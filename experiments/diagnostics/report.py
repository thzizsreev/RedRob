"""Assemble diagnostics_report.md and stdout summary."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from stats_utils import safe_spearman, to_float_array


def _get_std(exp1: pl.DataFrame, signal: str) -> str:
    row = exp1.filter(pl.col("signal_name") == signal)
    if row.is_empty():
        return "N/A"
    val = row["std"][0]
    return f"{val:.4f}" if val is not None else "N/A"


def _get_spearman(df: pl.DataFrame, a: str, b: str) -> str:
    if a not in df.columns or b not in df.columns:
        return "N/A"
    x = to_float_array(df[a])
    y = to_float_array(df[b])
    min_len = min(len(x), len(y))
    val = safe_spearman(x[:min_len], y[:min_len])
    return f"{val:.4f}" if val is not None else "N/A"


def _layer_spearman(exp3: pl.DataFrame, transition: str) -> str:
    row = exp3.filter(pl.col("transition") == transition)
    if row.is_empty():
        return "N/A"
    val = row["spearman"][0]
    return f"{val:.4f}" if val is not None else "N/A"


def write_report(
    output_dir: Path,
    exp1: pl.DataFrame,
    exp2: dict,
    exp3: pl.DataFrame,
    exp4: dict,
    exp5: pl.DataFrame,
    exp6: dict,
    scored_df: pl.DataFrame,
) -> tuple[Path, dict]:
    sorted_exp1 = exp1.sort("std", descending=True, nulls_last=True)
    high_corr = exp2["high_correlation_pairs"].filter(
        pl.col("flag").str.contains("HIGHLY_CORRELATED")
    )
    exp3_sorted = exp3.sort("spearman", nulls_last=False)
    summary = exp4["summary_dict"]

    lines: list[str] = [
        "# Stage 5 Diagnostics Report",
        "",
        "## Section 1 — Signal Discrimination Summary",
        "",
        "| Signal | Std | Flags |",
        "|---|---|---|",
    ]

    flat_signals = sorted_exp1.filter(pl.col("flags").str.contains("LOW_VARIANCE"))
    discriminating = sorted_exp1.filter(
        ~pl.col("flags").str.contains("LOW_VARIANCE") & pl.col("std").is_not_null()
    )

    for row in sorted_exp1.iter_rows(named=True):
        std = row["std"]
        std_s = f"{std:.4f}" if std is not None else "—"
        lines.append(f"| {row['signal_name']} | {std_s} | {row['flags']} |")

    flat_names = ", ".join(flat_signals["signal_name"].to_list()[:8]) or "none flagged"
    disc_names = ", ".join(discriminating["signal_name"].to_list()[:8]) or "none"
    lines.extend(
        [
            "",
            f"Signals with real discrimination power at Stage 5 include: {disc_names}. "
            f"Nearly flat signals (LOW_VARIANCE) include: {flat_names}.",
            "",
            "## Section 2 — Redundancy Map",
            "",
        ]
    )

    if high_corr.is_empty():
        lines.append("No pairs flagged HIGHLY_CORRELATED (|spearman| > 0.70).")
    else:
        for row in high_corr.iter_rows(named=True):
            lines.append(
                f"- {row['signal_a']} and {row['signal_b']} are measuring nearly the same thing "
                f"(spearman={row['spearman']:.4f}). Using both adds minimal new information."
            )

    lines.extend(
        [
            "",
            "## Section 3 — Layer Disruption Ranking",
            "",
            "| Transition | Spearman | Kendall | Moved 20+ | Flags |",
            "|---|---|---|---|---|",
        ]
    )

    for row in exp3_sorted.iter_rows(named=True):
        sp = row["spearman"]
        kt = row["kendall_tau"]
        sp_s = f"{sp:.4f}" if sp is not None else "—"
        kt_s = f"{kt:.4f}" if kt is not None else "—"
        lines.append(
            f"| {row['transition']} | {sp_s} | {kt_s} | {row['candidates_moved_20plus']} | {row['flags']} |"
        )

    dominant = exp3_sorted.filter(pl.col("flags").str.contains("DOMINANT_LAYER"))
    if not dominant.is_empty():
        d = dominant.row(0, named=True)
        lines.append(
            f"\n{d['transition']} is the most disruptive layer, reshuffling "
            f"{d['candidates_moved_20plus']} candidates by more than 20 positions."
        )

    lines.extend(
        [
            "",
            "## Section 4 — Availability Flip Summary",
            "",
            f"- Total wrong flips: {summary.get('wrong_flip_count', '—')} "
            f"({summary.get('wrong_flip_percentage', '—')}%)",
            f"- Large-gap wrong flips (>0.15 pre-avail diff): {summary.get('large_gap_flips', '—')}",
            f"- Top-100 candidates displaced by availability tier: {summary.get('top100_candidates_displaced', '—')}",
            f"- Top-100 candidates rescued by availability tier: {summary.get('top100_candidates_rescued', '—')}",
            "",
            "Wrong flips occur when a technically stronger candidate (higher score_after_t2) is ranked below "
            "a weaker one after tier3 (availability) is applied. Large-gap flips are the most "
            "damaging cases; displaced top-100 counts quantify submission-quality cost.",
            "",
            "## Section 5 — Boolean Signal Coverage",
            "",
            "| Signal | Condition | Count | Pct | Flags |",
            "|---|---|---|---|---|",
        ]
    )

    for row in exp5.iter_rows(named=True):
        lines.append(
            f"| {row['signal_name']} | {row['condition']} | {row['count_triggered']} | "
            f"{row['pct_triggered']} | {row['flags']} |"
        )

    flagged_exp5 = exp5.filter(
        pl.col("flags").str.contains("RARELY_TRIGGERED")
        | pl.col("flags").str.contains("ALWAYS_TRIGGERED")
    )
    if not flagged_exp5.is_empty():
        lines.append("")
        lines.append("Notes on flagged coverage:")
        for row in flagged_exp5.iter_rows(named=True):
            lines.append(
                f"- {row['signal_name']} {row['condition']}: {row['pct_triggered']}% triggered ({row['flags']})"
            )

    lines.extend(
        [
            "",
            "## Section 6 — Availability Sub-Factor Summary",
            "",
            "| Factor | Std | At Floor | At Ceiling | Missing Neutral | Flags |",
            "|---|---|---|---|---|---|",
        ]
    )

    subfactors = exp6["subfactors"]
    for row in subfactors.iter_rows(named=True):
        std = row["std"]
        std_s = f"{std:.4f}" if std is not None else "—"
        lines.append(
            f"| {row['factor_name']} | {std_s} | {row['count_at_floor']} | "
            f"{row['count_at_ceiling']} | {row['count_missing_neutral']} | {row['flags']} |"
        )

    keep = subfactors.filter(
        ~pl.col("flags").str.contains("NEAR_CONSTANT")
        & ~pl.col("flags").str.contains("MISSING_DOMINATED")
    )
    simplify = subfactors.filter(pl.col("flags").str.contains("FLOOR_HEAVY"))
    remove = subfactors.filter(pl.col("flags").str.contains("NEAR_CONSTANT"))

    lines.extend(
        [
            "",
            "**Data-driven recommendations:**",
            f"- Keep (discriminating): {', '.join(keep['factor_name'].to_list()) or 'none identified'}",
            f"- Simplify (floor-heavy): {', '.join(simplify['factor_name'].to_list()) or 'none identified'}",
            f"- Remove candidate (near-constant): {', '.join(remove['factor_name'].to_list()) or 'none identified'}",
            "",
            "## Section 7 — Raw Numbers for Formula Design",
            "",
            "| Metric | Value |",
            "|---|---|",
        ]
    )

    avail_unit = scored_df["avail_unit"].cast(pl.Int64)
    tier_c_pct = float((avail_unit == -1).sum()) / scored_df.height * 100 if scored_df.height else 0

    section7 = {
        "std(borda_primary)": _get_std(exp1, "borda_primary"),
        "std(tier2_scaled)": _get_std(exp1, "tier2_scaled"),
        "std(tier3_scaled)": _get_std(exp1, "tier3_scaled"),
        "std(tier4_scaled)": _get_std(exp1, "tier4_scaled"),
        "std(skill_score)": _get_std(exp1, "skill_score"),
        "spearman(cross_encoder_score, q1_score)": _get_spearman(
            scored_df, "cross_encoder_score", "q1_score"
        ),
        "spearman(cross_encoder_score, fused_score)": _get_spearman(
            scored_df, "cross_encoder_score", "fused_score"
        ),
        "spearman(cross_encoder_score, q2_score)": _get_spearman(
            scored_df, "cross_encoder_score", "q2_score"
        ),
        "spearman(cross_encoder_score, skill_score)": _get_spearman(
            scored_df, "cross_encoder_score", "skill_score"
        ),
        "spearman(q1_score, q2_score)": _get_spearman(scored_df, "q1_score", "q2_score"),
        "layer_spearman T0->T1": _layer_spearman(exp3, "T0→T1"),
        "layer_spearman T1->T2": _layer_spearman(exp3, "T1→T2"),
        "layer_spearman T2->T3": _layer_spearman(exp3, "T2→T3"),
        "layer_spearman T3->T4": _layer_spearman(exp3, "T3→T4"),
        "wrong_flip_count": str(summary.get("wrong_flip_count", "—")),
        "wrong_flip_large_gap_count": str(summary.get("large_gap_flips", "—")),
        "top100_candidates_displaced": str(summary.get("top100_candidates_displaced", "—")),
        "tier3_scaled std": _get_std(exp1, "tier3_scaled"),
        "avail_unit pct_tier_c": f"{tier_c_pct:.2f}%",
        "count in_sweet_spot == True": str(
            int(scored_df["in_sweet_spot"].cast(pl.Boolean, strict=False).sum())
        ),
        "count stale_coding == True": str(
            int(scored_df["stale_coding"].cast(pl.Boolean, strict=False).sum())
        ),
        "count consulting_heavy": str(
            int((scored_df["career_type"].cast(pl.Utf8) == "consulting_heavy").sum())
        ),
    }

    for metric, value in section7.items():
        lines.append(f"| {metric} | {value} |")

    report_path = output_dir / "diagnostics_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path, section7


def _ascii_safe(text: object) -> str:
    return (
        str(text)
        .replace("\u2192", "->")
        .replace("\u2014", "-")
        .replace("\u2194", "<->")
    )


def print_stdout_summary(
    exp1: pl.DataFrame,
    exp2: dict,
    exp3: pl.DataFrame,
    section7: dict,
) -> None:
    print("\n=== Stage 5 Diagnostics Summary ===\n")

    flagged = exp1.filter(pl.col("flags") != "")
    if flagged.is_empty():
        print("No variance flags triggered.")
    else:
        print("Key variance flags:")
        for row in flagged.head(10).iter_rows(named=True):
            print(f"  {row['signal_name']}: {row['flags']}")

    high = exp2["high_correlation_pairs"].filter(pl.col("flag").str.contains("HIGHLY_CORRELATED"))
    if not high.is_empty():
        print("\nHighly correlated pairs:")
        for row in high.head(5).iter_rows(named=True):
            print(f"  {row['signal_a']} <-> {row['signal_b']}: spearman={row['spearman']:.4f}")

    disrupted = exp3.filter(pl.col("flags").str.contains("HIGH_DISRUPTION"))
    if not disrupted.is_empty():
        print("\nHigh-disruption layers:")
        for row in disrupted.iter_rows(named=True):
            transition = _ascii_safe(row["transition"])
            print(f"  {transition}: spearman={row['spearman']:.4f}")

    print("\nSection 7 - Key metrics:")
    print("| Metric | Value |")
    print("|---|---|")
    for metric, value in section7.items():
        print(f"| {_ascii_safe(metric)} | {_ascii_safe(value)} |")
