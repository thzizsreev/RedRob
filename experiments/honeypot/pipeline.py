"""Orchestrate stratified sampling and multi-pass LLM honeypot study."""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from honeypot.config import (
    MANIFEST_FILENAME,
    RUN_SUMMARY_FILENAME,
    PipelineConfig,
    load_env,
)
from honeypot.llm_client import HoneypotLLMClient
from honeypot.persistence import ResultStore, utc_now_iso
from honeypot.sampling import (
    SampleEntry,
    build_and_write_manifest,
    build_and_write_manifest_from_filtered_ids,
    load_candidates_dict,
    load_manifest,
    manifest_entries,
)
from honeypot.schema import needs_pass2


def _vlog(config: PipelineConfig, message: str) -> None:
    if config.verbose:
        print(message, flush=True)


@dataclass
class PassStats:
    completed: int = 0
    skipped: int = 0
    failed: int = 0


def _resolve_manifest(config: PipelineConfig) -> tuple[dict[str, Any], Path]:
    output_dir = config.output_dir
    if config.manifest_path is not None:
        path = config.manifest_path
        return load_manifest(path), path

    manifest_path = output_dir / MANIFEST_FILENAME
    if manifest_path.exists():
        return load_manifest(manifest_path), manifest_path

    if config.filtered_ids_path is not None:
        manifest, path = build_and_write_manifest_from_filtered_ids(
            config.filtered_ids_path,
            config.candidates_path,
            output_dir,
        )
        print(f"Wrote filtered-ids manifest: {path}")
        return manifest, path

    manifest, path = build_and_write_manifest(
        config.candidates_path,
        output_dir,
        per_stratum=config.per_stratum,
        random_seed=config.random_seed,
    )
    print(f"Wrote sample manifest: {path}")
    return manifest, path


def _entry_map(entries: list[SampleEntry]) -> dict[str, SampleEntry]:
    return {e.candidate_id: e for e in entries}


def _process_one(
    *,
    pass_number: int,
    entry: SampleEntry,
    record: dict[str, Any],
    client: HoneypotLLMClient,
    store: ResultStore,
    config: PipelineConfig,
    pass1_row: dict[str, Any] | None,
) -> tuple[str, str]:
    """Returns (candidate_id, status) where status is completed|skipped|failed."""
    cid = entry.candidate_id
    if not config.force and store.has_result(pass_number, cid):
        _vlog(config, f"  [pass {pass_number}] SKIP {cid} (already in results)")
        return cid, "skipped"

    _vlog(config, f"  [pass {pass_number}] START {cid}")
    started = perf_counter()
    try:
        pass1_judgment = None
        if pass_number == 2 and pass1_row is not None:
            pass1_judgment = pass1_row.get("judgment")

        judgment = client.judge_candidate(
            record,
            pass_number=pass_number,
            pass1_judgment=pass1_judgment,
        )
        elapsed_ms = int((perf_counter() - started) * 1000)

        row: dict[str, Any] = {
            "candidate_id": cid,
            "pass": pass_number,
            "stratum": entry.stratum,
            "tags": entry.tags,
            "model": config.openai_model,
            "timestamp": utc_now_iso(),
            "latency_ms": elapsed_ms,
            "judgment": judgment.to_dict(),
        }

        if pass_number == 2 and pass1_row is not None:
            pass1_verdict = (pass1_row.get("judgment") or {}).get("verdict")
            row["pass1_reference"] = {
                "verdict": pass1_verdict,
                "confidence": (pass1_row.get("judgment") or {}).get("confidence"),
            }
            row["verdict_changed"] = pass1_verdict != judgment.verdict

        store.append_result(pass_number, row)
        verdict = judgment.verdict
        confidence = judgment.confidence
        contradiction = judgment.contradiction_type or "none"
        _vlog(
            config,
            f"  [pass {pass_number}] DONE  {cid}  "
            f"verdict={verdict}  confidence={confidence}  "
            f"contradiction={contradiction}  ({elapsed_ms}ms)",
        )
        return cid, "completed"
    except Exception as exc:
        store.append_failure(
            {
                "candidate_id": cid,
                "pass": pass_number,
                "stratum": entry.stratum,
                "tags": entry.tags,
                "timestamp": utc_now_iso(),
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
        )
        print(f"  FAILED {cid} pass {pass_number}: {exc}")
        return cid, "failed"


def _format_progress_line(
    pass_number: int,
    done: int,
    total: int,
    stats: PassStats,
    *,
    elapsed_sec: float,
) -> str:
    rate = done / elapsed_sec if elapsed_sec > 0 else 0.0
    return (
        f"  Pass {pass_number} progress: {done}/{total} "
        f"(ok={stats.completed}, skip={stats.skipped}, fail={stats.failed}, "
        f"{rate:.2f}/s)"
    )


def _run_pass(
    *,
    pass_number: int,
    work: list[tuple[SampleEntry, dict[str, Any], dict[str, Any] | None]],
    client: HoneypotLLMClient,
    store: ResultStore,
    config: PipelineConfig,
) -> PassStats:
    stats = PassStats()
    total = len(work)
    print(f"\nPass {pass_number}: {total} candidates, workers={config.max_workers}")
    pass_started = perf_counter()

    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = {
            executor.submit(
                _process_one,
                pass_number=pass_number,
                entry=entry,
                record=record,
                client=client,
                store=store,
                config=config,
                pass1_row=pass1_row,
            ): entry.candidate_id
            for entry, record, pass1_row in work
        }
        done = 0
        for future in as_completed(futures):
            _cid, status = future.result()
            if status == "completed":
                stats.completed += 1
            elif status == "skipped":
                stats.skipped += 1
            else:
                stats.failed += 1
            done += 1
            elapsed = perf_counter() - pass_started
            if config.verbose or done % 10 == 0 or done == total:
                print(_format_progress_line(pass_number, done, total, stats, elapsed_sec=elapsed))

    print(
        f"Pass {pass_number} done: completed={stats.completed}, "
        f"skipped={stats.skipped}, failed={stats.failed}, "
        f"api_retries={client.retries_total}, "
        f"elapsed={perf_counter() - pass_started:.1f}s"
    )
    return stats


def _build_pass1_work(
    entries: list[SampleEntry],
    records: dict[str, dict[str, Any]],
) -> list[tuple[SampleEntry, dict[str, Any], None]]:
    work: list[tuple[SampleEntry, dict[str, Any], None]] = []
    for entry in entries:
        record = records.get(entry.candidate_id)
        if record is None:
            print(f"  WARNING: missing record for {entry.candidate_id}, skipping")
            continue
        work.append((entry, record, None))
    return work


def _build_pass2_work(
    entries: list[SampleEntry],
    records: dict[str, dict[str, Any]],
    store: ResultStore,
) -> list[tuple[SampleEntry, dict[str, Any], dict[str, Any]]]:
    from honeypot.schema import HoneypotJudgment

    entry_by_id = _entry_map(entries)
    work: list[tuple[SampleEntry, dict[str, Any], dict[str, Any]]] = []

    for row in store.all_pass1():
        cid = str(row.get("candidate_id", ""))
        judgment_dict = row.get("judgment") or {}
        try:
            judgment = HoneypotJudgment.from_dict(judgment_dict)
        except ValueError:
            continue
        if not needs_pass2(judgment):
            continue
        entry = entry_by_id.get(cid)
        if entry is None:
            entry = SampleEntry(cid, ["pass2_extra"], "unknown")
        record = records.get(cid)
        if record is None:
            print(f"  WARNING: missing record for pass2 {cid}, skipping")
            continue
        work.append((entry, record, row))

    return work


def write_run_summary(
    output_dir: Path,
    *,
    manifest: dict[str, Any],
    pass1_stats: PassStats | None,
    pass2_stats: PassStats | None,
    store: ResultStore,
) -> Path:
    verdict_counts: Counter[str] = Counter()
    contradiction_counts: Counter[str] = Counter()
    stratum_counts: Counter[str] = Counter()
    pass2_changed = 0
    pass2_total = 0

    for row in store.all_pass1():
        j = row.get("judgment") or {}
        verdict_counts[str(j.get("verdict", "unknown"))] += 1
        contradiction_counts[str(j.get("contradiction_type", "unknown"))] += 1
        stratum_counts[str(row.get("stratum", "unknown"))] += 1

    pass2_rows = store.all_pass2()
    for row in pass2_rows:
        pass2_total += 1
        if row.get("verdict_changed"):
            pass2_changed += 1

    summary = {
        "generated_at": utc_now_iso(),
        "manifest_candidates_path": manifest.get("candidates_path"),
        "per_stratum": manifest.get("per_stratum"),
        "pass1": None if pass1_stats is None else pass1_stats.__dict__,
        "pass2": None if pass2_stats is None else pass2_stats.__dict__,
        "pass1_verdict_counts": dict(verdict_counts),
        "pass1_contradiction_type_counts": dict(contradiction_counts),
        "pass1_stratum_counts": dict(stratum_counts),
        "pass2_verdict_changed": pass2_changed,
        "pass2_total": pass2_total,
    }

    path = output_dir / RUN_SUMMARY_FILENAME
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return path


def run_pipeline(config: PipelineConfig) -> None:
    load_env()
    run_started = perf_counter()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.verbose:
        print("--- Honeypot run config ---")
        print(f"  candidates_path:   {config.candidates_path}")
        print(f"  output_dir:        {config.output_dir}")
        print(f"  filtered_ids_path: {config.filtered_ids_path}")
        print(f"  model:             {config.openai_model}")
        print(f"  workers:           {config.max_workers}")
        print(f"  rpm:               {config.requests_per_minute}")
        print(f"  pass_mode:         {config.pass_mode}")
        print(f"  force:             {config.force}")
        print(f"  sample_only:       {config.sample_only}")

    manifest, manifest_path = _resolve_manifest(config)
    entries = manifest_entries(manifest)
    candidates_path = Path(manifest.get("candidates_path") or config.candidates_path)

    _vlog(config, f"Loading candidates from {candidates_path}...")
    load_started = perf_counter()
    records = load_candidates_dict(candidates_path)
    _vlog(
        config,
        f"Loaded {len(records):,} candidate records in {perf_counter() - load_started:.1f}s",
    )

    print(f"Manifest: {manifest_path}")
    if manifest.get("sample_mode") == "filtered_ids":
        filtered_count = len(manifest.get("strata", {}).get("filtered", []))
        print(f"Sample mode: filtered_ids ({filtered_count} candidates)")
        print(f"Source: {manifest.get('filtered_ids_path')}")
    else:
        print(
            f"Sample size: A={len(manifest.get('strata', {}).get('A', []))}, "
            f"B={len(manifest.get('strata', {}).get('B', []))}, "
            f"C={len(manifest.get('strata', {}).get('C', []))}"
        )

    if config.sample_only:
        print("Sample-only mode — manifest written, no LLM calls.")
        return

    store = ResultStore(output_dir)
    client = HoneypotLLMClient(
        model=config.openai_model,
        requests_per_minute=config.requests_per_minute,
        max_retries=config.max_retries,
        initial_backoff_sec=config.initial_backoff_sec,
        verbose=config.verbose,
    )

    pass1_stats: PassStats | None = None
    pass2_stats: PassStats | None = None

    if config.pass_mode in ("1", "all"):
        work1 = _build_pass1_work(entries, records)
        _vlog(config, f"Pass 1 queue: {len(work1)} candidates")
        pass1_stats = _run_pass(
            pass_number=1,
            work=work1,
            client=client,
            store=store,
            config=config,
        )

    if config.pass_mode in ("2", "all"):
        work2 = _build_pass2_work(entries, records, store)
        print(f"Pass 2 eligible: {len(work2)} candidates")
        if work2:
            pass2_stats = _run_pass(
                pass_number=2,
                work=work2,
                client=client,
                store=store,
                config=config,
            )
        else:
            print("Pass 2: no eligible candidates (uncertain or honeypot+low).")

    summary_path = write_run_summary(
        output_dir,
        manifest=manifest,
        pass1_stats=pass1_stats,
        pass2_stats=pass2_stats,
        store=store,
    )
    print(f"\nRun summary: {summary_path}")
    if pass1_stats is not None:
        print(
            f"Pass 1 totals: ok={pass1_stats.completed}, "
            f"skip={pass1_stats.skipped}, fail={pass1_stats.failed}"
        )
    if pass2_stats is not None:
        print(
            f"Pass 2 totals: ok={pass2_stats.completed}, "
            f"skip={pass2_stats.skipped}, fail={pass2_stats.failed}"
        )
    print(f"Total wall time: {perf_counter() - run_started:.1f}s")
