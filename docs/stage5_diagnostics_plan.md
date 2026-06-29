# STAGE 5 — DIAGNOSTIC EXPERIMENTS PLAN
## For AI Coding Agent

---

## PURPOSE

This plan does **not** change the Stage 5 formula. It produces a diagnostic
report that tells us the nature of the data flowing through the current 7-layer
scoring pipeline. The outputs of these experiments will be used to redesign the
formula correctly — grounded in what the data actually shows rather than
assumptions.

The experiments answer six questions:

1. Which signals have real variance at Stage 5 and which have gone flat?
2. Which signals are measuring the same thing (correlated / redundant)?
3. How much does each layer actually reshuffle the ranking?
4. How many "wrong" rank flips is the availability multiplier causing?
5. How sparse are the boolean/categorical signals that drive penalties and bonuses?
6. Which availability sub-factors are discriminating and which are noise?

---

## IMPLEMENTATION CONTEXT

**Do not modify any existing Stage 5 scoring code.** This is a read-only
diagnostic. All experiments read existing artifacts and write a new report.
No scores, parquet files, or CSVs produced by Stage 5 are changed.

**Script to create:** `tracks/instructor/stage5/diagnostics.py`

**Entry point:**
```
python tracks/instructor/stage5/diagnostics.py
  --scored    artifacts/runtime/stage5/stage5_scored.parquet
  --jsonl     data/candidates.jsonl
  --out       artifacts/runtime/stage5/diagnostics/
```

The `--scored` file is `stage5_scored.parquet` — the full ~300-row intermediate
file that Stage 5 already writes. It contains every input signal, every
intermediate layer value, and the final score for all ~300 candidates that
entered Stage 5. This is the primary data source for all experiments.

The `--jsonl` file is needed only for Experiment 6 (availability sub-factors),
to recompute the seven individual factors which are not stored as separate
columns in the parquet.

**Output directory:** `artifacts/runtime/stage5/diagnostics/`

Create this directory if it does not exist. Write one file per experiment as
specified below. Also write a single combined `diagnostics_report.md` that
assembles all results in human-readable form for the operator to read.

---

## COLUMNS AVAILABLE IN `stage5_scored.parquet`

The agent must verify these columns exist before running any experiment.
Exit with a clear error listing any missing columns if the file does not
match this contract.

**Retrieval score columns (raw, pre-normalization):**
`cross_encoder_score`, `fused_score`, `q1_score`, `q2_score`, `q3_neg_sim`,
`skill_score`

**Normalized score columns (post-Layer 1 normalization):**
`ce_norm`, `fused_norm`, `q1_norm`, `q2_norm`, `q3_norm`

**Layer boundary columns (one per layer output):**
`core`, `core_floored`, `shaped`, `penalized`, `bonused`,
`availability_adj`, `final_score`

**Must-have coverage columns (Layer 2 intermediates):**
`keyword_ratio`, `assessment_cov`, `combined_coverage`,
`must_have_floor_multiplier`

**Career shape columns (Layer 3 inputs):**
`product_company_fraction`, `in_sweet_spot`, `exp_band`, `stale_coding`,
`has_any_production_role`, `shape_mult`

**Penalty columns (Layer 4 intermediates):**
`title_chasing_penalty`, `q3_residual_penalty`, `closed_source_penalty`,
`ambiguity_penalty`, `consulting_resid_penalty`, `total_penalty`,
`short_hop_count`, `title_ambiguous`, `career_type`, `external_validation_score`,
`has_github`

**Bonus columns (Layer 5 intermediates):**
`optional_bonus`

**Availability columns (Layer 6 inputs and output):**
`availability_multiplier`

**Logistics columns (Layer 7 intermediates and inputs):**
`location_adj`, `workmode_adj`, `notice_adj`, `logistics_adjustment`,
`location_tier`, `notice_period_days`

**Identity:**
`candidate_id`, `stage3_rank`, `stage4_rank`

---

## EXPERIMENT 1 — SIGNAL VARIANCE ANALYSIS

**Question:** Which signals have real spread across the 300 candidates
and which have gone flat after retrieval filtering?

**Why this matters:** A signal with near-zero variance contributes almost
nothing to ranking regardless of its assigned weight. The variance of a signal
determines its real influence, not the weight we assign to it.

### Signals to analyse

Compute the following statistics for each signal in two groups:

**Group A — Raw retrieval scores (before normalization):**
`cross_encoder_score`, `fused_score`, `q1_score`, `q2_score`, `q3_neg_sim`,
`skill_score`

**Group B — Normalized scores and layer outputs:**
`ce_norm`, `fused_norm`, `q1_norm`, `q2_norm`, `q3_norm`,
`core`, `core_floored`, `shaped`, `penalized`, `bonused`,
`availability_adj`, `final_score`

**Group C — Non-retrieval continuous signals:**
`product_company_fraction`, `external_validation_score`,
`must_have_floor_multiplier`, `shape_mult`, `total_penalty`,
`optional_bonus`, `availability_multiplier`

### Statistics to compute per signal

For every signal listed above, compute:

- `mean` — central tendency
- `std` — standard deviation (primary variance measure)
- `min` — minimum value in cohort
- `max` — maximum value in cohort
- `range` — max minus min
- `p5` — 5th percentile
- `p25` — 25th percentile
- `p50` — median
- `p75` — 75th percentile
- `p95` — 95th percentile
- `iqr` — interquartile range (p75 minus p25)
- `cv` — coefficient of variation (std / mean), only when mean != 0

### What to flag automatically

After computing statistics, apply these automatic flags and include them
in the output:

- Flag `LOW_VARIANCE` if `std < 0.05` for any signal in Group A or B.
  This means the signal is nearly constant across the cohort and is
  contributing almost no discrimination.

- Flag `COMPRESSED` if `iqr < 0.03` for any normalized score. This means
  the middle 50% of candidates are nearly indistinguishable on this signal.

- Flag `WIDE_RANGE` if `range > 0.5` for any signal in Group B. This means
  this signal has large absolute influence over the final score.

- Flag `FLOOR_DOMINATED` if `p25 == min` for `availability_multiplier`.
  This means many candidates are hitting the 0.5 floor clamp.

### Output file

`diagnostics/exp1_signal_variance.csv`

One row per signal. Columns: `signal_name`, `group`, `mean`, `std`, `min`,
`max`, `range`, `p5`, `p25`, `p50`, `p75`, `p95`, `iqr`, `cv`, `flags`

Also write a sorted version `diagnostics/exp1_signal_variance_sorted_by_std.csv`
sorted by `std` descending so the most discriminating signals appear first.

---

## EXPERIMENT 2 — PAIRWISE CORRELATION MATRIX

**Question:** Which signals are measuring the same underlying thing?
Which are genuinely independent?

**Why this matters:** If two signals correlate at 0.85, using both in the
formula is effectively double-counting the same information. Correlated signals
should either be merged or one should be dropped. Independent signals each
contribute unique information and deserve their own weight.

### Signals to correlate

Compute pairwise correlation for these signals:

**Retrieval group (primary interest — looking for redundancy):**
`ce_norm`, `fused_norm`, `q1_norm`, `q2_norm`, `q3_norm`, `skill_score`

**All signals combined (secondary — looking for unexpected relationships):**
All signals from Experiment 1 Group A, B, and C above.

### Correlation measures

For every pair of signals compute two correlation coefficients:

**Pearson correlation:** measures linear relationship. Formula:
```
pearson(x, y) = cov(x, y) / (std(x) × std(y))
```
Range: -1 to +1. Values above 0.7 or below -0.7 indicate strong correlation.

**Spearman rank correlation:** measures whether the two signals produce
similar orderings of candidates even if not linearly related. This is more
relevant than Pearson for a ranking system.
```
spearman(x, y) = pearson(rank(x), rank(y))
```
Range: -1 to +1. Values above 0.7 indicate the signals tend to rank
candidates in the same order.

### What to flag automatically

- Flag `HIGHLY_CORRELATED` for any pair with `|spearman| > 0.70`. These
  pairs are candidates for merging or dropping one.
- Flag `REDUNDANT_WITH_CE` specifically for any signal where
  `|spearman(signal, ce_norm)| > 0.70`. CE is the primary signal. Any
  signal that tracks CE closely adds minimal new information.
- Flag `GENUINELY_INDEPENDENT` for any pair with `|spearman| < 0.30`.
  These signals are measuring different things and both deserve weight.

### Output files

`diagnostics/exp2_pearson_matrix.csv` — full N×N Pearson matrix

`diagnostics/exp2_spearman_matrix.csv` — full N×N Spearman matrix

`diagnostics/exp2_high_correlation_pairs.csv` — filtered list of pairs
with `|spearman| > 0.50`, sorted by absolute correlation descending.
Columns: `signal_a`, `signal_b`, `pearson`, `spearman`, `flag`

---

## EXPERIMENT 3 — LAYER-BY-LAYER RANK STABILITY

**Question:** How much does each layer actually reshuffle the ranking?
Which layers are doing real work and which are contributing noise?

**Why this matters:** A layer that produces a rank correlation of 0.98
with its input is barely changing the ordering — it is almost a no-op.
A layer that produces 0.60 is significantly reshuffling candidates. This
tells us where the formula is actually making ranking decisions vs where
it is just adding complexity.

### What to compute

For each adjacent pair of layer outputs, compute:

1. **Spearman rank correlation** between the rank ordering at input and
   rank ordering at output. Closer to 1.0 means the layer barely changed
   the order.

2. **Kendall's tau** — an alternative rank correlation that is more
   sensitive to local swaps (adjacent rank changes). This catches cases
   where a layer is making many small swaps that Spearman misses.

3. **Position delta statistics** — for each candidate, compute the absolute
   difference in their rank position between the layer input and layer output.
   Then compute: mean position change, max position change, count of candidates
   who moved more than 10 positions, count who moved more than 20 positions.

### Layer pairs to analyse

Compute all of the above for these seven transitions:

| Transition | Input column | Output column |
|---|---|---|
| L0→L1 | `ce_norm` (as primary signal proxy) | `core` |
| L1→L2 | `core` | `core_floored` |
| L2→L3 | `core_floored` | `shaped` |
| L3→L4 | `shaped` | `penalized` |
| L4→L5 | `penalized` | `bonused` |
| L5→L6 | `bonused` | `availability_adj` |
| L6→L7 | `availability_adj` | `final_score` |

For L0→L1, use `ce_norm` as the proxy for "what the ranking would look like
using only the primary signal" — this shows how much the weighted blend in
Layer 1 changes the CE-only order.

### What to flag automatically

- Flag `HIGH_DISRUPTION` if spearman < 0.80 for any layer. This layer is
  significantly reshuffling the ranking.
- Flag `DOMINANT_LAYER` for the layer with the lowest spearman. This is
  the layer that has the most control over the final ranking — important
  to know if it should or not.
- Flag `NEAR_NOOP` if spearman > 0.97 for any layer. This layer is barely
  changing anything and may be simplified or removed.

### Output file

`diagnostics/exp3_layer_rank_stability.csv`

Columns: `transition`, `input_col`, `output_col`, `spearman`, `kendall_tau`,
`mean_position_delta`, `max_position_delta`, `candidates_moved_10plus`,
`candidates_moved_20plus`, `flags`

---

## EXPERIMENT 4 — AVAILABILITY RANK FLIP ANALYSIS

**Question:** How many candidates are being ranked incorrectly because
the availability multiplier overrides technical fitness differences?

**Why this matters:** This is the core problem identified in the Dhruv/Pranav
case. A technically stronger candidate (higher `bonused`) is being beaten by
a technically weaker candidate who has better availability. This experiment
quantifies how widespread this problem is across all 300 candidates.

### Definition of a "wrong flip"

A wrong flip occurs when:
```
bonused_i > bonused_j    (candidate i is technically stronger)
BUT
availability_adj_i < availability_adj_j    (candidate j ranks higher after availability)
```

This means the availability multiplier reversed the correct technical order.

### What to compute

**Step 1 — Count all wrong flips:**
Compare every pair of candidates (i, j) where i != j.
Count how many pairs satisfy the wrong flip condition above.
Total pairs in a 300-candidate cohort = 300×299/2 = 44,850 pairs.
Express wrong flips as both a raw count and a percentage of total pairs.

**Step 2 — Severity distribution:**
For each wrong flip pair, compute the technical gap
(`bonused_i - bonused_j`) and the availability gap
(`availability_multiplier_i - availability_multiplier_j`).

Bin wrong flips by technical gap size:
- Small gap: bonused difference < 0.05 (marginal technical difference)
- Medium gap: bonused difference 0.05–0.15 (meaningful technical difference)
- Large gap: bonused difference > 0.15 (substantial technical difference)

Large-gap wrong flips are the most damaging — a clearly stronger candidate
being beaten by a clearly weaker one purely due to availability.

**Step 3 — Availability multiplier distribution:**
Compute how many candidates hit the 0.5 floor clamp exactly (versus having
an unclamped value between 0.5 and 1.0). If many candidates are at 0.5, the
floor is compressing diversity and making availability less informative.

**Step 4 — Position impact of wrong flips:**
For the top-100 final ranking, identify how many candidates in the top 100
would not be there if availability were removed (bonused-only ranking).
And conversely, how many candidates in positions 101-300 would enter the
top 100 under bonused-only ranking.
This is the "true cost" of the current availability formula in terms of
submission quality.

### Output files

`diagnostics/exp4_availability_flips.csv`

Summary statistics:
Columns: `metric`, `value`
Rows:
- `total_pairs_compared`
- `wrong_flip_count`
- `wrong_flip_percentage`
- `small_gap_flips` (bonused diff < 0.05)
- `medium_gap_flips` (bonused diff 0.05–0.15)
- `large_gap_flips` (bonused diff > 0.15)
- `candidates_at_avail_floor` (availability_multiplier exactly 0.5)
- `top100_candidates_displaced` (would not be in top-100 without availability)
- `top100_candidates_rescued` (in top-100 only because of availability)

`diagnostics/exp4_large_gap_flips_detail.csv`

The individual wrong-flip pairs where bonused difference > 0.10.
Columns: `candidate_i`, `candidate_j`, `bonused_i`, `bonused_j`,
`bonused_gap`, `avail_mult_i`, `avail_mult_j`, `avail_gap`,
`final_rank_i`, `final_rank_j`

Sort by `bonused_gap` descending — worst cases first.

---

## EXPERIMENT 5 — BOOLEAN AND CATEGORICAL SIGNAL COVERAGE

**Question:** How many candidates actually trigger each boolean condition
or penalty/bonus? Are we writing formula logic for signals that affect
almost nobody?

**Why this matters:** If only 3 out of 300 candidates have `stale_coding == True`,
the stale_coding dampener affects 1% of the cohort. That may be correct
(it's a rare condition) or it may mean the feature is computed incorrectly
and most stale_coding cases are being missed. Either way, we need to know
which conditions are common vs rare to calibrate their formula weight.

### Signals to analyse

**Boolean signals:**
`in_sweet_spot`, `stale_coding`, `has_any_production_role`, `title_ambiguous`,
`has_github`

**Categorical signals with specific values that trigger penalties/bonuses:**
`exp_band` (count `in_band` vs `near_band`)
`career_type` (count `product_heavy`, `mixed`, `consulting_heavy`, `unknown`)
`location_tier` (count `preferred`, `acceptable`, `outside_india`, `unknown`)

**Penalty trigger counts:**
`title_chasing_penalty > 0` (count)
`closed_source_penalty > 0` (count)
`consulting_resid_penalty > 0` (count)
`ambiguity_penalty > 0` (count)
`q3_residual_penalty > 0.05` (meaningful Q3 penalty, count)

**Optional bonus trigger counts:**
`optional_bonus > 0` (any bonus triggered, count)
`optional_bonus >= 0.04` (two or more categories, count)
`optional_bonus >= 0.08` (cap hit, count)

**Penalty magnitude distribution:**
`total_penalty`: mean, std, p50, p75, p95, max

**Availability sub-factor distribution** (from Layer 6 output only —
detailed sub-factor analysis is in Experiment 6):
`availability_multiplier`: count at floor (== 0.5), count above 0.9,
count between 0.5 and 0.7, count between 0.7 and 0.9

### What to flag automatically

- Flag `RARELY_TRIGGERED` for any boolean that is True for fewer than 5%
  of candidates (fewer than ~15 of 300). Low trigger rate means either the
  signal is rare in this cohort (correct) or the detection logic is too
  conservative (worth investigating).

- Flag `ALWAYS_TRIGGERED` for any boolean that is True for more than 95%
  of candidates. Near-universal signals add no discrimination.

- Flag `CATEGORY_DOMINANT` for any categorical signal where one value
  accounts for more than 80% of candidates. If 85% of candidates are
  `location_tier == preferred`, the location tier is nearly constant and
  its weight in logistics is inflated relative to its actual discrimination.

- Flag `PENALTY_MISMATCH` if the sum of all penalty trigger counts is
  inconsistent (e.g. `consulting_resid_penalty > 0` count is very high
  but `career_type == consulting_heavy` count is low — something is wrong).

### Output file

`diagnostics/exp5_boolean_coverage.csv`

Columns: `signal_name`, `condition`, `count_triggered`, `pct_triggered`,
`flags`

One row per boolean condition and per categorical value.

---

## EXPERIMENT 6 — AVAILABILITY SUB-FACTOR DECOMPOSITION

**Question:** Of the seven factors that multiply together to produce
`availability_multiplier`, which ones are actually providing real
discrimination and which are near-constant noise?

**Why this matters:** Seven factors multiplied together can hit the 0.5
floor even when no single factor is severely bad. If three of the seven
factors have near-zero variance across the cohort (everyone scores similarly
on them), they are adding floor-compression without adding information.
We need to know which factors are real signals and which to simplify or remove.

### Setup

The seven sub-factors are NOT stored as separate columns in `stage5_scored.parquet`.
They must be recomputed from the raw behavioral signals in `candidates.jsonl`.

Load `candidates.jsonl`. Filter to only the ~300 candidates present in
`stage5_scored.parquet` (join on `candidate_id`). For each candidate,
recompute all seven factors using the exact same formulas from `layers.py`:

```
resp_factor    = clamp(recruiter_response_rate / 0.5, 0.6, 1.0)
                 if missing → 1.0

speed_factor   = clamp(1 - max(0, avg_response_time_hours - 24) / 168, 0.7, 1.0)
                 if missing → 1.0

days_inactive  = (date(2026-06-22) - last_active_date).days
recency_factor = 1.0 if days_inactive <= 30
                 else clamp(1.0 - (days_inactive - 30) / 180, 0.6, 1.0)
                 if missing → 1.0

open_factor    = 1.0 if open_to_work_flag else 0.85
                 if missing → 1.0

interview_factor = clamp(interview_completion_rate, 0.7, 1.0)
                   if missing or -1 → 1.0

offer_factor   = clamp(offer_acceptance_rate, 0.8, 1.0)
                 if missing or -1 → 1.0

market_factor  = 1.0 if (applications_submitted_30d > 0 or open_to_work_flag)
                 else 0.95
                 if missing → 1.0
```

Cross-check: for each candidate, verify that the product of the seven
recomputed factors (after clamping to [0.5, 1.0]) matches the
`availability_multiplier` column in the parquet within a tolerance of 0.001.
Log a warning for any mismatches.

### Statistics per sub-factor

For each of the seven factors compute:
`mean`, `std`, `min`, `max`, `p5`, `p25`, `p50`, `p75`, `p95`, `iqr`

Also compute:
- `count_at_floor`: how many candidates are at the factor's minimum clamp
  value (0.6 for resp, 0.7 for speed, etc.)
- `count_at_ceiling`: how many candidates are at 1.0 (the factor is not
  penalizing them at all)
- `count_missing_neutral`: how many candidates had the raw signal missing
  and defaulted to 1.0

### Pairwise correlation between sub-factors

Compute Spearman correlation between every pair of the seven sub-factors.
High correlation between two sub-factors means they are tracking the same
underlying availability behavior and double-penalizing it.

### Variance contribution analysis

Compute how much each sub-factor contributes to the variance of
`availability_multiplier` by comparing:
- Variance of `availability_multiplier` with all seven factors
- Variance of `availability_multiplier` when each factor is individually
  set to 1.0 (neutralized)

The factor whose neutralization most reduces the variance of
`availability_multiplier` is the most influential. The factor whose
neutralization has the least effect can potentially be removed without
changing the ranking.

### What to flag automatically

- Flag `NEAR_CONSTANT` for any sub-factor with `std < 0.03`. This factor
  is contributing noise rather than signal.
- Flag `FLOOR_HEAVY` for any sub-factor where `count_at_floor > 30%` of
  candidates. Many candidates are hitting the minimum, which means this
  factor is acting like a step function rather than a continuous signal.
- Flag `CEILING_HEAVY` for any sub-factor where `count_at_ceiling > 70%`
  of candidates. Most candidates score perfectly on this factor, so it
  adds no discrimination.
- Flag `MISSING_DOMINATED` for any sub-factor where `count_missing_neutral`
  > 40% of candidates. A signal that defaults to neutral for most candidates
  is providing sparse coverage.

### Output files

`diagnostics/exp6_avail_subfactors.csv`

One row per sub-factor. Columns: `factor_name`, `mean`, `std`, `min`,
`max`, `p5`, `p25`, `p50`, `p75`, `p95`, `iqr`, `count_at_floor`,
`count_at_ceiling`, `count_missing_neutral`, `flags`

`diagnostics/exp6_avail_subfactor_correlations.csv`

7×7 Spearman correlation matrix of the sub-factors.

`diagnostics/exp6_avail_variance_contribution.csv`

Columns: `factor_neutralized`, `avail_mult_variance_without_factor`,
`avail_mult_variance_baseline`, `variance_contribution_pct`

---

## COMBINED REPORT

After all six experiments complete, write `diagnostics/diagnostics_report.md`
with the following sections in order. This is the human-readable document
the operator reads to understand the data before redesigning the formula.

### Section 1 — Signal Discrimination Summary

A table showing each signal, its std, and its flag. Sorted by std descending.
One sentence interpretation: "These signals have real discrimination power at
Stage 5. These signals are nearly flat."

### Section 2 — Redundancy Map

List every pair flagged `HIGHLY_CORRELATED` (spearman > 0.70). For each pair,
one sentence: "Signal A and Signal B are measuring nearly the same thing
(spearman=X). Using both adds minimal new information."

### Section 3 — Layer Disruption Ranking

Table of all seven layer transitions sorted by spearman ascending (most
disruptive first). One sentence per layer: "Layer 6 (availability) is the
most disruptive layer, reshuffling X candidates by more than 20 positions."

### Section 4 — Availability Flip Summary

The key numbers from Experiment 4:
- Total wrong flips (with percentage)
- Large-gap wrong flips specifically
- How many top-100 candidates are displaced by availability
- One paragraph interpretation

### Section 5 — Boolean Signal Coverage

A table of all boolean and categorical signals with their trigger rates.
Flag any that are RARELY_TRIGGERED or ALWAYS_TRIGGERED with a note on
what this means for the formula.

### Section 6 — Availability Sub-Factor Summary

Which factors have real variance. Which are NEAR_CONSTANT or FLOOR_HEAVY.
A recommendation on which factors to keep, simplify, or remove — based purely
on what the data shows, not on design judgment. The design judgment happens
after this report is read.

### Section 7 — Raw Numbers for Formula Design

A single table with the key numbers that will directly feed into the new
formula design:

| Metric | Value |
|---|---|
| std(ce_norm) | — |
| std(q1_norm) | — |
| std(q2_norm) | — |
| std(fused_norm) | — |
| std(skill_score_norm) | — |
| spearman(ce_norm, q1_norm) | — |
| spearman(ce_norm, fused_norm) | — |
| spearman(ce_norm, q2_norm) | — |
| spearman(ce_norm, skill_score) | — |
| spearman(q1_norm, q2_norm) | — |
| layer_spearman L1→L2 | — |
| layer_spearman L2→L3 | — |
| layer_spearman L3→L4 | — |
| layer_spearman L4→L5 | — |
| layer_spearman L5→L6 | — |
| layer_spearman L6→L7 | — |
| wrong_flip_count | — |
| wrong_flip_large_gap_count | — |
| top100_candidates_displaced | — |
| availability_multiplier std | — |
| availability_multiplier pct_at_floor | — |
| count in_sweet_spot == True | — |
| count stale_coding == True | — |
| count consulting_heavy | — |

This table is the input to the formula redesign conversation.

---

## EXECUTION ORDER

Run experiments in this order. Each experiment is independent except that
Experiment 6 requires `candidates.jsonl` in addition to the parquet.

```
1. Validate parquet columns exist → exit with error list if any missing
2. Run Experiment 1 (variance)
3. Run Experiment 2 (correlation)
4. Run Experiment 3 (layer stability)
5. Run Experiment 4 (availability flips)
6. Run Experiment 5 (boolean coverage)
7. Run Experiment 6 (availability sub-factors) — requires JSONL
8. Write diagnostics_report.md combining all results
9. Print summary to stdout: key flags and the Section 7 table
```

---

## TECH STACK

| Library | Purpose |
|---|---|
| `polars` | Load and query `stage5_scored.parquet`, all DataFrame operations |
| `numpy` | Spearman/Pearson correlation, percentile computation |
| `scipy.stats` | `spearmanr`, `kendalltau`, `pearsonr` |
| `json` | Parse `candidates.jsonl` line by line for Experiment 6 |
| `pathlib` | Directory creation, path handling |
| `datetime` | Date arithmetic for recency factor in Experiment 6 |

Do not import `pandas`, `sklearn`, `torch`, or any visualization libraries.
This is a pure computation and reporting script — no plots, no models.

---

## WHAT THIS PLAN DOES NOT DO

This plan does NOT change the Stage 5 formula. It does NOT change any
weights, thresholds, or layer logic. It does NOT produce a new submission CSV.
It reads existing artifacts, computes statistics, and writes a diagnostic
report. The formula redesign happens after the operator reads this report.
