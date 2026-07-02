# STAGE 5 — FINAL IMPLEMENTATION PLAN
## Distribution-Aware Cascade Formula (v2)
### For AI Coding Agent — Read Entire Document Before Writing Any Code

---

## 0. CRITICAL DIRECTIVES

- **DO NOT** carry forward any logic from the old 7-layer formula unless explicitly listed as retained in Section 12.
- **DO NOT** hardcode any weight, threshold, ratio, or magnitude. Every numeric value comes from `config.yaml` under `stage5:`.
- **DO NOT** use PyTorch, Transformers, FAISS, ONNX, or any LLM API. This script is pure tabular arithmetic.
- **DO NOT** read `candidate_features.parquet` for behavioral signals. Read behavioral signals from `candidates.jsonl` exactly as the current Stage 5 implementation does. Do not change the data plumbing — only the scoring formula and reasoning generation change.
- **VERIFY** every required column exists at script startup. Fail loudly with a descriptive error naming every missing column before doing any computation.

---

## 1. WHAT THIS SCRIPT DOES

Takes ~300 candidates from Stage 4, applies a 4-tier distribution-aware cascade scoring formula, sorts to produce the final ranking, generates per-candidate reasoning text, writes the top-100 submission CSV.

**Script to replace:** `stages/stage5_rank.py` (or `stage5_lgbm.py` — replace whichever exists, in place).

**Inputs:**
- `artifacts/runtime/stage4_reranked.parquet` — ~300 candidates with all upstream columns
- `data/candidates.jsonl` — source of behavioral signals (existing join pattern, do not change)
- `config.yaml` — all parameters under `stage5:` namespace

**Outputs:**
- `artifacts/runtime/team_xxx.csv` — submission file, 101 rows (1 header + 100 data)
- `artifacts/runtime/stage5_full_scores.parquet` — debug file, all 300 candidates, all intermediate columns (NOT submitted)

---

## 2. INPUT COLUMN CONTRACT

### 2.1 From `stage4_reranked.parquet`

Verify all of these exist at startup. Hard error if any are missing.

| Column | Type | Description |
|---|---|---|
| `candidate_id` | String | Primary key. Format `CAND_XXXXXXX`. |
| `cross_encoder_score` | Float64 | Stage 4 cross-encoder score. Raw, not normalized. Higher = better. |
| `q1_score` | Float64 | Stage 3 dense retrieval score for Q1 (career history / retrieval subspace). Raw dot product. |
| `q2_score` | Float64 | Stage 3 dense retrieval score for Q2 (product-company career trajectory subspace). Raw dot product. |
| `q3_neg_sim` | Float64 | Stage 3 anti-pattern similarity. **NOT used in scoring.** Retained for debug output and reasoning only. |
| `fused_score` | Float64 | Stage 3 RRF fusion score. **NOT used in scoring.** Retained for debug output only. |
| `title_chasing_penalty` | Float64 | Pre-computed in Stage 2/5. Range [0, 0.15]. Formula: `min(0.15, short_hop_count × 0.03)`. |
| `ambiguity_penalty` | Float64 | Pre-computed. Range [0, 0.04]. Formula: `(title_ambiguous × 0.02) + (exp_band=="near_band" × 0.02)`. |
| `closed_source_penalty` | Float64 | Pre-computed. Range [0, 0.05]. Value is 0.05 if `external_validation_score < 0.1 AND NOT has_github`, else 0. |
| `optional_bonus` | Float64 | Pre-computed in Stage 5 Layer 5. Range [0, 0.08]. Already capped. |
| `in_sweet_spot` | Boolean | True if total_years_exp between 6 and 8 inclusive. |
| `location_tier` | String | Enum: `preferred`, `acceptable`, `outside_india`, `unknown`. Computed in Stage 2. |
| `total_years_exp` | Int64 | Total years of experience. Used in reasoning text only. |
| `career_type` | String | Enum: `product_heavy`, `mixed`, `consulting_heavy`, `unknown`. Used in reasoning only. |
| `short_hop_count` | Int64 | Number of jobs under 1.5 years. Used in reasoning only. |
| `external_validation_score` | Float64 | Composite of github + OSS presence. Range [0, 1]. Used in reasoning only. |
| `has_github` | Boolean | True if github_activity_score != -1. Used in reasoning only. |
| `title_ambiguous` | Boolean | Used in reasoning only. |
| `exp_band` | String | `in_band` or `near_band`. Used in reasoning only. |
| `technical_summary_sentence` | String | Pre-computed 30-word summary. May be null — see Section 9 fallback. |
| `stage3_rank` | Int64 | Rank from Stage 3. Used in debug output only. |
| `stage4_rank` | Int64 | Rank from Stage 4. Used in debug output only. |

### 2.2 From `candidates.jsonl` (behavioral signals — existing join, do not change)

The script already reads `candidates.jsonl` and extracts behavioral signals. **Keep this join exactly as it currently exists.** Only the signals listed below are used in the new formula. Others may still be read as before.

| Field in jsonl | Column name after join | Type | Used for |
|---|---|---|---|
| `redrob_signals.interview_completion_rate` | `interview_completion_rate` | Float64 | Tier 3 availability |
| `redrob_signals.offer_acceptance_rate` | `offer_acceptance_rate` | Float64 | Tier 3 availability |
| `redrob_signals.last_active_date` | `last_active_date` | String `YYYY-MM-DD` | Tier 3 availability |
| `redrob_signals.notice_period_days` | `notice_period_days` | Int64 | Tier 4 logistics |
| `redrob_signals.preferred_work_mode` | `preferred_work_mode` | String | Tier 4 logistics |

**Null handling for all behavioral signals:** if null after join, apply neutral defaults. Do not crash. Do not drop the candidate. Defaults:
- `interview_completion_rate` null → treat as Tier B (no classification trigger)
- `offer_acceptance_rate` null → same as -1 (skip offer condition in Tier A check)
- `offer_acceptance_rate == -1` → no prior offers, skip offer condition in Tier A check (not a penalty)
- `last_active_date` null → `days_since_active = 999` (triggers Tier C recency condition)
- `notice_period_days` null → `notice_unit = 0` (neutral)
- `preferred_work_mode` null → `workmode_unit = 0` (neutral)

---

## 3. THE FORMULA — OVERVIEW

Four passes. All additive at final assembly. No multipliers anywhere.

```
final_score = borda_primary + tier2_scaled + tier3_scaled + tier4_scaled
```

The cascade property is guaranteed by construction through distribution-aware rescaling:

```
std(borda_primary) = t1_std              ← anchor, set by data
std(tier2_scaled)  = tier2_ratio × t1_std   ← config: 0.25
std(tier3_scaled)  = tier3_ratio × t1_std   ← config: 0.15
std(tier4_scaled)  = tier4_ratio × t1_std   ← config: 0.05
```

Zero is anchored: a candidate with no penalties, no bonuses, neutral availability, neutral logistics scores exactly 0 on Tiers 2+3+4. Only Tier 1 differentiates them.

---

## 4. PASS 1 — TIER 1: BORDA PRIMARY

### Step 1.1 — Dense rank each signal

Rank all candidates on each of the three signals. Rank 1 = highest score (best candidate). Use **dense ranking**: tied candidates get the same rank, next rank is not skipped. N = actual row count of the DataFrame (do not hardcode 300).

```
rank_ce = dense_rank(cross_encoder_score, descending=True)   # integers [1, N]
rank_q1 = dense_rank(q1_score,            descending=True)   # integers [1, N]
rank_q2 = dense_rank(q2_score,            descending=True)   # integers [1, N]
```

### Step 1.2 — Amplify Q1 and Q2 only

CE is NOT amplified. Q1 and Q2 receive a convex power transformation that widens separation at the top of the ranking. Exponent from config: `borda.q_amplification_exponent = 1.4`.

```
rank_q1_amp = rank_q1 ^ 1.4    # float
rank_q2_amp = rank_q2 ^ 1.4    # float
```

Rationale: Q2 has the highest empirical std (0.401) and represents career history depth. Q1 second (0.353). CE lowest (0.238). Amplifying Q1/Q2 gives their ordering more weight at the top of the pool without simply multiplying their scores.

### Step 1.3 — Weighted Borda sum

Weights from config: `borda.w_ce = 0.25`, `borda.w_q1 = 0.35`, `borda.w_q2 = 0.40`. Sum = 1.0.

```
borda_sum = (w_ce × rank_ce) + (w_q1 × rank_q1_amp) + (w_q2 × rank_q2_amp)
```

Lower borda_sum = better candidate (lower rank number = better).

### Step 1.4 — Normalize and invert to [0, 1]

Min-max over actual values in the current pool. Do not use a fixed constant.

```
borda_min = min(borda_sum)
borda_max = max(borda_sum)

borda_primary = 1.0 - (borda_sum - borda_min) / (borda_max - borda_min)
```

After this: higher borda_primary = better candidate. Range is exactly [0.0, 1.0].

**Edge case:** if borda_max == borda_min (all candidates identical — should not occur), set `borda_primary = 0.5` for all and log a WARNING.

### Step 1.5 — Measure Tier 1 spread

This measurement is used by all subsequent tiers.

```
t1_std = std(borda_primary)    # computed over all N candidates
```

Store `t1_std` as a scalar. Log its value to stdout.

---

## 5. PASS 2 — TIER 2: CAREER SHAPE + PENALTIES (raw → rescaled)

### Step 2.1 — Compute raw Tier 2

All input columns already exist in `stage4_reranked.parquet`. No new computation.

**Positive signals:**
```
sweet_bonus = 0.04 if in_sweet_spot == True else 0.0
```

`optional_bonus` is already computed and capped at [0, 0.08]. Use directly.

**Negative signals (all already non-negative values):**
```
chase_pen  = title_chasing_penalty      # range [0, 0.15], 37.3% of candidates > 0
ambig_pen  = ambiguity_penalty          # range [0, 0.04], 34.3% of candidates > 0
closed_pen = closed_source_penalty      # range [0, 0.05], 24.0% of candidates > 0
```

**Raw combination:**
```
tier2_raw = sweet_bonus + optional_bonus - chase_pen - ambig_pen - closed_pen
```

Zero means: not in sweet spot, no optional bonus, no penalties. This is the natural neutral point. A candidate with nothing positive and nothing negative scores exactly 0. Candidates with active penalties go negative. Candidates with bonuses go positive.

### Step 2.2 — Measure raw spread

```
t2_std = std(tier2_raw)    # computed over all N candidates
```

### Step 2.3 — Rescale to cascade budget (zero-anchored, no mean subtraction)

Target std from config: `cascade.tier2_ratio = 0.25`.

```
target_t2_std = tier2_ratio × t1_std

if t2_std > 0:
    tier2_scaled = tier2_raw × (target_t2_std / t2_std)
else:
    tier2_scaled = 0.0 for all candidates    # log WARNING: tier2 is constant
```

**Why no mean subtraction:** subtracting the mean would push candidates with zero penalties into negative territory. A candidate with no active signals in this tier should score exactly 0, not be penalized for being "below average" on penalties. The zero anchor is preserved by multiplying raw values by the rescale factor without centering.

---

## 6. PASS 3 — TIER 3: AVAILABILITY (raw → rescaled)

### Step 6.1 — Compute days_since_active

Parse `last_active_date` (string `YYYY-MM-DD`) to a date object. Compute:

```
days_since_active = (run_date - last_active_date).days
```

Where `run_date = datetime.date.today()` at script execution time.

If `last_active_date` is null or unparseable: `days_since_active = 999`.

### Step 6.2 — Classify into three tiers using unit values

Evaluate **Tier C first**. If Tier C applies, do not check Tier A.

**Tier C — Clearly unavailable** (ANY one condition sufficient):
- `interview_completion_rate < tier_c_interview_max` (config: 0.30)
- `days_since_active > tier_c_recency_min_days` (config: 180)

→ `avail_unit = -1`

**Tier A — Clearly available** (ALL conditions must hold simultaneously):
- `interview_completion_rate >= tier_a_interview_min` (config: 0.70)
- `offer_acceptance_rate >= tier_a_offer_min` (config: 0.60) — **skip this condition entirely if offer_acceptance_rate is null OR == -1**
- `days_since_active <= tier_a_recency_max_days` (config: 30)

→ `avail_unit = +1`

**Tier B — Default** (everything not C or A):
→ `avail_unit = 0`

Store `avail_tier` as string column ("A", "B", "C") for reasoning and debug output.

### Step 6.3 — Measure raw spread

```
t3_std = std(avail_unit)    # computed over all N candidates
```

The distribution will be trimodal: some at -1, most at 0, some at +1.

### Step 6.4 — Rescale to cascade budget (zero-anchored)

Target std from config: `cascade.tier3_ratio = 0.15`.

```
target_t3_std = tier3_ratio × t1_std

if t3_std > 0:
    tier3_scaled = avail_unit × (target_t3_std / t3_std)
else:
    tier3_scaled = 0.0 for all candidates    # log WARNING: no availability variance
```

Tier B candidates (avail_unit = 0) stay exactly at 0. No centering applied.

---

## 7. PASS 4 — TIER 4: LOGISTICS (raw → rescaled)

### Step 7.1 — Compute unit values for each sub-signal

All three sub-signals use integer unit values. The magnitude comes from rescaling, not from the unit values themselves. Units encode relative priority within the tier.

**Location (from `location_tier` in Stage 4 parquet):**
```
location_unit = {
    "preferred":      1,
    "acceptable":     0,
    "outside_india": -2,    # double weight: no visa sponsorship, hard JD signal
    "unknown":        0,
}[location_tier]
```

If location_tier is null or any value not in the above dict: `location_unit = 0`.

**Work mode (from `preferred_work_mode` in candidates.jsonl):**

JD specifies hybrid as preferred (offices used Tue/Thu, flexible cadence).

```
workmode_unit = 1 if preferred_work_mode in {"hybrid", "flexible"} else 0
```

**Notice period (from `notice_period_days` in candidates.jsonl):**
```
if notice_period_days is null:    notice_unit = 0     # unknown → neutral
elif notice_period_days <= 30:    notice_unit = +1    # ideal per JD
elif notice_period_days <= 90:    notice_unit = 0     # acceptable
else:                             notice_unit = -1    # bar gets higher per JD
```

### Step 7.2 — Combine

```
tier4_raw = location_unit + workmode_unit + notice_unit
```

Theoretical range: [-4, +3]. Typical range much narrower in practice.

### Step 7.3 — Measure raw spread

```
t4_std = std(tier4_raw)    # computed over all N candidates
```

### Step 7.4 — Rescale to cascade budget (zero-anchored)

Target std from config: `cascade.tier4_ratio = 0.05`.

```
target_t4_std = tier4_ratio × t1_std

if t4_std > 0:
    tier4_scaled = tier4_raw × (target_t4_std / t4_std)
else:
    tier4_scaled = 0.0 for all candidates    # log WARNING: no logistics variance
```

---

## 8. FINAL SCORE ASSEMBLY

```
final_score = borda_primary + tier2_scaled + tier3_scaled + tier4_scaled
```

All additive. No clamping after assembly. Store all intermediate columns.

**Intermediate columns to store (needed for debug parquet and reasoning):**

From Tier 1: `rank_ce`, `rank_q1`, `rank_q2`, `rank_q1_amp`, `rank_q2_amp`, `borda_sum`, `borda_primary`, `t1_std`

From Tier 2: `sweet_bonus`, `tier2_raw`, `t2_std`, `target_t2_std`, `tier2_scaled`

From Tier 3: `days_since_active`, `avail_unit`, `avail_tier`, `t3_std`, `target_t3_std`, `tier3_scaled`

From Tier 4: `location_unit`, `workmode_unit`, `notice_unit`, `tier4_raw`, `t4_std`, `target_t4_std`, `tier4_scaled`

Final: `final_score`

---

## 9. SORTING, SLICING, MONOTONICITY

### Sort

Sort DataFrame by `final_score` descending.

**Tie-breaking:** if two candidates have identical `final_score`, break deterministically by `candidate_id` ascending (lexicographic). This ensures reproducibility.

### Slice

Take the first 100 rows. Assign `rank` as integers 1 through 100 (1-indexed, 1 = best).

### Monotonicity enforcement

Assert `final_score[i] >= final_score[i+1]` for all i in 0..98.

If any violation exists (possible at tie boundaries after secondary sort):
```
for i in range(1, 100):
    if score[i] > score[i-1]:
        score[i] = score[i-1]
```

Log a WARNING with count of positions corrected if this triggers. Should be rare.

---

## 10. REASONING COLUMN GENERATION

No LLM. No API calls. Construct from actual column values only.

**Anti-hallucination rule:** every fact in every clause must come from an actual column value. Never invent company names, years, scores, or skills not present as real column values. If a value is null, use the fallback text.

### Clause 1 — Primary strength (always present)

Evaluate in priority order, pick the first that applies:

1. `borda_primary >= 0.75` → `"Strong technical fit across retrieval, career depth, and JD alignment signals."`
2. `in_sweet_spot == True AND tier2_scaled > 0` → `"Within the 6–8 year target range with product-company background. {total_years_exp}y total experience."`
3. `cross_encoder_score >= 0.80` (raw value) → `"Closely matches role requirements on full-profile semantic evaluation."`
4. Default → `"Moderate technical signal across retrieval and career history dimensions."`

### Clause 2 — Supporting detail (always present)

If `technical_summary_sentence` is not null and not empty string:
→ use `technical_summary_sentence` directly.

Fallback if null or empty:
- `in_sweet_spot == True` → `"Experience within the 6–8 year target range."`
- `total_years_exp` not null → `"{total_years_exp}y total experience."`
- Both null → `"Profile details not available."`

### Clause 3 — Honesty clause (include only if a meaningful concern exists)

Evaluate in priority order. Use the first that applies. Omit entirely if none apply.

1. `avail_tier == "C" AND days_since_active > 180` → `"Note: inactive on platform for {days_since_active} days — availability uncertain."`
2. `avail_tier == "C" AND interview_completion_rate < 0.30` → `"Note: low interview completion rate ({int(interview_completion_rate*100)}%) — engagement concern."`
3. `notice_period_days > 90` → `"Notice period {notice_period_days} days raises the hiring bar per JD."`
4. `location_tier == "outside_india"` → `"Located outside India — no visa sponsorship per JD."`
5. `chase_pen > 0.06` (more than 2 short hops) → `"Title-chasing pattern noted: {short_hop_count} short tenures in history."`
6. `closed_pen > 0` → `"Closed-source background with limited external validation signals."`
7. `career_type == "mixed"` → `"Mixed product and consulting background — product-company fit is partial."`

### Assembly

```
reasoning = clause_1 + " " + clause_2
if clause_3 is not None:
    reasoning += " " + clause_3
```

Reasoning must not be empty. If somehow all clauses produce empty strings, use: `"Candidate ranked on composite of retrieval, career history, and availability signals."` as absolute fallback.

---

## 11. OUTPUT FILES

### 11.1 Submission CSV

**Path:** `artifacts/runtime/team_xxx.csv` (team_id from config)
**Encoding:** UTF-8, no BOM.
**Columns in exact order:** `candidate_id,rank,score,reasoning`
**Row count:** exactly 101 rows (1 header + 100 data rows).

**score column:** use `final_score` rounded to 6 decimal places.

**Pre-write validation — assert all of these, raise ValueError with descriptive message if any fails:**
1. Row count == 100
2. All ranks are unique integers from 1 to 100 inclusive
3. All candidate_ids are unique
4. All candidate_ids exist in the stage4_reranked.parquet input
5. `score` is monotonically non-increasing: score[i] >= score[i+1] for all i in 0..98
6. No null values in candidate_id, rank, or score columns
7. No empty strings in reasoning column

Use Polars `write_csv` with proper quoting so reasoning strings containing commas are correctly quoted.

### 11.2 Debug parquet

**Path:** `artifacts/runtime/stage5_full_scores.parquet`
**Contents:** ALL ~300 candidates with ALL intermediate columns listed in Section 8, plus:
- `in_top_100` (Boolean)
- `stage3_rank`, `stage4_rank` (carried from input)
- `q3_neg_sim`, `fused_score` (carried from input, not used in scoring)
- All reasoning clause values as separate columns for inspection

This file is for the defend-your-work interview and post-run analysis. It is NOT submitted.

---

## 12. STDOUT SUMMARY

Print after writing both files:

```
=== STAGE 5 COMPLETE ===
Input candidates:   {N}
t1_std (anchor):    {t1_std:.6f}

--- Tier 1 (Borda) ---
borda_primary:  min={min:.4f}  max={max:.4f}  mean={mean:.4f}  std={std:.4f}
Weights used:   ce={w_ce}  q1={w_q1}  q2={w_q2}  amplification_exp={exp}

--- Tier 2 (Career shape + penalties) ---
tier2_raw:      std={t2_std:.6f}
target_t2_std:  {target_t2_std:.6f}  (ratio={tier2_ratio})
rescale_factor: {target_t2_std/t2_std:.4f}
tier2_scaled:   min={min:.4f}  max={max:.4f}

--- Tier 3 (Availability) ---
Tier A:  {count_A} candidates  avail_unit=+1
Tier B:  {count_B} candidates  avail_unit= 0
Tier C:  {count_C} candidates  avail_unit=-1
t3_std: {t3_std:.6f}   target: {target_t3_std:.6f}   rescale: {factor:.4f}

--- Tier 4 (Logistics) ---
tier4_raw:      std={t4_std:.6f}
target_t4_std:  {target_t4_std:.6f}  (ratio={tier4_ratio})
rescale_factor: {target_t4_std/t4_std:.4f}

--- Final score ---
final_score:  min={min:.4f}  max={max:.4f}  mean={mean:.4f}  std={std:.4f}

--- Top 10 ---
rank | candidate_id   | final_score | borda_primary | tier2_scaled | avail_tier | location_tier
   1 | CAND_XXXXXXX   | 0.9821      | 0.9654        | 0.0142       | A          | preferred
   ...

--- Signals removed vs old formula (confirmed by diagnostics) ---
product_company_fraction:  NOT used (IQR=0.0, near-constant in pool)
has_any_production_role:   NOT used (95.3% triggered, near-constant)
q3_residual_penalty:       NOT used (std=0.0075, near-flat, double-penalizes)
fused_norm:                NOT used (amplified noise, raw std=0.0076)
stale_coding:              NOT used (0 triggers)
consulting_heavy:          NOT used (0 triggers)
market_factor:             NOT used (near-constant, 290/300 at ceiling)
resp_factor:               NOT used (215/300 at ceiling)
Multiplicative availability stack: REMOVED (caused 74% floor collapse)

Output written: artifacts/runtime/team_xxx.csv
Debug written:  artifacts/runtime/stage5_full_scores.parquet
```

---

## 13. WHAT IS REMOVED — COMPLETE LIST

Do not carry any of these forward. No config flags for them.

| Removed element | Reason from diagnostics |
|---|---|
| Old 7-layer formula entirely | Replaced by 4-pass cascade |
| `fused_norm` in Layer 1 | Raw std=0.0076, min-max was amplifying noise |
| `fused_score` in scoring | Represented by Q1+Q2 more cleanly; kept in debug only |
| `q3_residual_penalty` | q3_neg_sim std=0.0075, near-constant, double-penalizes |
| `product_company_fraction` in scoring | IQR=0.0, 86% at 1.0, near-constant after Stage 2 filtering |
| `has_any_production_role` in scoring | 95.3% triggered, effectively constant |
| `stale_coding` | 0 triggers in the 300-candidate pool |
| `consulting_heavy` / `consulting_resid_penalty` | 0 triggers in the 300-candidate pool |
| `market_factor` (availability) | 290/300 at ceiling, variance contribution -0.3% |
| `resp_factor` (availability) | 215/300 at ceiling, variance contribution -1.3% |
| `open_factor` (availability) | Dropped in favour of cleaner three-tier structure |
| Multiplicative availability stack (7-factor product) | 74% of candidates collapsed to 0.5 floor |
| `skill_score` | std=0.0 in Stage 5 pool, completely constant |
| `must_have_floor_multiplier` as separate layer | Spearman=0.998 with Q1 — redundant, Q1 already captures this |
| `shape_mult` as multiplicative layer | Replaced by additive tier2_scaled with rescaling |
| `core`, `core_floored`, `shaped`, `penalized`, `bonused` intermediate columns | Old pipeline intermediates — replaced by new 4-pass outputs |

---

## 14. SIGNALS RETAINED — COMPLETE LIST

| Signal | Source | Tier | Role |
|---|---|---|---|
| `cross_encoder_score` | stage4_reranked | 1 | Borda input, weight 0.25, raw rank (not amplified) |
| `q1_score` | stage4_reranked | 1 | Borda input, weight 0.35, rank amplified at ^1.4 |
| `q2_score` | stage4_reranked | 1 | Borda input, weight 0.40, rank amplified at ^1.4 |
| `in_sweet_spot` | stage4_reranked | 2 | sweet_bonus = +0.04 if True |
| `optional_bonus` | stage4_reranked | 2 | Already computed, used directly, range [0, 0.08] |
| `title_chasing_penalty` | stage4_reranked | 2 | Already computed, subtracted, range [0, 0.15] |
| `ambiguity_penalty` | stage4_reranked | 2 | Already computed, subtracted, range [0, 0.04] |
| `closed_source_penalty` | stage4_reranked | 2 | Already computed, subtracted, range [0, 0.05] |
| `interview_completion_rate` | candidates.jsonl | 3 | Availability tier classification |
| `offer_acceptance_rate` | candidates.jsonl | 3 | Availability tier classification (skip if null or -1) |
| `last_active_date` | candidates.jsonl | 3 | Converted to days_since_active |
| `location_tier` | stage4_reranked | 4 | Unit values: preferred=+1, acceptable=0, outside_india=-2, unknown=0 |
| `preferred_work_mode` | candidates.jsonl | 4 | Unit value: +1 if hybrid or flexible, else 0 |
| `notice_period_days` | candidates.jsonl | 4 | Unit values: <=30→+1, 31-90→0, >90→-1 |

Signals present in input but used ONLY for reasoning (not scoring):
`total_years_exp`, `career_type`, `short_hop_count`, `external_validation_score`,
`has_github`, `title_ambiguous`, `exp_band`, `technical_summary_sentence`,
`q3_neg_sim` (debug), `fused_score` (debug), `stage3_rank`, `stage4_rank`.

---

## 15. CONFIG BLOCK

Full required block in `config.yaml` under `stage5:`. Agent must read all values from here. No hardcoded numerics in scoring logic.

```yaml
stage5:

  team_id: "team_xxx"
  top_n: 100
  current_date: "auto"   # "auto" = use datetime.date.today(); or set explicit "YYYY-MM-DD"

  # --- Tier 1: Borda ---
  borda:
    w_ce:  0.25       # cross_encoder_score rank weight (raw rank, not amplified)
    w_q1:  0.35       # q1_score rank weight (amplified)
    w_q2:  0.40       # q2_score rank weight (amplified) — highest because std=0.401
    q_amplification_exponent: 1.4   # applied to rank_q1 and rank_q2 only

  # --- Cascade ratios (fraction of t1_std each tier may contribute) ---
  cascade:
    tier2_ratio: 0.25   # career shape + penalties: up to 25% of t1_std spread
    tier3_ratio: 0.15   # availability: up to 15% of t1_std spread
    tier4_ratio: 0.05   # logistics: up to 5% of t1_std spread

  # --- Tier 2: Career shape + penalties ---
  tier2:
    sweet_spot_bonus: 0.04   # added when in_sweet_spot == True

  # --- Tier 3: Availability thresholds ---
  availability:
    tier_a_interview_min:    0.70
    tier_a_offer_min:        0.60
    tier_a_recency_max_days: 30
    tier_c_interview_max:    0.30
    tier_c_recency_min_days: 180

  # --- Tier 4: Logistics unit values ---
  logistics:
    location_units:
      preferred:     1
      acceptable:    0
      outside_india: -2
      unknown:       0
    workmode_match_unit:  1    # applied when preferred_work_mode in {hybrid, flexible}
    notice_short_unit:    1    # notice_period_days <= 30
    notice_medium_unit:   0    # notice_period_days 31-90
    notice_long_unit:    -1    # notice_period_days > 90

  # --- Output paths ---
  output_csv:   "artifacts/runtime/team_xxx.csv"
  debug_parquet: "artifacts/runtime/stage5_full_scores.parquet"
```

---

## 16. CLI INTERFACE

```
python stages/stage5_rank.py \
    --input    artifacts/runtime/stage4_reranked.parquet \
    --jsonl    data/candidates.jsonl \
    --config   config.yaml \
    --output   artifacts/runtime/team_xxx.csv
```

All four arguments required. Fail with usage message if any missing.

Also expose a `run(input_path, jsonl_path, output_path, config)` function callable from `rank.py` wrapper.

---

## 17. DEPENDENCIES

No new dependencies. Uses only what is already in `requirements.txt`:

```
polars>=0.20.0
numpy>=1.24.0
pyyaml>=6.0
```

No LightGBM, ONNX, FAISS, torch, transformers, or API calls anywhere in this script.
