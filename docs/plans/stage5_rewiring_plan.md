# STAGE 5 — REWIRING PLAN (v1 + Option 1+2)
## Implementation Specification for AI Coding Agent

**READ THIS ENTIRE DOCUMENT BEFORE WRITING ANY CODE.**
**DO NOT ASSUME ANY VALUE, COLUMN NAME, WEIGHT, OR THRESHOLD NOT EXPLICITLY STATED HERE.**
**DO NOT CARRY FORWARD ANY LOGIC FROM THE OLD STAGE 5 FORMULA UNLESS EXPLICITLY LISTED AS RETAINED.**

---

## 1. CONTEXT AND SCOPE

This document specifies a **complete replacement** of the Stage 5 final scoring formula.
The old 7-layer formula is being removed entirely. There is no legacy flag, no fallback
path, no config toggle. The new formula (referred to as v1 + Option 1+2) replaces it
in full.

**What this script does:**
Takes ~300 candidates surviving from Stage 4, applies the new 4-tier composite scoring
formula, sorts to produce the final ranking, generates per-candidate reasoning text, and
writes the top-100 submission CSV.

**What this script does NOT do:**
- No FAISS, no ONNX, no embeddings, no LLM calls, no model inference of any kind.
- No network calls.
- No re-reading of the original candidates.jsonl.
- No processing of all 100K candidates — only the ~300 Stage 4 survivors.

---

## 2. FILE LOCATION

**Script to modify/replace:** `stages/stage5_rank.py`
(It may also exist as `stage5_lgbm.py` — if so, replace that file. Do not create a new
file alongside it; replace it in place.)

---

## 3. INPUT CONTRACTS

### 3.1 Primary input — Stage 4 output

**File:** `artifacts/runtime/stage4_reranked.parquet`
**Format:** Polars DataFrame, approximately 300 rows.

**Required columns from this file (exact names — do not assume aliases):**

| Column name | Type | Description |
|---|---|---|
| `candidate_id` | String | Primary key. Format: `CAND_XXXXXXX` |
| `cross_encoder_score` | Float64 | Stage 4 cross-encoder similarity score. Range approximately [0, 1]. Higher = better. |
| `q1_score` | Float64 | Stage 3 dense retrieval score for Q1 (career history / retrieval experience subspace). Raw dot product — NOT normalized. |
| `q2_score` | Float64 | Stage 3 dense retrieval score for Q2 (product-company career trajectory subspace). Raw dot product — NOT normalized. |
| `q3_neg_sim` | Float64 | Stage 3 anti-pattern similarity. Higher = more similar to anti-pattern query. DO NOT USE IN SCORING — this column is retained for debug output only. |
| `fused_score` | Float64 | Stage 3 RRF fusion score. DO NOT USE IN SCORING — retained for debug only. |
| `product_company_fraction` | Float64 | Fraction of career at product companies. Range [0, 1]. Computed in Stage 2. |
| `has_any_production_role` | Boolean | True if candidate has at least one production engineering role. Computed in Stage 2. |
| `total_years_exp` | Int64 | Total years of experience. Used only for reasoning text — not in scoring formula. |
| `in_sweet_spot` | Boolean | True if total_years_exp is between 6 and 8 inclusive. Computed in Stage 2. |
| `location_tier` | String | Enum: `preferred`, `acceptable`, `outside_india`, `unknown`. Computed in Stage 2. |
| `career_type` | String | Enum: `product`, `consulting`, `mixed`, `unknown`. Computed in Stage 2. |
| `technical_summary_sentence` | String | Pre-computed 30-word technical summary for reasoning column. From Track B precompute. May be null — see Section 9 for fallback. |

**VERIFY BEFORE PROCEEDING:** Print the column list of stage4_reranked.parquet at
script startup. If any of the above columns are missing, raise a descriptive ValueError
naming every missing column. Do not proceed with nulls substituted silently for missing
structural columns.

### 3.2 Secondary input — Behavioral signals

**File:** `artifacts/precomputed/candidate_features.parquet`
**Format:** Polars DataFrame, up to 100,000 rows (full candidate pool).

**Required columns from this file:**

| Column name | Type | Description |
|---|---|---|
| `candidate_id` | String | Join key. Must match format in Stage 4 output. |
| `interview_completion_rate` | Float64 | Redrob signal #19. Range [0, 1]. Fraction of interviews attended. |
| `offer_acceptance_rate` | Float64 | Redrob signal #20. Range [-1, 1]. -1 means no prior offers — NOT a bad signal. |
| `last_active_date` | String | Redrob signal #3. ISO date string `YYYY-MM-DD`. Date of last platform login. |
| `notice_period_days` | Int64 | Redrob signal #12. Range [0, 180]. Stated notice period. |
| `preferred_work_mode` | String | Redrob signal #14. Enum: `onsite`, `hybrid`, `remote`, `flexible`. |

**Join instruction:** After loading both files, perform a LEFT JOIN from stage4_reranked
onto candidate_features on `candidate_id`. The left side (stage4_reranked) is the master
— never drop a candidate because they are missing from candidate_features. If a candidate
is missing from candidate_features, all behavioral signal columns will be null for them
— apply the null handling defaults specified in Section 6.

**Load only the required columns from candidate_features.parquet.** Do not load all
columns — the file may be large. Use Polars column selection at read time:
```python
pl.read_parquet("...", columns=["candidate_id", "interview_completion_rate",
                                 "offer_acceptance_rate", "last_active_date",
                                 "notice_period_days", "preferred_work_mode"])
```

### 3.3 Config input

**File:** `config.yaml`, under the `stage5:` namespace.

The full required config block is specified in Section 10. The script must read all
parameters from config — no hardcoded values in the scoring logic. Every weight,
threshold, and adjustment value referenced below corresponds to a config key defined
in Section 10.

---

## 4. THE FORMULA — OVERVIEW

The new formula has exactly 4 tiers applied sequentially. All tiers are additive at
assembly. There are NO multiplicative layers anywhere in the new formula.

```
final_score = borda_primary + shape_adj + avail_adj + logistics_adj
```

| Tier | Name | Max positive contribution | Max negative contribution |
|---|---|---|---|
| 1 | Borda primary | ~1.0 | ~0.0 |
| 2 | Career shape | +0.08 | 0.00 |
| 3 | Availability | +0.04 | -0.04 |
| 4 | Logistics | +0.018 | -0.036 |

Tier 1 dominates. Tiers 2–4 are bounded adjustments. No signal in Tiers 2–4 can
override a meaningful technical gap created by Tier 1.

---

## 5. TIER 1 — BORDA PRIMARY SCORE

This is the foundation. It combines the three retrieval/reranking scores using
amplified rank combination (Option 1: reweighting + Option 2: rank amplification).

### Step 1 — Compute dense ranks

Convert each of the three scores to integer ranks over the ~300 candidates.
Rank 1 = highest score (best candidate). Use **dense ranking**: if two candidates
tie on a score, both receive the same rank and the next rank is NOT skipped.

```
rank_ce  = dense_rank(cross_encoder_score, descending=True)   # Range [1, N]
rank_q1  = dense_rank(q1_score,            descending=True)   # Range [1, N]
rank_q2  = dense_rank(q2_score,            descending=True)   # Range [1, N]
```

Where N = number of candidates in the Stage 4 output (approximately 300, but use the
actual count — do not hardcode 300).

### Step 2 — Amplify Q1 and Q2 ranks only

Apply a convex power transformation to Q1 and Q2 ranks. CE is NOT amplified — it
stays as a raw integer rank. This asymmetry is intentional: CE captures declared
skills and keyword fit (linear treatment). Q1 and Q2 capture career history depth
via dense vector subspaces (convex treatment to widen separation at the top).

```
rank_q1_amp = rank_q1 ^ 1.4
rank_q2_amp = rank_q2 ^ 1.4
```

The exponent is exactly **1.4**. Do not change this without explicit instruction.

What this does concretely (for reference, do not hardcode these):
- rank 1  → 1.0
- rank 10 → 25.1
- rank 50 → 264.0
- rank 100 → 630.9
- rank 300 → 3085.0

The amplified ranks have a much larger range than raw ranks. This is expected and
handled in Step 4.

### Step 3 — Weighted Borda sum

Combine the three rank values (two amplified, one raw) using fixed weights:

```
borda_sum = (w_ce × rank_ce) + (w_q1 × rank_q1_amp) + (w_q2 × rank_q2_amp)
```

Weights (from config — see Section 10):
- `w_ce  = 0.35`  (cross-encoder, raw rank)
- `w_q1  = 0.35`  (Q1 score, amplified rank)
- `w_q2  = 0.30`  (Q2 score, amplified rank)

Weights sum to exactly 1.0. Lower borda_sum = technically stronger candidate.

### Step 4 — Normalize to [0, 1] and invert

Because the amplified ranks produce a large raw range, normalize borda_sum using
min-max over the ACTUAL values in the current pool (not a fixed constant):

```
borda_min = min(borda_sum over all candidates)
borda_max = max(borda_sum over all candidates)

borda_primary = 1.0 - (borda_sum - borda_min) / (borda_max - borda_min)
```

After this operation: higher borda_primary = better candidate. Range is exactly [0, 1].

**Edge case:** if borda_max == borda_min (all candidates have identical scores — should
never happen in practice but must be handled), set borda_primary = 0.5 for all
candidates and log a WARNING.

---

## 6. TIER 2 — CAREER SHAPE ADJUSTMENT

A bounded additive term rewarding product-company background and production experience.
This tier has NO negative contribution — it can only add, never subtract.

```
shape_adj = (product_company_fraction × w_pcf) + (has_any_production_role_int × w_hpr)
```

Where:
- `product_company_fraction` — float [0, 1] from Stage 4 input (see Section 3.1)
- `has_any_production_role_int` — cast the boolean `has_any_production_role` to integer
  (True → 1, False → 0) before multiplying
- `w_pcf = 0.06` (from config)
- `w_hpr = 0.02` (from config)

Maximum possible shape_adj = **0.06 + 0.02 = 0.08**
Minimum possible shape_adj = **0.00**

**Null handling:** if `product_company_fraction` is null, treat as 0.0.
If `has_any_production_role` is null, treat as False (0).

---

## 7. TIER 3 — AVAILABILITY TIERED ADJUSTMENT

Classifies each candidate into one of three availability tiers based on the three
highest-variance behavioral subfactors. Applied as a fixed additive adjustment per tier.

### Step 1 — Compute days_since_active

Parse `last_active_date` (string, format `YYYY-MM-DD`) to a date. Compute:

```
days_since_active = (run_date - last_active_date).days
```

Where `run_date` = the date the script is executed (use `datetime.date.today()`).

**Null handling for last_active_date:** if null or unparseable, set
`days_since_active = 999` (treated as very stale → Tier C condition may trigger).

### Step 2 — Classify tier

Evaluate conditions in this exact order. Tier C is checked first because unavailability
is a stronger signal than availability:

**Tier C — Clearly unavailable** (any ONE of these conditions is sufficient):
- `interview_completion_rate < 0.30`
- `days_since_active > 180`

→ `avail_adj = -0.04`

**Tier A — Clearly available** (ALL THREE conditions must hold simultaneously):
- `interview_completion_rate >= 0.70`
- `offer_acceptance_rate >= 0.60`  ← skip this condition entirely if offer_acceptance_rate == -1
- `days_since_active <= 30`

→ `avail_adj = +0.04`

**Tier B — Uncertain / default** (everything not classified as Tier A or Tier C):
→ `avail_adj = 0.00`

**Null handling for behavioral signals:**
- `interview_completion_rate` null → treat as Tier B (do not trigger Tier C)
- `offer_acceptance_rate` null → same as -1 (skip the offer condition in Tier A check)
- `offer_acceptance_rate == -1` → skip the offer condition in Tier A check (not a penalty)

**Store the tier label** as a string column `avail_tier` (values: `"A"`, `"B"`, `"C"`)
for use in the reasoning column and debug output.

---

## 8. TIER 4 — LOGISTICS TIEBREAKER

Three sub-adjustments applied additively. All values are scaled down 5× relative to
the old Layer 7 values. Maximum positive swing: +0.018. Maximum negative swing: -0.036.

### Sub-adjustment 1 — Location

Read `location_tier` from Stage 4 input (string enum computed in Stage 2).

```
location_adj = {
    "preferred":     +0.006,
    "acceptable":     0.000,
    "unknown":        0.000,
    "outside_india": -0.020,
}[location_tier]
```

If `location_tier` is null or any value not in the above enum, treat as `"unknown"` → 0.000.

**What each tier maps to (for reference — these were computed in Stage 2, not here):**
- `preferred`: Noida, Pune, Delhi, Delhi NCR, Gurgaon, Gurugram, Faridabad, New Delhi
  (or any `acceptable` city where `willing_to_relocate == True`)
- `acceptable`: Hyderabad, Mumbai, Bangalore, Bengaluru, Chennai
- `outside_india`: any location clearly outside India (no visa sponsorship per JD)
- `unknown`: location could not be determined

### Sub-adjustment 2 — Work mode

Read `preferred_work_mode` from candidate_features (Redrob signal #14).

The JD specifies hybrid as the preferred work mode (offices in Noida and Pune, used
Tue/Thu, flexible cadence).

```
workmode_adj = +0.006 if preferred_work_mode in {"hybrid", "flexible"} else 0.000
```

If `preferred_work_mode` is null, treat as 0.000.

### Sub-adjustment 3 — Notice period

Read `notice_period_days` from candidate_features (Redrob signal #12).

```
if notice_period_days is null:     notice_adj = 0.000   # unknown = neutral
elif notice_period_days <= 30:     notice_adj = +0.006  # ideal, JD says sub-30 preferred
elif notice_period_days <= 90:     notice_adj =  0.000  # acceptable range
else:                              notice_adj = -0.016  # >90 days, bar gets higher per JD
```

### Assembly

```
logistics_adj = location_adj + workmode_adj + notice_adj
```

---

## 9. FINAL SCORE ASSEMBLY

```
final_score = borda_primary + shape_adj + avail_adj + logistics_adj
```

All four terms are floats. Addition only. No multiplication, no clamping after assembly.

**Expected range by construction:**
- Minimum theoretical: 0.0 + 0.0 + (-0.04) + (-0.036) = approximately -0.076
- Maximum theoretical: 1.0 + 0.08 + 0.04 + 0.018 = approximately 1.138

**Store all intermediate columns** in the working DataFrame for debug output:
`rank_ce`, `rank_q1`, `rank_q2`, `rank_q1_amp`, `rank_q2_amp`, `borda_sum`,
`borda_primary`, `shape_adj`, `avail_adj`, `avail_tier`, `location_adj`,
`workmode_adj`, `notice_adj`, `logistics_adj`, `final_score`.

---

## 10. SORTING, SLICING, AND MONOTONICITY

### Sort

Sort the DataFrame by `final_score` descending.

**Tie-breaking:** if two candidates have identical `final_score` after all four tiers,
break the tie deterministically by `candidate_id` ascending (lexicographic string sort).
This ensures reproducibility across runs.

### Slice top 100

Take the first 100 rows after sorting. Assign `rank` as integers 1 through 100
(1-indexed, 1 = best candidate).

### Monotonicity enforcement

After slicing, verify that `final_score[i] >= final_score[i+1]` for all i in 0..98.

If any violation exists (can occur at tie boundaries due to the secondary sort):
```
for i in range(1, 100):
    if final_score[i] > final_score[i-1]:
        final_score[i] = final_score[i-1]
```

This is a safety rail that should rarely trigger. Log a WARNING if it does trigger,
with the count of positions corrected.

---

## 11. REASONING COLUMN GENERATION

The reasoning column is heavily weighted in Stage 4 manual review. It must be specific
(real facts from the candidate's profile), connected to JD requirements, honest about
concerns, non-templated across candidates, and consistent with rank.

**No LLM, no API calls.** Construct reasoning from actual score decomposition columns.

### Construction logic

Build each reasoning string from 2–3 clauses. The clauses are selected based on
the candidate's actual dominant factors — not filled with fixed template text.

**Clause 1 — Primary strength (always present, pick the strongest true signal):**

Evaluate in this priority order and pick the first that applies:
1. If `borda_primary >= 0.75`: use `"Strong technical fit across retrieval, career depth, and JD alignment signals."`
2. Else if `product_company_fraction >= 0.7`: use `"{total_years_exp}y exp, {int(product_company_fraction*100)}% at product companies."`
3. Else if `cross_encoder_score >= 0.80` (unnormalized raw value): use `"Closely matches role requirements on full-profile semantic evaluation."`
4. Default: use `"Moderate technical signal across retrieval and career history dimensions."`

**Clause 2 — Supporting detail (always present, pick from pre-computed summary):**

If `technical_summary_sentence` is not null and not empty:
    use the value of `technical_summary_sentence` directly (it is pre-computed offline).

If `technical_summary_sentence` IS null or empty (fallback):
    Construct from real columns:
    - if `in_sweet_spot == True`: `"Experience within the 6–8 year target range."`
    - elif `total_years_exp is not null`: `"{total_years_exp}y total experience."`
    - else: `"Experience details not available."`

**Clause 3 — Honesty clause (present only if a meaningful concern exists):**

Evaluate in this priority order. Use the first that applies. If none apply, omit Clause 3.

1. `avail_tier == "C" and days_since_active > 180`: `"Note: inactive on platform for {days_since_active} days — availability uncertain."`
2. `avail_tier == "C" and interview_completion_rate < 0.30`: `"Note: low interview completion rate ({int(interview_completion_rate*100)}%) — engagement concern."`
3. `notice_period_days > 90`: `"Notice period {notice_period_days} days raises the hiring bar per JD."`
4. `location_tier == "outside_india"`: `"Located outside India — no visa sponsorship per JD."`
5. `career_type == "consulting"`: `"Significant consulting background — product-company fit is partial."`
6. `has_any_production_role == False`: `"No clear production engineering role identified in history."`

### Assembly

```
reasoning = clause_1 + " " + clause_2
if clause_3 exists:
    reasoning += " " + clause_3
```

**Anti-hallucination rule:** every fact in every clause must come from an actual column
value. Never invent company names, skill names, scores, or years that are not present
as real column values. If a column is null, use the fallback text, not an invented value.

---

## 12. CSV OUTPUT

**Filename:** `artifacts/runtime/team_xxx.csv`
(Replace `xxx` with the actual team ID if known from config. If not in config, use
literal `team_xxx.csv`.)

**Encoding:** UTF-8, no BOM.

**Columns, in this exact order:**
```
candidate_id,rank,score,reasoning
```

**Column types in output:**
- `candidate_id`: string, no quotes unless the value contains a comma (it should not)
- `rank`: integer, 1 through 100, no decimals
- `score`: float, rounded to 6 decimal places
- `reasoning`: string, double-quoted if it contains commas (use csv.writer or Polars
  write_csv with proper quoting — do not manually format)

**Row count:** exactly 101 rows (1 header + 100 data rows). Validate before writing.

**Pre-write validation checklist (assert all of these — raise ValueError with
a descriptive message if any fails):**
1. Row count == 100
2. All ranks are unique integers from 1 to 100 inclusive
3. All candidate_ids are unique
4. All candidate_ids exist in the original stage4_reranked.parquet input
5. `score` is monotonically non-increasing: score[i] >= score[i+1] for all i in 0..98
6. No null values in candidate_id, rank, or score columns
7. No empty strings in reasoning column

---

## 13. DEBUG OUTPUT

Write a second parquet file (NOT submitted, for analysis only):

**File:** `artifacts/runtime/stage5_full_scores.parquet`

**Contents:** ALL ~300 candidates (not just top 100) with ALL intermediate columns:
`candidate_id`, `cross_encoder_score`, `q1_score`, `q2_score`, `rank_ce`, `rank_q1`,
`rank_q2`, `rank_q1_amp`, `rank_q2_amp`, `borda_sum`, `borda_primary`,
`product_company_fraction`, `has_any_production_role`, `shape_adj`,
`interview_completion_rate`, `offer_acceptance_rate`, `days_since_active`,
`avail_tier`, `avail_adj`, `location_tier`, `notice_period_days`,
`preferred_work_mode`, `location_adj`, `workmode_adj`, `notice_adj`,
`logistics_adj`, `final_score`, plus a boolean column `in_top_100`.

This file is used for the defend-your-work interview and for diagnosing ranking
decisions post-run.

---

## 14. STDOUT SUMMARY

After writing both output files, print a human-readable summary to stdout:

```
=== STAGE 5 COMPLETE ===
Input candidates:        {N}
Output top-100 written:  artifacts/runtime/team_xxx.csv

--- Tier 1 (Borda) ---
borda_primary range:     [{min:.4f}, {max:.4f}]  mean={mean:.4f}  std={std:.4f}

--- Tier 2 (Career shape) ---
shape_adj range:         [{min:.4f}, {max:.4f}]  mean={mean:.4f}

--- Tier 3 (Availability) ---
Tier A (available):      {count_A} candidates  (+0.04)
Tier B (uncertain):      {count_B} candidates  (0.00)
Tier C (unavailable):    {count_C} candidates  (-0.04)

--- Tier 4 (Logistics) ---
logistics_adj range:     [{min:.4f}, {max:.4f}]

--- Final score ---
final_score range:       [{min:.4f}, {max:.4f}]  mean={mean:.4f}  std={std:.4f}

--- Top 10 candidates ---
rank | candidate_id  | final_score | borda_primary | avail_tier | location_tier
  1  | CAND_XXXXXXX  | 0.9432      | 0.9871        | A          | preferred
  ...

--- Signals NOT used (removed in v1 rewiring) ---
fused_score:     present in input but excluded from scoring (debug only)
q3_neg_sim:      present in input but excluded from scoring (debug only)
stale_coding:    removed (0 triggers in dataset)
consulting_heavy: removed (0 triggers in dataset)
market_factor:   removed (near-constant)
resp_factor:     removed (72% at ceiling)
```

---

## 15. CONFIG BLOCK (complete)

The following must be present in `config.yaml` under the `stage5:` namespace.
The coding agent must read all values from config — no hardcoded values in scoring logic.

```yaml
stage5:

  # --- Tier 1: Borda weights and amplification ---
  borda:
    w_ce: 0.35          # Weight for cross_encoder_score rank (raw, not amplified)
    w_q1: 0.35          # Weight for q1_score rank (amplified)
    w_q2: 0.30          # Weight for q2_score rank (amplified)
    q_amplification_exponent: 1.4   # Exponent applied to rank_q1 and rank_q2 only

  # --- Tier 2: Career shape ---
  career_shape:
    w_product_company_fraction: 0.06
    w_has_any_production_role: 0.02

  # --- Tier 3: Availability tiers ---
  availability:
    tier_a_interview_min: 0.70        # interview_completion_rate threshold for Tier A
    tier_a_offer_min: 0.60            # offer_acceptance_rate threshold for Tier A
    tier_a_recency_max_days: 30       # days_since_active threshold for Tier A
    tier_c_interview_max: 0.30        # interview_completion_rate threshold for Tier C
    tier_c_recency_min_days: 180      # days_since_active threshold for Tier C
    tier_a_adj: +0.04
    tier_b_adj: 0.00
    tier_c_adj: -0.04

  # --- Tier 4: Logistics ---
  logistics:
    location_preferred_adj: +0.006
    location_acceptable_adj: 0.000
    location_unknown_adj: 0.000
    location_outside_india_adj: -0.020
    workmode_match_adj: +0.006        # applied when preferred_work_mode is hybrid or flexible
    notice_short_adj: +0.006          # notice_period_days <= 30
    notice_medium_adj: 0.000          # notice_period_days 31-90
    notice_long_adj: -0.016           # notice_period_days > 90

  # --- Output ---
  output_path: "artifacts/runtime/team_xxx.csv"
  debug_output_path: "artifacts/runtime/stage5_full_scores.parquet"
  top_k: 100
```

---

## 16. WHAT IS REMOVED — COMPLETE LIST

The following signals and logic from the OLD Stage 5 formula are **fully removed**.
Do not carry any of these forward. Do not add config flags for them.

| Removed element | Reason |
|---|---|
| `fused_norm` (min-max normalized fused_score in Layer 1) | std=0.0075 raw; normalization was amplifying noise into fake signal |
| `fused_score` in scoring formula | Represented by Q1 and Q2 more cleanly; kept in debug output only |
| `q3_residual_penalty` (Layer 4) | q3_neg_sim std=0.0075, near-constant; double-penalized Q3 already in fused_score |
| `stale_coding` boolean penalty | 0 triggers in the 300-candidate Stage 5 pool |
| `consulting_heavy` boolean penalty | 0 triggers in the 300-candidate Stage 5 pool |
| `market_factor` in availability | Near-constant subfactor (profile_views, search_appearance); contributed <0.5% variance |
| `resp_factor` in availability | recruiter_response_rate: 72% at ceiling (1.0), contributed 1.3% variance |
| `open_factor` in availability | Borderline — dropped in favor of the three-tier structure which is cleaner |
| Multiplicative availability stack (7-factor product) | Caused 74% of candidates to collapse to the 0.5 floor — broken architecture |
| `skill_score` | std=0.0 in Stage 5 pool (constant); excluded until upstream stub is fixed |
| `title_chasing_penalty` (Layer 4) | Deferred — not enough signal at this stage without labeled data |
| All of Layer 2 (must-have floor multiplier) | Replaced by Tier 1 Borda which naturally depresses candidates thin on Q1/Q2 |
| All of Layer 5 (optional bonuses) | Deferred — additive bonuses for LoRA/XGBoost/OSS not included in v1 |
| Layer 7 logistics at old magnitude | Replaced by Tier 4 at 5× lower magnitude |

---

## 17. SIGNALS RETAINED — COMPLETE LIST

| Signal | Tier | Role in formula |
|---|---|---|
| `cross_encoder_score` | 1 | Borda input, weight 0.35, raw rank (not amplified) |
| `q1_score` | 1 | Borda input, weight 0.35, rank amplified at exponent 1.4 |
| `q2_score` | 1 | Borda input, weight 0.30, rank amplified at exponent 1.4 |
| `product_company_fraction` | 2 | Continuous, max +0.06 |
| `has_any_production_role` | 2 | Boolean cast to int, max +0.02 |
| `interview_completion_rate` | 3 | Availability tier classification |
| `offer_acceptance_rate` | 3 | Availability tier classification (skipped if -1) |
| `last_active_date` | 3 | Converted to days_since_active for tier classification |
| `location_tier` | 4 | Logistics tiebreaker, range [-0.020, +0.006] |
| `preferred_work_mode` | 4 | Logistics tiebreaker, +0.006 if hybrid or flexible |
| `notice_period_days` | 4 | Logistics tiebreaker, range [-0.016, +0.006] |

Signals present in input but used ONLY for reasoning text (not scoring):
`total_years_exp`, `in_sweet_spot`, `career_type`, `technical_summary_sentence`,
`q3_neg_sim` (debug only), `fused_score` (debug only).

---

## 18. DEPENDENCIES

No new dependencies required. Stage 5 uses only:

```
polars>=0.20.0
numpy>=1.24.0
pyyaml>=6.0
```

No LightGBM, no ONNX, no FAISS, no torch, no transformers, no API calls.

---

## 19. CLI INTERFACE

The script must expose a CLI with these arguments:

```
python stages/stage5_rank.py \
    --input   artifacts/runtime/stage4_reranked.parquet \
    --features artifacts/precomputed/candidate_features.parquet \
    --config  config.yaml \
    --output  artifacts/runtime/team_xxx.csv
```

All four arguments are required. Fail loudly with a usage message if any is missing.
The script must also expose a `run(input_path, features_path, output_path, config)`
function (in addition to the CLI) so it can be called programmatically from `rank.py`.
