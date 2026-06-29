# Stage 5 — Implementation Plan
**Formula version: v1 (Checkpoint locked)**

---

## Overview

Stage 5 takes the Top 300 candidates from Stage 3/4 and produces the final ranked Top 100 using the redesigned formula. This is a pure Python + Polars script — no model inference, no API calls, no heavy frameworks.

---

## Input Contracts

| Input | Source | Format |
|---|---|---|
| `stage5_candidates.parquet` | Stage 3/4 output | Polars DataFrame, 300 rows |
| Required columns | Track B payload | See column spec below |

### Required columns from parquet

| Column | Type | Source stage |
|---|---|---|
| `candidate_id` | string | All stages |
| `ce_score` | float [0,1] | Stage 4 cross-encoder |
| `q1_norm` | float [0,1] | Stage 3 dense retrieval |
| `q2_norm` | float [0,1] | Stage 3 hybrid fusion |
| `product_company_fraction` | float [0,1] | Precompute Track B |
| `has_any_production_role` | bool | Precompute Track B |
| `interview_completion_rate` | float [0,1] | Redrob signal #19 |
| `offer_acceptance_rate` | float [-1,1] | Redrob signal #20 |
| `last_active_date` | date string | Redrob signal #3 |
| `location_bucket` | string enum | Precompute Track B |
| `preferred_work_mode` | string enum | Redrob signal #14 |
| `notice_period_days` | int | Redrob signal #12 |
| `technical_summary_sentence` | string | Precompute Track B |

---

## Step-by-Step Implementation

---

### Step 1 — Load and validate input

Load `stage5_candidates.parquet` into a Polars DataFrame. Assert exactly 300 rows. Assert all required columns present. Assert no nulls in scoring columns (fill with sensible defaults if needed — document each default).

**Null handling defaults (document in code comments):**
- `offer_acceptance_rate == -1` → treat as Tier B (neutral), not a penalty
- `last_active_date` missing → treat as > 180 days (Tier C)
- `interview_completion_rate` missing → treat as Tier B

---

### Step 2 — Tier 1: Borda rank computation

Convert each of CE, Q1, Q2 into integer ranks among the 300 candidates. Rank 1 = highest score. Use dense ranking (no gaps for ties — if two candidates tie on CE, both get the same rank, next rank is not skipped).

Compute weighted Borda sum:

```
borda_sum = (0.50 × rank_ce) + (0.25 × rank_q1) + (0.25 × rank_q2)
```

Lower borda_sum = technically stronger candidate.

Invert and normalize to [0, 1]:

```
borda_primary = (300 - borda_sum) / 300
```

This is the dominant signal. Maximum value ≈ 1.0, minimum ≈ 0.0.

---

### Step 3 — Tier 2: Career shape additive adjustment

Compute shape_adj as a bounded additive term:

```
shape_adj = (product_company_fraction × 0.06) + (has_any_production_role × 0.02)
```

- `product_company_fraction` is a float — fractional contribution already.
- `has_any_production_role` is bool — cast to 0 or 1 before multiplying.
- Maximum possible shape_adj = **+0.08**
- Minimum possible shape_adj = **0.00** (no negative penalty here)

---

### Step 4 — Tier 3: Availability tiered adjustment

Compute `days_since_active` from `last_active_date` relative to run date.

Classify each candidate into one of three tiers:

**Tier A — Clearly available** (all three conditions must hold):
- `interview_completion_rate >= 0.70`
- `offer_acceptance_rate >= 0.60` (ignore if == -1, treat as not Tier A)
- `days_since_active <= 30`

→ `avail_adj = +0.04`

**Tier C — Clearly unavailable** (any one condition sufficient):
- `interview_completion_rate < 0.30`
- `days_since_active > 180`

→ `avail_adj = -0.04`

**Tier B — Uncertain / default** (everything else):
→ `avail_adj = 0.00`

Maximum swing: **±0.04** — cannot override a meaningful Tier 1 gap.

---

### Step 5 — Tier 4: Logistics tiebreaker

Three sub-adjustments, all scaled down 5× from original values:

**Location:**
- `india_preferred_city` (Pune, Noida, Hyderabad, Mumbai, Delhi NCR) → `+0.006`
- `outside_india` → `-0.020`
- All other India locations → `0.000`

**Work mode:**
- `preferred_work_mode` matches JD preference (hybrid) → `+0.006`
- Otherwise → `0.000`

**Notice period:**
- `notice_period_days <= 30` → `+0.006`
- `notice_period_days > 90` → `-0.016`
- 31–90 days → `0.000`

```
logistics_adj = location_adj + workmode_adj + notice_adj
```

Maximum positive: **+0.018**
Maximum negative: **-0.036**

---

### Step 6 — Final score assembly

```
final_score = borda_primary + shape_adj + avail_adj + logistics_adj
```

All additive. No multipliers. No compounding.

**Expected score range:**

| Component | Min | Max |
|---|---|---|
| borda_primary | ~0.00 | ~1.00 |
| shape_adj | 0.00 | +0.08 |
| avail_adj | -0.04 | +0.04 |
| logistics_adj | -0.036 | +0.018 |
| **final_score** | **~-0.08** | **~1.14** |

Tier 1 dominates. Tiers 2–4 are bounded and cannot flip a technically differentiated ranking.

---

### Step 7 — Sort and slice Top 100

Sort DataFrame by `final_score` descending. In case of exact score ties, break deterministically by `candidate_id` ascending (string sort). Slice top 100 rows. Assign `rank` as 1 through 100 (integer, 1-indexed).

---

### Step 8 — Monotonicity enforcement

After slicing, assert that `final_score` at rank N >= `final_score` at rank N+1 for all N in 1..99. If any violation exists (can happen with tie-breaking), apply a tiny epsilon decrement to enforce strict non-increase:

```
For each row where score[i] > score[i-1]: score[i] = score[i-1]
```

This is a safety rail — should rarely trigger given the deterministic sort.

---

### Step 9 — Reasoning column assembly

For each of the Top 100, look up `technical_summary_sentence` from Track B payload (pre-computed offline, already in the parquet).

Format:

```
reasoning = f"{technical_summary_sentence} [Availability: {tier}; Notice: {notice_period_days}d]"
```

This adds just enough signal-specific context to satisfy the Stage 4 reasoning checks (specific facts, JD connection, honest concerns) without requiring online LLM generation.

---

### Step 10 — CSV output

Write to `team_xxx.csv` (UTF-8) with exactly these columns in order:

```
candidate_id, rank, score, reasoning
```

101 rows total (1 header + 100 data rows). Validate before writing:
- Exactly 100 unique candidate_ids
- Ranks 1–100 each appearing exactly once
- Scores monotonically non-increasing
- No empty reasoning strings

---

## What is removed vs previous formula

| Signal | Decision | Reason |
|---|---|---|
| `fused_norm` | Removed | min-max was stretching std=0.0075 noise into fake [0,1] signal |
| `q3_residual_penalty` | Removed | Near-constant, double-penalizes Q3 already in fused |
| `stale_coding` | Removed | 0 triggers in dataset |
| `consulting_heavy` | Removed | 0 triggers in dataset |
| `market_factor` | Removed | Near-constant subfactor in availability |
| `resp_factor` | Removed | 72% at ceiling, 1.3% contribution |
| Multiplicative availability stack | Removed | Caused 74%/floor collapse in 300 candidates |
| `title_ambiguous` penalty | Deferred | Discussed but not included in v1 — revisit if needed |

---

## Signals retained and their tier

| Signal | Tier | Role |
|---|---|---|
| `ce_score` | 1 | Primary ranker (weight 0.50 in Borda) |
| `q1_norm` | 1 | Primary ranker (weight 0.25 in Borda) |
| `q2_norm` | 1 | Primary ranker (weight 0.25 in Borda) |
| `product_company_fraction` | 2 | Career shape, continuous, max +0.06 |
| `has_any_production_role` | 2 | Career shape, boolean, max +0.02 |
| `interview_completion_rate` | 3 | Availability tier classification |
| `offer_acceptance_rate` | 3 | Availability tier classification |
| `last_active_date` | 3 | Availability tier classification |
| `location_bucket` | 4 | Logistics tiebreaker |
| `preferred_work_mode` | 4 | Logistics tiebreaker |
| `notice_period_days` | 4 | Logistics tiebreaker |

---

## Dependency requirements (Stage 5 only)

No new dependencies beyond what is already in `requirements.txt`:

```
polars>=0.20.0
numpy>=1.24.0
```

No LightGBM, no ONNX, no FAISS — Stage 5 is pure tabular computation.

---

## File produced

`team_xxx.csv` — 101 rows, UTF-8, columns: candidate_id, rank, score, reasoning.
