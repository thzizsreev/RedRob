# REDROB HACKATHON — FINAL CONSOLIDATED RANKING PLAN
## Complete JD Coverage Across All Stages

This is the master plan. It cross-checks **every line of the job description** against the
pipeline and shows exactly which stage covers it. It supersedes the per-stage plans for the
purpose of verifying completeness. The per-stage implementation plans (Stage 2 modification,
Stage 3, Stage 5) remain the detailed build specs; this document is the coverage contract
that guarantees no JD intent is dropped.

---

## PART A — JD LINE-BY-LINE COVERAGE MATRIX

Every substantive requirement, preference, disqualifier, and trap from the JD, mapped to the
stage that handles it and the exact mechanism. If a row says "GAP" it is filled later in this
document under PART C.

### A.1 — Header / framing requirements

| JD line | Requirement | Stage | Mechanism |
|---|---|---|---|
| L5, L95 | Location: Pune/Noida preferred, Tier-1 India OK, outside-India case-by-case no visa | Stage 2 (compute) → Stage 5 (apply) | `location_tier` feature → Layer 7 logistics |
| L9, L41-43 | 5–9 years, flexible ("range not requirement") | Stage 2 | Check A: hard remove outside [4,10]; `exp_band`, `in_sweet_spot` flags |
| L19-25 | Shipper-over-researcher dual mode | Stage 3 + Stage 5 | Q2 career-shape query + Layer 3 career-shape multiplier |

### A.2 — "What you'd actually be doing" (the role mandate)

| JD line | Requirement | Stage | Mechanism |
|---|---|---|---|
| L29 | Owns ranking/retrieval/matching intelligence layer | Stage 3 | Q1 targets exactly retrieval+ranking+matching |
| L33-37 | Must know BM25, embeddings, hybrid retrieval, LLM re-ranking, eval infra, A/B | Stage 3 + Stage 5 | Q1 + Q4 (jargon) + Layer 2 must-have floor |
| L39 | Mentoring / leadership (growing 4→12) | Not scored | Not extractable from candidate profiles; correctly omitted |

### A.3 — The three hard disqualifiers ("we will / probably will not move forward")

| JD line | Disqualifier | Stage | Mechanism |
|---|---|---|---|
| L47 | Pure research, no production deployment | Stage 2 | **Check F** — research-only hard remove (with production escape hatch) |
| L49 | LangChain-only <12mo AI, no pre-LLM ML | Stage 2 | **Check H** — shallow-AI hard remove (with pre-LLM escape hatch) |
| L51 | No production code in 18 months (architect-only) | Stage 2 | **Check G** — `stale_coding` soft flag → Stage 5 Layer 3 penalty |

### A.4 — "Things you absolutely need" (must-haves)

| JD line | Must-have | Stage | Mechanism |
|---|---|---|---|
| L59 | Production embeddings retrieval + drift/refresh/regression ops | Stage 3 + Stage 5 | Q1 retrieval subspace + Layer 2 floor + assessment scores |
| L61 | Production vector DB / hybrid search ops | Stage 3 + Stage 5 | Q1 infra subspace + Q4 jargon + Layer 2 floor |
| L63 | Strong Python + code quality | Stage 5 | Layer 2 must-have category (python) |
| L65 | Evaluation frameworks (NDCG/MRR/MAP/A-B/offline-online) | Stage 3 + Stage 5 | Q1 eval subspace + Q4 jargon + Layer 2 floor |

### A.5 — "Things we'd like but won't reject for" (optional / nice-to-have)

| JD line | Optional | Stage | Mechanism |
|---|---|---|---|
| L69 | LLM fine-tuning (LoRA/QLoRA/PEFT) | Stage 5 | Layer 5 optional bonus (absence neutral) |
| L71 | Learning-to-rank (XGBoost/neural) | Stage 5 | Layer 5 optional bonus |
| L73 | HR-tech / recruiting / marketplace | Stage 5 | Layer 5 optional bonus |
| L75 | Distributed systems / inference optimization | Stage 5 | Layer 5 optional bonus |
| L77 | Open-source AI/ML contributions | Stage 5 | Layer 5 optional bonus + Layer 4 (validation corroboration) |

### A.6 — "Things we explicitly do NOT want" (penalties)

| JD line | Anti-pattern | Stage | Mechanism |
|---|---|---|---|
| L83 | Title-chasers (1.5y hops, want 3+ yr stay) | Stage 2 (compute) → Stage 5 (apply) | `avg_tenure_per_employer`, `short_hop_count`, `title_progression_jumps` → Layer 4 penalty |
| L85 | Framework enthusiasts (tutorial GitHub) | Stage 3 + Stage 5 | Q3 anti-pattern + Layer 4 q3_residual; corroborated by low `github_activity_score` + high `skill_kw_density` |
| L87 | Consulting-only career (named firms) | Stage 2 | **Check E** — consulting-only hard remove (with prior-product escape hatch) |
| L89 | CV/speech/robotics without NLP/IR | Stage 3 + Stage 5 | Q3 anti-pattern (semantic) + Layer 4 q3_residual |
| L91 | 5+ yr closed-source, no external validation | Stage 2 (compute) → Stage 5 (apply) | `external_validation_score`, `has_github` → Layer 4 penalty |

### A.7 — Logistics

| JD line | Item | Stage | Mechanism |
|---|---|---|---|
| L95 | Location tiers + no visa sponsorship | Stage 2 → Stage 5 | `location_tier` → Layer 7 |
| L95 | Hybrid, Tue/Thu offices, quarterly offsite | Stage 5 | `preferred_work_mode` (signal 14) → Layer 7 |
| L97 | Notice: sub-30 ideal, 30+ higher bar | Stage 2 → Stage 5 | `notice_period_days` (signal 12) → Layer 7 smooth decay |

### A.8 — "The vibe check" (culture)

| JD line | Item | Stage | Mechanism |
|---|---|---|---|
| L103 | Async-first, writes a lot | Not scored | No reliable profile proxy; documented as known non-coverage |
| L105 | Disagrees openly, decides quickly | Not scored | No profile proxy |
| L107 | Move fast / unstable codebase tolerance | Partially | Shipper-tilt captured via Q2; rest not extractable |

### A.9 — "How to read between the lines" (the ideal candidate)

| JD line | Item | Stage | Mechanism |
|---|---|---|---|
| L113 | 6-8 yrs total, 4-5 in applied ML at product cos | Stage 2 + Stage 5 | `in_sweet_spot` + `product_company_fraction` → Layer 3 |
| L115 | Shipped ≥1 end-to-end ranking/search/rec system at scale | Stage 3 | Q2 career-shape query |
| L117 | Strong defensible opinions on retrieval/eval/LLM | Stage 3 + Stage 4 | Q1/Q2 depth + cross-encoder holistic match |
| L119 | Located in / willing to relocate Noida/Pune | Stage 2 → Stage 5 | `location_tier` + `willing_to_relocate` (signal 15) |
| L121 | Active on platform / in job market | Stage 5 | Layer 6 availability multiplier |
| L123 | Prefer 10 great over 1000 maybes (precision) | Whole funnel | Precision-first design; must-have floor + adaptive cuts |

### A.10 — The hackathon traps (final note)

| JD line | Trap | Stage | Mechanism |
|---|---|---|---|
| L129, L131 | Keyword stuffer (Marketing Manager + AI skills) | Stage 2 | Check B: non-eng title + high kw density → hard remove |
| L131 | Tier-5 plain-language expert (no jargon, real systems) | Stage 3 | Q2 career-shape dense retrieval (semantic, not keyword) |
| L133 | Behavioral availability down-weight | Stage 5 | Layer 6 availability multiplier (6 signals) |
| (spec) | Honeypots (impossible profiles) | Stage 2 | Honeypot rules R1–R5, H3–H5 |

**Coverage conclusion:** every scoreable JD line is assigned to a stage. The only non-covered
items (L39 mentoring, L103-107 culture vibe) are genuinely not extractable from candidate
profile data and are documented as deliberate non-coverage, not gaps.

---

## PART B — THE COMPLETE PIPELINE (all stages, final state)

```
Stage 0   Offline precompute: 3-block Instructor vectors, FAISS IP index, BM25,
          candidate_features.parquet (23 signals + assessment scores + summaries)
   │
Stage 1   Clustering: 100K → ~6K
   │
Stage 2   HARD GATE (MODIFIED): ~6K → ~3-4K
          Existing: honeypot, experience band, keyword-stuffer, availability flags
          NEW Check E: consulting-only hard remove
          NEW Check F: research-only hard remove
          NEW Check G: stale-coding soft flag
          NEW Check H: shallow-AI hard remove
          NEW features: product_company_fraction, career_type, avg_tenure_per_employer,
                        short_hop_count, title_progression_jumps, location_tier,
                        external_validation_score, has_github, notice_period_days,
                        pre_llm_production_ml, recent_ai_only, research_fraction,
                        has_any_production_role, stale_coding
   │
Stage 3   HYBRID RETRIEVAL: ~3-4K → ~300-600
          Q1 dense (must-haves) + Q2 dense (career shape/Tier-5 rescue)
          + Q3 dense (anti-pattern penalty) + Q4 BM25 (jargon)
          → RRF fusion − α·Q3 → adaptive top-k
   │
Stage 4   CROSS-ENCODER: ~300-600 → ~300
          Joint (JD, candidate) MS-MARCO MiniLM scoring → cross_encoder_score
   │
Stage 5   COMPOSITE SCORER (FORMULA, not ML): ~300 → top 100
          7-layer transparent formula → final_score → reasoning → CSV
```

---

## PART C — STAGE 5 FINAL SPEC (with ALL 23 behavioral signals resolved)

This section is the authoritative Stage 5 spec. It supersedes the prior Stage 5 plan's
behavioral-signal handling by explicitly resolving all 23 signals.

### C.0 — Design decision (binding)

Stage 5 is a **deterministic weighted composite scorer**, NOT a trained model. No `lightgbm`,
no `torch`, no labels (ground truth is hidden), no online LLM. Reasons: no training labels
exist; the defend-your-work interview requires full explainability; the reasoning column needs
a decomposable score. Tech stack: `polars`, `numpy`, `pyyaml`, `argparse` only.

### C.1 — THE 23 BEHAVIORAL SIGNALS — COMPLETE RESOLUTION

Every signal is explicitly assigned a role or an exclusion with reason. **This table is the
contract — the coding agent implements exactly this, and excludes exactly the excluded ones.**

| # | Signal | Decision | Where / Why |
|---|---|---|---|
| 1 | profile_completeness_score | **EXCLUDE** | Hygiene, not fit. Rewards profile polish over substance. |
| 2 | signup_date | **EXCLUDE** | No fit signal. (Tenure on platform ≠ candidate quality.) |
| 3 | last_active_date | **USE** | Layer 6 — recency factor (the JD's "hasn't logged in 6 months") |
| 4 | open_to_work_flag | **USE** | Layer 6 — open_factor (also Stage 2 `not_open`) |
| 5 | profile_views_received_30d | **EXCLUDE (default OFF)** | Popularity, not fit; would help keyword stuffers. Optional micro-tiebreaker, capped, OFF by default. |
| 6 | applications_submitted_30d | **USE (weak)** | Layer 6 — job-market-active signal (the JD's "clear signal of being in job market") |
| 7 | recruiter_response_rate | **USE** | Layer 6 — resp_factor (the JD's "5% response rate") |
| 8 | avg_response_time_hours | **USE** | Layer 6 — response-speed factor |
| 9 | skill_assessment_scores | **USE (critical)** | Layer 2 — must-have floor corroboration. Un-fakeable competence evidence; the antidote to keyword stuffing. |
| 10 | connection_count | **EXCLUDE** | Social-graph noise, easily gamed. |
| 11 | endorsements_received | **EXCLUDE** | Easily gamed, social not technical. |
| 12 | notice_period_days | **USE** | Layer 7 — notice decay penalty |
| 13 | expected_salary_range | **EXCLUDE** | JD gives no band; judging it would be guessing. |
| 14 | preferred_work_mode | **USE** | Layer 7 — work-mode fit (role is Pune/Noida hybrid) |
| 15 | willing_to_relocate | **USE** | Stage 2 F2 — upgrades location_tier → Layer 7 |
| 16 | github_activity_score | **USE** | Stage 2 F3 — external_validation_score → Layer 4 |
| 17 | search_appearance_30d | **EXCLUDE (default OFF)** | Popularity proxy; helps stuffers. Optional micro-tiebreaker, OFF by default. |
| 18 | saved_by_recruiters_30d | **EXCLUDE (default OFF)** | Same as 17. |
| 19 | interview_completion_rate | **USE** | Layer 6 — interview_factor (do they show up) |
| 20 | offer_acceptance_rate | **USE** | Layer 6 — hireability factor (−1 = neutral) |
| 21 | verified_email | **EXCLUDE** | Hygiene, not fit. |
| 22 | verified_phone | **EXCLUDE** | Hygiene, not fit. |
| 23 | linkedin_connected | **EXCLUDE** | Hygiene, not fit. |

**Summary: 11 used, 12 excluded (3 of which are optional-OFF micro-tiebreakers).**

**Critical rule for ALL signals:** a value of `-1` or `null` means "not applicable" (per the
signals doc), NOT "bad." Treat as NEUTRAL (factor 1.0 / no penalty). This applies especially
to signals 9, 19, 20, 16. Misreading −1 as a low score is a correctness bug that punishes
candidates with no history.

**Why the exclusions matter as much as the inclusions:** signals 1, 5, 10, 11, 17, 18, 21, 22,
23 all measure popularity, social graph, or profile hygiene. The dataset's keyword-stuffer and
behavioral-rescue traps are *built to be rewarded* by exactly these signals. Including them
would make the ranking more gameable and would actively help trap candidates. The exclusion
list is a deliberate defense, documented so the coding agent does not "helpfully" add them.

### C.2 — THE SEVEN-LAYER FORMULA (final, with all signals wired)

```
core             = w_cross·CEnorm + w_fused·Fnorm + w_q1·Q1norm + w_q2·Q2norm    (L1)
core_floored     = core · must_have_floor_multiplier                              (L2)
shaped           = core_floored · career_shape_multiplier                         (L3)
penalized        = shaped − Σ penalties                                           (L4)
bonused          = penalized + Σ optional_bonuses                                 (L5)
availability_adj = bonused · availability_multiplier                              (L6)
final_score      = availability_adj + logistics_adjustment                        (L7)
```

#### LAYER 1 — Core relevance
Min-max normalize over the ~300: `cross_encoder_score`, `fused_score`, `q1_score`, `q2_score`
(and `q3_neg_sim` for L4). Blend:
```
core = 0.45·CEnorm + 0.25·Fnorm + 0.20·Q1norm + 0.10·Q2norm
```
Weights from `stage5.core_weights`, must sum to 1.0 (assert at load).

#### LAYER 2 — Must-have floor (THREE corroboration proxies now)
This is the most important anti-keyword-stuffer mechanic. Combine three independent proxies:
```
keyword_ratio   = must_have_categories_covered / 4   # skills keyword check (4 categories)
semantic_cov    = Q1norm                             # Q1 targets the must-haves
assessment_cov  = normalized mean of skill_assessment_scores for must-have-relevant skills
                  # signal 9; if no assessments present → treat as neutral (do NOT penalize):
                  # set assessment_cov = semantic_cov so it neither helps nor hurts

combined_coverage = max(keyword_ratio, semantic_cov, assessment_cov)
must_have_floor_multiplier = floor_min + (1 − floor_min)·combined_coverage   # floor_min=0.4
core_floored = core · must_have_floor_multiplier
```
The four must-have categories for the keyword check: retrieval, vector_db, eval, python
(keyword lists in config). Using `max()` of three proxies rescues Tier-5 candidates (strong on
one proxy, weak on others) while a candidate weak on ALL three is genuinely thin and gets
dampened to 0.4×.

**Assessment-score detail (signal 9):** `skill_assessment_scores` is a dict {skill: 0-100}.
Compute the mean score over skills that map to any must-have category; normalize to [0,1] by
dividing by 100. If the candidate has zero relevant assessments, set `assessment_cov =
semantic_cov` (neutral — never a penalty for not having taken tests).

#### LAYER 3 — Career-shape multiplier
```
shape_mult = 1.0
shape_mult *= product_floor + (1−product_floor)·product_company_fraction   # product_floor=0.5
if in_sweet_spot:           shape_mult *= 1.08
elif exp_band=="near_band": shape_mult *= 0.95
if stale_coding:            shape_mult *= 0.85     # Check G (JD L51)
if not has_any_production_role: shape_mult *= 0.80 # Check F escape signal
shaped = core_floored · shape_mult
```

#### LAYER 4 — Penalties (subtractive)
```
title_chasing   = min(0.15, short_hop_count·0.03)                    # JD L83
q3_residual     = 0.10·Q3norm                                        # JD L85,L89 (framework/CV)
closed_source   = 0.05 if (external_validation_score<0.1 and not has_github) else 0  # JD L91
ambiguity       = 0.02·title_ambiguous + 0.02·(exp_band=="near_band")
consulting_resid = 0.04 if career_type=="consulting_heavy" else 0    # JD L87 residual for kept-but-mixed
total_penalty   = title_chasing + q3_residual + closed_source + ambiguity + consulting_resid
penalized       = shaped − total_penalty
```
(Note: consulting_resid catches mixed-career candidates who survived Check E because they had
*some* product experience but are still consulting-heavy — a soft down-weight, not a removal.)

#### LAYER 5 — Optional bonus (additive, absence ALWAYS neutral)
```
five categories: fine_tuning(LoRA/QLoRA/PEFT), learning_to_rank, hr_tech, distributed_systems, oss
optional_bonus = min(0.08, count_present·0.02)     # JD L69-77
bonused = penalized + optional_bonus
```
OSS category requires `has_github==True` AND `external_validation_score` above a small floor
to count (avoids rewarding a claimed-but-unverified OSS skill).

#### LAYER 6 — Availability multiplier (ALL 6 Group-A signals, bounded, never rescues)
```
resp_factor      = clamp(recruiter_response_rate / 0.5, 0.6, 1.0)              # signal 7
speed_factor     = clamp(1 − max(0, avg_response_time_hours−24)/168, 0.7, 1.0) # signal 8 (>1wk decays)
recency_factor   = 1.0 if days_inactive<=30 else clamp(1−(days_inactive−30)/180, 0.6, 1.0)  # signal 3
open_factor      = 1.0 if open_to_work else 0.85                               # signal 4
interview_factor = clamp(interview_completion_rate, 0.7, 1.0) if present else 1.0  # signal 19 (−1=neutral)
offer_factor     = clamp(offer_acceptance_rate, 0.8, 1.0) if >=0 else 1.0      # signal 20 (−1=neutral)
market_factor    = 1.0 if applications_submitted_30d>0 or open_to_work else 0.95  # signal 6

availability_multiplier = resp·speed·recency·open·interview·offer·market
availability_multiplier = clamp(availability_multiplier, 0.5, 1.0)  # avail_min=0.5
availability_adj = bonused · availability_multiplier
```
Multiplicative + floored at 0.5 = can dampen a strong candidate but can NEVER lift a weak one
above a strong one (anti behavioral-rescue, JD L133 + plan.md rule). Every `−1`/null → 1.0.

#### LAYER 7 — Logistics (small additive, final tiebreak nudge)
```
location_adj = {preferred:+0.03, acceptable:+0.01, unknown:0, outside_india:−0.10}[location_tier]
workmode_adj = {hybrid:+0.01, flexible:+0.01, onsite:+0.01, remote:−0.02}[preferred_work_mode]  # signal 14
notice_adj   = 0 if notice_period_days<=30 or null
               else −min(0.08, (notice_period_days−30)/90·0.05)             # signal 12
logistics_adjustment = location_adj + workmode_adj + notice_adj
final_score  = availability_adj + logistics_adjustment
```

### C.3 — Procedure, reasoning, output, validation
Identical to the standalone Stage 5 plan (load+join, normalize, apply L1–L7, sort by
final_score desc with candidate_id tiebreak, top 100, generate reasoning, write CSV, validate).
Reasoning is composed from the score decomposition + `technical_summary_sentence` (no online
LLM); honesty clause picks each candidate's largest dampener/penalty so rank-consistency and
variation emerge naturally. Output: `team_xxx.csv`, exactly 101 lines,
`candidate_id,rank,score,reasoning`, UTF-8, score monotonic non-increasing.

---

## PART D — FINAL CONFIG ADDITIONS (`stage5:`)

```yaml
stage5:
  team_id: "team_xxx"
  top_n: 100
  current_date: "2026-06-22"

  core_weights: {w_cross: 0.45, w_fused: 0.25, w_q1: 0.20, w_q2: 0.10}
  must_have_floor_min: 0.4

  must_have_keywords:
    retrieval: [embeddings, semantic search, retrieval, rag, sentence-transformers, bge, e5, dense retrieval]
    vector_db: [faiss, pinecone, qdrant, weaviate, milvus, opensearch, elasticsearch, vector database, hnsw]
    eval: [ndcg, mrr, map, a/b test, evaluation, ranking metrics, offline-to-online]
    python: [python]

  career_shape:
    product_floor: 0.5
    sweet_spot_bonus: 1.08
    near_band_factor: 0.95
    stale_coding_factor: 0.85
    no_production_factor: 0.80

  penalties:
    per_hop_penalty: 0.03
    short_hop_penalty_cap: 0.15
    q3_penalty_weight: 0.10
    validation_floor: 0.1
    closed_source_penalty_value: 0.05
    title_ambiguous_penalty: 0.02
    near_band_penalty: 0.02
    consulting_heavy_penalty: 0.04

  optional_bonus:
    per_category_bonus: 0.02
    optional_bonus_cap: 0.08
    categories:
      fine_tuning: [lora, qlora, peft, fine-tuning, instruction tuning]
      learning_to_rank: [learning-to-rank, lambdamart, ranknet, xgboost ranking, lightgbm ranking]
      hr_tech: [ats, applicant tracking, recruiting, marketplace, talent]
      distributed_systems: [distributed systems, spark, ray, model serving, triton, quantization, inference optimization]
      oss: [open source, opensource, maintainer, contributor]

  availability:
    good_response_rate: 0.5
    response_floor: 0.6
    slow_response_hours: 24
    response_decay_window_hours: 168
    speed_floor: 0.7
    fresh_days: 30
    recency_decay_window: 180
    recency_floor: 0.6
    not_open_factor: 0.85
    interview_floor: 0.7
    offer_floor: 0.8
    market_inactive_factor: 0.95
    avail_min: 0.5

  logistics:
    loc_preferred_bonus: 0.03
    loc_acceptable_bonus: 0.01
    loc_outside_penalty: 0.10
    workmode_fit_bonus: 0.01
    workmode_remote_penalty: 0.02
    notice_penalty_scale: 0.05
    notice_penalty_cap: 0.08

  # optional popularity micro-tiebreakers — DEFAULT OFF (would help keyword stuffers)
  enable_popularity_tiebreak: false
  popularity_tiebreak_cap: 0.01
```

---

## PART E — FINAL COVERAGE VERIFICATION

Every JD disqualifier, must-have, optional, penalty, logistic, and trap is now assigned:

- **3 hard disqualifiers** (research-only, shallow-AI, consulting-only) → Stage 2 hard removes ✔
- **1 soft disqualifier** (stale coding) → Stage 2 flag → Stage 5 L3 penalty ✔
- **4 must-haves** → Stage 3 Q1/Q4 + Stage 5 L2 floor (3 proxies incl. assessment scores) ✔
- **5 optionals** → Stage 5 L5 bonus (absence neutral) ✔
- **5 explicit penalties** → Stage 2 (title-chasing/validation features) + Stage 3 Q3 + Stage 5 L4 ✔
- **3 logistics** (location, work-mode, notice) → Stage 2 features → Stage 5 L7 ✔
- **Ideal-candidate profile** (6-8y, product, shipped systems, opinions, located, active) → L3 + Q2 + L6 ✔
- **3 traps** (keyword stuffer, Tier-5 rescue, behavioral) → Stage 2 Check B + Stage 3 Q2 + Stage 5 L6 ✔
- **Honeypots** → Stage 2 rules ✔
- **All 23 behavioral signals** → 11 used, 12 excluded with documented reasons ✔

**Documented deliberate non-coverage** (not gaps — genuinely not in profile data):
mentoring/leadership (L39), async-writing culture (L103), disagreement style (L105), codebase-
instability tolerance (L107). The shipper-tilt aspect of culture IS partially captured via Q2.

No remaining gaps.
