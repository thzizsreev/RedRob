# REDROB HACKATHON — STAGED RANKING PIPELINE PLAN

**Scope of this document:** the online ranking funnel that turns the post-clustering
candidate set into the final Top-100 CSV. This plan covers **Stage 2 → Stage 5**.
Stage 1 (clustering, 100K → 6K) is assumed complete and is treated here only as an
upstream input contract.

**Status of detail:**
- **Stage 2 (Hard Tabular Gate)** — fully specified, intricate.
- **Stage 3 (Multi-Query Hybrid Retrieval)** — fully specified, intricate.
- **Stage 4 (Cross-Encoder Reranking)** — surface-level placeholder. To be deepened later.
- **Stage 5 (LightGBM Reranking)** — file-stub placeholder only. Model design is a separate research task.

---

## 0. EXECUTION MODEL — READ THIS FIRST

> **Every stage is implemented as its own standalone script and is run MANUALLY,
> one at a time, by the operator.**
>
> There is **no orchestrator**, no single `rank.py` that chains stages in-process.
> Each stage:
> 1. reads a well-defined input artifact from disk (the previous stage's output),
> 2. does its work,
> 3. writes a well-defined output artifact to disk,
> 4. exits.
>
> The operator inspects the output of stage *N* before running stage *N+1*.
> This is deliberate: it lets us validate, debug, and tune each stage in isolation,
> and it means a bad stage can be re-run without recomputing the whole funnel.

**Important reconciliation note for Stage 3 reproduction:** the hackathon spec requires
that the *ranking step* run in ≤5 min on CPU as a single reproducible command. The
manual multi-script model above is for **our development and tuning workflow**. Before
final submission we will provide a thin top-level wrapper (`rank.py`) that calls the
four stage modules in sequence so the official single-command reproduction requirement
is satisfied. The stage scripts must therefore each expose a callable `run(input_path,
output_path, config)` entry point in addition to their CLI, so the wrapper can import
and chain them. **Do not couple stages by shared in-memory state — only by disk artifacts.**

### Artifact flow overview

```
[Stage 1 output]                    stage1_clustered.parquet      (~6,000 rows)
        │
        ▼  python stage2_gate.py
[Stage 2: Hard Tabular Gate]        stage2_gated.parquet          (~3,000–4,000 rows)
        │
        ▼  python stage3_retrieve.py
[Stage 3: Hybrid Retrieval]         stage3_retrieved.parquet      (~300–600 rows)
        │
        ▼  python stage4_crossenc.py
[Stage 4: Cross-Encoder Rerank]     stage4_reranked.parquet       (~300 rows)
        │
        ▼  python stage5_lgbm.py
[Stage 5: LightGBM Rerank]          team_xxx.csv                  (exactly 100 rows)
```

### Directory layout

```
project/
├── ranking_pipeline_plan.md          # this file
├── artifacts/
│   ├── precomputed/                   # built offline (precompute.py), read-only at run time
│   │   ├── dense_vectors.npy          # (N, 768) float32, orthogonal subspace layout
│   │   ├── faiss_index.bin            # exact flat IP index over dense_vectors
│   │   ├── bm25_index.pkl             # serialized BM25 over clean jargon text blocks
│   │   ├── candidate_features.parquet # Track B tabular payload (23 signals + hard constraints + summaries)
│   │   └── id_row_map.parquet         # candidate_id <-> dense_vectors row index mapping
│   └── runtime/                       # stage outputs land here
│       ├── stage1_clustered.parquet
│       ├── stage2_gated.parquet
│       ├── stage3_retrieved.parquet
│       ├── stage4_reranked.parquet
│       └── team_xxx.csv
├── stages/
│   ├── stage2_gate.py
│   ├── stage3_retrieve.py
│   ├── stage4_crossenc.py            # placeholder, surface-level
│   └── stage5_lgbm.py                # placeholder stub only
├── rank.py                            # thin wrapper for final single-command reproduction
└── config.yaml                        # all tunable thresholds live here, not hardcoded
```

**Rule:** no stage hardcodes a threshold. Every magic number lives in `config.yaml`
under a per-stage namespace (`stage2:`, `stage3:`...). Each stage reads only its own block.

---

## STAGE 1 (UPSTREAM — INPUT CONTRACT ONLY)

Stage 1 is clustering and is already done. This plan does not implement it. It only
defines what Stage 2 is allowed to assume about its input.

**`stage1_clustered.parquet` — required columns:**

| Column | Type | Notes |
|---|---|---|
| `candidate_id` | string | `CAND_XXXXXXX`. Unique. Must exist in `candidates.jsonl`. |
| `cluster_id` | int | Which cluster the candidate was assigned to. |
| `dist_to_centroid` | float | Distance to assigned cluster centroid. **Carry this forward — it is a free Stage 5 feature.** |
| `cluster_rank` | int (nullable) | If clusters were themselves ranked vs the JD, the cluster's rank. Nullable if not computed. |

Row count expected: ~6,000. If the row count deviates by more than ±20% from 6,000,
Stage 2 should emit a loud warning (not crash) — it usually means an upstream change.

---

## STAGE 2 — HARD TABULAR GATE

**Script:** `stages/stage2_gate.py`
**Input:** `stage1_clustered.parquet` (+ `candidate_features.parquet`, `candidates.jsonl` for raw fields)
**Output:** `stage2_gated.parquet`
**Expected reduction:** ~6,000 → ~3,000–4,000
**Vectors used:** NONE. This is a pure tabular / rule stage. It runs before any embedding work.

### 2.0 Philosophy

This stage exists to **remove candidates who are structurally disqualified or
structurally suspicious before we spend retrieval compute on them.** It is cheap
(milliseconds over a few thousand rows) and it protects every downstream stage from
two of the dataset's deliberate traps: **keyword stuffers** and **honeypots**.

Two kinds of decisions are made here:
- **HARD REMOVE** — candidate is dropped from the funnel entirely.
- **SOFT FLAG** — candidate is kept, but a boolean/categorical flag is written to the
  output so Stage 5 can use it as a feature or penalty. We *flag rather than drop* when
  the JD itself signals nuance (e.g. it says experience is "a range, not a requirement").

> **Design principle:** when in doubt, FLAG, don't REMOVE. A wrongly-removed Tier-5
> candidate is invisible forever; a wrongly-flagged one can still be rescued by a strong
> model score downstream. Removal is reserved for things the JD treats as absolute or for
> logical impossibilities (honeypots).

### 2.1 Input fields this stage needs

Pulled from `candidate_features.parquet` and/or raw `candidates.jsonl`:

- `total_years_exp` (int)
- per-employer work history: list of `{company, title, start_date, end_date, is_current}`
- `company_founded_year` per employer (if available in raw data; else derive/skip the related honeypot rule)
- current/most-recent `title` (string)
- `skills`: list of `{name, proficiency, years_used}`
- `redrob_signals`: the 23-signal object (only `open_to_work_flag`, `last_active_date`,
  `recruiter_response_rate` are read here; the rest are deferred to Stage 5)
- `is_honeypot` field **if present in released data** — note: do NOT rely on this existing;
  the spec says honeypots must be detectable by inspection. Use it only as a cross-check
  in local validation, never as the production gate.

### 2.2 Check A — Experience Band Gate (configurable hard/soft)

The bible says filter `total_years_exp BETWEEN 5 AND 9`. The JD says 5–9 is "a range, not
a requirement" and it will "seriously consider candidates outside the band if other signals
are strong." These two documents are in tension. We resolve it with a **band + tolerance**:

```
hard_min      = 5
hard_max      = 9
soft_tolerance = 1     # candidates in [4,5) or (9,10] are KEPT but flagged
sweet_low     = 6
sweet_high    = 8
```

Logic:
- `total_years_exp` in `[hard_min, hard_max]` → KEEP, `exp_band = "in_band"`.
- `total_years_exp` in `[hard_min - soft_tolerance, hard_min)` or `(hard_max, hard_max + soft_tolerance]`
  → KEEP, `exp_band = "near_band"` (SOFT FLAG).
- Otherwise → **HARD REMOVE**, reason `exp_out_of_band`.
- Additionally compute `in_sweet_spot = sweet_low <= total_years_exp <= sweet_high`
  and write it as a column. This is **not** a gate — it is a positive feature for Stage 5,
  since the JD's "ideal candidate" is explicitly 6–8 years.

> Rationale for the tolerance band: a 4.5-year candidate who is otherwise a perfect Tier-5
> product-company ranker is exactly the kind of person the JD says it will bend the band for.
> Dropping them at a hard 5.0 cliff contradicts the JD's stated intent.

### 2.3 Check B — Title / Role Coherence (kills keyword stuffers)

This is the primary defense against the **keyword-stuffer trap** ("every AI buzzword but
title is 'Marketing Manager'"). We classify the candidate's most-recent title into a family
and combine that with skill-keyword density.

**Title families (maintained as a lookup in `config.yaml`, matched case-insensitively on
normalized title tokens):**

| Family | Example title tokens | Treatment |
|---|---|---|
| `core_eng` | engineer, ml engineer, ai engineer, software engineer, sde, backend, applied scientist, research engineer, search engineer, mlops, data engineer, platform engineer | ALLOW |
| `adjacent_eng` | data scientist, analytics engineer, research scientist, nlp scientist | ALLOW |
| `ambiguous` | analyst, consultant, lead, architect, specialist, developer | KEEP + SOFT FLAG `title_ambiguous` |
| `non_eng` | marketing, sales, hr, recruiter, accountant, operations manager, product manager (non-technical), designer, customer success | candidate for removal — see combined logic below |

**Combined keyword-stuffer logic:**

```
skill_kw_density = (# of JD-relevant ML/IR keywords present in candidate's skills)
                   / (max(1, total skills listed))
```

JD-relevant keyword set (from the JD's own vocabulary): embeddings, retrieval, ranking,
vector database, FAISS, Pinecone, Qdrant, Weaviate, Milvus, OpenSearch, Elasticsearch,
NDCG, MRR, MAP, sentence-transformers, BGE, E5, LoRA, QLoRA, learning-to-rank, RAG,
cross-encoder, A/B test, embedding drift, hybrid search.

Decision matrix:

| Title family | High kw density (≥ `stuffer_density`, default 0.5) | Low kw density |
|---|---|---|
| `core_eng` / `adjacent_eng` | KEEP | KEEP |
| `ambiguous` | KEEP + flag `title_ambiguous` | KEEP + flag `title_ambiguous` |
| `non_eng` | **HARD REMOVE** `keyword_stuffer` | **HARD REMOVE** `non_eng_title` |

> The key insight: a `non_eng` title with **high** AI-keyword density is the textbook
> keyword stuffer — they've loaded their skills section with buzzwords that contradict
> their actual role. A `non_eng` title with low density is just an unrelated candidate.
> Both are removed, but we record distinct reason codes so we can audit the trap-catch rate.

> **Caveat to validate against real data:** title parsing is messy. Before trusting this
> check, the operator should dump the distribution of inferred title families on the real
> 6K and eyeball the `non_eng` bucket for false positives (e.g. "Engineering Manager" must
> NOT be misfiled as non_eng — it should be `ambiguous` because of the 18-month-no-code
> disqualifier handled separately at Stage 5, not here).

### 2.4 Check C — Honeypot Consistency Rules (HARD REMOVE)

Honeypots are "subtly impossible profiles." They must be caught by **logical consistency**,
not relevance. Spec penalty is severe: >10% honeypot rate in Top 100 = disqualification.
We catch them as early as possible. Each rule below is independent; **tripping ANY single
rule is a HARD REMOVE** with reason `honeypot_<rule>`.

**Rule H1 — Tenure exceeds company age.**
For each employer with a known `company_founded_year`:
`years_at_company > (candidate_reference_year - company_founded_year)` → impossible.
(This is the spec's own example: "8 years of experience at a company founded 3 years ago.")

**Rule H2 — Summed tenure exceeds career span.**
`sum(role_durations) > (last_role_end - first_role_start) + overlap_tolerance`.
A small `overlap_tolerance` (default 0.5y) allows for legitimately concurrent roles;
beyond that the timeline is fabricated.

**Rule H3 — Expert proficiency with zero usage.**
Any skill with `proficiency == "expert"` AND `years_used == 0` → impossible.
(Spec example: "'expert' proficiency in 10 skills with 0 years used.") Generalize to a count:
trip if `count(expert skills with years_used == 0) >= expert_zero_threshold` (default 1, but
expose as config; a single typo'd skill maybe shouldn't nuke a real candidate — tune on data).

**Rule H4 — Total experience vs claimed skill-years impossibility.**
`max(skill.years_used) > total_years_exp + small_slack` → claiming more years in a single
skill than they've worked at all.

**Rule H5 — Future / inverted dates.**
Any `start_date > end_date`, or any date in the future relative to dataset reference date.

**Rule H6 — Statistical outlier backstop (optional, OFF by default in Stage 2).**
An Isolation Forest on tabular features can catch honeypots that slip past H1–H5. We
deliberately **do not** run it here by default — it risks removing real outlier-but-valid
candidates, and removal is irreversible. Instead, if enabled, it runs as a SOFT FLAG
(`honeypot_anomaly_score` float column) consumed at Stage 5, not as a Stage 2 removal.

> **Honeypot logging requirement:** Stage 2 must write a separate `stage2_honeypot_log.csv`
> listing every removed candidate_id, the rule(s) tripped, and the offending values. We need
> this to (a) tune thresholds, and (b) demonstrate the honeypot-catch methodology at Stage 4
> manual review / Stage 5 defend-your-work.

### 2.5 Check D — Availability Soft-Flag (NEVER a hard gate here)

The JD cares whether a candidate is "actually available," but availability must **never
override engineering fit** (that's the "behavioral rescue" trap, and its inverse). So Stage 2
only *flags*; the actual weighting happens at Stage 5.

Compute and write (do not filter on):
- `stale_profile = last_active_date older than stale_days` (default 180). Boolean flag only.
- `low_responder = recruiter_response_rate < min_response_rate` (default 0.10). Boolean flag only.
- `not_open = open_to_work_flag == False`. Boolean flag only.

> Why not gate here: a brilliant Tier-5 candidate who's been quiet for 5 months is still
> worth surfacing — the recruiter can reach out. We down-weight, we don't delete. The only
> place availability could justify a hard gate is an explicit business rule we don't have.

### 2.6 Stage 2 output contract

**`stage2_gated.parquet` columns:**

| Column | Type | Source |
|---|---|---|
| `candidate_id` | string | passthrough |
| `cluster_id`, `dist_to_centroid`, `cluster_rank` | passthrough | Stage 1 |
| `total_years_exp` | int | input |
| `exp_band` | enum(`in_band`,`near_band`) | Check A |
| `in_sweet_spot` | bool | Check A |
| `title_family` | enum | Check B |
| `skill_kw_density` | float | Check B |
| `title_ambiguous` | bool | Check B |
| `stale_profile` | bool | Check D |
| `low_responder` | bool | Check D |
| `not_open` | bool | Check D |
| `honeypot_anomaly_score` | float (nullable) | Check H6 if enabled |

Side outputs:
- `stage2_honeypot_log.csv` (removed honeypots + reasons)
- `stage2_removed_log.csv` (every removal: candidate_id, reason_code) — full audit trail

**Validation the operator should run after Stage 2:**
1. Row count in expected range (~3,000–4,000). If <2,000 or >5,000, a threshold is wrong.
2. Eyeball 20 random `non_eng` removals — are any obvious false positives?
3. Confirm honeypot log count is plausible (dataset has ~80 honeypots total across 100K;
   in a 6K sample expect a small handful, not hundreds — hundreds means H-rules are too aggressive).

---

## STAGE 3 — MULTI-QUERY HYBRID RETRIEVAL

**Script:** `stages/stage3_retrieve.py`
**Input:** `stage2_gated.parquet` (+ `dense_vectors.npy`, `faiss_index.bin`, `bm25_index.pkl`, `id_row_map.parquet`)
**Output:** `stage3_retrieved.parquet`
**Expected reduction:** ~3,000–4,000 → ~300–600
**Vectors used:** YES — dense (768-d orthogonal subspace) + sparse (BM25).

### 3.0 Philosophy

A single JD embedding is semantically lossy: the JD encodes at least four different
intents (hard requirements, ideal career shape, anti-patterns, exact tech vocabulary).
This stage **decomposes the JD into multiple query vectors**, retrieves per query,
fuses by rank, and **subtracts** an anti-pattern signal. This is what lets us find the
**Tier-5 plain-language experts** (via dense Q1/Q2) while still rewarding exact-jargon
matches (via sparse Q4) and actively penalizing **researcher / framework-enthusiast /
consulting-only** profiles (via negative Q3).

### 3.1 Restricting the search space to Stage 2 survivors

Stage 3 must only consider the candidates that passed Stage 2. Implementation:
1. Load surviving `candidate_id`s from `stage2_gated.parquet`.
2. Map them to dense-vector row indices via `id_row_map.parquet`.
3. Build a FAISS `IDSelector` bitmask of exactly those rows.
4. All dense queries run with this selector so the index **physically ignores** gated-out rows.
5. For BM25, restrict scoring to the surviving id set (BM25 has no native selector — filter
   the result list post-hoc, or pre-subset the corpus).

> This preserves the bible's `IDSelector`-bitmask design and means Stage 3 never re-evaluates
> a keyword stuffer or honeypot.

### 3.2 Dense-vector contract (consistency with offline precompute)

- `dense_vectors.npy` is `(N, 768)` float32 in **orthogonal subspace concatenation** layout:
  - dims `0–255` = Retrieval Systems subspace
  - dims `256–511` = Infrastructure subspace
  - dims `512–767` = Evaluation subspace
- The FAISS index is built with **Inner Product (dot product)** metric, **NOT cosine**.
  Global L2 normalization is forbidden — it collapses the orthogonal subspace boundaries.
  (If any per-subspace normalization is desired, it must be done **per 256-d block** offline,
  documented in `precompute.py`, and applied identically to query vectors.)
- Behavioral signals are intentionally absent from these vectors. They enter at Stage 5.

### 3.3 The four queries

All query texts are constructed once at the top of the stage from the JD and frozen in
`config.yaml` (`stage3.queries:`) so retrieval is reproducible.

**Q1 — Technical-requirements query (dense, positive).**
Text = the JD's "Things you absolutely need" section only:
production embeddings-based retrieval; production vector DB / hybrid search infra; strong
Python; evaluation framework design (NDCG/MRR/MAP, offline-to-online correlation, A/B tests).
Encode → 768-d. Apply subspace priority weights via **scalar multiplication on each 256-d
block** before search (per bible Step 4.1). Default weights (tune later):
`retrieval=0.35, infra=0.45, eval=0.20` — infra slightly highest because the JD lists vector-DB
ops first and emphasizes operational experience.

**Q2 — Career-shape query (dense, positive).**
Text = the "How to read between the lines / ideal candidate" paragraph, **rephrased as a
candidate self-description** rather than a job ad. Example phrasing:
"6–8 years total, 4–5 in applied ML at product companies (not services); shipped an
end-to-end ranking/search/recommendation system to real users at scale; opinionated about
hybrid vs dense retrieval and offline vs online evaluation, backed by systems actually built."
Encode → 768-d (same subspace weighting or a flatter `0.33/0.34/0.33` — tune).

> Why rephrase as a self-description: candidate profiles are written in the first person /
> résumé voice. A query written in JD voice ("we are looking for...") sits in a different
> region of embedding space than candidate text. Matching voice to voice improves recall,
> especially for Tier-5 candidates who never use ML jargon.

**Q3 — Negative / anti-pattern query (dense, SUBTRACTED).**
Text = the "Things we explicitly do NOT want" section: title-chasers, framework enthusiasts
(LangChain-tutorial GitHub), consulting-only careers, CV/speech/robotics-without-NLP,
closed-source-only-no-external-validation. Encode → 768-d. This vector is **not** retrieved
against; instead, for every candidate in the fused set we compute `sim(cand, Q3)` and
subtract a scaled penalty (see 3.5).

**Q4 — Sparse jargon query (BM25).**
Exact tokens: FAISS, Qdrant, Weaviate, Milvus, Pinecone, OpenSearch, Elasticsearch, HNSW,
NDCG, MRR, MAP, BGE, E5, sentence-transformers, cross-encoder, embedding drift, hybrid search,
learning-to-rank, A/B test, RAG. BM25 over the clean jargon text block built in precompute.

### 3.4 Per-query top-k BEFORE fusion (critical ordering)

RRF is **rank-based** and needs complete per-query rank lists. Do NOT fuse raw similarity
over the whole set. Order of operations:

```
Q1 dense  → retrieve top  per_query_k_dense  (default 1000)  → ranked list L1
Q2 dense  → retrieve top  per_query_k_dense  (default 1000)  → ranked list L2
Q4 BM25   → retrieve top  per_query_k_sparse (default  500)  → ranked list L4
candidate_union = unique(L1 ∪ L2 ∪ L4)        # ~1,500–2,000 unique
```

A candidate absent from a given list is treated as rank `per_query_k + 1` for that list
(a defined miss penalty), **not** dropped — this is what makes RRF robust to a candidate
being strong on one axis and unseen on another.

### 3.5 Fusion + negative penalty

**Reciprocal Rank Fusion** over the three positive lists:

```
RRF(c) = Σ_{L in {L1, L2, L4}}  1 / (rrf_k + rank_L(c))
```

`rrf_k` default 60 (standard). Then apply the negative anti-pattern penalty:

```
fused_score(c) = RRF(c)  −  alpha_neg * sim(c, Q3)
```

`alpha_neg` default 0.5 (tune carefully — too high and we punish anyone who *mentions*
LangChain even in a legitimate context; the JD penalizes framework *enthusiasts*, not
framework *users*).

Optionally also add a small positive nudge from Stage-1 cluster signal:

```
fused_score(c) += beta_cluster * (1 / (1 + dist_to_centroid))   # beta_cluster default 0.0 (OFF until validated)
```

Keep `beta_cluster` at 0 initially — only enable after confirming the clustering actually
correlates with JD relevance.

### 3.6 Adaptive top-k cut (not a flat top-500)

A flat top-k is fragile (score cliffs include junk; score plateaus exclude near-ties).
Use adaptive selection:

```
mu    = mean(fused_score over union)
sigma = std(fused_score over union)
threshold = mu - z * sigma            # z default 1.5

k = count(candidates with fused_score >= threshold)
k = max(k, min_k)                     # min_k default 300
k = min(k, max_k)                     # max_k default 600

select top-k by fused_score (descending)
```

- `min_k = 300` guarantees enough candidates for the cross-encoder to recover precision.
- `max_k = 600` caps Stage 4 cross-encoder inference time.
- The threshold floor adapts the cut to each run's actual score distribution.

Tie-break within the cut by `candidate_id` ascending (deterministic, per spec §3).

### 3.7 (Optional, v2) Pseudo-Relevance Feedback expansion

Document but leave OFF in v1:
after the first pass, take the top-20 fused candidates, extract their highest-weight BM25
terms, append to Q4, and re-run the sparse track once (Rocchio-style). Helps recover
Tier-5 candidates whose vocabulary the JD didn't anticipate. Gate behind `stage3.use_prf: false`.

### 3.8 Stage 3 output contract

**`stage3_retrieved.parquet` columns (one row per retrieved candidate, ~300–600 rows):**

| Column | Type | Source |
|---|---|---|
| `candidate_id` | string | passthrough |
| all Stage 2 columns | — | passthrough (we do NOT re-read raw data downstream) |
| `q1_score` | float | Q1 dense similarity |
| `q2_score` | float | Q2 dense similarity |
| `bm25_score` | float | Q4 |
| `q3_neg_sim` | float | similarity to anti-pattern vector |
| `rrf_score` | float | fusion before penalty |
| `fused_score` | float | final Stage 3 score (RRF − penalty) |
| `stage3_rank` | int | provisional rank by `fused_score`, 1 = best |

> Carry **all** scores forward as separate columns — Stage 5's LightGBM will consume them
> individually as features, not just the final fused number.

**Validation after Stage 3:**
1. Row count in `[min_k, max_k]`.
2. Manually read the top-10 profiles — do they look like real senior ML/IR engineers?
   Any honeypot or keyword stuffer in the top-10 means Stage 2 leaked; fix upstream.
3. Pick 3 candidates you *know* are Tier-5 plain-language fits (if you have labels) and
   confirm they survived — if not, Q2 phrasing or `alpha_neg` needs tuning.

---

## STAGE 4 — CROSS-ENCODER RERANKING  *(surface-level placeholder)*

**Script:** `stages/stage4_crossenc.py`
**Input:** `stage3_retrieved.parquet`
**Output:** `stage4_reranked.parquet`
**Expected reduction:** ~300–600 → ~300

### Intent (to be expanded later)

At a few hundred candidates we can afford a **cross-encoder** — a model that jointly encodes
(JD, candidate) pairs and scores relevance directly, which is meaningfully more accurate than
the bi-encoder similarity used in Stage 3. Too slow at 100K; feasible here on CPU.

### Surface-level shape

- Build a compact pairwise input per candidate: `(JD_summary_text, candidate_summary_text)`,
  using the pre-computed candidate technical summary from Track B.
- Run a small distilled cross-encoder exported to **ONNX** (e.g. an `ms-marco-MiniLM`-class
  model) on CPU via `onnxruntime`. No PyTorch online.
- Produce `cross_encoder_score` per candidate.
- Re-sort by `cross_encoder_score`, keep top ~300, write through all prior columns plus the
  new score.

### Open questions deferred to the deep-dive

- Exact model choice + ONNX export path + token budget per pair.
- Whether to feed the full JD or a distilled JD prompt.
- Whether Stage 4 *cuts* (600→300) or only *scores* (keeps all, lets Stage 5 cut).
- CPU latency budget check at 600 pairs.

**Output contract (provisional):** all Stage 3 columns + `cross_encoder_score` (float) +
`stage4_rank` (int). ~300 rows.

> This stage is intentionally underspecified for now. Do not over-build it until Stage 3
> output quality is confirmed.

---

## STAGE 5 — LIGHTGBM RERANKING  *(placeholder stub only)*

**Script:** `stages/stage5_lgbm.py`
**Input:** `stage4_reranked.parquet` (+ `candidate_features.parquet` for full 23 signals)
**Output:** `team_xxx.csv` (exactly 100 rows + header)

### Status

**STUB ONLY.** The model design — feature set finalization, label strategy, training data,
objective (`lambdarank` / NDCG@10), hyperparameters — is a **separate research task** and is
deliberately NOT specified here. For now this file should:

1. Load `stage4_reranked.parquet`.
2. Join the full 23 behavioral signals + tabular features from `candidate_features.parquet`.
3. Assemble the feature matrix (placeholder: just concatenate all available numeric features).
4. **Placeholder scoring:** until the real model exists, score = `cross_encoder_score` (or
   `fused_score` if Stage 4 not yet built). This lets the full pipeline run end-to-end and
   emit a valid CSV for format testing **before** the real model is ready.
5. Sort descending, slice top 100.
6. Write `team_xxx.csv` strictly conforming to the spec.

### Things the eventual model MUST respect (recorded now so we don't forget)

- Behavioral signals enter as features the model weights — **never** as an independent axis
  that can rescue a non-engineer (avoid "behavioral rescue").
- Availability flags (`stale_profile`, `low_responder`, `not_open`) should act as a learned
  **dampener**, not a hard zero.
- Output must satisfy spec §3 exactly: 100 data rows, ranks 1–100 each once, `score`
  monotonically non-increasing with rank, ties broken deterministically by `candidate_id`.
- `reasoning` column generated from the **pre-computed** Track B technical summary +
  candidate-specific facts (no online LLM). Must be specific, non-templated, and honest about
  concerns — these are exactly the Stage-4-manual-review checks in the submission spec.

### Output contract

`team_xxx.csv`: `candidate_id,rank,score,reasoning` — exactly 100 data rows, UTF-8.

---

## APPENDIX A — CONFIG NAMESPACES (for `config.yaml`)

```yaml
stage2:
  hard_min: 5
  hard_max: 9
  soft_tolerance: 1
  sweet_low: 6
  sweet_high: 8
  stuffer_density: 0.5
  expert_zero_threshold: 1
  overlap_tolerance_years: 0.5
  stale_days: 180
  min_response_rate: 0.10
  enable_isolation_forest: false

stage3:
  per_query_k_dense: 1000
  per_query_k_sparse: 500
  rrf_k: 60
  alpha_neg: 0.5
  beta_cluster: 0.0
  subspace_weights_q1: { retrieval: 0.35, infra: 0.45, eval: 0.20 }
  subspace_weights_q2: { retrieval: 0.33, infra: 0.34, eval: 0.33 }
  z_threshold: 1.5
  min_k: 300
  max_k: 600
  use_prf: false

stage4:
  keep_n: 300
  # model + onnx details TBD

stage5:
  team_id: "team_xxx"
  top_n: 100
  # model details TBD
```

## APPENDIX B — MANUAL RUN CHEAT SHEET

```bash
# Stage 2
python stages/stage2_gate.py \
  --in artifacts/runtime/stage1_clustered.parquet \
  --out artifacts/runtime/stage2_gated.parquet
# → inspect stage2_gated.parquet, stage2_honeypot_log.csv, stage2_removed_log.csv

# Stage 3
python stages/stage3_retrieve.py \
  --in artifacts/runtime/stage2_gated.parquet \
  --out artifacts/runtime/stage3_retrieved.parquet
# → inspect top-10 profiles manually

# Stage 4 (placeholder)
python stages/stage4_crossenc.py \
  --in artifacts/runtime/stage3_retrieved.parquet \
  --out artifacts/runtime/stage4_reranked.parquet

# Stage 5 (placeholder)
python stages/stage5_lgbm.py \
  --in artifacts/runtime/stage4_reranked.parquet \
  --out artifacts/runtime/team_xxx.csv
```

> Reminder: each command is run **manually and independently**. Verify the output of each
> stage before running the next. The final `rank.py` wrapper (for official single-command
> reproduction) chains these same `run()` entry points but is not used during development.
