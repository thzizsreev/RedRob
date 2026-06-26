# STAGE 3 — MULTI-QUERY HYBRID RETRIEVAL
## Implementation Plan for AI Coding Agent

---

## WHAT THIS DOCUMENT IS

This is a **complete replacement** of the previous Stage 3 plan. The overall purpose,
position in the pipeline, input/output contracts, FAISS mechanics, RRF fusion logic,
Q3 negative penalty, and adaptive top-k cutoff are all **unchanged**. The only change
is that the BM25 sparse track (L4) has been **replaced** by a structured skill-score
track (L3) built from precomputed per-candidate skill_weighted_score values.

The coding agent must implement this document from scratch. Do not carry over any BM25
code from a previous implementation. Do not import `rank_bm25`. Do not load
`bm25_index.pkl`. Everything else about Stage 3 is identical to what was previously
specified — the agent must preserve it exactly.

---

## OVERVIEW

**Script name:** `stages/stage3_retrieve.py`
**Runs after:** Stage 2 (`stage2_gate.py`) has completed and its output has been manually
inspected. Also requires precompute to have completed and written
`candidate_features.parquet` with the `skill_weighted_score` column present.
**Runs before:** Stage 4 (`stage4_crossenc.py`).

**What changed from the previous version:**
- BM25 sparse track (L4, `bm25_index.pkl`, `rank_bm25`) → **REMOVED**
- `skill_weighted_score` track (L3, from `candidate_features.parquet`) → **ADDED**
- RRF now fuses three lists (L1, L2, L3) instead of three lists (L1, L2, L4)
- Output column `bm25_score` / `bm25_rank` → **REMOVED**
- Output column `skill_score` / `skill_rank` → **ADDED**
- Everything else: unchanged

**Input artifacts (all read-only):**
- `artifacts/runtime/stage2_gated.parquet` — candidates that survived Stage 2, ~3,000–4,000 rows
- `artifacts/precomputed/dense_vectors.npy` — numpy array of shape `(N_total, D)`.
  N_total is all original candidates before any filtering. D is the full concatenated
  Instructor embedding dimension (3 subspaces concatenated). Each row is one candidate.
- `artifacts/precomputed/faiss_index.bin` — FAISS flat index built with Inner Product
  metric over `dense_vectors.npy`. Do NOT use cosine similarity.
- `artifacts/precomputed/id_row_map.parquet` — columns: `candidate_id` (string) and
  `dense_row_idx` (int). Maps each candidate_id to its row index in `dense_vectors.npy`.
- `artifacts/precomputed/candidate_features.parquet` — Track B tabular store. Stage 3
  reads exactly two columns from this file: `candidate_id` and `skill_weighted_score`.
  No other columns are needed by Stage 3.

**Removed input (do not load):**
- ~~`artifacts/precomputed/bm25_index.pkl`~~ — no longer used. Do not load.

**Output artifact:**
- `artifacts/runtime/stage3_retrieved.parquet` — ~300–600 rows, containing all Stage 2
  columns plus new score columns added by Stage 3.

**What this stage does in one sentence:**
Decomposes the Job Description into three distinct query signals — two dense Instructor
queries (Q1 technical requirements, Q2 career shape) and one structured skill-score query
(L3 from precomputed skill_weighted_score) — retrieves ranked lists independently per
signal, fuses via Reciprocal Rank Fusion, applies an anti-pattern penalty via Q3, and
cuts to the top ~300–600 candidates using an adaptive score threshold.

---

## CRITICAL DESIGN CONSTRAINTS

Read all of these before writing a single line of code. None of these changed from the
previous version.

**C1. No PyTorch online.** The Instructor model must be loaded via `onnxruntime` only.
Do not import `torch`, `transformers`, `sentence_transformers`, or `InstructorEmbedding`.
The ONNX export of the model must already exist at `models/instructor.onnx`.

**C2. FAISS must use Inner Product, not Cosine Similarity.** The index was built with
`faiss.METRIC_INNER_PRODUCT`. Do NOT L2-normalize query vectors before passing to FAISS.
Normalizing the full concatenated vector destroys the orthogonal subspace boundaries.

**C3. Instruction strings must exactly match precompute.** The three instruction strings
for Q1, Q2, Q3 subspace blocks are stored in `config.yaml` under `stage3.instructions`.
Read them from config. Never hardcode them. Any difference between precompute and Stage 3
instruction strings makes the dot products mathematically meaningless.

**C4. Stage 2 survivors only.** Build a FAISS `IDSelector` bitmask from Stage 2 survivor
row indices. All FAISS searches must use this selector. Candidates removed in Stage 2 must
never appear in any ranked list — they are excluded at the index level, not filtered
post-hoc.

**C5. skill_weighted_score is also filtered to Stage 2 survivors.** When building L3
from `candidate_features.parquet`, only include candidates whose `candidate_id` appears
in `stage2_gated.parquet`. Do not rank the full precomputed set — same restriction as FAISS.

**C6. All per-query scores written as separate columns.** Do not collapse into a single
number before writing. Downstream stages consume each score column independently.

**C7. No hardcoded thresholds or counts.** Every numeric parameter is read from
`config.yaml` under `stage3:`. The agent must not write a single numeric literal for
any threshold, k value, weight, or count.

**C8. Expose both CLI and `run()` function.** The CLI (`--input`, `--output`, `--config`)
is for manual runs. The `run(input_path, output_path, config)` function is called by the
final `rank.py` wrapper for hackathon submission reproduction.

---

## TECH STACK

| Library | Purpose |
|---|---|
| `polars` | All DataFrame operations — reading/writing parquet, joins, filtering, score assembly |
| `numpy` | Dense vector operations — loading `.npy`, dot products, score arrays |
| `faiss-cpu` | Dense vector search via Inner Product |
| `onnxruntime` | Inference for the Instructor ONNX model to encode query vectors |
| `pyyaml` | Reading config.yaml |
| `argparse` | CLI argument parsing |

**Removed from previous version:**
- ~~`rank_bm25`~~ — BM25 is gone. Do not import it.
- ~~`pickle`~~ — no longer needed (bm25_index.pkl is not loaded).

---

## BACKGROUND: HOW INSTRUCTOR EMBEDDINGS WORK IN THIS SYSTEM

The coding agent must understand this before implementing query encoding. This section
is unchanged from the previous plan.

During precompute, each candidate's profile text was encoded three times with the
Instructor ONNX model — once per subspace — each time using a different instruction
string. The three resulting vectors (each of dimension `d`) were concatenated in fixed
order: `[retrieval_vec | infra_vec | eval_vec]` producing a single `(3d,)` vector per
candidate. This is one row in `dense_vectors.npy`.

Instructor takes `(instruction, text)` as input and produces an embedding conditioned
on the instruction. The same text with a different instruction produces a semantically
different embedding. This is what enables subspace separation: the retrieval subspace
vector specifically encodes retrieval-relevant content from the profile, the infra
subspace encodes infrastructure content, and the eval subspace encodes evaluation content.

At query time in Stage 3, the same encoding procedure is applied to the query text:
encode it three times with the three subspace instructions, concatenate in the same
order, producing a `(3d,)` query vector. The query and candidate vectors live in the
same space and are comparable via dot product (Inner Product in FAISS).

Subspace weights are applied after encoding by scalar-multiplying each `d`-length block
of the query vector before passing to FAISS. This amplifies or dampens each subspace's
contribution to the dot product independently.

---

## BACKGROUND: WHY skill_weighted_score REPLACES BM25

The coding agent must understand the architectural reasoning to implement correctly.

**What BM25 did:** scored candidates by how many JD jargon tokens appeared in their
free-form profile text (jargon text block), weighted by corpus-level TF-IDF. It found
candidates who used the exact vocabulary of the JD.

**Why it was replaced:** the dense vectors (Q1, Q2) already capture semantic meaning from
free-form profile text — they find candidates who describe relevant experience regardless
of vocabulary. BM25's token-matching was a partial subset of what Q1 already covers. Its
real limitation was that it treated "FAISS: 6 years expert" and "FAISS: 3 months beginner"
identically — it had no concept of depth, proficiency, or rarity.

**What skill_weighted_score does instead:** it operates on the structured skills section
of the candidate profile — a different data source than the prose text that Q1/Q2 operate
on. It combines four signals per skill: tier-based relevance to the JD (via 5-tier
Gaussian classification), duration of experience (log-scaled), proficiency level, and
rarity across the full 100K candidate pool (IDF-based). This produces a score that
is orthogonal to the dense vectors — it captures structured depth rather than prose meaning.

**The three tracks are now genuinely orthogonal:**
- L1 (Q1 dense): semantic meaning of prose against technical requirements
- L2 (Q2 dense): semantic meaning of prose against career shape / Tier-5 rescue
- L3 (skill score): structured depth, duration, proficiency, rarity from skills section

A candidate with strong skills but sparse prose (writes tersely) is now retrieved via L3
even if L1/L2 rank them lower. A candidate who writes richly about relevant experience
but has a sparse skills section is retrieved via L1/L2. RRF rewards candidates who are
strong on multiple tracks.

---

## THE THREE QUERIES

### Q1 — Technical Requirements Query (dense, positive)

**Purpose:** Find candidates with production experience in the JD's must-haves:
embeddings-based retrieval, vector databases, evaluation frameworks. Finds both
ML-vocabulary-rich candidates AND Tier-5 plain-language candidates who describe these
systems without using jargon.

**Q1 text** (stored in `config.yaml` under `stage3.q1_text`):
```
Production experience building and operating embeddings-based retrieval systems deployed
to real users. Handled embedding drift, index refresh cycles, and retrieval quality
regression in production environments. Built or operated vector database infrastructure
using systems such as Pinecone, Qdrant, Weaviate, Milvus, OpenSearch, Elasticsearch,
or FAISS at meaningful scale. Designed and maintained evaluation frameworks for ranking
systems including NDCG, MRR, MAP, offline-to-online correlation, and A/B testing of
ranking changes. Strong Python engineering skills in production ML systems.
```

**Instruction strings** (from `config.yaml` under `stage3.instructions.q1`, must match
precompute exactly):
- Retrieval subspace: `"Represent the candidate's production experience with embeddings, semantic search, dense retrieval, and recommendation systems:"`
- Infra subspace: `"Represent the candidate's hands-on experience operating vector databases, search infrastructure, and ML systems at scale in production:"`
- Eval subspace: `"Represent the candidate's experience designing and running evaluation frameworks, ranking metrics, A/B tests, and offline-to-online validation for ML systems:"`

**Subspace weights** (from `config.yaml` under `stage3.subspace_weights_q1`):
- Retrieval: 0.35, Infra: 0.45, Eval: 0.20

Infra weighted highest because the JD prioritizes operational experience over theory.

**Encoding procedure:**
1. Encode Q1 text three times — once per subspace instruction — using `onnxruntime`.
2. Each encoding produces a `(d,)` vector where `d = D/3`.
3. Concatenate in order `[retrieval | infra | eval]` → `(D,)` vector.
4. Apply subspace weights: multiply dims `[0:d]` by 0.35, dims `[d:2d]` by 0.45,
   dims `[2d:3d]` by 0.20.
5. Reshape to `(1, D)` for FAISS.

**Retrieval:** FAISS search with IDSelector. Top `per_query_k_dense` (default 1000).
Output: ranked list `L1` — `(candidate_id, q1_score)` sorted by descending dot product.

---

### Q2 — Career Shape Query (dense, positive)

**Purpose:** Find candidates whose career trajectory matches the JD's ideal profile.
Specifically rescues Tier-5 plain-language candidates who don't use ML jargon but have
the right background. Phrased as a candidate self-description (not a job ad) because
candidate profile text and JD text sit in different regions of embedding space — matching
voice to voice improves recall.

**Q2 text** (stored in `config.yaml` under `stage3.q2_text`):
```
Six to eight years of total experience, four to five of which were in applied ML at
product companies. Shipped end-to-end ranking, search, or recommendation system to real
users at scale. Strong opinions on hybrid versus dense retrieval and offline versus online
evaluation, grounded in systems actually built and run in production. Prefer shipping a
working system and iterating over designing the perfect system. Not from a pure research
background. Built systems at product companies, not consulting firms.
```

**Instruction strings** (from `config.yaml` under `stage3.instructions.q2`):
- Retrieval subspace: `"Represent the candidate's career experience with search, ranking, recommendation, and retrieval systems they personally shipped to production:"`
- Infra subspace: `"Represent the candidate's background at product companies building and scaling ML infrastructure for real users:"`
- Eval subspace: `"Represent the candidate's hands-on experience with evaluation, iteration, and improvement of production ML systems based on real user feedback:"`

**Subspace weights** (from `config.yaml` under `stage3.subspace_weights_q2`):
- Retrieval: 0.33, Infra: 0.34, Eval: 0.33

Flat weights because Q2 looks for holistic career shape, not one specific technical axis.

**Encoding procedure:** identical to Q1 — three encodings, concatenate, apply weights,
reshape to `(1, D)`. Use Q2's instruction strings and weights, not Q1's.

**Retrieval:** FAISS search with IDSelector. Top `per_query_k_dense` (default 1000).
Output: ranked list `L2` — `(candidate_id, q2_score)` sorted by descending dot product.

---

### Q3 — Anti-Pattern Query (dense, NEGATIVE — used for penalty only, NOT retrieval)

**Purpose:** Identify candidates who resemble the JD's explicit "do NOT want" profiles.
Q3 is NEVER sent to FAISS for retrieval. After the candidate union is built from L1, L2,
L3, compute each candidate's dot product similarity to Q3 and subtract a scaled penalty
from their fused score. Candidates resembling framework enthusiasts, consulting-only
engineers, pure researchers, or CV/speech specialists without NLP/IR are down-ranked.

**Q3 text** (stored in `config.yaml` under `stage3.q3_text`):
```
Career shows switching companies every one to two years for senior titles. GitHub
consists primarily of LangChain tutorials and framework demos with no evidence of
production systems thinking. Entire career at consulting firms such as TCS, Infosys,
Wipro, Accenture with no product company experience at any point. Background is purely
academic research with no production deployments of any ML system to real users.
Primary expertise is in computer vision or speech processing without meaningful
information retrieval or NLP experience. Five or more years on closed-source proprietary
systems with no external validation through papers, talks, or open-source contributions.
```

**Instruction strings** (from `config.yaml` under `stage3.instructions.q3`):
- Retrieval subspace: `"Represent the candidate's experience or lack thereof with search, retrieval, and ranking in production:"`
- Infra subspace: `"Represent the candidate's background in terms of company type, deployment context, and engineering environment:"`
- Eval subspace: `"Represent the candidate's approach to building and validating ML systems, including signs of research-only or framework-only orientation:"`

**Encoding procedure:** same three-encoding, concatenate, reshape procedure. Use flat
subspace weights (0.33 / 0.34 / 0.33) — no amplification of any subspace for the
anti-pattern vector.

**How Q3 is used:** after the candidate union is assembled, extract the dense vectors
for all union candidates from `dense_vectors.npy` using their `dense_row_idx` values.
Compute numpy dot product of each candidate vector against the flat (non-reshaped) `(D,)`
Q3 vector:
```
union_vecs = dense_vectors[union_dense_row_idxs]   # shape: (union_size, D)
q3_sims    = union_vecs @ q3_vec                    # shape: (union_size,)
```
Attach as column `q3_neg_sim`. Higher value = more similar to disqualifier profile =
more penalty in the fusion formula.

---

### L3 — Skill-Score Track (structured, positive)

**Purpose:** This is the replacement for BM25. It ranks Stage 2 survivors by their
precomputed `skill_weighted_score` — a scalar that combines tier-based skill relevance,
years of experience, proficiency level, and IDF-based rarity across 100K candidates.
It operates on the structured skills section of the profile, which Q1 and Q2 do not
directly access. This surfaces candidates with deep, rare, relevant skills even if their
prose encoding does not rank them highly.

**Where skill_weighted_score comes from:** it was computed entirely offline in precompute
by the skill scoring pipeline (Blocks A–D). The agent does NOT implement the scoring
formula here — Stage 3 only consumes the precomputed scalar. The formula is:

```
Per skill s:
  Block A: relevance_score(s) = Σ_i w_Ti × tier_reward(Ti)
           where w_Ti = gaussian_weight_Ti / Σ gaussian_weight_Tj
           and   gaussian_weight_Ti = exp(−(1 − sim_Ti(s))² / 2σ²)
           and   tier_rewards: T1=1.0, T2=0.65, T3=0.35, T4=0.25, T5=0.05

  Block B: depth_score(s) = 0.6 × min(log(1+years)/log(11), 1.0)
                           + 0.4 × proficiency_score(s)
           proficiency_score: {null:0.5, beginner:0.3, intermediate:0.6,
                                advanced:0.85, expert:1.0}

  Block C: rarity_weight(s) = clip_normalize(log((N+1)/(df(s)+1)), idf_min, idf_max)

Per candidate c:
  Block D: skill_score_raw(c) = Σ relevance_score(s) × depth_score(s) × rarity_weight(s)
                                 over top-15 skills by (depth × rarity)
           skill_weighted_score(c) = min_max(clip(skill_score_raw, P95))
                                     computed globally over all 100K
```

The agent does not need to re-derive or verify this formula. It is documented here
only so the agent understands what the number means and why it is a valid retrieval signal.

**How to use skill_weighted_score in Stage 3:**

Step 1: Load `candidate_features.parquet`. Extract only the two columns needed:
`candidate_id` and `skill_weighted_score`. The parquet may contain many other columns —
do not load them all into memory. Use Polars column selection.

Step 2: Filter to Stage 2 survivors only. Inner join (or filter by is_in) the loaded
skill scores against the Stage 2 survivor candidate_id set. Drop any candidate not in
Stage 2's output. The result is a DataFrame of ~3,000–4,000 rows.

Step 3: Sort by `skill_weighted_score` descending. Take the top `per_query_k_skill`
(config default 500). Assign rank starting from 1. This is ranked list `L3`.

Step 4: Carry the raw `skill_weighted_score` value as the score column for L3.
Rename appropriately: `skill_score` (raw value), `skill_rank` (rank in L3, with
miss-penalty for candidates not in top 500).

---

## STEP-BY-STEP PROCEDURE

### STEP 1 — Parse Arguments and Load Config

Accept CLI args: `--input` (stage2_gated.parquet), `--output`
(stage3_retrieved.parquet), `--config` (default config.yaml). Load YAML. Extract the
full `stage3:` block. Every numeric parameter referenced below comes from this block.

Validate that all required config keys are present. Fail loudly with a descriptive
message naming the missing key if any is absent. Do not proceed with defaults for missing
required keys.

---

### STEP 2 — Load All Assets

Load in this order, failing loudly on any missing file:

1. `stage2_gated.parquet` → Polars DataFrame. Log row count. Verify `candidate_id`
   column exists.

2. `dense_vectors.npy` → numpy float32 array shape `(N_total, D)`. Log shape.
   Derive `d = D // 3` (per-subspace dimension).

3. `faiss_index.bin` → FAISS index via `faiss.read_index`. Verify it contains N_total
   vectors. Log total vector count.

4. `id_row_map.parquet` → Polars DataFrame, columns `candidate_id` and `dense_row_idx`.

5. `candidate_features.parquet` → Polars DataFrame, **select only** columns
   `candidate_id` and `skill_weighted_score`. Do not load the entire file.
   Verify `skill_weighted_score` column exists after load. If missing, halt with
   error: "skill_weighted_score column not found in candidate_features.parquet —
   ensure precompute skill scoring (Blocks A-D) has been run."

6. Instructor ONNX session → `onnxruntime.InferenceSession('models/instructor.onnx')`.
   Log input/output node names.

---

### STEP 3 — Build FAISS IDSelector

Join `stage2_gated.parquet` with `id_row_map.parquet` on `candidate_id`. Extract the
`dense_row_idx` values for all Stage 2 survivors as a numpy int64 array. Sort the array.

Create `faiss.IDSelectorBatch(dense_idxs)`. Store as `survivor_selector`.

This selector is passed to every FAISS search call. It physically restricts search to
Stage 2 survivors only — no post-hoc filtering needed for FAISS results.

---

### STEP 4 — Encode Q1, Q2, Q3

For each of Q1, Q2, Q3, execute the following encoding procedure. The procedure is
identical for all three — only the instruction strings, query text, and subspace weights
differ. Read all of these from config.

**4a.** For each of the three subspaces (retrieval, infra, eval):
- Retrieve the instruction string from `config.stage3.instructions.qN.subspace_name`
- Combine with the query text in the format used during precompute
  (VERIFY this format against `precompute.py` before implementing — the exact
  format of the Instructor input must match)
- Run inference via `onnxruntime` session using the input node name identified in Step 2
- Output is a `(d,)` vector

**4b.** Concatenate the three subspace vectors in fixed order `[retrieval | infra | eval]`
→ `(D,)` vector.

**4c.** For Q1 and Q2: apply subspace weights by scalar-multiplying each `d`-length block.
For Q3: apply flat weights (0.33 / 0.34 / 0.33) — do not amplify any subspace.

**4d.** For Q1 and Q2: reshape to `(1, D)` for FAISS. For Q3: keep as `(D,)` flat vector
for numpy dot product — do not reshape.

Do NOT L2-normalize any vector at any point.

After this step: `q1_vec` shape `(1, D)`, `q2_vec` shape `(1, D)`, `q3_vec` shape `(D,)`.

---

### STEP 5 — FAISS Retrieval for Q1 and Q2

For each of Q1 and Q2:

Create `faiss.SearchParameters` with `sel = survivor_selector`. Call:
```
scores, indices = faiss_index.search(query_vec, k=per_query_k_dense, params=search_params)
```
Returns `scores` shape `(1, k)` and `indices` shape `(1, k)`. Flatten both to 1D.

Map `indices` (FAISS row indices = dense_row_idx values) back to `candidate_id` using
`id_row_map`. Drop any index that is -1 (FAISS returns -1 for unfilled slots when fewer
than k valid candidates exist within the selector).

Build Polars DataFrames:
- `L1`: columns `candidate_id`, `q1_score` (raw dot product, no normalization),
  `q1_rank` (1 = highest score). Sort by descending q1_score before assigning rank.
- `L2`: columns `candidate_id`, `q2_score`, `q2_rank`. Same procedure.

---

### STEP 6 — Build L3 from skill_weighted_score

Step 6a: Filter the loaded `candidate_features` DataFrame (already restricted to
`candidate_id` and `skill_weighted_score`) to only include candidates present in
`stage2_gated`. Use an inner join or `is_in` filter on `candidate_id`. This enforces
the Stage 2 survivor restriction for the skill track.

Step 6b: Sort by `skill_weighted_score` descending. Take the top `per_query_k_skill`
rows (config default 500).

Step 6c: Assign `skill_rank` starting from 1 (rank 1 = highest skill_weighted_score).

Step 6d: Build Polars DataFrame `L3`: columns `candidate_id`, `skill_score`
(= `skill_weighted_score` value), `skill_rank`.

This is a direct read from a precomputed column. There is no encoding, no index search,
no heavy computation. It is a sort and slice.

---

### STEP 7 — Build Candidate Union

Take the union of all `candidate_id` values appearing in L1, L2, L3. A candidate may
appear in one, two, or all three lists. The union contains ~1,500–2,200 unique candidates.

Build a master union DataFrame with one row per unique `candidate_id`. Left-join L1, L2,
L3 onto this master on `candidate_id`. For candidates missing from a given list, score
and rank columns will be null after the join.

Fill null rank values with miss-penalty ranks:
- Null `q1_rank` → `per_query_k_dense + 1` (default 1001)
- Null `q2_rank` → `per_query_k_dense + 1` (default 1001)
- Null `skill_rank` → `per_query_k_skill + 1` (default 501)

Null score values (q1_score, q2_score, skill_score) remain null — they are stored as
columns in the final output but not used in any arithmetic directly (ranks are used for
RRF, not raw scores).

Also join `dense_row_idx` onto the union from `id_row_map`. This is needed in Step 8 for
Q3 dot product computation.

---

### STEP 8 — Compute RRF Score and Q3 Penalty

**8a. Reciprocal Rank Fusion:**

For each candidate in the union, compute:
```
RRF_score(c) = 1 / (rrf_k + q1_rank(c))
             + 1 / (rrf_k + q2_rank(c))
             + 1 / (rrf_k + skill_rank(c))
```
Where `rrf_k` = 60 (from config). Candidates missing from a list use the miss-penalty
rank filled in Step 7, producing a small but non-zero contribution. Compute as a Polars
expression over the union DataFrame. Store as column `rrf_score`.

**8b. Q3 Anti-Pattern Similarity:**

Extract `dense_row_idx` values for all union candidates as a numpy integer array. Gather
their dense vectors:
```
union_vecs = dense_vectors[union_dense_row_idxs]   # shape: (union_size, D)
q3_sims    = union_vecs @ q3_vec                    # shape: (union_size,)
```
Attach as column `q3_neg_sim` on the union DataFrame. Higher value = more similar to
the anti-pattern description = more penalty.

**8c. Fused Score:**
```
fused_score(c) = rrf_score(c)
               − alpha_neg × q3_neg_sim(c)
               + beta_cluster × (1 / (1 + dist_to_centroid(c)))
```
Where:
- `alpha_neg` = 0.5 (config) — controls anti-pattern penalty weight
- `beta_cluster` = 0.0 (config, OFF by default) — cluster proximity boost;
  only enable after validating Stage 1 clustering correlates with JD relevance
- `dist_to_centroid` is carried forward from Stage 1 via `stage2_gated.parquet`

Store as column `fused_score`.

---

### STEP 9 — Adaptive Top-K Cutoff

**9a.** Compute mean (`mu`) and std (`sigma`) of `fused_score` over the union.

**9b.** Threshold:
```
threshold = mu − z_threshold × sigma     (z_threshold default 1.5 from config)
```

**9c.** Filter: keep candidates with `fused_score >= threshold`. Count result.
- If count > `max_k` (default 600): take only top `max_k` by descending `fused_score`.
- If count < `min_k` (default 300): override threshold, take top `min_k` by descending
  `fused_score` regardless of threshold.

**9d.** Tiebreak: within any group of equal `fused_score`, sort by `candidate_id`
ascending. This is deterministic and matches the submission spec's tiebreak requirement.

**9e.** Assign `stage3_rank`: integer from 1 (best fused_score) to output row count.

---

### STEP 10 — Assemble Output DataFrame and Write

Join the Stage 2 DataFrame onto the final selected candidates using `candidate_id` as
join key (left join from selected candidates → Stage 2 to preserve ordering). Attach all
Stage 3 score columns.

**Output DataFrame columns — complete list:**

All Stage 2 columns (passthrough, do not drop any):
`candidate_id`, `cluster_id`, `cluster_rank`, `dist_to_centroid`, `total_years_exp`,
`exp_band`, `in_sweet_spot`, `title_family`, `skill_kw_density`, `title_ambiguous`,
`stale_profile`, `low_responder`, `not_open`, `honeypot_anomaly_score`,
plus all Stage 2 modification columns:
`product_company_count`, `consulting_company_count`, `product_company_fraction`,
`career_type`, `research_fraction`, `research_heavy`, `has_any_production_role`,
`stale_coding`, `currently_between_roles`, `months_since_last_ic_role`,
`pre_llm_production_ml`, `recent_ai_only`, `llm_framework_only`,
`ml_experience_start_year`, `avg_tenure_per_employer`, `short_hop_count`,
`title_progression_jumps`, `location_tier`, `external_validation_score`,
`has_github`, `notice_period_days`

New Stage 3 columns:

| Column | Type | Description |
|---|---|---|
| `q1_score` | float, nullable | Raw dot product similarity to Q1 vector. Null if not in L1. |
| `q1_rank` | int | Rank in L1 (1 = best). Miss-penalty value if not in L1. |
| `q2_score` | float, nullable | Raw dot product similarity to Q2 vector. Null if not in L2. |
| `q2_rank` | int | Rank in L2. Miss-penalty value if not in L2. |
| `skill_score` | float, nullable | Precomputed skill_weighted_score value. Null if not in L3. |
| `skill_rank` | int | Rank in L3 (1 = highest skill score). Miss-penalty if not in L3. |
| `q3_neg_sim` | float, non-null | Dot product similarity to Q3 anti-pattern vector. |
| `rrf_score` | float, non-null | Raw RRF fusion score before Q3 penalty. |
| `fused_score` | float, non-null | Final Stage 3 score: RRF − alpha×Q3 + beta×cluster. |
| `stage3_rank` | int | Provisional rank by fused_score. 1 = best. |

Verify output row count is between `min_k` and `max_k` before writing. Raise a hard
error if not. Write to `artifacts/runtime/stage3_retrieved.parquet` via Polars
`write_parquet`.

---

### STEP 11 — Logging and Debug Output

Print to stdout after writing:

1. Input row count (Stage 2 survivors)
2. L1 size, L2 size, L3 size
3. Union size (unique candidates across all three)
4. Overlap count: candidates appearing in all three lists
5. Score distribution over union: min, max, mean, std of `fused_score`
6. Threshold value used
7. Final output row count
8. Top 10 candidates: `candidate_id`, `fused_score`, `q1_score`, `q2_score`,
   `skill_score`, `q3_neg_sim`, `stage3_rank`
9. Bottom 5 candidates in output (lowest fused_score kept)

Also write debug artifact `artifacts/runtime/stage3_score_distribution.csv` — all union
candidates with their scores and whether they were kept or cut. This is for threshold
tuning. It is NOT submitted.

---

## CONFIG BLOCK (complete `stage3:` section for config.yaml)

```yaml
stage3:

  # === Query texts ===
  q1_text: |
    Production experience building and operating embeddings-based retrieval systems
    deployed to real users. Handled embedding drift, index refresh cycles, and retrieval
    quality regression in production environments. Built or operated vector database
    infrastructure using systems such as Pinecone, Qdrant, Weaviate, Milvus, OpenSearch,
    Elasticsearch, or FAISS at meaningful scale. Designed and maintained evaluation
    frameworks for ranking systems including NDCG, MRR, MAP, offline-to-online
    correlation, and A/B testing of ranking changes. Strong Python engineering skills
    in production ML systems.

  q2_text: |
    Six to eight years of total experience, four to five of which were in applied ML
    at product companies. Shipped end-to-end ranking, search, or recommendation system
    to real users at scale. Strong opinions on hybrid versus dense retrieval and offline
    versus online evaluation, grounded in systems actually built and run in production.
    Prefer shipping a working system and iterating over designing the perfect system.
    Not from a pure research background. Built systems at product companies, not
    consulting firms.

  q3_text: |
    Career shows switching companies every one to two years for senior titles. GitHub
    consists primarily of LangChain tutorials and framework demos with no evidence of
    production systems thinking. Entire career at consulting firms such as TCS, Infosys,
    Wipro, Accenture with no product company experience at any point. Background is
    purely academic research with no production deployments of any ML system to real
    users. Primary expertise is in computer vision or speech processing without
    meaningful information retrieval or NLP experience. Five or more years on
    closed-source proprietary systems with no external validation through papers,
    talks, or open-source contributions.

  # === Instruction strings (MUST match precompute.py exactly) ===
  instructions:
    q1:
      retrieval: "Represent the candidate's production experience with embeddings, semantic search, dense retrieval, and recommendation systems:"
      infra: "Represent the candidate's hands-on experience operating vector databases, search infrastructure, and ML systems at scale in production:"
      eval: "Represent the candidate's experience designing and running evaluation frameworks, ranking metrics, A/B tests, and offline-to-online validation for ML systems:"
    q2:
      retrieval: "Represent the candidate's career experience with search, ranking, recommendation, and retrieval systems they personally shipped to production:"
      infra: "Represent the candidate's background at product companies building and scaling ML infrastructure for real users:"
      eval: "Represent the candidate's hands-on experience with evaluation, iteration, and improvement of production ML systems based on real user feedback:"
    q3:
      retrieval: "Represent the candidate's experience or lack thereof with search, retrieval, and ranking in production:"
      infra: "Represent the candidate's background in terms of company type, deployment context, and engineering environment:"
      eval: "Represent the candidate's approach to building and validating ML systems, including signs of research-only or framework-only orientation:"

  # === Subspace weights ===
  subspace_weights_q1:
    retrieval: 0.35
    infra: 0.45
    eval: 0.20
  subspace_weights_q2:
    retrieval: 0.33
    infra: 0.34
    eval: 0.33
  subspace_weights_q3:
    retrieval: 0.33
    infra: 0.34
    eval: 0.33

  # === Retrieval counts ===
  per_query_k_dense: 1000     # top-k per FAISS query (Q1, Q2)
  per_query_k_skill: 500      # top-k for skill score track (L3)
                              # replaces per_query_k_sparse (BM25) — removed

  # === Miss-penalty ranks ===
  miss_penalty_dense: 1001    # = per_query_k_dense + 1
  miss_penalty_skill: 501     # = per_query_k_skill + 1

  # === RRF ===
  rrf_k: 60                   # standard smoothing constant

  # === Fusion weights ===
  alpha_neg: 0.5              # Q3 anti-pattern penalty weight
  beta_cluster: 0.0           # cluster proximity boost (OFF by default)

  # === Adaptive top-k ===
  z_threshold: 1.5
  min_k: 300
  max_k: 600
```

**Removed config keys (do not add back):**
- ~~`q4_tokens`~~ — BM25 jargon token list, no longer needed
- ~~`per_query_k_sparse`~~ — replaced by `per_query_k_skill`

---

## INPUT / OUTPUT CONTRACTS

### Input Contract

`stage2_gated.parquet` must contain at minimum: `candidate_id`, `cluster_id`,
`dist_to_centroid`, `cluster_rank`, and all Stage 2 flag and feature columns. If any
required column is missing, raise an error before any retrieval work.

`candidate_features.parquet` must contain `candidate_id` and `skill_weighted_score`.
If `skill_weighted_score` is absent, raise an error with the message specified in Step 2.

### Output Contract

`stage3_retrieved.parquet` must contain:
- All columns from `stage2_gated.parquet` (passthrough)
- `q1_score` (float, nullable), `q1_rank` (int)
- `q2_score` (float, nullable), `q2_rank` (int)
- `skill_score` (float, nullable), `skill_rank` (int)
- `q3_neg_sim` (float, non-null)
- `rrf_score` (float, non-null)
- `fused_score` (float, non-null)
- `stage3_rank` (int, 1 to output row count)

Row count: between `min_k` (300) and `max_k` (600). Sorted by ascending `stage3_rank`.

**Removed output columns:**
- ~~`bm25_score`~~ — no longer produced
- ~~`bm25_rank`~~ — no longer produced

---

## OPERATOR VALIDATION CHECKLIST

Run after Stage 3, before running Stage 4.

1. Output row count is between 300 and 600.
2. Top 10 candidates by `fused_score` look like senior ML/IR engineers. No marketing
   managers, no obvious honeypots, no pure researchers.
3. At least one candidate in the top 20 has a low `q1_score` / `q2_score` but a high
   `skill_score`. This confirms the skill track is surfacing candidates who write sparse
   prose but have deep structured skill data.
4. At least one candidate in the top 20 has a low `skill_score` but high `q1_score` /
   `q2_score`. This confirms the dense tracks are rescuing candidates whose skills section
   doesn't encode richly but whose prose does (Tier-5 plain-language candidates).
5. At least one candidate has a high `q3_neg_sim` but low `fused_score` — confirming
   the anti-pattern penalty is firing.
6. `fused_score` is monotonically consistent with `stage3_rank`.
7. No duplicate `candidate_id` in output.
8. Spot-check 5 random candidates — verify their `candidate_id` exists in
   `candidates.jsonl`.
9. Confirm `skill_weighted_score` column exists and is non-null for the candidates
   appearing in L3 (those with `skill_rank` <= 500).
