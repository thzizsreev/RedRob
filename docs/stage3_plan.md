# STAGE 3 — MULTI-QUERY HYBRID RETRIEVAL
## Implementation Plan for AI Coding Agent

---

## OVERVIEW

**Script name:** `stages/stage3_retrieve.py`
**Runs after:** Stage 2 (`stage2_gate.py`) has completed and its output has been manually inspected.
**Runs before:** Stage 4 (`stage4_crossenc.py`).

**Input artifacts (all read-only):**
- `artifacts/runtime/stage2_gated.parquet` — candidates that survived Stage 2, ~3,000–4,000 rows
- `artifacts/precomputed/dense_vectors.npy` — numpy array of shape `(N_total, D)` where N_total is all original candidates and D is the full concatenated Instructor embedding dimension. Each row corresponds to one candidate's three-block concatenated vector.
- `artifacts/precomputed/faiss_index.bin` — FAISS flat index built with Inner Product metric over `dense_vectors.npy`
- `artifacts/precomputed/bm25_index.pkl` — serialized BM25Okapi index over candidate jargon text blocks, one document per candidate
- `artifacts/precomputed/id_row_map.parquet` — two columns: `candidate_id` (string) and `dense_row_idx` (int). Maps each candidate_id to its row number in `dense_vectors.npy`
- `artifacts/precomputed/candidate_features.parquet` — full Track B tabular feature store; Stage 3 only needs `candidate_id` and the `jargon_text` field from this file

**Output artifact:**
- `artifacts/runtime/stage3_retrieved.parquet` — ~300–600 rows, containing all Stage 2 columns plus new score columns added by Stage 3

**What this stage does in one sentence:**
Decomposes the Job Description into four distinct query signals, retrieves candidates from the FAISS index and BM25 index independently per query, fuses the ranked lists using Reciprocal Rank Fusion, applies an anti-pattern penalty, and cuts the result to the top ~300–600 candidates using an adaptive score threshold.

---

## CRITICAL DESIGN CONSTRAINTS

Read these before writing a single line of code.

1. **No PyTorch online.** The Instructor model must be loaded via `onnxruntime`, not via the `InstructorEmbedding` or `sentence_transformers` Python library. The ONNX export of the model must already exist at `models/instructor.onnx`. `onnxruntime` is the only inference runtime allowed online.

2. **FAISS must use Inner Product, not Cosine Similarity.** The index was built with `faiss.METRIC_INNER_PRODUCT`. The query vectors must NOT be L2-normalized before being passed to FAISS. Normalizing the full 768-d vector would destroy the orthogonal subspace boundaries. Per-subspace normalization (if any) was done offline during precompute and must not be re-applied here.

3. **The same Instructor instruction strings used in precompute must be used here.** The three instruction strings for the three subspace blocks are stored in `config.yaml` under `stage3.instructions`. The coding agent must read them from config, never hardcode them. If the instruction strings differ even slightly between precompute and Stage 3, the dot products are mathematically meaningless.

4. **Stage 2 survivors only.** The FAISS index contains all N_total candidates. Stage 3 must never retrieve a candidate that was removed by Stage 2. This is enforced via a FAISS `IDSelector` bitmask built from the Stage 2 survivor indices. Candidates not in Stage 2's output must be physically excluded from the search, not just filtered post-hoc.

5. **All per-query scores must be written as separate columns in the output.** Do not collapse them into a single score before writing. Stage 5's LightGBM model consumes each score individually as a separate feature.

6. **No threshold or count is hardcoded.** Every numeric parameter (k values, RRF k, alpha, beta, z-score, min_k, max_k) is read from `config.yaml` under the `stage3:` namespace.

7. **The script must expose both a CLI entry point and a `run(input_path, output_path, config)` function.** The CLI is for manual runs. The `run()` function is called by the final `rank.py` wrapper for hackathon submission reproduction.

---

## TECH STACK

| Library | Purpose |
|---|---|
| `polars` | All DataFrame operations — reading/writing parquet, joins, filtering, score assembly |
| `numpy` | Dense vector operations — loading `.npy`, dot products, score arrays |
| `faiss-cpu` | Dense vector search via Inner Product over the precomputed FAISS index |
| `rank_bm25` | BM25 sparse retrieval via the precomputed BM25Okapi index |
| `onnxruntime` | Inference for the Instructor ONNX model to encode query vectors |
| `pyyaml` | Reading config.yaml |
| `pickle` | Deserializing `bm25_index.pkl` |
| `argparse` | CLI argument parsing |

No other libraries. Do not import `torch`, `transformers`, `sentence_transformers`, or `InstructorEmbedding`.

---

## DIRECTORY AND FILE STRUCTURE

The script lives at `stages/stage3_retrieve.py`. It does not create any subdirectories. All outputs go to `artifacts/runtime/`. All inputs are read from `artifacts/precomputed/` or `artifacts/runtime/` as specified above.

A separate `config.yaml` at the project root contains all Stage 3 parameters under a `stage3:` key. The script reads only the `stage3:` block. The full structure of the `stage3:` config block is specified in the CONFIG PARAMETERS section at the bottom of this document.

---

## HOW INSTRUCTOR EMBEDDINGS WORK IN THIS SYSTEM

This is background context the coding agent must understand before implementing the query encoding step.

During precompute, each candidate's profile text was encoded three times using the Instructor model — once per subspace — each time with a different instruction string. The three resulting vectors (each of dimension `d`, where `d` is the Instructor model's output dimension) were concatenated in order: `[retrieval_vec | infra_vec | eval_vec]` to produce a single `(3*d,)` vector. This is stored as one row in `dense_vectors.npy`.

The Instructor model takes `(instruction, text)` as input and produces an embedding that is conditioned on the instruction. The same model with a different instruction produces a semantically different vector from the same text. This is what enables subspace separation without multiple models.

At query time (Stage 3), we encode the query text the same way: three times, once per subspace instruction, then concatenate. The resulting query vector lives in the same space as the candidate vectors and can be compared via dot product.

The subspace weights are applied after encoding, before FAISS search, by scalar-multiplying each 256-d block of the query vector by its weight. This amplifies or dampens the contribution of each subspace to the dot product score during retrieval.

---

## THE FOUR QUERIES

Stage 3 decomposes the JD into four distinct query signals. Three are dense (Instructor-encoded), one is sparse (BM25 token list). Each query surfaces a different type of relevance.

### Q1 — Technical Requirements Query (dense, positive)

**Purpose:** Find candidates who have hands-on production experience with the specific technical systems this role requires — embeddings-based retrieval, vector databases, and evaluation frameworks. This is the "hard skills" query. It finds Tier-1 through Tier-3 candidates and also Tier-5 plain-language candidates who describe these systems without using ML vocabulary.

**Text content for this query** (stored in `config.yaml` under `stage3.q1_text`):
```
Production experience building and operating embeddings-based retrieval systems deployed
to real users. Handled embedding drift, index refresh cycles, and retrieval quality
regression in production environments. Built or operated vector database infrastructure
using systems such as Pinecone, Qdrant, Weaviate, Milvus, OpenSearch, Elasticsearch,
or FAISS at meaningful scale. Designed and maintained evaluation frameworks for ranking
systems including NDCG, MRR, MAP, offline-to-online correlation, and A/B testing of
ranking changes. Strong Python engineering skills in production ML systems.
```

**Instruction strings** (one per subspace, read from config):
- Retrieval subspace instruction: `"Represent the candidate's production experience with embeddings, semantic search, dense retrieval, and recommendation systems:"`
- Infra subspace instruction: `"Represent the candidate's hands-on experience operating vector databases, search infrastructure, and ML systems at scale in production:"`
- Eval subspace instruction: `"Represent the candidate's experience designing and running evaluation frameworks, ranking metrics, A/B tests, and offline-to-online validation for ML systems:"`

**Subspace weights for Q1** (from config `stage3.subspace_weights_q1`):
- Retrieval: 0.35
- Infra: 0.45
- Eval: 0.20

Infra is weighted highest because the JD explicitly prioritizes operational production experience over knowing the theory.

**How to encode Q1:** encode the Q1 text three times, once per subspace instruction, using the Instructor ONNX model. Each encoding produces a `(d,)` vector. Concatenate in order `[retrieval | infra | eval]` to get `(3d,)`. Apply subspace weights by scalar-multiplying each `d`-length block.

**Retrieval:** send the weighted Q1 vector to FAISS. Retrieve top `per_query_k_dense` candidates (default 1000). This produces ranked list `L1` — a list of `(candidate_id, q1_score)` pairs sorted by descending dot product similarity.

---

### Q2 — Career Shape Query (dense, positive)

**Purpose:** Find candidates whose career trajectory matches the "ideal candidate" profile described in the JD — product company background, shipped end-to-end systems, opinionated about retrieval and evaluation. This query specifically rescues Tier-5 plain-language candidates who do not use ML jargon but whose career story matches the profile. It is phrased as a candidate self-description, not as a job posting, because candidate profiles and JD text sit in slightly different regions of embedding space.

**Text content for this query** (stored in `config.yaml` under `stage3.q2_text`):
```
Six to eight years of total experience, four to five of which were spent in applied ML
and AI engineering roles at product companies rather than consulting firms or research
labs. Shipped at least one end-to-end ranking, search, or recommendation system to real
users at meaningful scale. Have strong opinions about hybrid versus dense retrieval,
offline versus online evaluation, and when to fine-tune a model versus prompt an existing
one, all grounded in systems I actually built and ran in production. Prefer shipping a
working system and iterating over designing a perfect system that never ships. Comfortable
owning an entire intelligence layer of a product, not just a model component.
```

**Instruction strings** (one per subspace, read from config):
- Retrieval subspace instruction: `"Represent the candidate's career experience with search, ranking, recommendation, and retrieval systems they personally shipped to production:"`
- Infra subspace instruction: `"Represent the candidate's background at product companies building and scaling ML infrastructure for real users:"`
- Eval subspace instruction: `"Represent the candidate's hands-on experience with evaluation, iteration, and improvement of production ML systems based on real user feedback:"`

**Subspace weights for Q2** (from config `stage3.subspace_weights_q2`):
- Retrieval: 0.33
- Infra: 0.34
- Eval: 0.33

Weights are approximately equal because Q2 is looking for holistic career shape, not a specific technical skill.

**How to encode Q2:** same procedure as Q1 — three encodings with three instructions, concatenate, apply weights. Results in a `(3d,)` weighted vector.

**Retrieval:** send to FAISS. Retrieve top `per_query_k_dense` candidates (default 1000). Produces ranked list `L2` — `(candidate_id, q2_score)` pairs.

---

### Q3 — Anti-Pattern Query (dense, NEGATIVE — not used for retrieval)

**Purpose:** Identify candidates who closely match the disqualifier profiles described in the JD. This vector is NOT sent to FAISS for retrieval. Instead, after the candidate union is assembled from L1, L2, and L4, we compute each candidate's dot product similarity to Q3 and use it as a penalty term subtracted from the fused score. Candidates who look like keyword stuffers, LangChain tutorial builders, consulting-only engineers, or pure researchers will have high Q3 similarity and will be down-ranked accordingly.

**Text content for this query** (stored in `config.yaml` under `stage3.q3_text`):
```
Career shows a pattern of switching companies every one to two years specifically to gain
a more senior title. GitHub or portfolio consists primarily of LangChain tutorials,
framework demos, and beginner AI projects with no evidence of production systems thinking.
Entire career spent at large consulting firms such as TCS, Infosys, Wipro, Accenture,
Cognizant, or Capgemini with no product company experience at any point. Background is
primarily in academic research or pure research roles with no production deployments of
any ML system to real users. Primary expertise is in computer vision, speech processing,
or robotics with no meaningful experience in information retrieval, NLP, or ranking systems.
Has spent five or more years working entirely on closed-source proprietary internal systems
with no external validation through papers, talks, blog posts, or open-source contributions.
```

**Instruction strings** (one per subspace, read from config):
- Retrieval subspace instruction: `"Represent the candidate's experience or lack thereof with search, retrieval, and ranking in production:"`
- Infra subspace instruction: `"Represent the candidate's background in terms of company type, deployment context, and engineering environment:"`
- Eval subspace instruction: `"Represent the candidate's approach to building and validating ML systems, including signs of research-only or framework-only orientation:"`

**Subspace weights for Q3:** use flat weights (0.33 / 0.34 / 0.33). No amplification needed — we want a balanced representation of the anti-pattern signal across all subspaces.

**How to use Q3:** after building the candidate union (see Fusion section), extract the dense vectors for all candidates in the union from `dense_vectors.npy` using their `dense_row_idx` values. Compute the dot product of each candidate's vector against the Q3 vector using numpy matrix multiplication. This produces a `(union_size,)` array of Q3 similarity scores. Attach this as the `q3_neg_sim` column to the union DataFrame. It will be subtracted in the fusion formula.

**Important nuance:** the JD penalizes framework *enthusiasts*, not framework *users*. The Q3 text reflects this — it describes someone whose entire portfolio is tutorials, not someone who has used LangChain as one tool among many. The `alpha_neg` weight (default 0.5) should be tuned to penalize without catastrophically punishing candidates who legitimately mention these tools. If `alpha_neg` is too high, candidates who list "LangChain" as one of twenty skills get unfairly punished.

---

### Q4 — Exact Jargon Query (sparse BM25, positive)

**Purpose:** Catch candidates who use the exact technical vocabulary of the JD. Dense retrieval excels at semantic matching but sometimes undersells candidates who happen to use precise technical terms. BM25 is a complementary signal — it rewards exact token matches with TF-IDF-style weighting. Together, dense + sparse = hybrid retrieval.

**Token list** (stored in `config.yaml` under `stage3.q4_tokens` as a list):
```
FAISS, HNSW, Qdrant, Weaviate, Milvus, Pinecone, OpenSearch, Elasticsearch,
NDCG, MRR, MAP, cross-encoder, bi-encoder, embedding drift, hybrid search,
learning-to-rank, sentence-transformers, BGE, E5, RAG, A/B test,
retrieval-augmented, reranking, vector database, dense retrieval,
sparse retrieval, BM25, semantic search, embedding model, fine-tuning,
LoRA, QLoRA, PEFT, lambdarank, XGBoost, LightGBM, inference optimization
```

**How to use Q4:** the BM25 index was built over candidate jargon text blocks offline. Call `bm25_index.get_scores(q4_tokens)` which returns a score array over the full corpus. Filter this array to only the candidates in Stage 2's output (using the `dense_row_idx` mapping). Sort descending. Take the top `per_query_k_sparse` (default 500). This produces ranked list `L4` — `(candidate_id, bm25_score)` pairs.

---

## STEP-BY-STEP PROCEDURE

### PROCEDURE STEP 1 — Parse Arguments and Load Config

The script accepts three CLI arguments: `--input` (path to `stage2_gated.parquet`), `--output` (path to write `stage3_retrieved.parquet`), and `--config` (path to `config.yaml`, default `config.yaml`).

Load the YAML config file and extract the full `stage3:` block. Every parameter reference below refers to a key in this block. The script never uses a numeric literal for any threshold or count.

---

### PROCEDURE STEP 2 — Load All Assets Into Memory

Load the following in order. Fail loudly with a descriptive error if any file is missing:

1. **Stage 2 DataFrame**: load `stage2_gated.parquet` into a Polars DataFrame. Verify it has a `candidate_id` column. Log the row count.

2. **Dense vectors**: load `dense_vectors.npy` into a numpy float32 array. Log its shape `(N_total, D)`. The value of `D` (the full concatenated embedding dimension) is derived from this shape, not hardcoded. The per-subspace dimension is `D // 3`.

3. **FAISS index**: load `faiss_index.bin` using `faiss.read_index`. Verify it is a flat index with Inner Product metric. Log the total number of vectors in the index — it should equal `N_total`.

4. **BM25 index**: deserialize `bm25_index.pkl` using Python's `pickle` module. This is a `BM25Okapi` instance from the `rank_bm25` library.

5. **ID row map**: load `id_row_map.parquet` into a Polars DataFrame with columns `candidate_id` and `dense_row_idx`. This is the bridge between string candidate IDs and integer numpy/FAISS row indices.

6. **Instructor ONNX session**: create an `onnxruntime.InferenceSession` from `models/instructor.onnx`. Log the input and output node names — these are needed to call inference correctly.

---

### PROCEDURE STEP 3 — Build the FAISS IDSelector

The FAISS index contains all N_total candidates, but Stage 3 must only retrieve from the ~3,000–4,000 that passed Stage 2.

Join the Stage 2 DataFrame with the ID row map on `candidate_id` to get the `dense_row_idx` for every Stage 2 survivor. Extract these indices as a numpy int64 array. Sort them.

Create a `faiss.IDSelectorBatch` from this array. This selector will be passed to every FAISS search call, physically restricting the search to Stage 2 survivors. FAISS will not evaluate candidates outside this selector regardless of their vector similarity.

Store the selector as a module-level variable — it is reused for both Q1 and Q2 searches.

---

### PROCEDURE STEP 4 — Encode Q1, Q2, Q3 Using the Instructor ONNX Model

This is the most critical step. The encoding must be identical in structure to what was done in precompute.

For each query (Q1, Q2, Q3), the procedure is:

**4a. Retrieve the instruction strings and query text from config.** There are three instruction strings per query (one per subspace: retrieval, infra, eval). These are stored in config under `stage3.instructions.q1`, `stage3.instructions.q2`, `stage3.instructions.q3` respectively, each containing subkeys `retrieval`, `infra`, `eval`. The query texts are stored under `stage3.q1_text`, `stage3.q2_text`, `stage3.q3_text`.

**4b. For each of the three subspaces (retrieval, infra, eval):**
- Construct the model input by combining the instruction string with the query text. The exact format must match what was used in precompute — check precompute.py to confirm the format. Typically: the Instructor model takes a list of `[instruction, text]` pairs.
- Run inference via `onnxruntime` session. Pass the tokenized input to the session's `run()` method using the input node name identified in Step 2.
- The output is a vector of dimension `d` (where `d = D // 3`). This is one subspace vector.

**4c. Concatenate the three subspace vectors** in the fixed order `[retrieval_vec | infra_vec | eval_vec]` to produce the full `(D,)` query vector.

**4d. Apply subspace weights** for Q1 and Q2 (Q3 uses flat weights):
- Slice the vector into three equal blocks of size `d`.
- Multiply the retrieval block by `subspace_weights.retrieval`.
- Multiply the infra block by `subspace_weights.infra`.
- Multiply the eval block by `subspace_weights.eval`.
- Reassemble the three blocks back into one `(D,)` vector.

**4e. Reshape to `(1, D)`** before passing to FAISS (FAISS expects a 2D array even for single queries).

Do NOT L2-normalize the vectors at any point.

After this step you have three vectors: `q1_vec` of shape `(1, D)`, `q2_vec` of shape `(1, D)`, `q3_vec` of shape `(D,)` (Q3 stays 1D since it is used for numpy dot product, not FAISS).

---

### PROCEDURE STEP 5 — FAISS Retrieval for Q1 and Q2

For each of Q1 and Q2:

Create a `faiss.SearchParameters` object and attach the `IDSelectorBatch` from Step 3 to it. This enforces that only Stage 2 survivors are searched.

Call `faiss_index.search(query_vec, k, params=search_params)` where `k = per_query_k_dense` from config (default 1000). The search returns two arrays: `scores` of shape `(1, k)` and `indices` of shape `(1, k)`. Flatten both to 1D.

The `indices` array contains FAISS row indices (which are `dense_row_idx` values). Map them back to `candidate_id`s using the ID row map. Drop any index that is -1 (FAISS returns -1 for unfilled slots if fewer than k valid candidates exist).

Build a Polars DataFrame `L1` (for Q1) or `L2` (for Q2) with columns `candidate_id` and `q1_score` (or `q2_score`). The score is the raw dot product value returned by FAISS. Do not normalize or transform it. Sort by descending score and assign a rank column `q1_rank` (or `q2_rank`) starting from 1.

---

### PROCEDURE STEP 6 — BM25 Retrieval for Q4

Load the Q4 token list from config under `stage3.q4_tokens`. This is a list of strings.

Call `bm25_index.get_scores(q4_tokens)` on the loaded BM25 index. This returns a numpy array of shape `(N_bm25_corpus,)` where each element is the BM25 score for one candidate. The BM25 corpus was built over all original candidates in the same order as they appear in the BM25 index — the mapping between BM25 corpus positions and candidate IDs was stored in `id_row_map.parquet` as `bm25_corpus_idx` column (verify this column name against precompute).

Filter the BM25 scores to only include candidates that are in the Stage 2 survivor set (using the `dense_row_idx` / `bm25_corpus_idx` mapping).

Sort by descending BM25 score. Take the top `per_query_k_sparse` candidates (default 500).

Build a Polars DataFrame `L4` with columns `candidate_id` and `bm25_score`. Assign a rank column `bm25_rank` starting from 1.

---

### PROCEDURE STEP 7 — Build the Candidate Union

Take the union of all candidate IDs appearing in `L1`, `L2`, and `L4`. A candidate may appear in one, two, or all three lists. The union typically contains ~1,500–2,000 unique candidates.

Build a master union DataFrame with one row per unique `candidate_id`. Left-join `L1`, `L2`, and `L4` onto this master DataFrame using `candidate_id` as the join key. For candidates missing from a given list, fill their score and rank columns with null.

For the RRF computation in the next step, missing ranks must be replaced with a miss-penalty rank. The penalty rank is `per_query_k_dense + 1` for Q1 and Q2 (i.e., 1001 if k=1000), and `per_query_k_sparse + 1` for Q4 (i.e., 501 if k=500). Apply these fills now: replace null rank values with their respective penalty ranks.

Also attach the `dense_row_idx` to each candidate in the union by joining with the ID row map. This is needed in the next step to look up Q3 similarity.

---

### PROCEDURE STEP 8 — Compute RRF Score and Q3 Penalty

**8a. Reciprocal Rank Fusion (RRF):**

RRF is the standard rank-based fusion formula from information retrieval. For each candidate in the union, their RRF score is:

```
RRF_score = 1 / (rrf_k + rank_in_L1)
           + 1 / (rrf_k + rank_in_L2)
           + 1 / (rrf_k + rank_in_L4)
```

Where `rrf_k` is a smoothing constant (default 60, read from config). Candidates that appeared in all three lists get contributions from all three terms. Candidates that appeared in only one list still get that one term; the other two terms use the miss-penalty rank values filled in Step 7, producing a small but non-zero contribution.

Compute this as a Polars expression over the union DataFrame. Store the result as column `rrf_score`.

**8b. Q3 Anti-Pattern Similarity:**

Extract the `dense_row_idx` values for all candidates in the union as a numpy integer array. Use these indices to gather the corresponding rows from `dense_vectors.npy`:

```
union_vecs = dense_vectors[union_dense_row_idxs]   # shape: (union_size, D)
```

Compute the dot product of each candidate's full vector against `q3_vec`:

```
q3_sims = union_vecs @ q3_vec   # shape: (union_size,)
```

Attach this as column `q3_neg_sim` to the union DataFrame. Higher values mean the candidate is more similar to the anti-pattern description — they will be penalized more.

**8c. Compute the Final Fused Score:**

```
fused_score = rrf_score
            - alpha_neg * q3_neg_sim
            + beta_cluster * (1 / (1 + dist_to_centroid))
```

Where:
- `alpha_neg` is read from config (default 0.5). Controls how heavily the anti-pattern penalty is applied.
- `beta_cluster` is read from config (default 0.0, meaning this term is OFF by default). If enabled, it gives a small boost to candidates who are close to their Stage 1 cluster centroid, based on the assumption that cluster centroids are themselves JD-relevant. Only enable this if you have validated that the Stage 1 clusters correlate with JD relevance.
- `dist_to_centroid` is the column carried forward from Stage 1 output.

Store the result as column `fused_score`.

---

### PROCEDURE STEP 9 — Adaptive Top-K Cutoff

The cutoff is adaptive — it depends on the actual distribution of `fused_score` values in the union, not a hardcoded number. This avoids including junk candidates just to hit a quota, and avoids excluding near-tie candidates due to an arbitrary cliff.

**9a. Compute score distribution statistics:**
Compute the mean (`mu`) and standard deviation (`sigma`) of the `fused_score` column over all candidates in the union.

**9b. Compute the threshold:**
```
threshold = mu - z_threshold * sigma
```
Where `z_threshold` is read from config (default 1.5). Candidates more than 1.5 standard deviations below the mean are statistically unlikely to be relevant.

**9c. Apply min/max bounds:**
Filter to candidates with `fused_score >= threshold`. Count them. If the count exceeds `max_k` (default 600), take only the top `max_k` by descending `fused_score`. If the count is below `min_k` (default 300), override the threshold and take the top `min_k` by descending `fused_score` regardless of threshold.

`min_k` exists because Stage 4's cross-encoder needs enough candidates to do meaningful reranking. `max_k` exists because Stage 4's cross-encoder inference time is linear in the number of candidates.

**9d. Tiebreak:**
Within any group of candidates with identical `fused_score`, sort by `candidate_id` ascending. This is deterministic and matches the hackathon spec's tiebreaking requirement.

**9e. Assign `stage3_rank`:**
After sorting, assign a `stage3_rank` integer column starting from 1 for the best candidate. This is a provisional rank — it will be superseded by Stage 4 and Stage 5 — but it is useful for debugging.

---

### PROCEDURE STEP 10 — Assemble Output DataFrame and Write

The output DataFrame must contain every column from `stage2_gated.parquet` plus the new Stage 3 columns. Do not drop any Stage 2 column. Stage 5 needs all of them.

**New columns added by Stage 3:**

| Column | Type | Description |
|---|---|---|
| `q1_score` | float | Raw dot product similarity to Q1 vector. Null if candidate was not in L1. |
| `q1_rank` | int | Rank in L1 (1 = most similar). Miss-penalty value if not in L1. |
| `q2_score` | float | Raw dot product similarity to Q2 vector. Null if not in L2. |
| `q2_rank` | int | Rank in L2. Miss-penalty value if not in L2. |
| `bm25_score` | float | BM25 score for Q4 token list. Null if not in L4. |
| `bm25_rank` | int | Rank in L4. Miss-penalty value if not in L4. |
| `q3_neg_sim` | float | Dot product similarity to Q3 anti-pattern vector. Higher = more penalized. |
| `rrf_score` | float | Raw RRF fusion score before Q3 penalty and cluster boost. |
| `fused_score` | float | Final Stage 3 score: RRF − alpha*Q3 + beta*cluster. |
| `stage3_rank` | int | Provisional rank by fused_score. 1 = best. |

Join the Stage 2 DataFrame onto the final top-k candidates using `candidate_id` as the join key (left join from top-k candidates onto Stage 2 to preserve the top-k row ordering). Attach all Stage 3 score columns.

Write the final DataFrame to `artifacts/runtime/stage3_retrieved.parquet` using Polars `write_parquet`. Verify the output row count is between `min_k` and `max_k` before writing. If not, raise an error with a clear message explaining which bound was violated and what the actual count was.

---

### PROCEDURE STEP 11 — Logging and Validation Outputs

The script must produce a human-readable summary printed to stdout after writing the parquet. This is what the operator reads before deciding to run Stage 4.

**Print the following:**

1. Input row count (Stage 2 survivors).
2. Union size (total unique candidates across L1 + L2 + L4).
3. L1 size, L2 size, L4 size — and how many candidates appeared in all three (overlap count).
4. Score distribution: min, max, mean, std of `fused_score` over the union.
5. Threshold value used for the adaptive cutoff.
6. Final output row count.
7. Top 10 candidates by `fused_score`: print `candidate_id`, `fused_score`, `q1_score`, `q2_score`, `bm25_score`, `q3_neg_sim`, `stage3_rank` for each. This is what the operator inspects manually to check retrieval quality.
8. Bottom 5 candidates in the output (lowest `fused_score` in the kept set) — useful for checking if the threshold is too permissive.

Also write a `artifacts/runtime/stage3_score_distribution.csv` file with all union candidates, their scores, and their ranks (both those kept and those cut). This is for offline analysis and threshold tuning. It does NOT go to Stage 4 — only the parquet does.

---

## CONFIG PARAMETERS

All Stage 3 parameters live under `stage3:` in `config.yaml`. Here is the full structure:

```yaml
stage3:

  # === ONNX model path ===
  onnx_model_path: models/instructor.onnx

  # === Query texts (the actual content of each query) ===
  q1_text: |
    Production experience building and operating embeddings-based retrieval systems
    deployed to real users. Handled embedding drift, index refresh cycles, and
    retrieval quality regression in production environments. Built or operated vector
    database infrastructure using systems such as Pinecone, Qdrant, Weaviate, Milvus,
    OpenSearch, Elasticsearch, or FAISS at meaningful scale. Designed and maintained
    evaluation frameworks for ranking systems including NDCG, MRR, MAP, A/B testing.

  q2_text: |
    Six to eight years of total experience, four to five of which were in applied ML
    at product companies. Shipped end-to-end ranking, search, or recommendation system
    to real users at scale. Strong opinions on hybrid versus dense retrieval and
    offline versus online evaluation, grounded in systems actually built and run in
    production. Prefer shipping a working system and iterating over designing the
    perfect system.

  q3_text: |
    Career shows switching companies every one to two years for senior titles. GitHub
    consists primarily of LangChain tutorials and framework demos. Entire career at
    consulting firms such as TCS, Infosys, Wipro, Accenture with no product company
    experience. Background is purely academic research with no production deployments.
    Primary expertise in computer vision or speech without NLP or information retrieval
    experience. Five or more years on closed-source proprietary systems with no
    external validation.

  # === Instruction strings per query per subspace ===
  # MUST match precompute.py exactly
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

  # === BM25 jargon token list ===
  q4_tokens:
    - FAISS
    - HNSW
    - Qdrant
    - Weaviate
    - Milvus
    - Pinecone
    - OpenSearch
    - Elasticsearch
    - NDCG
    - MRR
    - MAP
    - cross-encoder
    - bi-encoder
    - embedding drift
    - hybrid search
    - learning-to-rank
    - sentence-transformers
    - BGE
    - E5
    - RAG
    - retrieval-augmented
    - reranking
    - vector database
    - dense retrieval
    - BM25
    - semantic search
    - fine-tuning
    - LoRA
    - QLoRA
    - lambdarank

  # === Subspace weights ===
  subspace_weights_q1:
    retrieval: 0.35
    infra: 0.45
    eval: 0.20
  subspace_weights_q2:
    retrieval: 0.33
    infra: 0.34
    eval: 0.33

  # === Retrieval counts ===
  per_query_k_dense: 1000    # top-k per FAISS query (Q1, Q2)
  per_query_k_sparse: 500    # top-k for BM25 (Q4)

  # === RRF fusion ===
  rrf_k: 60                  # RRF smoothing constant, standard value

  # === Fusion penalty weights ===
  alpha_neg: 0.5             # weight of Q3 anti-pattern penalty on fused score
  beta_cluster: 0.0          # weight of cluster centroid proximity boost (0.0 = OFF)

  # === Adaptive top-k cutoff ===
  z_threshold: 1.5           # candidates below mean - z*std are cut
  min_k: 300                 # never output fewer than this many candidates
  max_k: 600                 # never output more than this many candidates
```

---

## INPUT / OUTPUT CONTRACTS

### Input Contract

`stage2_gated.parquet` must contain at minimum: `candidate_id` (string), `cluster_id` (int), `dist_to_centroid` (float), `cluster_rank` (int, nullable), and all Stage 2 flag columns (`exp_band`, `in_sweet_spot`, `title_family`, `skill_kw_density`, `title_ambiguous`, `stale_profile`, `low_responder`, `not_open`). If any of these are missing, the script must raise an error and halt before doing any retrieval work.

### Output Contract

`stage3_retrieved.parquet` must contain:
- All columns from `stage2_gated.parquet` (passthrough — do not drop any)
- `q1_score` (float, nullable), `q1_rank` (int)
- `q2_score` (float, nullable), `q2_rank` (int)
- `bm25_score` (float, nullable), `bm25_rank` (int)
- `q3_neg_sim` (float, non-null for all rows)
- `rrf_score` (float, non-null for all rows)
- `fused_score` (float, non-null for all rows)
- `stage3_rank` (int, 1 to output_row_count)

Row count: between `min_k` (300) and `max_k` (600). Sorted by ascending `stage3_rank`.

---

## OPERATOR VALIDATION CHECKLIST (run after Stage 3 before running Stage 4)

1. Output row count is between 300 and 600. If outside this range, check threshold math.
2. Top 10 candidates by `fused_score` look like senior ML/IR engineers — not marketing managers, not pure researchers, not obvious LangChain-tutorial profiles.
3. At least one candidate in the top 20 has a low `bm25_score` but a high `q2_score`. This indicates Tier-5 plain-language candidates are being found by the career-shape query even without jargon.
4. At least one candidate in the output has a high `q3_neg_sim` (close to the anti-pattern vector) but a low `fused_score` — confirming the penalty is working.
5. `fused_score` is monotonically consistent with `stage3_rank` (rank 1 has the highest score, rank N has the lowest). If not, there is a sorting bug.
6. No candidate_id appears more than once in the output.
7. Spot-check 5 random candidates — verify their `candidate_id` exists in the original `candidates.jsonl`.
