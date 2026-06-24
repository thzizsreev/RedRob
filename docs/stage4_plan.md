# STAGE 4 — CROSS-ENCODER RERANKING
## Implementation Plan for AI Coding Agent

---

## OVERVIEW

**Script name:** `tracks/instructor/stage4/run.py` (CLI) with logic in `tracks/instructor/stage4/`
**Runs after:** Stage 3 (`stage3/run.py`) has completed and its output has been manually inspected.
**Runs before:** Stage 5 (`stage5_lgbm.py`).

**Input artifacts (all read-only):**
- `artifacts/runtime/stage3_retrieved.parquet` — ~300–600 candidates with all Stage 2 + Stage 3 columns
- `artifacts/precomputed/candidate_features.parquet` — Track B tabular store; Stage 4 needs `candidate_id` and candidate text fields (see Pair Construction)
- `data/candidates.jsonl` (or `.jsonl.gz`) — fallback for building candidate passage text if Track B summaries are not yet populated
- `models/cross_encoder/` — offline-exported ONNX model + tokenizer (see Model Selection)

**Output artifacts:**
- `artifacts/runtime/stage4_reranked.parquet` — ~300 rows (configurable), all Stage 3 columns plus new Stage 4 score columns
- `artifacts/runtime/stage4_rank_delta.csv` — candidates whose rank moved by more than a threshold (debugging)
- `artifacts/runtime/stage4_summary.json` — counts, latency, model id, score distribution

**What this stage does in one sentence:**
Scores every Stage 3 survivor as a (JD, candidate) pair using a cross-encoder ONNX model on CPU, re-sorts by pairwise relevance, and cuts to the top ~300 candidates for Stage 5.

**Expected reduction:** ~300–600 → ~300 (`stage4.keep_n`, default 300).

---

## CRITICAL DESIGN CONSTRAINTS

Read these before writing a single line of code.

1. **No PyTorch online.** Cross-encoder inference must use `onnxruntime` only. Export the model offline once (see Offline Export). Do not import `torch`, `sentence_transformers`, or `CrossEncoder` at rank time.

2. **Pairs only — no index search.** Unlike Stage 3, Stage 4 does not query FAISS or BM25. It scores a fixed list of `(candidate_id)` from `stage3_retrieved.parquet`. Complexity is **O(n_pairs)** where n ≈migrate300–600.

3. **Preserve all upstream columns.** Write through every column from `stage3_retrieved.parquet`. Stage 5 consumes Stage 2 flags, Stage 3 scores, and `cross_encoder_score` together.

4. **No threshold or count is hardcoded.** Model path, batch size, token limits, JD text, `keep_n`, and tiebreak rules live under `stage4:` in `config.yaml`.

5. **Deterministic tiebreak.** When `cross_encoder_score` ties, sort by `candidate_id` ascending (same rule as Stage 3).

6. **Expose CLI + `run()` entry point.** Manual runs use the CLI; `rank.py` calls `run(stage3_path, output_dir, config_path)` for single-command reproduction.

7. **Latency budget.** Stage 4 must leave headroom within the global ≤5 min CPU budget for Stages 2, 3, and 5. Target **≤90 seconds** for 600 pairs on a typical hackathon CPU (see Latency Budget).

8. **Stage 4 answers text fit only.** Do not fold behavioral signals, availability flags, or honeypot rules into the cross-encoder score. Those belong in Stage 5.

---

## MODEL SELECTION

### Decision summary

| Role | Model | Hugging Face ID | Why |
|------|-------|-----------------|-----|
| **Primary (default)** | MS MARCO MiniLM L-6 | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Best speed/quality tradeoff for CPU reranking at 300–600 pairs; pre-exported ONNX on Hub; MS MARCO training matches query↔passage relevance |
| **Latency fallback** | MS MARCO TinyBERT L-2 | `cross-encoder/ms-marco-TinyBERT-L-2-v2` | ~4× faster than MiniLM-L-6; use if primary exceeds latency budget after batching/quantization |
| **Quality upgrade** | MS MARCO MiniLM L-12 | `cross-encoder/ms-marco-MiniLM-L-12-v2` | ~2× slower than L-6, modest NDCG gain on MS MARCO; try only if L-6 top-20 manual review is weak and latency allows |
| **Domain reranker (optional A/B)** | BGE Reranker Base | `BAAI/bge-reranker-base` | Strong on semantic retrieval reranking; 278M params — likely too slow for default CPU budget; benchmark offline before adopting |

**Selected for production:** `cross-encoder/ms-marco-MiniLM-L-6-v2`

**Config key:** `stage4.model_id: cross-encoder/ms-marco-MiniLM-L-6-v2`

### Why not use the Instructor bi-encoder as a “cross-encoder”?

Stage 3 already uses Instructor-large as a **bi-encoder** (separate query/candidate encodings + dot product). Stage 4 exists precisely because joint encoding is more accurate for fine-grained fit (Tier-5 plain-language experts vs keyword stuffers). Reusing Instructor in pairwise mode would require a different architecture and export; MS MARCO cross-encoders are purpose-built for this rerank step.

### Why MS MARCO MiniLM-L-6 over larger models?

| Model | Params | Typical CPU (600 pairs, batch 16) | Notes |
|-------|--------|-------------------------------------|-------|
| TinyBERT-L-2 | ~4M | ~15–30 s | Fastest; weaker discrimination |
| **MiniLM-L-6** | **22M** | **~45–90 s** | **Default — fits budget** |
| MiniLM-L-12 | 33M | ~90–150 s | Marginal quality gain |
| bge-reranker-base | 278M | ~3–8 min | Risk of blowing 5 min total budget |

Estimates assume INT8 or O3 ONNX, `max_length=512`, batch inference. **Measure on target hardware** before locking config.

### ONNX artifact layout

Download or export once to `models/cross_encoder/`:

```
models/cross_encoder/
├── model.onnx              # or model_O3.onnx / model_quint8_avx2.onnx for speed
├── config.json
└── tokenizer/              # vocab + tokenizer_config from HF
    ├── tokenizer.json
    ├── vocab.txt
    └── ...
```

**Preferred ONNX variant (x86 CPU):** `model_quint8_avx2.onnx` or `model_qint8_avx512_vnni.onnx` from the model’s Hub `onnx/` subfolder — already published by the cross-encoder maintainers.

**Offline export (if not using Hub ONNX):**

```bash
pip install optimum[exporters-onnx]
optimum-cli export onnx \
  --model cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --task text-classification \
  models/cross_encoder/
```

Export is a **one-time offline step** (PyTorch allowed here only). Commit `models/cross_encoder/` to artifact storage or document download in README — do not download weights during `rank.py`.

---

## HOW CROSS-ENCODERS WORK (vs Stage 3 bi-encoder)

**Bi-encoder (Stage 3):**

```
q_vec = encode(JD_query)          # separate forward pass
c_vec = encode(candidate_passage)   # separate forward pass
score = dot(q_vec, c_vec)
```

**Cross-encoder (Stage 4):**

```
score = model([CLS] JD_text [SEP] candidate_text [SEP])
```

The transformer attends across both texts in one forward pass, enabling alignments like “JD asks for production FAISS” ↔ “candidate describes operating HNSW indexes at scale.” This is slower but more accurate — feasible only after Stage 3 narrows to hundreds of candidates.

---

## PAIR CONSTRUCTION

Each candidate becomes one `(query, document)` pair fed to the cross-encoder.

### JD side (query text)

Use a **distilled JD summary**, not the raw full job posting. Stored in `config.yaml` as `stage4.jd_text`.

**Default `stage4.jd_text`** (aligned with Stage 3 Q1 + Q2 intent):

```
Senior AI Engineer role requiring production embeddings-based retrieval and hybrid search
systems deployed to real users; hands-on vector database operations (FAISS, Qdrant, Weaviate,
Milvus, Pinecone, OpenSearch, or Elasticsearch); evaluation framework design (NDCG, MRR, MAP,
offline-to-online correlation, A/B tests); strong Python in production ML. Ideal candidate has
6–8 years total experience, 4–5 in applied ML at product companies, shipped end-to-end ranking
or search to scale, and avoids consulting-only or research-only trajectories without production
deployments.
```

**Token budget:** truncate JD to `stage4.max_jd_tokens` (default 256) from the start — JD is the query, keep the full requirement list.

### Candidate side (document text)

Build from candidate record in priority order:

1. **`technical_summary_sentence`** from `candidate_features.parquet` if present (Track B precomputed ~30-word summary).
2. Else **`build_candidate_passage(record)`** from `tracks/instructor/extraction.py` — role-contextualized career descriptions + summary + current title.
3. Truncate to `stage4.max_candidate_tokens` (default 384) **from the end** (preserve leading / most recent roles), same policy as Instructor precompute.

**Do not** use jargon-only BM25 text (`build_jargon_text`) as the sole candidate input — it drops narrative evidence Stage 4 needs for Tier-5 plain-language experts.

### Pair format for MS MARCO models

MS MARCO cross-encoders expect `(query, passage)` order:

```python
# query = JD side, passage = candidate side
pairs = [(jd_text, candidate_text), ...]
```

Tokenizer: max length `stage4.max_pair_tokens` (default 512), truncate longest side first or use `truncation='only_second'` so JD query is preserved.

---

## TECH STACK

| Library | Purpose |
|---------|---------|
| `polars` | Read/write parquet, assemble output DataFrame |
| `numpy` | Score arrays |
| `onnxruntime` | Cross-encoder inference (CPU `CPUExecutionProvider`) |
| `transformers` | Tokenizer only (`AutoTokenizer`) — no model weights at runtime |
| `pyyaml` | Config |
| `argparse` | CLI |

**Do not** import `torch`, `sentence_transformers`, or `CrossEncoder` in the online path.

---

## DIRECTORY AND FILE STRUCTURE

```
tracks/instructor/stage4/
├── __init__.py
├── config.py          # load stage4: block from config.yaml
├── pairs.py           # build (jd_text, candidate_text) per candidate_id
├── score.py           # batched ONNX inference
├── io.py              # load stage3, join features, write outputs
└── run.py             # orchestrator + CLI

models/cross_encoder/  # offline ONNX + tokenizer (gitignored or LFS)
```

Outputs go to `artifacts/runtime/`.

---

## PROCEDURE

### STEP 1 — Load config and inputs

1. Load `stage4:` from `config.yaml`.
2. Load `stage3_retrieved.parquet`; assert row count in `[stage3.min_k, stage3.max_k]` or warn.
3. Load candidate text source: join `candidate_features.parquet` on `candidate_id`, or index `candidates.jsonl` for fallback passage building.

### STEP 2 — Build pairs

For each row in stage3 output (preserve order for logging):

1. Resolve `candidate_text` via pair construction rules above.
2. Pair with frozen `stage4.jd_text`.
3. Skip empty candidate text — assign `cross_encoder_score = -inf` (or sentinel `-1e9`) and log warning; do not crash.

### STEP 3 — Batched ONNX inference

1. Create `onnxruntime.InferenceSession` from `stage4.onnx_model_path`.
2. Tokenize pairs in batches of `stage4.batch_size` (default 16).
3. Run session; read logits — for MS MARCO sequence classification, take score from relevant logit (typically single output or `[batch, 1]`).
4. Store raw float score per `candidate_id`.

**Session options for CPU:**

```python
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = stage4.num_threads  # default: 4
sess_options.inter_op_num_threads = 1
providers = ["CPUExecutionProvider"]
```

### STEP 4 — Sort, rank, cut

1. Sort by `cross_encoder_score` descending; tiebreak `candidate_id` ascending.
2. Assign `stage4_rank` (1 = best).
3. Keep top `stage4.keep_n` rows (default 300).
4. Optionally log rank deltas: `stage3_rank - stage4_rank` for analysis.

### STEP 5 — Write outputs

1. **`stage4_reranked.parquet`** — all Stage 3 columns +:

| Column | Type | Description |
|--------|------|-------------|
| `cross_encoder_score` | float | Raw model output |
| `stage4_rank` | int | Rank after cross-encoder (1 = best) |
| `stage4_model_id` | string | Model HF id used (audit trail) |

2. **`stage4_rank_delta.csv`** — candidates with `abs(stage3_rank - stage4_rank) >= stage4.rank_delta_threshold` (default 50).

3. **`stage4_summary.json`** — input count, output count, model id, batch size, total inference seconds, score min/max/mean/std.

4. Print stdout summary (mirror Stage 3): top 10 by `cross_encoder_score` with `stage3_rank` vs `stage4_rank` for operator review.

---

## CONFIG PARAMETERS

Add to `config.yaml`:

```yaml
stage4:
  # === Model ===
  model_id: cross-encoder/ms-marco-MiniLM-L-6-v2
  onnx_model_path: models/cross_encoder/model_quint8_avx2.onnx
  tokenizer_path: models/cross_encoder/tokenizer

  # === Fallback models (document only; switch model_id to use) ===
  # model_id: cross-encoder/ms-marco-TinyBERT-L-2-v2      # latency fallback
  # model_id: cross-encoder/ms-marco-MiniLM-L-12-v2       # quality upgrade
  # model_id: BAAI/bge-reranker-base                      # domain A/B only

  # === JD / pair text ===
  jd_text: |
    Senior AI Engineer role requiring production embeddings-based retrieval and hybrid search
    systems deployed to real users; hands-on vector database operations (FAISS, Qdrant, Weaviate,
    Milvus, Pinecone, OpenSearch, or Elasticsearch); evaluation framework design (NDCG, MRR, MAP,
    offline-to-online correlation, A/B tests); strong Python in production ML. Ideal candidate has
    6–8 years total experience, 4–5 in applied ML at product companies, shipped end-to-end ranking
    or search to scale, and avoids consulting-only or research-only trajectories without production
    deployments.

  max_jd_tokens: 256
  max_candidate_tokens: 384
  max_pair_tokens: 512

  # === Inference ===
  batch_size: 16
  num_threads: 4

  # === Output cut ===
  keep_n: 300
  rank_delta_threshold: 50

  # === Paths (optional overrides) ===
  candidate_features_path: artifacts/precomputed/candidate_features.parquet
  candidates_jsonl_path: data/candidates.jsonl
```

---

## LATENCY BUDGET

| Stage | Target (600 candidates) |
|-------|---------------------------|
| Stage 4 inference | ≤ 90 s |
| Stage 4 total (I/O + tokenize + infer + write) | ≤ 120 s |

If over budget:

1. Increase `batch_size` (try 32).
2. Switch ONNX to `model_quint8_avx2.onnx` or `model_O3.onnx`.
3. Switch model to `cross-encoder/ms-marco-TinyBERT-L-2-v2`.
4. Reduce Stage 3 `max_k` to 400 (fewer pairs).

Log `pairs_per_second` in `stage4_summary.json` every run.

---

## INPUT / OUTPUT CONTRACTS

### Input

`stage3_retrieved.parquet` must contain at minimum: `candidate_id`, all Stage 2 flag columns, all Stage 3 score columns (`q1_score`, `q2_score`, `bm25_score`, `fused_score`, `stage3_rank`, etc.). Missing columns → error before inference.

### Output

`stage4_reranked.parquet`:

- All input columns (passthrough)
- `cross_encoder_score`, `stage4_rank`, `stage4_model_id`
- Row count = `min(input_count, keep_n)` — typically 300
- Sorted by ascending `stage4_rank`

---

## EXPERIMENTS (before locking production model)

Run these manually; record results in a spreadsheet or `experiments/stage4/`.

| # | Experiment | Vary | Success criterion |
|---|------------|------|-------------------|
| 1 | Model latency | MiniLM-L-6 vs TinyBERT-L-2 vs MiniLM-L-12 | L-6 fits ≤90 s at 600 pairs |
| 2 | JD text | Full JD vs `stage4.jd_text` distilled | Distilled wins on top-20 manual review |
| 3 | Candidate text | Summary-only vs full passage vs summary+last role | Best Tier-5 recall in top 20 |
| 4 | ONNX precision | FP32 vs quint8 | <1% rank change at top 50 |
| 5 | keep_n | 200 vs 300 vs 400 | Stage 5 recall of manual “strong” labels |
| 6 | Trap audit | Top 30 after Stage 4 only | No keyword stuffers / obvious honeypots in top 10 |
| 7 | Rank delta | \|stage3_rank − stage4_rank\| > 50 | Manual review confirms CE fixes bi-encoder mistakes |

**Manual review protocol:** Rate 30 stratified candidates (Strong / OK / Weak / Trap). Pick model + text config with best trap rejection and Tier-5 recall.

---

## OPERATOR VALIDATION CHECKLIST (run after Stage 4 before Stage 5)

1. Output row count equals `keep_n` (or input count if input < keep_n).
2. Top 10 by `cross_encoder_score` look like senior ML/IR engineers — fewer false positives than Stage 3 top 10.
3. At least one high `stage4_rank` candidate had low `bm25_score` but strong profile text (Tier-5 plain-language path).
4. No obvious keyword stuffer in top 20 (`title_family == non_eng` should not appear — if it does, Stage 2 leaked).
5. `cross_encoder_score` decreases monotonically with `stage4_rank`.
6. `stage4_summary.json` reports inference time within latency budget.
7. Compare top 10: note large rank movers (`stage3_rank` vs `stage4_rank`) and spot-check they make sense.

---

## RELATIONSHIP TO STAGE 5

Stage 4 produces **textual relevance** only. Stage 5 combines:

- `cross_encoder_score` (primary semantic feature)
- Stage 3 scores (`fused_score`, `q1_score`, `bm25_score`, …)
- Stage 2 flags (`exp_band`, `in_sweet_spot`, `stale_profile`, …)
- Full 23 `redrob_signals` from Track B

Do **not** use cross-encoder score alone as the final Top 100 unless running the Stage 5 placeholder for pipeline smoke tests.

---

## OFFLINE SETUP CHECKLIST

- [ ] Download or export `cross-encoder/ms-marco-MiniLM-L-6-v2` ONNX + tokenizer to `models/cross_encoder/`
- [ ] Verify ONNX session runs on CPU with a single test pair
- [ ] Benchmark 600 pairs; confirm ≤90 s inference
- [ ] Add `stage4:` block to `config.yaml`
- [ ] Confirm Stage 3 output passes validation checklist
- [ ] Run Stage 4; complete operator validation checklist before Stage 5

---

## MANUAL RUN

```bash
python tracks/instructor/stage4/run.py \
  --in artifacts/runtime/stage3_retrieved.parquet \
  --out artifacts/runtime/stage4_reranked.parquet \
  --config config.yaml
```

Expected wall time: ~1–2 minutes for 300–600 pairs on CPU with MiniLM-L-6 INT8 ONNX.
