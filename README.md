# RedRob — Candidate Retrieval

Hackathon project with two retrieval tracks:

| Track | Model | Vector dim | Precompute | Stage 1 | Rank |
|-------|-------|------------|------------|---------|------|
| **INSTRUCTOR (Track A)** | `hkunlp/instructor-large` via ONNX CUDA | 2304 (3×768 blocks) | `python tracks/instructor/stage0/run.py` | `python tracks/instructor/stage1/run_filter.py` | `python rank.py` |
| **Naive baseline** | `BAAI/bge-small-en-v1.5` | 384 | `python -m tracks.naive.precompute` | — | `python -m tracks.naive.rank` |

## Quick start (Track A)

1. Export ONNX model (once): `cd onnx && python export_to_onnx.py`
2. Edit `CANDIDATES_PATH` and `OUTPUT_DIR` in [`tracks/instructor/stage0/run.py`](tracks/instructor/stage0/run.py), then run from project root:
   ```bash
   python tracks/instructor/stage0/run.py
   ```
3. (Optional) Edit paths in [`tracks/instructor/stage1/run_filter.py`](tracks/instructor/stage1/run_filter.py), then run Stage 1 filter:
   ```bash
   python tracks/instructor/stage1/run_filter.py
   ```
4. Edit `ARTIFACTS_PATH` in [`rank.py`](rank.py), then run:
   ```bash
   python rank.py
   ```

ONNX smoke test (from project root): `python onnx/run_encode.py`

## Quick start (naive baseline)

```bash
python -m tracks.naive.precompute
python -m tracks.naive.rank
```

## Project layout

```
redrob/
├── rank.py                # Track A CPU-only retrieval
├── tracks/
│   ├── instructor/        # INSTRUCTOR 2304-d pipeline
│   │   ├── stage0/run.py  # offline index precompute
│   │   ├── stage1/        # cluster + filter (run_cluster.py, run_filter.py)
│   │   └── stage2/run.py  # hard tabular gate
│   ├── naive/             # BGE 384-d baseline
│   └── shared/            # Common paths (data/, artifacts/)
├── artifacts/
│   └── runtime/           # Instructor pipeline outputs by stage
│       ├── stage0/        # FAISS index, vectors, BM25, jd_query_vec
│       ├── stage1/        # cluster .npy + filter JSON
│       ├── stage2/        # gated parquet + logs
│       ├── stage3/        # retrieved parquet + score distribution
│       └── stage4/        # cross-encoder reranked parquet
├── models/
│   └── export_cross_encoder.py  # Phase A ONNX export (one-time)
├── onnx/                  # ONNX export tooling
├── tests/                 # retrieval + clustering tests
├── docs/                  # architecture notes
└── data/                  # candidate JSON / JSONL samples
```

## Tests

```bash
python tests/retrieval_test.py
python tests/run_clustering_test.py
python tests/run_filtering_test.py
```

## Stage 1 filtering (Track A)

Stage 1 is split into two steps:

| Step | Script | When | What |
|------|--------|------|------|
| **A** | `tracks/instructor/stage1/run_cluster.py` | Once per pool | FAISS export → UMAP → HDBSCAN → `stage1/*.npy` |
| **B** | `tracks/instructor/stage1/run_filter.py` | Repeat | Load `.npy` → rank by JD anchor → filter to floor → JSON |

Before retrieval, `rank.py` runs **Phase B only** by default (assumes Phase A was run offline):

1. Load precomputed cluster labels and vectors from `artifacts/<pool>/stage1/`
2. Rank clusters by **median** inner-product similarity to `jd_query_vec.npy`
3. Walk ranked clusters atomically until `floor=100` candidates are selected
4. Top-k retrieval runs only on that filtered pool

Production modules live under `tracks/instructor/clustering/` and `tracks/instructor/filtering/`. Tests import from `tracks/` only — production code never imports from `tests/`.

**Expected runtime:** Phase A (UMAP + HDBSCAN) can take minutes at 5k–100k scale. Phase B is seconds. Disable Stage 1 in retrieval with `retrieve(..., use_stage1_filter=False)` for A/B comparison.

Run Stage 1 in isolation (edit paths in [`run_cluster.py`](tracks/instructor/stage1/run_cluster.py) and [`run_filter.py`](tracks/instructor/stage1/run_filter.py)):

```bash
python tracks/instructor/stage1/run_cluster.py   # once: writes stage1/*.npy
python tracks/instructor/stage1/run_filter.py  # repeat: writes filtered JSON under stage1/
python tracks/instructor/stage2/run.py         # hard tabular gate → artifacts/runtime/stage2/
python tracks/instructor/stage3/run.py         # hybrid retrieval → artifacts/runtime/stage3/
```

The legacy test entry point `tests/run_filtering_test.py` mirrors the same two-phase flow programmatically.

## Stage 4 cross-encoder reranking (Track A)

Two phases — export once, rerank each run:

```bash
# Phase A — once per environment
pip install -r models/requirements.txt
python models/export_cross_encoder.py

# Phase B — after Stage 3
python tracks/instructor/stage4/run.py
```

Reads `artifacts/runtime/stage3/stage3_retrieved.parquet`, scores `(JD, candidate)` pairs with MS MARCO MiniLM ONNX on CPU, writes `artifacts/runtime/stage4/stage4_reranked.parquet` (~300 rows).

## Notes

- Track A rank is **CPU-only** (loads precomputed `jd_query_vec.npy` + FAISS index).
- Naive rank encodes the JD query at runtime with BGE-small.
- Large data files (`candidates.jsonl`, sample JSONs) may be gitignored — keep local copies.
