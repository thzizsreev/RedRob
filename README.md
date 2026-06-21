# RedRob — Candidate Retrieval

Hackathon project with two retrieval tracks:

| Track | Model | Vector dim | Precompute | Stage 1 | Rank |
|-------|-------|------------|------------|---------|------|
| **INSTRUCTOR (Track A)** | `hkunlp/instructor-large` via ONNX CUDA | 2304 (3×768 blocks) | `python precompute.py` | `python stage1.py` | `python rank.py` |
| **Naive baseline** | `BAAI/bge-small-en-v1.5` | 384 | `python -m tracks.naive.precompute` | — | `python -m tracks.naive.rank` |

## Quick start (Track A)

1. Export ONNX model (once): `cd onnx && python export_to_onnx.py`
2. Edit `CANDIDATES_PATH` and `OUTPUT_DIR` in [`precompute.py`](precompute.py), then run from project root:
   ```bash
   python precompute.py
   ```
3. (Optional) Edit `ARTIFACTS_PATH` and `OUTPUT_DIR` in [`stage1.py`](stage1.py), then run Stage 1 in isolation:
   ```bash
   python stage1.py
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
├── precompute.py          # Track A offline index build
├── stage1.py              # Track A Stage 1 cluster filtering (standalone)
├── rank.py                # Track A CPU-only retrieval
├── tracks/
│   ├── instructor/        # INSTRUCTOR 2304-d pipeline
│   ├── naive/             # BGE 384-d baseline
│   └── shared/            # Common paths (data/, artifacts/)
├── artifacts/             # Track A outputs (per-sample subdirs)
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
| **A** | `stage1_cluster.py` | Once per pool | FAISS export → UMAP → HDBSCAN → `stage1/*.npy` |
| **B** | `stage1.py` | Repeat | Load `.npy` → rank by JD anchor → filter to floor → JSON |

Before retrieval, `rank.py` runs **Phase B only** by default (assumes Phase A was run offline):

1. Load precomputed cluster labels and vectors from `artifacts/<pool>/stage1/`
2. Rank clusters by **median** inner-product similarity to `jd_query_vec.npy`
3. Walk ranked clusters atomically until `floor=100` candidates are selected
4. Top-k retrieval runs only on that filtered pool

Production modules live under `tracks/instructor/clustering/` and `tracks/instructor/filtering/`. Tests import from `tracks/` only — production code never imports from `tests/`.

**Expected runtime:** Phase A (UMAP + HDBSCAN) can take minutes at 5k–100k scale. Phase B is seconds. Disable Stage 1 in retrieval with `retrieve(..., use_stage1_filter=False)` for A/B comparison.

Run Stage 1 in isolation (edit paths in [`stage1_cluster.py`](stage1_cluster.py) and [`stage1.py`](stage1.py)):

```bash
python stage1_cluster.py   # once: writes stage1/*.npy
python stage1.py           # repeat: writes filtered JSON under stage1/
```

The legacy test entry point `tests/run_filtering_test.py` mirrors the same two-phase flow programmatically.

## Notes

- Track A rank is **CPU-only** (loads precomputed `jd_query_vec.npy` + FAISS index).
- Naive rank encodes the JD query at runtime with BGE-small.
- Large data files (`candidates.jsonl`, sample JSONs) may be gitignored — keep local copies.
