# RedRob — Candidate Retrieval

Hackathon project with two retrieval tracks:

| Track | Model | Vector dim | Precompute | Stage 1 | Rank |
|-------|-------|------------|------------|---------|------|
| **INSTRUCTOR (Track A)** | `hkunlp/instructor-large` via ONNX CUDA | 2304 (3×768 blocks) | `python precompute.py` | `python stage1.py` | `python rank.py` |
| **Naive baseline** | `BAAI/bge-small-en-v1.5` | 384 | `python -m tracks.naive.precompute` | — | `python -m tracks.naive.rank` |

## Quick start (Track A)

1. Export ONNX model (once): `cd onnx && python export_to_onnx.py`
2. Run precompute (edit paths or pass CLI args), from project root:
   ```bash
   python precompute.py --candidates data/candidates.jsonl --output-dir artifacts/candidates
   ```
   Default input: `data/candidates.jsonl`. Default output: `artifacts/candidates/` (derived from filename).
   Writes `candidate_index.faiss`, `id_map.json`, `jd_query_vec.npy` under `--output-dir`.
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

Before retrieval, `rank.py` runs **full online** Stage 1 by default:

1. UMAP reduce (12-d) → HDBSCAN cluster assignment
2. Rank clusters by **median** inner-product similarity to `jd_query_vec.npy`
3. Walk ranked clusters atomically until `floor=100` candidates are selected
4. Top-k retrieval runs only on that filtered pool

Production modules live under `tracks/instructor/clustering/` and `tracks/instructor/filtering/`. Tests import from `tracks/` only — production code never imports from `tests/`.

**Expected runtime:** UMAP + HDBSCAN on the full pool can take minutes at 5k–100k scale. Disable with `retrieve(..., use_stage1_filter=False)` for A/B comparison.

Run Stage 1 in isolation (edit paths in [`stage1.py`](stage1.py)):

```bash
python stage1.py
```

Output: configurable via `OUTPUT_DIR` in `stage1.py` (default: `artifacts/<sample>/stage1/`)

The legacy test entry point `tests/run_filtering_test.py` is frozen and not used by production.

## Notes

- Track A rank is **CPU-only** (loads precomputed `jd_query_vec.npy` + FAISS index).
- Naive rank encodes the JD query at runtime with BGE-small.
- Large data files (`candidates.jsonl`, sample JSONs) may be gitignored — keep local copies.
