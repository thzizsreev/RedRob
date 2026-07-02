# Stage 3 Isolated Test Environment

Self-contained sandbox implementing the **skill-track Stage 3** plan from
`docs/plans/stage3_new_plan.md` (BM25 L4 replaced by `skill_weighted_score` L3).

**Does not modify** any production pipeline code under `tracks/`.

## Prerequisites

- Stage 0 artifacts at `artifacts/runtime/stage0/` (FAISS index, vectors, id_map)
- Stage 2 output at `artifacts/runtime/stage2/stage2_gated.parquet`
- `data/candidates.jsonl` (for skill stub fixture)
- ONNX model at `onnx/models/instructor-large-encoder.onnx`

Uses a **CPU-only** ONNX embedder (`stage3/cpu_embedder.py`) so the test runs without CUDA.

## Quick start

From repo root:

```bash
python experiments/stage3/build_fixtures.py
python experiments/stage3/run.py
python experiments/stage3/validate_output.py
```

## Layout

| Path | Purpose |
|------|---------|
| `inputs/stage2_gated_sample.parquet` | ~500-row slice of Stage 2 survivors |
| `inputs/candidate_features_sample.parquet` | Stub `skill_weighted_score` per candidate |
| `inputs/artifacts_manifest.json` | Read-only pointer to Stage 0 artifacts |
| `outputs/` | All Stage 3 run artifacts |
| `stage3/` | Vendored implementation (skill L3 track) |

## Skill score stub

`candidate_features_sample.parquet` uses a **deterministic proxy** scorer in
`build_fixtures.py` — not production Blocks A–D precompute. It validates L3 wiring
only; ranking quality should be validated after real precompute exists.

## Outputs

- `outputs/stage3_retrieved.parquet` — final cut (80–150 rows for sample config)
- `outputs/stage3_score_distribution.csv` — full union with kept/cut flag
- `outputs/stage3_summary.json` — run metrics
- `outputs/stage3_retrieved.json` — JSON mirror of retrieved parquet
