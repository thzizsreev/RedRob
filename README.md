# RedRob

Precision candidate-ranking pipeline for a senior AI engineering role (production retrieval, vector search, ranking, and evaluation). The **INSTRUCTOR track** funnels ~100,000 anonymized profiles through offline precompute, CPU runtime ranking (Stages 1–5), and optional Stage 6 reasoning into a validated top-100 submission CSV.

---

## Docker sandbox (quick run)

CPU sandbox for hackathon reviewers — ranks a baked **1K pool** through Stages 1–5 in ~20s. **Not** the full 100K reproduce path (use [`rank.py`](#quick-start) for that).

**Image:** [hub.docker.com/r/thzizsreev/redrob-sandbox](https://hub.docker.com/r/thzizsreev/redrob-sandbox)

**Pull and run** — mount **repo root** to `/output` so the CSV is written beside this README:

```powershell
# Windows (PowerShell)
docker pull thzizsreev/redrob-sandbox:latest
docker run --rm -v "${PWD}:/output" thzizsreev/redrob-sandbox:latest
```

```bash
# macOS / Linux
docker pull thzizsreev/redrob-sandbox:latest
docker run --rm -v "$(pwd):/output" thzizsreev/redrob-sandbox:latest
```

| | |
|--|--|
| **Output** | `./SignalHunters_ranking.csv` (top 100, 3 columns) |
| **Constraints** | CPU only, ≤5 min, no network, baked Stage 0 artifacts |
| **Full 100K repro** | `python rank.py` (requires local Stage 0 artifacts + `data/candidates.jsonl`) |
| **Build / internals** | [`docker/README.md`](docker/README.md) |

**Build locally** (maintainers):

```bash
python docker/scripts/sample_pool1k.py --source data/candidates.jsonl --seed 42
python docker/scripts/make_sandbox_config.py
python docker/scripts/fetch_pool1k_artifacts.py
docker build -f docker/Dockerfile -t redrob-sandbox .
docker tag redrob-sandbox:latest thzizsreev/redrob-sandbox:latest
docker push thzizsreev/redrob-sandbox:latest
```

---

| Track | Embedding | Role |
|-------|-----------|------|
| **INSTRUCTOR (Track A)** | `hkunlp/instructor-large` (2304-d, ONNX) | Production pipeline (Stages 0–6) |
| **Naive baseline** | `BAAI/bge-small-en-v1.5` (384-d) | Comparison only |

**Team ID:** `SignalHunters` (set in [`config.yaml`](config.yaml) and [`tracks/shared/paths.py`](tracks/shared/paths.py)).

**Hackathon spec:** [`docs/reference/submission_spec.txt`](docs/reference/submission_spec.txt) · **Portal metadata:** [`submission_metadata.yaml`](submission_metadata.yaml)

**Deep dives:** [Guide](#guide) · [`guide/overview.md`](guide/overview.md)

---

## Quick start

One-time: install deps, place `data/candidates.jsonl`, run [Stage 0 precompute](#precompute-stage-0-gpu). Then from **repo root**:

```powershell
python -m venv env
.\env\Scripts\activate
pip install -r requirements.txt

# Ranking only (≤5 min CPU, spec reproduce command)
$env:REDROB_CPU_ONLY = "1"
python rank.py

# Full submission with reasoning (separate step, not in 5-min budget)
python apply_reasoning.py

# Validate upload file (must be named SignalHunters.csv per spec)
python tools/validate_submission.py SignalHunters.csv
```

| Step | Command | Output |
|------|---------|--------|
| Rank (Stages 1–5) | `python rank.py` | `./SignalHunters_ranking.csv` (3 columns) |
| Add reasoning (Stage 6) | `python apply_reasoning.py` | `./SignalHunters.csv` (4 columns) |
| Validate | `python tools/validate_submission.py SignalHunters.csv` | stdout pass/fail |

**Upload:** `SignalHunters.csv` (header: `candidate_id,rank,score,reasoning`).

---

## Pipeline overview

```
~100K  →  Stage 0 (precompute, GPU)  →  rank.py (Stages 1–5, CPU)  →  apply_reasoning.py (Stage 6)  →  SignalHunters.csv
```

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    OFFLINE (Stage 0, GPU)                    │
│  candidates.jsonl → INSTRUCTOR ONNX → vectors.npy + FAISS    │
│                  → skill features → Q1/Q2/Q3 vectors         │
│                  → UMAP/HDBSCAN → cluster artifacts        │
│                  → CE + paraphraser ONNX exports           │
└──────────────────────────┬──────────────────────────────────┘
                           │ artifacts/
┌──────────────────────────▼──────────────────────────────────┐
│                 RUNTIME (Stages 1–5, CPU)                    │
│  Cluster filter → Hard gate → FAISS+RRF → CE rerank →     │
│  4-tier cascade → stage5_scored_top100.parquet             │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                 POST (Stage 6, CPU)                           │
│  Template builder → T5 paraphrase (parallel workers) → CSV  │
└─────────────────────────────────────────────────────────────┘
```

| Stage | Purpose | Pool size | Guide |
|-------|---------|-----------|-------|
| 0 | Vectors, FAISS, skills, query vectors, ONNX exports | Full pool | [stage0-precompute.md](guide/stage0-precompute.md) |
| 1 | UMAP/HDBSCAN cluster filter (JD-anchored) | ~6K+ | [stage1-cluster-filter.md](guide/stage1-cluster-filter.md) |
| 2 | Hard tabular gate + honeypot | ~2–5K | [stage2-hard-gate.md](guide/stage2-hard-gate.md) |
| 3 | Hybrid retrieval (RRF + skill track) | 300–600 | [stage3-hybrid-retrieval.md](guide/stage3-hybrid-retrieval.md) |
| 4 | Cross-encoder rerank (ONNX CPU) | 300 | [stage4-cross-encoder.md](guide/stage4-cross-encoder.md) |
| 5 | 4-tier cascade scoring → top 100 | 100 | [stage5-cascade-scoring.md](guide/stage5-cascade-scoring.md) |
| 6 | Template + T5 paraphrase reasoning | 100 | [stage6-reasoning-builder.md](guide/stage6-reasoning-builder.md) |

**Entry points**

| Script | Stages | Output |
|--------|--------|--------|
| [`rank.py`](rank.py) | 1–5 | Ranking CSV, no `reasoning` column |
| [`apply_reasoning.py`](apply_reasoning.py) | 6 | `SignalHunters.csv` with full spec columns |
| [`tracks/instructor/run_pipeline.py`](tracks/instructor/run_pipeline.py) | 1–5 (dev) | Stage artifacts + CSV with template reasoning |
| [`tracks/instructor/stage6/run.py`](tracks/instructor/stage6/run.py) | 6 (dev) | Writes under `artifacts/runtime/stage6/` |

---

## Key files

### Repo root (generated / submission)

| File | Columns | When |
|------|---------|------|
| **`SignalHunters_ranking.csv`** | `candidate_id`, `rank`, `score` | After `rank.py` (Stage 3 reproduces this) |
| **`SignalHunters.csv`** | `candidate_id`, `rank`, `score`, `reasoning` | After `apply_reasoning.py` — **submit this** |

Both are gitignored. Team ID is defined in `tracks/shared/paths.py` (`TEAM_ID = "SignalHunters"`).

### Runtime artifacts (`artifacts/runtime/`, gitignored)

| Path | Contents |
|------|----------|
| `stage0/` | `candidate_index.faiss`, `candidate_vectors.npy`, `id_map.json`, skill features, query vectors |
| `stage1/` | Cluster labels, UMAP reduced vectors, filter survivors |
| `stage2/` | `stage2_gated.parquet`, honeypot log |
| `stage3/` | `stage3_retrieved.parquet` |
| `stage4/` | `stage4_reranked.parquet` |
| `stage5/` | `stage5_scored_top100.parquet`, `SignalHunters.csv` (ranking copy) |
| `stage6/` | `stage6_reasoning.parquet`, `stage6_summary.json`, audit CSV |

### Precomputed (`artifacts/precomputed/`, gitignored)

| Path | Purpose |
|------|---------|
| `reasoning_raw.parquet` | Pre-built s1/s2 template sentences (faster Stage 6) |
| `reasoning_lookup.json` | Cached paraphrased reasoning (skip re-run) |
| `candidate_features.parquet` | Skill assessments, tier-2 inputs |

### Models (export once, reuse)

| Path | Purpose |
|------|---------|
| `onnx/models/` | INSTRUCTOR-large encoder ONNX |
| `models/cross_encoder/` | MS MARCO cross-encoder for Stage 4 |
| `models/paraphraser/` | T5 paraphraser for Stage 6 |

### Config & metadata

| File | Role |
|------|------|
| [`config.yaml`](config.yaml) | `stage0_skill`, `stage2`–`stage6` parameters; `team_id` |
| [`submission_metadata.yaml`](submission_metadata.yaml) | Portal form mirror (reproduce commands, team info) |
| [`requirements.txt`](requirements.txt) | Unified Python dependencies |

---

## Commands reference

### `rank.py` — hackathon ranking (Stages 1–5)

Spec reproduce command. Requires Stage 0 artifacts. Enforces **≤5 min** CPU wall-clock.

```powershell
$env:REDROB_CPU_ONLY = "1"
python rank.py --candidates ./data/candidates.jsonl --out ./SignalHunters_ranking.csv
python rank.py   # same defaults
```

| Flag | Default | Description |
|------|---------|-------------|
| `--candidates` | `data/candidates.jsonl` | Candidate pool path |
| `--out` | `./SignalHunters_ranking.csv` | Ranking CSV output (3 columns) |
| `--config` | `config.yaml` | Pipeline config |
| `--runtime-dir` | `artifacts/runtime` | Parent of `stage0` … `stage5` |
| `--skip-validate` | off | Skip ranking CSV checks |

### `apply_reasoning.py` — Stage 6 reasoning

Reads ranking CSV, runs T5 paraphrase (or lookup cache), writes full submission file.

```powershell
python apply_reasoning.py
python apply_reasoning.py --input ./SignalHunters_ranking.csv --out ./SignalHunters.csv
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | `./SignalHunters_ranking.csv` | Ranking CSV from `rank.py` |
| `--out` | `./SignalHunters.csv` | Spec submission CSV (registered team ID) |
| `--config` | `config.yaml` | Stage 6 settings |
| `--artifacts-dir` | `artifacts/runtime/stage6` | Audit parquet + summary |
| `--skip-validate` | off | Skip spec validator |

### Precompute (Stage 0, GPU)

Run once per candidate pool. See [guide/stage0-precompute.md](guide/stage0-precompute.md).

```powershell
python tracks/instructor/stage0/run.py
python tracks/instructor/stage0/run_cluster.py
python tracks/instructor/stage0/run_cross_encoder.py
python tracks/instructor/stage0/run_paraphraser_export.py
python tracks/instructor/stage0/run_reasoning_raw_precompute.py   # optional, faster Stage 6
```

INSTRUCTOR ONNX export (first time only):

```powershell
cd onnx
pip install -r requirements.txt
python export_to_onnx.py
```

### Validation & audit

```powershell
python tools/validate_submission.py SignalHunters.csv
python tools/audit_top100_honeypots.py --submission SignalHunters.csv --candidates data/candidates.jsonl
python tools/build_reasoning_lookup.py artifacts/runtime/stage6/SignalHunters.csv
python tools/build_team_view.py
```

### Dev / alternate runners

```powershell
# Full pipeline with template reasoning in stage5 CSV (not spec path)
python tracks/instructor/run_pipeline.py

# Stage 6 via instructor runner (writes artifacts/runtime/stage6/)
python tracks/instructor/stage6/run.py

# Per-stage standalone
python tracks/instructor/stage1/run_filter.py
python tracks/instructor/stage2/run.py
python tracks/instructor/stage3/run.py
python tracks/instructor/stage4/run.py
python tracks/instructor/stage5/run.py
```

### Docker sandbox (1K demo)

Same commands as [Docker sandbox (quick run)](#docker-sandbox-quick-run) at the top of this README.

---

## Requirements

- **Python** 3.11+
- **GPU + CUDA** — Stage 0 precompute only (`onnxruntime-gpu`)
- **CPU** — Stages 1–6 runtime (`REDROB_CPU_ONLY=1` for ranking)
- **Disk** — tens of GB for full-pool artifacts
- **Data** — `data/candidates.jsonl` (local; not in repo)

Do not install CPU `onnxruntime` alongside `onnxruntime-gpu`.

---

## Repository layout

```
redrob/
├── rank.py                      # Hackathon entry: Stages 1–5 → SignalHunters_ranking.csv
├── apply_reasoning.py           # Stage 6 → SignalHunters.csv (upload)
├── SignalHunters_ranking.csv    # Ranking output (gitignored)
├── SignalHunters.csv            # Submission file (gitignored)
├── config.yaml                  # stage0_skill, stage2–stage6
├── submission_metadata.yaml     # Portal metadata template
├── requirements.txt
│
├── tracks/instructor/             # Stages 0–6 implementation
│   ├── stage0/ … stage6/
│   ├── run_pipeline.py          # Dev orchestrator (Stages 1–5)
│   └── core/                    # ONNX embedder, FAISS, I/O
├── artifacts/                   # Pipeline outputs (gitignored)
│   ├── runtime/stage0 … stage6/
│   └── precomputed/
├── data/                        # candidates.jsonl, samples
├── models/                      # cross_encoder/, paraphraser/
├── onnx/                        # INSTRUCTOR export
├── docker/                      # 1K CPU sandbox image
├── tools/                       # validate, team view, reasoning lookup
├── guide/                       # Per-stage operator handbooks
├── docs/                        # Plans, reports, submission spec
├── experiments/                 # Research prototypes (not production)
└── tests/
```

Full tree: [`docs/REPO_LAYOUT.md`](docs/REPO_LAYOUT.md).

---

## Configuration

| `config.yaml` block | Stages | Guide |
|---------------------|--------|-------|
| `stage0_skill` | Skill scoring (Stage 0) | [stage0-precompute.md](guide/stage0-precompute.md) |
| `stage2` | Hard gate | [stage2-hard-gate.md](guide/stage2-hard-gate.md) |
| `stage3` | Query vectors + retrieval | [stage3-hybrid-retrieval.md](guide/stage3-hybrid-retrieval.md) |
| `stage4` | Cross-encoder rerank | [stage4-cross-encoder.md](guide/stage4-cross-encoder.md) |
| `stage5` | Cascade scoring, `team_id` | [stage5-cascade-scoring.md](guide/stage5-cascade-scoring.md) |
| `stage6` | Paraphraser, `team_id`, lookup | [stage6-reasoning-builder.md](guide/stage6-reasoning-builder.md) |

Stage 1 constants: [`tracks/instructor/core/config.py`](tracks/instructor/core/config.py).

Set `stage5.team_id` and `stage6.team_id` to `SignalHunters`.

---

## Hackathon commands

| Field | Value |
|-------|-------|
| Sandbox link | https://hub.docker.com/r/thzizsreev/redrob-sandbox |
| Docker 1K demo | `docker run --rm -v "$(pwd):/output" thzizsreev/redrob-sandbox:latest` |
| Spec reproduce (100K) | `python rank.py --candidates ./data/candidates.jsonl --out ./SignalHunters_ranking.csv` |
| Shorthand | `python rank.py` |
| Full submission | `python apply_reasoning.py` → `SignalHunters.csv` |

See [Docker sandbox (quick run)](#docker-sandbox-quick-run) for pull/run commands and [Submission checklist](#submission-checklist) before upload.

---

## Submission checklist

Before uploading to the portal, confirm:

1. **CSV file:** `SignalHunters.csv` at repo root (exactly 101 lines: header + 100 rows).
2. **Validator:** `python tools/validate_submission.py SignalHunters.csv` prints `Submission is valid.`
3. **Reproduce command (portal):** `python rank.py` (Stage 3 reproduces ranking only; reasoning is offline Stage 6).
4. **GitHub repo:** `https://github.com/thzizsreev/RedRob` — README, `requirements.txt`, precomputed artifacts or rebuild scripts.
5. **Sandbox:** `https://hub.docker.com/r/thzizsreev/redrob-sandbox`
6. **Metadata mirror:** [`submission_metadata.yaml`](submission_metadata.yaml) matches portal fields (team, contacts, AI tools, methodology).

---

## Guide

| Guide | What it covers |
|-------|----------------|
| [`guide/overview.md`](guide/overview.md) | Full funnel, tech stack, config map, design rationale |
| [`guide/stage0-precompute.md`](guide/stage0-precompute.md) | Vectors, FAISS, skills, query vectors, ONNX exports |
| [`guide/stage1-cluster-filter.md`](guide/stage1-cluster-filter.md) | UMAP/HDBSCAN clustering and JD-anchor filter |
| [`guide/stage2-hard-gate.md`](guide/stage2-hard-gate.md) | Hard gate, honeypot rules, survivor enrichment |
| [`guide/stage3-hybrid-retrieval.md`](guide/stage3-hybrid-retrieval.md) | Q1/Q2/Q3 dense retrieval, skill track, RRF |
| [`guide/stage4-cross-encoder.md`](guide/stage4-cross-encoder.md) | Cross-encoder ONNX rerank to top 300 |
| [`guide/stage5-cascade-scoring.md`](guide/stage5-cascade-scoring.md) | 4-tier cascade scoring and top-100 selection |
| [`guide/stage6-reasoning-builder.md`](guide/stage6-reasoning-builder.md) | Template builder, T5 paraphrase, parallel workers |

---

## Further reading

| Document | Contents |
|----------|----------|
| [`docs/reference/submission_spec.txt`](docs/reference/submission_spec.txt) | Hackathon submission specification |
| [`docs/plans/final_consolidated_plan.md`](docs/plans/final_consolidated_plan.md) | JD coverage matrix |
| [`docs/REPO_LAYOUT.md`](docs/REPO_LAYOUT.md) | Full repository tree |
| [`docker/README.md`](docker/README.md) | Sandbox build and runtime flow |
