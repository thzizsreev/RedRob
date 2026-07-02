# Repository layout

```
redrob/
├── rank.py                  # Hackathon ranking entry point (Stages 1–5, CPU-only)
├── apply_reasoning.py       # Stage 6 → SignalHunters.csv (upload)
├── SignalHunters_ranking.csv   # Ranking-only output (gitignored; from rank.py)
├── SignalHunters.csv        # Submission file (gitignored; from apply_reasoning.py)
├── submission_metadata.yaml   # Portal metadata template
├── config.yaml              # Production pipeline config (stage2–6)
├── requirements.txt         # Python dependencies (full pipeline)
├── README.md
│
├── tracks/                  # MAIN PIPELINE — Stage 0–6 + naive baseline (do not relocate)
├── data/                    # Candidate pools (candidates.jsonl, samples)
├── artifacts/               # Pipeline outputs — rebuild via stage runners
├── docker/                  # Hackathon CPU sandbox image + scripts
│
├── guide/                   # Operator handbook + per-stage deep dives
├── docs/                    # Planning artifacts and reference material
│   ├── plans/               # Architecture & stage design plans
│   ├── reports/             # Implementation reports
│   ├── reference/           # submission_spec.txt, job description, honeypot notes
│   └── REPO_LAYOUT.md
│
├── experiments/             # Research prototypes (NOT production path)
├── tools/                   # CLI utilities & report builders
├── tests/                   # Unit & integration tests
├── models/                  # Cross-encoder & paraphraser ONNX export trees
├── onnx/                    # INSTRUCTOR ONNX export
└── outputs/                 # Generated views & test run artifacts (gitignored)
```

**Stage 3 reproduce:** `python rank.py` → `SignalHunters_ranking.csv` (3 columns, ≤5 min CPU).

**Portal upload:** `SignalHunters.csv` (4 columns, from `python apply_reasoning.py`).

Docker run (mount repo root to `/output`): writes ranking CSV for the baked 1K pool.

## Common commands

| Task | Command |
|------|---------|
| Ranking (spec, Stage 3) | `python rank.py --candidates data/candidates.jsonl --out SignalHunters_ranking.csv` |
| Ranking (defaults) | `python rank.py` |
| Full submission | `python apply_reasoning.py` |
| Validate upload | `python tools/validate_submission.py SignalHunters.csv` |
| Full pipeline (dev) | `python tracks/instructor/run_pipeline.py` |
| Docker sandbox | `docker run --rm -v "$(pwd):/output" thzizsreev/redrob-sandbox:latest` |

Production code lives under **`tracks/`**. Everything under **`experiments/`** and **`outputs/`** is supporting work.
