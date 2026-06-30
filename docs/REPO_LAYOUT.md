# Repository layout

```
redrob/
├── config.yaml              # Production pipeline config (stage2–5, stage3 query facets)
├── tracks/                  # MAIN PIPELINE — Stage 0–5 + naive baseline
│   ├── instructor/          # Track A (INSTRUCTOR-large, full funnel)
│   ├── naive/               # Track B baseline
│   └── shared/              # Path constants, shared helpers
├── artifacts/runtime/       # Pipeline outputs (stage0 … stage5) — rebuild via stage runners
├── data/                    # Candidate pools (candidates.jsonl, samples)
├── models/                  # Cross-encoder ONNX export
├── onnx/                    # INSTRUCTOR ONNX export
├── docs/                    # Architecture & stage plans
│
├── experiments/             # Research, prototypes, tuning (NOT production path)
│   ├── stage3/              # Stage 3 skill-track test harness
│   ├── q1_q2/               # Q1/Q2 facet-centroid vector tuning
│   ├── kmeans/              # K-means Stage 1 alternative
│   ├── honeypot/            # LLM honeypot study
│   ├── diagnostics/         # Stage 5 formula diagnostics
│   ├── pipeline/            # Legacy pipeline scratch
│   └── rishi/               # Personal experiment scratch
│
├── tools/                   # CLI utilities & report builders
│   ├── build_team_view.py   # Stage 5 → HTML ranking page
│   ├── build_eliminations_view.py
│   ├── build_collection.py
│   ├── validate_submission.py
│   ├── rank.py              # Fast CPU retrieval test
│   └── eliminations_view/   # Eliminations HTML generator
│
├── outputs/                 # Generated views & test run artifacts (gitignored)
│   ├── team_views/          # HTML/JSON team results (was team_results_view*)
│   ├── eliminations/        # Elimination funnel views
│   └── test_runs/           # Clustering/filtering/retrieval tests (was test_output/)
│
└── tests/                   # Unit & integration tests
```

## Common commands (updated paths)

| Task | Command |
|------|---------|
| Full pipeline | `python tracks/instructor/run_pipeline.py` |
| Team HTML view | `python tools/build_team_view.py` |
| Q1/Q2 vector test | `python experiments/q1_q2/run_test.py` |
| Stage 3 experiment | `python experiments/stage3/precompute/run.py` then `runner/run.py` |

Production code lives under **`tracks/`**. Everything under **`experiments/`** and **`outputs/`** is supporting work.
