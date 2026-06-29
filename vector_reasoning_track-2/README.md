# Vector Reasoning Track 2 — Plan 3 SONAR

Isolated experiment: encode → steer → decode in SONAR's native **1024d** space.

Source plan (read-only): [docs/vector_steering_plan_3_sonar.md](../docs/vector_steering_plan_3_sonar.md)

## Requirements

- **Python 3.10 or 3.11**
- **Linux or WSL** — `fairseq2n` native wheels are not published for Windows; native Windows pip install will fail at `fairseq2n`
- ~2.4GB model download on first run (cached in `~/.cache/fairseq2/`)

## Platform notes

| Environment | Supported |
|-------------|-----------|
| Linux (Python 3.10/3.11) | Yes |
| WSL Ubuntu (Python 3.10/3.11 + `python3-venv`) | Yes |
| Windows native Python | **No** — use WSL |

Quick check:

```bash
python validate_sonar.py
```

## Install (Linux / WSL)

```bash
cd vector_reasoning_track-2
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.0.1 torchaudio==2.0.2
pip install fairseq2==0.2.1 --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.0.1/cpu/
pip install sonar-space==0.2.1 numpy==1.24.0
python validate_sonar.py
```

If the fairseq2 wheel index fails, see the conda fallback in the plan doc.

## Run

```bash
python precompute.py      # once — 15 vectors in vectors/ (1024d)
python run_experiment.py  # blend + steer + decode → results/output.json
```

## Output

Final `reasoning` is **three decoded clauses only** (no template prefix). Each clause is produced by decoding a steered 1024d vector via SONAR beam search.

## Models

| Role | Model |
|------|--------|
| Encode | `text_sonar_basic_encoder` |
| Decode | `text_sonar_basic_decoder` |

Hyperparams: `GAMMA=0.55`, `DELTA=0.30`, `MAX_SEQ_LEN=64`.

All code and outputs stay in this folder. No imports from `tracks/`.
