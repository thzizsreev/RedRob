# Honeypot LLM Research Pipeline

Research-only script to study honeypot patterns in candidate profiles using OpenAI. **Not imported by `rank.py` or `precompute.py`.** Output is for human review — do not wire JSONL results into production ranking without re-implementing findings as explicit offline rules.

## Setup

```powershell
pip install -r honeypot/requirements.txt
copy .env.example .env
# Edit .env and set OPENAI_API_KEY
```

Optional env vars:

| Variable | Default |
|----------|---------|
| `OPENAI_API_KEY` | required |
| `OPENAI_MODEL` | `gpt-4o-mini` |

## Usage

From project root:

```powershell
# Build stratified sample manifest only (no API calls)
python -m honeypot sample --candidates data/sample1k.json --per-stratum 50

# Full pipeline: sample + pass1 + pass2
python -m honeypot run --candidates data/sample1k.json --output honeypot/output --per-stratum 50 --workers 4 --rpm 60

# Sample-only via run subcommand
python -m honeypot run --sample-only --per-stratum 50

# Re-run pass 2 only (after pass1 JSONL exists)
python -m honeypot run --pass 2 --manifest honeypot/output/manifest.json

# Force re-judge (ignore idempotency)
python -m honeypot run --force --per-stratum 3
```

## Stratified sampling

| Stratum | Purpose |
|---------|---------|
| **A** | AI/ML-adjacent candidates (broad keyword net) |
| **B** | Known trap categories: keyword stuffer, tier-5 plain language, behavioral rescue, consulting-only, CV/speech/robotics |
| **C** | Random control from full pool |

Manifest is written to `honeypot/output/manifest.json` before any LLM calls.

## Outputs

```
honeypot/output/
├── manifest.json
├── pass1_results.jsonl
├── pass2_results.jsonl      # uncertain + honeypot+low confidence from pass1
├── failures.jsonl           # per-candidate errors
└── run_summary.json
```

Each JSONL row includes full structured judgment plus metadata (`pass`, `stratum`, `model`, `latency_ms`, etc.).

## Pass 2

Second pass runs for candidates where pass 1 verdict is `uncertain`, or `honeypot` with `confidence: low`. Pass 2 includes pass 1 reasoning and logs `verdict_changed` when the verdict differs.

## Concurrency and retries

- Multi-threaded API calls (`--workers`, default 4)
- Global rate limit (`--rpm`, default 60 req/min)
- Exponential backoff + jitter on 429/5xx/connection errors (up to 6 retries)

Results append incrementally — safe to resume after interruption (skips existing `(candidate_id, pass)` unless `--force`).

## Data loading

Candidate records are loaded via existing project I/O only:

- `tracks.instructor.io.iter_candidates_from_path`
- `tracks.shared.paths` for default paths

All other logic lives under `honeypot/`.

## Memory note

The pipeline loads all candidates into memory when building the sample. Fine for `sample1k.json` (~1k records). For the full 100k JSONL pool, expect higher RAM usage or extend with streaming in a future version.
