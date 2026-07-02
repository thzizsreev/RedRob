# RedRob Docker Sandbox

CPU sandbox: **ranks the full 1K pool** through stages 1–5, writes a **top-100** submission CSV.

Stage 0 is baked offline (`fetch_pool1k_artifacts.py` — id_map + FAISS reconstruct from full-pool artifacts).

## Setup (before `docker build`)

```powershell
python docker/scripts/sample_pool1k.py --source data/candidates.jsonl --seed 42
python docker/scripts/make_sandbox_config.py
python docker/scripts/fetch_pool1k_artifacts.py
```

## Local run (default = full 1K)

```powershell
$env:REDROB_CPU_ONLY = "1"
python docker/scripts/run_demo.py
# → ./SignalHunters.csv in repo root (top 100, ranking only — 3 columns)
```

Optional upload (≤100 IDs, subset of pool1k only):

```powershell
python docker/scripts/run_demo.py --input path/to/candidates.jsonl
```

## Docker

Mount **repo root** to `/output` so the CSV is written beside `README.md`:

```powershell
docker build -f docker/Dockerfile -t redrob-sandbox .
docker run --rm -v "${PWD}:/output" redrob-sandbox
```

**Organizer demo:** default mode ranks the baked 1K pool. For ≤100 sample input (spec §10.5), use upload mode — avoid thin uploads that trigger padding to 100 rows.

```powershell
docker run --rm -v "${PWD}:/input" -v "${PWD}:/output" redrob-sandbox --input /input/candidates.jsonl
```

Full 100K reproduction uses root [`rank.py`](../rank.py), not the Docker image.

## Runtime flow

```
pool1k artifacts (1000) → stages 1–5 on all 1000 → stage5 top_n=100 → SignalHunters.csv
```

Upload mode subsets artifacts to the uploaded IDs first; default mode uses the full baked 1K index unchanged.
