# RedRob REST API

FastAPI backend wrapping the RedRob ranking pipeline (Stage 0 index + Stages 1–5 ranking).

## Prerequisites

One-time model export (same as CLI pipeline):

```bash
cd onnx && python export_to_onnx.py
python tracks/instructor/stage0/run_cross_encoder.py
```

Indexing requires **CUDA** (`onnxruntime-gpu`). Ranking Stages 1–5 use CPU cross-encoder ONNX.

## Install

```bash
pip install -r requirements.txt
```

## Run

From the repo root:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

OpenAPI docs: http://localhost:8000/docs

## End-to-end workflow

### 1. Check readiness

```bash
curl http://localhost:8000/api/v1/ready
```

### 2. Create a pool

```bash
curl -X POST http://localhost:8000/api/v1/pools \
  -H "Content-Type: application/json" \
  -d '{"name": "my-pool"}'
```

Save `pool_id` from the response.

### 3. Upload candidates (JSONL)

```bash
curl -X POST "http://localhost:8000/api/v1/pools/{pool_id}/candidates" \
  -F "file=@data/sample5k.jsonl"
```

For JSON arrays, convert to JSONL first. Each line must include a valid `candidate_id` (`CAND_XXXXXXX`).

### 4. Index pool (Stage 0 + cluster) — async job

```bash
curl -X POST "http://localhost:8000/api/v1/pools/{pool_id}/index"
```

Poll job status:

```bash
curl "http://localhost:8000/api/v1/jobs/{job_id}"
```

### 5. Run ranking (Stages 1–5) — async job

```bash
curl -X POST http://localhost:8000/api/v1/rankings \
  -H "Content-Type: application/json" \
  -d '{"pool_id": "{pool_id}"}'
```

### 6. Fetch results

JSON:

```bash
curl "http://localhost:8000/api/v1/rankings/{job_id}/results"
```

CSV download:

```bash
curl -O "http://localhost:8000/api/v1/rankings/{job_id}/results.csv"
```

## API routes

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/health` | Liveness |
| GET | `/api/v1/ready` | ONNX / config readiness |
| POST | `/api/v1/pools` | Create pool |
| GET | `/api/v1/pools` | List pools |
| GET | `/api/v1/pools/{pool_id}` | Pool metadata |
| POST | `/api/v1/pools/{pool_id}/candidates` | Upload JSONL |
| POST | `/api/v1/pools/{pool_id}/index` | Start indexing job |
| POST | `/api/v1/rankings` | Start ranking job |
| GET | `/api/v1/jobs/{job_id}` | Job status |
| GET | `/api/v1/rankings/{job_id}/results` | Top-N JSON |
| GET | `/api/v1/rankings/{job_id}/results.csv` | CSV download |

## Artifact layout

Pool data is stored under:

```
artifacts/api/pools/{pool_id}/
├── candidates.jsonl
├── pool_meta.json
├── stage0/ … stage5/
└── jobs/
```

Override root via env: `REDROB_API_POOLS_ROOT=/path/to/pools`

## MVP limitations

- Jobs are stored **in memory** — lost on server restart
- Only **one GPU-heavy job** runs at a time (process lock)
- Config is read from repo [`config.yaml`](../config.yaml) (optional override in ranking request)
- No authentication

## Tests

```bash
pytest tests/test_api.py -v
```
