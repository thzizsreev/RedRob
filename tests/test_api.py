"""FastAPI backend tests."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.settings import get_settings

FIXTURES = Path(__file__).parent / "_fixtures"
FIXTURES.mkdir(exist_ok=True)
FIXTURE_CSV = FIXTURES / "team_test.csv"

if not FIXTURE_CSV.exists():
    FIXTURE_CSV.write_text(
        "candidate_id,rank,score,reasoning\n"
        "CAND_0000001,1,0.95,Strong fit on retrieval\n",
        encoding="utf-8",
    )


@pytest.fixture
def client(tmp_path, monkeypatch):
    pools_root = tmp_path / "pools"
    pools_root.mkdir()
    monkeypatch.setenv("REDROB_API_POOLS_ROOT", str(pools_root))
    monkeypatch.setenv("REDROB_SYNC_JOBS", "true")
    get_settings.cache_clear()
    with TestClient(app) as test_client:
        yield test_client
    get_settings.cache_clear()


def test_health(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready(client):
    response = client.get("/api/v1/ready")
    assert response.status_code == 200
    body = response.json()
    assert "ready" in body
    assert "checks" in body


def test_create_pool(client):
    response = client.post("/api/v1/pools", json={"name": "test-pool"})
    assert response.status_code == 201
    body = response.json()
    assert body["pool_id"]
    assert body["status"] == "created"
    assert body["candidate_count"] == 0


def test_list_pools_empty(client):
    response = client.get("/api/v1/pools")
    assert response.status_code == 200
    assert response.json()["pools"] == []


def test_upload_candidates(client):
    create = client.post("/api/v1/pools", json={})
    pool_id = create.json()["pool_id"]

    sample_path = Path(__file__).resolve().parents[1] / "data" / "sample_candidates.json"
    records = json.loads(sample_path.read_text(encoding="utf-8"))[:3]
    jsonl = "\n".join(json.dumps(r) for r in records) + "\n"

    response = client.post(
        f"/api/v1/pools/{pool_id}/candidates",
        files={"file": ("candidates.jsonl", jsonl, "application/x-ndjson")},
    )
    assert response.status_code == 200
    assert response.json()["candidate_count"] == 3

    pool = client.get(f"/api/v1/pools/{pool_id}")
    assert pool.json()["candidate_count"] == 3


def test_index_job_queued(client):
    create = client.post("/api/v1/pools", json={})
    pool_id = create.json()["pool_id"]

    sample_path = Path(__file__).resolve().parents[1] / "data" / "sample_candidates.json"
    records = json.loads(sample_path.read_text(encoding="utf-8"))[:1]
    jsonl = json.dumps(records[0]) + "\n"
    client.post(
        f"/api/v1/pools/{pool_id}/candidates",
        files={"file": ("candidates.jsonl", jsonl, "application/x-ndjson")},
    )

    with patch("backend.api.routes.pools.check_readiness") as mock_ready:
        mock_ready.return_value = type(
            "R",
            (),
            {"checks": {"instructor_onnx": True, "cuda_available": True}},
        )()
        with patch("backend.services.index_service.run_index") as mock_index:
            mock_index.return_value = {"candidate_count": 1}

            response = client.post(f"/api/v1/pools/{pool_id}/index")
            assert response.status_code == 202
            job_id = response.json()["job_id"]

            job = None
            for _ in range(50):
                job = client.get(f"/api/v1/jobs/{job_id}").json()
                if job["status"] in ("completed", "failed"):
                    break
                time.sleep(0.05)

            assert job is not None
            assert job["status"] == "completed"
            mock_index.assert_called_once()


def test_ranking_job_with_mocks(client):
    create = client.post("/api/v1/pools", json={})
    pool_id = create.json()["pool_id"]

    with patch("backend.api.routes.rankings.check_readiness") as mock_ready:
        mock_ready.return_value = type(
            "R",
            (),
            {"ready": True, "checks": {"instructor_onnx": True, "cross_encoder_onnx": True}},
        )()
        with patch("backend.services.ranking_service.run_ranking") as mock_rank:
            mock_rank.return_value = {
                "final_csv_path": str(FIXTURE_CSV),
                "total_elapsed_seconds": 1.0,
                "timings": [{"stage": 1, "label": "filter", "elapsed_seconds": 0.1}],
                "stage1_filtered": 10,
                "stage2_survivors": 8,
                "stage3_output": 5,
                "stage4_output": 5,
            }

            response = client.post("/api/v1/rankings", json={"pool_id": pool_id})
            assert response.status_code == 202
            job_id = response.json()["job_id"]

            job = None
            for _ in range(50):
                job = client.get(f"/api/v1/jobs/{job_id}").json()
                if job["status"] in ("completed", "failed"):
                    break
                time.sleep(0.05)

            assert job is not None
            assert job["status"] == "completed"

            results = client.get(f"/api/v1/rankings/{job_id}/results")
            assert results.status_code == 200
            assert len(results.json()["items"]) == 1
