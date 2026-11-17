from pathlib import Path

from fastapi.testclient import TestClient

from ragservice.api import create_app
from ragservice.config import Settings
from ragservice.evaluation import evaluate, load_golden

ROOT = Path(__file__).resolve().parents[1]


def test_golden_set_quality_gate(indexed_pipeline):
    """Regression gate on the bundled corpus. A drop below these bars fails the build."""
    items = load_golden(ROOT / "eval" / "golden.jsonl")
    report = evaluate(indexed_pipeline, items, k=4)

    assert report["retrieval"]["recall_at_4"] >= 0.9, report
    assert report["retrieval"]["mrr"] >= 0.8, report
    assert report["abstention"]["correct_abstain_rate"] == 1.0, report
    assert report["abstention"]["false_abstain_rate"] <= 0.1, report
    assert report["generation"]["citation_rate"] == 1.0, report


def test_api_ingest_query_and_metrics_round_trip(make_pipeline):
    pipeline = make_pipeline()
    client = TestClient(create_app(Settings(), pipeline=pipeline))

    assert client.get("/healthz").json() == {"status": "ok", "chunks": 0}

    created = client.post(
        "/documents",
        json={
            "source": "policy.md",
            "content": "Audit logs are retained for 400 days and streamed to SIEM webhooks.",
        },
    )
    assert created.status_code == 201
    assert created.json()["chunks"] == 1

    answer = client.post("/query", json={"question": "How long are audit logs retained?"}).json()
    assert answer["abstained"] is False
    assert answer["citations"][0]["source"] == "policy.md"
    assert answer["metrics"]["cost_usd"] >= 0

    metrics = client.get("/metrics").json()
    assert metrics["requests_total"] == 1
    assert "latency_s" in metrics


def test_api_validates_input(make_pipeline):
    client = TestClient(create_app(Settings(), pipeline=make_pipeline()))
    assert client.post("/query", json={"question": "hi"}).status_code == 422
    too_many = {"question": "valid question", "top_k": 99}
    assert client.post("/query", json=too_many).status_code == 422
    bad_csv = client.post(
        "/documents", json={"source": "x.jsonl", "content": "[1,2]", "format": "jsonl"}
    )
    assert bad_csv.status_code == 422
