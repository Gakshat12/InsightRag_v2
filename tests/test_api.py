"""Integration tests for FastAPI endpoints: /health, /metrics, and /ask with mocked services."""
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.api.main import app, generator
from app.retrieval.service import RetrievalResult


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "reranker_enabled" in data
    assert "routing_mode" in data


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "insightrag_requests_total" in response.text


def test_ask_empty_query(client):
    response = client.post("/ask", json={"query": "   "})
    assert response.status_code == 400


def test_ask_json_mocked(client, monkeypatch):
    monkeypatch.setattr(generator, "chain", None)

    sample_doc = Document(
        page_content="InsightRAG leverages Reciprocal Rank Fusion for dense and sparse search.",
        metadata={"source": "whitepaper.pdf", "source_type": "kb", "page": 1, "chunk_id": 0}
    )
    mock_retrieval = [
        RetrievalResult(
            document=sample_doc,
            score=0.92,
            rank=1,
            metadata={"bm25_rank": 1, "faiss_rank": 1}
        )
    ]
    mock_meta = {
        "routing_mode": "cascade_router",
        "sources_queried": ["kb"],
        "router_decision": "kb_sufficient",
        "per_source_latency_ms": {"kb": 12.5},
    }

    monkeypatch.setattr(
        generator.orchestrator,
        "orchestrate",
        lambda query, routing_mode=None, top_k=None: (mock_retrieval, mock_meta)
    )

    response = client.post(
        "/ask",
        json={"query": "What does InsightRAG leverage?", "top_k": 3, "rerank": False, "stream": False}
    )

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "citations" in data
    assert "confidence" in data
    assert data["confidence"]["score"] > 0.0
    assert "routing_mode" in data
    assert "sources_queried" in data
    assert "stage_latency_ms" in data
    assert "estimated_cost_usd" in data


def test_ask_streaming_mocked(client, monkeypatch):
    monkeypatch.setattr(generator, "chain", None)

    sample_doc = Document(
        page_content="FastAPI supports high-performance streaming responses via SSE.",
        metadata={"source": "api.txt", "source_type": "web", "page": 1, "chunk_id": 0}
    )
    mock_retrieval = [
        RetrievalResult(
            document=sample_doc,
            score=0.88,
            rank=1,
            metadata={}
        )
    ]
    mock_meta = {
        "routing_mode": "fan_out",
        "sources_queried": ["kb", "wikipedia", "web"],
        "router_decision": "fan_out_all",
        "per_source_latency_ms": {"kb": 10.0, "wikipedia": 15.0, "web": 20.0},
    }

    monkeypatch.setattr(
        generator.orchestrator,
        "orchestrate",
        lambda query, routing_mode=None, top_k=None: (mock_retrieval, mock_meta)
    )

    response = client.post(
        "/ask",
        json={"query": "How does FastAPI stream?", "stream": True}
    )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    body = response.text
    assert "data: " in body
    assert "[DONE]" in body
