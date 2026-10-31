"""Unit tests for HybridRetriever, CrossEncoderReranker, and RetrievalService."""
import os
import tempfile
import pytest
from langchain_core.documents import Document
from langchain_community.embeddings import FakeEmbeddings

from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker, sigmoid
from app.retrieval.service import RetrievalService


@pytest.fixture
def test_indices():
    with tempfile.TemporaryDirectory() as tmpdir:
        index_dir = os.path.join(tmpdir, "indices")
        embeddings = FakeEmbeddings(size=128)
        pipeline = IngestionPipeline(embeddings=embeddings, index_dir=index_dir)

        docs = [
            Document(page_content="Kubernetes is an open-source container orchestration system.", metadata={"source": "k8s.txt", "page": 1, "chunk_id": 0}),
            Document(page_content="Docker packages applications into lightweight containers.", metadata={"source": "docker.txt", "page": 1, "chunk_id": 1}),
            Document(page_content="Python is a popular language for data science and AI.", metadata={"source": "py.txt", "page": 2, "chunk_id": 2}),
        ]
        pipeline.build_and_persist(docs)
        yield index_dir, embeddings


def test_hybrid_retrieval(test_indices):
    index_dir, embeddings = test_indices
    retriever = HybridRetriever(index_dir=index_dir, embeddings=embeddings, rrf_k=60)
    assert retriever.load_indices() is True

    # Search query
    results = retriever.retrieve("container orchestration", top_k=2)
    assert len(results) > 0
    top_doc, rrf_score, meta = results[0]
    assert rrf_score > 0.0
    assert "source" in top_doc.metadata


def test_sigmoid_math():
    assert sigmoid(0.0) == 0.5
    assert sigmoid(10.0) > 0.99
    assert sigmoid(-10.0) < 0.01


def test_reranker_fallback():
    reranker = CrossEncoderReranker()
    reranker._load_failed = True  # Simulate model load failure/fallback without network wait
    doc = Document(page_content="Test document content.", metadata={"source": "test.txt", "page": 1})
    candidates = [(doc, 0.025, {"bm25_rank": 1})]

    reranked = reranker.rerank("test query", candidates, top_k=1)
    assert len(reranked) == 1
    doc_out, norm_score, raw_score, meta = reranked[0]
    assert 0.0 <= norm_score <= 1.0


def test_retrieval_service_flow(test_indices):
    index_dir, embeddings = test_indices
    retriever = HybridRetriever(index_dir=index_dir, embeddings=embeddings)
    retriever.load_indices()

    service = RetrievalService(retriever=retriever)
    service.reranker = None  # test service flow with RRF normalization directly
    latencies = {}
    results = service.retrieve("Python data science", top_k=2, enable_rerank=False, latencies=latencies)

    assert len(results) > 0
    assert "hybrid_retrieval" in latencies
    assert 0.0 <= results[0].score <= 1.0
