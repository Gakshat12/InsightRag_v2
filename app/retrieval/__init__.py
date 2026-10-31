"""Retrieval package combining BM25, FAISS, CrossEncoder reranking, and multi-source routing."""
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker, sigmoid
from app.retrieval.service import RetrievalService, RetrievalResult
from app.retrieval.router import SourceOrchestrator, RouterChoice

__all__ = [
    "HybridRetriever",
    "CrossEncoderReranker",
    "sigmoid",
    "RetrievalService",
    "RetrievalResult",
    "SourceOrchestrator",
    "RouterChoice",
]
