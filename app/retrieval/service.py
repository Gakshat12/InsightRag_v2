"""Retrieval service orchestrating hybrid search and optional CrossEncoder reranking."""
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from app.core.config import get_settings, Settings
from app.core.logging import RequestStageTimer
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker


class RetrievalResult:
    """Standardized retrieved chunk representation."""
    def __init__(
        self,
        document: Document,
        score: float,
        rank: int,
        metadata: Dict[str, Any],
        raw_score: Optional[float] = None
    ):
        self.document = document
        self.score = score  # Normalized confidence relevance score in [0.0, 1.0]
        self.rank = rank
        self.metadata = metadata
        self.raw_score = raw_score


class RetrievalService:
    """SOLID Retrieval coordinator with injected retriever and reranker dependencies."""

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or get_settings()
        self.retriever = retriever or HybridRetriever()
        self.reranker = reranker or (CrossEncoderReranker() if self.settings.ENABLE_RERANKER else None)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        enable_rerank: Optional[bool] = None,
        latencies: Optional[Dict[str, float]] = None,
    ) -> List[RetrievalResult]:
        """Perform hybrid retrieval and optional reranking."""
        target_k = top_k if top_k is not None else self.settings.TOP_K
        do_rerank = enable_rerank if enable_rerank is not None else self.settings.ENABLE_RERANKER

        # Stage 1: Hybrid BM25 + FAISS
        with RequestStageTimer("hybrid_retrieval", latencies):
            # Fetch candidates (e.g. 2x top_k) for reranker if enabled
            fetch_k = target_k * 2 if do_rerank else target_k
            candidate_docs = self.retriever.retrieve(query, top_k=fetch_k)

        if not candidate_docs:
            return []

        # Stage 2: CrossEncoder Reranking
        results: List[RetrievalResult] = []
        if do_rerank and self.reranker:
            with RequestStageTimer("cross_encoder_rerank", latencies):
                reranked = self.reranker.rerank(query, candidate_docs, top_k=target_k)
                for rank, (doc, norm_score, raw_score, meta) in enumerate(reranked, start=1):
                    results.append(
                        RetrievalResult(
                            document=doc,
                            score=norm_score,
                            rank=rank,
                            metadata=meta,
                            raw_score=raw_score
                        )
                    )
        else:
            # Without reranker, normalize RRF score
            max_rrf = 1.0 / 60.0 + 1.0 / 60.0
            for rank, (doc, rrf_score, meta) in enumerate(candidate_docs[:target_k], start=1):
                norm_score = min(1.0, round(rrf_score / max_rrf, 4))
                results.append(
                    RetrievalResult(
                        document=doc,
                        score=norm_score,
                        rank=rank,
                        metadata=meta,
                        raw_score=rrf_score
                    )
                )

        return results

