"""Private Knowledge Base Source wrapping HybridRetriever."""
from typing import List, Optional, Any
from app.sources.base import BaseSource, SourceResult


class KBSource(BaseSource):
    """Retrieval source for the local private FAISS + BM25 knowledge base."""

    def __init__(self, retrieval_service: Optional[Any] = None):
        super().__init__(name="PrivateKB", source_type="kb", timeout=3.0, max_retries=1)
        if retrieval_service is not None:
            self.retrieval_service = retrieval_service
        else:
            from app.retrieval.service import RetrievalService
            self.retrieval_service = RetrievalService()

    def _search_implementation(self, query: str, top_k: int) -> List[SourceResult]:
        # Retrieve candidate chunks (reranking will be unified at aggregator stage)
        results = self.retrieval_service.retrieve(query=query, top_k=top_k, enable_rerank=False)
        converted: List[SourceResult] = []
        for r in results:
            doc = r.document
            title = doc.metadata.get("source", "kb_doc")
            page = doc.metadata.get("page", 1)
            meta = dict(doc.metadata)
            meta["raw_score"] = r.score
            converted.append(
                SourceResult(
                    content=doc.page_content,
                    source_type=self.source_type,
                    title=f"{title} (p.{page})",
                    url=None,
                    metadata=meta,
                )
            )
        return converted
