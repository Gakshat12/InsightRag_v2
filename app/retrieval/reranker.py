"""CrossEncoder reranker with lazy loading, sigmoid scoring, and fallback handling."""
import math
from typing import List, Tuple, Dict, Any, Optional
from langchain_core.documents import Document
from app.core.config import get_settings
from app.core.logging import logger


def sigmoid(x: float) -> float:
    """Compute standard logistic sigmoid."""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


class CrossEncoderReranker:
    """Reranks candidate documents using a Cross-Encoder transformer model."""

    def __init__(self, model_name: Optional[str] = None):
        settings = get_settings()
        self.model_name = model_name or settings.RERANKER_MODEL
        self._model = None
        self._load_failed = False

    def _get_model(self):
        """Lazy loader for CrossEncoder."""
        if self._model is None and not self._load_failed:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading CrossEncoder model: {self.model_name}")
                self._model = CrossEncoder(self.model_name)
            except Exception as e:
                logger.warning(f"Could not load CrossEncoder ({self.model_name}): {e}. Using fallback scoring.")
                self._load_failed = True
        return self._model

    def rerank(
        self,
        query: str,
        documents_with_scores: List[Tuple[Document, float, Dict[str, Any]]],
        top_k: Optional[int] = None,
    ) -> List[Tuple[Document, float, float, Dict[str, Any]]]:
        """
        Rerank documents.
        Returns list of (Document, rerank_score_normalized, raw_score, metadata).
        rerank_score_normalized is in [0.0, 1.0].
        """
        if not documents_with_scores:
            return []

        model = self._get_model()
        results = []

        if model is not None:
            pairs = [[query, doc.page_content] for doc, _, _ in documents_with_scores]
            try:
                raw_scores = model.predict(pairs)
                for (doc, rrf_score, meta), raw_score in zip(documents_with_scores, raw_scores):
                    norm_score = round(float(sigmoid(float(raw_score))), 4)
                    meta_copy = dict(meta)
                    meta_copy["rrf_score"] = rrf_score
                    meta_copy["rerank_raw"] = float(raw_score)
                    results.append((doc, norm_score, float(raw_score), meta_copy))
            except Exception as e:
                logger.error(f"Inference failed with CrossEncoder: {e}. Falling back to RRF.")
                results = []

        # Fallback if model failed or unavailable: normalize RRF scores to [0, 1]
        if not results:
            max_possible_rrf = 1.0 / 60.0 + 1.0 / 60.0  # Approx 0.0333 for rank 1 in both
            for doc, rrf_score, meta in documents_with_scores:
                norm_score = min(1.0, round(rrf_score / max_possible_rrf, 4))
                meta_copy = dict(meta)
                meta_copy["rrf_score"] = rrf_score
                results.append((doc, norm_score, rrf_score, meta_copy))

        # Sort descending by normalized score
        results.sort(key=lambda item: item[1], reverse=True)
        if top_k is not None:
            results = results[:top_k]
        return results

