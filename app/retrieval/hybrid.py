"""Hybrid retrieval combining FAISS vector search and BM25 with Reciprocal Rank Fusion (RRF)."""
import os
import pickle
from typing import List, Tuple, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
from app.core.config import get_settings
from app.core.logging import logger
from app.ingestion.pipeline import get_default_embeddings


class HybridRetriever:
    """Combines dense vector retrieval (FAISS) with sparse keyword retrieval (BM25) via RRF."""

    def __init__(
        self,
        index_dir: Optional[str] = None,
        embeddings: Optional[Embeddings] = None,
        rrf_k: Optional[int] = None,
    ):
        settings = get_settings()
        self.index_dir = index_dir or settings.INDEX_DIR
        self.embeddings = embeddings or get_default_embeddings()
        self.rrf_k = rrf_k if rrf_k is not None else settings.RRF_K

        self.faiss_store: Optional[FAISS] = None
        self.bm25: Optional[BM25Okapi] = None
        self.corpus_documents: List[Document] = []
        self._loaded = False

    def load_indices(self) -> bool:
        """Load persisted FAISS and BM25 indices from disk."""
        faiss_path = os.path.join(self.index_dir, "faiss_index")
        bm25_path = os.path.join(self.index_dir, "bm25_index.pkl")

        if not os.path.exists(faiss_path) or not os.path.exists(bm25_path):
            logger.warning(f"Indices not found in {self.index_dir}")
            return False

        try:
            # Load FAISS
            self.faiss_store = FAISS.load_local(
                faiss_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )

            # Load BM25
            with open(bm25_path, "rb") as f:
                bm25_data = pickle.load(f)
            self.bm25 = bm25_data["bm25"]
            self.corpus_documents = bm25_data["documents"]

            self._loaded = True
            logger.info(f"Loaded FAISS index and {len(self.corpus_documents)} BM25 corpus documents")
            return True
        except Exception as e:
            logger.error(f"Error loading indices: {e}")
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def search_bm25(self, query: str, top_k: int) -> List[Tuple[Document, int, float]]:
        """Perform BM25 search returning list of (document, rank, raw_score)."""
        if not self.bm25 or not self.corpus_documents:
            return []

        tokens = query.lower().split()
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            if scores[idx] > 0.0:
                results.append((self.corpus_documents[idx], rank, float(scores[idx])))
        return results

    def search_faiss(self, query: str, top_k: int) -> List[Tuple[Document, int, float]]:
        """Perform dense vector search returning list of (document, rank, distance/score)."""
        if not self.faiss_store:
            return []

        # search with distance score
        docs_and_scores = self.faiss_store.similarity_search_with_score(query, k=top_k)
        results = []
        for rank, (doc, score) in enumerate(docs_and_scores, start=1):
            results.append((doc, rank, float(score)))
        return results

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[Document, float, Dict[str, Any]]]:
        """
        Execute hybrid search and merge using Reciprocal Rank Fusion (RRF).
        RRF formula: RRF_score(d) = sum(1 / (k + rank_i(d)))
        """
        if not self._loaded and not self.load_indices():
            logger.warning("Retriever indices not available for search.")
            return []

        # Retrieve top candidates from both paths
        candidate_multiplier = 2
        fetch_k = max(top_k * candidate_multiplier, 10)

        bm25_results = self.search_bm25(query, fetch_k)
        faiss_results = self.search_faiss(query, fetch_k)

        # Unique document tracking by content + source
        doc_map: Dict[str, Document] = {}
        rrf_scores: Dict[str, float] = {}
        meta_scores: Dict[str, Dict[str, Any]] = {}

        def get_doc_id(doc: Document) -> str:
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", 0)
            chunk_id = doc.metadata.get("chunk_id", hash(doc.page_content[:50]))
            return f"{source}:{page}:{chunk_id}"

        # 1. Process BM25 ranks
        for doc, rank, score in bm25_results:
            doc_id = get_doc_id(doc)
            doc_map[doc_id] = doc
            reciprocal_rank = 1.0 / (self.rrf_k + rank)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + reciprocal_rank
            if doc_id not in meta_scores:
                meta_scores[doc_id] = {}
            meta_scores[doc_id]["bm25_rank"] = rank
            meta_scores[doc_id]["bm25_score"] = score

        # 2. Process FAISS ranks
        for doc, rank, score in faiss_results:
            doc_id = get_doc_id(doc)
            doc_map[doc_id] = doc
            reciprocal_rank = 1.0 / (self.rrf_k + rank)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + reciprocal_rank
            if doc_id not in meta_scores:
                meta_scores[doc_id] = {}
            meta_scores[doc_id]["faiss_rank"] = rank
            meta_scores[doc_id]["faiss_score"] = score

        # Sort documents by RRF score descending
        sorted_ids = sorted(rrf_scores.keys(), key=lambda d_id: rrf_scores[d_id], reverse=True)
        top_ids = sorted_ids[:top_k]

        results = []
        for doc_id in top_ids:
            results.append((doc_map[doc_id], round(rrf_scores[doc_id], 6), meta_scores[doc_id]))

        return results

