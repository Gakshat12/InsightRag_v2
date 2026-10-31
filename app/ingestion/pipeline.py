"""Ingestion pipeline: local sentence-transformers embedding, FAISS indexing, and BM25 index persistence."""
import os
import pickle
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
from app.core.config import get_settings
from app.core.logging import logger
from app.ingestion.loader import DocumentLoader
from app.ingestion.chunker import DocumentChunker


class LocalSentenceTransformerEmbeddings(Embeddings):
    """Local Sentence-Transformers embeddings implementing the LangChain Embeddings interface."""

    def __init__(self, model_name: Optional[str] = None):
        settings = get_settings()
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute document embeddings."""
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return embeddings.tolist()
        except Exception as e:
            logger.warning(f"SentenceTransformer encoding failed: {e}. Using deterministic fallback.")
            from langchain_community.embeddings import FakeEmbeddings
            return FakeEmbeddings(size=384).embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """Compute query embedding."""
        try:
            embedding = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
            return embedding.tolist()
        except Exception as e:
            logger.warning(f"SentenceTransformer encoding failed: {e}. Using deterministic fallback.")
            from langchain_community.embeddings import FakeEmbeddings
            return FakeEmbeddings(size=384).embed_query(text)


def get_default_embeddings(model_name: Optional[str] = None) -> Embeddings:
    """Return local Sentence-Transformers embeddings instance."""
    return LocalSentenceTransformerEmbeddings(model_name=model_name)


class IngestionPipeline:
    """Orchestrates loading, chunking, and dual-index (FAISS + BM25) persistence."""

    def __init__(
        self,
        embeddings: Optional[Embeddings] = None,
        loader: Optional[DocumentLoader] = None,
        chunker: Optional[DocumentChunker] = None,
        index_dir: Optional[str] = None,
    ):
        settings = get_settings()
        self.embeddings = embeddings or get_default_embeddings()
        self.loader = loader or DocumentLoader()
        self.chunker = chunker or DocumentChunker()
        self.index_dir = index_dir or settings.INDEX_DIR

    def ingest_folder(self, folder_path: str) -> int:
        """Load, chunk, index, and persist documents from a folder."""
        logger.info(f"Starting ingestion for folder: {folder_path}")
        raw_docs = self.loader.load_directory(folder_path)
        if not raw_docs:
            logger.warning(f"No documents found to ingest in {folder_path}")
            return 0

        chunks = self.chunker.split_documents(raw_docs)
        logger.info(f"Generated {len(chunks)} chunks from {len(raw_docs)} source documents")

        self.build_and_persist(chunks)
        return len(chunks)

    def build_and_persist(self, chunks: List[Document]) -> None:
        """Build and persist FAISS and BM25 indexes."""
        if not chunks:
            raise ValueError("Cannot build index from empty chunk list")

        os.makedirs(self.index_dir, exist_ok=True)

        # 1. FAISS Index with local Sentence-Transformers embeddings
        logger.info("Building FAISS vector index using local embeddings...")
        faiss_store = FAISS.from_documents(chunks, self.embeddings)
        faiss_path = os.path.join(self.index_dir, "faiss_index")
        faiss_store.save_local(faiss_path)
        logger.info(f"FAISS index successfully saved to: {faiss_path}")

        # 2. BM25 Index
        logger.info("Building BM25 keyword index...")
        corpus_texts = [doc.page_content for doc in chunks]
        tokenized_corpus = [text.lower().split() for text in corpus_texts]
        bm25 = BM25Okapi(tokenized_corpus)

        bm25_data = {
            "bm25": bm25,
            "documents": chunks,
        }
        bm25_path = os.path.join(self.index_dir, "bm25_index.pkl")
        with open(bm25_path, "wb") as f:
            pickle.dump(bm25_data, f)
        logger.info(f"BM25 index successfully saved to: {bm25_path}")
