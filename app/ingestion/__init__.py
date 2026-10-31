"""Ingestion module for document loading, chunking, and dual-indexing."""
from app.ingestion.loader import DocumentLoader
from app.ingestion.chunker import DocumentChunker
from app.ingestion.pipeline import (
    IngestionPipeline,
    LocalSentenceTransformerEmbeddings,
    get_default_embeddings,
)

__all__ = [
    "DocumentLoader",
    "DocumentChunker",
    "IngestionPipeline",
    "LocalSentenceTransformerEmbeddings",
    "get_default_embeddings",
]
