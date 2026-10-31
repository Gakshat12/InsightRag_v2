"""Unit tests for DocumentLoader, DocumentChunker, and IngestionPipeline."""
import os
import tempfile
import pytest
from langchain_core.documents import Document
from langchain_community.embeddings import FakeEmbeddings

from app.ingestion.loader import DocumentLoader
from app.ingestion.chunker import DocumentChunker
from app.ingestion.pipeline import IngestionPipeline


def test_document_loader_txt():
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_file = os.path.join(tmpdir, "sample.txt")
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write("Line 1: InsightRAG architecture.\nLine 2: Hybrid BM25 and FAISS search.")

        loader = DocumentLoader()
        docs = loader.load_file(sample_file)

        assert len(docs) == 1
        assert "InsightRAG" in docs[0].page_content
        assert docs[0].metadata["source"] == "sample.txt"
        assert docs[0].metadata["page"] == 1


def test_document_chunker():
    doc = Document(
        page_content="Word " * 300,
        metadata={"source": "test.txt", "page": 1}
    )
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.split_documents([doc])

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content) <= 120
        assert "chunk_id" in chunk.metadata


def test_ingestion_pipeline_build_and_persist():
    with tempfile.TemporaryDirectory() as tmpdir:
        index_dir = os.path.join(tmpdir, "indices")
        fake_embeddings = FakeEmbeddings(size=128)

        pipeline = IngestionPipeline(
            embeddings=fake_embeddings,
            index_dir=index_dir
        )

        chunks = [
            Document(page_content="Vector databases like FAISS store dense embeddings.", metadata={"source": "db.txt", "page": 1, "chunk_id": 0}),
            Document(page_content="BM25 is a classic probabilistic lexical retrieval algorithm.", metadata={"source": "lex.txt", "page": 1, "chunk_id": 1}),
        ]

        pipeline.build_and_persist(chunks)

        assert os.path.exists(os.path.join(index_dir, "faiss_index"))
        assert os.path.exists(os.path.join(index_dir, "bm25_index.pkl"))

