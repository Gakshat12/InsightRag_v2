"""CLI script to ingest documents from a folder into FAISS and BM25 indexes."""
import sys
import argparse
from app.core.logging import logger
from app.ingestion.loader import DocumentLoader
from app.ingestion.chunker import DocumentChunker
from app.ingestion.pipeline import IngestionPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Ingest PDF/TXT documents into FAISS and BM25 indices for InsightRAG."
    )
    parser.add_argument("folder", type=str, help="Path to directory containing PDF/TXT files")
    parser.add_argument("--chunk-size", type=int, default=None, help="Size of document chunks")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="Overlap between chunks")
    parser.add_argument("--index-dir", type=str, default=None, help="Output directory for index files")

    args = parser.parse_args()

    loader = DocumentLoader()
    chunker = DocumentChunker(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    pipeline = IngestionPipeline(loader=loader, chunker=chunker, index_dir=args.index_dir)

    try:
        count = pipeline.ingest_folder(args.folder)
        logger.info(f"Successfully indexed {count} chunks from '{args.folder}'")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

