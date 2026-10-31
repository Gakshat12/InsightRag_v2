"""Document chunking with configurable size and overlap."""
from typing import List, Optional
from langchain_core.documents import Document

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.core.config import get_settings


class DocumentChunker:
    """Splits documents into overlapping chunks with metadata preservation."""

    def __init__(self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None):
        settings = get_settings()
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split a list of documents into chunks and assign chunk IDs."""
        chunks = self.splitter.split_documents(documents)
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = idx
            chunk.page_content = chunk.page_content.strip()
        return [c for c in chunks if c.page_content]
