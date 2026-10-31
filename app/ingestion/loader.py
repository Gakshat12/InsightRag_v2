"""Document loading for PDF and TXT files using LangChain loaders."""
import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from app.core.logging import logger


class DocumentLoader:
    """Loads PDF and TXT documents from directories or files."""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

    def load_file(self, file_path: str) -> List[Document]:
        """Load a single document based on extension."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
        elif ext == ".pdf":
            loader = PyPDFLoader(file_path)
            docs = loader.load()
        else:
            logger.warning(f"Skipping unsupported file type: {file_path}")
            return []

        # Ensure standard metadata
        for doc in docs:
            doc.metadata["source"] = os.path.basename(file_path)
            doc.metadata["file_path"] = file_path
            if "page" not in doc.metadata:
                doc.metadata["page"] = 1
            else:
                # 1-indexed for citations
                doc.metadata["page"] = int(doc.metadata["page"]) + 1

        return docs

    def load_directory(self, directory_path: str) -> List[Document]:
        """Recursively load all supported files from a directory."""
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        documents: List[Document] = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in self.SUPPORTED_EXTENSIONS:
                    file_path = os.path.join(root, file)
                    try:
                        docs = self.load_file(file_path)
                        documents.extend(docs)
                    except Exception as e:
                        logger.error(f"Failed to load {file_path}: {e}")

        logger.info(f"Loaded {len(documents)} raw document chunks from {directory_path}")
        return documents

