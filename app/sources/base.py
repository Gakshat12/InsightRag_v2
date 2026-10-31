"""Base Source interface and standardized SourceResult model."""
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable
from pydantic import BaseModel, Field
from app.core.logging import logger


class SourceResult(BaseModel):
    """Standardized result returned by any retrieval source."""
    content: str = Field(..., description="Text content/passage retrieved")
    source_type: str = Field(..., description="Label identifying source origin: 'kb', 'wikipedia', or 'web'")
    title: str = Field(..., description="Title or filename of the document")
    url: Optional[str] = Field(None, description="Direct URL if available (web/wikipedia)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context metadata")


class BaseSource(ABC):
    """Abstract Base Class for all retrieval sources."""

    def __init__(self, name: str, source_type: str, timeout: float = 5.0, max_retries: int = 2, backoff_factor: float = 0.5):
        self.name = name
        self.source_type = source_type
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    @abstractmethod
    def _search_implementation(self, query: str, top_k: int) -> List[SourceResult]:
        """Internal search implementation to be overridden by subclasses."""
        pass

    def search(self, query: str, top_k: int = 5) -> List[SourceResult]:
        """
        Execute search with timeout, retry backoff, and graceful fallback.
        Never crashes the caller if a source fails or returns empty.
        """
        if not query or not query.strip():
            return []

        attempt = 0
        while attempt <= self.max_retries:
            try:
                results = self._search_implementation(query.strip(), top_k=top_k)
                return results or []
            except Exception as e:
                attempt += 1
                wait_time = self.backoff_factor * (2 ** (attempt - 1))
                if attempt <= self.max_retries:
                    logger.warning(
                        f"Source '{self.name}' attempt {attempt} failed ({e}). Retrying in {wait_time:.2f}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"Source '{self.name}' failed after {self.max_retries + 1} attempts: {e}. Graceful fallback to empty."
                    )
                    return []
        return []

