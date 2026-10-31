"""Sources package with unified interface, multi-source implementations, and citation models."""
from app.sources.base import BaseSource, SourceResult
from app.sources.kb_source import KBSource
from app.sources.wikipedia_source import WikipediaSource
from app.sources.web_source import WebSearchSource
from app.sources.citation import Citation, CitationManager

__all__ = [
    "BaseSource",
    "SourceResult",
    "KBSource",
    "WikipediaSource",
    "WebSearchSource",
    "Citation",
    "CitationManager",
]
