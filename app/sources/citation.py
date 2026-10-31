"""Citation extraction and source reference formatting with multi-source labels and URLs."""
import re
from typing import List, Any, Set, Optional
from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Structured representation of a source citation with type label and optional URL."""
    id: int = Field(description="Citation index referenced in the answer (e.g. 1 for [1])")
    source: str = Field(description="Source document title or filename")
    source_type: str = Field(default="kb", description="Origin source category: 'kb', 'wikipedia', or 'web'")
    url: Optional[str] = Field(None, description="Direct URL if available (web/wikipedia)")
    page: Optional[int] = Field(default=None, description="Page number for KB documents")
    snippet: str = Field(description="Excerpt text from the passage")
    text: Optional[str] = Field(default=None, description="Full passage text")
    score: float = Field(description="Relevance or reranker score of this chunk")


class CitationManager:
    """Manages numbering of retrieved chunks and links citations in LLM responses."""

    @staticmethod
    def build_context_block(chunks: List[Any]) -> str:
        """
        Format retrieved chunks into a numbered context block for LLM prompt.
        Format:
        [1] Source: [wikipedia] Title (URL: https://...)
        Content...
        """
        context_parts = []
        for idx, item in enumerate(chunks, start=1):
            doc = getattr(item, "document", item)
            source = doc.metadata.get("source", "doc")
            source_type = doc.metadata.get("source_type", "kb")
            url = doc.metadata.get("url")
            url_str = f" | URL: {url}" if url else ""
            page = doc.metadata.get("page")
            page_str = f" (Page {page})" if page else ""

            content = doc.page_content.strip()
            context_parts.append(f"[{idx}] Source: [{source_type}] {source}{page_str}{url_str}\n{content}")
        return "\n\n".join(context_parts)

    @staticmethod
    def extract_cited_sources(answer: str, chunks: List[Any]) -> List[Citation]:
        """
        Parse answer for citations like [1], [2] and map them to Citation objects.
        If no citations are explicitly mentioned in answer, return top retrieved chunks as context citations.
        """
        found_ids: Set[int] = set()
        matches = re.findall(r"\[(\d+)\]", answer)
        for m in matches:
            found_ids.add(int(m))

        citations: List[Citation] = []
        for idx, item in enumerate(chunks, start=1):
            if not found_ids or idx in found_ids:
                doc = getattr(item, "document", item)
                score = getattr(item, "score", 0.0)
                source = doc.metadata.get("source", "unknown")
                source_type = doc.metadata.get("source_type", "kb")
                url = doc.metadata.get("url")
                page = doc.metadata.get("page")
                snippet = doc.page_content[:250].replace("\n", " ").strip() + "..."

                citations.append(
                    Citation(
                        id=idx,
                        source=source,
                        source_type=source_type,
                        url=url,
                        page=page,
                        snippet=snippet,
                        text=doc.page_content.strip(),
                        score=round(float(score), 4),
                    )
                )

        return citations
