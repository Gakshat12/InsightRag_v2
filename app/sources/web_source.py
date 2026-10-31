"""Web search retrieval source using Tavily API with retries and fallback."""
import os
from typing import List, Optional
import httpx
from app.sources.base import BaseSource, SourceResult
from app.core.config import get_settings
from app.core.logging import logger


class WebSearchSource(BaseSource):
    """Retrieval source for live Web Search powered by Tavily."""

    API_URL = "https://api.tavily.com/search"

    def __init__(self, api_key: Optional[str] = None, timeout: float = 6.0, max_retries: int = 2):
        super().__init__(name="TavilyWeb", source_type="web", timeout=timeout, max_retries=max_retries)
        settings = get_settings()
        self.api_key = api_key or getattr(settings, "TAVILY_API_KEY", "") or os.getenv("TAVILY_API_KEY", "")

    def _search_implementation(self, query: str, top_k: int) -> List[SourceResult]:
        if not self.api_key:
            logger.warning("TAVILY_API_KEY is not set. WebSearchSource returning empty fallback.")
            return []

        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": max(1, min(top_k, 5)),
            "include_answer": False,
        }

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(self.API_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()

            results: List[SourceResult] = []
            for item in data.get("results", []):
                snippet = item.get("content", "").strip()
                title = item.get("title", "Web Result").strip()
                url = item.get("url", "")
                if snippet:
                    results.append(
                        SourceResult(
                            content=snippet,
                            source_type=self.source_type,
                            title=title,
                            url=url,
                            metadata={"score": item.get("score", 0.0)}
                        )
                    )
            return results

