"""Wikipedia retrieval source using MediaWiki search API with passage splitting and INFO logs."""
import urllib.parse
from typing import List

import httpx

from app.sources.base import BaseSource, SourceResult
from app.core.logging import logger


class WikipediaSource(BaseSource):
    """Retrieval source for Wikipedia using srsearch + intro extracts split into passages."""

    USER_AGENT = "InsightRAG/3.0 (https://github.com/example/insightrag; dev@insightrag.local)"
    API_URL = "https://en.wikipedia.org/w/api.php"

    def __init__(self, timeout: float = 8.0, max_retries: int = 2):
        super().__init__(name="Wikipedia", source_type="wikipedia", timeout=timeout, max_retries=max_retries)

    def _search_implementation(self, query: str, top_k: int) -> List[SourceResult]:
        headers = {"User-Agent": self.USER_AGENT}
        logger.info(f"[WikipediaSource] Searching with query: '{query}'")

        with httpx.Client(timeout=self.timeout, headers=headers) as client:
            # Step 1: MediaWiki srsearch – top 2 titles
            search_resp = client.get(self.API_URL, params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": 2,
                "format": "json",
            })
            search_resp.raise_for_status()
            hits = search_resp.json().get("query", {}).get("search", [])
            titles = [h["title"] for h in hits]
            logger.info(f"[WikipediaSource] Titles returned: {titles}")

            if not titles:
                logger.info("[WikipediaSource] No titles found, returning empty.")
                return []

            # Step 2: Fetch full plain-text extract per title and split into ~800-char passages (cap at 30)
            results: List[SourceResult] = []
            for title in titles:
                extract_resp = client.get(self.API_URL, params={
                    "action": "query",
                    "prop": "extracts",
                    "explaintext": "1",
                    "titles": title,
                    "format": "json",
                })
                extract_resp.raise_for_status()
                pages = extract_resp.json().get("query", {}).get("pages", {})
                page = next(iter(pages.values()), {})
                extract = page.get("extract", "").strip()

                if not extract:
                    logger.info(f"[WikipediaSource] Title='{title}': empty extract, skipping.")
                    continue

                # Strip headings and combine clean lines
                lines = [
                    line.strip() for line in extract.split("\n")
                    if line.strip() and not line.strip().startswith("==")
                ]
                clean_text = "\n\n".join(lines)

                # Split into ~800-character passages, capped at 30 per page
                passages: List[str] = []
                chunk_target = 800
                start = 0
                text_len = len(clean_text)

                while start < text_len and len(passages) < 30:
                    end = min(start + chunk_target, text_len)
                    if end < text_len:
                        break_point = clean_text.rfind("\n\n", start + 300, end)
                        if break_point == -1:
                            break_point = clean_text.rfind(". ", start + 300, end)
                            if break_point != -1:
                                break_point += 1
                        if break_point == -1:
                            break_point = clean_text.rfind(" ", start + 300, end)
                        if break_point != -1:
                            end = break_point

                    passage = clean_text[start:end].strip()
                    if len(passage) > 30:
                        passages.append(passage)
                    start = end

                logger.info(f"[WikipediaSource] Title='{title}' passages returned: {len(passages)}")

                safe_title = urllib.parse.quote(title.replace(" ", "_"))
                url = f"https://en.wikipedia.org/wiki/{safe_title}"

                for passage in passages:
                    results.append(SourceResult(
                        content=passage,
                        source_type=self.source_type,
                        title=title,
                        url=url,
                        metadata={"wikipedia_title": title},
                    ))

            logger.info(f"[WikipediaSource] Total passages returned: {len(results)}")
            return results
