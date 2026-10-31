"""Multi-source routing (fan_out and cascade_router) and cross-source reranking."""
import json
import time
from typing import List, Dict, Any, Tuple, Optional, Literal
from pydantic import BaseModel, Field
from langchain_core.documents import Document

from app.core.config import get_settings, Settings
from app.core.logging import logger
from app.core.metrics import SOURCE_QUERIES, SOURCE_FALLBACKS, ROUTER_DECISIONS
from app.core.llm_provider import LLMProvider
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.service import RetrievalResult
from app.sources.base import BaseSource, SourceResult


class RouterChoice(BaseModel):
    """Structured decision output from Gemini router."""
    selected_sources: List[Literal["wikipedia", "web"]] = Field(
        default_factory=list,
        description="List containing 'wikipedia', 'web', or both"
    )
    search_query: str = Field(
        default="",
        description="Concise keyword search query for external sources, e.g. 'B-tree'"
    )
    reasoning: str = Field(default="", description="Short reason for routing decision")


class SourceOrchestrator:
    """Orchestrates multi-source retrieval, routing modes, and CrossEncoder reranking."""

    def __init__(
        self,
        kb_source: Optional[BaseSource] = None,
        wikipedia_source: Optional[BaseSource] = None,
        web_source: Optional[BaseSource] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        llm_provider: Optional[LLMProvider] = None,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or get_settings()
        if kb_source is not None:
            self.kb_source = kb_source
        else:
            from app.sources.kb_source import KBSource
            self.kb_source = KBSource()

        if wikipedia_source is not None:
            self.wikipedia_source = wikipedia_source
        else:
            from app.sources.wikipedia_source import WikipediaSource
            self.wikipedia_source = WikipediaSource()

        if web_source is not None:
            self.web_source = web_source
        else:
            from app.sources.web_source import WebSearchSource
            self.web_source = WebSearchSource()

        self.reranker = reranker or (CrossEncoderReranker() if self.settings.ENABLE_RERANKER else None)
        self.llm_provider = llm_provider or LLMProvider(settings=self.settings)

    def _query_source(self, source: BaseSource, query: str, top_k: int) -> Tuple[List[SourceResult], float]:
        """Execute query on a single source with latency recording and Prometheus metrics."""
        start = time.perf_counter()
        SOURCE_QUERIES.labels(source=source.source_type).inc()
        results = source.search(query=query, top_k=top_k)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if not results:
            SOURCE_FALLBACKS.labels(source=source.source_type).inc()

        return results, round(elapsed_ms, 2)

    def route_external_sources_with_gemini(self, query: str, kb_context: str) -> Tuple[List[str], str]:
        """Ask Gemini with structured output which external sources to consult.

        Returns:
            (selected_sources, search_query) — keyword search_query for Wikipedia/web.
        """
        self._router_fallback = False
        llm = self.llm_provider.get_llm()
        if llm is None:
            logger.info("Router LLM not configured/offline. Defaulting to ['wikipedia', 'web'].")
            self._router_fallback = True
            return ["wikipedia", "web"], query

        prompt = (
            "You are a search query router. A private knowledge base produced insufficient confidence.\n"
            f"User Query: {query}\n"
            f"Private KB Snippet: {kb_context[:300]}\n\n"
            "Decide which external sources should be queried to answer this query.\n"
            "Options: 'wikipedia' (for encyclopedic/conceptual/historical topics), "
            "'web' (for recent/real-time/technical/news topics), or both.\n"
            "Also provide a concise keyword search_query (e.g. 'B-tree') for external sources.\n"
            "Output strictly valid JSON conforming to this schema:\n"
            '{"selected_sources": ["wikipedia" or "web"], "search_query": "string", "reasoning": "string"}'
        )

        try:
            if hasattr(llm, "with_structured_output"):
                structured_chain = llm.with_structured_output(RouterChoice)
                decision = structured_chain.invoke(prompt)
                if isinstance(decision, RouterChoice) and decision.selected_sources:
                    sq = decision.search_query.strip() or query
                    logger.info(f"[Router] search_query='{sq}', sources={decision.selected_sources}")
                    return list(decision.selected_sources), sq

            resp = llm.invoke(prompt)
            content = resp.content if hasattr(resp, "content") else str(resp)
            start_idx = content.find("{")
            end_idx = content.rfind("}")
            if start_idx != -1 and end_idx != -1:
                data = json.loads(content[start_idx:end_idx + 1])
                sources = [s for s in data.get("selected_sources", []) if s in ("wikipedia", "web")]
                sq = data.get("search_query", "").strip() or query
                if sources:
                    logger.info(f"[Router] search_query='{sq}', sources={sources}")
                    return sources, sq
        except Exception as e:
            logger.warning(f"Router Gemini call failed: {e}. Defaulting to both.")
            self._router_fallback = True

        self._router_fallback = True
        return ["wikipedia", "web"], query

    def _to_candidate_docs(self, results: List[SourceResult]) -> List[Tuple[Document, float, Dict[str, Any]]]:
        """Convert SourceResults to tuple format expected by CrossEncoder reranker."""
        candidate_docs: List[Tuple[Document, float, Dict[str, Any]]] = []
        for res in results:
            meta = {
                "source": res.title,
                "source_type": res.source_type,
                "url": res.url,
                **res.metadata
            }
            doc = Document(page_content=res.content, metadata=meta)
            base_score = float(res.metadata.get("raw_score", 0.02))
            candidate_docs.append((doc, base_score, meta))
        return candidate_docs

    def _rerank_candidates(
        self,
        query: str,
        candidate_docs: List[Tuple[Document, float, Dict[str, Any]]],
        top_k: int
    ) -> List[RetrievalResult]:
        """Rerank candidates using CrossEncoder if enabled, otherwise use base score."""
        retrieval_results: List[RetrievalResult] = []
        if self.reranker and candidate_docs:
            reranked = self.reranker.rerank(query, candidate_docs, top_k=top_k)
            for rank, (doc, norm_score, raw_score, meta) in enumerate(reranked, start=1):
                retrieval_results.append(
                    RetrievalResult(
                        document=doc,
                        score=norm_score,
                        rank=rank,
                        metadata=meta,
                        raw_score=raw_score
                    )
                )
        else:
            for rank, (doc, base_score, meta) in enumerate(candidate_docs[:top_k], start=1):
                retrieval_results.append(
                    RetrievalResult(
                        document=doc,
                        score=min(1.0, base_score),
                        rank=rank,
                        metadata=meta,
                        raw_score=base_score
                    )
                )
        return retrieval_results

    def orchestrate(
        self,
        query: str,
        routing_mode: Optional[str] = None,
        top_k: Optional[int] = None,
        answer_evaluator: Optional[Any] = None,
    ) -> Tuple[List[RetrievalResult], Dict[str, Any]]:
        """
        Orchestrate multi-source retrieval according to routing mode:
        - fan_out: queries KB, Wikipedia, and Web in parallel/sequence.
        - cascade_router: queries KB first. Evaluates KB answer confidence or retrieval score;
          if low confidence or 'I don't know', escalates to Gemini router.
        """
        mode = (routing_mode or self.settings.ROUTING_MODE).lower()
        k = top_k or self.settings.TOP_K

        per_source_latency: Dict[str, float] = {}
        sources_queried: List[str] = []
        router_decision: str = ""
        all_results: List[SourceResult] = []
        kb_score_for_log: float = 0.0
        kb_eval_data: Optional[Any] = None

        if mode == "fan_out":
            router_decision = "fan_out_all"
            ROUTER_DECISIONS.labels(decision="fan_out").inc()

            for src in [self.kb_source, self.wikipedia_source, self.web_source]:
                sources_queried.append(src.source_type)
                results, lat = self._query_source(src, query, top_k=k)
                per_source_latency[src.source_type] = lat
                all_results.extend(results)

            candidate_docs = self._to_candidate_docs(all_results)
            retrieval_results = self._rerank_candidates(query, candidate_docs, top_k=k)

        else:
            # cascade_router
            # Step 1: Query Private KB
            sources_queried.append("kb")
            kb_results, kb_lat = self._query_source(self.kb_source, query, top_k=k)
            per_source_latency["kb"] = kb_lat
            all_results.extend(kb_results)

            kb_candidate_docs = self._to_candidate_docs(kb_results)
            kb_retrieval_results = self._rerank_candidates(query, kb_candidate_docs, top_k=k)

            # Evaluate KB confidence via answer evaluator callback or retrieval score fallback
            if answer_evaluator is not None and callable(answer_evaluator):
                kb_confidence_score, is_kb_idk, kb_eval_data = answer_evaluator(kb_retrieval_results)
                is_sufficient = (not is_kb_idk) and (kb_confidence_score >= self.settings.CASCADE_KB_THRESHOLD)
                kb_score_for_log = kb_confidence_score
            else:
                kb_top_score = 0.0
                if kb_results:
                    kb_top_score = float(kb_results[0].metadata.get("raw_score", 0.0))
                is_sufficient = kb_top_score >= self.settings.CASCADE_KB_THRESHOLD and len(kb_results) >= 2
                kb_score_for_log = kb_top_score

            if is_sufficient:
                router_decision = "kb_sufficient"
                logger.info(f"Router decision: {router_decision}, KB-only confidence: {kb_score_for_log:.4f}")
                ROUTER_DECISIONS.labels(decision="kb_only").inc()
                retrieval_results = kb_retrieval_results
            else:
                # Step 2: Gemini routing decision
                kb_snippet = kb_results[0].content if kb_results else "No relevant documents found."
                selected, search_query = self.route_external_sources_with_gemini(query, kb_snippet)
                router_decision = "escalated_after_low_confidence"
                logger.info(f"Router decision: {router_decision}, KB-only confidence: {kb_score_for_log:.4f}")
                decision_label = "both" if len(selected) > 1 else (selected[0] if selected else "none")
                ROUTER_DECISIONS.labels(decision=decision_label).inc()

                if "wikipedia" in selected:
                    sources_queried.append("wikipedia")
                    wiki_results, wiki_lat = self._query_source(self.wikipedia_source, search_query, top_k=k)
                    per_source_latency["wikipedia"] = wiki_lat
                    all_results.extend(wiki_results)

                if "web" in selected:
                    sources_queried.append("web")
                    web_results, web_lat = self._query_source(self.web_source, search_query, top_k=k)
                    per_source_latency["web"] = web_lat
                    all_results.extend(web_results)

                candidate_docs = self._to_candidate_docs(all_results)
                retrieval_results = self._rerank_candidates(query, candidate_docs, top_k=k)

        wiki_count = sum(
            1 for r in retrieval_results
            if r.document.metadata.get("source_type") == "wikipedia"
        )
        logger.info(f"[Orchestrator] Wikipedia passages surviving rerank: {wiki_count}")

        metadata = {
            "routing_mode": mode,
            "sources_queried": sources_queried,
            "router_decision": router_decision,
            "kb_only_confidence": kb_score_for_log if mode == "cascade_router" else None,
            "kb_eval_data": kb_eval_data if (mode == "cascade_router" and is_sufficient) else None,
            "router_fallback": getattr(self, "_router_fallback", False) if (mode != "fan_out" and router_decision != "kb_sufficient") else False,
            "per_source_latency_ms": per_source_latency,
            "total_candidates_found": len(all_results),
        }

        return retrieval_results, metadata
