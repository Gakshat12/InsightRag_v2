"""Answer generation with multi-source routing, confidence checking, and streaming."""
import json
import re
from typing import List, Dict, Any, AsyncGenerator, Optional, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.core.config import get_settings, Settings
from app.core.logging import (
    logger,
    RequestStageTimer,
    calculate_cost,
    log_request_summary,
)
from app.core.metrics import (
    REQUEST_COUNT,
    STAGE_LATENCY,
    TOKEN_USAGE,
    ESTIMATED_COST,
)
from app.core.llm_provider import LLMProvider
from app.retrieval.service import RetrievalResult
from app.retrieval.router import SourceOrchestrator
from app.sources.citation import Citation, CitationManager
from app.api.schemas import ConfidenceDetail, QueryResponse

SYSTEM_PROMPT = """You are InsightRAG, an authoritative enterprise AI assistant.
Answer the user's question accurately and concisely, strictly based on the provided numbered context documents.

Rules:
1. Ground every substantive fact in the numbered source citations, e.g. [1], [2].
2. If the context does not contain sufficient information to answer the question, say clearly: "I don't know based on the available sources."
3. Do not make up facts or citation numbers that do not exist in the context.
4. Keep the answer direct and well-structured."""

USER_PROMPT = """Context documents:
{context}

Question:
{question}

Answer with source citations [1], [2]:"""


class AnswerGenerator:
    """Orchestrates multi-source RAG generation, confidence thresholding, and metrics."""

    def __init__(
        self,
        orchestrator: Optional[SourceOrchestrator] = None,
        llm_provider: Optional[LLMProvider] = None,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or get_settings()
        self.llm_provider = llm_provider or LLMProvider(settings=self.settings)
        self.orchestrator = orchestrator or SourceOrchestrator(
            llm_provider=self.llm_provider,
            settings=self.settings,
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("user", USER_PROMPT),
        ])
        self.output_parser = StrOutputParser()

        llm = self.llm_provider.get_llm()
        self.chain = (self.prompt | llm | self.output_parser) if llm is not None else None

    def count_tokens(self, text: str) -> int:
        """Estimate tokens using the LLM provider."""
        return self.llm_provider.count_tokens(text)

    def compute_confidence(
        self,
        answer: str,
        retrieved_results: List[RetrievalResult],
        rerank_enabled: bool,
    ) -> ConfidenceDetail:
        """
        Confidence Score C in [0.0, 1.0]:
        1. Retrieval support S_retrieval = 0.65 * S_top + 0.35 * S_avg
        2. Grounding factor F_grounding = 1.0 if answer cites valid sources [i], else 0.5
        3. C = round(S_retrieval * F_grounding, 4)
        """
        if not retrieved_results:
            return ConfidenceDetail(
                score=0.0,
                method="no_context",
                top_chunk_score=0.0,
                avg_chunk_score=0.0,
                grounded=False,
                passed_threshold=False,
            )

        scores = [r.score for r in retrieved_results]
        top_score = scores[0] if scores else 0.0
        avg_score = sum(scores) / len(scores) if scores else 0.0

        retrieval_support = 0.65 * top_score + 0.35 * avg_score

        has_citations = bool(re.search(r"\[\d+\]", answer))
        is_unknown = "i don't know" in answer.lower() or "not enough information" in answer.lower()
        grounded = has_citations and not is_unknown

        grounding_factor = 1.0 if grounded else 0.5
        final_score = max(0.0, min(1.0, round(retrieval_support * grounding_factor, 4)))

        passed = final_score >= self.settings.LOW_CONFIDENCE_THRESHOLD
        method_str = "cross_encoder_sigmoid" if rerank_enabled else "rrf_normalized"

        return ConfidenceDetail(
            score=final_score,
            method=method_str,
            top_chunk_score=round(top_score, 4),
            avg_chunk_score=round(avg_score, 4),
            grounded=grounded,
            passed_threshold=passed,
        )

    def _generate_from_candidates(
        self,
        query: str,
        retrieved_results: List[RetrievalResult],
        enable_rerank: Optional[bool] = None,
    ) -> Tuple[str, List[Citation], ConfidenceDetail, str]:
        """Generate answer, citations, and confidence for a given set of retrieval candidates."""
        context_text = CitationManager.build_context_block(retrieved_results)
        if self.chain and retrieved_results:
            prompt_value = self.prompt.format(context=context_text, question=query)
            answer = self.chain.invoke({"context": context_text, "question": query})
        else:
            prompt_value = f"{context_text}\n{query}"
            if retrieved_results:
                top_doc = retrieved_results[0].document
                source_name = top_doc.metadata.get("source", "doc")
                answer = f"Based on [{top_doc.metadata.get('source_type', 'source')}] {source_name} [1], {top_doc.page_content[:180]}."
            else:
                answer = "I don't know based on the available sources."

        citations = CitationManager.extract_cited_sources(answer, retrieved_results)
        rerank_active = enable_rerank if enable_rerank is not None else self.settings.ENABLE_RERANKER
        confidence = self.compute_confidence(answer, retrieved_results, rerank_active)
        return answer, citations, confidence, prompt_value

    def generate_answer(
        self,
        query: str,
        top_k: Optional[int] = None,
        enable_rerank: Optional[bool] = None,
        routing_mode: Optional[str] = None,
    ) -> QueryResponse:
        """Generate answer synchronously with complete metadata, citations, and metrics."""
        latencies: Dict[str, float] = {}
        rerank_active = enable_rerank if enable_rerank is not None else self.settings.ENABLE_RERANKER

        def answer_evaluator(kb_results: List[RetrievalResult]):
            ans, cits, conf, p_val = self._generate_from_candidates(query, kb_results, rerank_active)
            is_idk = (
                ans.strip().startswith("I don't know")
                or "i don't know based on the available sources" in ans.lower()
                or not conf.passed_threshold
                or not conf.grounded
                or not kb_results
            )
            return conf.score, is_idk, (ans, cits, conf, p_val)

        # 1. Multi-source routing & retrieval with cascade answer evaluation
        with RequestStageTimer("multi_source_retrieval", latencies):
            import inspect
            sig = inspect.signature(self.orchestrator.orchestrate)
            if "answer_evaluator" in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                retrieved_results, routing_meta = self.orchestrator.orchestrate(
                    query=query,
                    routing_mode=routing_mode,
                    top_k=top_k,
                    answer_evaluator=answer_evaluator,
                )
            else:
                retrieved_results, routing_meta = self.orchestrator.orchestrate(
                    query=query,
                    routing_mode=routing_mode,
                    top_k=top_k,
                )

        # 2. Generation stage (reuse precomputed KB answer if sufficient, else regenerate)
        kb_eval_data = routing_meta.get("kb_eval_data")
        if kb_eval_data is not None:
            answer, citations, confidence, prompt_value = kb_eval_data
        else:
            with RequestStageTimer("generation", latencies):
                answer, citations, confidence, prompt_value = self._generate_from_candidates(
                    query=query,
                    retrieved_results=retrieved_results,
                    enable_rerank=rerank_active,
                )

            # Enforce "I don't know" threshold strictly with no fabricated answer
            if not confidence.passed_threshold or not retrieved_results:
                answer = "I don't know based on the available sources."
                confidence.grounded = False

        # 3. Token & Cost calculation
        prompt_tokens = self.count_tokens(prompt_value)
        completion_tokens = self.count_tokens(answer)
        total_tokens = prompt_tokens + completion_tokens
        token_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

        cost_usd = calculate_cost(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            input_rate_per_million=self.settings.INPUT_COST_PER_MILLION,
            output_rate_per_million=self.settings.OUTPUT_COST_PER_MILLION,
        )

        total_latency = round(sum(latencies.values()), 2)

        # Record Prometheus metrics
        REQUEST_COUNT.labels(endpoint="/ask", status="success").inc()
        for stage, ms in latencies.items():
            STAGE_LATENCY.labels(stage=stage).observe(ms / 1000.0)
        TOKEN_USAGE.labels(type="prompt").inc(prompt_tokens)
        TOKEN_USAGE.labels(type="completion").inc(completion_tokens)
        ESTIMATED_COST.inc(cost_usd)

        # Per-request structured log
        extra_log = {
            "routing_mode": routing_meta.get("routing_mode"),
            "sources_queried": routing_meta.get("sources_queried"),
            "router_decision": routing_meta.get("router_decision"),
            "kb_only_confidence": routing_meta.get("kb_only_confidence"),
            "per_source_latency_ms": routing_meta.get("per_source_latency_ms"),
        }
        log_request_summary(
            query=query,
            latency_ms=latencies,
            tokens=token_usage,
            estimated_cost=cost_usd,
            status="success",
            extra=extra_log,
        )

        return QueryResponse(
            query=query,
            answer=answer,
            citations=citations,
            confidence=confidence,
            routing_mode=routing_meta.get("routing_mode", self.settings.ROUTING_MODE),
            sources_queried=routing_meta.get("sources_queried", []),
            router_decision=routing_meta.get("router_decision", ""),
            router_fallback=routing_meta.get("router_fallback") if routing_meta.get("router_decision") != "kb_sufficient" else False,
            per_source_latency_ms=routing_meta.get("per_source_latency_ms", {}),
            stage_latency_ms=latencies,
            total_latency_ms=total_latency,
            token_usage=token_usage,
            estimated_cost_usd=cost_usd,
        )

    async def stream_answer(
        self,
        query: str,
        top_k: Optional[int] = None,
        enable_rerank: Optional[bool] = None,
        routing_mode: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream answer tokens followed by final JSON metadata chunk (SSE format)."""
        latencies: Dict[str, float] = {}

        rerank_active = enable_rerank if enable_rerank is not None else self.settings.ENABLE_RERANKER

        def answer_evaluator(kb_results: List[RetrievalResult]):
            ans, cits, conf, p_val = self._generate_from_candidates(query, kb_results, rerank_active)
            is_idk = (
                ans.strip().startswith("I don't know")
                or "i don't know based on the available sources" in ans.lower()
                or not conf.passed_threshold
                or not conf.grounded
                or not kb_results
            )
            return conf.score, is_idk, (ans, cits, conf, p_val)

        # 1. Multi-source routing & retrieval
        with RequestStageTimer("multi_source_retrieval", latencies):
            import inspect
            sig = inspect.signature(self.orchestrator.orchestrate)
            if "answer_evaluator" in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                retrieved_results, routing_meta = self.orchestrator.orchestrate(
                    query=query,
                    routing_mode=routing_mode,
                    top_k=top_k,
                    answer_evaluator=answer_evaluator,
                )
            else:
                retrieved_results, routing_meta = self.orchestrator.orchestrate(
                    query=query,
                    routing_mode=routing_mode,
                    top_k=top_k,
                )

        context_text = CitationManager.build_context_block(retrieved_results)
        full_answer_parts: List[str] = []

        kb_eval_data = routing_meta.get("kb_eval_data")
        if kb_eval_data is not None:
            full_answer, citations, confidence, prompt_value = kb_eval_data
            for word in full_answer.split(" "):
                token = word + " "
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        else:
            prompt_value = self.prompt.format(context=context_text, question=query) if self.chain else query

            # 2. Generation stream
            with RequestStageTimer("generation", latencies):
                if self.chain and retrieved_results:
                    async for chunk in self.chain.astream({"context": context_text, "question": query}):
                        full_answer_parts.append(chunk)
                        yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
                else:
                    if retrieved_results:
                        top_doc = retrieved_results[0].document
                        source_name = top_doc.metadata.get("source", "doc")
                        mock_text = f"Based on [{top_doc.metadata.get('source_type', 'source')}] {source_name} [1], {top_doc.page_content[:180]}."
                    else:
                        mock_text = "I don't know based on the available sources."

                    for word in mock_text.split(" "):
                        token = word + " "
                        full_answer_parts.append(token)
                        yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            full_answer = "".join(full_answer_parts)

            # 3. Compute final metadata & verify confidence
            citations = CitationManager.extract_cited_sources(full_answer, retrieved_results)
            confidence = self.compute_confidence(full_answer, retrieved_results, rerank_active)

            if not confidence.passed_threshold or not retrieved_results:
                full_answer = "I don't know based on the available sources."
                confidence.grounded = False

        prompt_tokens = self.count_tokens(prompt_value)
        completion_tokens = self.count_tokens(full_answer)
        total_tokens = prompt_tokens + completion_tokens
        token_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

        cost_usd = calculate_cost(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            input_rate_per_million=self.settings.INPUT_COST_PER_MILLION,
            output_rate_per_million=self.settings.OUTPUT_COST_PER_MILLION,
        )

        total_latency = round(sum(latencies.values()), 2)

        # Emit Prometheus metrics
        REQUEST_COUNT.labels(endpoint="/ask", status="success").inc()
        for stage, ms in latencies.items():
            STAGE_LATENCY.labels(stage=stage).observe(ms / 1000.0)
        TOKEN_USAGE.labels(type="prompt").inc(prompt_tokens)
        TOKEN_USAGE.labels(type="completion").inc(completion_tokens)
        ESTIMATED_COST.inc(cost_usd)

        extra_log = {
            "routing_mode": routing_meta.get("routing_mode"),
            "sources_queried": routing_meta.get("sources_queried"),
            "router_decision": routing_meta.get("router_decision"),
            "per_source_latency_ms": routing_meta.get("per_source_latency_ms"),
        }
        log_request_summary(
            query=query,
            latency_ms=latencies,
            tokens=token_usage,
            estimated_cost=cost_usd,
            status="success",
            extra=extra_log,
        )

        final_payload = {
            "type": "final",
            "query": query,
            "answer": full_answer,
            "citations": [c.model_dump() for c in citations],
            "confidence": confidence.model_dump(),
            "routing_mode": routing_meta.get("routing_mode", self.settings.ROUTING_MODE),
            "sources_queried": routing_meta.get("sources_queried", []),
            "router_decision": routing_meta.get("router_decision", ""),
            "per_source_latency_ms": routing_meta.get("per_source_latency_ms", {}),
            "stage_latency_ms": latencies,
            "total_latency_ms": total_latency,
            "token_usage": token_usage,
            "estimated_cost_usd": cost_usd,
        }
        yield f"data: {json.dumps(final_payload)}\n\n"
        yield "data: [DONE]\n\n"
