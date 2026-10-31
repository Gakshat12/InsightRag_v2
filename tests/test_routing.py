"""Tests for multi-source routing, cascade decisions, source failure fallbacks, and I-don't-know path."""
import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document

from app.core.config import Settings
from app.core.llm_provider import LLMProvider
from app.sources.base import BaseSource, SourceResult
from app.retrieval.service import RetrievalResult
from app.retrieval.router import SourceOrchestrator
from app.api.generator import AnswerGenerator


class MockFailingSource(BaseSource):
    """Source that simulates repeated failures/timeouts."""
    def __init__(self, name="FailingSource", source_type="web"):
        super().__init__(name=name, source_type=source_type, max_retries=1, backoff_factor=0.01)

    def _search_implementation(self, query: str, top_k: int):
        raise ConnectionError("Simulated network timeout/failure")


class MockSuccessfulSource(BaseSource):
    """Source that returns predefined items."""
    def __init__(self, name, source_type, results):
        super().__init__(name=name, source_type=source_type)
        self._results = results

    def _search_implementation(self, query: str, top_k: int):
        return self._results


def test_source_failure_resilience():
    """Failing source must never crash, and returns graceful empty fallback."""
    failing = MockFailingSource()
    res = failing.search("query")
    assert res == []


def test_routing_fan_out():
    """fan_out mode must query all three sources."""
    settings = Settings(ROUTING_MODE="fan_out")
    kb = MockSuccessfulSource("KB", "kb", [SourceResult(content="KB content", source_type="kb", title="kb1")])
    wiki = MockSuccessfulSource("Wiki", "wikipedia", [SourceResult(content="Wiki content", source_type="wikipedia", title="wiki1")])
    web = MockSuccessfulSource("Web", "web", [SourceResult(content="Web content", source_type="web", title="web1")])

    mock_llm_provider = LLMProvider(api_key="")
    orchestrator = SourceOrchestrator(
        kb_source=kb,
        wikipedia_source=wiki,
        web_source=web,
        llm_provider=mock_llm_provider,
        settings=settings,
    )

    results, meta = orchestrator.orchestrate("quantum computing", routing_mode="fan_out")

    assert meta["routing_mode"] == "fan_out"
    assert set(meta["sources_queried"]) == {"kb", "wikipedia", "web"}
    assert meta["router_decision"] == "fan_out_all"
    assert len(results) == 3


def test_routing_cascade_kb_sufficient():
    """When KB confidence is high, cascade router must not call external sources."""
    settings = Settings(ROUTING_MODE="cascade_router", CASCADE_KB_THRESHOLD=0.50)
    kb_results = [
        SourceResult(content="High relevance chunk 1", source_type="kb", title="k1", metadata={"raw_score": 0.95}),
        SourceResult(content="High relevance chunk 2", source_type="kb", title="k2", metadata={"raw_score": 0.85}),
    ]
    kb = MockSuccessfulSource("KB", "kb", kb_results)
    wiki = MockSuccessfulSource("Wiki", "wikipedia", [])
    web = MockSuccessfulSource("Web", "web", [])

    mock_llm_provider = LLMProvider(api_key="")
    orchestrator = SourceOrchestrator(
        kb_source=kb,
        wikipedia_source=wiki,
        web_source=web,
        llm_provider=mock_llm_provider,
        settings=settings,
    )
    results, meta = orchestrator.orchestrate("internal policy", routing_mode="cascade_router")

    assert meta["sources_queried"] == ["kb"]
    assert meta["router_decision"] == "kb_sufficient"
    assert len(results) == 2


def test_routing_cascade_low_confidence_escalation(monkeypatch):
    """When KB confidence is low, cascade router escalates to Gemini router decision."""
    settings = Settings(ROUTING_MODE="cascade_router", CASCADE_KB_THRESHOLD=0.50)
    kb = MockSuccessfulSource("KB", "kb", [
        SourceResult(content="Vague mention", source_type="kb", title="k1", metadata={"raw_score": 0.15})
    ])
    wiki = MockSuccessfulSource("Wiki", "wikipedia", [
        SourceResult(content="Wikipedia article", source_type="wikipedia", title="W1", url="https://en.wikipedia.org/wiki/W1")
    ])
    web = MockSuccessfulSource("Web", "web", [
        SourceResult(content="Web snippet", source_type="web", title="Web1", url="https://example.com")
    ])

    mock_llm_provider = LLMProvider(api_key="")
    orchestrator = SourceOrchestrator(
        kb_source=kb,
        wikipedia_source=wiki,
        web_source=web,
        llm_provider=mock_llm_provider,
        settings=settings,
    )

    monkeypatch.setattr(
        orchestrator,
        "route_external_sources_with_gemini",
        lambda query, kb_context: (["wikipedia"], "Alan Turing")
    )

    results, meta = orchestrator.orchestrate("Alan Turing history", routing_mode="cascade_router")

    assert "kb" in meta["sources_queried"]
    assert "wikipedia" in meta["sources_queried"]
    assert "web" not in meta["sources_queried"]
    assert meta["router_decision"] == "escalated_after_low_confidence"


def test_low_confidence_path_idk():
    """When overall confidence is below threshold, must return I don't know without fabrication."""
    settings = Settings(LOW_CONFIDENCE_THRESHOLD=0.40)
    orchestrator = MagicMock()
    orchestrator.orchestrate.return_value = (
        [
            RetrievalResult(
                document=Document(page_content="Unrelated text", metadata={"source": "d1", "source_type": "kb"}),
                score=0.10,
                rank=1,
                metadata={},
            )
        ],
        {"routing_mode": "fan_out", "sources_queried": ["kb"], "router_decision": "none", "per_source_latency_ms": {}}
    )

    mock_llm_provider = LLMProvider(api_key="")
    generator = AnswerGenerator(orchestrator=orchestrator, llm_provider=mock_llm_provider, settings=settings)
    resp = generator.generate_answer("Obscure question")

    assert "I don't know based on the available sources." in resp.answer
    assert resp.confidence.passed_threshold is False


def test_empty_results_handling():
    """When no documents are found, must return I don't know."""
    orchestrator = MagicMock()
    orchestrator.orchestrate.return_value = (
        [],
        {"routing_mode": "fan_out", "sources_queried": ["kb"], "router_decision": "none", "per_source_latency_ms": {}}
    )

    mock_llm_provider = LLMProvider(api_key="")
    generator = AnswerGenerator(orchestrator=orchestrator, llm_provider=mock_llm_provider)
    resp = generator.generate_answer("Completely missing query")

    assert "I don't know based on the available sources." in resp.answer
    assert resp.confidence.score == 0.0


def test_wikipedia_passages_in_merged_results(monkeypatch):
    """Wikipedia passages must appear in merged results when cascade escalates to Wikipedia."""
    settings = Settings(ROUTING_MODE="cascade_router", CASCADE_KB_THRESHOLD=0.50)
    kb = MockSuccessfulSource("KB", "kb", [
        SourceResult(content="No relevant info", source_type="kb", title="k1", metadata={"raw_score": 0.1})
    ])
    wiki = MockSuccessfulSource("Wiki", "wikipedia", [
        SourceResult(
            content="A B-tree is a self-balancing tree data structure that maintains sorted data.",
            source_type="wikipedia",
            title="B-tree",
            url="https://en.wikipedia.org/wiki/B-tree",
        ),
        SourceResult(
            content="By allowing more children under one node than a regular binary search tree.",
            source_type="wikipedia",
            title="B-tree",
            url="https://en.wikipedia.org/wiki/B-tree",
        ),
    ])
    web = MockSuccessfulSource("Web", "web", [])

    mock_llm_provider = LLMProvider(api_key="")
    orchestrator = SourceOrchestrator(
        kb_source=kb,
        wikipedia_source=wiki,
        web_source=web,
        llm_provider=mock_llm_provider,
        settings=settings,
    )

    monkeypatch.setattr(
        orchestrator,
        "route_external_sources_with_gemini",
        lambda query, kb_context: (["wikipedia"], "B-tree"),
    )

    results, meta = orchestrator.orchestrate("What is a B-tree?", routing_mode="cascade_router")

    source_types = [r.document.metadata.get("source_type") for r in results]
    assert "wikipedia" in source_types, f"Wikipedia not in results: {source_types}"
    assert "wikipedia" in meta["sources_queried"]
    assert "web" not in meta["sources_queried"]


def test_escalation_after_low_confidence_kb_answer(monkeypatch):
    """When KB-only answer yields low confidence or 'I don't know', cascade router must escalate."""
    settings = Settings(ROUTING_MODE="cascade_router", CASCADE_KB_THRESHOLD=0.50, LOW_CONFIDENCE_THRESHOLD=0.35)
    kb = MockSuccessfulSource("KB", "kb", [
        SourceResult(content="Irrelevant document about unrelated topics.", source_type="kb", title="irrelevant_doc", metadata={"raw_score": 0.45})
    ])
    wiki = MockSuccessfulSource("Wiki", "wikipedia", [
        SourceResult(
            content="A B-tree is a self-balancing search tree data structure.",
            source_type="wikipedia",
            title="B-tree",
            url="https://en.wikipedia.org/wiki/B-tree"
        )
    ])
    web = MockSuccessfulSource("Web", "web", [])
    mock_llm_provider = LLMProvider(api_key="")

    orchestrator = SourceOrchestrator(
        kb_source=kb,
        wikipedia_source=wiki,
        web_source=web,
        llm_provider=mock_llm_provider,
        settings=settings,
    )
    generator = AnswerGenerator(orchestrator=orchestrator, llm_provider=mock_llm_provider, settings=settings)

    monkeypatch.setattr(
        orchestrator,
        "route_external_sources_with_gemini",
        lambda query, kb_context: (["wikipedia"], "B-tree")
    )

    def mock_chain_invoke(inputs):
        ctx = inputs.get("context", "")
        if "Irrelevant" in ctx and "B-tree" not in ctx:
            return "I don't know based on the available sources."
        return "A B-tree is a self-balancing tree [1]."

    generator.chain = MagicMock()
    generator.chain.invoke = mock_chain_invoke

    response = generator.generate_answer("What is a B-tree?", routing_mode="cascade_router")

    assert response.router_decision == "escalated_after_low_confidence"
    assert "wikipedia" in response.sources_queried
    assert any(c.source_type == "wikipedia" for c in response.citations)
