"""Prometheus metrics for InsightRAG system monitoring."""
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Request metrics
REQUEST_COUNT = Counter(
    "insightrag_requests_total",
    "Total number of requests handled",
    ["endpoint", "status"]
)

# Latency metrics
STAGE_LATENCY = Histogram(
    "insightrag_stage_latency_seconds",
    "Latency breakdown by stage in seconds",
    ["stage"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# Token metrics
TOKEN_USAGE = Counter(
    "insightrag_tokens_total",
    "Total tokens consumed",
    ["type"]  # "prompt" or "completion"
)

# Cost metrics
ESTIMATED_COST = Counter(
    "insightrag_estimated_cost_usd_total",
    "Total estimated cost in USD"
)

# Multi-source and routing metrics
SOURCE_QUERIES = Counter(
    "insightrag_source_queries_total",
    "Total queries sent to individual retrieval sources",
    ["source"]  # "kb", "wikipedia", "web"
)

SOURCE_FALLBACKS = Counter(
    "insightrag_source_fallbacks_total",
    "Total fallback or empty executions per source",
    ["source"]
)

ROUTER_DECISIONS = Counter(
    "insightrag_router_decisions_total",
    "Routing decisions taken by the cascade or fan_out router",
    ["decision"]  # "kb_only", "wikipedia", "web", "both", "fan_out"
)


def get_prometheus_metrics() -> tuple[bytes, str]:
    """Returns Prometheus formatted metrics and content-type."""
    return generate_latest(), CONTENT_TYPE_LATEST
