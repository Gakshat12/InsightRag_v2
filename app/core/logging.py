"""Structured logging with per-request latency, token, and cost tracking."""
import logging
import json
import time
from typing import Dict, Any, Optional

logger = logging.getLogger("insightrag")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class RequestStageTimer:
    """Context manager to measure and log execution latency for pipeline stages."""

    def __init__(self, stage_name: str, latencies_dict: Optional[Dict[str, float]] = None):
        self.stage_name = stage_name
        self.latencies_dict = latencies_dict
        self.start_time: float = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_ms = (time.perf_counter() - self.start_time) * 1000.0
        if self.latencies_dict is not None:
            self.latencies_dict[self.stage_name] = round(elapsed_ms, 2)


def calculate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    input_rate_per_million: float = 0.15,
    output_rate_per_million: float = 0.60,
) -> float:
    """Calculate estimated cost in USD based on token counts."""
    input_cost = (prompt_tokens / 1_000_000.0) * input_rate_per_million
    output_cost = (completion_tokens / 1_000_000.0) * output_rate_per_million
    return round(input_cost + output_cost, 6)


def log_request_summary(
    query: str,
    latency_ms: Dict[str, float],
    tokens: Dict[str, int],
    estimated_cost: float,
    status: str = "success",
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit a structured JSON log entry for query execution."""
    total_latency = sum(latency_ms.values()) if latency_ms else 0.0
    payload = {
        "event": "query_summary",
        "status": status,
        "query": query[:120] + ("..." if len(query) > 120 else ""),
        "total_latency_ms": round(total_latency, 2),
        "stage_latency_ms": latency_ms,
        "token_usage": tokens,
        "estimated_cost_usd": estimated_cost,
    }
    if extra:
        payload.update(extra)
    logger.info(json.dumps(payload))

