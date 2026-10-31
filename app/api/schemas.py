"""Pydantic schemas for API request and response data models."""
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from app.sources.citation import Citation


class QueryRequest(BaseModel):
    """Input payload for POST /ask."""
    query: str = Field(..., min_length=1, description="User's query")
    top_k: Optional[int] = Field(None, ge=1, le=50, description="Override top-k retrieved chunks")
    rerank: Optional[bool] = Field(None, description="Enable or disable CrossEncoder reranking")
    routing_mode: Optional[str] = Field(None, description="Routing mode: 'fan_out' or 'cascade_router'")
    stream: bool = Field(False, description="Enable Server-Sent Events (SSE) streaming response")


class ConfidenceDetail(BaseModel):
    """Breakdown of confidence score computation."""
    score: float = Field(..., ge=0.0, le=1.0, description="Final confidence score in [0.0, 1.0]")
    method: str = Field(..., description="Method used: 'cross_encoder_sigmoid' or 'rrf_normalized'")
    top_chunk_score: float = Field(..., description="Normalized score of the top-ranked chunk")
    avg_chunk_score: float = Field(..., description="Average normalized score of all retrieved chunks")
    grounded: bool = Field(..., description="Whether answer directly grounds in context citations")
    passed_threshold: bool = Field(True, description="Whether score meets the minimum low confidence threshold")


class QueryResponse(BaseModel):
    """Non-streaming response model for POST /ask."""
    query: str
    answer: str
    citations: List[Citation]
    confidence: ConfidenceDetail
    routing_mode: str
    sources_queried: List[str]
    router_decision: str
    per_source_latency_ms: Dict[str, float]
    stage_latency_ms: Dict[str, float]
    total_latency_ms: float
    token_usage: Dict[str, int]
    estimated_cost_usd: float
    router_fallback: Optional[bool] = Field(default=None, description="Whether router fallback was triggered")


class HealthResponse(BaseModel):
    """Response model for GET /health."""
    status: str
    version: str
    index_loaded: bool
    corpus_size: int
    reranker_enabled: bool
    routing_mode: str
