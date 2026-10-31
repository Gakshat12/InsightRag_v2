"""FastAPI Application providing /ask, /health, and /metrics endpoints."""
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app import __version__
from app.core.config import get_settings
from app.core.metrics import get_prometheus_metrics
from app.api.schemas import QueryRequest, QueryResponse, HealthResponse
from app.api.generator import AnswerGenerator
from app.retrieval.service import RetrievalService
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.router import SourceOrchestrator
from app.sources.kb_source import KBSource
from app.sources.wikipedia_source import WikipediaSource
from app.sources.web_source import WebSearchSource

settings = get_settings()

app = FastAPI(
    title="InsightRAG API",
    version=__version__,
    description="Enterprise Hybrid Retrieval-Augmented Generation Service",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Injected singletons
retriever = HybridRetriever()
retrieval_service = RetrievalService(retriever=retriever)

kb_source = KBSource(retrieval_service=retrieval_service)
wikipedia_source = WikipediaSource()
web_source = WebSearchSource()

orchestrator = SourceOrchestrator(
    kb_source=kb_source,
    wikipedia_source=wikipedia_source,
    web_source=web_source,
    settings=settings,
)

generator = AnswerGenerator(orchestrator=orchestrator, settings=settings)


@app.on_event("startup")
async def startup_event():
    """Attempt initial loading of indices on startup."""
    retriever.load_indices()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """System health check and index status."""
    is_loaded = retriever.is_loaded
    corpus_size = len(retriever.corpus_documents) if is_loaded else 0
    return HealthResponse(
        status="healthy" if is_loaded else "unindexed",
        version=__version__,
        index_loaded=is_loaded,
        corpus_size=corpus_size,
        reranker_enabled=settings.ENABLE_RERANKER,
        routing_mode=settings.ROUTING_MODE,
    )


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    body, content_type = get_prometheus_metrics()
    return Response(content=body, media_type=content_type)


@app.post("/ask", response_model=QueryResponse)
async def ask(request: QueryRequest):
    """
    RAG query endpoint supporting multi-source routing, JSON and streaming (SSE) response formats.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if request.stream:
        return StreamingResponse(
            generator.stream_answer(
                query=request.query,
                top_k=request.top_k,
                enable_rerank=request.rerank,
                routing_mode=request.routing_mode,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    try:
        return generator.generate_answer(
            query=request.query,
            top_k=request.top_k,
            enable_rerank=request.rerank,
            routing_mode=request.routing_mode,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
