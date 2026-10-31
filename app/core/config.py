"""InsightRAG application configuration module."""
import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for InsightRAG loaded from env / .env file."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # LLM Provider Configuration
    LLM_PROVIDER: str = "gemini"
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-3.8-flash"

    # External Search Keys
    TAVILY_API_KEY: str = ""

    # Routing Configuration
    ROUTING_MODE: str = "cascade_router"  # "fan_out" or "cascade_router"
    CASCADE_KB_THRESHOLD: float = 0.50
    LOW_CONFIDENCE_THRESHOLD: float = 0.35

    # Local Sentence-Transformers Embeddings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # CrossEncoder Reranker
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ENABLE_RERANKER: bool = True

    # Ingestion & Chunking
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # Retrieval
    TOP_K: int = 5
    RRF_K: int = 60
    INDEX_DIR: str = os.path.join("data", "indices")

    # API Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # Cost Estimation ($ per 1M tokens) - Gemini 1.5 Flash reference
    INPUT_COST_PER_MILLION: float = 0.075
    OUTPUT_COST_PER_MILLION: float = 0.30


@lru_cache()
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
