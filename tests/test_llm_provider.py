"""Unit tests for the swappable LLMProvider abstraction."""
import pytest
from app.core.config import Settings
from app.core.llm_provider import LLMProvider


def test_llm_provider_offline_fallback():
    # When GEMINI_API_KEY is empty, provider defaults gracefully to None
    settings = Settings(GEMINI_API_KEY="", LLM_PROVIDER="gemini")
    provider = LLMProvider(settings=settings)

    assert provider.provider == "gemini"
    assert provider.get_llm() is None


def test_llm_provider_token_counter():
    provider = LLMProvider(settings=Settings(GEMINI_API_KEY=""))
    text = "InsightRAG hybrid retrieval engine"
    count = provider.count_tokens(text)
    assert count > 0


def test_llm_provider_custom_swap():
    # Verify provider string can be swapped
    provider = LLMProvider(provider="custom_mock", api_key="test-key")
    assert provider.provider == "custom_mock"
    assert provider.get_llm() is None

