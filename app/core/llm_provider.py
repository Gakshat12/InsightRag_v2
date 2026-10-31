"""Swappable LLM Provider abstraction supporting Gemini and custom backends."""
from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel
from app.core.config import get_settings, Settings
from app.core.logging import logger


class LLMProvider:
    """Unified LLM Provider class for swappable model backends."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or get_settings()
        self.provider = (provider if provider is not None else self.settings.LLM_PROVIDER).lower()
        self.model_name = model_name if model_name is not None else self.settings.GEMINI_MODEL
        self.api_key = api_key if api_key is not None else self.settings.GEMINI_API_KEY
        self._llm: Optional[BaseChatModel] = None

    def get_llm(self) -> Optional[BaseChatModel]:
        """Return the initialized BaseChatModel for the active provider."""
        if self._llm is not None:
            return self._llm

        if self.provider == "gemini":
            if not self.api_key:
                logger.info("GEMINI_API_KEY is empty/offline. LLM operating in offline/mock mode.")
                return None
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                self._llm = ChatGoogleGenerativeAI(
                    model=self.model_name,
                    google_api_key=self.api_key,
                    temperature=0.0,
                    streaming=True,
                    timeout=30,
                    max_retries=1
                )
                logger.info(f"Initialized Gemini LLM Provider with model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize ChatGoogleGenerativeAI: {e}")
                self._llm = None
        else:
            logger.info(f"Custom or offline provider '{self.provider}'. Operating in offline/mock mode.")
            self._llm = None

        return self._llm

    def count_tokens(self, text: str) -> int:
        """Estimate token count without proprietary tokenizer dependencies."""
        if not text:
            return 0
        llm = self.get_llm()
        if llm is not None and hasattr(llm, "get_num_tokens"):
            try:
                return llm.get_num_tokens(text)
            except Exception:
                pass
        return max(1, len(text) // 4)
