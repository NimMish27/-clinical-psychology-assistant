from __future__ import annotations

import asyncio
from typing import Any

from config.settings import get_settings
from app_logging.logger import get_logger

_log = get_logger(__name__)


class LLMService:
    """Unified LLM client for the clinical pipeline.

    Wraps the Ollama LangChain integration so all pipeline stages
    share a single interface.  Callers provide a prompt and an optional
    system prompt; the service handles invocation, error wrapping, and
    timing.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        s = get_settings()
        self._model = model or s.llm.ollama_model
        self._base_url = base_url or s.llm.ollama_base_url
        self._temperature = temperature if temperature is not None else s.llm.temperature
        self._max_tokens = max_tokens or s.llm.max_tokens
        self._top_k = s.llm.top_k
        self._top_p = s.llm.top_p

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """Invoke the LLM and return the generated text."""
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                None,
                self._call_llm_sync,
                prompt,
                system_prompt,
            )
        except Exception as exc:
            _log.error(
                "clinical.llm_error",
                model=self._model,
                error=str(exc),
            )
            raise

    def _call_llm_sync(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        try:
            from langchain_ollama import OllamaLLM
            llm = OllamaLLM(
                model=self._model,
                base_url=self._base_url,
                temperature=self._temperature,
                top_k=self._top_k,
                top_p=self._top_p,
                num_predict=self._max_tokens,
            )
        except ImportError:
            from langchain_community.llms import Ollama
            llm = Ollama(
                model=self._model,
                base_url=self._base_url,
                temperature=self._temperature,
            )

        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        result = llm.invoke(full_prompt)
        return result.strip() if result else ""


_service: LLMService | None = None


def get_llm_service(**kwargs: Any) -> LLMService:
    global _service
    if _service is None:
        _service = LLMService(**kwargs)
    return _service
