from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from config.settings import invalidate_settings_cache
from observability.langfuse import get_langfuse_handler, reset_langfuse_handler


@pytest.fixture(autouse=True)
def _reset():
    invalidate_settings_cache()
    reset_langfuse_handler()
    yield


class TestGetLangfuseHandler:
    def test_returns_none_when_disabled(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        monkeypatch.setenv("LANGFUSE_ENABLED", "false")
        invalidate_settings_cache()
        reset_langfuse_handler()

        handler = get_langfuse_handler()
        assert handler is None

    def test_returns_none_when_not_set(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        monkeypatch.setenv("LANGFUSE_ENABLED", "false")
        invalidate_settings_cache()
        reset_langfuse_handler()

        handler = get_langfuse_handler()
        assert handler is None

    def test_initialises_handler_when_enabled(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_ENABLED", "true")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
        monkeypatch.setenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")
        invalidate_settings_cache()
        reset_langfuse_handler()

        handler = get_langfuse_handler()
        assert handler is not None

    def test_returns_singleton(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_ENABLED", "true")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
        monkeypatch.setenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")
        invalidate_settings_cache()
        reset_langfuse_handler()

        h1 = get_langfuse_handler()
        h2 = get_langfuse_handler()
        assert h1 is h2

    def test_handles_init_failure_gracefully(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_ENABLED", "true")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
        monkeypatch.setenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")
        invalidate_settings_cache()
        reset_langfuse_handler()

        # Langfuse and CallbackHandler are imported inside the function body
        with patch("langfuse.Langfuse", side_effect=RuntimeError("init failed")):
            handler = get_langfuse_handler()
            assert handler is None

    def test_reset_clears_singleton(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_ENABLED", "true")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
        monkeypatch.setenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")
        invalidate_settings_cache()
        reset_langfuse_handler()

        h1 = get_langfuse_handler()
        reset_langfuse_handler()
        h2 = get_langfuse_handler()
        assert h1 is not h2


class TestLLMServiceInstrumentation:
    @pytest.mark.asyncio
    async def test_passes_callbacks_when_handler_available(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_ENABLED", "true")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
        monkeypatch.setenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")
        invalidate_settings_cache()
        reset_langfuse_handler()

        handler = get_langfuse_handler()
        assert handler is not None

        fake_llm = MagicMock()
        fake_llm.invoke.return_value = "response text"

        from clinical.llm import LLMService

        svc = LLMService(model="fake-model", base_url="http://fake:11434")

        with patch("langchain_ollama.OllamaLLM", return_value=fake_llm):
            result = svc._call_llm_sync(prompt="hello", system_prompt="be helpful")
            assert result == "response text"
            call_kwargs = fake_llm.invoke.call_args
            assert call_kwargs is not None
            args, kwargs = call_kwargs
            assert "config" in kwargs
            assert "callbacks" in kwargs["config"]
            assert kwargs["config"]["callbacks"] == [handler]

    @pytest.mark.asyncio
    async def test_no_callbacks_when_disabled(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_ENABLED", "false")
        invalidate_settings_cache()
        reset_langfuse_handler()

        handler = get_langfuse_handler()
        assert handler is None

        fake_llm = MagicMock()
        fake_llm.invoke.return_value = "response text"

        from clinical.llm import LLMService

        svc = LLMService(model="fake-model", base_url="http://fake:11434")

        with patch("langchain_ollama.OllamaLLM", return_value=fake_llm):
            result = svc._call_llm_sync(prompt="hello", system_prompt="be helpful")
            assert result == "response text"
            fake_llm.invoke.assert_called_once_with(
                "be helpful\n\nhello"
            )
