from __future__ import annotations

from typing import Any

from config.settings import get_settings
from app_logging.logger import get_logger

_log = get_logger(__name__)

_handler: Any = None  # langfuse.langchain.CallbackHandler | None


def get_langfuse_handler() -> Any | None:
    """Return a singleton Langfuse CallbackHandler, or None if disabled.

    Follows the same singleton pattern as ``get_llm_service()`` and
    ``get_settings()``.  Callers should pass the returned handler in
    ``config={"callbacks": [handler]}`` to LangChain ``invoke()``.

    Respects ``LANGFUSE_ENABLED`` — when ``false`` (the default) the
    handler is never initialised and all calls return ``None``.
    """
    global _handler
    if _handler is not None:
        return _handler

    s = get_settings()
    if not s.langfuse.enabled:
        _log.debug("observability.langfuse_disabled")
        return None

    try:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler

        Langfuse(
            public_key=s.langfuse.public_key or None,
            secret_key=s.langfuse.secret_key or None,
            host=s.langfuse.base_url or None,
        )

        _handler = CallbackHandler()
        _log.info(
            "observability.langfuse_initialised",
            base_url=s.langfuse.base_url,
        )
    except Exception as exc:
        _log.error(
            "observability.langfuse_init_failed",
            error=str(exc),
        )

    return _handler


def reset_langfuse_handler() -> None:
    """Reset the singleton — useful in tests."""
    global _handler
    _handler = None
