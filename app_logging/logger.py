"""
logging/logger.py
──────────────────
Structured logging setup using structlog.

Features:
  - JSON output in production / staging (machine-parseable, audit-ready)
  - Pretty console output in development (via Rich)
  - Request-ID and session-ID context binding
  - Clinical audit trail helpers (never logs PII — document metadata only)
  - Automatic file rotation via loguru as the write backend

Usage:
    from logging.logger import get_logger, bind_request_context

    log = get_logger(__name__)
    log.info("retrieval.complete", query_len=42, docs_returned=4, latency_ms=180)

    # In middleware — bind per-request context:
    bind_request_context(request_id="abc123", session_id="sess-xyz")
"""

import logging
import sys
from pathlib import Path
from typing import Any

import structlog
from structlog.types import EventDict, WrappedLogger

# Lazy import — loguru is optional but preferred for file rotation
try:
    from loguru import logger as _loguru_logger
    _HAS_LOGURU = True
except ImportError:
    _HAS_LOGURU = False


# ── Processors ────────────────────────────────────────────────────────────────

def _add_app_metadata(
    logger: WrappedLogger, method: str, event_dict: EventDict
) -> EventDict:
    """Inject static app metadata into every log record."""
    event_dict.setdefault("app", "clinical-psychology-assistant")
    return event_dict


def _censor_pii(
    logger: WrappedLogger, method: str, event_dict: EventDict
) -> EventDict:
    """
    Strip fields that could contain patient-identifiable information.
    Raises if a forbidden key is present so callers fix their log calls.
    """
    _FORBIDDEN = {"patient_name", "patient_id", "dob", "ssn", "email", "phone"}
    found = _FORBIDDEN & set(event_dict.keys())
    if found:
        raise ValueError(
            f"Attempted to log PII fields: {found}. "
            "Remove them before logging or use audit_log() instead."
        )
    return event_dict


# ── File sink (loguru) ────────────────────────────────────────────────────────

def _configure_file_sink(log_file: Path, rotation: str, retention: str) -> None:
    """Add a rotating file sink via loguru if available."""
    if not _HAS_LOGURU:
        return

    _loguru_logger.remove()
    _loguru_logger.add(
        str(log_file),
        rotation=rotation,
        retention=retention,
        serialize=True,          # write as JSON
        level="DEBUG",
        enqueue=True,            # async — non-blocking
        backtrace=True,
        diagnose=False,          # disable in production (may expose locals)
    )


# ── Main setup entry point ────────────────────────────────────────────────────

def setup_logging(
    level: str = "INFO",
    log_format: str = "json",
    log_file: Path | None = None,
    rotation: str = "10 MB",
    retention: str = "30 days",
) -> None:
    """
    Configure structlog globally. Call once at application startup.

    Args:
        level:      Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_format: "json" for production, "console" for development.
        log_file:   Optional path for file logging (uses loguru rotation).
        rotation:   Loguru rotation policy, e.g. "10 MB" or "1 day".
        retention:  Loguru retention policy, e.g. "30 days".
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # ── stdlib logging baseline ───────────────────────────────────────────────
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )
    # Quiet noisy third-party loggers
    for noisy in ("httpx", "httpcore", "chromadb", "urllib3", "multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # ── File sink ─────────────────────────────────────────────────────────────
    if log_file:
        _configure_file_sink(log_file, rotation, retention)

    # ── Shared processors ─────────────────────────────────────────────────────
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        _add_app_metadata,
        _censor_pii,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]

    if log_format == "console":
        # Rich pretty-printing for local development
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    else:
        # Compact JSON for production / file output
        shared_processors.append(structlog.processors.format_exc_info)
        renderer = structlog.processors.JSONRenderer()

    # Use stdlib LoggerFactory so loggers have .name for add_logger_name
    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Return a named, bound structlog logger.

    Example:
        log = get_logger(__name__)
        log.info("rag.retrieve", query="anxiety DSM-5", docs=4, latency_ms=95)
    """
    return structlog.get_logger(name)


# ── Context helpers ───────────────────────────────────────────────────────────

def bind_request_context(**kwargs: Any) -> None:
    """
    Bind per-request key-value pairs to the current async context.
    Call from middleware; all subsequent log calls in the request carry these.

    Example:
        bind_request_context(request_id="req-abc123", session_id="sess-xyz")
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_request_context() -> None:
    """Clear context vars at the end of a request."""
    structlog.contextvars.clear_contextvars()


# ── Clinical audit logger ─────────────────────────────────────────────────────

class AuditLogger:
    """
    Dedicated logger for clinical audit events.
    Writes to a separate structlog logger tagged with audit=True.
    No PII — only document IDs, action types, and clinician IDs.

    Example:
        audit = AuditLogger()
        audit.log_query(clinician_id="dr-jones", query_hash="sha256:abc...", docs_retrieved=4)
    """

    def __init__(self) -> None:
        self._log = structlog.get_logger("audit")

    def log_query(
        self,
        clinician_id: str,
        query_hash: str,
        docs_retrieved: int,
        latency_ms: float,
        **kwargs: Any,
    ) -> None:
        self._log.info(
            "audit.query",
            audit=True,
            clinician_id=clinician_id,
            query_hash=query_hash,
            docs_retrieved=docs_retrieved,
            latency_ms=round(latency_ms, 2),
            **kwargs,
        )

    def log_safety_flag(
        self,
        session_id: str,
        risk_level: str,
        action_taken: str,
        **kwargs: Any,
    ) -> None:
        self._log.warning(
            "audit.safety_flag",
            audit=True,
            session_id=session_id,
            risk_level=risk_level,
            action_taken=action_taken,
            **kwargs,
        )

    def log_document_ingested(
        self,
        doc_id: str,
        source_type: str,
        chunk_count: int,
        **kwargs: Any,
    ) -> None:
        self._log.info(
            "audit.document_ingested",
            audit=True,
            doc_id=doc_id,
            source_type=source_type,
            chunk_count=chunk_count,
            **kwargs,
        )


# Module-level audit logger instance
audit_logger = AuditLogger()
