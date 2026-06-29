"""
api/main.py
────────────
FastAPI application factory for the Clinical Psychology Assistant API.

Structure
─────────
  create_app()          \u2190 application factory (testable, configurable)
    |
    \u2514\u2500 lifespan()       \u2190 startup: warm up model, create collection
    |                     shutdown: log graceful stop
    \u2514\u2500 Middleware stack
    |   \u2514\u2500 CORSMiddleware
    |   \u2514\u2500 RequestLoggingMiddleware
    |   \u2514\u2500 ProcessTimeMiddleware
    |
    \u2514\u2500 Exception handlers
    |   \u2514\u2500 RequestValidationError \u2192 422 with structured body
    |   \u2514\u2500 HTTPException         \u2192 consistent JSON envelope
    |
    \u2514\u2500 Routers
        \u2514\u2500 POST /chat
        \u2514\u2500 POST /ingest
        \u2514\u2500 GET  /health

Running the server
──────────────────
    # Development (auto-reload):
    uvicorn api.main:app --reload --port 8000

    # Production (multiple workers):
    uvicorn api.main:app --workers 4 --port 8000 --no-access-log

    # Via the CLI entry point:
    python -m api.main
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from api.middleware import ProcessTimeMiddleware, RequestLoggingMiddleware
from api.routers import chat, clinical, health, ingest
from api.schemas.models import ErrorDetail
from app_logging.logger import get_logger, setup_logging

_log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from config.settings import get_settings
    settings = get_settings()

    setup_logging(
        level=settings.logging.level,
        log_format=settings.logging.format,
        log_file=settings.logging.file,
        rotation=settings.logging.rotation,
        retention=settings.logging.retention,
    )

    _log.info(
        "api.startup",
        version=settings.app.version,
        env=settings.app.env,
        model=settings.llm.ollama_model,
        collection=settings.chroma.collection_name,
    )

    # Embedding model warm-up
    try:
        from rag.embeddings import load_embedding_model
        _log.info("api.loading_embedding_model", model=settings.embedding.model_name)
        load_embedding_model()
        _log.info("api.embedding_model_ready")
    except Exception as exc:
        _log.warning(
            "api.embedding_model_load_failed",
            error=str(exc),
            hint="Embedding model will be loaded on first request (cold start penalty).",
        )

    # ChromaDB collection
    try:
        from rag.vector_store import get_vector_store
        store = get_vector_store()
        info = store.create_collection(exist_ok=True)
        _log.info(
            "api.collection_ready",
            collection=info.name,
            documents=info.document_count,
        )
    except Exception as exc:
        _log.warning(
            "api.collection_init_failed",
            error=str(exc),
            hint="ChromaDB collection will be initialised on first use.",
        )

    _log.info("api.ready")
    yield

    _log.info("api.shutdown")


def _sanitize_errors(errors: list[dict]) -> list[dict]:
    """Convert non-serializable ``ctx.error`` to string for logging / JSON."""
    sanitized = []
    for err in errors:
        e = dict(err)
        ctx = e.get("ctx")
        if ctx and isinstance(ctx, dict) and "error" in ctx:
            ctx = dict(ctx)
            ctx["error"] = str(ctx["error"])
            e["ctx"] = ctx
        sanitized.append(e)
    return sanitized


async def _validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    errors = exc.errors()
    detail = "; ".join(
        f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}"
        for e in errors
    )
    sanitized = _sanitize_errors(errors)
    _log.warning(
        "api.validation_error",
        path=request.url.path,
        errors=sanitized,
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorDetail(
            code="VALIDATION_ERROR",
            message=detail,
            details={"errors": sanitized},
        ).model_dump(),
    )


async def _http_exception_handler(
    request: Request,
    exc: HTTPException,
) -> JSONResponse:
    code_map = {
        400: "BAD_REQUEST",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        413: "PAYLOAD_TOO_LARGE",
        422: "VALIDATION_ERROR",
        429: "RATE_LIMITED",
        503: "SERVICE_UNAVAILABLE",
    }
    code = code_map.get(exc.status_code, f"HTTP_{exc.status_code}")

    if exc.status_code >= 500:
        _log.error(
            "api.http_error",
            status_code=exc.status_code,
            path=request.url.path,
            detail=exc.detail,
        )
    else:
        _log.warning(
            "api.http_error",
            status_code=exc.status_code,
            path=request.url.path,
        )

    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorDetail(
            code=code,
            message=str(exc.detail),
        ).model_dump(),
    )


async def _unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    _log.error(
        "api.unhandled_exception",
        path=request.url.path,
        error_type=type(exc).__name__,
        error=str(exc),
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorDetail(
            code="INTERNAL_SERVER_ERROR",
            message="An unexpected error occurred. Please try again or contact support.",
        ).model_dump(),
    )


def create_app(*, settings=None) -> FastAPI:
    if settings is None:
        from config.settings import get_settings
        settings = get_settings()

    app = FastAPI(
        title=settings.app.name,
        version=settings.app.version,
        description=(
            "RAG-powered clinical psychology assistant for psychologists and clinicians. "
            "Powered by Llama 3.1 8B via Ollama, BAAI/bge-large-en-v1.5 embeddings, "
            "and ChromaDB vector storage."
        ),
        docs_url="/docs" if not settings.is_production() else None,
        redoc_url="/redoc" if not settings.is_production() else None,
        openapi_url="/openapi.json" if not settings.is_production() else None,
        lifespan=lifespan,
    )

    # ── Middleware (registered in reverse — last registered = outermost) ──────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.app.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Process-Time-Ms"],
    )
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(ProcessTimeMiddleware)

    # ── Exception handlers ────────────────────────────────────────────────────
    app.add_exception_handler(RequestValidationError, _validation_exception_handler)
    app.add_exception_handler(HTTPException, _http_exception_handler)
    app.add_exception_handler(Exception, _unhandled_exception_handler)

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(chat.router,   prefix="/api/v1")
    app.include_router(ingest.router, prefix="/api/v1")
    app.include_router(health.router)
    app.include_router(clinical.router)

    _log.debug(
        "api.app_created",
        routes=[r.path for r in app.routes],
    )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    from config.settings import get_settings

    s = get_settings()
    uvicorn.run(
        "api.main:app",
        host=s.server.host,
        port=s.server.port,
        reload=s.server.reload and not s.is_production(),
        workers=s.server.workers if s.is_production() else 1,
        log_level=s.logging.level.lower(),
        access_log=False,
    )
