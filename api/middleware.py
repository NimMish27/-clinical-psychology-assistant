"""
api/middleware.py
──────────────────
Request logging and process time middleware for FastAPI.

RequestLoggingMiddleware:
  - Assigns or propagates an X-Request-ID header per request.
  - Binds request context (request_id, path, method) to structlog.

ProcessTimeMiddleware:
  - Measures wall-clock time and sets the X-Process-Time-Ms response header.
"""

from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app_logging.logger import bind_request_context, clear_request_context, get_logger

_log = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id

        bind_request_context(
            request_id=request_id,
            path=request.url.path,
            method=request.method,
        )

        _log.info(
            "http.request",
            path=request.url.path,
            method=request.method,
            query_params=str(request.url.query),
        )

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        clear_request_context()

        return response


class ProcessTimeMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        t_start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
        return response
