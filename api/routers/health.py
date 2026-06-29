"""
api/routers/health.py
──────────────────────
GET /health — Deep health check with per-dependency probes.

Returns HTTP 200 (all OK), 206 (degraded), or 503 (critical down).

Checks performed:
  - Ollama:      GET /api/tags — verifies daemon is running + model listed
  - ChromaDB:    collection.count() — verifies persistence layer is writable
  - Embeddings:  EmbeddingModel.is_loaded — verifies model is in memory
  - Disk:        data/chroma directory is writable

Each check is run concurrently via asyncio.gather() with a per-check
timeout so a single slow dependency cannot delay the health response.
"""

from __future__ import annotations

import asyncio
import time

import httpx
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from api.dependencies import VectorStoreDep, get_settings_dep
from api.schemas.models import DependencyHealth, HealthResponse, ServiceStatus
from app_logging.logger import get_logger

_log = get_logger(__name__)

router = APIRouter(tags=["Health"])

_CHECK_TIMEOUT_S = 5.0

_START_TIME = time.perf_counter()


async def _check_ollama(settings) -> DependencyHealth:
    t = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=_CHECK_TIMEOUT_S) as client:
            r = await client.get(f"{settings.llm.ollama_base_url}/api/tags")
            r.raise_for_status()
            models = [m.get("name", "") for m in r.json().get("models", [])]
            model_name = settings.llm.ollama_model
            model_prefix = model_name.split(":")[0]
            model_found = any(m.startswith(model_prefix) for m in models)
            latency_ms = (time.perf_counter() - t) * 1000

            if not model_found:
                return DependencyHealth(
                    status=ServiceStatus.DEGRADED,
                    latency_ms=round(latency_ms, 2),
                    detail=(
                        f"Ollama running but model '{model_name}' not found. "
                        f"Run: ollama pull {model_name}"
                    ),
                )
            return DependencyHealth(
                status=ServiceStatus.OK,
                latency_ms=round(latency_ms, 2),
                detail=f"Model '{model_name}' ready",
            )

    except httpx.ConnectError:
        latency_ms = (time.perf_counter() - t) * 1000
        return DependencyHealth(
            status=ServiceStatus.DOWN,
            latency_ms=round(latency_ms, 2),
            detail="Cannot connect to Ollama. Run: ollama serve",
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - t) * 1000
        return DependencyHealth(
            status=ServiceStatus.DEGRADED,
            latency_ms=round(latency_ms, 2),
            detail=f"Ollama probe error: {type(exc).__name__}: {exc}",
        )


async def _check_chromadb(vector_store) -> DependencyHealth:
    t = time.perf_counter()
    try:
        loop = asyncio.get_event_loop()
        info = await asyncio.wait_for(
            loop.run_in_executor(None, vector_store.get_collection_info),
            timeout=_CHECK_TIMEOUT_S,
        )
        latency_ms = (time.perf_counter() - t) * 1000
        return DependencyHealth(
            status=ServiceStatus.OK,
            latency_ms=round(latency_ms, 2),
            detail=f"Collection '{info.name}' \u2014 {info.document_count:,} documents",
        )
    except asyncio.TimeoutError:
        latency_ms = (time.perf_counter() - t) * 1000
        return DependencyHealth(
            status=ServiceStatus.DEGRADED,
            latency_ms=round(latency_ms, 2),
            detail="ChromaDB health check timed out",
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - t) * 1000
        return DependencyHealth(
            status=ServiceStatus.DOWN,
            latency_ms=round(latency_ms, 2),
            detail=f"ChromaDB error: {type(exc).__name__}: {exc}",
        )


async def _check_embeddings() -> DependencyHealth:
    t = time.perf_counter()
    try:
        from rag.embeddings import EmbeddingModel
        model = EmbeddingModel.get_instance()
        latency_ms = (time.perf_counter() - t) * 1000
        h = model.health()
        return DependencyHealth(
            status=ServiceStatus.OK if h["is_loaded"] else ServiceStatus.DEGRADED,
            latency_ms=round(latency_ms, 2),
            detail=(
                f"Model '{h['model_name']}' loaded on {h['device']}, "
                f"dim={h['embedding_dim']}"
                if h["is_loaded"]
                else f"Model '{h['model_name']}' not yet loaded \u2014 "
                     "will load on first request"
            ),
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - t) * 1000
        return DependencyHealth(
            status=ServiceStatus.DEGRADED,
            latency_ms=round(latency_ms, 2),
            detail=f"Embedding probe error: {type(exc).__name__}: {exc}",
        )


async def _check_disk(settings) -> DependencyHealth:
    t = time.perf_counter()
    try:
        persist_dir = settings.chroma.persist_dir
        test_file = persist_dir / ".health_probe"
        test_file.touch()
        test_file.unlink()
        latency_ms = (time.perf_counter() - t) * 1000
        return DependencyHealth(
            status=ServiceStatus.OK,
            latency_ms=round(latency_ms, 2),
            detail=f"'{persist_dir}' writable",
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - t) * 1000
        return DependencyHealth(
            status=ServiceStatus.DOWN,
            latency_ms=round(latency_ms, 2),
            detail=f"Disk check failed: {exc}",
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Deep health check",
    description=(
        "Probes all external dependencies concurrently. "
        "Returns 200 (all OK), 206 (degraded), or 503 (critical dependency down)."
    ),
    responses={
        200: {"description": "All systems operational"},
        206: {"description": "One or more dependencies degraded"},
        503: {"description": "Critical dependency unavailable"},
    },
)
async def health(
    vector_store=VectorStoreDep,
    settings=Depends(get_settings_dep),
) -> JSONResponse:
    t_start = time.perf_counter()

    ollama_h, chroma_h, embed_h, disk_h = await asyncio.gather(
        _check_ollama(settings),
        _check_chromadb(vector_store),
        _check_embeddings(),
        _check_disk(settings),
        return_exceptions=False,
    )

    dependencies = {
        "ollama":     ollama_h,
        "chromadb":   chroma_h,
        "embeddings": embed_h,
        "disk":       disk_h,
    }

    statuses = {h.status for h in dependencies.values()}
    if ServiceStatus.DOWN in statuses:
        overall = ServiceStatus.DOWN
        http_code = 503
    elif ServiceStatus.DEGRADED in statuses:
        overall = ServiceStatus.DEGRADED
        http_code = 206
    else:
        overall = ServiceStatus.OK
        http_code = 200

    uptime_s = time.perf_counter() - _START_TIME
    elapsed_ms = (time.perf_counter() - t_start) * 1000

    _log.info(
        "health.checked",
        overall=overall,
        elapsed_ms=round(elapsed_ms, 2),
        statuses={k: v.status for k, v in dependencies.items()},
    )

    response_body = HealthResponse(
        status=overall,
        version=settings.app.version,
        environment=settings.app.env,
        uptime_s=round(uptime_s, 1),
        dependencies=dependencies,
    )

    return JSONResponse(
        content=response_body.model_dump(mode="json"),
        status_code=http_code,
    )
