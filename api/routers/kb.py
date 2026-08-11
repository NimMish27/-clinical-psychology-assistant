"""
api/routers/kb.py
──────────────────
Knowledge Base management endpoints.

Framework Knowledge (curated, repo-committed):
    GET    /kb/collections   → per-framework build status
    POST   /kb/rebuild       → run the build + index pipeline from the repo

User Documents (uploaded from the UI):
    GET    /kb/documents     → list uploaded source documents
    DELETE /kb/documents     → remove one uploaded source

The router only orchestrates: pipeline logic lives in
``knowledge_pipeline.builder`` / ``knowledge_pipeline.indexer``, and
collection access goes through ``rag.vector_store.VectorStore``.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, Query, status

from api.dependencies import RequestIdDep, SettingsDep
from api.schemas.models import (
    KbCollectionsResponse,
    KbCollectionInfo,
    KbDeleteResponse,
    KbDocumentsResponse,
    KbDocumentSummary,
    KbRebuildRequest,
    KbRebuildResponse,
)
from app_logging.logger import get_logger

if TYPE_CHECKING:
    from config.settings import Settings
    from knowledge_pipeline.models import BuildResult
    from rag.vector_store import VectorStore

_log = get_logger(__name__)

router = APIRouter(prefix="/kb", tags=["Knowledge Base"])

# Curated frameworks that own a knowledge base. Order matters for the UI.
CURATED_FRAMEWORKS = ("cbt", "act", "humanistic")

DEFAULT_USER_COLLECTION = "user_documents"

_rebuild_lock = asyncio.Lock()


def _load_manifest_lock() -> dict[str, dict[str, Any]]:
    """Read ``knowledge/index/manifest.lock``, returning {} when absent."""
    from knowledge_pipeline.paths import MANIFEST_LOCK

    if not MANIFEST_LOCK.exists():
        return {}
    try:
        return json.loads(MANIFEST_LOCK.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        _log.error("kb.manifest_lock_read_failed", error=str(exc))
        return {}


@router.get(
    "/collections",
    response_model=KbCollectionsResponse,
    summary="List curated framework knowledge bases",
)
async def list_collections(request_id: RequestIdDep) -> KbCollectionsResponse:
    lock = _load_manifest_lock()
    entries = []
    for framework in CURATED_FRAMEWORKS:
        info = lock.get(framework)
        if info:
            entries.append(
                KbCollectionInfo(
                    framework=framework,
                    built=True,
                    collection=info.get("collection"),
                    index_version=info.get("index_version"),
                    chunk_count=info.get("chunk_count"),
                    built_at=info.get("built_at"),
                    books=list(info.get("books") or []),
                )
            )
        else:
            entries.append(KbCollectionInfo(framework=framework, built=False))

    _log.info("kb.collections_listed", request_id=request_id, frameworks=len(entries))
    return KbCollectionsResponse(collections=entries)


@router.post(
    "/rebuild",
    response_model=KbRebuildResponse,
    summary="Rebuild curated framework knowledge bases from the repo",
)
async def rebuild_knowledge_base(
    body: KbRebuildRequest,
    request_id: RequestIdDep,
) -> KbRebuildResponse:
    if _rebuild_lock.locked():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A knowledge base rebuild is already in progress.",
        )

    async with _rebuild_lock:
        t_start = time.perf_counter()
        _log.info(
            "kb.rebuild_start",
            request_id=request_id,
            framework=body.framework,
        )

        def _run_pipeline() -> tuple[BuildResult, dict[str, Any] | None]:
            from knowledge_pipeline.builder import build_knowledge_base
            from knowledge_pipeline.indexer import index_chunks

            build = build_knowledge_base(framework=body.framework)
            if build.errors:
                return build, None
            lock = index_chunks(force_embed=False)
            return build, lock

        build, lock = await asyncio.to_thread(_run_pipeline)
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        if lock is not None:
            collections = [
                {
                    "framework": fw,
                    "collection": entry.get("collection"),
                    "chunk_count": entry.get("chunk_count"),
                }
                for fw, entry in sorted(lock.items())
            ]
            _log.info(
                "kb.rebuild_complete",
                request_id=request_id,
                children=build.children,
                collections=len(collections),
                elapsed_ms=round(elapsed_ms, 2),
            )
        else:
            collections = []
            _log.error(
                "kb.rebuild_failed",
                request_id=request_id,
                errors=build.errors,
                elapsed_ms=round(elapsed_ms, 2),
            )

        return KbRebuildResponse(
            status="ok" if lock is not None else "failed",
            built_at=build.built_at,
            source_files=build.source_files,
            parents=build.parents,
            children=build.children,
            total_tokens=build.total_tokens,
            errors=build.errors,
            skipped=build.skipped,
            collections=collections,
        )


def _build_store(collection: str, settings: Settings) -> VectorStore:
    from rag.vector_store import VectorStore

    return VectorStore(
        collection_name=collection,
        persist_dir=settings.chroma.persist_dir,
    )


@router.get(
    "/documents",
    response_model=KbDocumentsResponse,
    summary="List uploaded documents in a user collection",
)
async def list_documents(
    settings: SettingsDep,
    request_id: RequestIdDep,
    collection: str = Query(
        DEFAULT_USER_COLLECTION,
        min_length=1,
        max_length=128,
        description="Collection to list documents from",
    ),
) -> KbDocumentsResponse:
    store = _build_store(collection, settings)
    if not store.collection_exists():
        return KbDocumentsResponse(collection=collection, sources=[])

    sources = [
        KbDocumentSummary(
            source=entry["source"],
            chunks=entry["chunks"],
            user=entry.get("user"),
            project=entry.get("project"),
            document_type=entry.get("document_type"),
            upload_date=entry.get("upload_date"),
        )
        for entry in store.list_sources()
    ]
    _log.info(
        "kb.documents_listed",
        request_id=request_id,
        collection=collection,
        documents=len(sources),
    )
    return KbDocumentsResponse(collection=collection, sources=sources)


@router.delete(
    "/documents",
    response_model=KbDeleteResponse,
    summary="Delete an uploaded document from a user collection",
)
async def delete_document(
    settings: SettingsDep,
    request_id: RequestIdDep,
    collection: str = Query(
        DEFAULT_USER_COLLECTION,
        min_length=1,
        max_length=128,
        description="Collection the document belongs to",
    ),
    source: str = Query(
        ...,
        min_length=1,
        max_length=255,
        description="Source filename to delete (matched against the source metadata field)",
    ),
) -> KbDeleteResponse:
    store = _build_store(collection, settings)
    if not store.collection_exists():
        return KbDeleteResponse(collection=collection, source=source, deleted=0)

    deleted = store.delete_by_source(source)
    _log.info(
        "kb.document_deleted",
        request_id=request_id,
        collection=collection,
        source=source,
        deleted=deleted,
    )
    return KbDeleteResponse(collection=collection, source=source, deleted=deleted)
