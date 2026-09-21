"""
clinical/retrieval/collections.py
================================
Framework-aware Chroma collection resolution for the retrieval layer.

The knowledge pipeline writes ``knowledge/index/manifest.lock`` — an alias
map from framework to its active ``kb_{framework}_{index_version}``
collection. Retrievers resolve that alias at runtime so the active index can
be swapped (e.g. after a re-index) without touching retrieval code. When no
alias is recorded, or the aliased collection does not actually exist yet,
retrieval degrades gracefully to the app-default collection (legacy
behaviour).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from app_logging.logger import get_logger
from rag.vector_store import VectorStore, get_vector_store

if TYPE_CHECKING:
    from pathlib import Path

_log = get_logger(__name__)


def resolve_framework_collection(framework: str) -> str | None:
    """Return the active Chroma collection name for ``framework``.

    Resolves via ``knowledge_pipeline.indexer.read_active_collection``
    (which reads ``manifest.lock``). Returns ``None`` when no alias is
    recorded, the lock is missing/corrupt, or the pipeline package is
    unavailable, so retrieval falls back to the app-default collection
    instead of failing.
    """
    try:
        from knowledge_pipeline.indexer import read_active_collection

        return read_active_collection(framework)
    except Exception:
        return None


def resolve_framework_store(framework: str) -> tuple[VectorStore, str | None]:
    """Return ``(store, collection_name)`` for a framework.

    ``collection_name`` is the active ``manifest.lock`` collection when it
    exists on disk, otherwise ``None`` (meaning: use the app-default
    collection, matching legacy behaviour).
    """
    name = None
    try:
        name = resolve_framework_collection(framework)
    except Exception as exc:
        _log.warning(
            "retrieval.collection_resolution_failed",
            framework=framework,
            error=str(exc),
        )
    if name:
        try:
            store = VectorStore(
                collection_name=name,
                persist_dir=_chroma_persist_dir(),
            )
            if store.collection_exists():
                return store, name
        except Exception as exc:
            _log.warning(
                "retrieval.collection_resolution_failed",
                framework=framework,
                collection=name,
                error=str(exc),
            )
    return get_vector_store(), None


def get_framework_vector_store(framework: str) -> VectorStore:
    """Return the framework's active VectorStore, or the app default."""
    store, _ = resolve_framework_store(framework)
    return store


def _chroma_persist_dir() -> Path:
    from config.settings import get_settings

    return get_settings().chroma.persist_dir
