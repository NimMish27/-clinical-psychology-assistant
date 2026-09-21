"""
knowledge_pipeline/indexer.py
═════════════════════════════
Embeds built child chunks and stores them in ChromaDB, then records the
active collection alias in ``knowledge/index/manifest.lock``.

Collection naming (schema.yaml::collections):
    kb_{framework}_{index_version}    e.g. kb_act_2026-08

The manifest.lock is the indirection layer the retrieval code resolves at
runtime so the active index can be swapped without touching code.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from app_logging.logger import get_logger
from knowledge_pipeline.paths import (
    CHUNKS_JSONL,
    INDEX_DIR,
    MANIFEST_LOCK,
    VECTOR_STORE_DIR,
)
from rag.embeddings import EmbeddingModel, load_embedding_model
from rag.vector_store import VectorStore

if TYPE_CHECKING:
    from pathlib import Path

_log = get_logger(__name__)


class IndexerError(RuntimeError):
    pass


def _read_chunks(path: Path) -> list[dict]:
    if not path.is_file():
        raise IndexerError(f"chunks.jsonl not found at '{path}'. Run the build step first.")
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _sanitize_metadata(meta: dict) -> dict:
    """Make a chunk's metadata acceptable to ChromaDB.

    ChromaDB 1.5.x rejects *empty* list metadata values. ``None`` is allowed,
    so empty lists become None (the full list remains intact in the JSONL).
    """
    return {k: (None if v == [] else v) for k, v in meta.items()}


def _write_manifest_lock(entries: dict[str, dict]) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_LOCK.write_text(
        json.dumps(entries, sort_keys=True, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _log.info("indexer.manifest_lock_written", path=str(MANIFEST_LOCK))


def _read_existing_lock() -> dict[str, dict]:
    if not MANIFEST_LOCK.is_file():
        return {}
    try:
        return json.loads(MANIFEST_LOCK.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def index_chunks(
    chunks_path: Path = CHUNKS_JSONL,
    *,
    force_embed: bool = True,
) -> dict[str, dict]:
    """Embed chunks and upsert them into per-framework Chroma collections.

    Args:
        chunks_path: Path to built chunks.jsonl.
        force_embed: If True, re-embeds every chunk (idempotent upsert).
                     If False, skips chunks whose content_hash is unchanged.

    Returns:
        The manifest.lock mapping: {framework: {collection, ...}}.
    """
    rows = _read_chunks(chunks_path)
    if not rows:
        raise IndexerError("chunks.jsonl is empty — nothing to index.")

    model = load_embedding_model()
    embedding_model_name = model.model_name

    # Group by (framework, index_version) → one collection each.
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        framework = row.get("framework")
        index_version = row.get("index_version")
        if not framework or not index_version:
            _log.warning("indexer.row_missing_key", chunk_id=row.get("chunk_id"))
            continue
        groups.setdefault((framework, index_version), []).append(row)

    lock = _read_existing_lock()
    now = datetime.now(UTC).isoformat()

    for (framework, index_version), group_rows in sorted(groups.items()):
        collection_name = f"kb_{framework}_{index_version}"
        store = VectorStore(
            collection_name=collection_name,
            persist_dir=VECTOR_STORE_DIR,
        )
        store.create_collection(
            exist_ok=True,
            metadata={
                "framework": framework,
                "index_version": index_version,
                "embedding_model": embedding_model_name,
                "built_at": now,
            },
        )

        texts = [r["text"] for r in group_rows]
        ids = [r["chunk_id"] for r in group_rows]
        metadatas = [
            _sanitize_metadata({**r, "index_status": "embedded", "updated_at": now})
            for r in group_rows
        ]

        # Incremental skip: only embed rows whose content_hash changed.
        to_embed_ids = ids
        if not force_embed:
            current = store.peek(len(ids))
            existing = {d["id"]: d.get("metadata", {}) for d in current}
            to_embed_ids = [
                cid
                for cid, row in zip(ids, metadatas, strict=False)
                if existing.get(cid, {}).get("content_hash") != row.get("content_hash")
            ]
            _log.info(
                "indexer.incremental_plan",
                collection=collection_name,
                total=len(ids),
                changed=len(to_embed_ids),
            )
            if not to_embed_ids:
                lock[framework] = _lock_entry(
                    framework, index_version, embedding_model_name, now, group_rows
                )
                _write_manifest_lock(lock)
                continue

        index_map = {cid: i for i, cid in enumerate(ids)}
        embed_indices = [index_map[cid] for cid in to_embed_ids]
        embed_texts = [texts[i] for i in embed_indices]

        embedding_result = model.embed_documents(
            embed_texts,
            sources=[metadatas[i].get("source", "unknown") for i in embed_indices],
            pages=[int(metadatas[i].get("page", 0)) for i in embed_indices],
        )
        embedded_vectors = embedding_result.embeddings_only()

        # Upsert the embedded rows; leave unchanged rows untouched (upsert keeps them).
        upsert_ids = to_embed_ids
        upsert_docs = [texts[index_map[cid]] for cid in to_embed_ids]
        upsert_metas = [metadatas[index_map[cid]] for cid in to_embed_ids]

        insert = store.add_documents(
            ids=upsert_ids,
            documents=upsert_docs,
            embeddings=embedded_vectors,
            metadatas=upsert_metas,
        )

        info = store.get_collection_info()
        lock[framework] = _lock_entry(
            framework, index_version, embedding_model_name, now, group_rows
        )

        _log.info(
            "indexer.collection_indexed",
            collection=collection_name,
            inserted=insert.total_inserted,
            total_docs=info.document_count,
        )

    _write_manifest_lock(lock)
    return lock


def _lock_entry(
    framework: str,
    index_version: str,
    embedding_model: str,
    built_at: str,
    rows: list[dict],
) -> dict:
    books = sorted({str(r.get("book_slug")) for r in rows if r.get("book_slug")})
    return {
        "collection": f"kb_{framework}_{index_version}",
        "index_version": index_version,
        "framework": framework,
        "embedding_model": embedding_model,
        "active": True,
        "built_at": built_at,
        "books": books,
        "chunk_count": len(rows),
    }


def read_active_collection(framework: str) -> str | None:
    """Resolve the active Chroma collection for a framework via manifest.lock.

    Used by the retrieval layer to avoid hard-coding collection names.
    """
    lock = _read_existing_lock()
    entry = lock.get(framework)
    if not entry:
        return None
    return entry.get("collection")


# Re-export for convenience.
__all__ = [
    "IndexerError",
    "index_chunks",
    "read_active_collection",
    "EmbeddingModel",
]
