"""
knowledge_pipeline/builder.py
═════════════════════════════
Orchestrates the build stage: load → validate → chunk → write derived artifacts.

Outputs (all under knowledge/built/, gitignored):
    chunks.jsonl         one child chunk per line
    parents.jsonl        one parent (section) per line
    hash_registry.json   content hashes for incremental re-embedding
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import yaml

from app_logging.logger import get_logger
from knowledge_pipeline.chunker import Chunker
from knowledge_pipeline.loader import (
    chapter_source_files,
    discover_manifests,
    load_chunking_rules,
    load_citations,
    load_manifest,
    load_schema,
    load_source_file,
    resolve_source_path,
)
from knowledge_pipeline.models import BuildResult
from knowledge_pipeline.paths import (
    CHUNKS_JSONL,
    HASH_REGISTRY,
    PARENTS_JSONL,
    ensure_derived_dirs,
)
from knowledge_pipeline.validator import validate_book

if TYPE_CHECKING:
    from pathlib import Path

_log = get_logger(__name__)

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"


def _source_hash(manifest: dict) -> str:
    """sha256 of the whole-book manifest — a change forces full re-embed."""
    serialised = yaml.safe_dump(manifest, sort_keys=True)
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    _log.info("builder.jsonl_written", path=str(path), rows=len(rows))


def build_book(
    manifest_path: Path,
    *,
    schema: dict,
    rules: dict,
    citations: dict[str, str],
    chunker: Chunker,
    embedding_model: str,
) -> tuple[BuildResult, list[dict], list[dict], str | None]:
    """Build one book. Returns (result, parent_rows, chunk_rows, source_hash)."""
    manifest = load_manifest(manifest_path)
    result = BuildResult(built_at=datetime.now(UTC).isoformat())

    source_dir = resolve_source_path(manifest)
    if source_dir is None:
        result.errors.append(f"{manifest.get('book_slug')}: no curated sources yet")
        result.skipped.append(manifest.get("book_slug", ""))
        return result, [], [], None

    files = chapter_source_files(source_dir)
    front_matters = []
    documents = []
    for f in files:
        doc = load_source_file(f)
        documents.append(doc)
        front_matters.append(doc.front_matter)

    try:
        validate_book(manifest, schema, citations, front_matters)
    except Exception as exc:
        result.errors.append(f"{manifest.get('book_slug')}: {exc}")
        return result, [], [], None

    source_hash = _source_hash(manifest)
    index_version = manifest.get("index_version", "")
    parent_rows: list[dict] = []
    chunk_rows: list[dict] = []
    total_chars = 0
    total_tokens = 0

    for doc in documents:
        try:
            parents, children = chunker.chunk_source(
                source=doc,
                manifest=manifest,
                schema=schema,
                source_hash=source_hash,
                embedding_model=embedding_model,
            )
        except Exception as exc:
            result.errors.append(f"{doc.path}: {exc}")
            continue

        for parent in parents:
            parent_rows.append(
                parent.metadata | {"parent_id": parent.parent_id, "text": parent.text}
            )
            total_chars += len(parent.text)
            total_tokens += parent.token_count
        for child in children:
            chunk_rows.append(
                child.metadata | {"chunk_id": child.chunk_id, "text": child.text}
            )
            total_chars += len(child.text)
            total_tokens += child.token_count

    result.source_files = [str(f) for f in files]
    result.parents = len(parent_rows)
    result.children = len(chunk_rows)
    result.total_chars = total_chars
    result.total_tokens = total_tokens
    result.built_at = datetime.now(UTC).isoformat()

    _log.info(
        "builder.book_done",
        book=manifest.get("book_slug"),
        index_version=index_version,
        parents=result.parents,
        children=result.children,
    )
    return result, parent_rows, chunk_rows, source_hash


def build_knowledge_base(
    *,
    framework: str | None = None,
    book_slug: str | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    write_outputs: bool = True,
) -> BuildResult:
    """Build all (or selected) books. Returns an aggregate BuildResult.

    Args:
        framework:     Restrict to one framework's manifests.
        book_slug:     Restrict to a single book.
        embedding_model: Embedding model id stamped into chunk metadata.
        write_outputs: When True, writes chunks.jsonl, parents.jsonl and the
                       hash registry. When False, returns results in memory
                       (used by tests).
    """
    schema = load_schema()
    rules = load_chunking_rules()
    citations = load_citations()
    chunker = Chunker(rules=rules)

    manifests = discover_manifests(framework)
    if book_slug:
        manifests = [m for m in manifests if load_manifest(m).get("book_slug") == book_slug]

    if not manifests:
        _log.warning("builder.no_manifests", framework=framework, book=book_slug)
        return BuildResult(
            built_at=datetime.now(UTC).isoformat(),
            errors=["no manifests matched the given framework/book filters"],
        )

    aggregate = BuildResult(built_at=datetime.now(UTC).isoformat())
    all_parents: list[dict] = []
    all_chunks: list[dict] = []
    books_hashes: dict[str, str] = {}

    for manifest_path in manifests:
        result, parent_rows, chunk_rows, source_hash = build_book(
            manifest_path,
            schema=schema,
            rules=rules,
            citations=citations,
            chunker=chunker,
            embedding_model=embedding_model,
        )
        aggregate.source_files += result.source_files
        aggregate.parents += result.parents
        aggregate.children += result.children
        aggregate.total_chars += result.total_chars
        aggregate.total_tokens += result.total_tokens
        aggregate.errors += result.errors
        aggregate.skipped += result.skipped
        if source_hash:
            books_hashes[str(load_manifest(manifest_path).get("book_slug"))] = source_hash
        all_parents.extend(parent_rows)
        all_chunks.extend(chunk_rows)

    aggregate.built_at = datetime.now(UTC).isoformat()

    if write_outputs and not aggregate.errors:
        ensure_derived_dirs()
        _write_jsonl(PARENTS_JSONL, all_parents)
        _write_jsonl(CHUNKS_JSONL, all_chunks)
        registry = {
            "schema_version": str(schema.get("version", "")),
            "embedding_model": embedding_model,
            "generated_at": aggregate.built_at,
            "books": {
                slug: {
                    "source_hash": h,
                    "chunks": {
                        row["chunk_id"]: {
                            "content_hash": row["content_hash"],
                            "token_count": row["token_count"],
                        }
                        for row in all_chunks
                        if row.get("book_slug") == slug
                    },
                }
                for slug, h in books_hashes.items()
            },
        }
        HASH_REGISTRY.write_text(
            json.dumps(registry, sort_keys=True, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        _log.info("builder.hash_registry_written", path=str(HASH_REGISTRY))

    return aggregate
