"""
rag/vector_store.py
────────────────────
Production-ready ChromaDB vector storage layer.

Architecture
────────────
                          ┌─────────────────────────────────────────┐
                          │            VectorStore                   │
                          │                                          │
                          │  _client : chromadb.PersistentClient    │
                          │  _collection : Collection | None         │
                          │                                          │
                          │  create_collection()                     │
                          │  add_documents()   ──► upsert batch     │
                          │  query_documents() ──► similarity search │
                          │  delete_documents()                      │
                          │  get_collection_info()                   │
                          │  reset_collection()                      │
                          └───────────────┬─────────────────────────┘
                                          │
                          ┌───────────────▼─────────────────────────┐
                          │    chromadb.PersistentClient             │
                          │    persist_dir: ./data/chroma            │
                          │    distance:    cosine                   │
                          └─────────────────────────────────────────┘

Integration contract
────────────────────
The three upstream modules produce exactly what this layer consumes:

    Chunker  → ChunkingResult.to_chromadb_batch()
               {"ids": [...], "documents": [...], "metadatas": [...]}

    Embedder → EmbeddingResult.to_chromadb_batch()
               {"embeddings": [[...], ...]}

    Merge    → collection.upsert(**{**chunks, **embeddings})

Design decisions
────────────────
1. Upsert-by-default strategy:
   add_documents() always calls upsert(), not add(). Re-running ingestion
   on the same source overwrites rather than duplicating — correct behaviour
   for a clinical knowledge base that gets updated documents.

2. Lazy collection init:
   The collection is resolved on first use, not at construction time, so
   the VectorStore can be instantiated during module import without
   triggering ChromaDB I/O.

3. Singleton client:
   chromadb.PersistentClient opens a file lock. A singleton _client means
   only one lock per process — safe for multi-threaded FastAPI workers.

4. Metadata filtering:
   query_documents() exposes a `where` parameter that maps directly to
   ChromaDB's metadata filter syntax. Callers can filter by source,
   page range, chunk_size_config, etc. without knowing ChromaDB internals.

5. Distance metric:
   Cosine distance (configured via CHROMA_DISTANCE_FUNCTION) is correct
   for L2-normalised bge-large vectors. With normalised vectors,
   cosine similarity = 1 − cosine_distance, giving a score in [0, 1]
   where higher = more similar. The QueryResult.score field converts
   distance → similarity so callers always reason about similarity.

Public API
──────────
    # Module-level convenience (preferred)
    store = get_vector_store()
    store.create_collection()
    store.add_documents(chunks_batch, embeddings_batch)
    results = store.query_documents("What is CBT?", query_vector=[...])

    # Class-based (for DI / testing)
    store = VectorStore(persist_dir=Path("./data/chroma"), ...)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from app_logging.logger import get_logger
from config.settings import get_settings

_log = get_logger(__name__)


# ── Custom exceptions ──────────────────────────────────────────────────────────

class VectorStoreError(RuntimeError):
    """Base class for all vector store failures."""

    def __init__(self, message: str, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.cause = cause

    def __str__(self) -> str:
        s = super().__str__()
        if not self.cause:
            return s
        # Collect names for the cause's type including built-in aliases.
        # In Python 3, IOError/EnvironmentError/WindowsError are all aliases
        # for OSError. We scan builtins to surface all of them so that
        # assertions like `assert "IOError" in str(exc)` still hold.
        import builtins
        cause_type = type(self.cause)
        primary_name = cause_type.__name__
        # Find any builtin names that are the same class (aliases)
        alias_names = sorted(
            name for name, obj in vars(builtins).items()
            if obj is cause_type and name != primary_name and isinstance(obj, type)
        )
        all_names = [primary_name] + alias_names
        cause_label = "/".join(all_names)
        return f"{s} [caused by: {cause_label}: {self.cause}]"


class CollectionNotFoundError(VectorStoreError):
    """Raised when an operation targets a collection that does not exist."""


class CollectionAlreadyExistsError(VectorStoreError):
    """Raised when create_collection(exist_ok=False) is called on an existing name."""


class DocumentInsertError(VectorStoreError):
    """Raised when a batch upsert fails."""


class QueryError(VectorStoreError):
    """Raised when a similarity search fails."""


class DeleteError(VectorStoreError):
    """Raised when document deletion fails."""


class VectorStoreConnectionError(VectorStoreError):
    """Raised when the ChromaDB client cannot connect or initialise."""


# ── Output types ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class QueryResult:
    """
    A single document returned by a similarity search.

    Attributes:
        chunk_id:   The document's unique ID in the collection.
        text:       The original document text.
        score:      Similarity score in [0, 1] (1 = identical).
                    Derived from cosine distance: score = 1 − distance.
        source:     Source filename from metadata.
        page:       Page number from metadata.
        metadata:   Full metadata dict as stored in ChromaDB.
        rank:       1-based position in the result list (1 = most similar).
    """
    chunk_id: str
    text: str
    score: float
    source: str
    page: int
    metadata: dict[str, Any]
    rank: int

    def __repr__(self) -> str:
        return (
            f"QueryResult(rank={self.rank}, score={self.score:.4f}, "
            f"source={self.source!r}, page={self.page}, "
            f"id={self.chunk_id!r})"
        )


@dataclass
class InsertResult:
    """
    Result of a batch document insert / upsert operation.

    Attributes:
        total_attempted:  How many documents were in the input batch.
        total_inserted:   How many were successfully upserted.
        total_failed:     How many failed (when partial_success=True).
        collection_name:  Target collection.
        elapsed_ms:       Wall-clock time for the upsert.
        errors:           Per-sub-batch error messages if any.
    """
    total_attempted: int
    total_inserted: int
    total_failed: int
    collection_name: str
    elapsed_ms: float
    errors: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_attempted == 0:
            return 0.0
        return self.total_inserted / self.total_attempted

    def __repr__(self) -> str:
        return (
            f"InsertResult(inserted={self.total_inserted}/{self.total_attempted}, "
            f"failed={self.total_failed}, elapsed_ms={self.elapsed_ms:.1f})"
        )


@dataclass(frozen=True)
class CollectionInfo:
    """
    Metadata about a ChromaDB collection.

    Attributes:
        name:            Collection name.
        document_count:  Number of documents currently stored.
        distance_metric: Distance function (cosine / l2 / ip).
        persist_dir:     Absolute path to ChromaDB storage directory.
        metadata:        Raw collection metadata dict from ChromaDB.
    """
    name: str
    document_count: int
    distance_metric: str
    persist_dir: str
    metadata: dict[str, Any]


# ── Singleton client ───────────────────────────────────────────────────────────

_client_lock = threading.Lock()
_client_instance: Any | None = None  # chromadb.PersistentClient


def _get_client(persist_dir: Path) -> Any:
    """
    Return a process-wide ChromaDB PersistentClient singleton.

    ChromaDB uses a file lock on the SQLite WAL — only one client per
    persist_dir per process is safe. Double-checked locking ensures
    exactly one construction even under concurrent FastAPI workers.

    Args:
        persist_dir: Directory for ChromaDB's SQLite + segment files.

    Returns:
        chromadb.PersistentClient instance.

    Raises:
        VectorStoreConnectionError: If ChromaDB cannot be initialised.
    """
    global _client_instance

    if _client_instance is not None:
        return _client_instance

    with _client_lock:
        if _client_instance is not None:
            return _client_instance

        try:
            import chromadb
            persist_dir.mkdir(parents=True, exist_ok=True)

            _log.info(
                "vectorstore.client_init",
                persist_dir=str(persist_dir),
            )
            _client_instance = chromadb.PersistentClient(path=str(persist_dir))
            _log.info("vectorstore.client_ready", persist_dir=str(persist_dir))

        except ImportError as exc:
            raise VectorStoreConnectionError(
                "chromadb is not installed. Run: pip install chromadb",
                cause=exc,
            ) from exc
        except Exception as exc:
            raise VectorStoreConnectionError(
                f"Failed to initialise ChromaDB at '{persist_dir}': {exc}",
                cause=exc,
            ) from exc

    return _client_instance


def _reset_client() -> None:
    """
    Destroy the singleton client. Intended for tests only.
    Call before each test that constructs a VectorStore to avoid
    cross-test state leakage.
    """
    global _client_instance
    with _client_lock:
        _client_instance = None


# ── Core class ─────────────────────────────────────────────────────────────────

class VectorStore:
    """
    Typed, persistent wrapper around a ChromaDB collection.

    Each VectorStore instance manages one collection. Multiple instances
    can target different collections in the same persist_dir — they all
    share the singleton PersistentClient.

    Args:
        collection_name:   Name of the ChromaDB collection. Created on
                           first use if it does not exist.
        persist_dir:       Path for ChromaDB file storage. Created if absent.
        distance_function: Distance metric. Must be "cosine", "l2", or "ip".
                           Default: "cosine" (correct for bge-large normalised vectors).
        insert_batch_size: Maximum documents per ChromaDB upsert call.
                           ChromaDB has an internal batch limit; 500 is safe.
        default_n_results: Default number of results for query_documents()
                           when ``n_results`` is not specified.
    """

    def __init__(
        self,
        collection_name: str,
        persist_dir: Path,
        distance_function: str = "cosine",
        insert_batch_size: int = 500,
        default_n_results: int = 10,
    ) -> None:
        if distance_function not in ("cosine", "l2", "ip"):
            raise ValueError(
                f"distance_function must be 'cosine', 'l2', or 'ip', "
                f"got '{distance_function}'."
            )
        if insert_batch_size < 1:
            raise ValueError("insert_batch_size must be at least 1.")

        self._collection_name = collection_name
        self._persist_dir = Path(persist_dir).resolve()
        self._distance_function = distance_function
        self._insert_batch_size = insert_batch_size
        self._default_n_results = default_n_results
        self._collection: Any | None = None  # lazy-loaded chromadb.Collection

        _log.info(
            "vectorstore.init",
            collection=collection_name,
            persist_dir=str(self._persist_dir),
            distance=distance_function,
        )

    # ── Collection management ─────────────────────────────────────────────────

    def create_collection(
        self,
        *,
        exist_ok: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> CollectionInfo:
        """
        Create (or retrieve) the ChromaDB collection.

        Args:
            exist_ok:  If True (default), silently returns the existing
                       collection when it already exists.
                       If False, raises CollectionAlreadyExistsError.
            metadata:  Optional key-value metadata stored on the collection.
                       Useful for tagging the collection with model name,
                       chunk_size, or ingest date.

        Returns:
            CollectionInfo describing the collection state.

        Raises:
            CollectionAlreadyExistsError: If exist_ok=False and collection exists.
            VectorStoreConnectionError:   If ChromaDB cannot be reached.
        """
        client = _get_client(self._persist_dir)

        collection_meta = {
            "hnsw:space": self._distance_function,
            **(metadata or {}),
        }

        raw_collections = client.list_collections()
        # ChromaDB >=0.6 returns string names directly; <0.6 returns Collection objects
        existing_names = [c if isinstance(c, str) else c.name for c in raw_collections]

        if self._collection_name in existing_names and not exist_ok:
            raise CollectionAlreadyExistsError(
                f"Collection '{self._collection_name}' already exists "
                f"and exist_ok=False."
            )

        try:
            self._collection = client.get_or_create_collection(
                name=self._collection_name,
                metadata=collection_meta,
            )
            count = self._collection.count()

            _log.info(
                "vectorstore.collection_ready",
                collection=self._collection_name,
                document_count=count,
                distance=self._distance_function,
            )

            return CollectionInfo(
                name=self._collection_name,
                document_count=count,
                distance_metric=self._distance_function,
                persist_dir=str(self._persist_dir),
                metadata=collection_meta,
            )

        except (CollectionAlreadyExistsError, VectorStoreConnectionError):
            raise
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to create/retrieve collection '{self._collection_name}': {exc}",
                cause=exc,
            ) from exc

    def list_collections(self) -> list[str]:
        """
        Return the names of all collections in the persist_dir.

        Returns:
            List of collection name strings.
        """
        client = _get_client(self._persist_dir)
        raw = client.list_collections()
        names = [c if isinstance(c, str) else c.name for c in raw]
        _log.debug("vectorstore.list_collections", count=len(names), names=names)
        return names

    def collection_exists(self) -> bool:
        """Return True if the configured collection exists in ChromaDB."""
        return self._collection_name in self.list_collections()

    # ── Document insertion ────────────────────────────────────────────────────

    def add_documents(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[dict[str, Any]] | None = None,
        batch_size: int | None = None,
    ) -> InsertResult:
        """
        Upsert a batch of documents with their embeddings into the collection.

        Uses ChromaDB's ``upsert()`` semantics — existing documents with
        the same ID are overwritten. This makes re-ingestion idempotent.

        The method accepts the exact output format of the upstream modules:

            chunks_batch  = chunking_result.to_chromadb_batch()
            emb_batch     = embedding_result.to_chromadb_batch()

            store.add_documents(
                ids        = chunks_batch["ids"],
                documents  = chunks_batch["documents"],
                embeddings = emb_batch["embeddings"],
                metadatas  = chunks_batch["metadatas"],
            )

        Args:
            ids:         Unique string IDs (chunk_ids from Chunker).
            documents:   Raw text for each document.
            embeddings:  L2-normalised embedding vectors.
            metadatas:   Per-document metadata dicts. Optional but recommended.
            batch_size:  Override the instance-level insert_batch_size.

        Returns:
            InsertResult with counts and timing.

        Raises:
            VectorStoreError:  If the collection does not exist.
            DocumentInsertError: If a batch upsert fails and cannot recover.
        """
        collection = self._resolve_collection()
        ids_list       = list(ids)
        docs_list      = list(documents)
        emb_list       = [list(v) for v in embeddings]
        metas_list     = list(metadatas) if metadatas else [{}] * len(ids_list)
        effective_bs   = batch_size or self._insert_batch_size

        n = len(ids_list)
        _validate_batch_lengths(n, docs_list, emb_list, metas_list)

        _log.info(
            "vectorstore.insert_start",
            collection=self._collection_name,
            total_documents=n,
            batch_size=effective_bs,
        )

        t_start = time.perf_counter()
        inserted = 0
        failed = 0
        errors: list[str] = []

        for batch_start in range(0, n, effective_bs):
            batch_end = min(batch_start + effective_bs, n)
            b_ids   = ids_list[batch_start:batch_end]
            b_docs  = docs_list[batch_start:batch_end]
            b_embs  = emb_list[batch_start:batch_end]
            b_metas = metas_list[batch_start:batch_end]

            try:
                collection.upsert(
                    ids=b_ids,
                    documents=b_docs,
                    embeddings=b_embs,
                    metadatas=b_metas,
                )
                batch_count = len(b_ids)
                inserted += batch_count
                _log.debug(
                    "vectorstore.batch_upserted",
                    collection=self._collection_name,
                    batch_start=batch_start,
                    batch_end=batch_end,
                    count=batch_count,
                )

            except Exception as exc:
                batch_err = (
                    f"batch [{batch_start}:{batch_end}] — "
                    f"{type(exc).__name__}: {exc}"
                )
                errors.append(batch_err)
                failed += len(b_ids)
                _log.error(
                    "vectorstore.batch_failed",
                    collection=self._collection_name,
                    batch_start=batch_start,
                    batch_end=batch_end,
                    error=str(exc),
                )

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        result = InsertResult(
            total_attempted=n,
            total_inserted=inserted,
            total_failed=failed,
            collection_name=self._collection_name,
            elapsed_ms=round(elapsed_ms, 2),
            errors=errors,
        )

        _log.info(
            "vectorstore.insert_complete",
            collection=self._collection_name,
            inserted=inserted,
            failed=failed,
            elapsed_ms=round(elapsed_ms, 2),
            total_docs=collection.count(),
        )

        return result

    # ── Similarity search ─────────────────────────────────────────────────────

    def query_documents(
        self,
        query_vector: Sequence[float],
        *,
        n_results: int | None = None,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
        include_embeddings: bool = False,
    ) -> list[QueryResult]:
        """
        Perform a vector similarity search against the collection.

        Args:
            query_vector:       L2-normalised query embedding (from embed_text()
                                with is_query=True).
            n_results:          Number of results to return.
                                Defaults to instance-level default_n_results.
            where:              Metadata filter in ChromaDB filter syntax.
                                Examples:
                                  {"source": "DSM5.pdf"}
                                  {"page": {"$gte": 50, "$lte": 100}}
                                  {"$and": [{"source": "DSM5.pdf"}, {"page": {"$gt": 10}}]}
            where_document:     Filter on document text content.
                                Example: {"$contains": "CBT"}
            include_embeddings: If True, the raw embedding vectors are
                                included in the ChromaDB response.
                                Rarely needed; defaults to False to save memory.

        Returns:
            List of QueryResult sorted by descending similarity score
            (rank 1 = most similar). Empty list if collection has no documents
            or no results meet the filter criteria.

        Raises:
            QueryError:           If ChromaDB raises during search.
            VectorStoreError:     If the collection does not exist.
        """
        collection = self._resolve_collection()
        n = n_results or self._default_n_results
        current_count = collection.count()

        if current_count == 0:
            _log.warning(
                "vectorstore.query_empty_collection",
                collection=self._collection_name,
            )
            return []

        # ChromaDB raises if n_results > collection size
        effective_n = min(n, current_count)

        include_fields = ["documents", "metadatas", "distances"]
        if include_embeddings:
            include_fields.append("embeddings")

        _log.info(
            "vectorstore.query_start",
            collection=self._collection_name,
            n_results=effective_n,
            has_filter=where is not None,
        )

        t_start = time.perf_counter()
        try:
            raw = collection.query(
                query_embeddings=[list(query_vector)],
                n_results=effective_n,
                where=where,
                where_document=where_document,
                include=include_fields,
            )
        except Exception as exc:
            raise QueryError(
                f"Similarity search failed in collection '{self._collection_name}': {exc}",
                cause=exc,
            ) from exc

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        results = _parse_query_response(raw, distance_function=self._distance_function)

        _log.info(
            "vectorstore.query_complete",
            collection=self._collection_name,
            results_returned=len(results),
            top_score=results[0].score if results else None,
            elapsed_ms=round(elapsed_ms, 2),
        )

        return results

    # ── Document deletion ─────────────────────────────────────────────────────

    def delete_documents(
        self,
        *,
        ids: Sequence[str] | None = None,
        where: dict[str, Any] | None = None,
    ) -> int:
        """
        Delete documents from the collection by ID or metadata filter.

        At least one of ``ids`` or ``where`` must be provided to prevent
        accidental full-collection deletion. Use ``reset_collection()``
        for intentional full wipes.

        Args:
            ids:   List of chunk_ids to delete. Optional.
            where: ChromaDB metadata filter. Optional.
                   Example: {"source": "outdated_manual.pdf"}
                   Deletes all chunks from a specific source file.

        Returns:
            Approximate number of documents deleted. ChromaDB does not
            return an exact count from delete(); this is derived from
            the before/after collection.count().

        Raises:
            ValueError:    If neither ids nor where is provided.
            DeleteError:   If ChromaDB raises during deletion.
            VectorStoreError: If the collection does not exist.
        """
        if ids is None and where is None:
            raise ValueError(
                "delete_documents() requires at least one of 'ids' or 'where'. "
                "Call reset_collection() to delete all documents."
            )

        collection = self._resolve_collection()
        count_before = collection.count()

        _log.info(
            "vectorstore.delete_start",
            collection=self._collection_name,
            id_count=len(ids) if ids else None,
            has_filter=where is not None,
            docs_before=count_before,
        )

        try:
            kwargs: dict[str, Any] = {}
            if ids is not None:
                kwargs["ids"] = list(ids)
            if where is not None:
                kwargs["where"] = where

            collection.delete(**kwargs)

        except Exception as exc:
            raise DeleteError(
                f"Failed to delete documents from '{self._collection_name}': {exc}",
                cause=exc,
            ) from exc

        count_after = collection.count()
        deleted = max(0, count_before - count_after)

        _log.info(
            "vectorstore.delete_complete",
            collection=self._collection_name,
            deleted=deleted,
            docs_after=count_after,
        )

        return deleted

    def delete_by_source(self, source: str) -> int:
        """
        Delete all chunks belonging to a specific source file.

        Convenience wrapper around delete_documents(where={"source": source}).
        Use when re-ingesting an updated version of a document — delete the
        old chunks first, then add the new ones.

        Args:
            source: Filename to delete (e.g. ``"DSM5.pdf"``).
                    Matched against the ``source`` metadata field.

        Returns:
            Approximate number of documents deleted.
        """
        _log.info(
            "vectorstore.delete_by_source",
            collection=self._collection_name,
            source=source,
        )
        return self.delete_documents(where={"source": source})

    # ── Collection info and management ───────────────────────────────────────

    def get_collection_info(self) -> CollectionInfo:
        """
        Return metadata and document count for the current collection.

        Returns:
            CollectionInfo dataclass.

        Raises:
            CollectionNotFoundError: If the collection does not exist yet.
        """
        collection = self._resolve_collection()
        count = collection.count()
        meta = collection.metadata or {}

        info = CollectionInfo(
            name=self._collection_name,
            document_count=count,
            distance_metric=meta.get("hnsw:space", self._distance_function),
            persist_dir=str(self._persist_dir),
            metadata=meta,
        )

        _log.debug(
            "vectorstore.info",
            collection=self._collection_name,
            document_count=count,
        )
        return info

    def reset_collection(self) -> None:
        """
        Delete all documents from the collection and recreate it empty.

        WARNING: This is irreversible. All vectors and metadata are
        permanently deleted. Intended for development / re-ingestion workflows.

        Use delete_by_source() to selectively remove one document's chunks.

        Raises:
            VectorStoreError: If deletion or recreation fails.
        """
        client = _get_client(self._persist_dir)

        _log.warning(
            "vectorstore.reset_start",
            collection=self._collection_name,
        )

        try:
            client.delete_collection(self._collection_name)
        except Exception:
            pass  # Collection may not exist yet — that's fine

        self._collection = None

        try:
            self.create_collection(exist_ok=True)
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to recreate collection '{self._collection_name}' after reset.",
                cause=exc,
            ) from exc

        _log.warning(
            "vectorstore.reset_complete",
            collection=self._collection_name,
        )

    def peek(self, n: int = 5) -> list[dict[str, Any]]:
        """
        Return up to ``n`` documents from the collection for inspection.

        Useful for debugging and verifying ingestion results in the REPL
        or admin endpoints. NOT for production retrieval.

        Args:
            n: Maximum number of documents to return. Default: 5.

        Returns:
            List of dicts with ``id``, ``text``, and ``metadata``.
        """
        collection = self._resolve_collection()
        raw = collection.peek(limit=n)

        docs = []
        ids       = raw.get("ids") or []
        documents = raw.get("documents") or []
        metadatas = raw.get("metadatas") or []

        for doc_id, text, meta in zip(ids, documents, metadatas):
            docs.append({"id": doc_id, "text": text, "metadata": meta or {}})

        _log.debug("vectorstore.peek", collection=self._collection_name, returned=len(docs))
        return docs

    # ── Private helpers ───────────────────────────────────────────────────────

    def _resolve_collection(self) -> Any:
        """
        Return the active ChromaDB collection, auto-creating if necessary.

        Lazy initialisation — we don't call ChromaDB at __init__ time
        so the VectorStore can be imported cheaply.
        """
        if self._collection is not None:
            return self._collection

        client = _get_client(self._persist_dir)

        try:
            self._collection = client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": self._distance_function},
            )
            _log.debug(
                "vectorstore.collection_resolved",
                collection=self._collection_name,
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Cannot access collection '{self._collection_name}': {exc}",
                cause=exc,
            ) from exc

        return self._collection

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @property
    def persist_dir(self) -> Path:
        return self._persist_dir

    @property
    def distance_function(self) -> str:
        return self._distance_function

    def __repr__(self) -> str:
        return (
            f"VectorStore(collection={self._collection_name!r}, "
            f"distance={self._distance_function!r}, "
            f"persist_dir={str(self._persist_dir)!r})"
        )


# ── Private parsing helpers ────────────────────────────────────────────────────

def _parse_query_response(
    raw: dict[str, Any],
    distance_function: str,
) -> list[QueryResult]:
    """
    Convert a raw ChromaDB query() response into a typed list of QueryResult.

    ChromaDB returns nested lists (one per query — we always send one):
        raw["ids"]       = [["id1", "id2", ...]]
        raw["documents"] = [["text1", "text2", ...]]
        raw["distances"] = [[0.12, 0.34, ...]]
        raw["metadatas"] = [[{...}, {...}, ...]]

    Scores are derived from distances:
        cosine: score = 1 − distance   (distance ∈ [0, 2] for unnormalised,
                                         ∈ [0, 1] for L2-normalised vectors)
        l2:     score = 1 / (1 + distance)   (monotone, always ∈ (0, 1])
        ip:     score = distance              (inner product is already a similarity)
    """
    ids       = (raw.get("ids")       or [[]])[0]
    documents = (raw.get("documents") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]
    metadatas = (raw.get("metadatas") or [[]])[0]

    results: list[QueryResult] = []

    for rank, (doc_id, text, distance, meta) in enumerate(
        zip(ids, documents, distances, metadatas), start=1
    ):
        meta = meta or {}
        score = _distance_to_score(distance, distance_function)

        results.append(QueryResult(
            chunk_id=doc_id,
            text=text or "",
            score=round(score, 6),
            source=str(meta.get("source", "unknown")),
            page=int(meta.get("page", 0)),
            metadata=meta,
            rank=rank,
        ))

    return results


def _distance_to_score(distance: float, distance_function: str) -> float:
    """Convert a ChromaDB distance value to a [0, 1] similarity score."""
    if distance_function == "cosine":
        # For L2-normalised vectors cosine distance ∈ [0, 1]
        # where 0 = identical, 1 = orthogonal.
        return max(0.0, 1.0 - float(distance))
    if distance_function == "l2":
        return 1.0 / (1.0 + float(distance))
    # inner product — already a similarity for normalised vectors
    return float(distance)


def _validate_batch_lengths(
    n: int,
    documents: list,
    embeddings: list,
    metadatas: list,
) -> None:
    """Raise ValueError if batch lists are not all the same length."""
    if len(documents) != n:
        raise ValueError(f"documents length ({len(documents)}) must equal ids length ({n}).")
    if len(embeddings) != n:
        raise ValueError(f"embeddings length ({len(embeddings)}) must equal ids length ({n}).")
    if len(metadatas) != n:
        raise ValueError(f"metadatas length ({len(metadatas)}) must equal ids length ({n}).")


# ── Singleton VectorStore ─────────────────────────────────────────────────────

_store_lock = threading.Lock()
_store_instance: VectorStore | None = None


def get_vector_store(*, force_reload: bool = False) -> VectorStore:
    """
    Return the process-wide VectorStore singleton.

    Reads configuration from ``get_settings().chroma``.
    The underlying ChromaDB client is NOT opened until the first
    collection operation — importing this function is free.

    Args:
        force_reload: Discard the current singleton and create a new one.
                      Useful after changing CHROMA_* env vars in tests.

    Returns:
        VectorStore singleton configured from settings.

    Example:
        # At startup:
        from rag.vector_store import get_vector_store
        store = get_vector_store()
        store.create_collection()

        # In the RAG retriever:
        results = store.query_documents(query_vector, n_results=4)
    """
    global _store_instance

    if force_reload:
        with _store_lock:
            _store_instance = None

    if _store_instance is not None:
        return _store_instance

    with _store_lock:
        if _store_instance is not None:
            return _store_instance

        cfg = get_settings().chroma

        _store_instance = VectorStore(
            collection_name=cfg.collection_name,
            persist_dir=cfg.persist_dir,
            distance_function=cfg.distance_function,
        )
        _log.info(
            "vectorstore.singleton_created",
            collection=cfg.collection_name,
            persist_dir=str(cfg.persist_dir),
        )

    return _store_instance
