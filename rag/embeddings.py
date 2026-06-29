"""
rag/embeddings.py
──────────────────
Production-ready text embedding module backed by SentenceTransformers
and the BAAI/bge-large-en-v1.5 model.

Architecture overview
─────────────────────
                    ┌─────────────────────────────┐
                    │        EmbeddingModel        │  ← singleton wrapper
                    │                              │
                    │  _model: SentenceTransformer │  ← loaded once, reused
                    │  _lock:  threading.Lock      │  ← thread-safe init
                    │                              │
                    │  embed_text(str)  → list[f]  │
                    │  embed_documents(...)         │
                    │    → EmbeddingResult         │
                    └──────────────┬───────────────┘
                                   │
              ┌────────────────────▼──────────────────────┐
              │           SentenceTransformer              │
              │         BAAI/bge-large-en-v1.5            │
              │  dim=1024  |  max_seq_len=512 tokens      │
              └────────────────────────────────────────────┘

Key design decisions
────────────────────
1. Singleton via threading.Lock (not module-level global):
   The model (~1.3 GB on CPU) must be loaded exactly once per process.
   A double-checked locking pattern ensures thread-safety without
   a performance penalty on every call after first load.

2. BGE query prefix:
   BAAI/bge-large-en-v1.5 is an asymmetric model — queries and
   documents are embedded differently for best retrieval performance.
   Queries must be prefixed with "Represent this sentence: " at
   embed time. Documents (chunks) are embedded without the prefix.
   This module handles both cases transparently.

3. Batch processing with progress:
   Large corpora (hundreds of chunks) are processed in configurable
   batches. A progress callback allows the API layer to stream status
   to the client without coupling the embedding module to any HTTP layer.

4. Normalised vectors:
   Embeddings are L2-normalised so cosine similarity is equivalent to
   dot-product — the ChromaDB default. Normalisation is enabled in the
   SentenceTransformer encode() call.

5. Pure output types (no numpy in the public API):
   numpy arrays are converted to Python lists before returning, keeping
   the public API dependency-light and JSON-serialisable.

Public API
──────────
    # Module-level convenience functions (preferred)
    load_embedding_model()                   # warm up at startup
    embed_text(text, *, is_query)            # single string → list[float]
    embed_documents(texts, sources, pages)   # batch → EmbeddingResult

    # Class-based API (for DI / testing)
    model = EmbeddingModel.get_instance()
    result = model.embed_documents(...)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

from app_logging.logger import get_logger
from config.settings import get_settings

_log = get_logger(__name__)

# ── BGE model constants ────────────────────────────────────────────────────────

# BAAI/bge-large-en-v1.5 requires this prefix on *query* strings.
# Document chunks are embedded WITHOUT the prefix.
# Reference: https://huggingface.co/BAAI/bge-large-en-v1.5
_BGE_QUERY_PREFIX: str = "Represent this sentence: "

# Embedding dimension for bge-large-en-v1.5
_BGE_EMBEDDING_DIM: int = 1024

# Hard cap on text length before truncation warning is issued.
# bge-large-en-v1.5 has a 512-token context window; ~2048 chars is a
# generous upper bound before silent truncation becomes a concern.
_TRUNCATION_WARN_CHARS: int = 2048


# ── Custom exceptions ─────────────────────────────────────────────────────────

class EmbeddingError(RuntimeError):
    """Base class for embedding failures."""

    def __init__(self, message: str, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.cause = cause

    def __str__(self) -> str:
        base = super().__str__()
        return f"{base} [caused by: {type(self.cause).__name__}: {self.cause}]" if self.cause else base


class ModelLoadError(EmbeddingError):
    """Raised when the SentenceTransformer model cannot be loaded."""


class EmbeddingInferenceError(EmbeddingError):
    """Raised when encoding fails during inference."""


class EmptyInputError(EmbeddingError):
    """Raised when an empty text or empty batch is submitted."""


# ── Output types ──────────────────────────────────────────────────────────────

Vector = list[float]  # a single embedding vector


@dataclass(frozen=True)
class EmbeddedDocument:
    """
    A single embedded document with its source metadata.

    Attributes:
        text:      The original text that was embedded.
        embedding: L2-normalised embedding vector (dim=1024 for bge-large).
        source:    Source filename (e.g. ``DSM5.pdf``).
        page:      1-based page number from the source document.
        dim:       Embedding dimension (convenience, always 1024 for this model).
    """
    text: str
    embedding: Vector
    source: str
    page: int
    dim: int = field(default=_BGE_EMBEDDING_DIM)

    def to_chromadb_embedding(self) -> Vector:
        """Return the embedding vector ready for ChromaDB's ``embeddings`` param."""
        return self.embedding


@dataclass
class EmbeddingResult:
    """
    Batch embedding result for a collection of documents.

    Attributes:
        documents:          List of EmbeddedDocument, one per input text.
        total_embedded:     How many texts were successfully embedded.
        total_failed:       How many texts failed (included as None vectors).
        model_name:         Name of the model used.
        elapsed_ms:         Wall-clock time for the entire batch.
        texts_per_second:   Throughput metric.
        errors:             Error details for any failed texts.
    """
    documents: list[EmbeddedDocument]
    total_embedded: int
    total_failed: int
    model_name: str
    elapsed_ms: float
    texts_per_second: float
    errors: list[str] = field(default_factory=list)

    def embeddings_only(self) -> list[Vector]:
        """Return just the embedding vectors — for ChromaDB batch upsert."""
        return [d.embedding for d in self.documents]

    def to_chromadb_batch(self) -> dict[str, list]:
        """
        Produce the ``embeddings`` component of a ChromaDB batch upsert.
        Pair with ``ChunkingResult.to_chromadb_batch()`` for full upsert payload.

        Returns:
            {"embeddings": [[...], [...], ...]}
        """
        return {"embeddings": self.embeddings_only()}

    def __repr__(self) -> str:
        return (
            f"EmbeddingResult(embedded={self.total_embedded}, "
            f"failed={self.total_failed}, "
            f"elapsed_ms={self.elapsed_ms:.1f}, "
            f"tps={self.texts_per_second:.1f})"
        )


# ── Singleton embedding model ─────────────────────────────────────────────────

class EmbeddingModel:
    """
    Thread-safe singleton wrapper around SentenceTransformer.

    Loading a SentenceTransformer model is expensive (~3–10 s on CPU,
    ~1 s on GPU). This class ensures the model is loaded exactly once
    per process, regardless of how many threads or async tasks call it.

    Usage:
        model = EmbeddingModel.get_instance()
        vector = model.embed_text("What is CBT?", is_query=True)
    """

    _instance: EmbeddingModel | None = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self, model_name: str, device: str, batch_size: int) -> None:
        # Private — use get_instance() or load_embedding_model()
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._model = None  # loaded lazily on first use
        self._model_lock = threading.Lock()  # guards _model initialisation
        self._loaded = False

    # ── Singleton factory ─────────────────────────────────────────────────────

    @classmethod
    def get_instance(cls, *, force_reload: bool = False) -> "EmbeddingModel":
        """
        Return the process-wide EmbeddingModel singleton.

        Reads configuration from ``get_settings().embedding``.
        The model is NOT loaded at construction time — use
        ``load_embedding_model()`` or let the first ``embed_*`` call
        trigger it.

        Args:
            force_reload: Destroy the existing singleton and create a new one.
                          Use in tests or when the model config changes at runtime.

        Returns:
            EmbeddingModel singleton.
        """
        if force_reload and cls._instance is not None:
            with cls._lock:
                cls._instance = None

        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cfg = get_settings().embedding
                    cls._instance = cls(
                        model_name=cfg.model_name,
                        device=cfg.device,
                        batch_size=cfg.batch_size,
                    )
                    _log.info(
                        "embedding.singleton_created",
                        model_name=cfg.model_name,
                        device=cfg.device,
                        batch_size=cfg.batch_size,
                    )

        return cls._instance

    # ── Model loading ─────────────────────────────────────────────────────────

    def load(self) -> None:
        """
        Load the SentenceTransformer model into memory.

        Thread-safe double-checked locking — only the first caller pays
        the load cost; all subsequent callers return immediately.

        Raises:
            ModelLoadError: If the model cannot be loaded (missing, network
                            error on first download, CUDA OOM, etc.)
        """
        if self._loaded:
            return

        with self._model_lock:
            if self._loaded:
                return  # another thread loaded it while we waited

            _log.info(
                "embedding.model_loading",
                model_name=self._model_name,
                device=self._device,
            )
            t_start = time.perf_counter()

            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(
                    self._model_name,
                    device=self._device,
                )
                # Trigger a warm-up forward pass so the first real call
                # doesn't show inflated latency in logs.
                self._model.encode(
                    ["warmup"],
                    batch_size=1,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
                self._loaded = True

            except ImportError as exc:
                raise ModelLoadError(
                    "sentence-transformers is not installed. "
                    "Run: pip install sentence-transformers",
                    cause=exc,
                ) from exc
            except Exception as exc:
                raise ModelLoadError(
                    f"Failed to load embedding model '{self._model_name}' "
                    f"on device '{self._device}': {exc}",
                    cause=exc,
                ) from exc

            elapsed = (time.perf_counter() - t_start) * 1000
            _log.info(
                "embedding.model_ready",
                model_name=self._model_name,
                device=self._device,
                load_time_ms=round(elapsed, 2),
            )

    def _ensure_loaded(self) -> None:
        """Load model if not already loaded. Called before every encode."""
        if not self._loaded:
            self.load()

    # ── Core embedding methods ────────────────────────────────────────────────

    def embed_text(self, text: str, *, is_query: bool = False) -> Vector:
        """
        Embed a single string and return its L2-normalised vector.

        Args:
            text:     The text to embed. Must be non-empty.
            is_query: If True, prepend the BGE query prefix
                      ``"Represent this sentence: "``.
                      Set True when embedding a user's search query.
                      Set False (default) when embedding document chunks.

        Returns:
            L2-normalised embedding vector as ``list[float]`` (dim=1024).

        Raises:
            EmptyInputError:        ``text`` is empty or whitespace-only.
            ModelLoadError:         Model failed to load.
            EmbeddingInferenceError: Encoding raised an unexpected error.
        """
        text = text.strip()
        if not text:
            raise EmptyInputError("embed_text() received an empty string.")

        self._ensure_loaded()
        self._warn_if_long(text, label="text")

        input_text = f"{_BGE_QUERY_PREFIX}{text}" if is_query else text

        _log.debug(
            "embedding.embed_text",
            is_query=is_query,
            char_count=len(text),
        )

        t_start = time.perf_counter()
        try:
            vector = self._model.encode(
                input_text,
                batch_size=1,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        except Exception as exc:
            raise EmbeddingInferenceError(
                f"Encoding failed for single text (is_query={is_query})", cause=exc
            ) from exc

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        _log.debug("embedding.embed_text_done", elapsed_ms=round(elapsed_ms, 2))

        return vector.tolist()

    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        sources: Sequence[str] | None = None,
        pages: Sequence[int] | None = None,
        batch_size: int | None = None,
        on_batch_complete: Callable[[int, int], None] | None = None,
    ) -> EmbeddingResult:
        """
        Embed a batch of document texts with full metadata and error handling.

        Designed for embedding chunks produced by ``Chunker.chunk_pages()``.
        Each batch is processed independently — a failure in one batch is
        logged and recorded but does not abort the remaining batches.

        Args:
            texts:             Sequence of document strings to embed.
                               Must be non-empty.
            sources:           Optional sequence of source filenames, aligned
                               with ``texts``. Defaults to "unknown" for each.
            pages:             Optional sequence of 1-based page numbers, aligned
                               with ``texts``. Defaults to 0 for each.
            batch_size:        Override the default batch size from config.
            on_batch_complete: Optional callback ``(batches_done, total_batches)``
                               called after each batch. Use to stream progress
                               to an API client.

        Returns:
            EmbeddingResult with an EmbeddedDocument per successfully embedded text.

        Raises:
            EmptyInputError: ``texts`` is empty.
            ModelLoadError:  Model failed to load.

        Note:
            Per-text errors do NOT raise — they are recorded in
            ``EmbeddingResult.errors`` with the text index.
        """
        if not texts:
            raise EmptyInputError("embed_documents() received an empty texts list.")

        self._ensure_loaded()

        n = len(texts)
        sources_resolved = list(sources) if sources else ["unknown"] * n
        pages_resolved   = list(pages)   if pages   else [0] * n
        effective_batch  = batch_size or self._batch_size

        if len(sources_resolved) != n or len(pages_resolved) != n:
            raise ValueError(
                f"texts ({n}), sources ({len(sources_resolved)}), and "
                f"pages ({len(pages_resolved)}) must all have the same length."
            )

        _log.info(
            "embedding.batch_start",
            total_texts=n,
            batch_size=effective_batch,
            model=self._model_name,
        )

        t_start = time.perf_counter()
        documents: list[EmbeddedDocument] = []
        errors: list[str] = []
        total_failed = 0

        # Split into batches
        batches = _make_batches(list(texts), effective_batch)
        total_batches = len(batches)

        for batch_idx, batch_slice in enumerate(batches):
            batch_start_idx = batch_idx * effective_batch
            batch_texts, batch_sources, batch_pages = [], [], []

            for local_i, text in enumerate(batch_slice):
                global_i = batch_start_idx + local_i
                cleaned = text.strip() if text else ""
                if not cleaned:
                    errors.append(f"index {global_i}: empty text skipped")
                    total_failed += 1
                    continue
                self._warn_if_long(cleaned, label=f"index {global_i}")
                batch_texts.append(cleaned)
                batch_sources.append(sources_resolved[global_i])
                batch_pages.append(pages_resolved[global_i])

            if not batch_texts:
                _log.debug(
                    "embedding.batch_all_empty",
                    batch_idx=batch_idx,
                )
                if on_batch_complete:
                    on_batch_complete(batch_idx + 1, total_batches)
                continue

            t_batch = time.perf_counter()
            try:
                vectors = self._model.encode(
                    batch_texts,
                    batch_size=effective_batch,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
            except Exception as exc:
                batch_err = (
                    f"batch {batch_idx} (indices {batch_start_idx}–"
                    f"{batch_start_idx + len(batch_texts) - 1}): "
                    f"{type(exc).__name__}: {exc}"
                )
                errors.append(batch_err)
                total_failed += len(batch_texts)
                _log.error(
                    "embedding.batch_failed",
                    batch_idx=batch_idx,
                    error=str(exc),
                )
                if on_batch_complete:
                    on_batch_complete(batch_idx + 1, total_batches)
                continue

            batch_ms = (time.perf_counter() - t_batch) * 1000
            _log.debug(
                "embedding.batch_done",
                batch_idx=batch_idx,
                batch_size=len(batch_texts),
                elapsed_ms=round(batch_ms, 2),
            )

            for vec, src, pg, txt in zip(vectors, batch_sources, batch_pages, batch_texts):
                documents.append(
                    EmbeddedDocument(
                        text=txt,
                        embedding=vec.tolist(),
                        source=src,
                        page=pg,
                    )
                )

            if on_batch_complete:
                on_batch_complete(batch_idx + 1, total_batches)

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        total_embedded = len(documents)
        tps = (total_embedded / (elapsed_ms / 1000)) if elapsed_ms > 0 else 0.0

        result = EmbeddingResult(
            documents=documents,
            total_embedded=total_embedded,
            total_failed=total_failed,
            model_name=self._model_name,
            elapsed_ms=round(elapsed_ms, 2),
            texts_per_second=round(tps, 2),
            errors=errors,
        )

        _log.info(
            "embedding.batch_complete",
            total_texts=n,
            embedded=total_embedded,
            failed=total_failed,
            elapsed_ms=round(elapsed_ms, 2),
            texts_per_second=round(tps, 2),
        )

        return result

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        """True if the underlying SentenceTransformer model is in memory."""
        return self._loaded

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def device(self) -> str:
        return self._device

    @property
    def embedding_dim(self) -> int:
        """
        Return the embedding dimension.
        Requires model to be loaded; returns the BGE constant otherwise.
        """
        if self._loaded and self._model is not None:
            try:
                return self._model.get_sentence_embedding_dimension()
            except Exception:
                pass
        return _BGE_EMBEDDING_DIM

    def health(self) -> dict:
        """
        Return a health-check dict for the ``/health`` API endpoint.

        Returns:
            {
                "status": "ok" | "not_loaded",
                "model_name": "...",
                "device": "...",
                "embedding_dim": 1024,
                "is_loaded": bool
            }
        """
        return {
            "status": "ok" if self._loaded else "not_loaded",
            "model_name": self._model_name,
            "device": self._device,
            "embedding_dim": self.embedding_dim,
            "is_loaded": self._loaded,
        }

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not_loaded"
        return f"EmbeddingModel(model={self._model_name!r}, device={self._device!r}, {status})"

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _warn_if_long(text: str, label: str = "text") -> None:
        if len(text) > _TRUNCATION_WARN_CHARS:
            _log.warning(
                "embedding.text_may_truncate",
                label=label,
                char_count=len(text),
                threshold=_TRUNCATION_WARN_CHARS,
                hint=(
                    f"bge-large-en-v1.5 has a 512-token window. "
                    f"Text at {label} is {len(text)} chars and may be silently truncated. "
                    "Consider reducing chunk_size."
                ),
            )


# ── Private helpers ────────────────────────────────────────────────────────────

def _make_batches(items: list, batch_size: int) -> list[list]:
    """Partition a list into consecutive sub-lists of at most batch_size."""
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


# ── Module-level convenience functions (public API) ────────────────────────────

def load_embedding_model(*, force_reload: bool = False) -> EmbeddingModel:
    """
    Warm up the embedding model singleton.

    Call once at application startup (e.g. in FastAPI's ``lifespan``)
    to pay the load cost before the first request arrives.

    Args:
        force_reload: Discard the current singleton and reload.
                      Useful after changing ``EMBEDDING_MODEL`` in .env
                      without restarting the process.

    Returns:
        The loaded EmbeddingModel singleton.

    Raises:
        ModelLoadError: If the model cannot be loaded.

    Example:
        # In FastAPI lifespan:
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            load_embedding_model()
            yield
    """
    model = EmbeddingModel.get_instance(force_reload=force_reload)
    model.load()
    return model


def embed_text(text: str, *, is_query: bool = False) -> Vector:
    """
    Embed a single string and return its normalised vector.

    This is the recommended entry point for embedding a user's search
    query before passing it to the ChromaDB retriever.

    Args:
        text:     Non-empty string to embed.
        is_query: True → prepend BGE query prefix (use for user queries).
                  False → embed as-is (use for document chunks).

    Returns:
        L2-normalised ``list[float]`` of length 1024.

    Raises:
        EmptyInputError:         ``text`` is empty.
        ModelLoadError:          Model not installed or failed to load.
        EmbeddingInferenceError: Encoding error.

    Example:
        query_vec = embed_text("What are the DSM-5 criteria for MDD?", is_query=True)
        chunk_vec = embed_text("Criterion A: depressed mood...", is_query=False)
    """
    return EmbeddingModel.get_instance().embed_text(text, is_query=is_query)


def embed_documents(
    texts: Sequence[str],
    *,
    sources: Sequence[str] | None = None,
    pages: Sequence[int] | None = None,
    batch_size: int | None = None,
    on_batch_complete: Callable[[int, int], None] | None = None,
) -> EmbeddingResult:
    """
    Embed a batch of document chunks and return structured results.

    This is the recommended entry point for the ingestion pipeline.
    Pairs directly with ``Chunker.chunk_pages()`` output.

    Args:
        texts:             Texts to embed (document chunks, not queries).
        sources:           Aligned source filenames. Optional.
        pages:             Aligned 1-based page numbers. Optional.
        batch_size:        Override batch size from config.
        on_batch_complete: Progress callback ``(batches_done, total_batches)``.

    Returns:
        ``EmbeddingResult`` with one ``EmbeddedDocument`` per successful text.

    Example:
        from ingestion.processors import Chunker
        from rag.embeddings import embed_documents

        result    = Chunker().chunk_pages(pages)
        texts     = [c.text   for c in result.chunks]
        sources   = [c.source for c in result.chunks]
        pages_    = [c.page   for c in result.chunks]

        embedded  = embed_documents(texts, sources=sources, pages=pages_)
        print(embedded)
        # EmbeddingResult(embedded=42, failed=0, elapsed_ms=1840.2, tps=22.8)
    """
    return EmbeddingModel.get_instance().embed_documents(
        texts,
        sources=sources,
        pages=pages,
        batch_size=batch_size,
        on_batch_complete=on_batch_complete,
    )
