"""
rag/retriever.py
─────────────────
Clinical document retriever — the query-time entry point for the RAG pipeline.

Architecture
────────────
                    +--------------------------------------------------+
                    |                  Retriever                       |
                    |                                                  |
                    |  retrieve(query)                                 |
                    |    |                                             |
                    |    +- 1. validate & preprocess query             |
                    |    +- 2. embed_text(query, is_query=True)        |
                    |    |       -> L2-normalised 1024-dim vector      |
                    |    +- 3. vector_store.query_documents(vector)    |
                    |    |       -> list[QueryResult]  (cosine search) |
                    |    +- 4. apply similarity_threshold filter       |
                    |    +- 5. re-rank by score (already sorted)       |
                    |    +- 6. return list[RetrievedChunk]             |
                    |                                                  |
                    |  retrieve_with_sources(query)                    |
                    |    -> groups results by source file              |
                    |                                                  |
                    |  aretrieve(query)  <- async version              |
                    +--------------------------------------------------+

Integration in the LangGraph RAG workflow
─────────────────────────────────────────
    retriever = Retriever()

    # In retrieve_node:
    chunks = retriever.retrieve("What are DSM-5 criteria for MDD?")
    context = "\n\n".join(c.text for c in chunks)

    # In diagnostic_agent with source filter:
    chunks = retriever.retrieve(
        "CBT techniques for GAD",
        source_filter="CBT_manual.pdf",
    )

Public API
──────────
    # Module-level singleton (preferred)
    from rag.retriever import get_retriever
    retriever = get_retriever()
    chunks    = retriever.retrieve("What is CBT?")

    # Class-level (for DI / testing)
    from rag.retriever import Retriever
    retriever = Retriever(n_results=5, similarity_threshold=0.4)
"""

from __future__ import annotations

import asyncio
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from app_logging.logger import get_logger

_log = get_logger(__name__)

# Conditional imports for hybrid retrieval — gracefully degrade if missing
try:
    from rag.retrieval.fusion import reciprocal_rank_fusion
except ImportError:
    reciprocal_rank_fusion = None  # type: ignore[assignment]
    _log.warning("retriever.fusion_unavailable")


# ---- Custom exceptions ----------------------------------------------------------

class RetrieverError(RuntimeError):
    """Base class for retriever failures."""

    def __init__(self, message: str, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.cause = cause

    def __str__(self) -> str:
        s = super().__str__()
        return f"{s} [caused by: {type(self.cause).__name__}: {self.cause}]" if self.cause else s


class EmptyQueryError(RetrieverError):
    """Raised when the query string is empty or whitespace-only."""


class EmbeddingFailedError(RetrieverError):
    """Raised when query embedding fails."""


class SearchFailedError(RetrieverError):
    """Raised when the ChromaDB similarity search fails."""


class NoResultsError(RetrieverError):
    """
    Raised (optionally) when no chunks pass the similarity threshold.
    Only raised when raise_on_empty=True -- by default an empty list is returned.
    """


# ---- Output model ---------------------------------------------------------------

@dataclass(frozen=True)
class RetrievedChunk:
    """
    A single retrieved chunk, ready for injection into the LLM prompt.

    This is the canonical output type of the retriever. Downstream consumers
    (LangGraph nodes, agents, API endpoints) should depend only on this type,
    not on QueryResult from the vector store layer.

    Attributes:
        text:       The chunk's plain text content.
        source:     Source PDF filename (e.g. ``"DSM5.pdf"``).
        page:       1-based page number in the source document.
        score:      Cosine similarity in [0, 1] -- higher is more relevant.
        chunk_id:   Unique ID from the vector store (``"DSM5__p0012__c0003"``).
        rank:       1-based retrieval rank (1 = most similar).
        metadata:   Full metadata dict from ChromaDB for audit/debug use.
    """
    text: str
    source: str
    page: int
    score: float
    chunk_id: str
    rank: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Canonical output format matching the project specification.

        Returns:
            {"text": "...", "source": "...", "page": 15, "score": 0.91}
        """
        return {
            "text": self.text,
            "source": self.source,
            "page": self.page,
            "score": self.score,
        }

    def to_prompt_citation(self) -> str:
        """
        Compact citation string for injection into the LLM prompt.

        Example:
            "[DSM5.pdf, p.12, score=0.91] Criterion A: depressed mood..."
        """
        preview = self.text[:120].replace("\n", " ")
        ellipsis = "\u2026" if len(self.text) > 120 else ""
        return f"[{self.source}, p.{self.page}, score={self.score:.2f}] {preview}{ellipsis}"

    def __repr__(self) -> str:
        return (
            f"RetrievedChunk(rank={self.rank}, score={self.score:.4f}, "
            f"source={self.source!r}, page={self.page}, "
            f"chars={len(self.text)})"
        )


@dataclass
class RetrievalResult:
    """
    Complete result of a single retrieval call.

    Attributes:
        query:           The original (preprocessed) query string.
        chunks:          Retrieved chunks sorted by descending score.
        total_candidates: How many results ChromaDB returned before threshold filtering.
        threshold_used:  The similarity_threshold applied.
        n_results_requested: The n_results value sent to ChromaDB.
        elapsed_ms:      Wall-clock time for the full retrieve() call.
        embedding_ms:    Time spent embedding the query.
        search_ms:       Time spent in ChromaDB similarity search.
    """
    query: str
    chunks: list[RetrievedChunk]
    total_candidates: int
    threshold_used: float
    n_results_requested: int
    elapsed_ms: float
    embedding_ms: float
    search_ms: float

    @property
    def found(self) -> bool:
        """True if at least one chunk passed the similarity threshold."""
        return len(self.chunks) > 0

    @property
    def top(self) -> RetrievedChunk | None:
        """The single most relevant chunk, or None if no results."""
        return self.chunks[0] if self.chunks else None

    def to_dicts(self) -> list[dict[str, Any]]:
        """Return chunks as list of canonical spec dicts."""
        return [c.to_dict() for c in self.chunks]

    def to_context_string(self, separator: str = "\n\n---\n\n") -> str:
        """
        Concatenate chunk texts for direct injection into an LLM prompt.

        Args:
            separator: String placed between chunks. Default is a markdown
                       horizontal rule to help the LLM distinguish boundaries.

        Returns:
            Single string of all chunk texts joined by separator.
        """
        return separator.join(c.text for c in self.chunks)

    def to_cited_context(self) -> str:
        """
        Context string where each chunk is prefixed with its citation.
        Useful for prompts that require inline source attribution.

        Example output:
            [DSM5.pdf, p.12, score=0.91] Criterion A: depressed mood...

            ---

            [CBT_manual.pdf, p.45, score=0.87] Cognitive restructuring...
        """
        return "\n\n---\n\n".join(c.to_prompt_citation() for c in self.chunks)

    def __repr__(self) -> str:
        return (
            f"RetrievalResult(found={self.found}, chunks={len(self.chunks)}, "
            f"top_score={self.top.score if self.top else None}, "
            f"elapsed_ms={self.elapsed_ms:.1f})"
        )


# ---- Core retriever class -------------------------------------------------------

class Retriever:
    """
    Query-time retriever: embed query -> similarity search -> filtered results.

    The retriever is the single entry point for all RAG lookups. It is
    stateless beyond its configuration and can be shared safely across
    threads and async tasks.

    Args:
        n_results:           Number of chunks to request from ChromaDB.
                             Default: reads from ``settings.rag.top_k`` (10).
        similarity_threshold: Minimum similarity score for a chunk to be
                             included in the result. Chunks below this score
                             are silently dropped. Range [0, 1].
                             Default: reads from ``settings.rag.similarity_threshold`` (0.35).
        raise_on_empty:      If True, raise NoResultsError when no chunks
                             pass the threshold. If False (default), return
                             an empty RetrievalResult.
        max_query_length:    Truncate queries longer than this many characters
                             before embedding. Protects against accidental
                             full-document queries. Default: 1000.
    """

    def __init__(
        self,
        *,
        n_results: int | None = None,
        similarity_threshold: float | None = None,
        raise_on_empty: bool = False,
        max_query_length: int = 1000,
    ) -> None:
        from config.settings import get_settings
        cfg = get_settings().rag

        self._n_results           = n_results            if n_results is not None else cfg.top_k
        self._similarity_threshold = similarity_threshold if similarity_threshold is not None else cfg.similarity_threshold
        self._raise_on_empty      = raise_on_empty
        self._max_query_length    = max_query_length
        self._bm25_weight         = cfg.bm25_weight
        self._bm25_enabled        = cfg.bm25_enabled
        self._rerank_enabled      = cfg.rerank_enabled
        self._rerank_top_k        = cfg.rerank_top_k
        self._hybrid_top_k_ratio  = cfg.hybrid_top_k_ratio

        if not 0.0 <= self._similarity_threshold <= 1.0:
            raise ValueError(
                f"similarity_threshold must be in [0, 1], got {self._similarity_threshold}."
            )
        if self._n_results < 1:
            raise ValueError(f"n_results must be >= 1, got {self._n_results}.")

        _log.info(
            "retriever.init",
            n_results=self._n_results,
            similarity_threshold=self._similarity_threshold,
            raise_on_empty=raise_on_empty,
        )

    # ---- Primary synchronous interface ------------------------------------------

    def retrieve(
        self,
        query: str,
        *,
        n_results: int | None = None,
        source_filter: str | None = None,
        page_range: tuple[int, int] | None = None,
        where: dict[str, Any] | None = None,
        similarity_threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
        use_hybrid: bool | None = None,
        use_rerank: bool | None = None,
        bm25_weight: float | None = None,
    ) -> RetrievalResult:
        """
        Retrieve the most relevant clinical document chunks for a query.

        Flow (default, dense-only):
            1. Validate and preprocess query text.
            2. Embed query with BGE query prefix (``is_query=True``).
            3. Run cosine similarity search in ChromaDB.
            4. Filter by similarity threshold.
            5. Return typed RetrievalResult.

        When ``use_hybrid=True`` (dense + BM25 + fusion):
            1-2. Same as above.
            3.   Run dense vector search AND BM25 keyword search.
            4.   Fuse results with Reciprocal Rank Fusion.
            5.   Optionally rerank with cross-encoder if ``use_rerank=True``.
            6.   Return typed RetrievalResult.

        Args:
            query:               Natural language query from the clinician.
            n_results:           Override instance-level n_results for this call.
            source_filter:       Restrict search to a single source file.
                                 Example: ``"DSM5.pdf"``
            page_range:          Restrict search to pages in (start, end) inclusive.
                                 Example: ``(50, 100)``
            where:               Raw ChromaDB metadata filter. Overrides
                                 source_filter, page_range, and metadata_filter
                                 if provided.
            similarity_threshold: Override instance-level threshold for this call.
            metadata_filter:     Structured metadata filter dict. Each key is a
                                 metadata field name, value is the exact value
                                 to match. Supports ``$in`` for list matching.
                                 Example: ``{"topic": "depression", "therapy": {"$in": ["CBT", "ACT"]}}``
            use_hybrid:          Enable hybrid dense + BM25 retrieval. Defaults
                                 to instance-level ``bm25_enabled`` setting.
            use_rerank:          Enable cross-encoder reranking after retrieval.
                                 Defaults to instance-level ``rerank_enabled``.
            bm25_weight:         Weight for BM25 scores in hybrid fusion
                                 (0.0 = dense only, 1.0 = BM25 only).
                                 Defaults to instance-level ``bm25_weight``.

        Returns:
            RetrievalResult containing matched chunks and timing metadata.

        Raises:
            EmptyQueryError:    Query is empty or whitespace.
            EmbeddingFailedError: Query embedding failed.
            SearchFailedError:  ChromaDB search raised an error.
            NoResultsError:     No chunks found and raise_on_empty=True.
        """
        t_total = time.perf_counter()

        # ---- 1. Validate & preprocess -------------------------------------------
        clean_query = self._preprocess_query(query)
        effective_n = n_results or self._n_results
        effective_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self._similarity_threshold
        )
        effective_hybrid = (
            use_hybrid if use_hybrid is not None else self._bm25_enabled
        )
        effective_rerank = (
            use_rerank if use_rerank is not None else self._rerank_enabled
        )
        effective_bm25_w = (
            bm25_weight if bm25_weight is not None else self._bm25_weight
        )

        # ---- Route to hybrid or dense-only pipeline ----------------------------
        if effective_hybrid or effective_rerank:
            return self._retrieve_hybrid(
                query=clean_query,
                n_results=effective_n,
                threshold=effective_threshold,
                source_filter=source_filter,
                page_range=page_range,
                where=where,
                metadata_filter=metadata_filter,
                use_rerank=effective_rerank,
                bm25_weight=effective_bm25_w,
                t_total=t_total,
            )

        _log.info(
            "retriever.retrieve_start",
            query_length=len(clean_query),
            n_results=effective_n,
            threshold=effective_threshold,
            source_filter=source_filter,
            page_range=page_range,
        )

        # ---- 2. Build metadata filter -------------------------------------------
        chroma_filter = _build_where_filter(
            where=where,
            source_filter=source_filter,
            page_range=page_range,
            metadata_filter=metadata_filter,
        )

        # ---- 3. Embed query -----------------------------------------------------
        t_embed = time.perf_counter()
        query_vector = self._embed_query(clean_query)
        embedding_ms = (time.perf_counter() - t_embed) * 1000

        # ---- 4. Similarity search -----------------------------------------------
        t_search = time.perf_counter()
        raw_results = self._search(query_vector, effective_n, chroma_filter)
        search_ms = (time.perf_counter() - t_search) * 1000

        total_candidates = len(raw_results)

        # ---- 5. Apply threshold & convert ---------------------------------------
        chunks = _to_retrieved_chunks(raw_results, effective_threshold)

        # ---- 6. Handle empty results --------------------------------------------
        if not chunks and self._raise_on_empty:
            raise NoResultsError(
                f"No chunks found for query (threshold={effective_threshold:.2f}, "
                f"candidates={total_candidates}). "
                "Consider lowering RAG_SIMILARITY_THRESHOLD in .env."
            )

        elapsed_ms = (time.perf_counter() - t_total) * 1000

        result = RetrievalResult(
            query=clean_query,
            chunks=chunks,
            total_candidates=total_candidates,
            threshold_used=effective_threshold,
            n_results_requested=effective_n,
            elapsed_ms=round(elapsed_ms, 2),
            embedding_ms=round(embedding_ms, 2),
            search_ms=round(search_ms, 2),
        )

        _log.info(
            "retriever.retrieve_complete",
            query_length=len(clean_query),
            candidates=total_candidates,
            returned=len(chunks),
            top_score=chunks[0].score if chunks else None,
            elapsed_ms=round(elapsed_ms, 2),
            embedding_ms=round(embedding_ms, 2),
            search_ms=round(search_ms, 2),
        )

        return result

    def retrieve_chunks(
        self,
        query: str,
        **kwargs: Any,
    ) -> list[RetrievedChunk]:
        """
        Convenience wrapper -- returns just the chunk list.

        Equivalent to ``retrieve(query, **kwargs).chunks``.
        Use when you only need the chunks and don't care about timing metadata.
        """
        return self.retrieve(query, **kwargs).chunks

    def retrieve_with_sources(
        self,
        query: str,
        **kwargs: Any,
    ) -> dict[str, list[RetrievedChunk]]:
        """
        Retrieve and group results by source filename.

        Useful for building a "sources consulted" section in the clinical
        response, or for deduplicating across documents.

        Returns:
            Dict mapping source filename -> list of RetrievedChunk from that source,
            ordered by descending score within each source.

        Example:
            {
                "DSM5.pdf": [RetrievedChunk(...), RetrievedChunk(...)],
                "CBT_manual.pdf": [RetrievedChunk(...)],
            }
        """
        result = self.retrieve(query, **kwargs)
        grouped: dict[str, list[RetrievedChunk]] = {}
        for chunk in result.chunks:
            grouped.setdefault(chunk.source, []).append(chunk)
        return grouped

    # ---- Async interface --------------------------------------------------------

    async def aretrieve(
        self,
        query: str,
        **kwargs: Any,
    ) -> RetrievalResult:
        """
        Async wrapper around retrieve() for FastAPI endpoints and LangGraph nodes.

        Both embedding and ChromaDB search are CPU/IO bound operations.
        We run them in a thread-pool executor to avoid blocking the event loop.

        Usage in a LangGraph node:
            async def retrieve_node(state: GraphState) -> GraphState:
                retriever = get_retriever()
                result = await retriever.aretrieve(state["query"])
                return {**state, "context": result.to_context_string()}
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.retrieve(query, **kwargs),
        )

    async def aretrieve_chunks(self, query: str, **kwargs: Any) -> list[RetrievedChunk]:
        """Async convenience wrapper -- returns just the chunk list."""
        result = await self.aretrieve(query, **kwargs)
        return result.chunks

    # ---- Private helpers --------------------------------------------------------

    def _preprocess_query(self, query: str) -> str:
        """
        Validate and normalise the query string.

        Steps:
          1. Strip outer whitespace.
          2. Raise EmptyQueryError if empty after stripping.
          3. Collapse internal whitespace runs (double spaces, stray newlines).
          4. Truncate to max_query_length with a warning if too long.

        We intentionally do NOT:
          - Lower-case (bge-large handles casing at encode time)
          - Remove punctuation (question marks affect embedding direction)
          - Stem or lemmatise (embedding model handles this)
        """
        cleaned = query.strip()
        if not cleaned:
            raise EmptyQueryError(
                "retrieve() received an empty query string. "
                "Provide a non-empty clinical question."
            )

        # Collapse internal whitespace
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{2,}", "\n", cleaned)

        if len(cleaned) > self._max_query_length:
            _log.warning(
                "retriever.query_truncated",
                original_length=len(cleaned),
                max_length=self._max_query_length,
            )
            cleaned = cleaned[: self._max_query_length]

        return cleaned

    def _embed_query(self, query: str) -> list[float]:
        """
        Embed the query using the BGE query prefix.

        Args:
            query: Preprocessed query string.

        Returns:
            L2-normalised 1024-dim embedding vector.

        Raises:
            EmbeddingFailedError: Wraps any exception from the embedding layer.
        """
        try:
            from rag.embeddings import embed_text
            return embed_text(query, is_query=True)
        except Exception as exc:
            raise EmbeddingFailedError(
                f"Failed to embed query (length={len(query)}): {exc}",
                cause=exc,
            ) from exc

    def _search(
        self,
        query_vector: list[float],
        n_results: int,
        where: dict[str, Any] | None,
    ) -> list[Any]:
        """
        Run similarity search in ChromaDB.

        Args:
            query_vector: L2-normalised query embedding.
            n_results:    Number of results to request.
            where:        Optional metadata filter.

        Returns:
            List of QueryResult from the vector store.

        Raises:
            SearchFailedError: Wraps any exception from the vector store layer.
        """
        try:
            from rag.vector_store import get_vector_store
            store = get_vector_store()
            return store.query_documents(
                query_vector,
                n_results=n_results,
                where=where,
            )
        except Exception as exc:
            raise SearchFailedError(
                f"ChromaDB similarity search failed: {exc}",
                cause=exc,
            ) from exc

    # ---- Hybrid retrieval pipeline ---------------------------------------------

    def _retrieve_hybrid(
        self,
        *,
        query: str,
        n_results: int,
        threshold: float,
        source_filter: str | None,
        page_range: tuple[int, int] | None,
        where: dict[str, Any] | None,
        metadata_filter: dict[str, Any] | None,
        use_rerank: bool,
        bm25_weight: float,
        t_total: float,
    ) -> RetrievalResult:
        """Run dense + BM25 hybrid retrieval with optional cross-encoder rerank."""

        chroma_filter = _build_where_filter(
            where=where,
            source_filter=source_filter,
            page_range=page_range,
            metadata_filter=metadata_filter,
        )
        bm25_filter = _build_bm25_filter(metadata_filter=metadata_filter, source_filter=source_filter)

        # ---- Dense retrieval ---------------------------------------------------
        t_embed = time.perf_counter()
        query_vector = self._embed_query(query)
        embedding_ms = (time.perf_counter() - t_embed) * 1000

        candidate_n = int(n_results * self._hybrid_top_k_ratio)
        t_search = time.perf_counter()
        dense_raw = self._search(query_vector, candidate_n, chroma_filter)
        dense_ms = (time.perf_counter() - t_search) * 1000

        dense_candidates = _to_retrieved_chunks(dense_raw, threshold)
        _log.info(
            "retriever.dense_complete",
            candidates=len(dense_candidates),
            top_score=dense_candidates[0].score if dense_candidates else None,
            elapsed_ms=round(dense_ms, 2),
        )

        # ---- BM25 retrieval ----------------------------------------------------
        t_sparse = time.perf_counter()
        sparse_candidates: list[dict[str, Any]] = []
        try:
            from rag.retrieval.bm25 import get_bm25_retriever
            bm25 = get_bm25_retriever()
            sparse_raw = bm25.search(
                query,
                n_results=candidate_n,
                metadata_filter=bm25_filter,
            )
            sparse_candidates = sparse_raw
        except Exception as exc:
            _log.warning("retriever.bm25_failed", error=str(exc))
        sparse_ms = (time.perf_counter() - t_sparse) * 1000
        _log.info(
            "retriever.bm25_complete",
            candidates=len(sparse_candidates),
            elapsed_ms=round(sparse_ms, 2),
        )

        # ---- Fusion (RRF) ------------------------------------------------------
        dense_weight = 1.0 - bm25_weight
        if sparse_candidates and reciprocal_rank_fusion is not None:
            fused = reciprocal_rank_fusion(
                dense_results=_chunks_to_dicts(dense_candidates),
                sparse_results=sparse_candidates,
                top_k=n_results,
                dense_weight=dense_weight,
                sparse_weight=bm25_weight,
            )
        elif sparse_candidates:
            fused = sparse_candidates[:n_results]
        else:
            fused = _chunks_to_dicts(dense_candidates)[:n_results]

        total_candidates = len(dense_raw) + len(sparse_candidates)

        # ---- Cross-encoder rerank ----------------------------------------------
        if use_rerank and fused:
            try:
                from rag.retrieval.reranker import get_reranker
                reranker = get_reranker()
                fused = reranker.rerank(query, fused, top_k=n_results)
                _log.info("retriever.rerank_complete", candidates=len(fused))
            except Exception as exc:
                _log.warning("retriever.rerank_failed", error=str(exc))

        # ---- Convert to RetrievedChunk -----------------------------------------
        chunks = _dicts_to_chunks(fused)

        # ---- Handle empty results ----------------------------------------------
        if not chunks and self._raise_on_empty:
            raise NoResultsError(
                f"No chunks found for query (threshold={threshold:.2f}, "
                f"candidates={total_candidates}). "
                "Consider lowering RAG_SIMILARITY_THRESHOLD in .env."
            )

        elapsed_ms = (time.perf_counter() - t_total) * 1000
        total_ms = embedding_ms + dense_ms + sparse_ms

        result = RetrievalResult(
            query=query,
            chunks=chunks,
            total_candidates=total_candidates,
            threshold_used=threshold,
            n_results_requested=n_results,
            elapsed_ms=round(elapsed_ms, 2),
            embedding_ms=round(embedding_ms, 2),
            search_ms=round(dense_ms, 2),
        )

        _log.info(
            "retriever.hybrid_complete",
            query_length=len(query),
            total_candidates=total_candidates,
            dense=len(dense_candidates),
            bm25=len(sparse_candidates),
            returned=len(chunks),
            top_score=chunks[0].score if chunks else None,
            elapsed_ms=round(elapsed_ms, 2),
            embedding_ms=round(embedding_ms, 2),
            dense_ms=round(dense_ms, 2),
            sparse_ms=round(sparse_ms, 2),
        )

        return result

    # ---- Configuration introspection --------------------------------------------

    @property
    def n_results(self) -> int:
        return self._n_results

    @property
    def similarity_threshold(self) -> float:
        return self._similarity_threshold

    @property
    def config(self) -> dict[str, Any]:
        return {
            "n_results": self._n_results,
            "similarity_threshold": self._similarity_threshold,
            "raise_on_empty": self._raise_on_empty,
            "max_query_length": self._max_query_length,
        }

    def __repr__(self) -> str:
        return (
            f"Retriever(n_results={self._n_results}, "
            f"threshold={self._similarity_threshold})"
        )


# ---- Private module helpers -----------------------------------------------------

def _build_where_filter(
    *,
    where: dict[str, Any] | None,
    source_filter: str | None,
    page_range: tuple[int, int] | None,
    metadata_filter: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """
    Build a ChromaDB ``where`` filter from convenience parameters.

    Priority: explicit ``where`` dict > source_filter + page_range + metadata_filter.
    If ``where`` is provided, it is used as-is (caller owns the ChromaDB
    filter syntax). Otherwise, source_filter, page_range, and metadata_filter
    are combined with a ``$and`` operator when multiple are present.

    The ``metadata_filter`` dict supports:
        - Exact match: ``{"topic": "depression"}`` → ``{"topic": {"$eq": "depression"}}``
        - ``$in``: ``{"therapy": {"$in": ["CBT", "ACT"]}}``
        - ``$eq``: ``{"disorder": {"$eq": "MDD"}}``

    Args:
        where:           Raw ChromaDB filter dict. Overrides all others.
        source_filter:   Filename to restrict search to.
        page_range:      (start, end) tuple for inclusive page filtering.
        metadata_filter: Structured metadata filter (see above).

    Returns:
        ChromaDB-compatible filter dict, or None if no filtering needed.
    """
    if where is not None:
        return where

    conditions: list[dict[str, Any]] = []

    if source_filter:
        conditions.append({"source": source_filter})

    if page_range:
        start, end = page_range
        if start > end:
            raise ValueError(
                f"page_range start ({start}) must be <= end ({end})."
            )
        conditions.append({"page": {"$gte": start}})
        conditions.append({"page": {"$lte": end}})

    if metadata_filter:
        for key, condition in metadata_filter.items():
            if isinstance(condition, dict) and "$in" in condition:
                conditions.append({key: {"$in": condition["$in"]}})
            elif isinstance(condition, dict) and "$eq" in condition:
                conditions.append({key: {"$eq": condition["$eq"]}})
            else:
                conditions.append({key: {"$eq": condition}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _build_bm25_filter(
    metadata_filter: dict[str, Any] | None = None,
    source_filter: str | None = None,
) -> dict[str, Any] | None:
    """Build a metadata filter dict for BM25 post-filtering.

    BM25 doesn't support ChromaDB's ``$and``/``$eq`` syntax — it does its
    own matching.  This function converts the shared parameters into a
    simple key-value dict that ``BM25Retriever._match_metadata`` understands.
    """
    filt: dict[str, Any] = {}
    if metadata_filter:
        for key, condition in metadata_filter.items():
            if isinstance(condition, dict) and "$in" in condition:
                filt[key] = {"$in": condition["$in"]}
            elif isinstance(condition, dict) and "$eq" in condition:
                filt[key] = condition["$eq"]
            else:
                filt[key] = condition
    if source_filter:
        filt["source"] = source_filter
    return filt or None


def _chunks_to_dicts(chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
    """Convert RetrievedChunk objects to dicts for hybrid fusion."""
    return [
        {
            "chunk_id": c.chunk_id,
            "text": c.text,
            "source": c.source,
            "page": c.page,
            "score": c.score,
            "rank": c.rank,
            "metadata": c.metadata,
        }
        for c in chunks
    ]


def _dicts_to_chunks(dicts: list[dict[str, Any]]) -> list[RetrievedChunk]:
    """Convert dicts (from fusion/rerank) back to RetrievedChunk."""
    chunks: list[RetrievedChunk] = []
    for rank, d in enumerate(dicts, start=1):
        chunks.append(RetrievedChunk(
            text=str(d.get("text", "")),
            source=str(d.get("source", "unknown")),
            page=int(d.get("page", 0)),
            score=float(d.get("score", 0.0)),
            chunk_id=str(d.get("chunk_id", "")),
            rank=rank,
            metadata=dict(d.get("metadata", {})),
        ))
    return chunks


def _to_retrieved_chunks(
    raw_results: list[Any],
    threshold: float,
) -> list[RetrievedChunk]:
    """
    Convert QueryResult objects from the vector store into RetrievedChunk
    objects, applying the similarity threshold filter and re-assigning ranks.

    Args:
        raw_results: List of QueryResult from VectorStore.query_documents().
        threshold:   Minimum score; results below this are dropped.

    Returns:
        List of RetrievedChunk sorted by descending score, re-ranked from 1.
    """
    kept: list[RetrievedChunk] = []

    for raw in raw_results:
        if raw.score < threshold:
            _log.debug(
                "retriever.chunk_below_threshold",
                chunk_id=raw.chunk_id,
                score=raw.score,
                threshold=threshold,
            )
            continue

        kept.append(RetrievedChunk(
            text=raw.text,
            source=raw.source,
            page=raw.page,
            score=raw.score,
            chunk_id=raw.chunk_id,
            rank=raw.rank,          # re-assigned below after filtering
            metadata=raw.metadata,
        ))

    # Re-assign ranks after threshold filtering
    # (original ranks may have gaps if middle results were filtered out)
    return [
        RetrievedChunk(
            text=c.text,
            source=c.source,
            page=c.page,
            score=c.score,
            chunk_id=c.chunk_id,
            rank=new_rank,
            metadata=c.metadata,
        )
        for new_rank, c in enumerate(kept, start=1)
    ]


# ---- Singleton factory ----------------------------------------------------------

_retriever_lock = threading.Lock()
_retriever_instance: Retriever | None = None


def get_retriever(*, force_reload: bool = False) -> Retriever:
    """
    Return the process-wide Retriever singleton.

    Reads n_results and similarity_threshold from ``settings.rag``.
    Call once at startup (e.g. in FastAPI lifespan) so the first
    request doesn't pay the construction cost.

    Args:
        force_reload: Discard the existing singleton and rebuild.
                      Useful after updating RAG_* env vars in tests.

    Returns:
        Retriever singleton.

    Example:
        from rag.retriever import get_retriever

        # At startup:
        retriever = get_retriever()

        # In a LangGraph node:
        result = retriever.retrieve("What is the DSM-5 criterion for MDD?")
        context = result.to_context_string()
    """
    global _retriever_instance

    if force_reload:
        with _retriever_lock:
            _retriever_instance = None

    if _retriever_instance is not None:
        return _retriever_instance

    with _retriever_lock:
        if _retriever_instance is not None:
            return _retriever_instance

        _retriever_instance = Retriever()
        _log.info("retriever.singleton_created")

    return _retriever_instance
