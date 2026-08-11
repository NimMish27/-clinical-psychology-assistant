"""
clinical/retrieval/humanistic_retriever.py — Focused Humanistic literature retriever.

Provides hybrid retrieval (dense + BM25 + RRF fusion) with cross-encoder
reranking, always filtered to the humanistic framework. Returns structured
``HumanisticEvidenceResult`` objects suitable for framework lifecycle methods.

Follows the same pattern as ``CBTRetriever`` and ``ACTRetriever``.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any

from app_logging.logger import get_logger
from clinical.retrieval.collections import resolve_framework_store
from clinical.retrieval.models import (
    HumanisticEvidenceFilter,
    HumanisticEvidenceItem,
    HumanisticEvidenceResult,
)

if TYPE_CHECKING:
    from rag.vector_store import VectorStore

try:
    from rag.retrieval.fusion import reciprocal_rank_fusion
except ImportError:
    reciprocal_rank_fusion = None  # type: ignore[assignment]

_log = get_logger(__name__)

_RRF_K: float = 60.0


class HumanisticRetriever:
    """Humanistic-focused evidence retriever with hybrid search + reranking.

    Always applies a ``framework`` metadata filter (default ``"humanistic"``).
    Additional typed filters can be passed via ``HumanisticEvidenceFilter``.
    """

    def __init__(
        self,
        framework: str = "humanistic",
        n_results: int = 10,
        similarity_threshold: float = 0.35,
        hybrid_top_k_ratio: float = 3.0,
        bm25_weight: float = 0.3,
        enable_rerank: bool = True,
        rerank_top_k: int = 10,
    ) -> None:
        self._framework = framework
        self._n_results = n_results
        self._similarity_threshold = similarity_threshold
        self._hybrid_top_k_ratio = hybrid_top_k_ratio
        self._bm25_weight = bm25_weight
        self._enable_rerank = enable_rerank
        self._rerank_top_k = rerank_top_k

        _log.info(
            "humanistic_retriever.init",
            framework=framework,
            n_results=n_results,
            threshold=similarity_threshold,
            hybrid_ratio=hybrid_top_k_ratio,
            bm25_weight=bm25_weight,
            rerank=enable_rerank,
        )

    # ── Public API ─────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        filters: HumanisticEvidenceFilter | None = None,
    ) -> HumanisticEvidenceResult:
        """Run Humanistic evidence retrieval.

        Args:
            query:   Natural-language search query.
            filters: Optional typed metadata filters.

        Returns:
            ``HumanisticEvidenceResult`` with retrieved items sorted by relevance.
        """
        t_start = time.perf_counter()

        if not query or not query.strip():
            _log.warning("humanistic_retriever.empty_query")
            return HumanisticEvidenceResult(
                query=query,
                retrieved=[],
                synthesis="Empty query — no evidence retrieved.",
                key_findings=[],
                evidence_gaps=["Query was empty or whitespace-only"],
                confidence=0.0,
                total_candidates=0,
                elapsed_ms=0.0,
            )

        effective_filter = filters or HumanisticEvidenceFilter(framework=self._framework)
        candidate_n = int(self._n_results * self._hybrid_top_k_ratio)

        chroma_where = _build_where_from_filter(effective_filter)
        bm25_filter = _build_bm25_filter_from_filter(effective_filter)

        # Resolve the framework-scoped collection (manifest.lock alias).
        store, collection_name = resolve_framework_store(self._framework)

        # Dense vector search
        t_dense = time.perf_counter()
        dense_raw = _search_dense(query, candidate_n, chroma_where, store=store)
        dense_ms = (time.perf_counter() - t_dense) * 1000

        dense_candidates = _to_humanistic_items(dense_raw, self._similarity_threshold)
        dense_candidate_dicts = _items_to_dicts(dense_candidates)

        # BM25 keyword search
        t_sparse = time.perf_counter()
        sparse_candidate_dicts: list[dict[str, Any]] = []
        try:
            from rag.retrieval.bm25 import get_bm25_retriever

            bm25 = get_bm25_retriever()
            sparse_raw = bm25.search(
                query,
                n_results=candidate_n,
                metadata_filter=bm25_filter,
                collection_name=collection_name,
            )
            sparse_candidate_dicts = sparse_raw
        except Exception as exc:
            _log.warning("humanistic_retriever.bm25_failed", error=str(exc))
        sparse_ms = (time.perf_counter() - t_sparse) * 1000

        total_candidates = len(dense_raw) + len(sparse_candidate_dicts)

        # RRF Fusion
        dense_weight = 1.0 - self._bm25_weight
        if sparse_candidate_dicts and reciprocal_rank_fusion is not None:
            fused = reciprocal_rank_fusion(
                dense_results=dense_candidate_dicts,
                sparse_results=sparse_candidate_dicts,
                top_k=self._n_results,
                dense_weight=dense_weight,
                sparse_weight=self._bm25_weight,
            )
        elif sparse_candidate_dicts:
            fused = sparse_candidate_dicts[: self._n_results]
        else:
            fused = dense_candidate_dicts[: self._n_results]

        # Cross-encoder rerank
        if self._enable_rerank and fused:
            try:
                from rag.retrieval.reranker import get_reranker

                reranker = get_reranker()
                fused = reranker.rerank(query, fused, top_k=self._rerank_top_k)
            except Exception as exc:
                _log.warning("humanistic_retriever.rerank_failed", error=str(exc))

        items = _dicts_to_items(fused, self._framework)
        if len(items) > self._n_results:
            items = items[: self._n_results]

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        result = HumanisticEvidenceResult(
            query=query,
            retrieved=items,
            synthesis=_build_synthesis(items, query),
            key_findings=[item.content[:200] for item in items[:3]],
            evidence_gaps=[
                "Individual treatment response varies; literature may not reflect this specific presentation",
            ],
            confidence=_compute_confidence(items),
            total_candidates=total_candidates,
            elapsed_ms=round(elapsed_ms, 2),
        )

        _log.info(
            "humanistic_retriever.retrieve_complete",
            query_length=len(query),
            total_candidates=total_candidates,
            dense=len(dense_candidates),
            bm25=len(sparse_candidate_dicts),
            returned=len(items),
            top_score=items[0].quality_rating if items else None,
            elapsed_ms=round(elapsed_ms, 2),
        )

        return result

    async def retrieve_async(
        self,
        query: str,
        filters: HumanisticEvidenceFilter | None = None,
    ) -> HumanisticEvidenceResult:
        """Async version of ``retrieve()``."""
        import asyncio

        return await asyncio.to_thread(self.retrieve, query, filters)

    @property
    def framework(self) -> str:
        return self._framework

    @property
    def config(self) -> dict[str, Any]:
        return {
            "framework": self._framework,
            "n_results": self._n_results,
            "similarity_threshold": self._similarity_threshold,
            "hybrid_top_k_ratio": self._hybrid_top_k_ratio,
            "bm25_weight": self._bm25_weight,
            "enable_rerank": self._enable_rerank,
            "rerank_top_k": self._rerank_top_k,
        }


# ── Filter builders ──────────────────────────────────────────────────────


def _build_where_from_filter(f: HumanisticEvidenceFilter) -> dict[str, Any] | None:
    """Build a ChromaDB ``where`` filter dict from a typed evidence filter."""
    conditions: list[dict[str, Any]] = []
    conditions.append({"framework": {"$eq": f.framework}})

    for field_name in ("chapter", "topic", "intervention", "symptom", "population"):
        val = getattr(f, field_name, None)
        if val is None:
            continue
        if isinstance(val, list):
            conditions.append({field_name: {"$in": val}})
        else:
            conditions.append({field_name: {"$eq": val}})

    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _build_bm25_filter_from_filter(f: HumanisticEvidenceFilter) -> dict[str, Any] | None:
    """Build a flat metadata filter dict for BM25 post-filtering."""
    filt: dict[str, Any] = {}
    filt["framework"] = f.framework
    for field_name in ("chapter", "topic", "intervention", "symptom", "population"):
        val = getattr(f, field_name, None)
        if val is None:
            continue
        if isinstance(val, list):
            filt[field_name] = {"$in": val}
        else:
            filt[field_name] = val
    return filt or None


# ── Dense search ─────────────────────────────────────────────────────────


def _search_dense(
    query: str,
    n_results: int,
    where: dict[str, Any] | None,
    *,
    store: VectorStore | None = None,
) -> list[Any]:
    """Embed the query and run ChromaDB similarity search."""
    from rag.embeddings import embed_text
    from rag.vector_store import get_vector_store

    query_vector = embed_text(query, is_query=True)
    store = store or get_vector_store()
    return store.query_documents(
        query_vector,
        n_results=n_results,
        where=where,
    )


# ── Conversion helpers ───────────────────────────────────────────────────


def _to_humanistic_items(
    raw_results: list[Any],
    threshold: float,
) -> list[HumanisticEvidenceItem]:
    """Convert QueryResult objects to HumanisticEvidenceItem with threshold filtering."""
    kept: list[HumanisticEvidenceItem] = []
    for raw in raw_results:
        if raw.score < threshold:
            continue
        meta = raw.metadata or {}
        kept.append(HumanisticEvidenceItem(
            source=str(raw.source),
            content=str(raw.text),
            relevance=_build_relevance(raw.score),
            quality_rating=float(raw.score),
            framework=str(meta.get("framework", "humanistic")),
            chapter=str(meta.get("chapter")) if meta.get("chapter") else None,
            topic=str(meta.get("topic")) if meta.get("topic") else None,
            intervention=str(meta.get("intervention")) if meta.get("intervention") else None,
            symptom=str(meta.get("symptom")) if meta.get("symptom") else None,
            population=str(meta.get("population")) if meta.get("population") else None,
            page=int(raw.page),
            chunk_id=str(raw.chunk_id),
        ))
    return kept


def _items_to_dicts(items: list[HumanisticEvidenceItem]) -> list[dict[str, Any]]:
    """Convert HumanisticEvidenceItem list to dicts for fusion/rerank."""
    return [
        {
            "chunk_id": item.chunk_id,
            "text": item.content,
            "source": item.source,
            "page": item.page,
            "score": item.quality_rating or 0.0,
            "rank": idx + 1,
            "metadata": {
                "framework": item.framework,
                "chapter": item.chapter,
                "topic": item.topic,
                "intervention": item.intervention,
                "symptom": item.symptom,
                "population": item.population,
            },
        }
        for idx, item in enumerate(items)
    ]


def _dicts_to_items(
    dicts: list[dict[str, Any]],
    framework: str,
) -> list[HumanisticEvidenceItem]:
    """Convert dicts (from fusion/rerank) back to HumanisticEvidenceItem."""
    items: list[HumanisticEvidenceItem] = []
    for rank, d in enumerate(dicts, start=1):
        meta = dict(d.get("metadata", {}))
        score = float(d.get("score", 0.0))
        items.append(HumanisticEvidenceItem(
            source=str(d.get("source", "unknown")),
            content=str(d.get("text", "")),
            relevance=_build_relevance(score),
            quality_rating=score,
            framework=str(meta.get("framework", framework)),
            chapter=str(meta.get("chapter")) if meta.get("chapter") else None,
            topic=str(meta.get("topic")) if meta.get("topic") else None,
            intervention=str(meta.get("intervention")) if meta.get("intervention") else None,
            symptom=str(meta.get("symptom")) if meta.get("symptom") else None,
            population=str(meta.get("population")) if meta.get("population") else None,
            page=int(d.get("page", 0)),
            chunk_id=str(d.get("chunk_id", "")),
        ))
    return items


def _build_relevance(score: float) -> str:
    """Generate a human-readable relevance description from a similarity score."""
    if score >= 0.9:
        return "Highly relevant direct match to the query"
    if score >= 0.75:
        return "Strongly relevant to the clinical query"
    if score >= 0.6:
        return "Moderately relevant context"
    if score >= 0.45:
        return "Somewhat relevant background information"
    return "Tangentially related reference"


def _build_synthesis(items: list[HumanisticEvidenceItem], query: str) -> str:
    """Generate a summary sentence from retrieved items."""
    if not items:
        return "No evidence was retrieved. Consider broadening the search query or adjusting metadata filters."
    sources = {item.source for item in items}
    topics = {item.topic for item in items if item.topic}
    topic_str = f" covering topics such as {', '.join(sorted(topics)[:3])}" if topics else ""
    return (
        f"Retrieved {len(items)} evidence item(s) from {len(sources)} source(s)"
        f"{topic_str} relevant to: {query[:200]}"
    )


def _compute_confidence(items: list[HumanisticEvidenceItem]) -> float:
    """Compute an overall confidence score from the top items."""
    if not items:
        return 0.0
    scores = [item.quality_rating or 0.0 for item in items[:5]]
    return round(sum(scores) / len(scores), 2)


# ── Singleton factory ────────────────────────────────────────────────────

_retriever_lock = threading.Lock()
_retriever_instance: HumanisticRetriever | None = None


def get_humanistic_retriever(*, force_reload: bool = False) -> HumanisticRetriever:
    """Return the process-wide HumanisticRetriever singleton.

    Usage::

        from clinical.retrieval import get_humanistic_retriever
        retriever = get_humanistic_retriever()
        result = retriever.retrieve("empathy and UPR in therapy")
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
        _retriever_instance = HumanisticRetriever()
        _log.info("humanistic_retriever.singleton_created")

    return _retriever_instance
