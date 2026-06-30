from __future__ import annotations

import threading
import time
from typing import Any

from app_logging.logger import get_logger

_log = get_logger(__name__)


class CrossEncoderReranker:
    """Cross-encoder reranker for precision re-ranking of retrieval results.

    Takes a query and a list of candidate chunks, scores each (query, chunk)
    pair with a cross-encoder model, and returns the top candidates sorted
    by the cross-encoder relevance score.

    The cross-encoder is loaded lazily on first use and cached as a singleton
    to avoid repeated model loading overhead.

    Usage::

        reranker = CrossEncoderReranker()
        results = reranker.rerank(query, candidates, top_k=4)
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self._model_name = model_name
        self._model: Any = None
        self._lock = threading.Lock()
        self._loaded = False
        _log.info("reranker.init", model_name=model_name)

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Rerank candidate chunks using cross-encoder scores.

        Args:
            query:       The original search query.
            candidates:  List of candidate dicts, each with at least ``text``,
                         ``source``, ``page``, ``score``, ``chunk_id``, ``metadata``.
            top_k:       Number of results to return after reranking.
                         Defaults to all candidates.

        Returns:
            Candidates sorted by descending cross-encoder score, with an
            additional ``rerank_score`` key and updated ``rank`` and ``score``
            fields.
        """
        if not candidates:
            return []

        if not query.strip():
            _log.warning("reranker.empty_query")
            return candidates

        self._ensure_loaded()
        if self._model is None:
            _log.warning("reranker.model_unavailable_returning_original_order")
            return candidates

        pairs = [(query, c["text"]) for c in candidates]
        t_start = time.perf_counter()

        try:
            scores = self._model.predict(pairs)
        except Exception as exc:
            _log.error("reranker.predict_failed", error=str(exc))
            return candidates

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        for c, ce_score in zip(candidates, scores):
            c["rerank_score"] = round(float(ce_score), 6)
            c["score"] = c["rerank_score"]

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        effective_k = top_k if top_k is not None else len(candidates)
        kept = candidates[:effective_k]

        for rank, c in enumerate(kept, start=1):
            c["rank"] = rank

        _log.info(
            "reranker.complete",
            candidates_in=len(candidates),
            candidates_out=len(kept),
            top_score=kept[0]["rerank_score"] if kept else None,
            elapsed_ms=round(elapsed_ms, 2),
        )
        return kept

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            try:
                from sentence_transformers.cross_encoder import CrossEncoder
                self._model = CrossEncoder(self._model_name)
                self._loaded = True
                _log.info("reranker.model_loaded", model_name=self._model_name)
            except ImportError:
                _log.warning(
                    "reranker.sentence_transformers_not_installed, "
                    "cross-encoder reranking disabled"
                )
                self._model = None
                self._loaded = True
            except Exception as exc:
                _log.error(
                    "reranker.load_failed",
                    model_name=self._model_name,
                    error=str(exc),
                )
                self._model = None
                self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_name(self) -> str:
        return self._model_name

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not_loaded"
        return f"CrossEncoderReranker(model={self._model_name!r}, {status})"


_reranker_lock = threading.Lock()
_reranker_instance: CrossEncoderReranker | None = None


def get_reranker(*, force_reload: bool = False) -> CrossEncoderReranker:
    global _reranker_instance
    if force_reload:
        with _reranker_lock:
            _reranker_instance = None
    if _reranker_instance is not None:
        return _reranker_instance
    with _reranker_lock:
        if _reranker_instance is not None:
            return _reranker_instance
        from config.settings import get_settings
        cfg = get_settings().rag
        _reranker_instance = CrossEncoderReranker(
            model_name=cfg.cross_encoder_model,
        )
        _log.info("reranker.singleton_created")
    return _reranker_instance
