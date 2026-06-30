from __future__ import annotations

import re
import threading
import time
from typing import Any

from app_logging.logger import get_logger

_log = get_logger(__name__)


class BM25Retriever:
    """Sparse keyword retriever using the BM25 ranking function.

    Builds an in-memory BM25 index from all chunk texts in the ChromaDB
    collection.  Designed to be used alongside the dense (vector) retriever
    in a hybrid retrieval pipeline.

    The index is built lazily on first query and refreshed when
    ``force_reload=True`` is passed to ``search()``.

    Usage::

        bm25 = BM25Retriever(k1=1.5, b=0.75)
        results = bm25.search("academic burnout", n_results=5)
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._k1 = k1
        self._b = b
        self._corpus: list[str] = []
        self._chunk_ids: list[str] = []
        self._metadatas: list[dict[str, Any]] = []
        self._bm25: Any = None
        self._lock = threading.Lock()
        self._loaded = False
        _log.info("bm25.init", k1=k1, b=b)

    def search(
        self,
        query: str,
        *,
        n_results: int = 10,
        metadata_filter: dict[str, Any] | None = None,
        force_reload: bool = False,
    ) -> list[dict[str, Any]]:
        """Run BM25 search against the corpus.

        Args:
            query:            Keyword query string.
            n_results:        Number of results to return.
            metadata_filter:  Filter results by metadata field values.
                              Supported operators: ``$in``, ``$eq`` (exact match).
            force_reload:     Reload corpus from ChromaDB before searching.

        Returns:
            List of dicts with keys ``chunk_id``, ``text``, ``score``,
            ``source``, ``page``, ``metadata``, ``rank``, sorted by
            descending BM25 score.
        """
        if force_reload:
            self._loaded = False

        if not self._loaded:
            self._load()

        if not self._corpus:
            _log.warning("bm25.empty_corpus")
            return []

        tokenised_query = self._tokenize(query)

        if not tokenised_query:
            _log.warning("bm25.empty_query_after_tokenization")
            return []

        scores = self._bm25.get_scores(tokenised_query)

        scored: list[tuple[int, float]] = [
            (i, float(scores[i])) for i in range(len(scores))
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        results: list[dict[str, Any]] = []
        for idx, bm25_score in scored:
            meta = self._metadatas[idx]
            if metadata_filter and not self._match_metadata(meta, metadata_filter):
                continue
            results.append({
                "chunk_id": self._chunk_ids[idx],
                "text": self._corpus[idx],
                "score": round(bm25_score, 6),
                "source": str(meta.get("source", "unknown")),
                "page": int(meta.get("page", 0)),
                "metadata": meta,
                "rank": 0,
            })
            if len(results) >= n_results:
                break

        for rank, r in enumerate(results, start=1):
            r["rank"] = rank

        _log.info(
            "bm25.search_complete",
            query_length=len(query),
            candidates=len(scored),
            returned=len(results),
        )
        return results

    def _load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            try:
                from rag.vector_store import get_vector_store
                store = get_vector_store()
                collection = store._resolve_collection()
                raw = collection.get(include=["documents", "metadatas"])
            except Exception as exc:
                _log.error("bm25.load_failed", error=str(exc))
                self._corpus = []
                self._chunk_ids = []
                self._metadatas = []
                self._bm25 = None
                self._loaded = True
                return

            ids_raw = raw.get("ids") or []
            docs_raw = raw.get("documents") or []
            metas_raw = raw.get("metadatas") or []

            self._chunk_ids = list(ids_raw)
            self._corpus = [d or "" for d in docs_raw]
            self._metadatas = [m or {} for m in metas_raw]

            if not self._corpus:
                _log.warning("bm25.load_empty_collection")
                self._bm25 = None
                self._loaded = True
                return

            t_start = time.perf_counter()
            tokenized_corpus = [self._tokenize(doc) for doc in self._corpus]
            try:
                from rank_bm25 import BM25Okapi
                self._bm25 = BM25Okapi(tokenized_corpus, k1=self._k1, b=self._b)
            except ImportError:
                _log.warning("bm25.rank_bm25_not_installed, using simple BM25")
                self._bm25 = _SimpleBM25(tokenized_corpus, k1=self._k1, b=self._b)
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            self._loaded = True
            _log.info(
                "bm25.index_built",
                documents=len(self._corpus),
                elapsed_ms=round(elapsed_ms, 2),
            )

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return [t for t in text.split() if len(t) > 1]

    def _match_metadata(
        self,
        meta: dict[str, Any],
        filt: dict[str, Any],
    ) -> bool:
        for key, condition in filt.items():
            val = meta.get(key)
            if isinstance(condition, dict) and "$in" in condition:
                if val not in condition["$in"]:
                    return False
            elif isinstance(condition, dict) and "$eq" in condition:
                if val != condition["$eq"]:
                    return False
            else:
                if val != condition:
                    return False
        return True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def corpus_size(self) -> int:
        return len(self._corpus)

    def __repr__(self) -> str:
        return f"BM25Retriever(k1={self._k1}, b={self._b}, loaded={self._loaded}, docs={len(self._corpus)})"


class _SimpleBM25:
    """Minimal BM25 implementation when rank_bm25 is not installed."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self._k1 = k1
        self._b = b
        self._corpus = corpus
        self._doc_len = [len(d) for d in corpus]
        self._avgdl = sum(self._doc_len) / len(corpus) if corpus else 0.0
        self._nd = len(corpus)

        from collections import Counter
        self._df: dict[str, int] = {}
        for doc in corpus:
            for term in set(doc):
                self._df[term] = self._df.get(term, 0) + 1

    def get_scores(self, query: list[str]) -> list[float]:
        from collections import Counter
        scores = [0.0] * self._nd
        for term in query:
            if term not in self._df:
                continue
            idf = self._idf(term)
            for i, doc in enumerate(self._corpus):
                tf = Counter(doc).get(term, 0)
                if tf == 0:
                    continue
                denom = tf + self._k1 * (1 - self._b + self._b * self._doc_len[i] / self._avgdl)
                scores[i] += idf * (tf * (self._k1 + 1)) / denom
        return scores

    def _idf(self, term: str) -> float:
        import math
        n = self._df.get(term, 0)
        return math.log(1 + (self._nd - n + 0.5) / (n + 0.5))


_bm25_lock = threading.Lock()
_bm25_instance: BM25Retriever | None = None


def get_bm25_retriever(*, force_reload: bool = False) -> BM25Retriever:
    global _bm25_instance
    if force_reload:
        with _bm25_lock:
            _bm25_instance = None
    if _bm25_instance is not None:
        return _bm25_instance
    with _bm25_lock:
        if _bm25_instance is not None:
            return _bm25_instance
        from config.settings import get_settings
        cfg = get_settings().rag
        _bm25_instance = BM25Retriever(k1=cfg.bm25_k1, b=cfg.bm25_b)
        _log.info("bm25.singleton_created")
    return _bm25_instance
