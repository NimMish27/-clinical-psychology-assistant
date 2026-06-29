"""
tests/unit/rag/test_retriever.py
──────────────────────────────────
Unit tests for the clinical document retriever.

All external dependencies (embed_text, VectorStore) are mocked.
Tests cover: exception hierarchy, output models, query preprocessing,
filter building, threshold filtering, rank re-assignment, async interface,
singleton lifecycle, and the full retrieve() pipeline.

Run:
    pytest tests/unit/rag/test_retriever.py -v
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from rag.retriever import (
    EmptyQueryError,
    EmbeddingFailedError,
    NoResultsError,
    Retriever,
    RetrievalResult,
    RetrievedChunk,
    RetrieverError,
    SearchFailedError,
    _build_where_filter,
    _to_retrieved_chunks,
    get_retriever,
)


# ---- Fixtures & helpers ---------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_singleton():
    """Clear singleton before every test."""
    import rag.retriever as r_mod
    r_mod._retriever_instance = None
    yield
    r_mod._retriever_instance = None


def _make_settings(n_results: int = 5, threshold: float = 0.35, top_k: int = 10):
    mock = MagicMock()
    mock.rag.top_k = top_k
    mock.rag.similarity_threshold = threshold
    return mock


def _make_retriever(**kwargs) -> Retriever:
    """Build a Retriever with settings mocked."""
    with patch("config.settings.get_settings", return_value=_make_settings()):
        return Retriever(**kwargs)


@dataclass
class FakeQueryResult:
    """Mimics rag.vector_store.QueryResult without importing it."""
    chunk_id: str
    text: str
    score: float
    source: str
    page: int
    metadata: dict
    rank: int


def _fake_qr(
    rank: int = 1,
    score: float = 0.9,
    source: str = "DSM5.pdf",
    page: int = 12,
    text: str = "Clinical criterion text here.",
) -> FakeQueryResult:
    return FakeQueryResult(
        chunk_id=f"DSM5__p{page:04d}__c{rank:04d}",
        text=text,
        score=score,
        source=source,
        page=page,
        metadata={"source": source, "page": page, "chunk_index": rank - 1},
        rank=rank,
    )


def _patch_embed(vector: list[float] | None = None):
    """Patch embed_text to return a fixed vector."""
    v = vector or [0.1] * 1024
    return patch("rag.retriever.Retriever._embed_query", return_value=v)


def _patch_search(results: list):
    """Patch Retriever._search to return fixed results."""
    return patch("rag.retriever.Retriever._search", return_value=results)


# ---- Exception hierarchy --------------------------------------------------------

class TestExceptions:
    def test_base_error(self):
        exc = RetrieverError("base")
        assert "base" in str(exc)
        assert exc.cause is None

    def test_base_error_with_cause(self):
        cause = ValueError("root")
        exc = RetrieverError("wrapper", cause=cause)
        assert "ValueError" in str(exc)
        assert "root" in str(exc)

    def test_all_subclass_retriever_error(self):
        for klass in (EmptyQueryError, EmbeddingFailedError, SearchFailedError, NoResultsError):
            assert issubclass(klass, RetrieverError)


# ---- RetrievedChunk model -------------------------------------------------------

class TestRetrievedChunk:
    def _make(self, **kw) -> RetrievedChunk:
        defaults = dict(
            text="Criterion A: persistent depressed mood.",
            source="DSM5.pdf",
            page=12,
            score=0.91,
            chunk_id="DSM5__p0012__c0000",
            rank=1,
            metadata={"source": "DSM5.pdf", "page": 12},
        )
        return RetrievedChunk(**{**defaults, **kw})

    def test_to_dict_format(self):
        chunk = self._make()
        d = chunk.to_dict()
        assert set(d.keys()) == {"text", "source", "page", "score"}
        assert d["text"] == "Criterion A: persistent depressed mood."
        assert d["source"] == "DSM5.pdf"
        assert d["page"] == 12
        assert d["score"] == 0.91

    def test_to_prompt_citation_format(self):
        chunk = self._make()
        citation = chunk.to_prompt_citation()
        assert "DSM5.pdf" in citation
        assert "p.12" in citation
        assert "0.91" in citation

    def test_to_prompt_citation_truncates_long_text(self):
        chunk = self._make(text="A" * 200)
        citation = chunk.to_prompt_citation()
        assert "\u2026" in citation

    def test_to_prompt_citation_no_ellipsis_short_text(self):
        chunk = self._make(text="Short text.")
        citation = chunk.to_prompt_citation()
        assert "\u2026" not in citation

    def test_frozen(self):
        chunk = self._make()
        with pytest.raises(Exception):
            chunk.score = 0.5  # type: ignore

    def test_repr(self):
        chunk = self._make()
        r = repr(chunk)
        assert "rank=1" in r
        assert "0.9100" in r
        assert "DSM5.pdf" in r


# ---- RetrievalResult model ------------------------------------------------------

class TestRetrievalResult:
    def _make_result(self, n: int = 3, score: float = 0.88) -> RetrievalResult:
        chunks = [
            RetrievedChunk(
                text=f"Chunk text {i}.", source="DSM5.pdf", page=i + 1,
                score=score - i * 0.05, chunk_id=f"id_{i}", rank=i + 1,
            )
            for i in range(n)
        ]
        return RetrievalResult(
            query="test query",
            chunks=chunks,
            total_candidates=n + 2,
            threshold_used=0.35,
            n_results_requested=5,
            elapsed_ms=45.2,
            embedding_ms=12.1,
            search_ms=33.1,
        )

    def test_found_true_when_chunks(self):
        result = self._make_result(3)
        assert result.found is True

    def test_found_false_when_empty(self):
        result = self._make_result(0)
        assert result.found is False

    def test_top_returns_first_chunk(self):
        result = self._make_result(3)
        assert result.top is result.chunks[0]

    def test_top_returns_none_when_empty(self):
        result = self._make_result(0)
        assert result.top is None

    def test_to_dicts_format(self):
        result = self._make_result(2)
        dicts = result.to_dicts()
        assert len(dicts) == 2
        assert all(set(d.keys()) == {"text", "source", "page", "score"} for d in dicts)

    def test_to_context_string_joins_texts(self):
        result = self._make_result(3)
        ctx = result.to_context_string()
        assert "Chunk text 0." in ctx
        assert "Chunk text 2." in ctx

    def test_to_context_string_custom_separator(self):
        result = self._make_result(2)
        ctx = result.to_context_string(separator=" | ")
        assert " | " in ctx

    def test_to_context_string_empty(self):
        result = self._make_result(0)
        assert result.to_context_string() == ""

    def test_to_cited_context_contains_citations(self):
        result = self._make_result(2)
        ctx = result.to_cited_context()
        assert "DSM5.pdf" in ctx
        assert "score=" in ctx

    def test_repr(self):
        result = self._make_result(3)
        r = repr(result)
        assert "found=True" in r
        assert "chunks=3" in r


# ---- _build_where_filter --------------------------------------------------------

class TestBuildWhereFilter:
    def test_no_filters_returns_none(self):
        assert _build_where_filter(where=None, source_filter=None, page_range=None) is None

    def test_explicit_where_returned_as_is(self):
        w = {"source": "DSM5.pdf"}
        result = _build_where_filter(where=w, source_filter="other.pdf", page_range=None)
        assert result is w   # exact same object, not a copy

    def test_source_filter_only(self):
        result = _build_where_filter(where=None, source_filter="DSM5.pdf", page_range=None)
        assert result == {"source": "DSM5.pdf"}

    def test_page_range_only(self):
        result = _build_where_filter(where=None, source_filter=None, page_range=(10, 50))
        assert result is not None
        # Should use $and with two page conditions
        conditions = result["$and"]
        assert any("$gte" in str(c) for c in conditions)
        assert any("$lte" in str(c) for c in conditions)

    def test_source_and_page_range_combined(self):
        result = _build_where_filter(
            where=None,
            source_filter="DSM5.pdf",
            page_range=(100, 200),
        )
        assert result is not None
        assert "$and" in result
        conditions = result["$and"]
        assert len(conditions) == 3   # source + page gte + page lte

    def test_invalid_page_range_raises(self):
        with pytest.raises(ValueError, match="page_range"):
            _build_where_filter(where=None, source_filter=None, page_range=(50, 10))

    def test_single_condition_no_and_wrapper(self):
        result = _build_where_filter(where=None, source_filter="doc.pdf", page_range=None)
        assert "$and" not in result   # single condition, no wrapper needed


# ---- _to_retrieved_chunks -------------------------------------------------------

class TestToRetrievedChunks:
    def test_all_above_threshold_kept(self):
        raw = [_fake_qr(rank=i, score=0.9 - i * 0.05) for i in range(1, 4)]
        chunks = _to_retrieved_chunks(raw, threshold=0.35)
        assert len(chunks) == 3

    def test_below_threshold_filtered(self):
        raw = [
            _fake_qr(rank=1, score=0.92),
            _fake_qr(rank=2, score=0.34),   # below 0.35
            _fake_qr(rank=3, score=0.81),
        ]
        chunks = _to_retrieved_chunks(raw, threshold=0.35)
        assert len(chunks) == 2
        assert all(c.score >= 0.35 for c in chunks)

    def test_ranks_re_assigned_after_filtering(self):
        raw = [
            _fake_qr(rank=1, score=0.90),
            _fake_qr(rank=2, score=0.20),  # filtered out
            _fake_qr(rank=3, score=0.75),
        ]
        chunks = _to_retrieved_chunks(raw, threshold=0.35)
        assert len(chunks) == 2
        assert chunks[0].rank == 1   # re-assigned
        assert chunks[1].rank == 2   # re-assigned from 3

    def test_empty_input_returns_empty(self):
        assert _to_retrieved_chunks([], threshold=0.5) == []

    def test_all_filtered_returns_empty(self):
        raw = [_fake_qr(rank=i, score=0.1) for i in range(1, 4)]
        chunks = _to_retrieved_chunks(raw, threshold=0.5)
        assert chunks == []

    def test_threshold_zero_keeps_all(self):
        raw = [_fake_qr(rank=i, score=0.01) for i in range(1, 4)]
        chunks = _to_retrieved_chunks(raw, threshold=0.0)
        assert len(chunks) == 3

    def test_fields_mapped_correctly(self):
        raw = [_fake_qr(rank=1, score=0.88, source="CBT.pdf", page=42)]
        chunks = _to_retrieved_chunks(raw, threshold=0.0)
        c = chunks[0]
        assert c.text == "Clinical criterion text here."
        assert c.source == "CBT.pdf"
        assert c.page == 42
        assert c.score == 0.88
        assert c.rank == 1


# ---- Retriever initialisation ---------------------------------------------------

class TestRetrieverInit:
    def test_default_values_from_settings(self):
        with patch("config.settings.get_settings") as mock_settings:
            mock_settings.return_value.rag.top_k = 7
            mock_settings.return_value.rag.similarity_threshold = 0.42
            r = Retriever()
        assert r.n_results == 7
        assert r.similarity_threshold == 0.42

    def test_explicit_overrides_settings(self):
        r = _make_retriever(n_results=3, similarity_threshold=0.6)
        assert r.n_results == 3
        assert r.similarity_threshold == 0.6

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError, match="similarity_threshold"):
            _make_retriever(similarity_threshold=1.5)

    def test_invalid_n_results_raises(self):
        with pytest.raises(ValueError, match="n_results"):
            _make_retriever(n_results=0)

    def test_config_property(self):
        r = _make_retriever(n_results=5, similarity_threshold=0.4)
        cfg = r.config
        assert cfg["n_results"] == 5
        assert cfg["similarity_threshold"] == 0.4

    def test_repr(self):
        r = _make_retriever(n_results=5, similarity_threshold=0.4)
        assert "5" in repr(r)
        assert "0.4" in repr(r)


# ---- Query preprocessing --------------------------------------------------------

class TestQueryPreprocessing:
    def _preprocess(self, query: str, max_len: int = 1000) -> str:
        r = _make_retriever(max_query_length=max_len)
        return r._preprocess_query(query)

    def test_empty_query_raises(self):
        r = _make_retriever()
        with pytest.raises(EmptyQueryError):
            r._preprocess_query("")

    def test_whitespace_only_raises(self):
        r = _make_retriever()
        with pytest.raises(EmptyQueryError):
            r._preprocess_query("   \n\t  ")

    def test_strips_outer_whitespace(self):
        assert self._preprocess("  hello  ") == "hello"

    def test_collapses_internal_spaces(self):
        result = self._preprocess("What  is   CBT?")
        assert "  " not in result

    def test_collapses_newlines(self):
        result = self._preprocess("What is CBT?\n\n\nExplain criteria.")
        assert "\n\n" not in result

    def test_truncates_long_query(self):
        long = "A" * 2000
        result = self._preprocess(long, max_len=500)
        assert len(result) == 500

    def test_short_query_not_truncated(self):
        query = "What is DSM-5?"
        result = self._preprocess(query)
        assert result == query

    def test_preserves_clinical_terms(self):
        query = "DSM-5 Criterion A for MDD: CBT-I vs pharmacotherapy?"
        result = self._preprocess(query)
        assert "DSM-5" in result
        assert "CBT-I" in result


# ---- retrieve() pipeline --------------------------------------------------------

class TestRetrieve:
    def _raw_results(self, n: int = 5) -> list[FakeQueryResult]:
        return [_fake_qr(rank=i + 1, score=0.95 - i * 0.05) for i in range(n)]

    def test_happy_path_returns_result(self):
        r = _make_retriever(n_results=5, similarity_threshold=0.35)
        raw = self._raw_results(5)
        with _patch_embed(), _patch_search(raw):
            result = r.retrieve("What are the DSM-5 criteria for MDD?")

        assert isinstance(result, RetrievalResult)
        assert result.found is True
        assert len(result.chunks) == 5

    def test_returns_top_n_results(self):
        r = _make_retriever(n_results=5, similarity_threshold=0.0)
        raw = self._raw_results(5)
        with _patch_embed(), _patch_search(raw):
            result = r.retrieve("CBT techniques")
        assert len(result.chunks) == 5

    def test_threshold_filters_low_scores(self):
        r = _make_retriever(n_results=5, similarity_threshold=0.80)
        raw = [
            _fake_qr(rank=1, score=0.92),
            _fake_qr(rank=2, score=0.85),
            _fake_qr(rank=3, score=0.71),  # below 0.80
            _fake_qr(rank=4, score=0.60),  # below 0.80
        ]
        with _patch_embed(), _patch_search(raw):
            result = r.retrieve("anxiety disorders")
        assert len(result.chunks) == 2
        assert all(c.score >= 0.80 for c in result.chunks)

    def test_per_call_threshold_override(self):
        r = _make_retriever(n_results=5, similarity_threshold=0.35)
        raw = [_fake_qr(rank=1, score=0.50), _fake_qr(rank=2, score=0.30)]
        with _patch_embed(), _patch_search(raw):
            result = r.retrieve("query", similarity_threshold=0.60)
        assert len(result.chunks) == 0  # both below 0.60

    def test_per_call_n_results_override(self):
        r = _make_retriever(n_results=10, similarity_threshold=0.0)
        raw = self._raw_results(3)
        with _patch_embed() as mock_embed, _patch_search(raw) as mock_search:
            r.retrieve("query", n_results=3)
        # _search should have been called with n=3, not 10
        mock_search.assert_called_once()
        call_args = mock_search.call_args
        assert call_args[0][1] == 3

    def test_ranks_correct_in_output(self):
        r = _make_retriever(similarity_threshold=0.0)
        raw = self._raw_results(3)
        with _patch_embed(), _patch_search(raw):
            result = r.retrieve("query")
        assert [c.rank for c in result.chunks] == [1, 2, 3]

    def test_timing_fields_populated(self):
        r = _make_retriever()
        with _patch_embed(), _patch_search(self._raw_results(2)):
            result = r.retrieve("timing test query")
        assert result.elapsed_ms >= 0
        assert result.embedding_ms >= 0
        assert result.search_ms >= 0

    def test_query_stored_in_result(self):
        r = _make_retriever()
        with _patch_embed(), _patch_search([]):
            result = r.retrieve("What is exposure therapy?")
        assert result.query == "What is exposure therapy?"

    def test_empty_query_raises(self):
        r = _make_retriever()
        with pytest.raises(EmptyQueryError):
            r.retrieve("")

    def test_empty_results_no_raise_by_default(self):
        r = _make_retriever(raise_on_empty=False)
        with _patch_embed(), _patch_search([]):
            result = r.retrieve("obscure query")
        assert result.found is False
        assert result.chunks == []

    def test_empty_results_raises_when_configured(self):
        r = _make_retriever(raise_on_empty=True)
        with _patch_embed(), _patch_search([]):
            with pytest.raises(NoResultsError):
                r.retrieve("obscure query")

    def test_embedding_error_wrapped(self):
        r = _make_retriever()
        with patch("rag.embeddings.embed_text",
                   side_effect=RuntimeError("model not loaded")):
            with pytest.raises(EmbeddingFailedError):
                r.retrieve("query")

    def test_search_error_wrapped(self):
        r = _make_retriever()
        mock_store = MagicMock()
        mock_store.query_documents.side_effect = RuntimeError("ChromaDB down")
        with _patch_embed():
            with patch("rag.vector_store.get_vector_store",
                       return_value=mock_store):
                with pytest.raises(SearchFailedError):
                    r.retrieve("query")

    def test_total_candidates_reflects_raw_count(self):
        r = _make_retriever(similarity_threshold=0.80)
        raw = self._raw_results(5)  # 5 raw results, some may be filtered
        with _patch_embed(), _patch_search(raw):
            result = r.retrieve("query")
        assert result.total_candidates == 5

    def test_to_dict_output_format(self):
        r = _make_retriever(similarity_threshold=0.0)
        raw = [_fake_qr(rank=1, score=0.91, source="DSM5.pdf", page=15,
                        text="Criterion text.")]
        with _patch_embed(), _patch_search(raw):
            result = r.retrieve("query")
        d = result.to_dicts()[0]
        assert d == {"text": "Criterion text.", "source": "DSM5.pdf",
                     "page": 15, "score": 0.91}


# ---- source_filter and page_range -----------------------------------------------

class TestFilters:
    def test_source_filter_passed_to_search(self):
        r = _make_retriever()
        with _patch_embed() as mock_embed, \
             patch("rag.retriever.Retriever._search", return_value=[]) as mock_search:
            r.retrieve("query", source_filter="DSM5.pdf")
        call_kwargs = mock_search.call_args
        where_arg = call_kwargs[0][2]   # positional: (vector, n, where)
        assert where_arg == {"source": "DSM5.pdf"}

    def test_page_range_passed_to_search(self):
        r = _make_retriever()
        with _patch_embed(), \
             patch("rag.retriever.Retriever._search", return_value=[]) as mock_search:
            r.retrieve("query", page_range=(10, 50))
        where_arg = mock_search.call_args[0][2]
        assert "$and" in where_arg

    def test_explicit_where_overrides_source_filter(self):
        r = _make_retriever()
        custom_where = {"$and": [{"source": "ICD11.pdf"}]}
        with _patch_embed(), \
             patch("rag.retriever.Retriever._search", return_value=[]) as mock_search:
            r.retrieve("query", where=custom_where, source_filter="DSM5.pdf")
        where_arg = mock_search.call_args[0][2]
        assert where_arg is custom_where


# ---- retrieve_chunks and retrieve_with_sources -----------------------------------

class TestConvenienceMethods:
    def _raw(self) -> list[FakeQueryResult]:
        return [
            _fake_qr(rank=1, score=0.90, source="DSM5.pdf", page=1),
            _fake_qr(rank=2, score=0.85, source="CBT.pdf", page=5),
            _fake_qr(rank=3, score=0.80, source="DSM5.pdf", page=3),
        ]

    def test_retrieve_chunks_returns_list(self):
        r = _make_retriever(similarity_threshold=0.0)
        with _patch_embed(), _patch_search(self._raw()):
            chunks = r.retrieve_chunks("query")
        assert isinstance(chunks, list)
        assert all(isinstance(c, RetrievedChunk) for c in chunks)

    def test_retrieve_with_sources_groups_by_source(self):
        r = _make_retriever(similarity_threshold=0.0)
        with _patch_embed(), _patch_search(self._raw()):
            grouped = r.retrieve_with_sources("query")
        assert "DSM5.pdf" in grouped
        assert "CBT.pdf" in grouped
        assert len(grouped["DSM5.pdf"]) == 2
        assert len(grouped["CBT.pdf"]) == 1

    def test_retrieve_with_sources_empty(self):
        r = _make_retriever(similarity_threshold=0.99)
        with _patch_embed(), _patch_search(self._raw()):
            grouped = r.retrieve_with_sources("query")
        assert grouped == {}


# ---- Async interface ------------------------------------------------------------

class TestAsyncRetriever:
    def test_aretrieve_returns_retrieval_result(self):
        r = _make_retriever(similarity_threshold=0.0)
        raw = [_fake_qr(rank=1, score=0.88)]
        with _patch_embed(), _patch_search(raw):
            result = asyncio.run(r.aretrieve("What is CBT?"))
        assert isinstance(result, RetrievalResult)
        assert result.found is True

    def test_aretrieve_chunks_returns_list(self):
        r = _make_retriever(similarity_threshold=0.0)
        raw = [_fake_qr(rank=1, score=0.88), _fake_qr(rank=2, score=0.75)]
        with _patch_embed(), _patch_search(raw):
            chunks = asyncio.run(r.aretrieve_chunks("query"))
        assert len(chunks) == 2

    def test_aretrieve_propagates_errors(self):
        r = _make_retriever()
        async def run():
            with pytest.raises(EmptyQueryError):
                await r.aretrieve("")
        asyncio.run(run())


# ---- Singleton: get_retriever ---------------------------------------------------

class TestGetRetriever:
    def test_returns_retriever_instance(self):
        with patch("config.settings.get_settings", return_value=_make_settings()):
            r = get_retriever()
        assert isinstance(r, Retriever)

    def test_same_instance_returned_twice(self):
        with patch("config.settings.get_settings", return_value=_make_settings()):
            a = get_retriever()
            b = get_retriever()
        assert a is b

    def test_force_reload_creates_new_instance(self):
        with patch("config.settings.get_settings", return_value=_make_settings()):
            a = get_retriever()
            b = get_retriever(force_reload=True)
        assert a is not b

    def test_thread_safety(self):
        instances = []
        barrier = threading.Barrier(6)

        def get_it():
            barrier.wait()
            with patch("config.settings.get_settings", return_value=_make_settings()):
                instances.append(get_retriever())

        threads = [threading.Thread(target=get_it) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(i is instances[0] for i in instances)
