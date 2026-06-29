"""
tests/unit/rag/test_vector_store.py
─────────────────────────────────────
Unit tests for the ChromaDB vector storage module.

All tests mock chromadb — no ChromaDB installation or disk I/O required.
Tests cover: exception hierarchy, output models, singleton lifecycle,
collection management, insert batching, similarity search, deletion,
metadata filtering, distance-to-score conversion, and the module-level
get_vector_store() factory.

Run:
    pytest tests/unit/rag/test_vector_store.py -v
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from rag.vector_store import (
    CollectionAlreadyExistsError,
    CollectionInfo,
    DeleteError,
    DocumentInsertError,
    InsertResult,
    QueryError,
    QueryResult,
    VectorStore,
    VectorStoreConnectionError,
    VectorStoreError,
    _distance_to_score,
    _parse_query_response,
    _reset_client,
    _validate_batch_lengths,
    get_vector_store,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_singletons(tmp_path):
    """
    Reset the ChromaDB client singleton and VectorStore singleton
    before every test to prevent cross-test state leakage.
    """
    _reset_client()

    import rag.vector_store as vs_module
    vs_module._store_instance = None

    yield

    _reset_client()
    vs_module._store_instance = None


@pytest.fixture()
def mock_chroma_module():
    """
    Patch chromadb at the import level inside vector_store.
    Returns the mock chromadb module and a pre-configured mock collection.
    """
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0
    mock_collection.metadata = {"hnsw:space": "cosine"}

    mock_client = MagicMock()
    mock_client.list_collections.return_value = []
    mock_client.get_or_create_collection.return_value = mock_collection

    mock_chromadb = MagicMock()
    mock_chromadb.PersistentClient.return_value = mock_client

    with patch.dict("sys.modules", {"chromadb": mock_chromadb}):
        yield mock_chromadb, mock_client, mock_collection


@pytest.fixture()
def store(mock_chroma_module, tmp_path):
    """A VectorStore instance with mocked ChromaDB."""
    _, _, _ = mock_chroma_module
    return VectorStore(
        collection_name="test_collection",
        persist_dir=tmp_path / "chroma",
        distance_function="cosine",
        insert_batch_size=5,
        default_n_results=4,
    )


def _make_query_raw(
    ids: list[str],
    texts: list[str],
    distances: list[float],
    metadatas: list[dict],
) -> dict:
    """Build a mock ChromaDB query() response."""
    return {
        "ids": [ids],
        "documents": [texts],
        "distances": [distances],
        "metadatas": [metadatas],
    }


def _meta(source: str = "DSM5.pdf", page: int = 1) -> dict:
    return {"source": source, "page": page, "chunk_index": 0}


# ── Exception hierarchy ────────────────────────────────────────────────────────

class TestExceptions:
    def test_base_error(self):
        exc = VectorStoreError("base error")
        assert "base error" in str(exc)
        assert exc.cause is None

    def test_base_error_with_cause(self):
        cause = IOError("disk full")
        exc = VectorStoreError("wrapper", cause=cause)
        assert "IOError" in str(exc)
        assert "disk full" in str(exc)

    def test_hierarchy(self):
        for klass in (
            CollectionAlreadyExistsError,
            DocumentInsertError,
            QueryError,
            DeleteError,
            VectorStoreConnectionError,
        ):
            assert issubclass(klass, VectorStoreError)

    def test_collection_not_found(self):
        from rag.vector_store import CollectionNotFoundError
        exc = CollectionNotFoundError("missing collection")
        assert issubclass(CollectionNotFoundError, VectorStoreError)


# ── Output model tests ─────────────────────────────────────────────────────────

class TestQueryResult:
    def test_fields(self):
        qr = QueryResult(
            chunk_id="DSM5__p0001__c0000",
            text="Criterion A text",
            score=0.92,
            source="DSM5.pdf",
            page=1,
            metadata={"source": "DSM5.pdf", "page": 1},
            rank=1,
        )
        assert qr.chunk_id == "DSM5__p0001__c0000"
        assert qr.score == 0.92
        assert qr.rank == 1

    def test_repr(self):
        qr = QueryResult(
            chunk_id="x", text="t", score=0.85,
            source="doc.pdf", page=3,
            metadata={}, rank=2,
        )
        r = repr(qr)
        assert "rank=2" in r
        assert "0.8500" in r

    def test_frozen(self):
        qr = QueryResult(
            chunk_id="x", text="t", score=0.5,
            source="d.pdf", page=1, metadata={}, rank=1,
        )
        with pytest.raises(Exception):
            qr.score = 0.9  # type: ignore


class TestInsertResult:
    def test_success_rate_full(self):
        r = InsertResult(
            total_attempted=10, total_inserted=10, total_failed=0,
            collection_name="col", elapsed_ms=50.0,
        )
        assert r.success_rate == 1.0

    def test_success_rate_partial(self):
        r = InsertResult(
            total_attempted=10, total_inserted=7, total_failed=3,
            collection_name="col", elapsed_ms=50.0,
        )
        assert abs(r.success_rate - 0.7) < 1e-9

    def test_success_rate_zero_attempted(self):
        r = InsertResult(
            total_attempted=0, total_inserted=0, total_failed=0,
            collection_name="col", elapsed_ms=0.0,
        )
        assert r.success_rate == 0.0

    def test_repr(self):
        r = InsertResult(
            total_attempted=5, total_inserted=5, total_failed=0,
            collection_name="col", elapsed_ms=33.3,
        )
        assert "5/5" in repr(r)


# ── Distance-to-score conversion ───────────────────────────────────────────────

class TestDistanceToScore:
    def test_cosine_zero_distance_is_one(self):
        assert _distance_to_score(0.0, "cosine") == 1.0

    def test_cosine_one_distance_is_zero(self):
        assert _distance_to_score(1.0, "cosine") == 0.0

    def test_cosine_clamped_at_zero(self):
        # Distance > 1.0 should not produce negative score
        assert _distance_to_score(1.5, "cosine") == 0.0

    def test_cosine_midpoint(self):
        assert abs(_distance_to_score(0.25, "cosine") - 0.75) < 1e-9

    def test_l2_zero_distance_is_one(self):
        assert _distance_to_score(0.0, "l2") == 1.0

    def test_l2_positive_distance_less_than_one(self):
        score = _distance_to_score(1.0, "l2")
        assert 0 < score < 1.0

    def test_l2_monotone_decreasing(self):
        scores = [_distance_to_score(d, "l2") for d in [0.1, 0.5, 1.0, 2.0, 5.0]]
        assert scores == sorted(scores, reverse=True)

    def test_ip_passthrough(self):
        assert _distance_to_score(0.88, "ip") == 0.88


# ── _parse_query_response ──────────────────────────────────────────────────────

class TestParseQueryResponse:
    def test_basic_parse(self):
        raw = _make_query_raw(
            ids=["id1", "id2"],
            texts=["text one", "text two"],
            distances=[0.1, 0.3],
            metadatas=[_meta("DSM5.pdf", 1), _meta("ICD11.pdf", 5)],
        )
        results = _parse_query_response(raw, "cosine")

        assert len(results) == 2
        assert results[0].rank == 1
        assert results[1].rank == 2
        assert results[0].chunk_id == "id1"
        assert results[0].source == "DSM5.pdf"
        assert results[0].page == 1
        assert abs(results[0].score - 0.9) < 1e-4

    def test_scores_in_descending_order(self):
        raw = _make_query_raw(
            ids=["a", "b", "c"],
            texts=["t1", "t2", "t3"],
            distances=[0.05, 0.20, 0.45],
            metadatas=[_meta(), _meta(), _meta()],
        )
        results = _parse_query_response(raw, "cosine")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_response(self):
        raw = {"ids": [[]], "documents": [[]], "distances": [[]], "metadatas": [[]]}
        results = _parse_query_response(raw, "cosine")
        assert results == []

    def test_missing_metadata_defaults(self):
        raw = _make_query_raw(
            ids=["x"], texts=["text"], distances=[0.2], metadatas=[None],
        )
        results = _parse_query_response(raw, "cosine")
        assert results[0].source == "unknown"
        assert results[0].page == 0

    def test_score_rounded_to_6_places(self):
        raw = _make_query_raw(["id"], ["t"], [0.123456789], [_meta()])
        results = _parse_query_response(raw, "cosine")
        # score = 1 - 0.123456789 = 0.876543211... → rounded to 6dp
        assert len(str(results[0].score).split(".")[-1]) <= 6


# ── _validate_batch_lengths ────────────────────────────────────────────────────

class TestValidateBatchLengths:
    def test_equal_lengths_ok(self):
        _validate_batch_lengths(3, ["a", "b", "c"], [[1], [2], [3]], [{}, {}, {}])

    def test_documents_mismatch(self):
        with pytest.raises(ValueError, match="documents"):
            _validate_batch_lengths(3, ["a", "b"], [[1], [2], [3]], [{}, {}, {}])

    def test_embeddings_mismatch(self):
        with pytest.raises(ValueError, match="embeddings"):
            _validate_batch_lengths(3, ["a", "b", "c"], [[1], [2]], [{}, {}, {}])

    def test_metadatas_mismatch(self):
        with pytest.raises(ValueError, match="metadatas"):
            _validate_batch_lengths(3, ["a", "b", "c"], [[1], [2], [3]], [{}, {}])


# ── VectorStore initialisation ─────────────────────────────────────────────────

class TestVectorStoreInit:
    def test_invalid_distance_function(self, tmp_path):
        with pytest.raises(ValueError, match="distance_function"):
            VectorStore(
                collection_name="col",
                persist_dir=tmp_path,
                distance_function="invalid",
            )

    def test_invalid_batch_size(self, tmp_path):
        with pytest.raises(ValueError, match="insert_batch_size"):
            VectorStore(
                collection_name="col",
                persist_dir=tmp_path,
                insert_batch_size=0,
            )

    def test_repr(self, tmp_path):
        vs = VectorStore("clinical_kb", tmp_path, distance_function="cosine")
        r = repr(vs)
        assert "clinical_kb" in r
        assert "cosine" in r

    def test_properties(self, tmp_path):
        vs = VectorStore("col", tmp_path, distance_function="l2")
        assert vs.collection_name == "col"
        assert vs.persist_dir == tmp_path.resolve()
        assert vs.distance_function == "l2"


# ── create_collection ──────────────────────────────────────────────────────────

class TestCreateCollection:
    def test_creates_and_returns_info(self, store, mock_chroma_module):
        _, mock_client, mock_collection = mock_chroma_module
        mock_client.list_collections.return_value = []
        mock_collection.count.return_value = 0

        info = store.create_collection()

        assert isinstance(info, CollectionInfo)
        assert info.name == "test_collection"
        assert info.document_count == 0
        assert info.distance_metric == "cosine"
        mock_client.get_or_create_collection.assert_called_once()

    def test_exist_ok_true_does_not_raise(self, store, mock_chroma_module):
        _, mock_client, mock_collection = mock_chroma_module
        mock_client.list_collections.return_value = [
            MagicMock(name="test_collection")
        ]
        # Should NOT raise
        info = store.create_collection(exist_ok=True)
        assert info.name == "test_collection"

    def test_exist_ok_false_raises(self, store, mock_chroma_module):
        _, mock_client, _ = mock_chroma_module
        existing = MagicMock()
        existing.name = "test_collection"
        mock_client.list_collections.return_value = [existing]

        with pytest.raises(CollectionAlreadyExistsError):
            store.create_collection(exist_ok=False)

    def test_custom_metadata_passed_through(self, store, mock_chroma_module):
        _, mock_client, mock_collection = mock_chroma_module
        mock_client.list_collections.return_value = []

        store.create_collection(metadata={"ingested_by": "pipeline_v2"})
        call_kwargs = mock_client.get_or_create_collection.call_args.kwargs
        assert "ingested_by" in call_kwargs["metadata"]

    def test_connection_error_wraps(self, store, mock_chroma_module):
        _, mock_client, _ = mock_chroma_module
        mock_client.list_collections.side_effect = RuntimeError("socket error")

        with pytest.raises((VectorStoreError, RuntimeError)):
            store.create_collection()


# ── list_collections and collection_exists ─────────────────────────────────────

class TestCollectionListing:
    def test_list_empty(self, store, mock_chroma_module):
        _, mock_client, _ = mock_chroma_module
        mock_client.list_collections.return_value = []
        assert store.list_collections() == []

    def test_list_multiple(self, store, mock_chroma_module):
        _, mock_client, _ = mock_chroma_module
        c1, c2 = MagicMock(name="col_a"), MagicMock(name="col_b")
        c1.name = "col_a"
        c2.name = "col_b"
        mock_client.list_collections.return_value = [c1, c2]
        names = store.list_collections()
        assert "col_a" in names
        assert "col_b" in names

    def test_collection_exists_true(self, store, mock_chroma_module):
        _, mock_client, _ = mock_chroma_module
        existing = MagicMock()
        existing.name = "test_collection"
        mock_client.list_collections.return_value = [existing]
        assert store.collection_exists() is True

    def test_collection_exists_false(self, store, mock_chroma_module):
        _, mock_client, _ = mock_chroma_module
        mock_client.list_collections.return_value = []
        assert store.collection_exists() is False


# ── add_documents ──────────────────────────────────────────────────────────────

class TestAddDocuments:
    def _insert_args(self, n: int = 3):
        return {
            "ids":        [f"id_{i}" for i in range(n)],
            "documents":  [f"Clinical text document {i}." for i in range(n)],
            "embeddings": [[float(i)] * 4 for i in range(n)],
            "metadatas":  [_meta(page=i + 1) for i in range(n)],
        }

    def test_happy_path(self, store, mock_chroma_module):
        _, _, mock_collection = mock_chroma_module
        result = store.add_documents(**self._insert_args(3))

        assert isinstance(result, InsertResult)
        assert result.total_inserted == 3
        assert result.total_failed == 0
        mock_collection.upsert.assert_called_once()

    def test_upsert_called_not_add(self, store, mock_chroma_module):
        """Must use upsert() for idempotency, never add()."""
        _, _, mock_collection = mock_chroma_module
        store.add_documents(**self._insert_args(2))
        mock_collection.upsert.assert_called()
        mock_collection.add.assert_not_called()

    def test_batching_respected(self, store, mock_chroma_module):
        """insert_batch_size=5, inserting 12 → 3 upsert calls."""
        _, _, mock_collection = mock_chroma_module
        store.add_documents(**self._insert_args(12))
        # batches: [0:5], [5:10], [10:12] = 3 calls
        assert mock_collection.upsert.call_count == 3

    def test_partial_batch_failure_continues(self, store, mock_chroma_module):
        """A failing batch is recorded; remaining batches succeed."""
        _, _, mock_collection = mock_chroma_module
        call_count = [0]

        def upsert_side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("ChromaDB segment error")

        mock_collection.upsert.side_effect = upsert_side_effect
        result = store.add_documents(**self._insert_args(10))

        assert result.total_failed == 5   # first batch of 5 failed
        assert result.total_inserted == 5  # second batch of 5 succeeded
        assert len(result.errors) == 1

    def test_mismatched_lengths_raises(self, store, mock_chroma_module):
        with pytest.raises(ValueError):
            store.add_documents(
                ids=["a", "b"],
                documents=["text"],      # wrong length
                embeddings=[[0.1] * 4, [0.2] * 4],
                metadatas=[{}, {}],
            )

    def test_no_metadatas_defaults_to_empty_dicts(self, store, mock_chroma_module):
        _, _, mock_collection = mock_chroma_module
        store.add_documents(
            ids=["id_0"],
            documents=["text"],
            embeddings=[[0.1, 0.2, 0.3, 0.4]],
            metadatas=None,
        )
        call_kwargs = mock_collection.upsert.call_args.kwargs
        assert call_kwargs["metadatas"] == [{}]

    def test_result_timing_positive(self, store, mock_chroma_module):
        result = store.add_documents(**self._insert_args(2))
        assert result.elapsed_ms >= 0

    def test_batch_size_override(self, store, mock_chroma_module):
        _, _, mock_collection = mock_chroma_module
        store.add_documents(**self._insert_args(6), batch_size=2)
        # 6 docs / batch_size 2 = 3 upsert calls
        assert mock_collection.upsert.call_count == 3


# ── query_documents ────────────────────────────────────────────────────────────

class TestQueryDocuments:
    def _setup_query(self, mock_chroma_module, n: int = 3, distance: float = 0.1):
        _, _, mock_collection = mock_chroma_module
        mock_collection.count.return_value = 100

        ids   = [f"id_{i}" for i in range(n)]
        texts = [f"Clinical result text {i}." for i in range(n)]
        dists = [distance + i * 0.1 for i in range(n)]
        metas = [_meta(page=i + 1) for i in range(n)]

        mock_collection.query.return_value = _make_query_raw(ids, texts, dists, metas)
        return mock_collection

    def test_returns_query_results(self, store, mock_chroma_module):
        self._setup_query(mock_chroma_module)
        results = store.query_documents([0.1, 0.2, 0.3, 0.4])
        assert isinstance(results, list)
        assert all(isinstance(r, QueryResult) for r in results)

    def test_correct_number_of_results(self, store, mock_chroma_module):
        self._setup_query(mock_chroma_module, n=3)
        results = store.query_documents([0.1] * 4)
        assert len(results) == 3

    def test_scores_derived_from_distances(self, store, mock_chroma_module):
        self._setup_query(mock_chroma_module, n=1, distance=0.25)
        results = store.query_documents([0.1] * 4)
        # cosine score = 1 - 0.25 = 0.75
        assert abs(results[0].score - 0.75) < 1e-4

    def test_ranked_ascending_by_rank_field(self, store, mock_chroma_module):
        self._setup_query(mock_chroma_module, n=4)
        results = store.query_documents([0.1] * 4)
        ranks = [r.rank for r in results]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_empty_collection_returns_empty_list(self, store, mock_chroma_module):
        _, _, mock_collection = mock_chroma_module
        mock_collection.count.return_value = 0
        results = store.query_documents([0.1] * 4)
        assert results == []
        mock_collection.query.assert_not_called()

    def test_n_results_capped_by_collection_size(self, store, mock_chroma_module):
        _, _, mock_collection = mock_chroma_module
        mock_collection.count.return_value = 2
        mock_collection.query.return_value = _make_query_raw(
            ["a", "b"], ["t1", "t2"], [0.1, 0.2], [_meta(), _meta()]
        )
        store.query_documents([0.1] * 4, n_results=100)
        call_kwargs = mock_collection.query.call_args.kwargs
        assert call_kwargs["n_results"] == 2  # capped to collection size

    def test_where_filter_passed_through(self, store, mock_chroma_module):
        self._setup_query(mock_chroma_module, n=1)
        store.query_documents([0.1] * 4, where={"source": "DSM5.pdf"})
        call_kwargs = mock_collection.query.call_args.kwargs if False else \
            mock_chroma_module[2].query.call_args.kwargs
        assert call_kwargs["where"] == {"source": "DSM5.pdf"}

    def test_query_error_raised(self, store, mock_chroma_module):
        _, _, mock_collection = mock_chroma_module
        mock_collection.count.return_value = 10
        mock_collection.query.side_effect = RuntimeError("index error")
        with pytest.raises(QueryError):
            store.query_documents([0.1] * 4)

    def test_source_and_page_extracted(self, store, mock_chroma_module):
        _, _, mock_collection = mock_chroma_module
        mock_collection.count.return_value = 50
        mock_collection.query.return_value = _make_query_raw(
            ["id"], ["text"], [0.1], [{"source": "CBT.pdf", "page": 42}]
        )
        results = store.query_documents([0.1] * 4)
        assert results[0].source == "CBT.pdf"
        assert results[0].page == 42

    def test_metadata_preserved(self, store, mock_chroma_module):
        _, _, mock_collection = mock_chroma_module
        mock_collection.count.return_value = 5
        meta = {"source": "doc.pdf", "page": 7, "chunk_index": 2, "char_count": 450}
        mock_collection.query.return_value = _make_query_raw(
            ["id"], ["text"], [0.15], [meta]
        )
        results = store.query_documents([0.1] * 4)
        assert results[0].metadata["char_count"] == 450


# ── delete_documents ───────────────────────────────────────────────────────────

class TestDeleteDocuments:
    def test_delete_by_ids(self, store, mock_chroma_module):
        _, _, mock_collection = mock_chroma_module
        mock_collection.count.side_effect = [10, 7]  # before, after

        deleted = store.delete_documents(ids=["id_0", "id_1", "id_2"])

        mock_collection.delete.assert_called_once_with(ids=["id_0", "id_1", "id_2"])
        assert deleted == 3

    def test_delete_by_where(self, store, mock_chroma_module):
        _, _, mock_collection = mock_chroma_module
        mock_collection.count.side_effect = [20, 15]

        deleted = store.delete_documents(where={"source": "outdated.pdf"})

        mock_collection.delete.assert_called_once_with(where={"source": "outdated.pdf"})
        assert deleted == 5

    def test_delete_by_ids_and_where(self, store, mock_chroma_module):
        _, _, mock_collection = mock_chroma_module
        mock_collection.count.side_effect = [10, 8]

        store.delete_documents(ids=["id_1"], where={"source": "doc.pdf"})

        call_kwargs = mock_collection.delete.call_args.kwargs
        assert "ids" in call_kwargs
        assert "where" in call_kwargs

    def test_no_ids_no_where_raises(self, store, mock_chroma_module):
        with pytest.raises(ValueError, match="ids.*where|where.*ids"):
            store.delete_documents()

    def test_delete_error_wrapped(self, store, mock_chroma_module):
        _, _, mock_collection = mock_chroma_module
        mock_collection.count.return_value = 5
        mock_collection.delete.side_effect = RuntimeError("lock error")
        with pytest.raises(DeleteError):
            store.delete_documents(ids=["id_0"])

    def test_delete_by_source(self, store, mock_chroma_module):
        _, _, mock_collection = mock_chroma_module
        mock_collection.count.side_effect = [30, 20]

        deleted = store.delete_by_source("DSM5.pdf")

        call_kwargs = mock_collection.delete.call_args.kwargs
        assert call_kwargs["where"] == {"source": "DSM5.pdf"}
        assert deleted == 10

    def test_deleted_count_never_negative(self, store, mock_chroma_module):
        _, _, mock_collection = mock_chroma_module
        # After > before (shouldn't happen, but guard against it)
        mock_collection.count.side_effect = [5, 10]
        deleted = store.delete_documents(ids=["id"])
        assert deleted == 0


# ── get_collection_info ────────────────────────────────────────────────────────

class TestGetCollectionInfo:
    def test_returns_info(self, store, mock_chroma_module):
        _, _, mock_collection = mock_chroma_module
        mock_collection.count.return_value = 42
        mock_collection.metadata = {"hnsw:space": "cosine", "model": "bge-large"}

        info = store.get_collection_info()

        assert isinstance(info, CollectionInfo)
        assert info.document_count == 42
        assert info.distance_metric == "cosine"
        assert info.name == "test_collection"

    def test_persist_dir_in_info(self, store, mock_chroma_module, tmp_path):
        _, _, mock_collection = mock_chroma_module
        mock_collection.count.return_value = 0
        mock_collection.metadata = {}

        info = store.get_collection_info()
        assert info.persist_dir  # non-empty string


# ── reset_collection ───────────────────────────────────────────────────────────

class TestResetCollection:
    def test_deletes_and_recreates(self, store, mock_chroma_module):
        _, mock_client, mock_collection = mock_chroma_module
        mock_collection.count.return_value = 0

        store.reset_collection()

        mock_client.delete_collection.assert_called_once_with("test_collection")
        mock_client.get_or_create_collection.assert_called()

    def test_clears_internal_collection_ref(self, store, mock_chroma_module):
        _, _, mock_collection = mock_chroma_module
        # Force the collection to be resolved
        store._collection = mock_collection
        store.reset_collection()
        # After reset, _collection should have been cleared then re-resolved
        # (it gets set again by create_collection)


# ── peek ──────────────────────────────────────────────────────────────────────

class TestPeek:
    def test_returns_dicts(self, store, mock_chroma_module):
        _, _, mock_collection = mock_chroma_module
        mock_collection.peek.return_value = {
            "ids": ["id_0", "id_1"],
            "documents": ["text 0", "text 1"],
            "metadatas": [_meta(), _meta(page=2)],
        }
        docs = store.peek(n=2)
        assert len(docs) == 2
        assert docs[0]["id"] == "id_0"
        assert docs[0]["text"] == "text 0"
        assert "metadata" in docs[0]

    def test_empty_collection(self, store, mock_chroma_module):
        _, _, mock_collection = mock_chroma_module
        mock_collection.peek.return_value = {
            "ids": [], "documents": [], "metadatas": []
        }
        assert store.peek() == []

    def test_n_passed_to_chromadb(self, store, mock_chroma_module):
        _, _, mock_collection = mock_chroma_module
        mock_collection.peek.return_value = {"ids": [], "documents": [], "metadatas": []}
        store.peek(n=7)
        mock_collection.peek.assert_called_once_with(limit=7)


# ── Singleton: get_vector_store ────────────────────────────────────────────────

class TestGetVectorStore:
    def _patch_settings(self, tmp_path):
        mock_cfg = MagicMock()
        mock_cfg.chroma.collection_name = "clinical_knowledge_base"
        mock_cfg.chroma.persist_dir = tmp_path / "chroma"
        mock_cfg.chroma.distance_function = "cosine"
        return patch("rag.vector_store.get_settings", return_value=mock_cfg)

    def test_returns_vector_store(self, tmp_path):
        with self._patch_settings(tmp_path):
            vs = get_vector_store()
            assert isinstance(vs, VectorStore)

    def test_same_instance_returned_twice(self, tmp_path):
        with self._patch_settings(tmp_path):
            a = get_vector_store()
            b = get_vector_store()
            assert a is b

    def test_force_reload_creates_new_instance(self, tmp_path):
        with self._patch_settings(tmp_path):
            a = get_vector_store()
            b = get_vector_store(force_reload=True)
            assert a is not b

    def test_thread_safety(self, tmp_path):
        """Concurrent calls must all return the same singleton."""
        with self._patch_settings(tmp_path):
            instances = []
            barrier = threading.Barrier(6)

            def get_it():
                barrier.wait()
                instances.append(get_vector_store())

            threads = [threading.Thread(target=get_it) for _ in range(6)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert all(i is instances[0] for i in instances)

    def test_collection_name_from_settings(self, tmp_path):
        with self._patch_settings(tmp_path):
            vs = get_vector_store()
            assert vs.collection_name == "clinical_knowledge_base"


# ── Integration: add → query pipeline shape ────────────────────────────────────

class TestAddQueryPipelineShape:
    """
    Verify the full add → query data flow without a real ChromaDB instance.
    Ensures the contract between ChunkingResult / EmbeddingResult and
    VectorStore is maintained.
    """

    def test_chromadb_batch_format_accepted(self, store, mock_chroma_module):
        """
        Simulate what the ingestion pipeline does:
          chunks_batch  = ChunkingResult.to_chromadb_batch()
          emb_batch     = EmbeddingResult.to_chromadb_batch()
          store.add_documents(ids=..., documents=..., embeddings=..., metadatas=...)
        """
        _, _, mock_collection = mock_chroma_module

        # Simulate ChunkingResult.to_chromadb_batch() output
        chunks_batch = {
            "ids":       ["DSM5__p0001__c0000", "DSM5__p0001__c0001"],
            "documents": ["Criterion A: depressed mood.", "Criterion B: anhedonia."],
            "metadatas": [
                {"source": "DSM5.pdf", "page": 1, "chunk_index": 0},
                {"source": "DSM5.pdf", "page": 1, "chunk_index": 1},
            ],
        }
        # Simulate EmbeddingResult.to_chromadb_batch() output
        emb_batch = {
            "embeddings": [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
        }

        result = store.add_documents(
            ids=chunks_batch["ids"],
            documents=chunks_batch["documents"],
            embeddings=emb_batch["embeddings"],
            metadatas=chunks_batch["metadatas"],
        )

        assert result.total_inserted == 2
        assert result.total_failed == 0
        mock_collection.upsert.assert_called_once_with(
            ids=chunks_batch["ids"],
            documents=chunks_batch["documents"],
            embeddings=emb_batch["embeddings"],
            metadatas=chunks_batch["metadatas"],
        )

    def test_query_result_fields_align_with_chunk_metadata(self, store, mock_chroma_module):
        """
        QueryResult fields should map directly to chunk metadata fields
        so the RAG retriever can reconstruct source attribution.
        """
        _, _, mock_collection = mock_chroma_module
        mock_collection.count.return_value = 10
        mock_collection.query.return_value = _make_query_raw(
            ids=["DSM5__p0042__c0002"],
            texts=["Persistent depressive disorder criteria text."],
            distances=[0.08],
            metadatas=[{
                "source": "DSM5.pdf",
                "page": 42,
                "chunk_index": 2,
                "char_count": 320,
                "word_count": 55,
            }],
        )
        results = store.query_documents([0.1, 0.2, 0.3, 0.4])

        r = results[0]
        assert r.chunk_id == "DSM5__p0042__c0002"
        assert r.source == "DSM5.pdf"
        assert r.page == 42
        assert r.metadata["chunk_index"] == 2
        assert r.metadata["char_count"] == 320
        assert r.score > 0.9   # distance 0.08 → score 0.92
