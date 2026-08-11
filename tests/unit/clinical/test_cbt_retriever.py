"""
tests/unit/clinical/test_cbt_retriever.py — Tests for the CBT evidence retriever.

Covers models, filter builders, conversion helpers, and integration
with the CBTRetriever class (with mocked retrieval layer).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from clinical.retrieval.cbt_retriever import (
    CBTRetriever,
    _build_bm25_filter_from_filter,
    _build_relevance,
    _build_synthesis,
    _build_where_from_filter,
    _compute_confidence,
    _dicts_to_items,
    _items_to_dicts,
    _to_cbt_items,
    get_cbt_retriever,
)
from clinical.retrieval.models import (
    CBTEvidenceFilter,
    CBTEvidenceItem,
    CBTEvidenceResult,
)


# ═══════════════════════════════════════════════════════════════════════
# Mock QueryResult (simulates rag.vector_store.QueryResult)
# ═══════════════════════════════════════════════════════════════════════


class _MockQueryResult:
    """Minimal mock of rag.vector_store.QueryResult for testing."""

    def __init__(
        self,
        text: str,
        source: str = "cbt_manual.pdf",
        page: int = 10,
        score: float = 0.85,
        chunk_id: str = "chunk_001",
        metadata: dict | None = None,
    ) -> None:
        self.text = text
        self.source = source
        self.page = page
        self.score = score
        self.chunk_id = chunk_id
        self.metadata = metadata or {"framework": "cbt"}


# ═══════════════════════════════════════════════════════════════════════
# Tests: CBTEvidenceFilter
# ═══════════════════════════════════════════════════════════════════════


class TestCBTEvidenceFilter:
    def test_default_framework(self):
        f = CBTEvidenceFilter()
        assert f.framework == "cbt"

    def test_with_all_fields(self):
        f = CBTEvidenceFilter(
            framework="cbt",
            chapter="Chapter 3",
            topic="depression",
            intervention="cognitive restructuring",
            symptom="low mood",
            population="adults",
        )
        assert f.chapter == "Chapter 3"
        assert f.topic == "depression"
        assert f.intervention == "cognitive restructuring"
        assert f.symptom == "low mood"
        assert f.population == "adults"

    def test_with_list_values(self):
        f = CBTEvidenceFilter(
            topic=["depression", "anxiety"],
            intervention=["CBT", "ACT"],
        )
        assert isinstance(f.topic, list)
        assert len(f.topic) == 2

    def test_to_metadata_filter_default(self):
        f = CBTEvidenceFilter()
        result = f.to_metadata_filter()
        assert result == {"framework": "cbt"}

    def test_to_metadata_filter_with_fields(self):
        f = CBTEvidenceFilter(
            framework="cbt",
            chapter="Ch 2",
            topic="anxiety",
        )
        result = f.to_metadata_filter()
        assert result == {"framework": "cbt", "chapter": "Ch 2", "topic": "anxiety"}

    def test_to_metadata_filter_with_list(self):
        f = CBTEvidenceFilter(topic=["depression", "anxiety"])
        result = f.to_metadata_filter()
        assert result["framework"] == "cbt"
        assert result["topic"] == {"$in": ["depression", "anxiety"]}


# ═══════════════════════════════════════════════════════════════════════
# Tests: CBTEvidenceItem
# ═══════════════════════════════════════════════════════════════════════


class TestCBTEvidenceItem:
    def test_minimal_item(self):
        item = CBTEvidenceItem(
            source="cbt_manual.pdf",
            content="Cognitive restructuring involves identifying and challenging negative automatic thoughts.",
            relevance="Describes core CBT technique for thought modification",
        )
        assert item.framework == "cbt"
        assert item.quality_rating is None
        assert item.chapter is None

    def test_full_item(self):
        item = CBTEvidenceItem(
            source="beck_2011.pdf",
            content="Behavioural activation is an effective intervention for depression.",
            relevance="Supports behavioural approach for low mood",
            quality_rating=0.92,
            framework="cbt",
            chapter="Ch 4: Behavioural Interventions",
            topic="depression",
            intervention="behavioural activation",
            symptom="low mood",
            population="adults",
            page=42,
            chunk_id="beck_2011__p0042__c0001",
        )
        assert item.quality_rating == 0.92
        assert item.chapter == "Ch 4: Behavioural Interventions"
        assert item.page == 42

    def test_content_min_length_enforced(self):
        with pytest.raises(Exception):
            CBTEvidenceItem(
                source="test.pdf",
                content="short",
                relevance="too short relevance",
            )

    def test_relevance_min_length_enforced(self):
        with pytest.raises(Exception):
            CBTEvidenceItem(
                source="test.pdf",
                content="This is a sufficiently long content string for testing.",
                relevance="short",
            )

    def test_quality_rating_clamped(self):
        with pytest.raises(Exception):
            CBTEvidenceItem(
                source="test.pdf",
                content="This is a sufficiently long content string for testing purposes.",
                relevance="Sufficiently long relevance description for testing.",
                quality_rating=1.5,
            )


# ═══════════════════════════════════════════════════════════════════════
# Tests: CBTEvidenceResult
# ═══════════════════════════════════════════════════════════════════════


class TestCBTEvidenceResult:
    def test_minimal_result(self):
        result = CBTEvidenceResult(
            query="What is CBT for depression?",
            retrieved=[],
            synthesis="No evidence found.",
            key_findings=[],
            evidence_gaps=[],
        )
        assert result.confidence == 0.5
        assert result.total_candidates == 0

    def test_with_items(self):
        items = [
            CBTEvidenceItem(
                source="test.pdf",
                content="This is a sufficiently long content for testing purposes here.",
                relevance="Relevance description that is sufficiently long for testing.",
                quality_rating=0.9,
            )
        ]
        result = CBTEvidenceResult(
            query="test query",
            retrieved=items,
            synthesis="Found 1 item.",
            key_findings=["Finding 1"],
            evidence_gaps=["Gap 1"],
            confidence=0.8,
            total_candidates=10,
            elapsed_ms=42.0,
        )
        assert len(result.retrieved) == 1
        assert result.confidence == 0.8
        assert result.total_candidates == 10

    def test_to_framework_evidence_result(self):
        item = CBTEvidenceItem(
            source="test.pdf",
            content="This is a sufficiently long content string to pass validation.",
            relevance="This is a sufficiently long relevance description for testing.",
            quality_rating=0.85,
        )
        result = CBTEvidenceResult(
            query="test",
            retrieved=[item],
            synthesis="Synthesis that is sufficiently long for validation purposes.",
            key_findings=["Finding"],
            evidence_gaps=["Gap"],
            confidence=0.7,
        )
        fw_result = result.to_framework_evidence_result()
        assert fw_result.synthesis == result.synthesis
        assert len(fw_result.retrieved) == 1
        assert fw_result.retrieved[0].source == "test.pdf"
        assert fw_result.confidence == 0.7


# ═══════════════════════════════════════════════════════════════════════
# Tests: Filter builders
# ═══════════════════════════════════════════════════════════════════════


class TestBuildWhereFromFilter:
    def test_default_filter(self):
        f = CBTEvidenceFilter()
        result = _build_where_from_filter(f)
        assert result == {"framework": {"$eq": "cbt"}}

    def test_single_extra_field(self):
        f = CBTEvidenceFilter(topic="depression")
        result = _build_where_from_filter(f)
        assert result == {"$and": [
            {"framework": {"$eq": "cbt"}},
            {"topic": {"$eq": "depression"}},
        ]}

    def test_multiple_extra_fields(self):
        f = CBTEvidenceFilter(chapter="Ch 3", topic="anxiety", symptom="worry")
        result = _build_where_from_filter(f)
        assert result["$and"][0] == {"framework": {"$eq": "cbt"}}
        assert {"chapter": {"$eq": "Ch 3"}} in result["$and"]
        assert {"topic": {"$eq": "anxiety"}} in result["$and"]
        assert {"symptom": {"$eq": "worry"}} in result["$and"]

    def test_list_values(self):
        f = CBTEvidenceFilter(topic=["depression", "anxiety"])
        result = _build_where_from_filter(f)
        assert {"topic": {"$in": ["depression", "anxiety"]}} in result["$and"]

    def test_all_fields(self):
        f = CBTEvidenceFilter(
            framework="cbt",
            chapter="Ch 1",
            topic="depression",
            intervention="CBT",
            symptom="low mood",
            population="adults",
        )
        result = _build_where_from_filter(f)
        assert len(result["$and"]) == 6


class TestBuildBM25FilterFromFilter:
    def test_default(self):
        f = CBTEvidenceFilter()
        result = _build_bm25_filter_from_filter(f)
        assert result == {"framework": "cbt"}

    def test_with_extra_fields(self):
        f = CBTEvidenceFilter(chapter="Ch 2", topic="anxiety")
        result = _build_bm25_filter_from_filter(f)
        assert result["framework"] == "cbt"
        assert result["chapter"] == "Ch 2"
        assert result["topic"] == "anxiety"

    def test_list_values(self):
        f = CBTEvidenceFilter(topic=["depression", "anxiety"])
        result = _build_bm25_filter_from_filter(f)
        assert result["topic"] == {"$in": ["depression", "anxiety"]}


# ═══════════════════════════════════════════════════════════════════════
# Tests: Helper functions
# ═══════════════════════════════════════════════════════════════════════


class TestBuildRelevance:
    def test_high_relevance(self):
        assert "Highly relevant" in _build_relevance(0.95)

    def test_strong_relevance(self):
        assert "Strongly relevant" in _build_relevance(0.80)

    def test_moderate_relevance(self):
        assert "Moderately relevant" in _build_relevance(0.65)

    def test_somewhat_relevant(self):
        assert "Somewhat relevant" in _build_relevance(0.50)

    def test_tangential(self):
        assert "Tangentially" in _build_relevance(0.30)

    def test_edge_low(self):
        assert "Tangentially" in _build_relevance(0.0)


class TestBuildSynthesis:
    def test_empty_items(self):
        result = _build_synthesis([], "test query")
        assert "No evidence" in result

    def test_with_items(self):
        items = [
            CBTEvidenceItem(
                source="a.pdf",
                content="This is a sufficiently long content string for testing purposes.",
                relevance="Relevance description that is sufficiently long for testing.",
                topic="depression",
            ),
            CBTEvidenceItem(
                source="b.pdf",
                content="Another sufficiently long content string for testing purposes here.",
                relevance="Another relevance description that is sufficiently long here.",
                topic="anxiety",
            ),
        ]
        result = _build_synthesis(items, "CBT for mood")
        assert "2 evidence item(s)" in result
        assert "2 source(s)" in result
        assert "depression" in result

    def test_without_topics(self):
        items = [
            CBTEvidenceItem(
                source="a.pdf",
                content="This is a sufficiently long content string for testing purposes.",
                relevance="Relevance description that is sufficiently long for testing.",
            ),
        ]
        result = _build_synthesis(items, "query")
        assert "1 evidence item(s)" in result


class TestComputeConfidence:
    def test_empty_items(self):
        assert _compute_confidence([]) == 0.0

    def test_single_item(self):
        items = [
            CBTEvidenceItem(
                source="a.pdf",
                content="This is a sufficiently long content string for testing purposes.",
                relevance="Relevance description that is sufficiently long for testing.",
                quality_rating=0.8,
            ),
        ]
        assert _compute_confidence(items) == 0.8

    def test_multiple_items(self):
        items = [
            CBTEvidenceItem(
                source=f"{i}.pdf",
                content=f"This is a sufficiently long content string for testing item {i}.",
                relevance=f"Relevance description that is sufficiently long for item {i}.",
                quality_rating=score,
            )
            for i, score in enumerate([0.9, 0.8, 0.7, 0.6, 0.5])
        ]
        avg = (0.9 + 0.8 + 0.7 + 0.6 + 0.5) / 5
        assert _compute_confidence(items) == round(avg, 2)

    def test_truncated_to_top_5(self):
        items = [
            CBTEvidenceItem(
                source=f"{i}.pdf",
                content=f"This is a sufficiently long content string for testing item {i}.",
                relevance=f"Relevance description that is sufficiently long for item {i}.",
                quality_rating=score,
            )
            for i, score in enumerate([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
        ]
        # Only first 5 items considered (all 1.0)
        assert _compute_confidence(items) == 1.0


# ═══════════════════════════════════════════════════════════════════════
# Tests: Conversion helpers
# ═══════════════════════════════════════════════════════════════════════


class TestToCbtItems:
    def test_empty_results(self):
        result = _to_cbt_items([], threshold=0.3)
        assert result == []

    def test_below_threshold_filtered(self):
        raw = [
            _MockQueryResult(text="Low score item", score=0.2),
            _MockQueryResult(text="High score item", score=0.9),
        ]
        items = _to_cbt_items(raw, threshold=0.5)
        assert len(items) == 1
        assert items[0].quality_rating == 0.9

    def test_metadata_preserved(self):
        raw = [
            _MockQueryResult(
                text="Content with metadata for testing purposes here.",
                score=0.85,
                metadata={
                    "framework": "cbt",
                    "chapter": "Ch 3",
                    "topic": "depression",
                    "intervention": "CBT",
                    "symptom": "low mood",
                    "population": "adults",
                },
            ),
        ]
        items = _to_cbt_items(raw, threshold=0.0)
        assert items[0].chapter == "Ch 3"
        assert items[0].topic == "depression"
        assert items[0].intervention == "CBT"


class TestItemsToDicts:
    def test_empty(self):
        assert _items_to_dicts([]) == []

    def test_conversion(self):
        items = [
            CBTEvidenceItem(
                source="test.pdf",
                content="This is a sufficiently long content string for testing purposes here.",
                relevance="Relevance description that is sufficiently long for testing.",
                quality_rating=0.85,
                chunk_id="chunk_001",
                page=10,
                framework="cbt",
                topic="depression",
            ),
        ]
        dicts = _items_to_dicts(items)
        assert len(dicts) == 1
        assert dicts[0]["chunk_id"] == "chunk_001"
        assert dicts[0]["score"] == 0.85
        assert dicts[0]["metadata"]["framework"] == "cbt"


class TestDictsToItems:
    def test_empty(self):
        assert _dicts_to_items([], framework="cbt") == []

    def test_conversion(self):
        dicts = [
            {
                "chunk_id": "chunk_001",
                "text": "This is a sufficiently long content string for testing.",
                "source": "test.pdf",
                "page": 10,
                "score": 0.85,
                "rank": 1,
                "metadata": {
                    "framework": "cbt",
                    "topic": "depression",
                },
            },
        ]
        items = _dicts_to_items(dicts, framework="cbt")
        assert len(items) == 1
        assert items[0].source == "test.pdf"
        assert items[0].quality_rating == 0.85
        assert items[0].topic == "depression"

    def test_framework_fallback(self):
        dicts = [
            {
                "chunk_id": "chunk_001",
                "text": "This is a sufficiently long content string for testing.",
                "source": "test.pdf",
                "page": 10,
                "score": 0.85,
                "rank": 1,
                "metadata": {},
            },
        ]
        items = _dicts_to_items(dicts, framework="act")
        assert items[0].framework == "act"


# ═══════════════════════════════════════════════════════════════════════
# Tests: CBTRetriever (with mocked dependencies)
# ═══════════════════════════════════════════════════════════════════════


class TestCBTRetrieverInit:
    def test_default_config(self):
        r = CBTRetriever()
        assert r.framework == "cbt"
        assert r.config["n_results"] == 10
        assert r.config["similarity_threshold"] == 0.35
        assert r.config["enable_rerank"] is True

    def test_custom_config(self):
        r = CBTRetriever(
            framework="act",
            n_results=5,
            similarity_threshold=0.5,
            enable_rerank=False,
        )
        assert r.framework == "act"
        assert r.config["n_results"] == 5
        assert r.config["similarity_threshold"] == 0.5
        assert r.config["enable_rerank"] is False


class TestCBTRetrieverRetrieve:
    def test_retrieve_with_empty_query_returns_synthesis(self):
        r = CBTRetriever(
            n_results=5,
            similarity_threshold=0.0,
            enable_rerank=False,
        )
        result = r.retrieve("")
        assert isinstance(result, CBTEvidenceResult)
        assert len(result.retrieved) >= 0  # may be empty but not crash

    @patch("clinical.retrieval.cbt_retriever._search_dense")
    def test_retrieve_returns_correct_type(self, mock_search):
        r = CBTRetriever(n_results=5, similarity_threshold=0.0, enable_rerank=False)
        mock_search.return_value = [
            _MockQueryResult(
                text="Cognitive restructuring is a core CBT technique for modifying automatic thoughts.",
                score=0.85,
                metadata={"framework": "cbt", "topic": "depression"},
            ),
        ]

        with patch(
            "rag.retrieval.bm25.get_bm25_retriever",
            return_value=MagicMock(search=MagicMock(return_value=[])),
        ):
            result = r.retrieve("CBT for depression")

        assert isinstance(result, CBTEvidenceResult)
        assert len(result.retrieved) > 0
        assert result.retrieved[0].framework == "cbt"
        assert result.retrieved[0].quality_rating is not None

    @patch("clinical.retrieval.cbt_retriever._search_dense")
    def test_retrieve_with_filter(self, mock_search):
        r = CBTRetriever(n_results=5, similarity_threshold=0.0, enable_rerank=False)
        mock_search.return_value = [
            _MockQueryResult(
                text="Behavioural activation for depression.",
                score=0.78,
                metadata={"framework": "cbt", "topic": "depression"},
            ),
        ]

        with patch(
            "rag.retrieval.bm25.get_bm25_retriever",
            return_value=MagicMock(search=MagicMock(return_value=[])),
        ):
            result = r.retrieve(
                "behavioural activation",
                filters=CBTEvidenceFilter(topic="depression", intervention="behavioural activation"),
            )

        assert isinstance(result, CBTEvidenceResult)
        assert len(result.retrieved) >= 0

    def test_retrieve_empty_literature(self):
        """When no documents exist, retrieval should return empty gracefully."""
        r = CBTRetriever(n_results=5, similarity_threshold=0.0, enable_rerank=False)
        result = r.retrieve("CBT techniques")
        assert isinstance(result, CBTEvidenceResult)

    @patch("clinical.retrieval.cbt_retriever._search_dense")
    def test_retrieve_uses_framework_resolved_store(self, mock_search):
        """Dense search must use the manifest.lock-resolved store and pass the
        collection name to BM25 so both stages share the same collection."""
        class _FakeStore:
            collection_name = "kb_cbt_2026-08"

        fake_store = _FakeStore()
        r = CBTRetriever(n_results=5, similarity_threshold=0.0, enable_rerank=False)
        mock_search.return_value = [
            _MockQueryResult(
                text="Cognitive restructuring is a core CBT technique for modifying automatic thoughts.",
                score=0.85,
                metadata={"framework": "cbt", "topic": "depression"},
            ),
        ]
        bm25_mock = MagicMock(search=MagicMock(return_value=[]))

        with patch(
            "clinical.retrieval.cbt_retriever.resolve_framework_store",
            return_value=(fake_store, "kb_cbt_2026-08"),
        ), patch(
            "rag.retrieval.bm25.get_bm25_retriever",
            return_value=bm25_mock,
        ):
            result = r.retrieve("CBT for depression")

        assert isinstance(result, CBTEvidenceResult)
        assert len(result.retrieved) > 0
        _, dense_kwargs = mock_search.call_args
        assert dense_kwargs["store"] is fake_store
        _, bm25_kwargs = bm25_mock.search.call_args
        assert bm25_kwargs["collection_name"] == "kb_cbt_2026-08"

    @patch("clinical.retrieval.cbt_retriever._search_dense")
    def test_retrieve_without_alias_uses_default_store(self, mock_search):
        """When no manifest alias exists, dense search must fall back to the
        app-default vector store and BM25 to the default collection."""
        default_store = MagicMock()
        r = CBTRetriever(n_results=5, similarity_threshold=0.0, enable_rerank=False)
        mock_search.return_value = [
            _MockQueryResult(
                text="Behavioural activation is effective for depression.",
                score=0.7,
                metadata={"framework": "cbt"},
            ),
        ]
        bm25_mock = MagicMock(search=MagicMock(return_value=[]))

        with patch(
            "clinical.retrieval.cbt_retriever.resolve_framework_store",
            return_value=(default_store, None),
        ), patch(
            "rag.retrieval.bm25.get_bm25_retriever",
            return_value=bm25_mock,
        ):
            result = r.retrieve("behavioural activation")

        assert isinstance(result, CBTEvidenceResult)
        _, dense_kwargs = mock_search.call_args
        assert dense_kwargs["store"] is default_store
        _, bm25_kwargs = bm25_mock.search.call_args
        assert bm25_kwargs["collection_name"] is None


class TestCBTRetrieverAsync:
    @pytest.mark.asyncio
    @patch("clinical.retrieval.cbt_retriever._search_dense")
    async def test_retrieve_async(self, mock_search):
        r = CBTRetriever(n_results=5, similarity_threshold=0.0, enable_rerank=False)
        mock_search.return_value = [
            _MockQueryResult(
                text="CBT is an effective treatment for anxiety disorders.",
                score=0.88,
                metadata={"framework": "cbt"},
            ),
        ]

        with patch(
            "rag.retrieval.bm25.get_bm25_retriever",
            return_value=MagicMock(search=MagicMock(return_value=[])),
        ):
            result = await r.retrieve_async("CBT for anxiety")

        assert isinstance(result, CBTEvidenceResult)
        assert len(result.retrieved) >= 0

    @pytest.mark.asyncio
    async def test_retrieve_async_empty(self):
        r = CBTRetriever(n_results=5, similarity_threshold=0.0, enable_rerank=False)
        result = await r.retrieve_async("")
        assert isinstance(result, CBTEvidenceResult)


# ═══════════════════════════════════════════════════════════════════════
# Tests: Singleton factory
# ═══════════════════════════════════════════════════════════════════════


class TestGetCBTRetriever:
    def tearDown(self):
        # Reset singleton after test
        import clinical.retrieval.cbt_retriever as m

        m._retriever_instance = None

    def test_returns_singleton(self):
        r1 = get_cbt_retriever()
        r2 = get_cbt_retriever()
        assert r1 is r2

    def test_force_reload_creates_new(self):
        r1 = get_cbt_retriever()
        r2 = get_cbt_retriever(force_reload=True)
        assert r1 is not r2

    def test_default_config(self):
        r = get_cbt_retriever()
        assert r.framework == "cbt"
