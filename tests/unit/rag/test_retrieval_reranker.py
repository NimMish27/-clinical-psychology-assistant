from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rag.retrieval.reranker import CrossEncoderReranker, get_reranker


@pytest.fixture
def candidates():
    return [
        {"chunk_id": "id1", "text": "Cognitive behavioral therapy for anxiety", "source": "doc1.pdf", "page": 1,
         "score": 0.9, "rank": 1, "metadata": {"source": "doc1.pdf"}},
        {"chunk_id": "id2", "text": "Depression medication SSRIs", "source": "doc2.pdf", "page": 2,
         "score": 0.8, "rank": 2, "metadata": {"source": "doc2.pdf"}},
        {"chunk_id": "id3", "text": "Mindfulness based stress reduction techniques", "source": "doc3.pdf", "page": 3,
         "score": 0.7, "rank": 3, "metadata": {"source": "doc3.pdf"}},
    ]


class TestCrossEncoderReranker:
    def test_initialisation(self):
        r = CrossEncoderReranker(model_name="test-model")
        assert r.model_name == "test-model"
        assert r.is_loaded is False

    def test_repr(self):
        r = CrossEncoderReranker()
        assert "CrossEncoderReranker" in repr(r)

    def test_empty_candidates_returns_empty(self):
        r = CrossEncoderReranker()
        assert r.rerank("query", []) == []

    def test_empty_query_returns_candidates_unchanged(self):
        r = CrossEncoderReranker()
        candidates = [{"chunk_id": "id1", "text": "text", "source": "doc.pdf", "page": 1,
                       "score": 0.9, "rank": 1, "metadata": {}}]
        result = r.rerank("", candidates)
        assert result == candidates

    def test_rerank_assigns_rerank_score(self, candidates):
        r = CrossEncoderReranker()
        r._loaded = True
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5, 0.9, 0.3]
        r._model = mock_model

        result = r.rerank("CBT therapy", candidates, top_k=3)
        assert len(result) == 3
        for c in result:
            assert "rerank_score" in c
        # Highest rerank score should be first
        assert result[0]["chunk_id"] == "id2"

    def test_top_k_limits_results(self, candidates):
        r = CrossEncoderReranker()
        r._loaded = True
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5, 0.9, 0.3]
        r._model = mock_model

        result = r.rerank("CBT therapy", candidates, top_k=1)
        assert len(result) == 1

    def test_score_updated_after_rerank(self, candidates):
        r = CrossEncoderReranker()
        r._loaded = True
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5, 0.9, 0.3]
        r._model = mock_model

        result = r.rerank("CBT therapy", candidates, top_k=3)
        # score should be updated to rerank_score
        assert result[0]["score"] == result[0]["rerank_score"]

    def test_ranks_sequential_after_rerank(self, candidates):
        r = CrossEncoderReranker()
        r._loaded = True
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5, 0.9, 0.3]
        r._model = mock_model

        result = r.rerank("CBT therapy", candidates, top_k=3)
        ranks = [c["rank"] for c in result]
        assert ranks == [1, 2, 3]

    def test_model_unavailable_returns_original_order(self, candidates):
        r = CrossEncoderReranker()
        r._loaded = True
        r._model = None

        result = r.rerank("CBT therapy", candidates, top_k=3)
        assert result == candidates

    def test_predict_failure_returns_original_order(self, candidates):
        r = CrossEncoderReranker()
        r._loaded = True
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("predict failed")
        r._model = mock_model

        result = r.rerank("CBT therapy", candidates, top_k=3)
        assert result == candidates

    def test_singleton(self):
        with patch("config.settings.get_settings") as mock_settings:
            cfg = MagicMock()
            cfg.rag.cross_encoder_model = "test-model"
            mock_settings.return_value = cfg

            a = get_reranker()
            b = get_reranker()
            assert a is b

    def test_force_reload(self):
        with patch("config.settings.get_settings") as mock_settings:
            cfg = MagicMock()
            cfg.rag.cross_encoder_model = "test-model"
            mock_settings.return_value = cfg

            a = get_reranker()
            b = get_reranker(force_reload=True)
            assert a is not b
