from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from rag.retrieval.bm25 import BM25Retriever, _SimpleBM25
from rag.retrieval.bm25 import get_bm25_retriever


@pytest.fixture
def bm25():
    return BM25Retriever(k1=1.5, b=0.75)


class TestSimpleBM25:
    def test_basic_scoring(self):
        corpus = [["hello", "world"], ["foo", "bar"]]
        bm25 = _SimpleBM25(corpus)
        scores = bm25.get_scores(["hello"])
        assert len(scores) == 2
        assert scores[0] > 0
        assert scores[1] == 0

    def test_empty_query(self):
        corpus = [["hello", "world"]]
        bm25 = _SimpleBM25(corpus)
        scores = bm25.get_scores([])
        assert scores == [0.0]

    def test_unknown_term(self):
        corpus = [["hello", "world"]]
        bm25 = _SimpleBM25(corpus)
        scores = bm25.get_scores(["unknown"])
        assert scores == [0.0]


class TestBM25Retriever:
    def test_initialisation(self, bm25):
        assert bm25.is_loaded is False
        assert bm25.corpus_size == 0

    def test_repr(self, bm25):
        r = repr(bm25)
        assert "BM25Retriever" in r
        assert "k1=1.5" in r

    def test_tokenize_removes_punctuation(self, bm25):
        tokens = bm25._tokenize("Hello, World! How's it going?")
        assert "hello" in tokens
        assert "world" in tokens
        assert "how" in tokens
        assert "s" not in tokens  # single chars are removed
        assert "going" in tokens

    def test_tokenize_lowercases(self, bm25):
        tokens = bm25._tokenize("CBT for GAD")
        assert "cbt" in tokens
        assert "gad" in tokens

    def test_tokenize_empty(self, bm25):
        assert bm25._tokenize("") == []

    def test_match_metadata_exact_match(self, bm25):
        meta = {"source": "DSM5.pdf", "topic": "depression"}
        assert bm25._match_metadata(meta, {"source": "DSM5.pdf"})
        assert not bm25._match_metadata(meta, {"source": "other.pdf"})

    def test_match_metadata_in_operator(self, bm25):
        meta = {"therapy": "CBT"}
        filt = {"therapy": {"$in": ["CBT", "ACT"]}}
        assert bm25._match_metadata(meta, filt)
        filt2 = {"therapy": {"$in": ["ACT", "DBT"]}}
        assert not bm25._match_metadata(meta, filt2)

    def test_match_metadata_eq_operator(self, bm25):
        meta = {"disorder": "MDD"}
        filt = {"disorder": {"$eq": "MDD"}}
        assert bm25._match_metadata(meta, filt)
        filt2 = {"disorder": {"$eq": "GAD"}}
        assert not bm25._match_metadata(meta, filt2)

    def test_search_empty_corpus_returns_empty(self, bm25):
        bm25._loaded = True
        bm25._corpus = []
        results = bm25.search("test query")
        assert results == []

    def test_search_empty_query_returns_empty(self, bm25):
        bm25._loaded = True
        bm25._corpus = ["some text"]
        bm25._chunk_ids = ["id1"]
        bm25._metadatas = [{}]
        bm25._bm25 = _SimpleBM25([["some", "text"]])
        results = bm25.search("")
        assert results == []

    def test_search_whitespace_query_returns_empty(self, bm25):
        bm25._loaded = True
        bm25._corpus = ["some text"]
        bm25._chunk_ids = ["id1"]
        bm25._metadatas = [{}]
        bm25._bm25 = _SimpleBM25([["some", "text"]])
        results = bm25.search("  ")
        assert results == []

    def test_search_returns_ranked_results(self, bm25):
        bm25._loaded = True
        bm25._corpus = ["cognitive behavioral therapy", "depression medication", "therapy for anxiety"]
        bm25._chunk_ids = ["id1", "id2", "id3"]
        bm25._metadatas = [{"source": "doc1"}, {"source": "doc2"}, {"source": "doc3"}]
        tokenized = [bm25._tokenize(d) for d in bm25._corpus]
        bm25._bm25 = _SimpleBM25(tokenized)

        results = bm25.search("therapy", n_results=2)
        assert len(results) == 2
        for r in results:
            assert "chunk_id" in r
            assert "text" in r
            assert "score" in r
            assert "rank" in r
        # Therapy-related docs should score higher
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_with_metadata_filter(self, bm25):
        bm25._loaded = True
        docs = [
            ("id1", "therapy for depression", {"source": "doc1", "topic": "depression"}),
            ("id2", "therapy for anxiety", {"source": "doc2", "topic": "anxiety"}),
            ("id3", "depression medication", {"source": "doc3", "topic": "depression"}),
        ]
        bm25._chunk_ids = [d[0] for d in docs]
        bm25._corpus = [d[1] for d in docs]
        bm25._metadatas = [d[2] for d in docs]
        tokenized = [bm25._tokenize(d) for d in bm25._corpus]
        bm25._bm25 = _SimpleBM25(tokenized)

        results = bm25.search("therapy", n_results=10, metadata_filter={"topic": "depression"})
        assert len(results) == 2
        assert all(r["metadata"]["topic"] == "depression" for r in results)

    def test_singleton(self):
        with patch("config.settings.get_settings") as mock_settings:
            cfg = MagicMock()
            cfg.rag.bm25_k1 = 1.5
            cfg.rag.bm25_b = 0.75
            mock_settings.return_value = cfg

            a = get_bm25_retriever()
            b = get_bm25_retriever()
            assert a is b

    def test_force_reload(self):
        with patch("config.settings.get_settings") as mock_settings:
            cfg = MagicMock()
            cfg.rag.bm25_k1 = 1.5
            cfg.rag.bm25_b = 0.75
            mock_settings.return_value = cfg

            a = get_bm25_retriever()
            b = get_bm25_retriever(force_reload=True)
            assert a is not b
