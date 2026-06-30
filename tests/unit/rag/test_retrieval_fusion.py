from __future__ import annotations

import pytest

from rag.retrieval.fusion import reciprocal_rank_fusion, weighted_score_fusion


def _make_dense(cid: str, rank: int, score: float) -> dict:
    return {
        "chunk_id": cid,
        "text": f"text_{cid}",
        "source": "doc.pdf",
        "page": rank,
        "score": score,
        "rank": rank,
        "metadata": {"source": "doc.pdf"},
    }


def _make_sparse(cid: str, rank: int, score: float) -> dict:
    return {
        "chunk_id": cid,
        "text": f"text_{cid}",
        "source": "doc.pdf",
        "page": rank,
        "score": score,
        "rank": rank,
        "metadata": {"source": "doc.pdf"},
    }


class TestReciprocalRankFusion:
    def test_both_empty(self):
        assert reciprocal_rank_fusion([], []) == []

    def test_dense_only(self):
        dense = [_make_dense("id1", 1, 0.9), _make_dense("id2", 2, 0.8)]
        fused = reciprocal_rank_fusion(dense, [], top_k=5)
        assert len(fused) == 2
        assert fused[0]["chunk_id"] == "id1"
        assert fused[0]["fusion_score"] > 0

    def test_sparse_only(self):
        sparse = [_make_sparse("id1", 1, 10.0), _make_sparse("id2", 2, 8.0)]
        fused = reciprocal_rank_fusion([], sparse, top_k=5)
        assert len(fused) == 2

    def test_both_sources_boost_overlap(self):
        dense = [_make_dense("id1", 1, 0.9), _make_dense("id2", 2, 0.8)]
        sparse = [_make_sparse("id2", 1, 10.0), _make_sparse("id3", 2, 8.0)]
        fused = reciprocal_rank_fusion(dense, sparse, top_k=5)
        # id2 appears in both -> should rank higher than id1 or id3
        id2 = [f for f in fused if f["chunk_id"] == "id2"][0]
        id1 = [f for f in fused if f["chunk_id"] == "id1"][0]
        assert id2["fusion_score"] > id1["fusion_score"]

    def test_respects_top_k(self):
        dense = [_make_dense(f"id{i}", i, 0.9) for i in range(1, 11)]
        sparse = [_make_sparse(f"id{i}", i, 10.0) for i in range(1, 11)]
        fused = reciprocal_rank_fusion(dense, sparse, top_k=3)
        assert len(fused) == 3

    def test_weights_affect_ranking(self):
        dense = [_make_dense("id1", 1, 0.9), _make_dense("id2", 10, 0.1)]
        sparse = [_make_sparse("id1", 10, 1.0), _make_sparse("id2", 1, 10.0)]

        dense_heavy = reciprocal_rank_fusion(dense, sparse, top_k=2, dense_weight=0.9, sparse_weight=0.1)
        sparse_heavy = reciprocal_rank_fusion(dense, sparse, top_k=2, dense_weight=0.1, sparse_weight=0.9)

        # With dense_heavy, id1 should rank first; with sparse_heavy, id2
        assert dense_heavy[0]["chunk_id"] == "id1"
        assert sparse_heavy[0]["chunk_id"] == "id2"

    def test_scores_are_sorted_descending(self):
        dense = [_make_dense(f"id{i}", i, 0.9) for i in range(1, 6)]
        sparse = [_make_sparse(f"id{i}", i, 10.0) for i in range(1, 6)]
        fused = reciprocal_rank_fusion(dense, sparse, top_k=5)
        scores = [f["fusion_score"] for f in fused]
        assert scores == sorted(scores, reverse=True)

    def test_ranks_are_sequential(self):
        dense = [_make_dense(f"id{i}", i, 0.9) for i in range(1, 6)]
        sparse = [_make_sparse("id_extra", 1, 10.0)]
        fused = reciprocal_rank_fusion(dense, sparse, top_k=6)
        ranks = [f["rank"] for f in fused]
        assert ranks == list(range(1, len(fused) + 1))


class TestWeightedScoreFusion:
    def test_both_empty(self):
        assert weighted_score_fusion([], []) == []

    def test_dense_only(self):
        dense = [_make_dense("id1", 1, 0.9), _make_dense("id2", 2, 0.8)]
        fused = weighted_score_fusion(dense, [], top_k=5)
        assert len(fused) == 2

    def test_sparse_only(self):
        sparse = [_make_sparse("id1", 1, 10.0)]
        fused = weighted_score_fusion([], sparse, top_k=5)
        assert len(fused) == 1

    def test_combines_scores(self):
        dense = [_make_dense("id1", 1, 0.9)]
        sparse = [_make_sparse("id1", 1, 10.0)]
        fused = weighted_score_fusion(dense, sparse, top_k=5, dense_weight=0.7, sparse_weight=0.3)
        assert len(fused) == 1
        assert fused[0]["fusion_score"] > 0
        assert "dense_score" in fused[0]
        assert "sparse_score" in fused[0]

    def test_respects_top_k(self):
        dense = [_make_dense(f"id{i}", i, 0.9) for i in range(1, 11)]
        sparse = [_make_sparse(f"id{i}", i, 10.0) for i in range(1, 11)]
        fused = weighted_score_fusion(dense, sparse, top_k=3)
        assert len(fused) == 3

    def test_scores_are_sorted_descending(self):
        dense = [_make_dense(f"id{i}", i, 0.9 - i * 0.1) for i in range(1, 4)]
        sparse = [_make_sparse(f"id{i}", i, 10.0 - i * 2.0) for i in range(1, 4)]
        fused = weighted_score_fusion(dense, sparse, top_k=3)
        scores = [f["fusion_score"] for f in fused]
        assert scores == sorted(scores, reverse=True)
