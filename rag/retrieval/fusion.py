from __future__ import annotations

from typing import Any

from app_logging.logger import get_logger

_log = get_logger(__name__)

_RRF_K: float = 60.0


def reciprocal_rank_fusion(
    dense_results: list[dict[str, Any]],
    sparse_results: list[dict[str, Any]],
    *,
    top_k: int = 10,
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
    rrf_k: float = _RRF_K,
) -> list[dict[str, Any]]:
    """Fuse dense and sparse retrieval results using Reciprocal Rank Fusion.

    Each result from both systems receives an RRF score::

        rrf_score(d) = dense_weight / (k + rank_dense(d))
                     + sparse_weight / (k + rank_sparse(d))

    Where ``k`` (default 60) is a constant that softens the impact of high
    ranks.  Documents that appear in both systems get a natural boost.

    Args:
        dense_results:   Results from the dense (vector) retriever, each
                         with ``chunk_id`` and ``rank`` fields.
        sparse_results:  Results from the sparse (BM25) retriever, each
                         with ``chunk_id`` and ``rank`` fields.
        top_k:           Number of results to return after fusion.
        dense_weight:    Weight for dense system scores (0.0 - 1.0).
        sparse_weight:   Weight for sparse system scores (0.0 - 1.0).
        rrf_k:           RRF constant (default 60).

    Returns:
        Top ``top_k`` results, sorted by descending combined RRF score,
        with an additional ``fusion_score`` key.
    """
    if not dense_results and not sparse_results:
        return []

    chunk_map: dict[str, dict[str, Any]] = {}
    seen: dict[str, dict[str, float]] = {}

    for r in dense_results:
        cid = r["chunk_id"]
        seen.setdefault(cid, {"dense": rrf_k, "sparse": rrf_k})
        seen[cid]["dense"] = r["rank"]
        chunk_map[cid] = r

    for r in sparse_results:
        cid = r["chunk_id"]
        seen.setdefault(cid, {"dense": rrf_k, "sparse": rrf_k})
        seen[cid]["sparse"] = r["rank"]
        if cid not in chunk_map:
            chunk_map[cid] = r

    fused: list[dict[str, Any]] = []
    for cid, ranks in seen.items():
        rrf_score = (
            dense_weight / (rrf_k + ranks["dense"])
            + sparse_weight / (rrf_k + ranks["sparse"])
        )
        chunk = dict(chunk_map[cid])
        chunk["fusion_score"] = round(rrf_score, 6)
        chunk["dense_rank"] = int(ranks["dense"]) if ranks["dense"] < rrf_k else None
        chunk["sparse_rank"] = int(ranks["sparse"]) if ranks["sparse"] < rrf_k else None
        chunk["score"] = chunk["fusion_score"]
        fused.append(chunk)

    fused.sort(key=lambda x: x["fusion_score"], reverse=True)

    kept = fused[:top_k]
    for rank, c in enumerate(kept, start=1):
        c["rank"] = rank

    _log.info(
        "fusion.rrf_complete",
        dense_in=len(dense_results),
        sparse_in=len(sparse_results),
        unique=len(seen),
        fused_out=len(kept),
    )
    return kept


def weighted_score_fusion(
    dense_results: list[dict[str, Any]],
    sparse_results: list[dict[str, Any]],
    *,
    top_k: int = 10,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
) -> list[dict[str, Any]]:
    """Fuse dense and sparse results using weighted score combination.

    Scores from each system are min-max normalised to [0, 1] before
    combination::

        norm_score(d) = (score(d) - min) / (max - min)
        combined(d)   = dense_weight * norm_score_dense(d)
                      + sparse_weight * norm_score_sparse(d)

    Args:
        dense_results:   Results from dense retriever with ``chunk_id`` and ``score``.
        sparse_results:  Results from sparse retriever with ``chunk_id`` and ``score``.
        top_k:           Number of results to return.
        dense_weight:    Weight for dense scores.
        sparse_weight:   Weight for sparse scores.

    Returns:
        Top ``top_k`` results sorted by descending combined score.
    """
    if not dense_results and not sparse_results:
        return []

    if not dense_results:
        return sparse_results[:top_k]
    if not sparse_results:
        return dense_results[:top_k]

    def normalise(items: list[dict[str, Any]], key: str) -> None:
        scores = [r.get(key, 0.0) for r in items]
        lo, hi = min(scores), max(scores)
        if hi - lo < 1e-9:
            for r in items:
                r["_norm"] = 0.5
        else:
            for r in items:
                r["_norm"] = (r.get(key, 0.0) - lo) / (hi - lo)

    normalise(dense_results, "score")
    normalise(sparse_results, "score")

    dense_map = {r["chunk_id"]: r for r in dense_results}
    sparse_map = {r["chunk_id"]: r for r in sparse_results}
    all_ids = set(dense_map) | set(sparse_map)

    fused: list[dict[str, Any]] = []
    for cid in all_ids:
        d = dense_map.get(cid)
        s = sparse_map.get(cid)
        combined = (
            dense_weight * (d["_norm"] if d else 0.0)
            + sparse_weight * (s["_norm"] if s else 0.0)
        )
        chunk = dict(d or s)
        chunk["fusion_score"] = round(combined, 6)
        chunk["score"] = chunk["fusion_score"]
        chunk["dense_score"] = d["score"] if d else None
        chunk["sparse_score"] = s["score"] if s else None
        fused.append(chunk)

    fused.sort(key=lambda x: x["fusion_score"], reverse=True)
    kept = fused[:top_k]
    for rank, c in enumerate(kept, start=1):
        c["rank"] = rank

    _log.info(
        "fusion.weighted_complete",
        dense_in=len(dense_results),
        sparse_in=len(sparse_results),
        unique=len(all_ids),
        fused_out=len(kept),
    )
    return kept
