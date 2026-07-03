from __future__ import annotations

from clinical.evaluation.benchmarks.cases import BenchmarkCase
from clinical.evaluation.metrics.base import BaseMetric, MetricResult


class RetrievalPrecisionMetric(BaseMetric):
    """Evaluate what fraction of top-5 retrieved chunks are topically relevant.

    Uses keyword overlap between chunk text and the case's expected topics
    as a proxy for relevance — works without requiring pre-annotated relevant
    chunk IDs.
    """

    name = "retrieval_precision@5"

    async def score_case(
        self,
        case: BenchmarkCase,
        pipeline_state: dict,
    ) -> MetricResult:
        chunks = pipeline_state.get("retrieved_chunks", [])
        if not chunks:
            return MetricResult(
                metric=self.name,
                case_id=case.case_id,
                score=0.0,
                details={"top_k": 5, "chunks_retrieved": 0},
            )

        top_k = min(5, len(chunks))
        top_chunks = chunks[:top_k]
        expected = [t.lower() for t in case.ground_truth.expected_topics]

        relevant_count = 0
        chunk_scores: list[dict] = []
        for c in top_chunks:
            text_lower = c.text.lower() if hasattr(c, "text") else str(c).lower()
            matched_topics = [t for t in expected if t in text_lower]
            is_relevant = len(matched_topics) > 0
            if is_relevant:
                relevant_count += 1
            chunk_scores.append({
                "text_preview": text_lower[:100],
                "score": c.score if hasattr(c, "score") else None,
                "is_relevant": is_relevant,
                "matched_topics": matched_topics,
            })

        precision = relevant_count / top_k if top_k > 0 else 0.0

        return MetricResult(
            metric=self.name,
            case_id=case.case_id,
            score=round(precision, 4),
            details={
                "top_k": top_k,
                "relevant_count": relevant_count,
                "chunks_retrieved": len(chunks),
                "chunk_details": chunk_scores,
                "expected_topics": expected,
            },
        )
