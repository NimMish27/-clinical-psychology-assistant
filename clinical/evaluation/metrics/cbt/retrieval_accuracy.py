"""
clinical/evaluation/metrics/cbt/retrieval_accuracy.py

Evaluates retrieval accuracy for CBT framework evaluation.

Two components:
  1. Standard pipeline retrieval — checks if ``retrieved_chunks`` cover
     CBT-relevant expected topics (reuses the chunk-level approach from
     ``RetrievalPrecisionMetric``).
  2. CBT framework literature retrieval — inspects the CBT framework's
     guidance text for citations and references to expected literature topics.

The final score is a weighted combination of both components.
"""

from __future__ import annotations

import re

from clinical.evaluation.benchmarks.cases import BenchmarkCase
from clinical.evaluation.metrics.base import BaseMetric, MetricResult


class CBTRetrievalAccuracyMetric(BaseMetric):
    """Evaluate retrieval accuracy for CBT-specific evidence.

    Measures whether:
      - Retrieved chunks cover CBT-relevant expected topics (60% weight).
      - The CBT framework guidance references expected CBT literature (40% weight).

    A high score requires both relevant chunks to be retrieved and the CBT
    framework to reference appropriate literature in its output.
    """

    name = "cbt_retrieval_accuracy"

    async def score_case(
        self,
        case: BenchmarkCase,
        pipeline_state: dict,
    ) -> MetricResult:
        # ── Component 1: Chunk-level retrieval precision against CBT topics ──
        chunk_score = self._score_chunk_retrieval(case, pipeline_state)

        # ── Component 2: CBT framework literature references ─────────────────
        literature_score = self._score_literature_references(case, pipeline_state)

        # ── Weighted combination ─────────────────────────────────────────────
        overall = round(chunk_score * 0.6 + literature_score * 0.4, 4)

        return MetricResult(
            metric=self.name,
            case_id=case.case_id,
            score=overall,
            details={
                "chunk_retrieval_score": round(chunk_score, 4),
                "literature_reference_score": round(literature_score, 4),
                "chunk_details": self._last_chunk_details,
                "literature_details": self._last_lit_details,
            },
        )

    # ── Internal state for detail extraction ─────────────────────────────────

    _last_chunk_details: list[dict] = []
    _last_lit_details: dict = {}

    # ── Chunk retrieval ──────────────────────────────────────────────────────

    def _score_chunk_retrieval(
        self,
        case: BenchmarkCase,
        pipeline_state: dict,
    ) -> float:
        chunks = pipeline_state.get("retrieved_chunks", [])
        cbt_topics = case.ground_truth.expected_cbt_literature_topics
        all_topics = list(
            set(
                [t.lower() for t in case.ground_truth.expected_topics]
                + [t.lower() for t in cbt_topics]
            )
        )

        if not chunks:
            self._last_chunk_details = [
                {"chunks_retrieved": 0, "note": "no chunks available"}
            ]
            return 0.0

        top_k = min(10, len(chunks))
        top_chunks = chunks[:top_k]
        relevant_count = 0
        self._last_chunk_details = []

        for c in top_chunks:
            text_lower = c.text.lower() if hasattr(c, "text") else str(c).lower()
            matched = [t for t in all_topics if t in text_lower]
            is_relevant = len(matched) > 0
            if is_relevant:
                relevant_count += 1
            self._last_chunk_details.append({
                "text_preview": text_lower[:120],
                "is_relevant": is_relevant,
                "matched_topics": matched[:3],
                "score": round(c.score, 4) if hasattr(c, "score") and c.score else None,
            })

        return relevant_count / top_k if top_k > 0 else 0.0

    # ── Literature references in CBT guidance ────────────────────────────────

    _CITATION_RE = re.compile(
        r"(?:\([A-Z][A-Za-z]+(?:\s+(?:et\s+al\.?|&\s+[A-Z][A-Za-z]+))?\s*,?\s*\d{4}\)"
        r"|[A-Z][A-Za-z]+\s+et\s+al\.?\s*\(?\d{4}\)?)",
    )

    def _score_literature_references(
        self,
        case: BenchmarkCase,
        pipeline_state: dict,
    ) -> float:
        framework_results = pipeline_state.get("framework_results", {})
        cbt_result = framework_results.get("cbt")

        guidance_text = ""
        if cbt_result and hasattr(cbt_result, "guidance"):
            guidance_text = cbt_result.guidance

        response = pipeline_state.get("response")
        response_text = ""
        if response:
            response_text = (
                response.markdown
                if hasattr(response, "markdown")
                else str(response)
            )

        combined = (guidance_text + "\n" + response_text).lower()
        expected_lit = [
            t.lower() for t in case.ground_truth.expected_cbt_literature_topics
        ]

        citations = self._CITATION_RE.findall(combined)
        topic_matches = 0
        matched_topics: list[str] = []

        for topic in expected_lit:
            if topic in combined:
                topic_matches += 1
                matched_topics.append(topic)

        # Also check for key CBT author names
        cbt_author_refs = [
            "beck", "clark", "barlow",  "foa", "salkovskis",
            "wells", "fairburn", "shafran", "egan", "padesky",
            "dugas", "borkovec", "vlaeyen", "watkins", "spielman",
            "novaco", "martell", "nolen-hoeksema", "öst", "fennell",
        ]
        author_count = sum(
            1 for author in cbt_author_refs if author in combined
        )

        total_lit = max(len(expected_lit), 1)
        topic_ratio = topic_matches / total_lit
        author_bonus = min(0.3, author_count * 0.05)

        # Citation presence bonus
        citation_bonus = 0.1 if citations else 0.0

        raw_score = topic_ratio + author_bonus + citation_bonus

        self._last_lit_details = {
            "citations_in_output": len(citations),
            "author_references_found": author_count,
            "expected_literature_topics": expected_lit,
            "matched_literature_topics": matched_topics,
            "unmatched_literature_topics": [
                t for t in expected_lit if t not in matched_topics
            ],
            "topic_ratio": round(topic_ratio, 4),
            "author_bonus": round(author_bonus, 4),
            "citation_bonus": round(citation_bonus, 4),
            "guidance_available": bool(guidance_text),
        }

        return min(1.0, raw_score)
