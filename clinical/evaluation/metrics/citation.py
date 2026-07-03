from __future__ import annotations

import re

from clinical.evaluation.benchmarks.cases import BenchmarkCase
from clinical.evaluation.metrics.base import BaseMetric, MetricResult
from clinical.evaluation.metrics.utils import (
    citation_matches_inline,
    extract_references,
    extract_sections,
    normalize_text,
)


class CitationAccuracyMetric(BaseMetric):
    """Evaluate whether inline citations match entries in the reference section."""

    name = "citation_accuracy"

    _CITATION_RE = re.compile(
        r"(?:\([A-Z][A-Za-z]+(?:\s+(?:et\s+al\.?|&\s+[A-Z][A-Za-z]+))?\s*,?\s*\d{4}\)"
        r"|[A-Z][A-Za-z]+\s+et\s+al\.?\s*\(?\d{4}\)?)",
    )

    _REF_HEADERS = {"references", "10. references", "reference"}

    async def score_case(
        self,
        case: BenchmarkCase,
        pipeline_state: dict,
    ) -> MetricResult:
        response = pipeline_state.get("response")
        if not response:
            return MetricResult(
                metric=self.name,
                case_id=case.case_id,
                score=0.0,
                details={"error": "no response generated"},
            )

        markdown = response.markdown if hasattr(response, "markdown") else str(response)
        sections = extract_sections(markdown)
        normalized_refs = extract_references(markdown, self._REF_HEADERS)

        body_text = "\n".join(
            v for k, v in sections.items() if k not in self._REF_HEADERS
        ) if sections else markdown

        citations_found: list[dict] = []
        for match in self._CITATION_RE.finditer(body_text):
            citation = match.group()
            is_matched = citation_matches_inline(citation, normalized_refs)
            citations_found.append({
                "citation": citation,
                "matched_in_refs": is_matched,
            })

        if not citations_found:
            return MetricResult(
                metric=self.name,
                case_id=case.case_id,
                score=0.0,
                details={"citations_found": 0, "refs_found": len(normalized_refs)},
            )

        matched_count = sum(1 for c in citations_found if c["matched_in_refs"])
        accuracy = matched_count / len(citations_found)

        return MetricResult(
            metric=self.name,
            case_id=case.case_id,
            score=round(accuracy, 4),
            details={
                "citations_found": len(citations_found),
                "matched": matched_count,
                "unmatched": len(citations_found) - matched_count,
                "citations": citations_found,
                "references_section_found": bool(normalized_refs),
                "ref_count": len(normalized_refs),
            },
        )
