from __future__ import annotations

import re

from clinical.evaluation.benchmarks.cases import BenchmarkCase
from clinical.evaluation.metrics.base import BaseMetric, MetricResult
from clinical.evaluation.metrics.utils import (
    citation_matches_inline,
    extract_references,
    extract_sections,
)


class HallucinationRateMetric(BaseMetric):
    """Estimate hallucination rate by checking citations against references.

    A lower score is better (0.0 = no hallucinations detected).
    """

    name = "hallucination_rate"

    _CITATION_RE = re.compile(
        r"(?:\([A-Z][A-Za-z]+(?:\s+(?:et\s+al\.?|&\s+[A-Z][A-Za-z]+))?\s*,?\s*\d{4}\)"
        r"|[A-Z][A-Za-z]+\s+et\s+al\.?\s*\(?\d{4}\)?)",
    )

    _FACTUAL_CLAIM_PATTERNS: list[re.Pattern[str]] = [
        re.compile(r"\b(?:prevalence|incidence|rate|percentage)\s+(?:of|is|are)\b", re.IGNORECASE),
        re.compile(r"\b(?:affects|impacts)\s+(?:approximately|about|around|~)\s+\d+", re.IGNORECASE),
        re.compile(r"\b(?:studies show|research indicates|evidence suggests|literature demonstrates)\b", re.IGNORECASE),
        re.compile(r"\b(?:according to|per)\s+(?:the\s+)?(?:DSM|ICD|WHO|NICE|APA)\b", re.IGNORECASE),
    ]

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
                score=1.0,
                details={"error": "no response generated"},
            )

        markdown = response.markdown if hasattr(response, "markdown") else str(response)
        sections = extract_sections(markdown)
        normalized_refs = extract_references(markdown, self._REF_HEADERS)

        body_text = "\n".join(
            v for k, v in sections.items() if k not in self._REF_HEADERS
        ) if sections else markdown

        unmatched_citations: list[str] = []
        matched_citations: list[str] = []

        for match in self._CITATION_RE.finditer(body_text):
            citation = match.group()
            is_matched = citation_matches_inline(citation, normalized_refs)
            if is_matched:
                matched_citations.append(citation)
            else:
                unmatched_citations.append(citation)

        uncited_claims = 0
        for pattern in self._FACTUAL_CLAIM_PATTERNS:
            for claim_match in pattern.finditer(body_text):
                pos = claim_match.start()
                before = body_text[max(0, pos - 80):pos + 80]
                has_nearby_citation = bool(self._CITATION_RE.search(before))
                if not has_nearby_citation:
                    uncited_claims += 1

        total_citations = len(matched_citations) + len(unmatched_citations)

        if total_citations == 0 and uncited_claims == 0:
            hallucination_rate = 0.0
        elif total_citations == 0:
            hallucination_rate = min(1.0, uncited_claims / max(uncited_claims, 1))
        else:
            mismatched_ratio = len(unmatched_citations) / max(total_citations, 1)
            claim_penalty = min(1.0, uncited_claims / max(total_citations, 1))
            hallucination_rate = (mismatched_ratio + claim_penalty) / 2.0

        hallucination_rate = min(1.0, hallucination_rate)

        return MetricResult(
            metric=self.name,
            case_id=case.case_id,
            score=round(hallucination_rate, 4),
            details={
                "total_citations_found": total_citations,
                "matched_citations": len(matched_citations),
                "unmatched_citations": len(unmatched_citations),
                "uncited_factual_claims": uncited_claims,
                "unmatched_list": unmatched_citations[:10],
            },
        )
