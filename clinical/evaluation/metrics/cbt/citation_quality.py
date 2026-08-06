"""
clinical/evaluation/metrics/cbt/citation_quality.py

Evaluates citation quality specifically for CBT framework output.

Measures:
  - Whether citations in the CBT guidance and overall response are
    well-formed and match expected CBT references.
  - Whether the output cites empirical CBT sources (author names,
    years, treatment outcome references).
  - Proportion of citations that match references in a reference section.

Leverages the same ``citation_matches_inline`` helper as the existing
``CitationAccuracyMetric``, but focuses specifically on CBT literature.
"""

from __future__ import annotations

import re

from clinical.evaluation.benchmarks.cases import BenchmarkCase
from clinical.evaluation.metrics.base import BaseMetric, MetricResult
from clinical.evaluation.metrics.utils import (
    citation_matches_inline,
    extract_references,
    extract_sections,
)


class CBTCitationQualityMetric(BaseMetric):
    """Evaluate citation quality for CBT framework output.

    Three sub-scores weighted equally:
      1. Citation presence — are citations present at all?
      2. Citation accuracy — do inline citations match reference entries?
      3. CBT relevance — do citations reference known CBT sources?
    """

    name = "cbt_citation_quality"

    _CITATION_RE = re.compile(
        r"(?:\([A-Z][A-Za-z]+(?:\s+(?:et\s+al\.?|&\s+[A-Z][A-Za-z]+))?\s*,?\s*\d{4}\)"
        r"|[A-Z][A-Za-z]+\s+et\s+al\.?\s*\(?\d{4}\)?)",
    )

    _CBT_AUTHORS: set[str] = {
        "beck", "clark", "barlow", "foa", "salkovskis",
        "wells", "fairburn", "shafran", "egan", "padesky",
        "dugas", "borkovec", "vlaeyen", "watkins", "spielman",
        "novaco", "martell", "nolen-hoeksema", "öst", "fennell",
        "dobson", "hollon", "dimidjian", "cuijpers", "butler",
        "chapman", "linehan", "hayes", "hofmann",
    }

    _REF_HEADERS = {"references", "citations", "bibliography", "sources"}

    async def score_case(
        self,
        case: BenchmarkCase,
        pipeline_state: dict,
    ) -> MetricResult:
        # ── Extract text from CBT guidance and response ──────────────────
        guidance, response_text = self._extract_texts(pipeline_state)
        combined = (guidance + "\n" + response_text)

        # ── Sub-score 1: Citation presence ──────────────────────────────
        citations = self._CITATION_RE.findall(combined)
        presence_score = min(1.0, len(citations) * 0.15)

        # ── Sub-score 2: Citation accuracy ──────────────────────────────
        accuracy_score = self._score_accuracy(combined, citations)

        # ── Sub-score 3: CBT relevance ──────────────────────────────────
        relevance_score = self._score_cbt_relevance(combined, citations)

        overall = round(
            (presence_score + accuracy_score + relevance_score) / 3.0, 4
        )

        return MetricResult(
            metric=self.name,
            case_id=case.case_id,
            score=overall,
            details={
                "presence_score": round(presence_score, 4),
                "accuracy_score": round(accuracy_score, 4),
                "relevance_score": round(relevance_score, 4),
                "citations_found": len(citations),
                "cbt_author_matches": self._last_author_matches,
                "matched_references": self._last_ref_matches,
                "guidance_citations": len(self._CITATION_RE.findall(guidance)),
                "response_citations": len(self._CITATION_RE.findall(response_text)),
            },
        )

    # ── Internal detail storage ─────────────────────────────────────────────

    _last_author_matches: list[str] = []
    _last_ref_matches: dict = {}

    # ── Extraction ──────────────────────────────────────────────────────────

    @staticmethod
    def _extract_texts(pipeline_state: dict) -> tuple[str, str]:
        framework_results = pipeline_state.get("framework_results", {})
        cbt_result = framework_results.get("cbt")
        guidance = (
            cbt_result.guidance
            if cbt_result and hasattr(cbt_result, "guidance")
            else ""
        )
        response = pipeline_state.get("response")
        response_text = (
            response.markdown
            if response and hasattr(response, "markdown")
            else (str(response) if response else "")
        )
        return guidance, response_text

    # ── Accuracy scoring ────────────────────────────────────────────────────

    def _score_accuracy(
        self,
        combined: str,
        citations: list[str],
    ) -> float:
        normalized_refs = extract_references(combined, self._REF_HEADERS)

        if not citations:
            return 0.0

        matched_count = 0
        match_details: list[dict] = []
        for cite in citations:
            is_matched = citation_matches_inline(cite, normalized_refs)
            if is_matched:
                matched_count += 1
            match_details.append({"citation": cite, "matched": is_matched})

        self._last_ref_matches = {
            "ref_count": len(normalized_refs),
            "matched_citations": matched_count,
            "total_citations": len(citations),
            "match_details": match_details,
        }

        if not normalized_refs:
            return 0.2  # partial credit even without a refs section

        return matched_count / len(citations)

    # ── CBT relevance scoring ───────────────────────────────────────────────

    def _score_cbt_relevance(
        self,
        combined: str,
        citations: list[str],
    ) -> float:
        combined_lower = combined.lower()

        author_matches: list[str] = []
        for author in self._CBT_AUTHORS:
            if author in combined_lower:
                author_matches.append(author)

        self._last_author_matches = author_matches

        # Base score from author matches
        author_score = min(1.0, len(author_matches) * 0.1)

        # Citation format check — APA-like citations get a bonus
        has_apa = bool(re.search(r"\([A-Za-z]+\s+et\s+al\..*?\d{4}\)", combined, re.IGNORECASE))
        format_bonus = 0.15 if has_apa else 0.0

        # Year presence bonus
        has_years = bool(re.search(r"\b(19|20)\d{2}\b", combined))
        year_bonus = 0.1 if has_years else 0.0

        return min(1.0, author_score + format_bonus + year_bonus)
