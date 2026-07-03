from __future__ import annotations

import re

from clinical.evaluation.benchmarks.cases import BenchmarkCase
from clinical.evaluation.metrics.base import BaseMetric, MetricResult


class FormulationQualityMetric(BaseMetric):
    """Evaluate formulation quality on a multi-axis rubric.

    Axes:
      - Tentativeness: avoids definitive diagnosis language
      - Evidence-grounded: references evidence or retrieved chunks
      - Specificity: addresses the specific case, not generic
      - Alternative considerations: mentions differentials/uncertainties
      - Confidence-appropriate: expresses appropriate uncertainty
    """

    name = "formulation_quality"

    _DIAGNOSIS_PATTERNS: list[re.Pattern[str]] = [
        re.compile(r"\bdiagnos(?:ed|is)\s+(?:with|of)\s+", re.IGNORECASE),
        re.compile(r"\bs(?:uffer|uffers|uffering)\s+from\b", re.IGNORECASE),
        re.compile(r"\b(?:patient|client)\s+has\s+(?:\w+\s+){0,4}(?:disorder|depression|anxiety|syndrome)\b", re.IGNORECASE),
        re.compile(r"\ba\s+case\s+of\s+", re.IGNORECASE),
    ]

    _TENTATIVE_WORDS = {
        "may", "might", "could", "suggests", "suggesting", "appears",
        "tentative", "hypothesis", "perhaps", "possibly", "potential",
        "consistent with", "may reflect", "may indicate",
    }

    async def score_case(
        self,
        case: BenchmarkCase,
        pipeline_state: dict,
    ) -> MetricResult:
        response = pipeline_state.get("response")
        formulation = pipeline_state.get("formulation")

        if not response and not formulation:
            return MetricResult(
                metric=self.name,
                case_id=case.case_id,
                score=0.0,
                details={"error": "no formulation data available"},
            )

        formulation_text = ""
        if formulation:
            formulation_text = (
                formulation.case_summary if hasattr(formulation, "case_summary")
                else str(formulation)
            )
        elif response:
            markdown = response.markdown if hasattr(response, "markdown") else str(response)
            sections = self._extract_sections(markdown)
            for key in sections:
                if "clinical formulation" in key or "formulation" in key:
                    formulation_text = sections[key]
                    break

        if not formulation_text:
            return MetricResult(
                metric=self.name,
                case_id=case.case_id,
                score=0.0,
                details={"error": "could not extract formulation text"},
            )

        text_lower = formulation_text.lower()
        axes: dict[str, float] = {}

        axis_scores = {
            "tentativeness": self._score_tentativeness(text_lower),
            "evidence-grounded": self._score_evidence_grounded(text_lower),
            "specificity": self._score_specificity(text_lower, case),
            "alternative-considerations": self._score_alternatives(text_lower),
            "confidence-appropriate": self._score_confidence(text_lower),
        }
        axes.update(axis_scores)

        overall = sum(axis_scores.values()) / max(len(axis_scores), 1)

        return MetricResult(
            metric=self.name,
            case_id=case.case_id,
            score=round(overall, 4),
            details={
                "axes": axes,
                "formulation_excerpt": formulation_text[:500],
                "usable_formulation": bool(formulation_text),
            },
        )

    def _score_tentativeness(self, text: str) -> float:
        score = 1.0
        for pattern in self._DIAGNOSIS_PATTERNS:
            if pattern.search(text):
                score -= 0.25
        tentative_count = sum(1 for w in self._TENTATIVE_WORDS if w in text)
        score = min(1.0, score + tentative_count * 0.1)
        return max(0.0, score)

    def _score_evidence_grounded(self, text: str) -> float:
        has_citation = bool(re.search(r"\(\w+\s+et\s+al|\(\d{4}\)|\[\d+\]", text))
        has_evidence_ref = any(
            phrase in text for phrase in
            ["evidence", "research", "literature", "study", "studies", "findings"]
        )
        score = 0.0
        if has_citation:
            score += 0.6
        if has_evidence_ref:
            score += 0.4
        return min(1.0, score)

    def _score_specificity(self, text: str, case: BenchmarkCase) -> float:
        topics = [t.lower() for t in case.ground_truth.expected_topics]
        if not topics:
            return 0.5
        topic_words = set()
        for t in topics:
            for w in t.split():
                if len(w) > 3:
                    topic_words.add(w)
        matches = sum(1 for w in topic_words if w in text)
        ratio = matches / max(len(topic_words), 1)
        return min(1.0, ratio * 1.5)

    def _score_alternatives(self, text: str) -> float:
        indicators = [
            "alternative", "differential", "uncertainty", "uncertain",
            "cannot rule out", "may also", "other possible",
            "alternative explanation", "differential consideration",
        ]
        count = sum(1 for i in indicators if i in text)
        return min(1.0, count * 0.25)

    def _score_confidence(self, text: str) -> float:
        has_confidence = any(
            phrase in text for phrase in
            ["confidence", "tentative", "preliminary", "based on available"]
        )
        has_overconfident = bool(re.search(
            r"\b(definitely|certainly|undoubtedly|guaranteed)\b",
            text, re.IGNORECASE,
        ))
        score = 0.5
        if has_confidence:
            score += 0.3
        if has_overconfident:
            score -= 0.3
        return max(0.0, min(1.0, score))

    @staticmethod
    def _extract_sections(markdown: str) -> dict[str, str]:
        sections: dict[str, str] = {}
        current_header: str | None = None
        current_lines: list[str] = []
        header_re = re.compile(r"^##\s+\d*\.?\s*(.+)$", re.MULTILINE)

        for line in markdown.split("\n"):
            m = header_re.match(line)
            if m:
                if current_header:
                    sections[current_header.strip().lower()] = "\n".join(current_lines).strip()
                current_header = m.group(1).strip()
                current_lines = []
            elif current_header:
                current_lines.append(line)

        if current_header:
            sections[current_header.strip().lower()] = "\n".join(current_lines).strip()
        return sections
