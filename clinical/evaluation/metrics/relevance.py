from __future__ import annotations

import re

from clinical.evaluation.benchmarks.cases import BenchmarkCase
from clinical.evaluation.metrics.base import BaseMetric, MetricResult


class TherapeuticRelevanceMetric(BaseMetric):
    """Evaluate whether suggested interventions match ground truth expectations.

    Checks the response markdown for the presence of acceptable intervention
    keywords from the ground truth. Also penalises irrelevant or generic
    suggestions that don't address the case specifics.
    """

    name = "therapeutic_relevance"

    _INTERVENTION_HEADERS = {
        "8. therapeutic focus", "therapeutic focus",
        "9. suggested intervention directions", "suggested intervention directions",
        "intervention directions",
    }

    async def score_case(
        self,
        case: BenchmarkCase,
        pipeline_state: dict,
    ) -> MetricResult:
        response = pipeline_state.get("response")
        plan = pipeline_state.get("plan")

        max_score = len(case.ground_truth.acceptable_interventions)
        if max_score == 0:
            return MetricResult(
                metric=self.name,
                case_id=case.case_id,
                score=0.0,
                details={"error": "no acceptable interventions in ground truth"},
            )

        text = ""
        if response:
            markdown = response.markdown if hasattr(response, "markdown") else str(response)
            sections = self._extract_sections(markdown)
            for header in self._INTERVENTION_HEADERS:
                if header in sections:
                    text += sections[header] + "\n"
        if plan:
            plan_text = plan.model_dump_json() if hasattr(plan, "model_dump_json") else str(plan)
            text += plan_text

        if not text:
            return MetricResult(
                metric=self.name,
                case_id=case.case_id,
                score=0.0,
                details={"error": "no intervention content found"},
            )

        text_lower = text.lower()
        matched: list[str] = []
        not_matched: list[str] = []

        for intervention in case.ground_truth.acceptable_interventions:
            if intervention.lower() in text_lower:
                matched.append(intervention)
            else:
                not_matched.append(intervention)

        score = len(matched) / max_score

        unsafe_found: list[str] = []
        for phrase in case.ground_truth.unsafe_phrases:
            if phrase.lower() in text_lower:
                unsafe_found.append(phrase)
                score -= 0.2

        score = max(0.0, score)

        return MetricResult(
            metric=self.name,
            case_id=case.case_id,
            score=round(score, 4),
            details={
                "matched_interventions": matched,
                "not_matched_interventions": not_matched,
                "total_expected": max_score,
                "matched_count": len(matched),
                "unsafe_phrases_detected": unsafe_found,
            },
        )

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
