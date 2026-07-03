from __future__ import annotations

import re

from clinical.evaluation.benchmarks.cases import BenchmarkCase
from clinical.evaluation.metrics.base import BaseMetric, MetricResult


class MissingInfoDetectionMetric(BaseMetric):
    """Evaluate what fraction of expected missing-info items the pipeline identifies.

    Compares the ground truth missing information against:
    1. The ``missing_info`` module output (structured list)
    2. The "MISSING INFORMATION" section of the response markdown

    Score = true_positives / (true_positives + false_negatives)
    Precision = true_positives / (true_positives + false_positives)
    """

    name = "missing_info_detection"

    _MISSING_INFO_HEADERS = {
        "6. missing information", "missing information",
        "missing info",
    }

    async def score_case(
        self,
        case: BenchmarkCase,
        pipeline_state: dict,
    ) -> MetricResult:
        expected = case.ground_truth.expected_missing_info_items
        if not expected:
            return MetricResult(
                metric=self.name,
                case_id=case.case_id,
                score=1.0,
                details={"note": "no missing info items expected"},
            )

        detected_phrases: list[str] = []

        missing_info = pipeline_state.get("missing_info")
        if missing_info:
            if hasattr(missing_info, "missing_information"):
                for item in missing_info.missing_information:
                    if hasattr(item, "info_gap") and item.info_gap:
                        detected_phrases.append(item.info_gap.lower())
                    if hasattr(item, "clinical_relevance") and item.clinical_relevance:
                        detected_phrases.append(item.clinical_relevance.lower())

        response = pipeline_state.get("response")
        if response:
            markdown = response.markdown if hasattr(response, "markdown") else str(response)
            sections = self._extract_sections(markdown)
            for header in self._MISSING_INFO_HEADERS:
                if header in sections:
                    detected_phrases.append(sections[header].lower())

        combined_text = " ".join(detected_phrases)

        true_positives: list[str] = []
        false_negatives: list[str] = []

        for item in expected:
            keywords = item.lower().split()
            if any(kw in combined_text for kw in keywords):
                true_positives.append(item)
            else:
                false_negatives.append(item)

        recall = len(true_positives) / max(len(expected), 1)
        f1 = 2 * recall / (1 + recall) if recall > 0 else 0.0

        return MetricResult(
            metric=self.name,
            case_id=case.case_id,
            score=round(f1, 4),
            details={
                "recall": round(recall, 4),
                "true_positives": true_positives,
                "false_negatives": false_negatives,
                "expected_count": len(expected),
                "detected_count": len(true_positives),
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
