from __future__ import annotations

import re

from clinical.evaluation.benchmarks.cases import BenchmarkCase
from clinical.evaluation.metrics.base import BaseMetric, MetricResult


class ClinicalHelpfulnessMetric(BaseMetric):
    """Evaluate overall clinical helpfulness against a rubric.

    Three axes scored 0-1 each:
      - Validation: does the response acknowledge and validate the client's
        experience using expected empathetic language?
      - Safety: does the response address safety concerns appropriately?
      - Practicality: does the response provide actionable, specific guidance?

    Each axis checks rubric keywords from ground truth and general
    clinical language markers. The overall score is the average.
    """

    name = "clinical_helpfulness"

    _VALIDATION_MARKERS = [
        "validates", "acknowledges", "normalises", "validates the client",
        "recognises", "sensitive to", "respectful",
        "it is important to recognise", "the client's experience",
        "the client may be feeling", "it is understandable",
    ]

    _SAFETY_MARKERS = [
        "safety", "risk assessment", "risk", "crisis",
        "emergency", "urgent", "referral", "signpost",
        "if you are concerned", "seek help", "support services",
        "helpline", "crisis team", "gp", "general practitioner",
    ]

    _PRACTICALITY_MARKERS = [
        "specific", "suggestion", "recommend", "consider",
        "may wish to", "could explore", "might consider",
        "therapeutic approach", "treatment option",
        "referral to", "liaise with", "collaborate with",
    ]

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
        text_lower = markdown.lower()

        rubric = case.ground_truth.clinical_helpfulness_rubric

        validation_score = self._score_axis(
            text_lower,
            rubric.get("validation", []),
            self._VALIDATION_MARKERS,
            "validation",
        )
        safety_score = self._score_axis(
            text_lower,
            rubric.get("safety", []),
            self._SAFETY_MARKERS,
            "safety",
        )
        practicality_score = self._score_axis(
            text_lower,
            rubric.get("practicality", []),
            self._PRACTICALITY_MARKERS,
            "practicality",
        )

        overall = (validation_score + safety_score + practicality_score) / 3.0

        axes = {
            "validation": round(validation_score, 4),
            "safety": round(safety_score, 4),
            "practicality": round(practicality_score, 4),
        }

        return MetricResult(
            metric=self.name,
            case_id=case.case_id,
            score=round(overall, 4),
            details={
                "axes": axes,
                "rubric_matches": {
                    "validation": self._find_matches(text_lower, rubric.get("validation", [])),
                    "safety": self._find_matches(text_lower, rubric.get("safety", [])),
                    "practicality": self._find_matches(text_lower, rubric.get("practicality", [])),
                },
                "response_length": len(markdown),
            },
        )

    def _score_axis(
        self,
        text: str,
        rubric_phrases: list[str],
        general_markers: list[str],
        axis: str,
    ) -> float:
        if not rubric_phrases and not general_markers:
            return 0.5

        score = 0.0
        max_score = 0

        # Match rubric phrases via keyword overlap
        for phrase in rubric_phrases:
            max_score += 1
            keywords = [w for w in phrase.lower().split() if len(w) > 3]
            if not keywords:
                if phrase.lower() in text:
                    score += 1.0
            else:
                word_matches = sum(1 for kw in keywords if kw in text)
                if word_matches / max(len(keywords), 1) >= 0.5:
                    score += 1.0

        # General marker matches
        general_hits = sum(1 for m in general_markers if m in text)
        general_max = min(len(general_markers), 5)
        max_score += general_max
        score += min(general_hits, general_max) * (1.0 / max(general_max, 1))

        return min(1.0, score / max(max_score, 1)) if max_score > 0 else 0.5

    @staticmethod
    def _find_matches(text: str, phrases: list[str]) -> list[str]:
        result = []
        for p in phrases:
            keywords = [w for w in p.lower().split() if len(w) > 3]
            if not keywords:
                if p.lower() in text:
                    result.append(p)
            else:
                word_matches = sum(1 for kw in keywords if kw in text)
                if word_matches / max(len(keywords), 1) >= 0.5:
                    result.append(p)
        return result
