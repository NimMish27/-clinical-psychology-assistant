from __future__ import annotations

from typing import Any

from clinical.framework_evaluation.benchmarks import BenchmarkCase
from clinical.framework_evaluation.metrics.base import BaseFrameworkMetric, FrameworkMetricResult
from clinical.frameworks.base import BaseFramework
from clinical.frameworks.models import FrameworkResult


def _try_parse_json(raw: str | None) -> dict[str, Any] | None:
    """Parse raw output into a dict, or None if it is not valid JSON."""
    if not raw:
        return None
    try:
        data = BaseFramework._parse_json_response(raw)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


class SchemaConformanceMetric(BaseFrameworkMetric):
    """Score field-level schema conformance of the raw JSON output.

    The runner treats JSON parsing and basic schema validity as a binary
    precondition (fail fast). This metric goes a step further and scores
    how fully the raw output conforms to the FrameworkResult schema,
    weighting the mandatory fields over the recommended ones.
    """

    name = "schema_conformance"
    scope = "framework"

    _REQUIRED_KEYS: tuple[tuple[str, str], ...] = (
        ("conceptualisation", "str"),
        ("confidence", "number"),
    )

    _RECOMMENDED_KEYS: tuple[tuple[str, str], ...] = (
        ("key_concepts", "list"),
        ("therapeutic_approaches", "list"),
        ("strengths", "list"),
        ("limitations", "list"),
        ("evidence_alignment", "str"),
    )

    async def score_framework(
        self,
        case: BenchmarkCase,
        result: FrameworkResult,
        raw_output: str | None,
    ) -> FrameworkMetricResult:
        data = _try_parse_json(raw_output)
        if data is None:
            return FrameworkMetricResult(
                metric=self.name,
                framework_name=result.framework_name,
                case_id=case.case_id,
                score=0.0,
                details={"error": "raw output is not valid JSON"},
            )

        checks: dict[str, bool] = {}
        for key, expected in self._REQUIRED_KEYS:
            checks[f"required:{key}"] = self._check_field(data, key, expected)

        for key, expected in self._RECOMMENDED_KEYS:
            checks[f"recommended:{key}"] = self._check_field(data, key, expected)

        total_weight = 0.0
        score = 0.0
        for key, passed in checks.items():
            weight = 2.0 if key.startswith("required:") else 1.0
            total_weight += weight
            if passed:
                score += weight

        overall = score / total_weight if total_weight else 0.0

        return FrameworkMetricResult(
            metric=self.name,
            framework_name=result.framework_name,
            case_id=case.case_id,
            score=round(overall, 4),
            details={
                "checks": checks,
                "missing_keys": [
                    k for k, ok in checks.items() if not ok
                ],
            },
        )

    @staticmethod
    def _check_field(data: dict[str, Any], key: str, expected: str) -> bool:
        if key not in data:
            return False
        value = data[key]
        if expected == "str":
            return isinstance(value, str)
        if expected == "list":
            return isinstance(value, list)
        if expected == "number":
            try:
                num = float(value)
            except (TypeError, ValueError):
                return False
            return 0.0 <= num <= 1.0
        return False


class StructuralCompletenessMetric(BaseFrameworkMetric):
    """Score the substantive completeness of a framework's response.

    Measures whether the framework output filled out every field of the
    FrameworkResult with meaningful content, weighted toward the
    conceptualization (the core of the analysis).
    """

    name = "structural_completeness"
    scope = "framework"

    _MIN_CONCEPTUALIZATION = 100
    _MIN_FIELD = 20

    async def score_framework(
        self,
        case: BenchmarkCase,
        result: FrameworkResult,
        raw_output: str | None,
    ) -> FrameworkMetricResult:
        checks: dict[str, float] = {
            "conceptualization": self._length_score(
                result.conceptualization, self._MIN_CONCEPTUALIZATION
            ),
            "key_concepts": self._list_score(result.key_concepts),
            "therapeutic_approaches": self._list_score(result.therapeutic_approaches),
            "strengths": self._list_score(result.strengths),
            "limitations": self._list_score(result.limitations),
            "evidence_alignment": self._length_score(result.evidence_alignment, self._MIN_FIELD),
            "confidence_reported": 1.0 if result.confidence > 0.0 else 0.0,
        }

        weights = {
            "conceptualization": 3.0,
            "key_concepts": 1.0,
            "therapeutic_approaches": 1.0,
            "strengths": 1.0,
            "limitations": 1.0,
            "evidence_alignment": 1.0,
            "confidence_reported": 1.0,
        }

        total = sum(weights.values())
        score = sum(checks[k] * weights[k] for k in checks) / total

        return FrameworkMetricResult(
            metric=self.name,
            framework_name=result.framework_name,
            case_id=case.case_id,
            score=round(score, 4),
            details={
                "checks": {k: round(v, 4) for k, v in checks.items()},
                "conceptualization_length": len(result.conceptualization),
                "field_counts": {
                    "key_concepts": len(result.key_concepts),
                    "therapeutic_approaches": len(result.therapeutic_approaches),
                    "strengths": len(result.strengths),
                    "limitations": len(result.limitations),
                },
            },
        )

    @staticmethod
    def _list_score(values: list[str], min_items: int = 1) -> float:
        if not values:
            return 0.0
        non_empty = sum(1 for v in values if len(v.strip()) >= 3)
        if non_empty < min_items:
            return 0.0
        return min(1.0, non_empty / min_items)

    @staticmethod
    def _length_score(text: str, minimum: int) -> float:
        length = len(text.strip())
        if length <= 0:
            return 0.0
        return min(1.0, length / minimum)
