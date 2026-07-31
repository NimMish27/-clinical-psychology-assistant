from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from clinical.framework_evaluation.benchmarks import BenchmarkCase
from clinical.frameworks.models import FrameworkResult


@dataclass
class FrameworkMetricResult:
    """Score for a single metric against a single framework output (or case)."""

    metric: str
    framework_name: str
    case_id: str
    score: float
    details: dict[str, Any] = field(default_factory=dict)
    passed: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "framework_name": self.framework_name,
            "case_id": self.case_id,
            "score": self.score,
            "details": self.details,
        }


class BaseFrameworkMetric:
    """Base class for framework-evaluation metrics.

    A metric either scores a single framework output (``score_framework``)
    or scores a whole case across all frameworks (``score_case`` — used by
    case-level metrics such as framework diversity). At least one of the two
    methods must be implemented.
    """

    name: str = "base"
    scope: str = "framework"  # "framework" or "case"

    async def score_framework(
        self,
        case: BenchmarkCase,
        result: FrameworkResult,
        raw_output: str | None,
    ) -> FrameworkMetricResult:
        """Score a single framework's output for a case.

        Raises NotImplementedError unless overridden.
        """
        raise NotImplementedError

    async def score_case(
        self,
        case: BenchmarkCase,
        results: dict[str, FrameworkResult],
        raw_outputs: dict[str, str | None],
    ) -> FrameworkMetricResult:
        """Score a case across all framework outputs.

        Raises NotImplementedError unless overridden.
        """
        raise NotImplementedError
