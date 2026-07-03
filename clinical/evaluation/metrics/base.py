from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from clinical.evaluation.benchmarks.cases import BenchmarkCase


@dataclass
class MetricResult:
    metric: str
    case_id: str
    score: float
    details: dict[str, Any] = field(default_factory=dict)
    passed: bool | None = None


class BaseMetric:
    name: str = "base"

    async def score_case(
        self,
        case: BenchmarkCase,
        pipeline_state: dict,
    ) -> MetricResult:
        raise NotImplementedError
