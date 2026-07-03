from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from clinical.evaluation.benchmarks.cases import ALL_CASES, BenchmarkCase


@dataclass
class EvaluationResult:
    case_id: str
    pipeline_elapsed_ms: float
    metric_results: list
    pipeline_state: dict = field(repr=False)
    error: str | None = None


@dataclass
class EvaluationReport:
    cases: list[BenchmarkCase] = field(default_factory=lambda: ALL_CASES)
    results: list[EvaluationResult] = field(default_factory=list)

    def aggregate(self) -> dict[str, Any]:
        """Compute aggregate scores across all cases for each metric."""
        metric_scores: dict[str, list[float]] = {}

        for result in self.results:
            for mr in result.metric_results:
                if mr.metric not in metric_scores:
                    metric_scores[mr.metric] = []
                metric_scores[mr.metric].append(mr.score)

        aggregates: dict[str, Any] = {}
        for metric, scores in metric_scores.items():
            if scores:
                aggregates[metric] = {
                    "mean": round(sum(scores) / len(scores), 4),
                    "min": round(min(scores), 4),
                    "max": round(max(scores), 4),
                    "std": round(self._std(scores), 4),
                    "values": [round(s, 4) for s in scores],
                }
            else:
                aggregates[metric] = {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0, "values": []}

        return aggregates

    def _std(self, values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        return variance ** 0.5


def format_report(report: EvaluationReport) -> str:
    """Generate a human-readable evaluation report string."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("  CLINICAL PSYCHOLOGY ASSISTANT — EVALUATION REPORT")
    lines.append("=" * 72)
    lines.append("")

    aggregates = report.aggregate()
    lines.append(f"{'Metric':<38} {'Mean':>8} {'Min':>8} {'Max':>8} {'Std':>8}")
    lines.append("-" * 72)
    for metric, agg in sorted(aggregates.items()):
        lines.append(
            f"{metric:<38} {agg['mean']:>8.4f} {agg['min']:>8.4f} "
            f"{agg['max']:>8.4f} {agg['std']:>8.4f}"
        )
    lines.append("")

    lines.append("-" * 72)
    lines.append("  PER-CASE DETAILS")
    lines.append("-" * 72)
    lines.append("")

    for result in report.results:
        case_obj = next((c for c in report.cases if c.case_id == result.case_id), None)
        title = case_obj.title if case_obj else result.case_id

        lines.append(f"  ┌─ {title} ({result.case_id})")
        if result.error:
            lines.append(f"  │  ERROR: {result.error}")
            continue
        lines.append(f"  │  Pipeline time: {result.pipeline_elapsed_ms:.0f} ms")
        lines.append(f"  │")
        for mr in result.metric_results:
            label = f"  │  {mr.metric}:"
            lines.append(f"  │    {mr.metric:<38} {mr.score:.4f}")
        lines.append(f"  └─")
        lines.append("")

    return "\n".join(lines)


def format_json_report(report: EvaluationReport) -> dict[str, Any]:
    """Generate a JSON-serialisable evaluation report."""
    aggregates = report.aggregate()
    cases_list = []
    for result in report.results:
        case_obj = next((c for c in report.cases if c.case_id == result.case_id), None)
        cases_list.append({
            "case_id": result.case_id,
            "title": case_obj.title if case_obj else result.case_id,
            "pipeline_elapsed_ms": result.pipeline_elapsed_ms,
            "error": result.error,
            "metrics": {
                mr.metric: {
                    "score": mr.score,
                    "details": mr.details,
                }
                for mr in result.metric_results
            },
        })

    return {
        "summary": aggregates,
        "cases": cases_list,
    }
