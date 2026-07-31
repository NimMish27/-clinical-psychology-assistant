from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from clinical.framework_evaluation.benchmarks import BenchmarkCase, get_all_cases
from clinical.framework_evaluation.metrics.base import FrameworkMetricResult
from clinical.frameworks.models import FrameworkResult


@dataclass
class FrameworkValidation:
    """Result of the runner's fail-fast preconditions (JSON parse + schema)."""

    valid: bool = True
    json_valid: bool = True
    schema_errors: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class FrameworkOutputResult:
    """Evaluation output for a single framework against a single case."""

    framework_name: str
    result: FrameworkResult | None
    validation: FrameworkValidation
    metric_results: list[FrameworkMetricResult] = field(default_factory=list)

    def metrics_map(self) -> dict[str, float]:
        return {mr.metric: mr.score for mr in self.metric_results}


@dataclass
class FrameworkCaseResult:
    """Evaluation output for a single benchmark case across all frameworks."""

    case_id: str
    title: str
    frameworks: dict[str, FrameworkOutputResult] = field(default_factory=dict)
    case_level_metrics: list[FrameworkMetricResult] = field(default_factory=list)


@dataclass
class FrameworkEvaluationReport:
    """Aggregated results of a framework evaluation run."""

    cases: list[BenchmarkCase] = field(default_factory=lambda: get_all_cases())
    framework_names: list[str] = field(default_factory=list)
    framework_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    results: list[FrameworkCaseResult] = field(default_factory=list)

    def get_case(self, case_id: str) -> FrameworkCaseResult | None:
        return next((r for r in self.results if r.case_id == case_id), None)

    def framework_aggregates(self) -> dict[str, dict[str, Any]]:
        """Aggregate per-framework metric scores across all cases.

        Returns ``{framework_name: {metric: {mean, min, max, std, values}}}``.
        """
        scores: dict[str, dict[str, list[float]]] = {
            fw: {} for fw in self.framework_names
        }
        for case in self.results:
            for fw_name, fw_out in case.frameworks.items():
                for mr in fw_out.metric_results:
                    scores.setdefault(fw_name, {}).setdefault(mr.metric, []).append(mr.score)

        aggregates: dict[str, dict[str, Any]] = {}
        for fw_name, metrics in scores.items():
            aggregates[fw_name] = {
                metric: self._aggregate(metric_scores)
                for metric, metric_scores in metrics.items()
            }
        return aggregates

    def case_level_aggregates(self) -> dict[str, Any]:
        """Aggregate case-level metrics (e.g. framework_diversity) across cases."""
        scores: dict[str, list[float]] = {}
        for case in self.results:
            for mr in case.case_level_metrics:
                scores.setdefault(mr.metric, []).append(mr.score)
        return {
            metric: self._aggregate(metric_scores)
            for metric, metric_scores in scores.items()
        }

    @staticmethod
    def _aggregate(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0, "values": []}
        mean = sum(values) / len(values)
        std = 0.0
        if len(values) > 1:
            variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
            std = variance ** 0.5
        return {
            "mean": round(mean, 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "std": round(std, 4),
            "values": [round(v, 4) for v in values],
        }


def format_framework_report(report: FrameworkEvaluationReport) -> str:
    """Render a human-readable side-by-side evaluation report."""
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("  FRAMEWORK EVALUATION REPORT")
    lines.append("=" * 78)
    lines.append("")

    if not report.results:
        lines.append("No evaluation results.")
        return "\n".join(lines)

    # Per-case side-by-side matrices.
    for case_result in report.results:
        lines.append(f"┌─ {case_result.title} ({case_result.case_id})")
        lines.append(f"│  Frameworks: {', '.join(report.framework_names)}")
        lines.append("│")
        header = f"{'Metric':<28}" + "".join(f"{fw:>10}" for fw in report.framework_names)
        lines.append("│  " + header)
        lines.append("│  " + "-" * (28 + 10 * len(report.framework_names)))

        metric_names: list[str] = []
        for fw_out in case_result.frameworks.values():
            for mr in fw_out.metric_results:
                if mr.metric not in metric_names:
                    metric_names.append(mr.metric)

        for metric in metric_names:
            row = [f"{metric:<28}"]
            for fw_name in report.framework_names:
                fw_out = case_result.frameworks.get(fw_name)
                if fw_out is None:
                    row.append(f"{'-':>8}")
                elif not fw_out.validation.valid:
                    row.append(f"{'INVALID':>8}")
                else:
                    score = fw_out.metrics_map().get(metric)
                    row.append(f"{score:>8.4f}" if score is not None else f"{'-':>8}")
            lines.append("│  " + "  ".join(row))

        for mr in case_result.case_level_metrics:
            lines.append(f"│  {mr.metric}: {mr.score:.4f}")

        lines.append("│")
        for fw_name, fw_out in case_result.frameworks.items():
            if fw_out.validation.valid:
                continue
            err = fw_out.validation.error or "failed preconditions"
            lines.append(f"│  ⚠ {fw_name}: INVALID — {err}")
        lines.append("└─")
        lines.append("")

    # Aggregate tables.
    lines.append("-" * 78)
    lines.append("  AGGREGATE — PER-FRAMEWORK METRICS (mean across cases)")
    lines.append("-" * 78)
    lines.append("")
    agg = report.framework_aggregates()
    metric_names = sorted({m for fw in agg.values() for m in fw})
    header = f"{'Metric':<30}" + "".join(f"{fw:>12}" for fw in report.framework_names)
    lines.append(header)
    lines.append("-" * 78)
    for metric in metric_names:
        row = f"{metric:<30}"
        for fw in report.framework_names:
            value = agg.get(fw, {}).get(metric, {}).get("mean")
            row += f"{value:>12.4f}" if value is not None else f"{'-':>12}"
        lines.append(row)
    lines.append("")

    if report.case_level_aggregates():
        lines.append("  AGGREGATE — CASE-LEVEL METRICS (mean across cases)")
        lines.append("")
        for metric, details in report.case_level_aggregates().items():
            lines.append(
                f"  {metric:<30} mean={details['mean']:.4f}  "
                f"min={details['min']:.4f}  max={details['max']:.4f}  "
                f"std={details['std']:.4f}"
            )
        lines.append("")

    return "\n".join(lines)


def format_json_framework_report(report: FrameworkEvaluationReport) -> dict[str, Any]:
    """Serialize a framework evaluation report to a JSON-able dict."""
    cases_payload: list[dict[str, Any]] = []
    for case_result in report.results:
        frameworks_payload: dict[str, Any] = {}
        for fw_name, fw_out in case_result.frameworks.items():
            frameworks_payload[fw_name] = {
                "valid": fw_out.validation.valid,
                "validation_error": fw_out.validation.error,
                "json_valid": fw_out.validation.json_valid,
                "schema_errors": fw_out.validation.schema_errors,
                "confidence": fw_out.result.confidence if fw_out.result else None,
                "metrics": {
                    mr.metric: {"score": mr.score, "details": mr.details}
                    for mr in fw_out.metric_results
                },
            }
        cases_payload.append({
            "case_id": case_result.case_id,
            "title": case_result.title,
            "frameworks": frameworks_payload,
            "case_level_metrics": {
                mr.metric: {"score": mr.score, "details": mr.details}
                for mr in case_result.case_level_metrics
            },
        })

    return {
        "framework_names": report.framework_names,
        "framework_metadata": report.framework_metadata,
        "framework_aggregates": report.framework_aggregates(),
        "case_level_aggregates": report.case_level_aggregates(),
        "cases": cases_payload,
    }
