from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app_logging.logger import get_logger
from clinical.framework_evaluation.report import FrameworkEvaluationReport

_log = get_logger(__name__)


@dataclass
class BaselineCaseEntry:
    """Stored per-case framework scores from a baseline run."""

    case_id: str
    title: str
    frameworks: dict[str, dict[str, float]] = field(default_factory=dict)
    case_level_metrics: dict[str, float] = field(default_factory=dict)

    @staticmethod
    def from_report_case(case_result) -> BaselineCaseEntry:
        frameworks: dict[str, dict[str, float]] = {}
        for fw_name, fw_out in case_result.frameworks.items():
            if fw_out.validation.valid:
                frameworks[fw_name] = fw_out.metrics_map()
        return BaselineCaseEntry(
            case_id=case_result.case_id,
            title=case_result.title,
            frameworks=frameworks,
            case_level_metrics={
                mr.metric: mr.score for mr in case_result.case_level_metrics
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "title": self.title,
            "frameworks": self.frameworks,
            "case_level_metrics": self.case_level_metrics,
        }


@dataclass
class FrameworkBaseline:
    """Stored baseline of framework evaluation scores."""

    framework_names: list[str] = field(default_factory=list)
    cases: list[BaselineCaseEntry] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_report(
        cls,
        report: FrameworkEvaluationReport,
        meta: dict[str, Any] | None = None,
    ) -> FrameworkBaseline:
        return cls(
            framework_names=list(report.framework_names),
            cases=[BaselineCaseEntry.from_report_case(r) for r in report.results],
            meta={
                "created_at": datetime.now(UTC).isoformat(),
                **(meta or {}),
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "framework_names": self.framework_names,
            "cases": [c.to_dict() for c in self.cases],
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FrameworkBaseline:
        return cls(
            framework_names=list(data.get("framework_names", [])),
            cases=[
                BaselineCaseEntry(
                    case_id=c.get("case_id", ""),
                    title=c.get("title", ""),
                    frameworks=c.get("frameworks", {}),
                    case_level_metrics=c.get("case_level_metrics", {}),
                )
                for c in data.get("cases", [])
            ],
            meta=data.get("meta", {}),
        )


def save_baseline(report: FrameworkEvaluationReport, path: str | Path) -> Path:
    """Persist a framework evaluation report as a JSON baseline."""
    baseline = FrameworkBaseline.from_report(report)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(baseline.to_dict(), indent=2), encoding="utf-8")
    _log.info("framework_baseline.saved", path=str(target))
    return target


def load_baseline(path: str | Path) -> FrameworkBaseline:
    """Load a previously saved framework baseline from JSON."""
    target = Path(path)
    data = json.loads(target.read_text(encoding="utf-8"))
    _log.info("framework_baseline.loaded", path=str(target))
    return FrameworkBaseline.from_dict(data)


@dataclass
class RegressionDelta:
    """Score change for a single framework metric on a single case."""

    framework_name: str
    metric: str
    case_id: str
    baseline_score: float
    new_score: float
    delta: float
    regressed: bool


@dataclass
class RegressionReport:
    """Comparison of a new evaluation run against a stored baseline."""

    deltas: list[RegressionDelta] = field(default_factory=list)
    tolerance: float = 0.05

    @property
    def regressions(self) -> list[RegressionDelta]:
        return [d for d in self.deltas if d.regressed]

    @property
    def improvements(self) -> list[RegressionDelta]:
        return [d for d in self.deltas if not d.regressed and d.delta > self.tolerance]

    def summary(self) -> dict[str, Any]:
        return {
            "total_comparisons": len(self.deltas),
            "regressions": len(self.regressions),
            "improvements": len(self.improvements),
            "unchanged": len(self.deltas) - len(self.regressions) - len(self.improvements),
            "worst_delta": min((d.delta for d in self.deltas), default=0.0),
        }


def compare(
    report: FrameworkEvaluationReport,
    baseline: FrameworkBaseline,
    tolerance: float = 0.05,
) -> RegressionReport:
    """Compare a new evaluation report against a stored baseline.

    A regression is flagged when a new score falls below the baseline
    score by more than ``tolerance``.
    """
    baseline_by_case = {c.case_id: c for c in baseline.cases}
    deltas: list[RegressionDelta] = []

    for case_result in report.results:
        baseline_case = baseline_by_case.get(case_result.case_id)
        if baseline_case is None:
            continue

        for fw_name, fw_out in case_result.frameworks.items():
            baseline_metrics = baseline_case.frameworks.get(fw_name, {})
            for metric, new_score in fw_out.metrics_map().items():
                if metric not in baseline_metrics:
                    continue
                baseline_score = baseline_metrics[metric]
                delta = new_score - baseline_score
                deltas.append(RegressionDelta(
                    framework_name=fw_name,
                    metric=metric,
                    case_id=case_result.case_id,
                    baseline_score=baseline_score,
                    new_score=new_score,
                    delta=round(delta, 4),
                    regressed=delta < -tolerance,
                ))

        for mr in case_result.case_level_metrics:
            baseline_score = baseline_case.case_level_metrics.get(mr.metric)
            if baseline_score is None:
                continue
            delta = mr.score - baseline_score
            deltas.append(RegressionDelta(
                framework_name="all_frameworks",
                metric=mr.metric,
                case_id=case_result.case_id,
                baseline_score=baseline_score,
                new_score=mr.score,
                delta=round(delta, 4),
                regressed=delta < -tolerance,
            ))

    return RegressionReport(deltas=deltas, tolerance=tolerance)


def format_regression_report(report: RegressionReport) -> str:
    """Render a human-readable regression comparison."""
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("  FRAMEWORK REGRESSION REPORT")
    lines.append("=" * 78)
    lines.append(f"  Tolerance: ±{report.tolerance:.2f}")
    summary = report.summary()
    lines.append(
        f"  Compared: {summary['total_comparisons']} | "
        f"Regressions: {summary['regressions']} | "
        f"Improvements: {summary['improvements']} | "
        f"Unchanged: {summary['unchanged']}"
    )
    lines.append("-" * 78)
    lines.append("")

    if not report.regressions:
        lines.append("No regressions detected.")
    else:
        lines.append("REGRESSIONS:")
        lines.append(f"{'Framework':<16}{'Metric':<28}{'Case':<20}{'Base':>8}{'New':>8}{'Δ':>8}")
        for d in report.regressions:
            lines.append(
                f"{d.framework_name:<16}{d.metric:<28}{d.case_id:<20}"
                f"{d.baseline_score:>8.4f}{d.new_score:>8.4f}{d.delta:>8.4f}"
            )
        lines.append("")

    improvements = report.improvements
    if improvements:
        lines.append("IMPROVEMENTS:")
        for d in improvements:
            lines.append(
                f"  {d.framework_name:<14} {d.metric:<26} "
                f"{d.case_id:<20} Δ={d.delta:+.4f}"
            )
        lines.append("")

    return "\n".join(lines)
