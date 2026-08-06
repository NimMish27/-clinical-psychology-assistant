"""
scripts/run_cbt_evaluation.py
──────────────────────────────
Run the CBT framework evaluation suite against benchmark cases.

Evaluates the CBT framework plugin's output across four CBT-specific
metrics as well as the 7 general evaluation metrics.

Usage:
    python scripts/run_cbt_evaluation.py                            # all 20 CBT cases
    python scripts/run_cbt_evaluation.py --case cbt_depression_triad_001  # single case
    python scripts/run_cbt_evaluation.py --json                     # JSON output
    python scripts/run_cbt_evaluation.py --output results.json      # save to file
    python scripts/run_cbt_evaluation.py --general-metrics           # include all 7 general metrics
    python scripts/run_cbt_evaluation.py --list-cases               # list available cases

The script:
  1. Loads the 20 CBT benchmark cases
  2. Runs the full LangGraph pipeline (which includes framework analysis)
  3. Computes CBT-specific metrics:
       - cbt_retrieval_accuracy: CBT literature retrieval precision
       - cbt_conceptualization_quality: CBT model fidelity
       - cbt_citation_quality: CBT citation relevance and accuracy
       - cbt_never_diagnose: safety check for diagnostic labels
  4. Optionally runs all 7 general evaluation metrics
  5. Outputs a formatted report or JSON

Requires: all pipeline dependencies (Ollama, ChromaDB, embedding model).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import os; os.environ["LOG_FORMAT"] = "console"

from app_logging.logger import setup_logging, get_logger
from clinical.evaluation import (
    ALL_CBT_CASES,
    BenchmarkCase,
    CBTConceptualizationQualityMetric,
    CBTCitationQualityMetric,
    CBTNeverDiagnoseMetric,
    CBTRetrievalAccuracyMetric,
)
from clinical.evaluation.benchmarks.cbt_cases import ALL_CBT_CASES as CBT_CASES
from clinical.evaluation.metrics.base import MetricResult
from clinical.evaluation.report import EvaluationReport, EvaluationResult, format_report, format_json_report

setup_logging(level="INFO", log_format="console")
_log = get_logger(__name__)

# ── Default CBT metric set ────────────────────────────────────────────

CBT_METRICS = [
    CBTRetrievalAccuracyMetric(),
    CBTConceptualizationQualityMetric(),
    CBTCitationQualityMetric(),
    CBTNeverDiagnoseMetric(),
]

# ── General metrics (optional) ────────────────────────────────────────

GENERAL_METRICS = [
    # Imported lazily inside the run function to avoid circular imports
]


def _get_general_metrics():
    if not GENERAL_METRICS:
        from clinical.evaluation.metrics.citation import CitationAccuracyMetric
        from clinical.evaluation.metrics.formulation import FormulationQualityMetric
        from clinical.evaluation.metrics.hallucination import HallucinationRateMetric
        from clinical.evaluation.metrics.helpfulness import ClinicalHelpfulnessMetric
        from clinical.evaluation.metrics.missing_info import MissingInfoDetectionMetric
        from clinical.evaluation.metrics.retrieval import RetrievalPrecisionMetric
        from clinical.evaluation.metrics.relevance import TherapeuticRelevanceMetric

        GENERAL_METRICS.extend([
            RetrievalPrecisionMetric(),
            CitationAccuracyMetric(),
            FormulationQualityMetric(),
            TherapeuticRelevanceMetric(),
            MissingInfoDetectionMetric(),
            HallucinationRateMetric(),
            ClinicalHelpfulnessMetric(),
        ])
    return GENERAL_METRICS


def list_cases() -> None:
    """Print available CBT benchmark cases."""
    print("CBT Evaluation — Available Benchmark Cases\n")
    print(f"{'Case ID':<42} {'Title'}")
    print("-" * 100)
    for c in CBT_CASES:
        print(f"{c.case_id:<42} {c.title}")
    print(f"\nTotal: {len(CBT_CASES)} cases")


async def evaluate_cbt(
    cases: list[BenchmarkCase] | None = None,
    include_general: bool = False,
) -> EvaluationReport:
    """Run the CBT evaluation suite.

    Args:
        cases: Cases to evaluate. Defaults to ALL_CBT_CASES.
        include_general: Whether to also run all 7 general metrics.

    Returns:
        EvaluationReport with per-case and aggregate scores.
    """
    cases = cases or CBT_CASES
    metrics = list(CBT_METRICS)
    if include_general:
        metrics.extend(_get_general_metrics())

    results: list[EvaluationResult] = []

    for case in cases:
        _log.info("cbt_eval.running_case", case_id=case.case_id, title=case.title)
        t_start = time.perf_counter()

        try:
            from clinical.graph.graph import run_pipeline

            state = await run_pipeline(case.clinical_text)
        except Exception as exc:
            _log.error("cbt_eval.case_failed", case_id=case.case_id, error=str(exc))
            results.append(EvaluationResult(
                case_id=case.case_id,
                pipeline_elapsed_ms=0.0,
                metric_results=[],
                pipeline_state={},
                error=str(exc),
            ))
            continue

        pipeline_ms = (time.perf_counter() - t_start) * 1000

        metric_results: list[MetricResult] = []
        for metric in metrics:
            try:
                mr = await metric.score_case(case, state)
                metric_results.append(mr)
            except Exception as exc:
                _log.error(
                    "cbt_eval.metric_failed",
                    metric=metric.name,
                    case_id=case.case_id,
                    error=str(exc),
                )
                metric_results.append(MetricResult(
                    metric=metric.name,
                    case_id=case.case_id,
                    score=0.0,
                    details={"error": str(exc)},
                ))

        results.append(EvaluationResult(
            case_id=case.case_id,
            pipeline_elapsed_ms=round(pipeline_ms, 2),
            metric_results=metric_results,
            pipeline_state=state,
        ))

    return EvaluationReport(cases=cases, results=results)


def format_cbt_report(report: EvaluationReport) -> str:
    """Generate a formatted CBT evaluation report."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("  CBT FRAMEWORK EVALUATION REPORT")
    lines.append("=" * 72)
    lines.append("")

    aggregates = report.aggregate()
    lines.append(f"{'Metric':<42} {'Mean':>8} {'Min':>8} {'Max':>8} {'Std':>8}")
    lines.append("-" * 72)
    for metric, agg in sorted(aggregates.items()):
        lines.append(
            f"{metric:<42} {agg['mean']:>8.4f} {agg['min']:>8.4f} "
            f"{agg['max']:>8.4f} {agg['std']:>8.4f}"
        )
    lines.append("")

    # Never-diagnose summary
    nd_values = aggregates.get("cbt_never_diagnose", {}).get("values", [])
    if nd_values:
        violations = sum(1 for v in nd_values if v < 1.0)
        lines.append(f"Never-Diagnose Compliance: {len(nd_values) - violations}/{len(nd_values)} cases clean")
        lines.append("")

    lines.append("-" * 72)
    lines.append("  PER-CASE DETAILS")
    lines.append("-" * 72)
    lines.append("")

    for result in report.results:
        case_obj = next((c for c in report.cases if c.case_id == result.case_id), None)
        title = case_obj.title if case_obj else result.case_id

        lines.append(f"  Case: {title} ({result.case_id})")
        if result.error:
            lines.append(f"  ERROR: {result.error}")
            lines.append("")
            continue
        lines.append(f"  Pipeline: {result.pipeline_elapsed_ms:.0f} ms")
        for mr in result.metric_results:
            label = mr.metric
            score_display = f"{mr.score:.4f}"
            if mr.passed is not None:
                status = "PASS" if mr.passed else "FAIL"
                lines.append(f"    {label:<40} {score_display} [{status}]")
            else:
                lines.append(f"    {label:<40} {score_display}")
        lines.append("")

    return "\n".join(lines)


def format_cbt_json_report(report: EvaluationReport) -> dict:
    """Generate a JSON-serialisable CBT evaluation report."""
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
                    "passed": mr.passed,
                    "details": mr.details,
                }
                for mr in result.metric_results
            },
        })

    never_diagnose_agg = aggregates.get("cbt_never_diagnose", {})
    return {
        "eval_type": "cbt_framework",
        "case_count": len(report.results),
        "summary": aggregates,
        "never_diagnose_compliance": {
            "total_cases": len(never_diagnose_agg.get("values", [])),
            "clean_cases": sum(
                1 for v in never_diagnose_agg.get("values", []) if v >= 1.0
            ),
            "violation_cases": sum(
                1 for v in never_diagnose_agg.get("values", []) if v < 1.0
            ),
        },
        "cases": cases_list,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CBT Framework Evaluation Suite",
    )
    parser.add_argument(
        "--case", type=str, default=None,
        help="Run a single case by ID (e.g. cbt_depression_triad_001)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save results to a JSON file",
    )
    parser.add_argument(
        "--general-metrics", action="store_true",
        help="Include all 7 general evaluation metrics alongside CBT metrics",
    )
    parser.add_argument(
        "--list-cases", action="store_true",
        help="List available benchmark cases and exit",
    )
    args = parser.parse_args()

    if args.list_cases:
        list_cases()
        sys.exit(0)

    cases: list[BenchmarkCase] = CBT_CASES
    if args.case:
        filtered = [c for c in cases if c.case_id == args.case]
        if not filtered:
            _log.error("cbt_eval.case_not_found", case_id=args.case)
            sys.exit(1)
        cases = filtered

    _log.info(
        "cbt_eval.start",
        case_count=len(cases),
        general_metrics=args.general_metrics,
        cases=[c.case_id for c in cases],
    )

    t_start = time.perf_counter()

    import asyncio
    report = asyncio.run(evaluate_cbt(
        cases=cases,
        include_general=args.general_metrics,
    ))

    elapsed_s = time.perf_counter() - t_start

    if args.json or args.output:
        data = format_cbt_json_report(report)
        data["elapsed_s"] = round(elapsed_s, 2)
        output = json.dumps(data, indent=2, default=str)

        if args.output:
            Path(args.output).write_text(output, encoding="utf-8")
            _log.info("cbt_eval.saved", path=args.output)
        else:
            print(output)
    else:
        print(format_cbt_report(report))
        print(f"\n  Total evaluation time: {elapsed_s:.1f}s")

    _log.info(
        "cbt_eval.complete",
        case_count=len(cases),
        elapsed_s=round(elapsed_s, 2),
    )


if __name__ == "__main__":
    main()
