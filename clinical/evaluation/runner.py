from __future__ import annotations

import time
from typing import Any

from clinical.evaluation.benchmarks.cases import ALL_CASES, BenchmarkCase
from clinical.evaluation.metrics.citation import CitationAccuracyMetric
from clinical.evaluation.metrics.formulation import FormulationQualityMetric
from clinical.evaluation.metrics.hallucination import HallucinationRateMetric
from clinical.evaluation.metrics.helpfulness import ClinicalHelpfulnessMetric
from clinical.evaluation.metrics.missing_info import MissingInfoDetectionMetric
from clinical.evaluation.metrics.retrieval import RetrievalPrecisionMetric
from clinical.evaluation.metrics.relevance import TherapeuticRelevanceMetric
from clinical.evaluation.report import EvaluationReport, EvaluationResult
from app_logging.logger import get_logger

_log = get_logger(__name__)


_ALL_METRICS = [
    RetrievalPrecisionMetric(),
    CitationAccuracyMetric(),
    FormulationQualityMetric(),
    TherapeuticRelevanceMetric(),
    MissingInfoDetectionMetric(),
    HallucinationRateMetric(),
    ClinicalHelpfulnessMetric(),
]


async def evaluate_all(
    cases: list[BenchmarkCase] | None = None,
    metrics: list | None = None,
) -> EvaluationReport:
    """Run all evaluation metrics against all benchmark cases.

    Args:
        cases:  Cases to evaluate. Defaults to ALL_CASES.
        metrics: Metrics to run. Defaults to all seven.

    Returns:
        EvaluationReport with per-case and aggregate scores.
    """
    cases = cases or ALL_CASES
    metrics = metrics or _ALL_METRICS

    results: list[EvaluationResult] = []

    for case in cases:
        _log.info("eval.running_case", case_id=case.case_id, title=case.title)
        t_start = time.perf_counter()

        try:
            from clinical.graph.graph import run_pipeline

            state = await run_pipeline(case.clinical_text)
        except Exception as exc:
            _log.error("eval.case_failed", case_id=case.case_id, error=str(exc))
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
                    "eval.metric_failed",
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
