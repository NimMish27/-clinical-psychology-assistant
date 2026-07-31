from __future__ import annotations

from typing import Any

from app_logging.logger import get_logger
from clinical.case_understanding.models import CaseUnderstandingResult
from clinical.evidence_synthesis.models import EvidenceSynthesisResult, Finding
from clinical.framework_evaluation.benchmarks import BenchmarkCase, get_all_cases
from clinical.framework_evaluation.metrics.base import BaseFrameworkMetric, FrameworkMetricResult
from clinical.framework_evaluation.metrics.clinical import (
    CitationQualityMetric,
    EvidenceGroundingMetric,
    FrameworkDiversityMetric,
    SafetyMetric,
)
from clinical.framework_evaluation.metrics.structural import (
    SchemaConformanceMetric,
    StructuralCompletenessMetric,
)
from clinical.framework_evaluation.report import (
    FrameworkCaseResult,
    FrameworkEvaluationReport,
    FrameworkOutputResult,
    FrameworkValidation,
)
from clinical.frameworks.base import BaseFramework
from clinical.frameworks.models import FrameworkContext, FrameworkResult
from clinical.frameworks.registry import get_framework_registry
from clinical.frameworks.router import ParallelFrameworkRouter
from clinical.llm import get_llm_service

_log = get_logger(__name__)

DEFAULT_STRUCTURAL_METRICS: list[BaseFrameworkMetric] = [
    SchemaConformanceMetric(),
    StructuralCompletenessMetric(),
]

DEFAULT_CLINICAL_METRICS: list[BaseFrameworkMetric] = [
    EvidenceGroundingMetric(),
    CitationQualityMetric(),
    SafetyMetric(),
]

DEFAULT_METRICS: list[BaseFrameworkMetric] = (
    DEFAULT_STRUCTURAL_METRICS + DEFAULT_CLINICAL_METRICS
)

DEFAULT_CASE_LEVEL_METRICS: list[BaseFrameworkMetric] = [
    FrameworkDiversityMetric(),
]


class RecordingLLM:
    """LLM adapter that records the raw response for validation.

    Wraps any object exposing ``async generate(prompt, system_prompt=None)``
    and stores the last raw output so the runner can validate JSON/schema
    as a precondition before scoring.
    """

    def __init__(self, delegate: Any):
        self._delegate = delegate
        self.last_output: str | None = None

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        raw = await self._delegate.generate(prompt=prompt, system_prompt=system_prompt)
        self.last_output = raw
        return raw


def _build_understanding(case: BenchmarkCase) -> CaseUnderstandingResult:
    """Build a minimal, valid CaseUnderstandingResult for a benchmark case."""
    return CaseUnderstandingResult(raw_text=case.clinical_text)


def _build_evidence(case: BenchmarkCase) -> EvidenceSynthesisResult:
    """Build synthetic evidence grounded in the benchmark ground truth."""
    findings = [
        Finding(
            statement=topic,
            supporting_sources=["benchmark_references.pdf, p.1"],
            confidence=0.8,
        )
        for topic in case.ground_truth.expected_evidence_topics
    ]
    summary = (
        "Synthesised benchmark evidence. Relevant areas: "
        + ", ".join(case.ground_truth.expected_evidence_topics or ["case evidence"])
        + "."
    )
    return EvidenceSynthesisResult(
        key_findings=findings,
        overall_summary=summary,
        confidence=0.8,
        chunks_analysed=len(findings),
    )


def validate_framework_output(
    raw: str | None,
    result: FrameworkResult,
) -> FrameworkValidation:
    """Fail-fast preconditions for a framework output.

    Checks (1) that raw JSON parses and (2) that it conforms to the
    FrameworkResult schema's mandatory fields. Invalid outputs do not
    proceed to metric evaluation.
    """
    if not raw:
        return FrameworkValidation(
            valid=False,
            json_valid=False,
            error="no raw LLM output captured",
        )

    try:
        from clinical.frameworks.base import BaseFramework

        data = BaseFramework._parse_json_response(raw)
    except Exception as exc:
        return FrameworkValidation(
            valid=False,
            json_valid=False,
            error=f"invalid JSON: {exc}",
        )

    schema_errors: list[str] = []
    conceptualisation = data.get("conceptualisation")
    if not isinstance(conceptualisation, str) or len(conceptualisation.strip()) < 20:
        schema_errors.append("conceptualisation must be a non-trivial string (>= 20 chars)")

    confidence = data.get("confidence")
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = None
    if confidence is None or not (0.0 <= confidence <= 1.0):
        schema_errors.append("confidence must be a number in [0, 1]")

    if result.conceptualization.endswith("could not be completed.") and result.confidence == 0.0:
        schema_errors.append("framework returned a failure placeholder")

    return FrameworkValidation(
        valid=not schema_errors,
        json_valid=True,
        schema_errors=schema_errors,
    )


async def evaluate_frameworks(
    cases: list[BenchmarkCase] | None = None,
    frameworks: list[BaseFramework] | None = None,
    llm: Any | None = None,
    metrics: list[BaseFrameworkMetric] | None = None,
    case_level_metrics: list[BaseFrameworkMetric] | None = None,
) -> FrameworkEvaluationReport:
    """Evaluate all framework plugins against the benchmark cases.

    Args:
        cases: Benchmark cases to evaluate. Defaults to all registered cases.
        frameworks: Framework plugins to evaluate. Defaults to all enabled.
        llm: LLM backend (any object with ``async generate(...)``). Defaults
             to the shared LLMService.
        metrics: Per-framework metrics to score. Defaults to the standard set.
        case_level_metrics: Case-level metrics (e.g. diversity). Defaults to
                            the standard set.

    Returns:
        FrameworkEvaluationReport with per-framework and case-level results.

    JSON validity and schema conformance are enforced as preconditions:
    outputs that fail are recorded as INVALID and are not scored further.
    """
    cases = list(cases) if cases else get_all_cases()
    registry = get_framework_registry()
    fws = list(frameworks) if frameworks else registry.get_enabled()
    llm = llm if llm is not None else get_llm_service()
    metrics = list(metrics) if metrics else DEFAULT_METRICS
    case_metrics = list(case_level_metrics) if case_level_metrics else DEFAULT_CASE_LEVEL_METRICS

    router = ParallelFrameworkRouter()
    report = FrameworkEvaluationReport(
        cases=cases,
        framework_names=[fw.name for fw in fws],
        framework_metadata={
            fw.name: fw.metadata.model_dump() for fw in fws
        },
    )

    for case in cases:
        _log.info("framework_eval.running_case", case_id=case.case_id, title=case.title)
        understanding = _build_understanding(case)
        evidence = _build_evidence(case)

        case_result = FrameworkCaseResult(case_id=case.case_id, title=case.title)

        valid_results: dict[str, FrameworkResult] = {}
        valid_raws: dict[str, str | None] = {}

        for fw in fws:
            recorder = RecordingLLM(llm)
            context = FrameworkContext(
                understanding=understanding,
                evidence=evidence,
                raw_text=case.clinical_text,
                session_id=f"eval:{case.case_id}",
                llm=recorder,
            )

            try:
                routed = await router.route([fw], context)
                result = routed[fw.name]
            except Exception as exc:
                _log.error("framework_eval.framework_failed", framework=fw.name, error=str(exc))
                result = FrameworkResult(
                    framework_name=fw.name,
                    framework_description=fw.description,
                    conceptualization=f"{fw.name} analysis could not be completed.",
                    confidence=0.0,
                )

            validation = validate_framework_output(recorder.last_output, result)

            fw_out = FrameworkOutputResult(
                framework_name=fw.name,
                result=result,
                validation=validation,
            )

            if validation.valid:
                valid_results[fw.name] = result
                valid_raws[fw.name] = recorder.last_output
                for metric in metrics:
                    try:
                        mr = await metric.score_framework(case, result, recorder.last_output)
                        fw_out.metric_results.append(mr)
                    except Exception as exc:
                        _log.error(
                            "framework_eval.metric_failed",
                            metric=metric.name,
                            framework=fw.name,
                            case_id=case.case_id,
                            error=str(exc),
                        )
                        fw_out.metric_results.append(FrameworkMetricResult(
                            metric=metric.name,
                            framework_name=fw.name,
                            case_id=case.case_id,
                            score=0.0,
                            details={"error": str(exc)},
                        ))
            else:
                _log.warning(
                    "framework_eval.invalid_output",
                    framework=fw.name,
                    case_id=case.case_id,
                    error=validation.error,
                )

            case_result.frameworks[fw.name] = fw_out

        for metric in case_metrics:
            try:
                mr = await metric.score_case(case, valid_results, valid_raws)
                case_result.case_level_metrics.append(mr)
            except Exception as exc:
                _log.error(
                    "framework_eval.case_metric_failed",
                    metric=metric.name,
                    case_id=case.case_id,
                    error=str(exc),
                )
                case_result.case_level_metrics.append(FrameworkMetricResult(
                    metric=metric.name,
                    framework_name="all_frameworks",
                    case_id=case.case_id,
                    score=0.0,
                    details={"error": str(exc)},
                ))

        report.results.append(case_result)

    return report
