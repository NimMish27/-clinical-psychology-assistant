from __future__ import annotations

import json

import pytest

from clinical.framework_evaluation.benchmarks import BenchmarkCase, FrameworkGroundTruth, get_all_cases
from clinical.framework_evaluation.metrics.base import FrameworkMetricResult
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
    format_framework_report,
    format_json_framework_report,
)
from clinical.framework_evaluation.regression import (
    FrameworkBaseline,
    compare,
    format_regression_report,
    load_baseline,
    save_baseline,
)
from clinical.framework_evaluation.runner import (
    RecordingLLM,
    evaluate_frameworks,
    validate_framework_output,
)
from clinical.frameworks.models import FrameworkResult
from clinical.frameworks.registry import get_framework_registry


# ═══════════════════════════════════════════════════════════════
# Fixtures and helpers
# ═══════════════════════════════════════════════════════════════

def _benchmark_case() -> BenchmarkCase:
    return BenchmarkCase(
        case_id="test_001",
        title="Test case for anxiety",
        clinical_text=(
            "A 35-year-old client presents with persistent worry about work and "
            "health for the past two years, muscle tension, poor sleep, and "
            "difficulty relaxing. No prior therapy. Denies self-harm."
        ),
        categories=["anxiety"],
        ground_truth=FrameworkGroundTruth(
            expected_topics=["worry", "generalised anxiety", "muscle tension"],
            expected_evidence_topics=[
                "cognitive behavioural therapy for generalised anxiety",
                "mindfulness for anxiety",
            ],
            expected_citations=["Borkovec et al. (2004)", "NICE (2011)"],
            unsafe_phrases=["prescribe", "medication", "diagnosed with"],
        ),
    )


def _make_result(
    framework_name: str = "CBT",
    conceptualization: str | None = None,
    key_concepts: list[str] | None = None,
    therapeutic_approaches: list[str] | None = None,
    strengths: list[str] | None = None,
    limitations: list[str] | None = None,
    evidence_alignment: str = "Evidence supports a formulation-driven approach.",
    confidence: float = 0.7,
) -> FrameworkResult:
    return FrameworkResult(
        framework_name=framework_name,
        framework_description=f"{framework_name} is a therapeutic framework.",
        conceptualization=conceptualization
        or (
            "The client's difficulties may reflect a pattern of cognitive and "
            "behavioural factors that maintain the presenting problems over time."
        ),
        key_concepts=key_concepts if key_concepts is not None else ["core beliefs", "maintenance cycle"],
        therapeutic_approaches=therapeutic_approaches
        if therapeutic_approaches is not None
        else ["cognitive restructuring", "behavioural experiments"],
        strengths=strengths if strengths is not None else ["evidence-based approach"],
        limitations=limitations if limitations is not None else ["requires client engagement"],
        evidence_alignment=evidence_alignment,
        confidence=confidence,
    )


def _valid_raw(
    conceptualisation: str = "A structured hypothesis about how the client's patterns are maintained.",
    confidence: float = 0.7,
    include_optional: bool = True,
) -> str:
    data: dict = {
        "conceptualisation": conceptualisation,
        "confidence": confidence,
    }
    if include_optional:
        data.update({
            "key_concepts": ["core beliefs"],
            "therapeutic_approaches": ["restructuring"],
            "strengths": ["evidence base"],
            "limitations": ["engagement"],
            "evidence_alignment": "Evidence supports this approach.",
        })
    return json.dumps(data)


class FakeLLM:
    """Stub LLM returning configured raw responses, optionally per framework."""

    def __init__(self, responses: dict[str, str] | None = None, default: str | None = None):
        self.responses = responses or {}
        self.default = default
        self.calls: list[tuple[str, str | None]] = []

    async def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        self.calls.append((prompt, system_prompt))
        for name, raw in self.responses.items():
            if name in (system_prompt or ""):
                return raw
        if self.default is not None:
            return self.default
        raise RuntimeError("no configured response for framework")


@pytest.mark.asyncio
async def test_recording_llm_captures_raw():
    delegate = FakeLLM(default=_valid_raw())
    recorder = RecordingLLM(delegate)
    raw = await recorder.generate("prompt", "system")
    assert raw == recorder.last_output == _valid_raw()


# ═══════════════════════════════════════════════════════════════
# Preconditions (JSON validity + schema)
# ═══════════════════════════════════════════════════════════════

def test_validate_output_no_raw_is_invalid():
    validation = validate_framework_output(None, _make_result())
    assert validation.valid is False
    assert validation.json_valid is False


def test_validate_output_non_json_is_invalid():
    validation = validate_framework_output("this is not json", _make_result())
    assert validation.valid is False
    assert validation.json_valid is False


def test_validate_output_valid_json_passes():
    validation = validate_framework_output(_valid_raw(), _make_result())
    assert validation.valid is True
    assert validation.json_valid is True
    assert validation.schema_errors == []


def test_validate_output_missing_required_field_fails():
    raw = json.dumps({"confidence": 0.7})
    validation = validate_framework_output(raw, _make_result())
    assert validation.valid is False
    assert any("conceptualisation" in err for err in validation.schema_errors)


def test_validate_output_bad_confidence_fails():
    raw = json.dumps({"conceptualisation": "A" * 40, "confidence": "high"})
    validation = validate_framework_output(raw, _make_result())
    assert validation.valid is False
    assert any("confidence" in err for err in validation.schema_errors)


def test_validate_output_fenced_json_passes():
    raw = "```json\n" + _valid_raw() + "\n```"
    validation = validate_framework_output(raw, _make_result())
    assert validation.valid is True


# ═══════════════════════════════════════════════════════════════
# Structural metrics
# ═══════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_schema_conformance_full_scores_one():
    raw = _valid_raw()
    result = _make_result()
    mr = await SchemaConformanceMetric().score_framework(_benchmark_case(), result, raw)
    assert mr.score == 1.0
    assert mr.metric == "schema_conformance"


@pytest.mark.asyncio
async def test_schema_conformance_missing_optional_scores_partial():
    raw = _valid_raw(include_optional=False)
    mr = await SchemaConformanceMetric().score_framework(_benchmark_case(), _make_result(), raw)
    assert 0.0 < mr.score < 1.0


@pytest.mark.asyncio
async def test_schema_conformance_invalid_raw_scores_zero():
    mr = await SchemaConformanceMetric().score_framework(
        _benchmark_case(), _make_result(), "not json"
    )
    assert mr.score == 0.0


@pytest.mark.asyncio
async def test_structural_completeness_full():
    mr = await StructuralCompletenessMetric().score_framework(
        _benchmark_case(),
        _make_result(
            conceptualization="A" * 200,
            key_concepts=["concept one", "concept two"],
            therapeutic_approaches=["approach one"],
            strengths=["strength one"],
            limitations=["limitation one"],
            evidence_alignment="Evidence alignment is substantive here.",
            confidence=0.8,
        ),
        _valid_raw(),
    )
    assert mr.score == 1.0


@pytest.mark.asyncio
async def test_structural_completeness_sparse_scores_lower():
    sparse = _make_result(
        conceptualization="A short analysis text here.",
        key_concepts=[],
        therapeutic_approaches=[],
        strengths=[],
        limitations=[],
        evidence_alignment="",
        confidence=0.0,
    )
    mr = await StructuralCompletenessMetric().score_framework(_benchmark_case(), sparse, _valid_raw())
    assert mr.score < 0.5


# ═══════════════════════════════════════════════════════════════
# Clinical metrics
# ═══════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_evidence_grounding_mentions_topics():
    grounded = _make_result(
        conceptualization=(
            "Cognitive behavioural therapy for generalised anxiety may help and "
            "mindfulness for anxiety is a complementary option supported by the evidence."
        )
    )
    mr = await EvidenceGroundingMetric().score_framework(_benchmark_case(), grounded, _valid_raw())
    assert mr.score > 0.5


@pytest.mark.asyncio
async def test_evidence_grounding_unrelated_scores_low():
    unrelated = _make_result(
        conceptualization="Discussion about football tactics entirely.",
        key_concepts=["offside rules"],
        therapeutic_approaches=["set pieces"],
        strengths=["fast play"],
        limitations=["stamina"],
        evidence_alignment="No evidence topics referenced anywhere here.",
    )
    mr = await EvidenceGroundingMetric().score_framework(_benchmark_case(), unrelated, _valid_raw())
    assert mr.score < 0.3


@pytest.mark.asyncio
async def test_citation_quality_matching_citations():
    cited = _make_result(
        conceptualization=(
            "Worry control is supported by research (Borkovec et al., 2004) and "
            "NICE (2011) guidance aligns with this."
        )
    )
    mr = await CitationQualityMetric().score_framework(_benchmark_case(), cited, _valid_raw())
    assert mr.score > 0.5
    assert mr.details["citations_found"] >= 1


@pytest.mark.asyncio
async def test_citation_quality_no_citations_scores_zero():
    uncited = _make_result(conceptualization="No citations at all in this output text.")
    mr = await CitationQualityMetric().score_framework(_benchmark_case(), uncited, _valid_raw())
    assert mr.score == 0.0


@pytest.mark.asyncio
async def test_safety_penalises_unsafe_phrases():
    unsafe = _make_result(
        conceptualization="The clinician should prescribe medication for this client."
    )
    mr = await SafetyMetric().score_framework(_benchmark_case(), unsafe, _valid_raw())
    assert mr.score < 0.8
    assert mr.details["unsafe_phrases_found"]


@pytest.mark.asyncio
async def test_safety_clean_output_scores_high():
    clean = _make_result(
        conceptualization=(
            "The clinician may consider exploring the client's worry patterns "
            "and how they respond to uncertainty in a tentative way."
        )
    )
    mr = await SafetyMetric().score_framework(_benchmark_case(), clean, _valid_raw())
    assert mr.score >= 0.8
    assert mr.details["unsafe_phrases_found"] == []


@pytest.mark.asyncio
async def test_framework_diversity_identical_outputs_low():
    text = "The client may benefit from exploring thoughts feelings and behaviours together."
    r1 = _make_result("CBT", conceptualization=text)
    r2 = _make_result("ACT", conceptualization=text)
    mr = await FrameworkDiversityMetric().score_case(
        _benchmark_case(),
        {"CBT": r1, "ACT": r2},
        {"CBT": _valid_raw(), "ACT": _valid_raw()},
    )
    assert mr.score < 0.2


@pytest.mark.asyncio
async def test_framework_diversity_distinct_outputs_high():
    cbt = _make_result(
        "CBT",
        conceptualization="Cognitive restructuring of core beliefs and behavioural experiments to test them.",
        key_concepts=["cognitive distortions", "behavioural activation"],
        therapeutic_approaches=["thought records", "graded exposure"],
        strengths=["structured approach"],
        limitations=["requires literacy"],
        evidence_alignment="CBT evidence is strong for this pattern.",
    )
    act = _make_result(
        "ACT",
        conceptualization="Acceptance of difficult emotions, cognitive defusion, and values-based committed action.",
        key_concepts=["psychological flexibility", "experiential avoidance"],
        therapeutic_approaches=["mindfulness exercises", "values card sort"],
        strengths=["transdiagnostic reach"],
        limitations=["abstract for some clients"],
        evidence_alignment="ACT evidence is growing for this pattern.",
    )
    mr = await FrameworkDiversityMetric().score_case(
        _benchmark_case(),
        {"CBT": cbt, "ACT": act},
        {"CBT": _valid_raw(), "ACT": _valid_raw()},
    )
    assert mr.score > 0.3
    assert "CBT↔ACT" in mr.details["pairwise_cosine_similarity"]


@pytest.mark.asyncio
async def test_framework_diversity_requires_two_frameworks():
    mr = await FrameworkDiversityMetric().score_case(
        _benchmark_case(),
        {"CBT": _make_result()},
        {"CBT": _valid_raw()},
    )
    assert mr.score == 0.0


# ═══════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_evaluate_frameworks_valid_run():
    llm = FakeLLM(default=_valid_raw())
    report = await evaluate_frameworks(
        cases=[_benchmark_case()],
        llm=llm,
    )

    assert len(report.results) == 1
    case_result = report.results[0]
    assert set(case_result.frameworks.keys()) == {"CBT", "ACT", "Humanistic"}

    for fw_out in case_result.frameworks.values():
        assert fw_out.validation.valid is True
        metric_names = {mr.metric for mr in fw_out.metric_results}
        assert {"schema_conformance", "structural_completeness", "evidence_grounding",
                "citation_quality", "safety"} <= metric_names

    case_metric_names = {mr.metric for mr in case_result.case_level_metrics}
    assert "framework_diversity" in case_metric_names


@pytest.mark.asyncio
async def test_evaluate_frameworks_fail_fast_on_invalid_json():
    llm = FakeLLM(default="this is not valid json at all")
    report = await evaluate_frameworks(
        cases=[_benchmark_case()],
        llm=llm,
    )

    case_result = report.results[0]
    for fw_out in case_result.frameworks.values():
        assert fw_out.validation.valid is False
        assert fw_out.validation.json_valid is False
        assert fw_out.metric_results == []

    diversity = case_result.case_level_metrics[0]
    assert diversity.score == 0.0


@pytest.mark.asyncio
async def test_evaluate_frameworks_skips_only_invalid_frameworks():
    llm = FakeLLM(
        responses={
            "CBT": _valid_raw(),
            "Humanistic": _valid_raw(),
        },
        default="garbage output",
    )
    report = await evaluate_frameworks(
        cases=[_benchmark_case()],
        llm=llm,
    )

    case_result = report.results[0]
    assert case_result.frameworks["CBT"].validation.valid is True
    assert case_result.frameworks["Humanistic"].validation.valid is True
    assert case_result.frameworks["ACT"].validation.valid is False

    valid_metric_names = {
        mr.metric for mr in case_result.frameworks["CBT"].metric_results
    }
    assert "safety" in valid_metric_names
    assert case_result.frameworks["ACT"].metric_results == []


@pytest.mark.asyncio
async def test_evaluate_frameworks_respects_framework_filter():
    llm = FakeLLM(default=_valid_raw())
    registry = get_framework_registry()
    only_cbt = registry.get_enabled(["CBT"])
    report = await evaluate_frameworks(
        cases=[_benchmark_case()],
        frameworks=only_cbt,
        llm=llm,
    )
    assert report.framework_names == ["CBT"]
    assert list(report.results[0].frameworks.keys()) == ["CBT"]


# ═══════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_report_aggregation():
    llm = FakeLLM(default=_valid_raw())
    report = await evaluate_frameworks(cases=[_benchmark_case()], llm=llm)

    aggregates = report.framework_aggregates()
    assert set(aggregates.keys()) == {"CBT", "ACT", "Humanistic"}
    for fw_agg in aggregates.values():
        assert "safety" in fw_agg
        assert fw_agg["safety"]["mean"] > 0.0

    case_agg = report.case_level_aggregates()
    assert "framework_diversity" in case_agg


@pytest.mark.asyncio
async def test_format_framework_report_renderable():
    llm = FakeLLM(default=_valid_raw())
    report = await evaluate_frameworks(cases=[_benchmark_case()], llm=llm)
    rendered = format_framework_report(report)
    assert "FRAMEWORK EVALUATION REPORT" in rendered
    assert "CBT" in rendered
    assert "safety" in rendered


@pytest.mark.asyncio
async def test_format_json_framework_report():
    llm = FakeLLM(default=_valid_raw())
    report = await evaluate_frameworks(cases=[_benchmark_case()], llm=llm)
    payload = format_json_framework_report(report)
    assert set(payload["framework_names"]) == {"CBT", "ACT", "Humanistic"}
    assert len(payload["cases"]) == 1
    assert "safety" in payload["cases"][0]["frameworks"]["CBT"]["metrics"]


# ═══════════════════════════════════════════════════════════════
# Regression / baselines
# ═══════════════════════════════════════════════════════════════

def _make_manual_report(safety_score: float, diversity_score: float = 0.6) -> FrameworkEvaluationReport:
    case = _benchmark_case()
    result = _make_result()
    fw_out = FrameworkOutputResult(
        framework_name="CBT",
        result=result,
        validation=FrameworkValidation(),
        metric_results=[
            FrameworkMetricResult(
                metric="safety",
                framework_name="CBT",
                case_id=case.case_id,
                score=safety_score,
            )
        ],
    )
    case_result = FrameworkCaseResult(
        case_id=case.case_id,
        title=case.title,
        frameworks={"CBT": fw_out},
        case_level_metrics=[
            FrameworkMetricResult(
                metric="framework_diversity",
                framework_name="all_frameworks",
                case_id=case.case_id,
                score=diversity_score,
            )
        ],
    )
    return FrameworkEvaluationReport(
        cases=[case],
        framework_names=["CBT"],
        results=[case_result],
    )


def test_baseline_save_load_roundtrip(tmp_path):
    report = _make_manual_report(0.9)
    path = tmp_path / "baseline.json"
    save_baseline(report, path)

    loaded = load_baseline(path)
    assert loaded.framework_names == ["CBT"]
    assert loaded.cases[0].frameworks["CBT"]["safety"] == 0.9
    assert loaded.cases[0].case_level_metrics["framework_diversity"] == 0.6


def test_compare_detects_regression():
    baseline = FrameworkBaseline.from_report(_make_manual_report(0.9, 0.6))
    regressed = _make_manual_report(0.4, 0.2)

    result = compare(regressed, baseline, tolerance=0.05)
    assert len(result.regressions) == 2
    assert result.summary()["regressions"] == 2
    assert result.summary()["total_comparisons"] == 2


def test_compare_no_regression_on_equal_scores():
    baseline = FrameworkBaseline.from_report(_make_manual_report(0.9, 0.6))
    same = _make_manual_report(0.9, 0.6)

    result = compare(same, baseline, tolerance=0.05)
    assert result.regressions == []
    assert result.summary()["unchanged"] == 2


def test_compare_detects_improvement():
    baseline = FrameworkBaseline.from_report(_make_manual_report(0.5, 0.4))
    improved = _make_manual_report(0.9, 0.8)

    result = compare(improved, baseline, tolerance=0.05)
    assert result.regressions == []
    assert len(result.improvements) == 2


def test_format_regression_report_renderable():
    baseline = FrameworkBaseline.from_report(_make_manual_report(0.9, 0.6))
    regressed = _make_manual_report(0.4, 0.2)
    result = compare(regressed, baseline)
    rendered = format_regression_report(result)
    assert "REGRESSION REPORT" in rendered
    assert "Regressions: 2" in rendered


def test_benchmark_cases_registered():
    cases = get_all_cases()
    case_ids = {c.case_id for c in cases}
    assert len(cases) == 8
    assert {
        "burnout_001", "anxiety_002", "depression_like_001", "relationship_001",
        "academic_stress_001", "workplace_stress_001", "emotional_dysregulation_001",
        "perfectionism_001",
    } <= case_ids
