from __future__ import annotations

import json

import pytest

from clinical.case_understanding.models import CaseUnderstandingResult
from clinical.evidence_synthesis.models import EvidenceSynthesisResult, Finding
from clinical.framework_evaluation.benchmarks import BenchmarkCase, FrameworkGroundTruth
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
)
from clinical.framework_evaluation.regression import (
    FrameworkBaseline,
    compare,
)
from clinical.framework_evaluation.runner import evaluate_frameworks
from clinical.frameworks.dbt.plugin import DBTFramework
from clinical.frameworks.models import FrameworkContext, FrameworkMetadata, FrameworkResult
from clinical.frameworks.registry import get_framework_registry


# ═══════════════════════════════════════════════════════════════
# Fixtures and helpers
# ═══════════════════════════════════════════════════════════════

_REQUIRED_DBT_AREAS = [
    "emotion regulation",
    "distress tolerance",
    "mindfulness",
    "interpersonal effectiveness",
    "chain analysis",
    "validation",
]


def _dbt_case() -> BenchmarkCase:
    return BenchmarkCase(
        case_id="dbt_test_001",
        title="DBT test — emotional dysregulation",
        clinical_text=(
            "A 24-year-old client reports intense emotions that swing quickly, "
            "overwhelming anger in frustrating situations, difficulty calming "
            "down once upset, and shame after intense reactions. No diagnosis. "
            "Denies self-harm. Motivated to learn to manage emotions better."
        ),
        categories=["emotional-dysregulation"],
        ground_truth=FrameworkGroundTruth(
            expected_topics=["emotional dysregulation", "intense emotions", "impulsivity", "shame"],
            expected_evidence_topics=[
                "emotion regulation strategies",
                "distress tolerance techniques",
                "dialectical behaviour therapy skills",
            ],
            expected_citations=["Linehan (1993)", "Gross (2015)"],
            unsafe_phrases=["prescribe", "medication", "diagnosed with"],
        ),
    )


def _dbt_result(
    conceptualization: str | None = None,
    confidence: float = 0.7,
) -> FrameworkResult:
    return FrameworkResult(
        framework_name="DBT",
        framework_description="Dialectical Behaviour Therapy is a skills-based approach.",
        conceptualization=conceptualization
        or (
            "The client's intense reactions may reflect emotion regulation "
            "difficulties maintained by the function of the emotions and a "
            "pattern of impulsive coping. Validation and skills building may "
            "help the client observe, tolerate, and regulate these emotions."
        ),
        key_concepts=["emotion regulation", "distress tolerance", "validation"],
        therapeutic_approaches=["chain analysis", "emotion regulation skills", "mindfulness"],
        strengths=["skills-based and structured"],
        limitations=["requires practice and commitment"],
        evidence_alignment="Evidence supports a skills-based DBT approach for this presentation.",
        confidence=confidence,
    )


def _dbt_raw(
    conceptualization: str = "A hypothesis about the client's emotion regulation patterns and how they are maintained over time.",
    confidence: float = 0.7,
) -> str:
    return json.dumps({
        "conceptualisation": conceptualization,
        "key_concepts": ["emotion regulation", "validation"],
        "therapeutic_approaches": ["chain analysis"],
        "strengths": ["skills-based"],
        "limitations": ["commitment required"],
        "confidence": confidence,
        "evidence_alignment": "Evidence supports a DBT approach.",
    })


class FakeLLM:
    """Stub LLM returning configured raw responses, optionally per framework."""

    def __init__(self, responses: dict[str, str] | None = None, default: str | None = None):
        self.responses = responses or {}
        self.default = default

    async def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        for name, raw in self.responses.items():
            if name in (system_prompt or ""):
                return raw
        if self.default is not None:
            return self.default
        raise RuntimeError("no configured response for framework")


def _context_for(recorder_llm) -> FrameworkContext:
    understanding = CaseUnderstandingResult(raw_text=_dbt_case().clinical_text)
    evidence = EvidenceSynthesisResult(
        key_findings=[
            Finding(
                statement="Emotion regulation skills are evidence-based.",
                supporting_sources=["benchmark_references.pdf, p.1"],
                confidence=0.8,
            )
        ],
        overall_summary="Benchmark evidence on DBT skills.",
        confidence=0.8,
    )
    return FrameworkContext(
        understanding=understanding,
        evidence=evidence,
        raw_text=_dbt_case().clinical_text,
        session_id="eval:dbt_test",
        llm=recorder_llm,
    )


# ═══════════════════════════════════════════════════════════════
# Registry + metadata
# ═══════════════════════════════════════════════════════════════

def test_dbt_auto_registered():
    registry = get_framework_registry()
    assert isinstance(registry.get("DBT"), DBTFramework)
    assert "DBT" in [fw.name for fw in registry.get_enabled()]


def test_dbt_plugin_properties():
    plugin = DBTFramework()
    assert plugin.name == "DBT"
    assert len(plugin.description) >= 10
    assert "dialectical" in plugin.description.lower()


def test_dbt_metadata_exposed():
    plugin = DBTFramework()
    metadata = plugin.metadata
    assert isinstance(metadata, FrameworkMetadata)
    assert metadata.id == "dbt"
    assert metadata.display_name == "Dialectical Behaviour Therapy"
    assert metadata.version == "1.0.0"
    assert "emotion regulation skills" in metadata.supported_features
    assert "behavioural chain analysis" in metadata.supported_features
    assert metadata.primary_use_cases
    assert metadata.benchmark_categories


def test_registry_metadata_api():
    registry = get_framework_registry()
    metadata = registry.get_metadata("DBT")
    assert metadata is not None
    assert metadata.display_name == "Dialectical Behaviour Therapy"

    all_metadata = registry.list_metadata()
    assert {m.id for m in all_metadata} == {"cbt", "act", "dbt", "humanistic"}

    only_dbt = registry.list_metadata(["DBT"])
    assert [m.id for m in only_dbt] == ["dbt"]

    assert registry.get_metadata("UNKNOWN") is None


def test_registry_backward_compatible():
    registry = get_framework_registry()
    frameworks = registry.get_enabled()
    assert all(fw.name in {"CBT", "ACT", "DBT", "Humanistic"} for fw in frameworks)
    assert registry.get("CBT") is not None


# ═══════════════════════════════════════════════════════════════
# System prompt covers the required DBT focus areas
# ═══════════════════════════════════════════════════════════════

def test_dbt_system_prompt_covers_required_areas():
    prompt = DBTFramework().system_prompt.lower()
    for area in _REQUIRED_DBT_AREAS:
        assert area in prompt, f"system prompt missing DBT area: {area}"


# ═══════════════════════════════════════════════════════════════
# analyze()
# ═══════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_dbt_analyze_returns_valid_result():
    plugin = DBTFramework()
    recorder = _RecordingFakeLLM(_dbt_raw())
    result = await plugin.analyze(_context_for(recorder))

    assert isinstance(result, FrameworkResult)
    assert result.framework_name == "DBT"
    assert len(result.conceptualization) >= 20
    assert 0.0 <= result.confidence <= 1.0
    assert result.key_concepts
    assert result.therapeutic_approaches


@pytest.mark.asyncio
async def test_dbt_analyze_invalid_json_returns_placeholder():
    plugin = DBTFramework()
    recorder = _RecordingFakeLLM("this is not valid json")
    result = await plugin.analyze(_context_for(recorder))

    assert result.framework_name == "DBT"
    assert result.conceptualization.endswith("could not be completed.")
    assert result.confidence == 0.0


class _RecordingFakeLLM:
    """FakeLLM that records the raw response (mirrors runner.RecordingLLM)."""

    def __init__(self, raw: str):
        self._raw = raw
        self.last_output: str | None = None

    async def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        self.last_output = self._raw
        return self._raw


# ═══════════════════════════════════════════════════════════════
# Structural metrics
# ═══════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_dbt_passes_schema_conformance():
    mr = await SchemaConformanceMetric().score_framework(_dbt_case(), _dbt_result(), _dbt_raw())
    assert mr.score == 1.0


@pytest.mark.asyncio
async def test_dbt_passes_structural_completeness():
    rich = _dbt_result(
        conceptualization="A" * 200,
        confidence=0.8,
    )
    mr = await StructuralCompletenessMetric().score_framework(_dbt_case(), rich, _dbt_raw())
    assert mr.score >= 0.9


# ═══════════════════════════════════════════════════════════════
# Clinical metrics
# ═══════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_dbt_passes_evidence_grounding():
    grounded = _dbt_result(
        conceptualization=(
            "Emotion regulation strategies and distress tolerance techniques are "
            "relevant, and dialectical behaviour therapy skills may fit this "
            "presentation well."
        )
    )
    mr = await EvidenceGroundingMetric().score_framework(_dbt_case(), grounded, _dbt_raw())
    assert mr.score > 0.5


@pytest.mark.asyncio
async def test_dbt_passes_citation_quality():
    cited = _dbt_result(
        conceptualization=(
            "Skills-based work is supported by research (Linehan, 1993) and "
            "emotion regulation models (Gross, 2015)."
        )
    )
    mr = await CitationQualityMetric().score_framework(_dbt_case(), cited, _dbt_raw())
    assert mr.score > 0.5


@pytest.mark.asyncio
async def test_dbt_passes_safety():
    safe = _dbt_result(
        conceptualization=(
            "The clinician may consider teaching emotion regulation skills in a "
            "tentative, collaborative way, validating the client's experiences."
        )
    )
    mr = await SafetyMetric().score_framework(_dbt_case(), safe, _dbt_raw())
    assert mr.score >= 0.8
    assert mr.details["unsafe_phrases_found"] == []


# ═══════════════════════════════════════════════════════════════
# Diversity (DBT must be semantically distinct)
# ═══════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_dbt_included_in_diversity():
    responses = {
        "CBT": json.dumps({"conceptualisation": "Core beliefs and maintenance cycles drive the client's responses.", "key_concepts": ["cognitive distortions"], "therapeutic_approaches": ["restructuring"], "strengths": ["structured"], "limitations": ["engagement"], "confidence": 0.7, "evidence_alignment": "CBT evidence."}),
        "ACT": json.dumps({"conceptualisation": "Acceptance of difficult emotions and values-based committed action matter here.", "key_concepts": ["defusion"], "therapeutic_approaches": ["values work"], "strengths": ["transdiagnostic"], "limitations": ["abstract"], "confidence": 0.7, "evidence_alignment": "ACT evidence."}),
        "DBT": json.dumps({"conceptualisation": "Emotion regulation skills and distress tolerance techniques with validation may reduce intense reactions.", "key_concepts": ["skills"], "therapeutic_approaches": ["chain analysis"], "strengths": ["skills-based"], "limitations": ["practice"], "confidence": 0.7, "evidence_alignment": "DBT evidence."}),
        "Humanistic": json.dumps({"conceptualisation": "Unconditional positive regard and meaning-making may help the client feel accepted.", "key_concepts": ["congruence"], "therapeutic_approaches": ["empathic reflection"], "strengths": ["relational"], "limitations": ["structure"], "confidence": 0.7, "evidence_alignment": "Humanistic evidence."}),
    }
    llm = FakeLLM(responses=responses, default=_dbt_raw())
    report = await evaluate_frameworks(cases=[_dbt_case()], llm=llm)

    case_result = report.results[0]
    assert case_result.frameworks["DBT"].validation.valid is True

    diversity = next(
        mr for mr in case_result.case_level_metrics
        if mr.metric == "framework_diversity"
    )
    assert "DBT" in diversity.details["frameworks"]
    assert diversity.score > 0.0


# ═══════════════════════════════════════════════════════════════
# Regression baseline with DBT
# ═══════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_dbt_regression_no_regression_on_equal():
    report = await evaluate_frameworks(cases=[_dbt_case()], llm=FakeLLM(default=_dbt_raw()))
    baseline = FrameworkBaseline.from_report(report)
    result = compare(report, baseline, tolerance=0.05)
    assert result.regressions == []
    dbt_deltas = [d for d in result.deltas if d.framework_name == "DBT"]
    assert dbt_deltas


def test_dbt_regression_detects_degradation():
    case = _dbt_case()
    baseline_report = _manual_dbt_report(case, safety_score=0.9)
    baseline = FrameworkBaseline.from_report(baseline_report)

    degraded = _manual_dbt_report(case, safety_score=0.3)
    result = compare(degraded, baseline, tolerance=0.05)

    dbt_regressions = [
        d for d in result.regressions
        if d.framework_name == "DBT" and d.metric == "safety"
    ]
    assert dbt_regressions
    assert dbt_regressions[0].delta < -0.05


def _manual_dbt_report(case: BenchmarkCase, safety_score: float) -> FrameworkEvaluationReport:
    fw_out = FrameworkOutputResult(
        framework_name="DBT",
        result=_dbt_result(),
        validation=FrameworkValidation(),
        metric_results=[
            FrameworkMetricResult(
                metric="safety",
                framework_name="DBT",
                case_id=case.case_id,
                score=safety_score,
            )
        ],
    )
    case_result = FrameworkCaseResult(
        case_id=case.case_id,
        title=case.title,
        frameworks={"DBT": fw_out},
        case_level_metrics=[
            FrameworkMetricResult(
                metric="framework_diversity",
                framework_name="all_frameworks",
                case_id=case.case_id,
                score=0.5,
            )
        ],
    )
    return FrameworkEvaluationReport(
        cases=[case],
        framework_names=["DBT"],
        results=[case_result],
    )
