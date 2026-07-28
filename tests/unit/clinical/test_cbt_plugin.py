"""
tests/unit/clinical/test_cbt_plugin.py — Tests for CBT Plugin with lifecycle overrides.
"""

from __future__ import annotations

import pytest
from clinical.framework.cbt.plugin import (
    CBTPlugin,
    _DIAGNOSTIC_PATTERNS,
    _diagnosis_replacement,
    _strip_diagnoses,
    _strip_diagnoses_from_result,
)
from clinical.framework.models import (
    CaseAnalysis,
    CaseConceptualization,
    EvidenceResult,
    FollowupQuestionsResult,
    InterventionPlanResult,
    LimitationsResult,
    SessionObjectivesResult,
    TherapeuticFocusResult,
    FrameworkCaseInput,
)
from clinical.framework.prompts.manager import PromptManager
from clinical.framework.registry import FrameworkRegistry


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_registry():
    FrameworkRegistry._instance = None
    yield
    FrameworkRegistry._instance = None


@pytest.fixture
def plugin() -> CBTPlugin:
    return CBTPlugin()


@pytest.fixture
def case_input() -> FrameworkCaseInput:
    return FrameworkCaseInput(
        case_understanding={
            "presenting_concerns": "Feeling very down, can't get out of bed, constant worry about everything",
            "thoughts": "I'm a failure, nothing will get better, everyone is judging me",
            "behaviours": "Stopped going to work, avoiding friends, staying in bed all day",
        },
        raw_text=(
            "I feel hopeless. I can't concentrate at work. My mind is always racing "
            "with worst-case scenarios. I've been avoiding social situations."
        ),
    )


# ── Test: CBT metadata ─────────────────────────────────────────────────────


class TestCBTMetadata:
    def test_metadata_id(self):
        meta = CBTPlugin.metadata()
        assert meta.id == "cbt"

    def test_metadata_display_name(self):
        meta = CBTPlugin.metadata()
        assert meta.display_name == "Cognitive Behavioral Therapy"

    def test_metadata_version(self):
        meta = CBTPlugin.metadata()
        assert meta.version == "2.0.0"

    def test_metadata_literature_collection(self):
        meta = CBTPlugin.metadata()
        assert meta.literature_collection == "cbt_literature"

    def test_metadata_tags_include_cbt(self):
        meta = CBTPlugin.metadata()
        assert "cbt" in meta.tags

    def test_metadata_supported_use_cases(self):
        meta = CBTPlugin.metadata()
        assert len(meta.supported_use_cases) >= 5
        assert any("depress" in c.lower() for c in meta.supported_use_cases)


# ── Test: is_applicable ────────────────────────────────────────────────────


class TestCBTApplicability:
    def test_applicable_for_anxiety(self):
        assert CBTPlugin.is_applicable({"presenting_concerns": "anxiety and worry"})

    def test_applicable_for_depression(self):
        assert CBTPlugin.is_applicable({"presenting_concerns": "depression and low mood"})

    def test_applicable_for_avoidance(self):
        assert CBTPlugin.is_applicable({"presenting_concerns": "avoidance of social situations"})

    def test_applicable_for_ocd(self):
        assert CBTPlugin.is_applicable({"presenting_concerns": "obsessive thoughts and compulsive rituals"})

    def test_not_applicable_for_unrelated(self):
        assert not CBTPlugin.is_applicable({"presenting_concerns": "career counselling for mid-life change"})

    def test_empty_case_not_applicable(self):
        assert not CBTPlugin.is_applicable({})

    def test_not_applicable_for_grief_only(self):
        assert not CBTPlugin.is_applicable({"presenting_concerns": "grief and loss after bereavement"})


# ── Test: Never-diagnose enforcement ───────────────────────────────────────


class TestNeverDiagnose:
    def test_strip_diagnoses_mdd(self):
        text = "Client meets criteria for major depressive disorder"
        result = _strip_diagnoses(text)
        assert "major depressive disorder" not in result
        assert "persistent low mood" in result

    def test_strip_diagnoses_gad(self):
        text = "Presenting with generalized anxiety disorder"
        result = _strip_diagnoses(text)
        assert "generalized anxiety disorder" not in result
        assert "persistent and excessive worry" in result

    def test_strip_diagnoses_ptsd(self):
        text = "Trauma-related symptoms consistent with PTSD"
        result = _strip_diagnoses(text)
        assert "PTSD" not in result
        assert "trauma-related distress" in result

    def test_strip_diagnoses_ocd(self):
        text = "Obsessive compulsive disorder symptoms present"
        result = _strip_diagnoses(text)
        assert "obsessive compulsive disorder" not in result
        assert "recurrent intrusive thoughts" in result

    def test_strip_diagnoses_adhd(self):
        text = "ADHD symptoms affecting work performance"
        result = _strip_diagnoses(text)
        assert "ADHD" not in result

    def test_strip_diagnoses_bpd(self):
        text = "Borderline personality disorder traits"
        result = _strip_diagnoses(text)
        assert "borderline personality disorder" not in result
        assert "emotional and interpersonal instability" in result

    def test_strip_diagnoses_panic(self):
        text = "Panic disorder with agoraphobia"
        result = _strip_diagnoses(text)
        assert "panic disorder" not in result
        assert "intense fear episodes" in result

    def test_strip_diagnoses_normal_text_unchanged(self):
        text = "Client experiences anxiety in social situations and avoids gatherings"
        result = _strip_diagnoses(text)
        assert result == text

    def test_strip_diagnoses_multiple_labels(self):
        text = "MDD and GAD are both present; BPD traits also noted"
        result = _strip_diagnoses(text)
        assert "MDD" not in result
        assert "GAD" not in result
        assert "BPD" not in result

    def test_strip_diagnoses_case_insensitive(self):
        text = "Major Depressive Disorder and Generalised Anxiety Disorder"
        result = _strip_diagnoses(text)
        assert "Major Depressive Disorder" not in result
        assert "Generalised Anxiety Disorder" not in result
        assert "persistent low mood" in result.lower()

    def test_diagnosis_replacement_unknown_returns_original(self):
        assert _diagnosis_replacement("unknown_label") == "unknown_label"


# ── Test: Never-diagnose on Pydantic results ───────────────────────────────


class TestStripDiagnosesFromResult:
    def test_strip_from_case_analysis(self):
        result = CaseAnalysis(
            key_features=["Symptoms of major depressive disorder"],
            presenting_patterns=["GAD pattern of worry"],
            framework_observations=["CBT-relevant: mdd cognitions"],
            severity_indicators=["PTSD flashbacks"],
            strengths_and_resources=["Good insight"],
            analysis_summary="Analysis with MDD and GAD references",
        )
        cleaned = _strip_diagnoses_from_result(result)
        assert "major depressive disorder" not in str(cleaned)
        assert "GAD" not in str(cleaned)
        assert "PTSD" not in str(cleaned)
        assert "MDD" not in str(cleaned)

    def test_strip_from_evidence_result(self):
        result = EvidenceResult(
            retrieved=[],
            synthesis="Evidence for MDD treatment",
            key_findings=["CBT effective for GAD"],
            evidence_gaps=["No data on BPD"],
            confidence=0.5,
        )
        cleaned = _strip_diagnoses_from_result(result)
        assert "MDD" not in cleaned.synthesis
        assert "GAD" not in str(cleaned.key_findings)
        assert "BPD" not in str(cleaned.evidence_gaps)

    def test_strip_from_conceptualization(self):
        result = CaseConceptualization(
            framework_formulation="Core beliefs in depression (MDD)",
            maintaining_factors=["Avoidance perpetuating GAD"],
            core_mechanisms=["Temporal lobe dysfunction in PTSD"],
            developmental_factors=["Early trauma"],
            conceptualization_summary="Summary with BPD traits",
        )
        cleaned = _strip_diagnoses_from_result(result)
        assert "MDD" not in cleaned.framework_formulation
        assert "GAD" not in str(cleaned.maintaining_factors)
        assert "PTSD" not in str(cleaned.core_mechanisms)

    def test_strip_from_followup_questions(self):
        result = FollowupQuestionsResult(
            assessment_questions=["Have you been diagnosed with OCD?"],
            therapeutic_probes=["How does the GAD affect you?"],
            outcome_monitoring=["PHQ-9 for MDD"],
            engagement_questions=["How comfortable are you?"],
        )
        cleaned = _strip_diagnoses_from_result(result)
        assert "OCD" not in str(cleaned.assessment_questions)
        assert "GAD" not in str(cleaned.therapeutic_probes)
        assert "MDD" not in str(cleaned.outcome_monitoring)

    def test_strip_from_limitations(self):
        result = LimitationsResult(
            contraindications=["PTSD with dissociation"],
            precautions=["Monitor for MDD worsening"],
            alternative_frameworks=["CFT for BPD"],
            caveats=["Caveat about OCD treatment"],
        )
        cleaned = _strip_diagnoses_from_result(result)
        assert "PTSD" not in str(cleaned.contraindications)
        assert "MDD" not in str(cleaned.precautions)
        assert "BPD" not in str(cleaned.alternative_frameworks)

    def test_strip_from_session_objectives(self):
        result = SessionObjectivesResult(
            immediate_objectives=["Psychoeducation on MDD"],
            short_term_goals=["Challenge GAD cognitions"],
            long_term_goals=["Relapse prevention for OCD"],
            progress_indicators=["Reduced avoidance in SAD"],
        )
        cleaned = _strip_diagnoses_from_result(result)
        assert "MDD" not in str(cleaned.immediate_objectives)
        assert "GAD" not in str(cleaned.short_term_goals)
        assert "OCD" not in str(cleaned.long_term_goals)
        assert "SAD" not in str(cleaned.progress_indicators)

    def test_strip_empty_fields_unchanged(self):
        result = CaseAnalysis(
            key_features=[],
            presenting_patterns=[],
            framework_observations=[],
            severity_indicators=[],
            strengths_and_resources=[],
            analysis_summary="No diagnostic labels here",
        )
        cleaned = _strip_diagnoses_from_result(result)
        assert cleaned.analysis_summary == "No diagnostic labels here"


# ── Test: _build_search_query ──────────────────────────────────────────────


class TestBuildSearchQuery:
    def test_build_from_case_input(self, case_input: FrameworkCaseInput):
        query = CBTPlugin._build_search_query(case_input)
        assert isinstance(query, str)
        assert len(query) > 20

    def test_build_from_raw_text(self):
        inp = FrameworkCaseInput(raw_text="Client reports feeling anxious in crowds")
        query = CBTPlugin._build_search_query(inp)
        assert "anxious" in query

    def test_build_fallback_for_empty_input(self):
        inp = FrameworkCaseInput()
        query = CBTPlugin._build_search_query(inp)
        assert query == "CBT evidence for clinical case"


# ── Test: Lifecycle methods return correct types (LLM=None) ────────────────


class TestLifecycleMethods:
    async def test_analyze_case_returns_default(self, plugin: CBTPlugin, case_input):
        result = await plugin.analyze_case(case_input, llm=None)
        assert isinstance(result, CaseAnalysis)
        assert "Analysis unavailable" in result.analysis_summary

    async def test_retrieve_evidence_returns_default(self, plugin: CBTPlugin, case_input):
        result = await plugin.retrieve_evidence(case_input, llm=None)
        assert isinstance(result, EvidenceResult)
        assert result.confidence == 0.0

    async def test_case_conceptualization_returns_default(self, plugin: CBTPlugin, case_input):
        result = await plugin.generate_case_conceptualization(case_input, llm=None)
        assert isinstance(result, CaseConceptualization)
        assert "Unable to generate" in result.conceptualization_summary

    async def test_therapeutic_focus_returns_default(self, plugin: CBTPlugin, case_input):
        result = await plugin.generate_therapeutic_focus(case_input, llm=None)
        assert isinstance(result, TherapeuticFocusResult)
        assert "Unable to determine focus" in result.primary_focus_areas

    async def test_intervention_plan_returns_default(self, plugin: CBTPlugin, case_input):
        result = await plugin.generate_intervention_plan(case_input, llm=None)
        assert isinstance(result, InterventionPlanResult)
        assert "Plan unavailable" in result.provisional_strategy

    async def test_session_objectives_returns_default(self, plugin: CBTPlugin, case_input):
        result = await plugin.generate_session_objectives(case_input, llm=None)
        assert isinstance(result, SessionObjectivesResult)
        assert "Unable to generate goals" in result.short_term_goals

    async def test_followup_questions_returns_default(self, plugin: CBTPlugin, case_input):
        result = await plugin.generate_followup_questions(case_input, llm=None)
        assert isinstance(result, FollowupQuestionsResult)
        assert all(getattr(result, f) == [] for f in
                   ["assessment_questions", "therapeutic_probes",
                    "outcome_monitoring", "engagement_questions"])

    async def test_limitations_returns_default(self, plugin: CBTPlugin, case_input):
        result = await plugin.generate_limitations(case_input, llm=None)
        assert isinstance(result, LimitationsResult)
        assert "Limitations unavailable" in result.caveats[0]

    async def test_run_full_analysis_aggregates_results(self, plugin: CBTPlugin, case_input):
        result = await plugin.run_full_analysis(case_input, llm=None)
        assert result.metadata.id == "cbt"
        assert isinstance(result.case_analysis, CaseAnalysis)
        assert isinstance(result.evidence, EvidenceResult)
        assert isinstance(result.conceptualization, CaseConceptualization)
        assert isinstance(result.therapeutic_focus, TherapeuticFocusResult)
        assert isinstance(result.intervention_plan, InterventionPlanResult)
        assert isinstance(result.session_objectives, SessionObjectivesResult)
        assert isinstance(result.followup_questions, FollowupQuestionsResult)
        assert isinstance(result.limitations, LimitationsResult)
        assert result.processing_ms >= 0.0


# ── Test: Prompt templates ─────────────────────────────────────────────────


class TestCBTPrompts:
    def test_all_prompts_load_via_manager(self):
        mgr = PromptManager.for_framework("cbt")
        tmpl = mgr.list_templates("cbt")
        assert len(tmpl) == 8

    def test_analyze_case_prompt_exists(self):
        mgr = PromptManager.for_framework("cbt")
        rendered = mgr.render("cbt", "analyze_case", {
            "system_header": "test",
            "case_text": "test",
            "output_schema": "{}",
        })
        assert rendered is not None
        assert "automatic thoughts" in rendered.lower()
        assert "cognitive distortions" in rendered.lower()
        assert "never assign" in rendered.lower()

    def test_retrieve_evidence_prompt_exists(self):
        mgr = PromptManager.for_framework("cbt")
        rendered = mgr.render("cbt", "retrieve_evidence", {
            "system_header": "test",
            "case_text": "test",
            "output_schema": "{}",
        })
        assert rendered is not None
        assert "meta-analyses" in rendered.lower() or "outcome research" in rendered.lower()

    def test_case_conceptualization_prompt_exists(self):
        mgr = PromptManager.for_framework("cbt")
        rendered = mgr.render("cbt", "case_conceptualization", {
            "system_header": "test",
            "case_text": "test",
            "output_schema": "{}",
        })
        assert rendered is not None
        assert "core beliefs" in rendered.lower()
        assert "intermediate beliefs" in rendered.lower()

    def test_intervention_plan_prompt_exists(self):
        mgr = PromptManager.for_framework("cbt")
        rendered = mgr.render("cbt", "intervention_plan", {
            "system_header": "test",
            "case_text": "test",
            "output_schema": "{}",
        })
        assert rendered is not None
        assert "cognitive restructuring" in rendered.lower()
        assert "behavioural" in rendered.lower()

    def test_therapeutic_focus_prompt_exists(self):
        mgr = PromptManager.for_framework("cbt")
        rendered = mgr.render("cbt", "therapeutic_focus", {
            "system_header": "test",
            "case_text": "test",
            "output_schema": "{}",
        })
        assert rendered is not None
        assert "primary" in rendered.lower()
        assert "staging" in rendered.lower()

    def test_session_objectives_prompt_exists(self):
        mgr = PromptManager.for_framework("cbt")
        rendered = mgr.render("cbt", "session_objectives", {
            "system_header": "test",
            "case_text": "test",
            "output_schema": "{}",
        })
        assert rendered is not None
        assert "agenda" in rendered.lower() or "homework" in rendered.lower()

    def test_followup_questions_prompt_exists(self):
        mgr = PromptManager.for_framework("cbt")
        rendered = mgr.render("cbt", "followup_questions", {
            "system_header": "test",
            "case_text": "test",
            "output_schema": "{}",
        })
        assert rendered is not None
        assert "?" in rendered

    def test_limitations_prompt_exists(self):
        mgr = PromptManager.for_framework("cbt")
        rendered = mgr.render("cbt", "limitations", {
            "system_header": "test",
            "case_text": "test",
            "output_schema": "{}",
        })
        assert rendered is not None
        assert "contraindications" in rendered.lower() or "alternative" in rendered.lower()

    def test_every_prompt_has_never_diagnose(self):
        mgr = PromptManager.for_framework("cbt")
        for method in ["analyze_case", "retrieve_evidence", "case_conceptualization",
                       "therapeutic_focus", "intervention_plan", "session_objectives",
                       "followup_questions", "limitations"]:
            rendered = mgr.render("cbt", method, {
                "system_header": "test",
                "case_text": "test",
                "output_schema": "{}",
            })
            assert rendered is not None, f"Failed to render {method}"
            assert "never assign" in rendered.lower() or "never" in rendered.lower(), (
                f"Missing never-diagnose in {method}"
            )

    def test_all_templates_have_upgraded_version(self):
        mgr = PromptManager.for_framework("cbt")
        template_names = mgr.list_templates("cbt")
        for name in template_names:
            tmpl = mgr.get_template("cbt", name)
            assert tmpl is not None, f"Template {name} not found"
            assert tmpl.version >= "1.1.0", (
                f"Template {name} has old version {tmpl.version}"
            )


# ── Test: Vector retrieval graceful degradation ────────────────────────────


class TestVectorRetrieval:
    async def test_try_vector_retrieval_returns_none_when_no_collection(
        self, plugin: CBTPlugin, case_input,
    ):
        result = await plugin._try_vector_retrieval(case_input)
        assert result is None

    async def test_retrieve_evidence_falls_back_gracefully(
        self, plugin: CBTPlugin, case_input,
    ):
        result = await plugin.retrieve_evidence(case_input, llm=None)
        assert isinstance(result, EvidenceResult)
        assert result.confidence == 0.0


# ── Test: Registry integration ─────────────────────────────────────────────


class TestCBTRegistryIntegration:
    def test_registry_discovers_cbt(self):
        registry = FrameworkRegistry.get_instance()
        registry.discover()
        assert "cbt" in registry
        meta = registry.get_metadata("cbt")
        assert meta.id == "cbt"
        assert meta.display_name == "Cognitive Behavioral Therapy"

    def test_cbt_is_applicable_through_registry(self):
        registry = FrameworkRegistry.get_instance()
        registry.discover()
        cls = registry.get("cbt")
        assert cls.is_applicable({"presenting_concerns": "anxiety and depression"})
        assert not cls.is_applicable({"presenting_concerns": "career planning"})
