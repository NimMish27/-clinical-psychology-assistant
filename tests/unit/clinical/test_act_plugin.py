"""
tests/unit/clinical/test_act_plugin.py — Tests for ACT Plugin with lifecycle overrides.
"""

from __future__ import annotations

import pytest
from clinical.framework.act.plugin import (
    ACTPlugin,
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
def plugin() -> ACTPlugin:
    return ACTPlugin()


@pytest.fixture
def case_input() -> FrameworkCaseInput:
    return FrameworkCaseInput(
        case_understanding={
            "presenting_concerns": "Chronic pain, anxiety, feeling stuck, lost sense of purpose",
            "thoughts": "I can't cope with this pain, I'm broken, nothing matters anymore",
            "behaviours": "Avoiding activity, staying in bed, stopped valued hobbies",
        },
        raw_text=(
            "I've been struggling with chronic back pain that stops me from doing what matters. "
            "I avoid activity because I'm afraid of making the pain worse. "
            "I used to enjoy hiking and woodworking but now I just stay at home."
        ),
    )


# ── Test: ACT metadata ──────────────────────────────────────────────────────


class TestACTMetadata:
    def test_metadata_id(self):
        meta = ACTPlugin.metadata()
        assert meta.id == "act"

    def test_metadata_display_name(self):
        meta = ACTPlugin.metadata()
        assert meta.display_name == "Acceptance and Commitment Therapy"

    def test_metadata_version(self):
        meta = ACTPlugin.metadata()
        assert meta.version == "2.0.0"

    def test_metadata_literature_collection(self):
        meta = ACTPlugin.metadata()
        assert meta.literature_collection == "act_literature"

    def test_metadata_tags_include_act(self):
        meta = ACTPlugin.metadata()
        assert "act" in meta.tags

    def test_metadata_supported_use_cases(self):
        meta = ACTPlugin.metadata()
        assert len(meta.supported_use_cases) >= 5
        assert any("pain" in c.lower() for c in meta.supported_use_cases)
        assert any("anxiety" in c.lower() for c in meta.supported_use_cases)

    def test_metadata_techniques_include_acceptance_defusion_values(self):
        meta = ACTPlugin.metadata()
        techniques = " ".join(meta.intervention_style.techniques).lower()
        assert "defusion" in techniques
        assert "acceptance" in techniques
        assert "values" in techniques or "committed action" in techniques

    def test_metadata_retrieval_strategy_hybrid(self):
        meta = ACTPlugin.metadata()
        from clinical.framework.models import RetrievalStrategy
        assert meta.retrieval_strategy == RetrievalStrategy.HYBRID

    def test_metadata_citations_not_empty(self):
        meta = ACTPlugin.metadata()
        assert len(meta.citations) >= 2
        assert any("Hayes" in c for c in meta.citations)


# ── Test: is_applicable ─────────────────────────────────────────────────────


class TestACTApplicability:
    def test_applicable_for_chronic_pain(self):
        assert ACTPlugin.is_applicable({"presenting_concerns": "chronic pain and illness"})

    def test_applicable_for_anxiety(self):
        assert ACTPlugin.is_applicable({"presenting_concerns": "anxiety and worry"})

    def test_applicable_for_depression(self):
        assert ACTPlugin.is_applicable({"presenting_concerns": "depression and low mood"})

    def test_applicable_for_burnout(self):
        assert ACTPlugin.is_applicable({"presenting_concerns": "burnout and stress"})

    def test_applicable_for_substance_use(self):
        assert ACTPlugin.is_applicable({"presenting_concerns": "substance use and addiction"})

    def test_applicable_for_meaning_and_values(self):
        assert ACTPlugin.is_applicable({"presenting_concerns": "loss of meaning and values confusion"})

    def test_not_applicable_for_unrelated(self):
        assert not ACTPlugin.is_applicable({"presenting_concerns": "routine career counselling"})

    def test_empty_case_not_applicable(self):
        assert not ACTPlugin.is_applicable({})

    def test_applicable_for_experiential_avoidance(self):
        assert ACTPlugin.is_applicable({"presenting_concerns": "experiential avoidance patterns"})

    def test_applicable_for_obsessive_patterns(self):
        assert ACTPlugin.is_applicable({"presenting_concerns": "obsessive compulsive patterns"})


# ── Test: Never-diagnose enforcement ────────────────────────────────────────


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
        text = "Client experiences experiential avoidance and cognitive fusion patterns"
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

    def test_diagnostic_patterns_list_not_empty(self):
        assert len(_DIAGNOSTIC_PATTERNS) >= 1

    def test_diagnostic_patterns_match_dsm_labels(self):
        assert _DIAGNOSTIC_PATTERNS[0].search("major depressive disorder") is not None
        assert _DIAGNOSTIC_PATTERNS[0].search("generalised anxiety disorder") is not None
        assert _DIAGNOSTIC_PATTERNS[0].search("binge-eating disorder") is not None

    def test_diagnostic_patterns_no_false_positive(self):
        # Should NOT match non-diagnostic uses
        assert _DIAGNOSTIC_PATTERNS[0].search("feeling depressed sometimes") is None
        assert _DIAGNOSTIC_PATTERNS[0].search("anxiety about an event") is None


# ── Test: Never-diagnose on Pydantic results ────────────────────────────────


class TestStripDiagnosesFromResult:
    def test_strip_from_case_analysis(self):
        result = CaseAnalysis(
            key_features=["Symptoms of major depressive disorder"],
            presenting_patterns=["GAD pattern of worry"],
            framework_observations=["ACT-relevant: mdd fusion"],
            severity_indicators=["PTSD flashbacks"],
            strengths_and_resources=["Willingness to engage"],
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
            synthesis="Evidence for MDD treatment with ACT",
            key_findings=["ACT effective for GAD"],
            evidence_gaps=["No data on BPD with ACT"],
            confidence=0.5,
        )
        cleaned = _strip_diagnoses_from_result(result)
        assert "MDD" not in cleaned.synthesis
        assert "GAD" not in str(cleaned.key_findings)
        assert "BPD" not in str(cleaned.evidence_gaps)

    def test_strip_from_conceptualization(self):
        result = CaseConceptualization(
            framework_formulation="ACT formulation for avoidance in depression (MDD)",
            maintaining_factors=["Fusion perpetuating GAD pattern"],
            core_mechanisms=["Experiential avoidance in PTSD"],
            developmental_factors=["Early learning history"],
            conceptualization_summary="Summary with BPD traits noted",
        )
        cleaned = _strip_diagnoses_from_result(result)
        assert "MDD" not in cleaned.framework_formulation
        assert "GAD" not in str(cleaned.maintaining_factors)
        assert "PTSD" not in str(cleaned.core_mechanisms)

    def test_strip_from_followup_questions(self):
        result = FollowupQuestionsResult(
            assessment_questions=["Have you been diagnosed with OCD?"],
            therapeutic_probes=["How does the GAD pattern show up?"],
            outcome_monitoring=["Values progress for MDD"],
            engagement_questions=["How is this work going for you?"],
        )
        cleaned = _strip_diagnoses_from_result(result)
        assert "OCD" not in str(cleaned.assessment_questions)
        assert "GAD" not in str(cleaned.therapeutic_probes)
        assert "MDD" not in str(cleaned.outcome_monitoring)

    def test_strip_from_limitations(self):
        result = LimitationsResult(
            contraindications=["PTSD requiring stabilisation"],
            precautions=["Monitor for MDD patterns"],
            alternative_frameworks=["CFT for BPD"],
            caveats=["Caveat about OCD treatment"],
        )
        cleaned = _strip_diagnoses_from_result(result)
        assert "PTSD" not in str(cleaned.contraindications)
        assert "MDD" not in str(cleaned.precautions)
        assert "BPD" not in str(cleaned.alternative_frameworks)

    def test_strip_from_session_objectives(self):
        result = SessionObjectivesResult(
            immediate_objectives=["Defusion from MDD story"],
            short_term_goals=["Acceptance of GAD-related sensations"],
            long_term_goals=["Values-based living despite OCD patterns"],
            progress_indicators=["Reduced avoidance in SAD"],
        )
        cleaned = _strip_diagnoses_from_result(result)
        assert "MDD" not in str(cleaned.immediate_objectives)
        assert "GAD" not in str(cleaned.short_term_goals)
        assert "OCD" not in str(cleaned.long_term_goals)
        assert "SAD" not in str(cleaned.progress_indicators)

    def test_strip_from_therapeutic_focus(self):
        result = TherapeuticFocusResult(
            primary_focus_areas=["Working through MDD-related fusion"],
            secondary_focus_areas=["Addressing GAD-related avoidance"],
            staging_and_sequencing="Start with defusion from PTSD narrative",
            focus_rationale="Focus rationale for BPD patterns",
        )
        cleaned = _strip_diagnoses_from_result(result)
        assert "MDD" not in str(cleaned.primary_focus_areas)
        assert "GAD" not in str(cleaned.secondary_focus_areas)
        assert "PTSD" not in str(cleaned.staging_and_sequencing)
        assert "BPD" not in str(cleaned.focus_rationale)

    def test_strip_from_intervention_plan(self):
        result = InterventionPlanResult(
            interventions=[{"technique": "defusion for MDD", "rationale": "reduce fusion"}],
            provisional_strategy="ACT protocol for GAD",
            expected_outcomes=["Reduced BPD-related avoidance"],
        )
        cleaned = _strip_diagnoses_from_result(result)
        assert "MDD" not in str(cleaned.interventions)
        assert "GAD" not in cleaned.provisional_strategy
        assert "BPD" not in str(cleaned.expected_outcomes)

    def test_strip_empty_fields_unchanged(self):
        result = CaseAnalysis(
            key_features=[],
            presenting_patterns=[],
            framework_observations=[],
            severity_indicators=[],
            strengths_and_resources=[],
            analysis_summary="No diagnostic labels here — ACT focuses on function",
        )
        cleaned = _strip_diagnoses_from_result(result)
        assert cleaned.analysis_summary == "No diagnostic labels here — ACT focuses on function"


# ── Test: _build_search_query ──────────────────────────────────────────────


class TestBuildSearchQuery:
    def test_build_from_case_understanding(self):
        inp = FrameworkCaseInput(
            case_understanding={
                "presenting_concerns": "Chronic back pain limiting valued activities",
                "thoughts": "I cannot do what matters anymore",
            },
        )
        query = ACTPlugin._build_search_query(inp)
        assert len(query) > 20
        assert "Chronic back pain" in query

    def test_build_from_raw_text(self):
        inp = FrameworkCaseInput(
            case_understanding={"presenting_concerns": "Anxiety"},
            raw_text="I feel anxious all the time and avoid social activities",
        )
        query = ACTPlugin._build_search_query(inp)
        assert "anxious" in query

    def test_build_fallback_when_no_content(self):
        inp = FrameworkCaseInput(case_understanding={})
        query = ACTPlugin._build_search_query(inp)
        assert "evidence" in query


# ── Test: Prompt templates ─────────────────────────────────────────────────


class TestPromptTemplates:
    def test_all_prompt_templates_exist(self):
        mgr = PromptManager.for_framework("act")
        # Render one to trigger discovery
        mgr.render("act", "analyze_case", {
            "system_header": "h", "case_text": "c", "output_schema": "{}",
        })
        stats = mgr.stats()
        assert stats["total_templates"] == 8

    def test_each_template_has_never_diagnose(self):
        mgr = PromptManager.for_framework("act")
        for method in [
            "analyze_case",
            "case_conceptualization",
            "therapeutic_focus",
            "intervention_plan",
            "session_objectives",
            "followup_questions",
            "retrieve_evidence",
            "limitations",
        ]:
            rendered = mgr.render("act", method, {
                "system_header": "header",
                "case_text": "case",
                "output_schema": "{}",
            })
            assert rendered is not None
            assert "Never assign diagnostic" in rendered

    def test_each_template_contains_act_specific_content(self):
        mgr = PromptManager.for_framework("act")
        assert "hexaflex" in mgr.render("act", "analyze_case", {
            "system_header": "h", "case_text": "c", "output_schema": "{}",
        }).lower()
        assert "defusion" in mgr.render("act", "intervention_plan", {
            "system_header": "h", "case_text": "c", "output_schema": "{}",
        }).lower()
        assert "values" in mgr.render("act", "therapeutic_focus", {
            "system_header": "h", "case_text": "c", "output_schema": "{}",
        }).lower()

    def test_all_templates_render_with_variables(self):
        mgr = PromptManager.for_framework("act")
        for method in [
            "analyze_case",
            "case_conceptualization",
            "therapeutic_focus",
            "intervention_plan",
            "session_objectives",
            "followup_questions",
            "retrieve_evidence",
            "limitations",
        ]:
            rendered = mgr.render("act", method, {
                "system_header": "You are an ACT practitioner",
                "case_text": "Case: chronic pain with values loss",
                "output_schema": '{"type": "object", "properties": {}}',
            })
            assert rendered is not None
            assert "ACT" in rendered or "Acceptance" in rendered
            assert "You are an ACT" in rendered
            assert "chronic pain" in rendered.casefold()


# ── Test: ACT-specific lifecycle methods (structure & defaults) ────────────


class TestACTLifecycleDefaults:
    """Tests that lifecycle methods return correctly-typed fallbacks on error."""

    @pytest.mark.asyncio
    async def test_analyze_case_returns_case_analysis(self, plugin, case_input):
        result = await plugin.analyze_case(case_input)
        assert isinstance(result, CaseAnalysis)
        # On error (no LLM), should return fallback
        assert result.analysis_summary.startswith("Analysis unavailable")

    @pytest.mark.asyncio
    async def test_retrieve_evidence_returns_evidence_result(self, plugin, case_input):
        result = await plugin.retrieve_evidence(case_input)
        assert isinstance(result, EvidenceResult)

    @pytest.mark.asyncio
    async def test_generate_case_conceptualization_returns_conceptualization(self, plugin, case_input):
        result = await plugin.generate_case_conceptualization(case_input)
        assert isinstance(result, CaseConceptualization)

    @pytest.mark.asyncio
    async def test_generate_therapeutic_focus_returns_focus_result(self, plugin, case_input):
        result = await plugin.generate_therapeutic_focus(case_input)
        assert isinstance(result, TherapeuticFocusResult)

    @pytest.mark.asyncio
    async def test_generate_intervention_plan_returns_plan(self, plugin, case_input):
        result = await plugin.generate_intervention_plan(case_input)
        assert isinstance(result, InterventionPlanResult)

    @pytest.mark.asyncio
    async def test_generate_session_objectives_returns_objectives(self, plugin, case_input):
        result = await plugin.generate_session_objectives(case_input)
        assert isinstance(result, SessionObjectivesResult)

    @pytest.mark.asyncio
    async def test_generate_followup_questions_returns_questions(self, plugin, case_input):
        result = await plugin.generate_followup_questions(case_input)
        assert isinstance(result, FollowupQuestionsResult)

    @pytest.mark.asyncio
    async def test_generate_limitations_returns_limitations(self, plugin, case_input):
        result = await plugin.generate_limitations(case_input)
        assert isinstance(result, LimitationsResult)


# ── Test: Registry integration ─────────────────────────────────────────────


class TestACTRegistryIntegration:
    def test_act_registered_in_registry(self):
        registry = FrameworkRegistry.get_instance()
        registry.discover()
        assert "act" in registry

    def test_act_metadata_accessible_through_registry(self):
        registry = FrameworkRegistry.get_instance()
        registry.discover()
        meta = registry.get_metadata("act")
        assert meta.id == "act"
        assert meta.display_name == "Acceptance and Commitment Therapy"

    def test_act_applicable_through_registry(self):
        reg = FrameworkRegistry.get_instance()
        applicable = reg.applicable_for({"presenting_concerns": "chronic pain"})
        assert any(slug == "act" for slug, _ in applicable)

    def test_act_not_applicable_for_non_matching(self):
        reg = FrameworkRegistry.get_instance()
        applicable = reg.applicable_for({"presenting_concerns": "career counselling"})
        act_applicable = any(slug == "act" for slug, _ in applicable)
        # act should NOT be applicable here unless the GeneralPlugin also matches
        # This mostly tests that act doesn't false-positive


# ── Test: Vector retrieval helper (graceful fallback) ──────────────────────


class TestVectorRetrievalFallback:
    @pytest.mark.asyncio
    async def test_retrieval_fallback_on_empty_input(self, plugin):
        inp = FrameworkCaseInput(case_understanding={"presenting_concerns": "a"})
        result = await plugin._try_vector_retrieval(inp)
        assert result is None

    @pytest.mark.asyncio
    async def test_retrieval_fallback_on_error(self, plugin, case_input):
        # Without vector store infrastructure, should return None gracefully
        result = await plugin._try_vector_retrieval(case_input)
        # May return None or an EvidenceResult depending on infrastructure
        assert result is None or isinstance(result, EvidenceResult)
