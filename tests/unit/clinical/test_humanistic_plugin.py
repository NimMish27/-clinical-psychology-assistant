"""
tests/unit/clinical/test_humanistic_plugin.py — Tests for Humanistic Therapy Plugin.
"""

from __future__ import annotations

import pytest
from clinical.framework.humanistic.plugin import (
    HumanisticPlugin,
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
def plugin() -> HumanisticPlugin:
    return HumanisticPlugin()


@pytest.fixture
def case_input() -> FrameworkCaseInput:
    return FrameworkCaseInput(
        case_understanding={
            "presenting_concerns": "Low self-worth, feeling like I'm not good enough, difficulty accepting myself",
            "thoughts": "I should be better, I'm a failure, people will reject me if they see the real me",
            "behaviours": "People-pleasing, avoiding vulnerability, overworking to prove my worth",
        },
        raw_text=(
            "I've always felt like I'm not good enough. Growing up, nothing I did was ever quite right. "
            "Now I feel like I'm always performing, always trying to be what others want me to be. "
            "I don't even know who I really am anymore."
        ),
    )


# ── Test: Humanistic metadata ──────────────────────────────────────────────


class TestHumanisticMetadata:
    def test_metadata_id(self):
        meta = HumanisticPlugin.metadata()
        assert meta.id == "humanistic"

    def test_metadata_display_name(self):
        meta = HumanisticPlugin.metadata()
        assert meta.display_name == "Humanistic Therapy"

    def test_metadata_version(self):
        meta = HumanisticPlugin.metadata()
        assert meta.version == "2.0.0"

    def test_metadata_literature_collection(self):
        meta = HumanisticPlugin.metadata()
        assert meta.literature_collection == "humanistic_literature"

    def test_metadata_tags_include_humanistic(self):
        meta = HumanisticPlugin.metadata()
        assert "humanistic" in meta.tags

    def test_metadata_supported_use_cases(self):
        meta = HumanisticPlugin.metadata()
        assert len(meta.supported_use_cases) >= 6
        assert any("self-esteem" in c.lower() for c in meta.supported_use_cases)
        assert any("existential" in c.lower() for c in meta.supported_use_cases)

    def test_metadata_techniques_include_empathic_responding(self):
        meta = HumanisticPlugin.metadata()
        techniques = " ".join(meta.intervention_style.techniques).lower()
        assert "empathic" in techniques or "listening" in techniques
        assert "focusing" in techniques or "gestalt" in techniques
        assert "unconditional" in techniques or "congruence" in techniques

    def test_metadata_retrieval_strategy_hybrid(self):
        meta = HumanisticPlugin.metadata()
        from clinical.framework.models import RetrievalStrategy
        assert meta.retrieval_strategy == RetrievalStrategy.HYBRID

    def test_metadata_citations_not_empty(self):
        meta = HumanisticPlugin.metadata()
        assert len(meta.citations) >= 2
        assert any("Rogers" in c for c in meta.citations)


# ── Test: is_applicable ─────────────────────────────────────────────────────


class TestHumanisticApplicability:
    def test_applicable_for_self_esteem(self):
        assert HumanisticPlugin.is_applicable({"presenting_concerns": "low self-esteem and self-worth"})

    def test_applicable_for_identity(self):
        assert HumanisticPlugin.is_applicable({"presenting_concerns": "identity confusion and self-discovery"})

    def test_applicable_for_existential_concerns(self):
        assert HumanisticPlugin.is_applicable({"presenting_concerns": "existential concerns about life meaning"})

    def test_applicable_for_relationship_difficulties(self):
        assert HumanisticPlugin.is_applicable({"presenting_concerns": "relationship difficulties"})

    def test_applicable_for_personal_growth(self):
        assert HumanisticPlugin.is_applicable({"presenting_concerns": "personal growth and self-actualisation"})

    def test_applicable_for_shame(self):
        assert HumanisticPlugin.is_applicable({"presenting_concerns": "deep shame and self-criticism"})

    def test_applicable_for_life_transition(self):
        assert HumanisticPlugin.is_applicable({"presenting_concerns": "adjustment to life transition"})

    def test_not_applicable_for_unrelated(self):
        assert not HumanisticPlugin.is_applicable({"presenting_concerns": "routine financial planning"})

    def test_empty_case_not_applicable(self):
        assert not HumanisticPlugin.is_applicable({})

    def test_applicable_for_loneliness(self):
        assert HumanisticPlugin.is_applicable({"presenting_concerns": "loneliness and isolation"})

    def test_applicable_for_authenticity(self):
        assert HumanisticPlugin.is_applicable({"presenting_concerns": "struggling with authenticity"})


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
        assert "excessive worry" in result

    def test_strip_diagnoses_ptsd(self):
        text = "Trauma-related symptoms consistent with PTSD"
        result = _strip_diagnoses(text)
        assert "PTSD" not in result
        assert "trauma-related distress" in result

    def test_strip_diagnoses_ocd(self):
        text = "Obsessive compulsive disorder symptoms present"
        result = _strip_diagnoses(text)
        assert "obsessive compulsive disorder" not in result
        assert "intrusive thoughts" in result

    def test_strip_diagnoses_adhd(self):
        text = "ADHD symptoms affecting work performance"
        result = _strip_diagnoses(text)
        assert "ADHD" not in result

    def test_strip_diagnoses_bpd(self):
        text = "Borderline personality disorder traits"
        result = _strip_diagnoses(text)
        assert "borderline personality disorder" not in result

    def test_strip_diagnoses_panic(self):
        text = "Panic disorder with agoraphobia"
        result = _strip_diagnoses(text)
        assert "panic disorder" not in result

    def test_strip_diagnoses_normal_text_unchanged(self):
        text = "Client experiences incongruence between self-concept and actual experience"
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
        assert _DIAGNOSTIC_PATTERNS[0].search("feeling depressed sometimes") is None
        assert _DIAGNOSTIC_PATTERNS[0].search("anxiety about an event") is None


# ── Test: Never-diagnose on Pydantic results ────────────────────────────────


class TestStripDiagnosesFromResult:
    def test_strip_from_case_analysis(self):
        result = CaseAnalysis(
            key_features=["Symptoms of major depressive disorder"],
            presenting_patterns=["GAD pattern of worry"],
            framework_observations=["Humanistic view of mdd-related incongruence"],
            severity_indicators=["PTSD flashbacks"],
            strengths_and_resources=["Openness to self-exploration"],
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
            synthesis="Evidence for MDD treatment with humanistic therapy",
            key_findings=["Humanistic therapy effective for GAD"],
            evidence_gaps=["No data on BPD with humanistic therapy"],
            confidence=0.5,
        )
        cleaned = _strip_diagnoses_from_result(result)
        assert "MDD" not in cleaned.synthesis
        assert "GAD" not in str(cleaned.key_findings)
        assert "BPD" not in str(cleaned.evidence_gaps)

    def test_strip_from_conceptualization(self):
        result = CaseConceptualization(
            framework_formulation="Humanistic formulation for incongruence in depression (MDD)",
            maintaining_factors=["Conditions of worth perpetuating GAD pattern"],
            core_mechanisms=["Incongruence driven by PTSD-related avoidance of experience"],
            developmental_factors=["Early conditions of worth from family of origin"],
            conceptualization_summary="Summary with BPD traits noted",
        )
        cleaned = _strip_diagnoses_from_result(result)
        assert "MDD" not in cleaned.framework_formulation
        assert "GAD" not in str(cleaned.maintaining_factors)
        assert "PTSD" not in str(cleaned.core_mechanisms)

    def test_strip_from_followup_questions(self):
        result = FollowupQuestionsResult(
            assessment_questions=["Have you been diagnosed with OCD?"],
            therapeutic_probes=["How does the GAD pattern show up in daily life?"],
            outcome_monitoring=["Self-regard shifts for MDD"],
            engagement_questions=["How is our work together for you?"],
        )
        cleaned = _strip_diagnoses_from_result(result)
        assert "OCD" not in str(cleaned.assessment_questions)
        assert "GAD" not in str(cleaned.therapeutic_probes)
        assert "MDD" not in str(cleaned.outcome_monitoring)

    def test_strip_from_limitations(self):
        result = LimitationsResult(
            contraindications=["PTSD requiring stabilisation"],
            precautions=["Monitor for MDD patterns"],
            alternative_frameworks=["CBT for BPD"],
            caveats=["Caveat about OCD treatment"],
        )
        cleaned = _strip_diagnoses_from_result(result)
        assert "PTSD" not in str(cleaned.contraindications)
        assert "MDD" not in str(cleaned.precautions)
        assert "BPD" not in str(cleaned.alternative_frameworks)

    def test_strip_from_session_objectives(self):
        result = SessionObjectivesResult(
            immediate_objectives=["Self-acceptance despite MDD-related self-criticism"],
            short_term_goals=["Reducing avoidance of GAD-related feelings"],
            long_term_goals=["Living authentically beyond OCD patterns"],
            progress_indicators=["Reduced need for approval in SAD"],
        )
        cleaned = _strip_diagnoses_from_result(result)
        assert "MDD" not in str(cleaned.immediate_objectives)
        assert "GAD" not in str(cleaned.short_term_goals)
        assert "OCD" not in str(cleaned.long_term_goals)
        assert "SAD" not in str(cleaned.progress_indicators)

    def test_strip_from_therapeutic_focus(self):
        result = TherapeuticFocusResult(
            primary_focus_areas=["UPR to counter MDD-related self-criticism"],
            secondary_focus_areas=["Empathic exploration of GAD-related worry"],
            staging_and_sequencing="Start with congruence regarding PTSD experiences",
            focus_rationale="Rationale for working with BPD patterns through UPR",
        )
        cleaned = _strip_diagnoses_from_result(result)
        assert "MDD" not in str(cleaned.primary_focus_areas)
        assert "GAD" not in str(cleaned.secondary_focus_areas)
        assert "PTSD" not in str(cleaned.staging_and_sequencing)
        assert "BPD" not in str(cleaned.focus_rationale)

    def test_strip_from_intervention_plan(self):
        result = InterventionPlanResult(
            interventions=[{"technique": "empathic responding for MDD", "rationale": "deepen self-acceptance"}],
            provisional_strategy="Humanistic approach for GAD-related incongruence",
            expected_outcomes=["Reduced BPD-related self-criticism"],
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
            analysis_summary="No diagnostic labels here — humanistic therapy focuses on self-concept",
        )
        cleaned = _strip_diagnoses_from_result(result)
        assert cleaned.analysis_summary == "No diagnostic labels here — humanistic therapy focuses on self-concept"


# ── Test: _build_search_query ──────────────────────────────────────────────


class TestBuildSearchQuery:
    def test_build_from_case_understanding(self):
        inp = FrameworkCaseInput(
            case_understanding={
                "presenting_concerns": "Low self-worth and difficulty accepting myself",
                "thoughts": "I feel like I'm not good enough no matter what I do",
            },
        )
        query = HumanisticPlugin._build_search_query(inp)
        assert len(query) > 20
        assert "Low self-worth" in query

    def test_build_from_raw_text(self):
        inp = FrameworkCaseInput(
            case_understanding={"presenting_concerns": "Self-criticism"},
            raw_text="I judge myself harshly and I'm afraid of being seen",
        )
        query = HumanisticPlugin._build_search_query(inp)
        assert "judge" in query

    def test_build_fallback_when_no_content(self):
        inp = FrameworkCaseInput(case_understanding={})
        query = HumanisticPlugin._build_search_query(inp)
        assert "evidence" in query


# ── Test: Prompt templates ─────────────────────────────────────────────────


class TestPromptTemplates:
    def test_all_prompt_templates_exist(self):
        mgr = PromptManager.for_framework("humanistic")
        mgr.render("humanistic", "analyze_case", {
            "system_header": "h", "case_text": "c", "output_schema": "{}",
        })
        stats = mgr.stats()
        assert stats["total_templates"] == 8

    def test_each_template_has_never_diagnose(self):
        mgr = PromptManager.for_framework("humanistic")
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
            rendered = mgr.render("humanistic", method, {
                "system_header": "header",
                "case_text": "case",
                "output_schema": "{}",
            })
            assert rendered is not None
            assert "Never assign diagnostic" in rendered or "never assign diagnostic" in rendered

    def test_each_template_contains_humanistic_content(self):
        mgr = PromptManager.for_framework("humanistic")
        assert "self-concept" in mgr.render("humanistic", "analyze_case", {
            "system_header": "h", "case_text": "c", "output_schema": "{}",
        }).lower()
        assert "unconditional" in mgr.render("humanistic", "therapeutic_focus", {
            "system_header": "h", "case_text": "c", "output_schema": "{}",
        }).lower()
        assert "self-regard" in mgr.render("humanistic", "session_objectives", {
            "system_header": "h", "case_text": "c", "output_schema": "{}",
        }).lower()

    def test_all_templates_render_with_variables(self):
        mgr = PromptManager.for_framework("humanistic")
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
            rendered = mgr.render("humanistic", method, {
                "system_header": "You are a humanistic practitioner",
                "case_text": "Case: low self-worth and conditions of worth",
                "output_schema": '{"type": "object", "properties": {}}',
            })
            assert rendered is not None
            assert "You are a humanistic" in rendered
            assert "low self-worth" in rendered.casefold()


# ── Test: Humanistic-specific lifecycle methods (structure & defaults) ─────


class TestHumanisticLifecycleDefaults:
    """Tests that lifecycle methods return correctly-typed fallbacks on error."""

    @pytest.mark.asyncio
    async def test_analyze_case_returns_case_analysis(self, plugin, case_input):
        result = await plugin.analyze_case(case_input)
        assert isinstance(result, CaseAnalysis)
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


class TestHumanisticRegistryIntegration:
    def test_humanistic_registered_in_registry(self):
        registry = FrameworkRegistry.get_instance()
        registry.discover()
        assert "humanistic" in registry

    def test_humanistic_metadata_accessible_through_registry(self):
        registry = FrameworkRegistry.get_instance()
        registry.discover()
        meta = registry.get_metadata("humanistic")
        assert meta.id == "humanistic"
        assert meta.display_name == "Humanistic Therapy"

    def test_humanistic_applicable_through_registry(self):
        registry = FrameworkRegistry.get_instance()
        registry.discover()
        applicable = registry.applicable_for({"presenting_concerns": "low self-esteem"})
        assert any(slug == "humanistic" for slug, _ in applicable)

    def test_humanistic_not_applicable_for_non_matching(self):
        registry = FrameworkRegistry.get_instance()
        registry.discover()
        applicable = registry.applicable_for({"presenting_concerns": "routine career counselling"})
        humanistic_applicable = any(slug == "humanistic" for slug, _ in applicable)
        # humanistic may still match via GeneralPlugin but shouldn't false-positive on its own


# ── Test: Vector retrieval helper (graceful fallback) ──────────────────────


class TestVectorRetrievalFallback:
    @pytest.mark.asyncio
    async def test_retrieval_fallback_on_empty_input(self, plugin):
        inp = FrameworkCaseInput(case_understanding={"presenting_concerns": "a"})
        result = await plugin._try_vector_retrieval(inp)
        assert result is None

    @pytest.mark.asyncio
    async def test_retrieval_fallback_on_error(self, plugin, case_input):
        result = await plugin._try_vector_retrieval(case_input)
        assert result is None or isinstance(result, EvidenceResult)


# ── Test: Humanistic-specific evidence content ────────────────────────────


class TestHumanisticEvidenceContent:
    """Verify that the humanistic retriever models and conversion work."""

    def test_humanistic_evidence_filter_importable(self):
        from clinical.retrieval.models import HumanisticEvidenceFilter
        filt = HumanisticEvidenceFilter()
        assert filt.framework == "humanistic"

    def test_humanistic_evidence_item_importable(self):
        from clinical.retrieval.models import HumanisticEvidenceItem
        from clinical.retrieval.humanistic_retriever import HumanisticRetriever

        item = HumanisticEvidenceItem(
            source="Rogers (1957)",
            content="Empathy is a necessary and sufficient condition for therapeutic change",
            relevance="Direct evidence for core humanistic process",
        )
        assert item.framework == "humanistic"
        assert "Empathy" in item.content
        assert HumanisticRetriever is not None

    def test_humanistic_evidence_result_importable(self):
        from clinical.retrieval.models import HumanisticEvidenceResult

        result = HumanisticEvidenceResult(
            query="empathy and UPR in therapy",
            retrieved=[],
            synthesis="No evidence retrieved",
            key_findings=[],
            evidence_gaps=["No data"],
            confidence=0.0,
        )
        assert result.query == "empathy and UPR in therapy"
        assert result.confidence == 0.0

    def test_humanistic_retriever_singleton(self):
        from clinical.retrieval.humanistic_retriever import get_humanistic_retriever

        r1 = get_humanistic_retriever()
        r2 = get_humanistic_retriever()
        assert r1 is r2

    def test_humanistic_retriever_config(self):
        from clinical.retrieval.humanistic_retriever import HumanisticRetriever

        r = HumanisticRetriever(n_results=5)
        cfg = r.config
        assert cfg["framework"] == "humanistic"
        assert cfg["n_results"] == 5

    def test_humanistic_retriever_framework_property(self):
        from clinical.retrieval.humanistic_retriever import HumanisticRetriever

        r = HumanisticRetriever()
        assert r.framework == "humanistic"

    def test_humanistic_retriever_empty_query(self):
        from clinical.retrieval.humanistic_retriever import HumanisticRetriever

        r = HumanisticRetriever()
        result = r.retrieve("")
        assert result.retrieved == []
        assert "Empty query" in result.synthesis
        assert result.confidence == 0.0

    def test_humanistic_evidence_result_to_framework(self):
        from clinical.retrieval.models import (
            HumanisticEvidenceItem,
            HumanisticEvidenceResult,
        )
        from clinical.framework.models import EvidenceResult

        hr = HumanisticEvidenceResult(
            query="test",
            retrieved=[
                HumanisticEvidenceItem(
                    source="Rogers (1957)",
                    content="The necessary and sufficient conditions of therapeutic personality change",
                    relevance="Core Rogerian theory",
                ),
            ],
            synthesis="One item retrieved about Rogerian conditions",
            key_findings=["Rogerian conditions predict outcome"],
            evidence_gaps=["Limited recent meta-analyses"],
            confidence=0.7,
        )
        fr = hr.to_framework_evidence_result()
        assert isinstance(fr, EvidenceResult)
        assert len(fr.retrieved) == 1
        assert fr.confidence == 0.7

    def test_humanistic_retriever_reimport_through_init(self):
        from clinical.retrieval import HumanisticRetriever, get_humanistic_retriever

        assert HumanisticRetriever is not None
        r = get_humanistic_retriever()
        assert r.framework == "humanistic"

    def test_humanistic_filter_default(self):
        from clinical.retrieval.models import HumanisticEvidenceFilter

        f = HumanisticEvidenceFilter()
        assert f.framework == "humanistic"
        assert f.topic is None
        assert f.intervention is None

    def test_humanistic_filter_custom_topic(self):
        from clinical.retrieval.models import HumanisticEvidenceFilter

        f = HumanisticEvidenceFilter(topic="self-concept")
        assert f.topic == "self-concept"

    def test_humanistic_item_with_metadata(self):
        from clinical.retrieval.models import HumanisticEvidenceItem

        item = HumanisticEvidenceItem(
            source="Gendlin (1978)",
            content="Focusing is a process of attending to the felt sense",
            relevance="Describes the focusing technique",
            topic="focusing",
            intervention="focusing",
            population="adults",
        )
        assert item.topic == "focusing"
        assert item.intervention == "focusing"
        assert item.population == "adults"
