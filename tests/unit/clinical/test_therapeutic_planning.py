from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from clinical.therapeutic_planning import (
    ACTStrategy,
    BehaviouralActivationSuggestion,
    CBTStrategy,
    DBTStrategy,
    HomeworkIdea,
    InterventionDirection,
    PsychEducationSuggestion,
    SelfCompassionStrategy,
    TherapeuticFocus,
    TherapeuticPlanResult,
    TherapeuticPlanner,
    TreatmentGoal,
)


# ═══════════════════════════════════════════════════════════════
# Model unit tests
# ═══════════════════════════════════════════════════════════════

class TestTherapeuticFocus:
    def test_basic_creation(self):
        f = TherapeuticFocus(
            area="Cognitive patterns and self-critical thinking",
            rationale="Perfectionist beliefs maintain the cycle of overwork and withdrawal",
        )
        assert "Cognitive" in f.area
        assert "perfectionist" in f.rationale.lower()

    def test_frozen(self):
        f = TherapeuticFocus(area="Cognitive patterns area", rationale="Rationale text for testing purposes.")
        with pytest.raises(Exception):
            f.area = "changed"

    def test_min_length_enforced_area(self):
        with pytest.raises(ValueError):
            TherapeuticFocus(area="abc", rationale="Valid rationale text for testing.")


class TestTreatmentGoal:
    def test_basic_creation(self):
        g = TreatmentGoal(
            goal="Reduce social avoidance from daily to twice per week",
            suggested_measurement="Behavioural record",
            indicative_timeframe="8 sessions",
        )
        assert "social" in g.goal.lower()
        assert g.suggested_measurement == "Behavioural record"
        assert g.indicative_timeframe == "8 sessions"

    def test_optional_fields(self):
        g = TreatmentGoal(
            goal="Improve mood regulation over 12 weeks",
        )
        assert g.suggested_measurement is None
        assert g.indicative_timeframe is None

    def test_frozen(self):
        g = TreatmentGoal(goal="Valid treatment goal with enough characters here.")
        with pytest.raises(Exception):
            g.goal = "changed"


class TestCBTStrategy:
    def test_basic_creation(self):
        s = CBTStrategy(
            technique="Thought record",
            rationale="Helps identify automatic thoughts about social situations",
            application="Introduce using a recent social situation",
        )
        assert s.technique == "Thought record"
        assert s.application is not None

    def test_optional_application(self):
        s = CBTStrategy(technique="Thought record", rationale="Rationale text for testing purposes.")
        assert s.application is None


class TestACTStrategy:
    def test_basic_creation(self):
        s = ACTStrategy(
            process="Cognitive defusion",
            rationale="Client appears fused with self-critical thoughts",
        )
        assert s.process == "Cognitive defusion"

    def test_with_application(self):
        s = ACTStrategy(
            process="Values clarification",
            rationale="Rationale text for testing purposes.",
            application="Explore using values card sort",
        )
        assert s.application == "Explore using values card sort"


class TestDBTStrategy:
    def test_basic_creation(self):
        s = DBTStrategy(
            skill="Distress tolerance TIPP skills",
            rationale="Client reports intense emotional reactions",
        )
        assert "TIPP" in s.skill


class TestPsychEducationSuggestion:
    def test_basic_creation(self):
        p = PsychEducationSuggestion(
            topic="The anxiety cycle",
            key_points=["Avoidance maintains anxiety", "Short-term relief reinforces avoidance"],
        )
        assert p.topic == "The anxiety cycle"
        assert len(p.key_points) == 2


class TestBehaviouralActivationSuggestion:
    def test_basic_creation(self):
        b = BehaviouralActivationSuggestion(
            activity_domain="Social connection",
            suggested_activities=["Coffee with a friend", "Joining a group"],
            rationale="Graded social exposure may reduce avoidance",
        )
        assert b.activity_domain == "Social connection"
        assert len(b.suggested_activities) == 2


class TestSelfCompassionStrategy:
    def test_basic_creation(self):
        s = SelfCompassionStrategy(
            practice="Compassionate letter writing",
            rationale="May help develop a kinder internal voice",
        )
        assert "compassionate" in s.practice.lower()


class TestHomeworkIdea:
    def test_basic_creation(self):
        h = HomeworkIdea(
            activity="Complete a thought record for one social situation",
            purpose="Build awareness of automatic thoughts",
            frequency="Once before next session",
        )
        assert "thought record" in h.activity.lower()
        assert h.frequency == "Once before next session"

    def test_optional_frequency(self):
        h = HomeworkIdea(
            activity="Practice relaxation exercise daily",
            purpose="Rationale text for testing the purpose field.",
        )
        assert h.frequency is None


class TestInterventionDirection:
    def test_basic_creation(self):
        d = InterventionDirection(
            area="Cognitive restructuring",
            suggested_approaches=["Socratic questioning", "behavioural experiments"],
            rationale="Challenging perfectionist beliefs may reduce avoidance",
        )
        assert "cognitive" in d.area.lower()
        assert len(d.suggested_approaches) == 2


class TestTherapeuticPlanResult:
    def test_defaults_not_allowed_disclaimer_required(self):
        # disclaimer is required, so we must provide it
        r = TherapeuticPlanResult(disclaimer="All suggestions require clinical judgement and are not treatment prescriptions.")
        assert r.therapeutic_focus == []
        assert r.treatment_goals == []
        assert r.cbt_strategies == []
        assert r.act_strategies == []
        assert r.dbt_strategies == []
        assert r.psychoeducation_suggestions == []
        assert r.behavioural_activation_suggestions == []
        assert r.self_compassion_strategies == []
        assert r.homework_ideas == []
        assert r.planning_ms >= 0.0
        assert r.planned_at is not None

    def test_with_all_sections(self):
        r = TherapeuticPlanResult(
            disclaimer="All suggestions require clinical judgement and are not treatment prescriptions.",
            therapeutic_focus=[TherapeuticFocus(area="Cognitive patterns", rationale="Key maintaining factor identified")],
            treatment_goals=[TreatmentGoal(goal="Reduce avoidance over 8 sessions")],
            intervention_directions=[InterventionDirection(
                area="CBT techniques", suggested_approaches=["thought records"], rationale="Evidence-based approach",
            )],
            cbt_strategies=[CBTStrategy(technique="Thought record", rationale="Identifies automatic thoughts")],
            act_strategies=[ACTStrategy(process="Defusion", rationale="Reduces fusion with thoughts")],
            dbt_strategies=[DBTStrategy(skill="TIPP skills", rationale="Manages intense emotions")],
            psychoeducation_suggestions=[PsychEducationSuggestion(
                topic="Anxiety", key_points=["Avoidance maintains anxiety"],
            )],
            behavioural_activation_suggestions=[BehaviouralActivationSuggestion(
                activity_domain="Social", suggested_activities=["Call a friend"], rationale="Reduces isolation",
            )],
            self_compassion_strategies=[SelfCompassionStrategy(
                practice="Letter writing", rationale="Builds self-kindness",
            )],
            homework_ideas=[HomeworkIdea(activity="Thought record", purpose="Build awareness")],
        )
        assert len(r.therapeutic_focus) == 1
        assert len(r.treatment_goals) == 1
        assert len(r.intervention_directions) == 1
        assert len(r.cbt_strategies) == 1
        assert len(r.act_strategies) == 1
        assert len(r.dbt_strategies) == 1
        assert len(r.psychoeducation_suggestions) == 1
        assert len(r.behavioural_activation_suggestions) == 1
        assert len(r.self_compassion_strategies) == 1
        assert len(r.homework_ideas) == 1

    def test_frozen(self):
        r = TherapeuticPlanResult(disclaimer="All suggestions require clinical judgement and are not treatment prescriptions.")
        with pytest.raises(Exception):
            r.disclaimer = "changed"


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

GOOD_JSON_RESPONSE = """{
  "disclaimer": "The following suggestions are evidence-informed and intended for clinician consideration only. They do not constitute a treatment prescription and should be adapted based on clinical judgement and client preferences.",
  "therapeutic_focus": [
    {
      "area": "Cognitive patterns and self-critical thinking",
      "rationale": "The formulation identifies perfectionist beliefs as a key maintaining factor."
    }
  ],
  "treatment_goals": [
    {
      "goal": "Reduce frequency of social avoidance from daily to twice per week over 8 sessions",
      "suggested_measurement": "Behavioural record",
      "indicative_timeframe": "8 sessions"
    }
  ],
  "intervention_directions": [
    {
      "area": "Cognitive restructuring",
      "suggested_approaches": ["Socratic questioning", "behavioural experiments"],
      "rationale": "Challenging perfectionist beliefs may reduce avoidance and improve mood."
    }
  ],
  "cbt_strategies": [
    {
      "technique": "Thought record",
      "rationale": "Helps the client identify and evaluate automatic thoughts about social situations",
      "application": "Introduce using a recent social situation"
    }
  ],
  "act_strategies": [
    {
      "process": "Cognitive defusion",
      "rationale": "The client appears fused with self-critical thoughts",
      "application": "Begin with 'I notice I am having the thought that...' phrasing"
    }
  ],
  "dbt_strategies": [
    {
      "skill": "Distress tolerance TIPP skills",
      "rationale": "Client reports intense emotional reactions to perceived criticism",
      "application": "Practise TIPP in session before applying to real-world situations"
    }
  ],
  "psychoeducation_suggestions": [
    {
      "topic": "The anxiety cycle",
      "key_points": ["Avoidance maintains anxiety in the long term", "Short-term relief reinforces avoidance"]
    }
  ],
  "behavioural_activation_suggestions": [
    {
      "activity_domain": "Social connection",
      "suggested_activities": ["Coffee with a friend", "Joining a low-pressure group activity"],
      "rationale": "Graded social exposure may reduce avoidance and increase positive reinforcement"
    }
  ],
  "self_compassion_strategies": [
    {
      "practice": "Compassionate letter writing",
      "rationale": "May help the client develop a kinder internal voice"
    }
  ],
  "homework_ideas": [
    {
      "activity": "Complete a thought record for one social situation between sessions",
      "purpose": "Build awareness of automatic thoughts and their emotional impact",
      "frequency": "Once before next session"
    }
  ]
}"""

EMPTY_JSON_RESPONSE = """{
  "disclaimer": "No specific suggestions could be generated from the provided information.",
  "therapeutic_focus": [],
  "treatment_goals": [],
  "intervention_directions": [],
  "cbt_strategies": [],
  "act_strategies": [],
  "dbt_strategies": [],
  "psychoeducation_suggestions": [],
  "behavioural_activation_suggestions": [],
  "self_compassion_strategies": [],
  "homework_ideas": []
}"""

MARKDOWN_FENCED_RESPONSE = """```json
{
  "disclaimer": "The following suggestions are evidence-informed and intended for clinician consideration only.",
  "therapeutic_focus": [
    {
      "area": "Markdown fenced focus area",
      "rationale": "Testing markdown fence parsing in the planner module."
    }
  ],
  "treatment_goals": [],
  "intervention_directions": [],
  "cbt_strategies": [],
  "act_strategies": [],
  "dbt_strategies": [],
  "psychoeducation_suggestions": [],
  "behavioural_activation_suggestions": [],
  "self_compassion_strategies": [],
  "homework_ideas": []
}
```"""

INVALID_JSON_RESPONSE = "This is not JSON at all"

PARTIAL_JSON_RESPONSE = """{
  "disclaimer": "Partial plan with only one section.",
  "therapeutic_focus": [
    {
      "area": "A single focus area for testing the partial response",
      "rationale": "Rationale text for testing purposes here."
    }
  ]
}"""


# ═══════════════════════════════════════════════════════════════
# TherapeuticPlanner tests
# ═══════════════════════════════════════════════════════════════

class TestTherapeuticPlanner:
    def make_planner(self, response: str = GOOD_JSON_RESPONSE) -> TherapeuticPlanner:
        llm = MagicMock()
        llm.generate = AsyncMock(return_value=response)
        return TherapeuticPlanner(llm)

    # ── Happy path ──────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_plan_from_kwargs(self):
        planner = self.make_planner()
        result = await planner.plan(
            case_summary="Client presents with anxiety and social avoidance.",
            formulations_text=["Cognitive-behavioural understanding centred on avoidant coping"],
            supporting_evidence=["Clear temporal link between workplace stressor and symptom onset"],
            alternative_explanations=["Physical health conditions may account for some symptoms"],
            missing_information=["Developmental history", "Trauma screening"],
            caution="This formulation is tentative.",
            evidence_summary="CBT is supported as first-line treatment for anxiety disorders.",
            key_findings=["CBT shows large effect sizes for GAD"],
            evidence_themes=["CBT as gold standard for anxiety"],
        )
        assert isinstance(result, TherapeuticPlanResult)
        assert "clinician" in result.disclaimer.lower()
        assert len(result.therapeutic_focus) == 1
        assert len(result.treatment_goals) == 1
        assert len(result.intervention_directions) == 1
        assert len(result.cbt_strategies) == 1
        assert len(result.act_strategies) == 1
        assert len(result.dbt_strategies) == 1
        assert len(result.psychoeducation_suggestions) == 1
        assert len(result.behavioural_activation_suggestions) == 1
        assert len(result.self_compassion_strategies) == 1
        assert len(result.homework_ideas) == 1
        assert result.planning_ms > 0

    @pytest.mark.asyncio
    async def test_plan_with_full_formulation_text(self):
        planner = self.make_planner()
        result = await planner.plan(
            formulation='{"case_summary": "Test case summary with enough chars.", "confidence": 0.65}',
            evidence_summary="CBT is well-established for anxiety.",
        )
        assert len(result.therapeutic_focus) == 1
        assert len(result.treatment_goals) == 1

    # ── Empty / edge cases ──────────────────────────────────

    @pytest.mark.asyncio
    async def test_empty_input(self):
        planner = self.make_planner()
        result = await planner.plan()
        assert result.therapeutic_focus == []
        assert "No formulation" in result.disclaimer

    @pytest.mark.asyncio
    async def test_empty_llm_response(self):
        planner = self.make_planner(response=EMPTY_JSON_RESPONSE)
        result = await planner.plan(
            case_summary="Test case summary with enough characters for validation.",
            evidence_summary="Test evidence.",
        )
        assert result.therapeutic_focus == []
        assert result.treatment_goals == []

    # ── Response parsing ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_markdown_fenced_response(self):
        planner = self.make_planner(response=MARKDOWN_FENCED_RESPONSE)
        result = await planner.plan(
            case_summary="Test case summary for markdown test with enough characters.",
        )
        assert len(result.therapeutic_focus) == 1
        assert "Markdown" in result.therapeutic_focus[0].area

    @pytest.mark.asyncio
    async def test_invalid_json_response_falls_back(self):
        planner = self.make_planner(response=INVALID_JSON_RESPONSE)
        result = await planner.plan(
            case_summary="Test case summary for invalid JSON test with enough chars.",
        )
        assert "could not be completed" in result.disclaimer.lower()

    @pytest.mark.asyncio
    async def test_partial_json_response(self):
        planner = self.make_planner(response=PARTIAL_JSON_RESPONSE)
        result = await planner.plan(
            case_summary="Test case summary for partial JSON with enough characters.",
        )
        assert len(result.therapeutic_focus) == 1
        assert result.treatment_goals == []

    # ── LLM error handling ──────────────────────────────────

    @pytest.mark.asyncio
    async def test_llm_raises_exception(self):
        llm = MagicMock()
        llm.generate = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
        planner = TherapeuticPlanner(llm)
        result = await planner.plan(
            case_summary="Test case summary for LLM error test with enough chars.",
        )
        assert "could not be completed" in result.disclaimer.lower()

    # ── Builder resilience ──────────────────────────────────

    def test_build_focus_skips_invalid(self):
        planner = self.make_planner()
        raw = [
            {"area": "Cognitive patterns area", "rationale": "Rationale text for testing purposes."},
            {"area": "", "rationale": "text"},
            {"not_area": "missing"},
            42,
        ]
        items = planner._build_focus(raw)
        assert len(items) == 1
        assert items[0].area == "Cognitive patterns area"

    def test_build_goals_skips_invalid(self):
        planner = self.make_planner()
        raw = [
            {"goal": "Valid goal with enough characters for the min length requirement."},
            {"goal": "", "suggested_measurement": "test"},
            "invalid",
        ]
        items = planner._build_goals(raw)
        assert len(items) == 1

    def test_build_directions_skips_invalid(self):
        planner = self.make_planner()
        raw = [
            {"area": "Cognitive", "suggested_approaches": ["CBT"], "rationale": "Rationale for testing purposes here."},
            {"area": ""},
            None,
        ]
        items = planner._build_directions(raw)
        assert len(items) == 1

    def test_build_cbt_skips_invalid(self):
        planner = self.make_planner()
        raw = [
            {"technique": "Thought record", "rationale": "Rationale text for testing purposes."},
            {"technique": ""},
            {},
        ]
        items = planner._build_cbt(raw)
        assert len(items) == 1

    def test_build_act_skips_invalid(self):
        planner = self.make_planner()
        raw = [
            {"process": "Defusion", "rationale": "Rationale text for testing purposes."},
            {"process": ""},
            {},
        ]
        items = planner._build_act(raw)
        assert len(items) == 1

    def test_build_dbt_skips_invalid(self):
        planner = self.make_planner()
        raw = [
            {"skill": "TIPP skills", "rationale": "Rationale text for testing purposes."},
            {"skill": ""},
            {},
        ]
        items = planner._build_dbt(raw)
        assert len(items) == 1

    def test_build_psychoeducation_skips_invalid(self):
        planner = self.make_planner()
        raw = [
            {"topic": "Anxiety cycle", "key_points": ["Avoidance maintains anxiety"]},
            {"topic": "", "key_points": ["test"]},
            {},
        ]
        items = planner._build_psychoeducation(raw)
        assert len(items) == 1

    def test_build_ba_skips_invalid(self):
        planner = self.make_planner()
        raw = [
            {"activity_domain": "Social", "suggested_activities": ["Call a friend"], "rationale": "Rationale text here."},
            {"activity_domain": ""},
            {},
        ]
        items = planner._build_ba(raw)
        assert len(items) == 1

    def test_build_self_compassion_skips_invalid(self):
        planner = self.make_planner()
        raw = [
            {"practice": "Letter writing", "rationale": "Rationale text for testing purposes."},
            {"practice": ""},
            {},
        ]
        items = planner._build_self_compassion(raw)
        assert len(items) == 1

    def test_build_homework_skips_invalid(self):
        planner = self.make_planner()
        raw = [
            {"activity": "Valid activity with enough characters for validation purposes.", "purpose": "Purpose text for testing here."},
            {"activity": ""},
            {},
        ]
        items = planner._build_homework(raw)
        assert len(items) == 1

    # ── Standalone instantiation ────────────────────────────

    def test_can_be_instantiated_without_pipeline(self):
        llm = MagicMock()
        planner = TherapeuticPlanner(llm)
        assert planner is not None
        assert planner._llm is llm

    def test_result_is_valid_pydantic(self):
        r = TherapeuticPlanResult(disclaimer="All suggestions require clinical judgement and are not treatment prescriptions.")
        assert isinstance(r, TherapeuticPlanResult)
        assert r.model_config.get("frozen")
