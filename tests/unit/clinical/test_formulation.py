from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from clinical.formulation import ClinicalFormulationResult, ClinicalFormulator, Formulation


# ═══════════════════════════════════════════════════════════════
# Model unit tests
# ═══════════════════════════════════════════════════════════════

class TestFormulation:
    def test_basic_creation(self):
        f = Formulation(
            label="Cognitive-behavioural understanding",
            explanation="This formulation considers how early experiences shaped core beliefs leading to current difficulties.",
            supporting_symptoms=["Reports social avoidance", "Endorses negative beliefs about self"],
            confidence=0.7,
        )
        assert "Cognitive" in f.label
        assert len(f.supporting_symptoms) == 2
        assert f.confidence == 0.7

    def test_frozen(self):
        f = Formulation(
            label="Cognitive-behavioural understanding",
            explanation="This formulation considers how early experiences shaped core beliefs leading to current difficulties.",
            supporting_symptoms=["Reports social avoidance"],
            confidence=0.5,
        )
        with pytest.raises(Exception):
            f.label = "changed"

    def test_min_length_enforced_label(self):
        with pytest.raises(ValueError):
            Formulation(
                label="abc",
                explanation="This is a valid explanation that meets the minimum length requirement for testing.",
                supporting_symptoms=["a"],
                confidence=0.5,
            )

    def test_min_length_enforced_explanation(self):
        with pytest.raises(ValueError):
            Formulation(
                label="A valid label for testing",
                explanation="Too short",
                supporting_symptoms=["a"],
                confidence=0.5,
            )

    def test_min_length_enforced_symptoms(self):
        with pytest.raises(ValueError):
            Formulation(
                label="A valid label for testing",
                explanation="This is a valid explanation that meets the minimum length requirement for testing.",
                supporting_symptoms=[],
                confidence=0.5,
            )

    def test_confidence_clamped(self):
        with pytest.raises(ValueError):
            Formulation(
                label="A valid label for testing",
                explanation="This is a valid explanation that meets the minimum length requirement for testing.",
                supporting_symptoms=["a"],
                confidence=1.5,
            )
        with pytest.raises(ValueError):
            Formulation(
                label="A valid label for testing",
                explanation="This is a valid explanation that meets the minimum length requirement for testing.",
                supporting_symptoms=["a"],
                confidence=-0.1,
            )


class TestClinicalFormulationResult:
    def test_minimal(self):
        r = ClinicalFormulationResult(
            case_summary="A 35-year-old professional presenting with anxiety and low mood.",
        )
        assert r.possible_formulations == []
        assert r.supporting_evidence == []
        assert r.alternative_explanations == []
        assert r.missing_assessment_information == []
        assert r.caution == ""
        assert r.confidence == 0.0
        assert r.formulation_ms >= 0.0
        assert r.formulated_at is not None

    def test_with_formulations(self):
        r = ClinicalFormulationResult(
            case_summary="A 35-year-old professional presenting with anxiety and low mood.",
            possible_formulations=[
                Formulation(
                    label="Cognitive-behavioural understanding",
                    explanation="This formulation considers how early experiences shaped core beliefs leading to current difficulties.",
                    supporting_symptoms=["Reports social avoidance", "Endorses negative beliefs"],
                    confidence=0.7,
                ),
            ],
            supporting_evidence=["Avoidance pattern is consistent across settings"],
            alternative_explanations=["Physical health conditions cannot be ruled out"],
            missing_assessment_information=["Developmental history", "Trauma history"],
            caution="This is a tentative formulation based on available information.",
            confidence=0.65,
        )
        assert len(r.possible_formulations) == 1
        assert len(r.supporting_evidence) == 1
        assert len(r.alternative_explanations) == 1
        assert len(r.missing_assessment_information) == 2
        assert "tentative" in r.caution
        assert r.confidence == 0.65

    def test_frozen(self):
        r = ClinicalFormulationResult(case_summary="A 35-year-old professional presenting with anxiety and low mood.")
        with pytest.raises(Exception):
            r.case_summary = "changed"


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

GOOD_JSON_RESPONSE = """{
  "case_summary": "A 35-year-old professional presents with escalating anxiety and low mood over the past 6 months. Symptoms include social avoidance, rumination, and disrupted sleep. Contextual factors include workplace stress and recent relationship difficulties. Protective factors include supportive family and prior positive response to therapy.",
  "possible_formulations": [
    {
      "label": "Cognitive-behavioural formulation centred on avoidant coping",
      "explanation": "The client's reported social avoidance may function as a short-term anxiety management strategy that, over time, prevents disconfirmatory learning and maintains fear of negative evaluation. Early perfectionist standards may have created vulnerability to workplace stress, triggering a cycle of rumination, withdrawal, and low mood.",
      "supporting_symptoms": [
        "Reports avoiding social gatherings for 6 months",
        "Describes racing thoughts before anticipated social events",
        "Endorses beliefs about being judged negatively by colleagues"
      ],
      "confidence": 0.7
    },
    {
      "label": "Psychodynamic understanding centred on relational patterns",
      "explanation": "Alternatively, the client's difficulties may reflect recurring relational patterns stemming from early attachment experiences. The recent relationship difficulty may have activated fears of rejection, leading to withdrawal and low mood as a protective response.",
      "supporting_symptoms": [
        "History of relationship difficulties",
        "Reports feeling 'not good enough' in multiple contexts",
        "Pattern of withdrawal following perceived criticism"
      ],
      "confidence": 0.55
    }
  ],
  "supporting_evidence": [
    "Clear temporal link between workplace stressor and symptom onset",
    "Avoidance pattern is consistent across multiple domains",
    "Prior positive response to therapy suggests good prognostic factors"
  ],
  "alternative_explanations": [
    "Physical health conditions (e.g. thyroid dysfunction) may account for some symptoms and warrant medical investigation",
    "Substance use or medication side effects have not been fully assessed",
    "Cultural factors around achievement and success may shape how distress is experienced"
  ],
  "missing_assessment_information": [
    "Standardised measures of anxiety and depression (e.g. GAD-7, PHQ-9)",
    "Developmental and attachment history",
    "Trauma screening",
    "Medical assessment to rule out organic causes"
  ],
  "caution": "This formulation is based on the available clinical information and should be considered tentative. It does not replace a comprehensive clinical assessment. Comorbidity and cultural factors should be explored further.",
  "confidence": 0.65
}"""

EMPTY_JSON_RESPONSE = """{
  "case_summary": "No formulation could be generated from the provided information.",
  "possible_formulations": [],
  "supporting_evidence": [],
  "alternative_explanations": [],
  "missing_assessment_information": [],
  "caution": "Insufficient information to generate a meaningful formulation.",
  "confidence": 0.0
}"""

MARKDOWN_FENCED_RESPONSE = """```json
{
  "case_summary": "A client presenting with anxiety following workplace stress.",
  "possible_formulations": [
    {
      "label": "Cognitive-behavioural formulation",
      "explanation": "Workplace stress may have activated perfectionist beliefs, leading to a cycle of overwork, exhaustion, and withdrawal that maintains low mood.",
      "supporting_symptoms": ["Workplace stress trigger", "Social withdrawal", "Low energy"],
      "confidence": 0.6
    }
  ],
  "supporting_evidence": ["Temporal link to stressor"],
  "alternative_explanations": ["Physical health factors not ruled out"],
  "missing_assessment_information": ["Medical assessment", "Trauma history"],
  "caution": "Tentative formulation based on limited information.",
  "confidence": 0.6
}
```"""

INVALID_JSON_RESPONSE = "This is not JSON at all"

PARTIAL_JSON_RESPONSE = """{
  "case_summary": "Partial formulation result.",
  "possible_formulations": [
    {
      "label": "A single formulation",
      "explanation": "This is a valid explanation that meets the minimum length requirement for testing purposes.",
      "supporting_symptoms": ["Symptom one", "Symptom two"],
      "confidence": 0.6
    }
  ]
}"""

FLAT_DICT = {
    "age": 35,
    "gender": "female",
    "occupation": "teacher",
    "presenting_concerns": ["anxiety", "low mood"],
    "symptoms": ["social avoidance", "rumination", "sleep disturbance"],
    "emotional_indicators": ["tearful", "irritable"],
    "behavioural_indicators": ["withdrawn", "reduced activity"],
    "stressors": ["workplace stress", "relationship difficulty"],
    "protective_factors": ["supportive family"],
    "risk_factors": ["previous episode of depression"],
    "functional_impairment": "difficulty concentrating at work",
    "social_context": "lives with partner, limited social network",
    "duration": "6 months",
    "previous_treatment": "CBT 2 years ago with partial response",
    "severity": "moderate",
}


# ═══════════════════════════════════════════════════════════════
# ClinicalFormulator tests
# ═══════════════════════════════════════════════════════════════

class TestClinicalFormulator:
    def make_formulator(self, response: str = GOOD_JSON_RESPONSE) -> ClinicalFormulator:
        llm = MagicMock()
        llm.generate = AsyncMock(return_value=response)
        return ClinicalFormulator(llm)

    # ── Happy path ──────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_formulate_from_flat_dict(self):
        formulator = self.make_formulator()
        result = await formulator.formulate(FLAT_DICT)
        assert isinstance(result, ClinicalFormulationResult)
        assert "35-year-old" in result.case_summary
        assert len(result.possible_formulations) == 2
        assert len(result.supporting_evidence) == 3
        assert len(result.alternative_explanations) == 3
        assert len(result.missing_assessment_information) == 4
        assert result.caution
        assert result.confidence == 0.65
        assert result.formulation_ms > 0

    @pytest.mark.asyncio
    async def test_formulate_from_kwargs(self):
        formulator = self.make_formulator()
        result = await formulator.formulate(
            case_summary="Client presents with anxiety and low mood.",
            symptoms=["social avoidance", "rumination"],
            contextual_factors=["workplace stress", "relationship difficulties"],
            risk_factors=["previous depression"],
            protective_factors=["supportive family"],
            duration="6 months",
            previous_treatment="CBT 2 years ago",
            evidence_synthesis="CBT is supported as first-line treatment for anxiety.",
        )
        assert len(result.possible_formulations) == 2
        assert result.confidence == 0.65

    @pytest.mark.asyncio
    async def test_formulate_with_evidence_synthesis(self):
        formulator = self.make_formulator()
        result = await formulator.formulate(
            FLAT_DICT,
            evidence_synthesis="CBT is well-established for anxiety disorders with large effect sizes.",
        )
        assert len(result.possible_formulations) == 2

    # ── Empty / edge cases ──────────────────────────────────

    @pytest.mark.asyncio
    async def test_empty_input(self):
        formulator = self.make_formulator()
        result = await formulator.formulate()
        assert result.possible_formulations == []
        assert "No case information" in result.case_summary

    @pytest.mark.asyncio
    async def test_empty_input_explicit_none(self):
        formulator = self.make_formulator()
        result = await formulator.formulate(case_data=None)
        assert result.possible_formulations == []
        assert "No case information" in result.case_summary

    @pytest.mark.asyncio
    async def test_empty_llm_response(self):
        formulator = self.make_formulator(response=EMPTY_JSON_RESPONSE)
        result = await formulator.formulate(FLAT_DICT)
        assert result.possible_formulations == []
        assert result.supporting_evidence == []

    # ── Response parsing ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_markdown_fenced_response(self):
        formulator = self.make_formulator(response=MARKDOWN_FENCED_RESPONSE)
        result = await formulator.formulate(FLAT_DICT)
        assert len(result.possible_formulations) == 1
        assert "Cognitive-behavioural" in result.possible_formulations[0].label

    @pytest.mark.asyncio
    async def test_invalid_json_response_falls_back(self):
        formulator = self.make_formulator(response=INVALID_JSON_RESPONSE)
        result = await formulator.formulate(FLAT_DICT)
        assert "Failed" in result.case_summary or "failed" in result.case_summary

    @pytest.mark.asyncio
    async def test_partial_json_response(self):
        formulator = self.make_formulator(response=PARTIAL_JSON_RESPONSE)
        result = await formulator.formulate(FLAT_DICT)
        assert len(result.possible_formulations) == 1
        assert result.caution == ""

    # ── LLM error handling ──────────────────────────────────

    @pytest.mark.asyncio
    async def test_llm_raises_exception(self):
        llm = MagicMock()
        llm.generate = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
        formulator = ClinicalFormulator(llm)
        result = await formulator.formulate(FLAT_DICT)
        assert "failed" in result.case_summary.lower()

    # ── Confidence clamping ─────────────────────────────────

    @pytest.mark.asyncio
    async def test_confidence_clamping(self):
        llm = MagicMock()
        llm.generate = AsyncMock(return_value=
            '{"case_summary": "Test case summary with enough characters to satisfy the minimum length constraint.", '
            '"possible_formulations": ['
            '  {"label": "Test formulation label", "explanation": "This is a valid explanation that meets minimum length requirements for testing purposes.", "supporting_symptoms": ["a"], "confidence": 1.5}'
            '], '
            '"supporting_evidence": [], "alternative_explanations": [], '
            '"missing_assessment_information": [], "caution": "", "confidence": 1.2}')
        formulator = ClinicalFormulator(llm)
        result = await formulator.formulate(FLAT_DICT)
        assert result.possible_formulations[0].confidence == 1.0
        assert result.confidence == 1.0

    # ── Builder resilience ──────────────────────────────────

    def test_build_formulations_skips_invalid(self):
        formulator = self.make_formulator()
        raw = [
            {
                "label": "A valid formulation label",
                "explanation": "This is a valid explanation that meets the minimum length requirement for testing purposes.",
                "supporting_symptoms": ["Symptom A", "Symptom B"],
                "confidence": 0.7,
            },
            {"label": "", "explanation": "Valid explanation text here for testing purposes.", "supporting_symptoms": ["a"], "confidence": 0.5},
            {"not_label": "missing"},
            42,
            None,
        ]
        formulations = formulator._build_formulations(raw)
        assert len(formulations) == 1
        assert formulations[0].label == "A valid formulation label"

    def test_clamp_confidence_edge_cases(self):
        assert ClinicalFormulator._clamp_confidence(None) == 0.0
        assert ClinicalFormulator._clamp_confidence("not_a_number") == 0.0
        assert ClinicalFormulator._clamp_confidence(0.5) == 0.5
        assert ClinicalFormulator._clamp_confidence(1.5) == 1.0
        assert ClinicalFormulator._clamp_confidence(-0.5) == 0.0

    # ── Standalone instantiation ────────────────────────────

    def test_can_be_instantiated_without_pipeline(self):
        llm = MagicMock()
        formulator = ClinicalFormulator(llm)
        assert formulator is not None
        assert formulator._llm is llm

    def test_result_is_valid_pydantic(self):
        r = ClinicalFormulationResult(case_summary="A valid case summary for testing purposes.")
        assert isinstance(r, ClinicalFormulationResult)
        assert r.model_config.get("frozen")
