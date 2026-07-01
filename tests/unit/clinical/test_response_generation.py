from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from clinical.response_generation import ClinicalResponseResult, ResponseGenerator

# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

_MOCK_MARKDOWN = """\
## 1. CASE SUMMARY

A 34-year-old female professional presented with persistent low mood.

## 2. PRESENTING CONCERNS

- Persistent low mood and loss of interest
- Excessive worry about work performance

## 3. OBSERVED SYMPTOMS

- Low mood, anhedonia, reduced energy
- Excessive worry, irritability, muscle tension

## 4. CLINICAL FORMULATION

Perfectionist standards and emotional avoidance maintaining the cycle.

## 5. POSSIBLE DIFFERENTIAL CONSIDERATIONS

Generalised anxiety difficulties cannot be ruled out.

## 6. MISSING INFORMATION

History of previous episodes would strengthen the formulation.

## 7. EVIDENCE SUMMARY

CBT for perfectionism shows moderate effect sizes.

## 8. THERAPEUTIC FOCUS

Cognitive restructuring of perfectionist beliefs.

## 9. SUGGESTED INTERVENTION DIRECTIONS

CBT-informed therapy with graded exposure.

## 10. REFERENCES

Beck, A. T. (1976). Cognitive therapy.

## 11. CONFIDENCE LEVEL

Moderate confidence. Rating: 0.7
"""

_DEFAULT_LLM_JSON = json.dumps({
    "markdown": _MOCK_MARKDOWN,
    "sections_generated": 11,
    "confidence": 0.7,
})


def _make_response_json(markdown: str | None = None, sections: int = 11, conf: float = 0.7) -> str:
    return json.dumps({
        "markdown": markdown or _MOCK_MARKDOWN,
        "sections_generated": sections,
        "confidence": conf,
    })


# ═══════════════════════════════════════════════════════════════
# Model unit tests
# ═══════════════════════════════════════════════════════════════

class TestClinicalResponseResult:
    def test_basic_creation(self):
        r = ClinicalResponseResult(
            markdown=_MOCK_MARKDOWN,
            sections_generated=11,
            confidence=0.7,
        )
        assert "CASE SUMMARY" in r.markdown
        assert r.sections_generated == 11
        assert r.confidence == 0.7
        assert r.generation_ms == 0.0

    def test_min_length_enforced(self):
        with pytest.raises(ValueError):
            ClinicalResponseResult(markdown="too short", sections_generated=0, confidence=0.0)

    def test_sections_generated_clamped(self):
        with pytest.raises(ValueError):
            ClinicalResponseResult(markdown="x" * 100, sections_generated=12, confidence=0.5)
        with pytest.raises(ValueError):
            ClinicalResponseResult(markdown="x" * 100, sections_generated=-1, confidence=0.5)

    def test_confidence_clamped(self):
        with pytest.raises(ValueError):
            ClinicalResponseResult(markdown="x" * 100, sections_generated=5, confidence=1.5)
        with pytest.raises(ValueError):
            ClinicalResponseResult(markdown="x" * 100, sections_generated=5, confidence=-0.1)

    def test_frozen(self):
        r = ClinicalResponseResult(markdown=_MOCK_MARKDOWN, sections_generated=11, confidence=0.7)
        with pytest.raises(Exception):
            r.markdown = "changed"

    def test_generated_at_defaults_to_utc(self):
        r = ClinicalResponseResult(markdown=_MOCK_MARKDOWN, sections_generated=11, confidence=0.7)
        assert r.generated_at is not None
        assert r.generated_at.tzinfo is not None
        assert str(r.generated_at.tzinfo) == "UTC"

    def test_default_confidence(self):
        r = ClinicalResponseResult(markdown=_MOCK_MARKDOWN, sections_generated=0, confidence=0.0)
        assert r.confidence == 0.0
        assert r.sections_generated == 0


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.generate = AsyncMock()
    return llm


@pytest.fixture
def generator(mock_llm):
    return ResponseGenerator(llm=mock_llm)


# ═══════════════════════════════════════════════════════════════
# Generator unit tests
# ═══════════════════════════════════════════════════════════════

class TestGenerate:
    @pytest.mark.asyncio
    async def test_generate_with_full_data(self, generator, mock_llm):
        mock_llm.generate.return_value = _DEFAULT_LLM_JSON

        result = await generator.generate(
            case_summary="A 34-year-old female with low mood and anxiety for 6 months.",
            presenting_concerns=["Low mood", "Anxiety", "Sleep disturbance"],
            observed_symptoms=["Anhedonia", "Poor concentration", "Fatigue"],
            formulation_text="Perfectionist standards and emotional avoidance.",
            formulation_confidence=0.7,
            differential_considerations=["GAD", "Adjustment difficulties"],
            missing_information="Trauma history, family history.",
            evidence_summary="CBT for perfectionism shows moderate effect sizes.",
            evidence_findings=["Perfectionism correlates with depression"],
            therapeutic_focus=["Cognitive restructuring", "Behavioural activation"],
            intervention_directions="CBT-informed therapy with graded exposure.",
            cbt_strategies=["Thought records", "Behavioural experiments"],
            act_strategies=["Values clarification", "Defusion"],
            dbt_strategies=["Mindfulness", "Distress tolerance"],
            references=["Beck (1976)", "Ehlers & Clark (2000)"],
            caution="Risk assessment recommended before trauma work.",
        )

        assert "CASE SUMMARY" in result.markdown
        assert result.sections_generated == 11
        assert result.confidence == 0.7
        assert result.generation_ms > 0
        mock_llm.generate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_empty_input_returns_early_no_llm_call(self, generator, mock_llm):
        result = await generator.generate()

        assert "No clinical data was provided" in result.markdown
        assert result.sections_generated == 0
        assert result.confidence == 0.0
        mock_llm.generate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_partial_input_with_only_presenting_concerns(self, generator, mock_llm):
        mock_llm.generate.return_value = _DEFAULT_LLM_JSON

        result = await generator.generate(
            presenting_concerns=["Low mood", "Anxiety"],
        )

        assert "CASE SUMMARY" in result.markdown
        mock_llm.generate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_llm_returns_malformed_json(self, generator, mock_llm):
        mock_llm.generate.return_value = "this is not json at all and has no braces"

        result = await generator.generate(
            case_summary="A 34-year-old female with low mood.",
        )

        assert "could not be generated" in result.markdown.lower()
        assert result.sections_generated == 0
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_llm_returns_json_with_markdown_fence(self, generator, mock_llm):
        mock_llm.generate.return_value = f"```json\n{_DEFAULT_LLM_JSON}\n```"

        result = await generator.generate(
            case_summary="A 34-year-old female with low mood.",
        )

        assert "CASE SUMMARY" in result.markdown
        assert result.sections_generated == 11

    @pytest.mark.asyncio
    async def test_llm_failure_graceful_degradation(self, generator, mock_llm):
        mock_llm.generate.side_effect = RuntimeError("Ollama unavailable")

        result = await generator.generate(
            case_summary="A 34-year-old female with low mood.",
        )

        assert "could not be generated" in result.markdown.lower()
        assert result.sections_generated == 0
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_generation_timing_is_recorded(self, generator, mock_llm):
        mock_llm.generate.return_value = _DEFAULT_LLM_JSON

        result = await generator.generate(
            case_summary="A 34-year-old female with low mood.",
        )

        assert result.generation_ms > 0

    @pytest.mark.asyncio
    async def test_clamps_sections_generated_out_of_range(self, generator, mock_llm):
        mock_llm.generate.return_value = _make_response_json(sections=99)

        result = await generator.generate(
            case_summary="A 34-year-old female with low mood.",
        )

        assert result.sections_generated == 11

    @pytest.mark.asyncio
    async def test_clamps_confidence_out_of_range(self, generator, mock_llm):
        mock_llm.generate.return_value = _make_response_json(sections=5, conf=2.5)

        result = await generator.generate(
            case_summary="A 34-year-old female with low mood.",
        )

        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_clamps_negative_confidence(self, generator, mock_llm):
        mock_llm.generate.return_value = _make_response_json(sections=5, conf=-1.0)

        result = await generator.generate(
            case_summary="A 34-year-old female with low mood.",
        )

        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_non_numeric_sections_generated_defaults_to_zero(self, generator, mock_llm):
        raw = '{"markdown": "Some markdown text here for the response report.", "sections_generated": "lots", "confidence": 0.5}'
        mock_llm.generate.return_value = raw

        result = await generator.generate(
            case_summary="A 34-year-old female with low mood.",
        )

        assert result.sections_generated == 0

    @pytest.mark.asyncio
    async def test_non_numeric_confidence_defaults_to_zero(self, generator, mock_llm):
        raw = '{"markdown": "Some markdown text here for the response report.", "sections_generated": 5, "confidence": "high"}'
        mock_llm.generate.return_value = raw

        result = await generator.generate(
            case_summary="A 34-year-old female with low mood.",
        )

        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_missing_markdown_field_falls_back_to_error(self, generator, mock_llm):
        mock_llm.generate.return_value = '{"sections_generated": 0, "confidence": 0.0}'

        result = await generator.generate(
            case_summary="A 34-year-old female with low mood.",
        )

        assert "could not be generated" in result.markdown.lower()


class TestContentCheck:
    def test_has_content_positive(self):
        gen = ResponseGenerator(MagicMock())
        assert gen._has_content("## Case Summary\nSome text here\n")

    def test_has_content_empty_string(self):
        gen = ResponseGenerator(MagicMock())
        assert not gen._has_content("")

    def test_has_content_whitespace_only(self):
        gen = ResponseGenerator(MagicMock())
        assert not gen._has_content("   \n  \n  ")

    def test_has_content_only_headers(self):
        gen = ResponseGenerator(MagicMock())
        assert not gen._has_content("## Case Summary\n## Presenting Concerns\n")

    def test_has_content_with_data_and_header(self):
        gen = ResponseGenerator(MagicMock())
        assert gen._has_content("## Case Summary\nSome text\n## Empty\n")

    def test_has_content_with_list(self):
        gen = ResponseGenerator(MagicMock())
        assert gen._has_content("- Item 1\n- Item 2\n")


class TestBuildPrompt:
    def test_empty_prompt_when_no_args(self):
        gen = ResponseGenerator(MagicMock())
        prompt = gen._build_prompt(
            case_summary=None,
            presenting_concerns=None,
            observed_symptoms=None,
            formulation_text=None,
            formulation_confidence=None,
            differential_considerations=None,
            missing_information=None,
            evidence_summary=None,
            evidence_findings=None,
            therapeutic_focus=None,
            intervention_directions=None,
            cbt_strategies=None,
            act_strategies=None,
            dbt_strategies=None,
            references=None,
            caution=None,
        )
        assert not prompt or prompt.strip() == ""

    def test_prompt_includes_case_summary(self):
        gen = ResponseGenerator(MagicMock())
        prompt = gen._build_prompt(
            case_summary="Some summary text",
            presenting_concerns=None,
            observed_symptoms=None,
            formulation_text=None,
            formulation_confidence=None,
            differential_considerations=None,
            missing_information=None,
            evidence_summary=None,
            evidence_findings=None,
            therapeutic_focus=None,
            intervention_directions=None,
            cbt_strategies=None,
            act_strategies=None,
            dbt_strategies=None,
            references=None,
            caution=None,
        )
        assert "Some summary text" in prompt
        assert "## Case Summary" in prompt

    def test_prompt_includes_lists(self):
        gen = ResponseGenerator(MagicMock())
        prompt = gen._build_prompt(
            case_summary=None,
            presenting_concerns=["Concern A", "Concern B"],
            observed_symptoms=["Symptom 1"],
            formulation_text=None,
            formulation_confidence=None,
            differential_considerations=["Diff A"],
            missing_information=None,
            evidence_summary=None,
            evidence_findings=["Finding 1"],
            therapeutic_focus=["Focus A"],
            intervention_directions=None,
            cbt_strategies=["CBT 1"],
            act_strategies=["ACT 1"],
            dbt_strategies=["DBT 1"],
            references=["Ref 1"],
            caution=None,
        )
        assert "- Concern A" in prompt
        assert "- Symptom 1" in prompt
        assert "- Diff A" in prompt
        assert "- Finding 1" in prompt
        assert "- Focus A" in prompt
        assert "- CBT 1" in prompt
        assert "- ACT 1" in prompt
        assert "- DBT 1" in prompt
        assert "- Ref 1" in prompt

    def test_prompt_includes_text_fields(self):
        gen = ResponseGenerator(MagicMock())
        prompt = gen._build_prompt(
            case_summary=None,
            presenting_concerns=None,
            observed_symptoms=None,
            formulation_text="Formulation narrative text",
            formulation_confidence=0.8,
            differential_considerations=None,
            missing_information="Missing info text",
            evidence_summary="Evidence text",
            intervention_directions="Intervention text",
            references=None,
            caution="Caution text",
            evidence_findings=None,
            therapeutic_focus=None,
            cbt_strategies=None,
            act_strategies=None,
            dbt_strategies=None,
        )
        assert "Formulation narrative text" in prompt
        assert "0.8" in prompt
        assert "Missing info text" in prompt
        assert "Evidence text" in prompt
        assert "Intervention text" in prompt
        assert "Caution text" in prompt
        assert "## Caution" in prompt
        assert "## Formulation Confidence" in prompt


class TestParseResponse:
    @pytest.mark.asyncio
    async def test_raises_on_empty_response(self, generator, mock_llm):
        mock_llm.generate.return_value = ""

        result = await generator.generate(
            case_summary="Some text.",
        )

        assert result.sections_generated == 0

    @pytest.mark.asyncio
    async def test_raises_on_no_braces(self, generator, mock_llm):
        mock_llm.generate.return_value = "just text no json here"

        result = await generator.generate(
            case_summary="Some text.",
        )

        assert result.sections_generated == 0

    @pytest.mark.asyncio
    async def test_handles_nested_json_in_response(self, generator, mock_llm):
        mock_llm.generate.return_value = _DEFAULT_LLM_JSON

        result = await generator.generate(
            case_summary="Some text.",
        )

        assert "CASE SUMMARY" in result.markdown


class TestInstantiation:
    def test_can_be_instantiated_without_pipeline(self, mock_llm):
        gen = ResponseGenerator(llm=mock_llm)
        assert gen is not None
        assert hasattr(gen, "generate")

    def test_llm_service_is_used(self, mock_llm):
        gen = ResponseGenerator(llm=mock_llm)
        assert gen._llm is mock_llm
