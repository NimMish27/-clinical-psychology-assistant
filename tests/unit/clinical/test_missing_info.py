from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from clinical.missing_info import MissingInfoDetector, MissingInfoItem, MissingInfoResult


# ═══════════════════════════════════════════════════════════════
# Model unit tests
# ═══════════════════════════════════════════════════════════════

class TestMissingInfoItem:
    def test_basic_creation(self):
        item = MissingInfoItem(
            info_gap="Sleep quality and duration",
            clinical_relevance="Sleep disturbance affects mood and guides treatment choices",
            suggested_questions=[
                "How many hours do you typically sleep?",
                "Do you have trouble falling asleep?",
            ],
        )
        assert "sleep" in item.info_gap.lower()
        assert len(item.suggested_questions) == 2

    def test_frozen(self):
        item = MissingInfoItem(
            info_gap="Sleep quality and duration area",
            clinical_relevance="Clinical relevance text for testing purposes here.",
            suggested_questions=["A valid question here?"],
        )
        with pytest.raises(Exception):
            item.info_gap = "changed"

    def test_min_length_enforced_info_gap(self):
        with pytest.raises(ValueError):
            MissingInfoItem(
                info_gap="abc",
                clinical_relevance="Valid clinical relevance text for testing purposes here.",
                suggested_questions=["A question?"],
            )

    def test_min_length_enforced_clinical_relevance(self):
        with pytest.raises(ValueError):
            MissingInfoItem(
                info_gap="Sleep quality",
                clinical_relevance="Short",
                suggested_questions=["A question?"],
            )

    def test_min_length_enforced_suggested_questions(self):
        with pytest.raises(ValueError):
            MissingInfoItem(
                info_gap="Sleep quality",
                clinical_relevance="Valid clinical relevance text for testing purposes here.",
                suggested_questions=[],
            )


class TestMissingInfoResult:
    def test_defaults(self):
        r = MissingInfoResult()
        assert r.missing_information == []
        assert r.input_summary == ""
        assert r.overall_assessment == ""
        assert r.detection_ms >= 0.0
        assert r.detected_at is not None

    def test_with_data(self):
        r = MissingInfoResult(
            input_summary="Client reports waking up tired.",
            missing_information=[
                MissingInfoItem(
                    info_gap="Sleep quality",
                    clinical_relevance="Sleep disturbance affects mood and treatment planning.",
                    suggested_questions=["How many hours do you sleep?"],
                ),
            ],
            overall_assessment="Several clinically relevant gaps identified.",
        )
        assert len(r.missing_information) == 1
        assert "tired" in r.input_summary
        assert r.overall_assessment

    def test_frozen(self):
        r = MissingInfoResult()
        with pytest.raises(Exception):
            r.input_summary = "changed"


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

GOOD_JSON_RESPONSE = """{
  "input_summary": "Client reports waking up tired every day with no additional clinical context.",
  "missing_information": [
    {
      "info_gap": "Sleep quality and duration",
      "clinical_relevance": "Sleep disturbance is a transdiagnostic factor that can worsen mood, anxiety, and cognitive function. It also guides treatment choices such as CBT-i versus medication.",
      "suggested_questions": [
        "Can you tell me about your sleep — how many hours do you typically get?",
        "Do you have trouble falling asleep, staying asleep, or waking too early?",
        "How does your sleep affect how you feel during the day?"
      ]
    },
    {
      "info_gap": "Suicidal ideation and self-harm risk",
      "clinical_relevance": "Any expression of distress warrants a safety assessment. Identifying suicidal ideation is essential for risk management and treatment planning.",
      "suggested_questions": [
        "Have you had any thoughts about hurting yourself or ending your life?",
        "What supports do you have in place if you feel unsafe?"
      ]
    },
    {
      "info_gap": "Duration and onset of symptoms",
      "clinical_relevance": "Knowing how long the fatigue has been present helps distinguish acute from chronic conditions and guides urgency of intervention.",
      "suggested_questions": [
        "How long have you been waking up tired?",
        "Did it start gradually or suddenly?"
      ]
    }
  ],
  "overall_assessment": "The provided information indicates low energy but lacks essential details on sleep quality, duration, risk, mood, and medical history. A comprehensive assessment is needed before any clinical decisions can be made."
}"""

EMPTY_JSON_RESPONSE = """{
  "input_summary": "No specific gaps identified.",
  "missing_information": [],
  "overall_assessment": "The information provided is comprehensive."
}"""

MARKDOWN_FENCED_RESPONSE = """```json
{
  "input_summary": "Markdown fenced response for testing.",
  "missing_information": [
    {
      "info_gap": "Test gap from markdown fence",
      "clinical_relevance": "This is a test gap with enough characters for validation purposes.",
      "suggested_questions": ["Is this a test question?"]
    }
  ],
  "overall_assessment": "Test assessment."
}
```"""

INVALID_JSON_RESPONSE = "This is not JSON at all"

PARTIAL_JSON_RESPONSE = """{
  "input_summary": "Partial response for testing.",
  "missing_information": [
    {
      "info_gap": "A single gap from partial response",
      "clinical_relevance": "Clinical relevance text for testing the partial response path.",
      "suggested_questions": ["What about X?"]
    }
  ]
}"""


# ═══════════════════════════════════════════════════════════════
# MissingInfoDetector tests
# ═══════════════════════════════════════════════════════════════

class TestMissingInfoDetector:
    def make_detector(self, response: str = GOOD_JSON_RESPONSE) -> MissingInfoDetector:
        llm = MagicMock()
        llm.generate = AsyncMock(return_value=response)
        return MissingInfoDetector(llm)

    # ── Happy path ──────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_detect_basic(self):
        detector = self.make_detector()
        result = await detector.detect("I wake up tired every day.")
        assert isinstance(result, MissingInfoResult)
        assert len(result.missing_information) == 3
        assert result.overall_assessment
        assert result.detection_ms > 0

    @pytest.mark.asyncio
    async def test_detect_with_context(self):
        detector = self.make_detector()
        result = await detector.detect(
            "I feel sad sometimes.",
            context="Client is a 35-year-old professional with workplace stress.",
        )
        assert len(result.missing_information) == 3

    @pytest.mark.asyncio
    async def test_detect_different_statements(self):
        detector = self.make_detector()
        result = await detector.detect("I have been feeling anxious about social situations for months.")
        assert len(result.missing_information) > 0

    # ── Empty / edge cases ──────────────────────────────────

    @pytest.mark.asyncio
    async def test_empty_input(self):
        detector = self.make_detector()
        result = await detector.detect("")
        assert result.missing_information == []
        assert "empty input" in result.overall_assessment

    @pytest.mark.asyncio
    async def test_whitespace_input(self):
        detector = self.make_detector()
        result = await detector.detect("   ")
        assert result.missing_information == []

    @pytest.mark.asyncio
    async def test_empty_llm_response(self):
        detector = self.make_detector(response=EMPTY_JSON_RESPONSE)
        result = await detector.detect("I feel fine now.")
        assert result.missing_information == []

    # ── Response parsing ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_markdown_fenced_response(self):
        detector = self.make_detector(response=MARKDOWN_FENCED_RESPONSE)
        result = await detector.detect("Test input for markdown parsing.")
        assert len(result.missing_information) == 1
        assert "markdown" in result.missing_information[0].info_gap.lower()

    @pytest.mark.asyncio
    async def test_invalid_json_response_falls_back(self):
        detector = self.make_detector(response=INVALID_JSON_RESPONSE)
        result = await detector.detect("Test input for invalid JSON.")
        assert "could not be completed" in result.overall_assessment.lower()

    @pytest.mark.asyncio
    async def test_partial_json_response(self):
        detector = self.make_detector(response=PARTIAL_JSON_RESPONSE)
        result = await detector.detect("Test input for partial JSON.")
        assert len(result.missing_information) == 1
        assert result.overall_assessment == ""

    # ── LLM error handling ──────────────────────────────────

    @pytest.mark.asyncio
    async def test_llm_raises_exception(self):
        llm = MagicMock()
        llm.generate = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
        detector = MissingInfoDetector(llm)
        result = await detector.detect("Test input for LLM error.")
        assert "could not be completed" in result.overall_assessment.lower()

    # ── Builder resilience ──────────────────────────────────

    def test_build_items_skips_invalid(self):
        detector = self.make_detector()
        raw = [
            {
                "info_gap": "Sleep quality and duration",
                "clinical_relevance": "Clinical relevance text for testing purposes here.",
                "suggested_questions": ["How many hours do you sleep?"],
            },
            {"info_gap": "", "clinical_relevance": "text", "suggested_questions": ["q"]},
            {"not_info_gap": "missing"},
            42,
            None,
        ]
        items = detector._build_items(raw)
        assert len(items) == 1
        assert items[0].info_gap == "Sleep quality and duration"

    # ── Standalone instantiation ────────────────────────────

    def test_can_be_instantiated_without_pipeline(self):
        llm = MagicMock()
        detector = MissingInfoDetector(llm)
        assert detector is not None
        assert detector._llm is llm

    def test_result_is_valid_pydantic(self):
        r = MissingInfoResult()
        assert isinstance(r, MissingInfoResult)
        assert r.model_config.get("frozen")
