from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from clinical.case_understanding import (
    CaseUnderstandingExtractor,
    CaseUnderstandingResult,
    ClinicalPresentation,
    ConfidenceRating,
    ContextualFactors,
    DemographicInfo,
    Duration,
    ExtractedField,
    OverallSeverity,
    PreviousTreatment,
    Severity,
    TreatmentHistory,
)


# ═══════════════════════════════════════════════════════════════
# Model unit tests
# ═══════════════════════════════════════════════════════════════

class TestExtractedField:
    def test_basic_creation(self):
        f = ExtractedField(value="male", confidence=ConfidenceRating.HIGH)
        assert f.value == "male"
        assert f.confidence == ConfidenceRating.HIGH

    def test_with_source_text(self):
        f = ExtractedField(
            value=35,
            confidence=ConfidenceRating.MEDIUM,
            source_text="35-year-old male",
        )
        assert f.source_text == "35-year-old male"

    def test_frozen(self):
        f = ExtractedField(value="test", confidence=ConfidenceRating.LOW)
        with pytest.raises(Exception):
            f.value = "changed"


class TestDuration:
    def test_full(self):
        d = Duration(value=2, unit="weeks", original_text="for 2 weeks")
        assert d.value == 2
        assert d.unit == "weeks"

    def test_partial(self):
        d = Duration(original_text="chronic")
        assert d.value is None
        assert d.unit is None

    def test_frozen(self):
        d = Duration(value=1, unit="month")
        with pytest.raises(Exception):
            d.value = 2


class TestPreviousTreatment:
    def test_full(self):
        t = PreviousTreatment(
            modality="CBT",
            response="improved",
            duration="6 months",
            original_text="had CBT for 6 months with improvement",
        )
        assert t.modality == "CBT"

    def test_minimal(self):
        t = PreviousTreatment(modality="medication")
        assert t.response is None


class TestDemographicInfo:
    def test_all_fields_present(self):
        d = DemographicInfo(
            age=ExtractedField(value=30, confidence=ConfidenceRating.HIGH),
            gender=ExtractedField(value="F", confidence=ConfidenceRating.HIGH),
            occupation=ExtractedField(value="teacher", confidence=ConfidenceRating.MEDIUM),
        )
        assert d.age.value == 30
        assert d.gender.value == "F"
        assert d.occupation.value == "teacher"

    def test_all_none(self):
        d = DemographicInfo()
        assert d.age is None
        assert d.gender is None
        assert d.occupation is None


class TestClinicalPresentation:
    def test_with_symptoms(self):
        p = ClinicalPresentation(
            presenting_concerns=[
                ExtractedField(value="anxiety", confidence=ConfidenceRating.HIGH),
            ],
            symptoms=[
                ExtractedField(value="worry", confidence=ConfidenceRating.HIGH),
                ExtractedField(value="fatigue", confidence=ConfidenceRating.MEDIUM),
            ],
            emotional_indicators=[
                ExtractedField(value="irritable", confidence=ConfidenceRating.LOW),
            ],
        )
        assert len(p.symptoms) == 2
        assert p.duration is None

    def test_with_duration(self):
        p = ClinicalPresentation(
            presenting_concerns=[],
            symptoms=[],
            duration=Duration(value=3, unit="months"),
        )
        assert p.duration.value == 3


class TestContextualFactors:
    def test_with_stressors(self):
        c = ContextualFactors(
            stressors=[ExtractedField(value="job loss", confidence=ConfidenceRating.HIGH)],
            protective_factors=[ExtractedField(value="strong family support", confidence=ConfidenceRating.MEDIUM)],
            risk_factors=[ExtractedField(value="social isolation", confidence=ConfidenceRating.HIGH)],
        )
        assert len(c.stressors) == 1
        assert len(c.protective_factors) == 1
        assert len(c.risk_factors) == 1

    def test_empty(self):
        c = ContextualFactors()
        assert c.stressors == []
        assert c.protective_factors == []
        assert c.risk_factors == []
        assert c.functional_impairment is None


class TestTreatmentHistory:
    def test_with_treatments(self):
        h = TreatmentHistory(
            previous_treatment=[
                PreviousTreatment(modality="CBT", response="improved"),
                PreviousTreatment(modality="SSRI", response="no change"),
            ]
        )
        assert len(h.previous_treatment) == 2

    def test_empty(self):
        h = TreatmentHistory()
        assert h.previous_treatment == []


class TestOverallSeverity:
    def test_severity_validation(self):
        s = OverallSeverity(severity=Severity.SEVERE, confidence=ConfidenceRating.HIGH)
        assert s.severity == Severity.SEVERE

    def test_defaults(self):
        s = OverallSeverity()
        assert s.severity == Severity.UNSPECIFIED
        assert s.confidence == ConfidenceRating.UNKNOWN
        assert s.rationale is None


class TestCaseUnderstandingResult:
    def test_minimal(self):
        r = CaseUnderstandingResult(raw_text="Patient feels sad.")
        assert r.demographic is not None
        assert r.clinical_presentation is not None
        assert r.contextual_factors is not None
        assert r.treatment_history is not None
        assert r.overall_severity is not None

    def test_to_flat_dict(self):
        r = CaseUnderstandingResult(
            demographic=DemographicInfo(
                age=ExtractedField(value=45, confidence=ConfidenceRating.HIGH),
                gender=ExtractedField(value="M", confidence=ConfidenceRating.HIGH),
            ),
            clinical_presentation=ClinicalPresentation(
                presenting_concerns=[
                    ExtractedField(value="depression", confidence=ConfidenceRating.HIGH),
                ],
                symptoms=[
                    ExtractedField(value="low mood", confidence=ConfidenceRating.HIGH),
                ],
                duration=Duration(value=6, unit="months", original_text="for 6 months"),
            ),
            contextual_factors=ContextualFactors(
                stressors=[ExtractedField(value="divorce", confidence=ConfidenceRating.HIGH)],
            ),
            treatment_history=TreatmentHistory(
                previous_treatment=[
                    PreviousTreatment(modality="CBT", response="improved"),
                ]
            ),
            overall_severity=OverallSeverity(
                severity=Severity.MODERATE,
                confidence=ConfidenceRating.MEDIUM,
            ),
            raw_text="45-year-old male with depression for 6 months.",
        )
        flat = r.to_flat_dict()
        assert flat["age"] == 45
        assert flat["gender"] == "M"
        assert flat["presenting_concerns"] == ["depression"]
        assert flat["symptoms"] == ["low mood"]
        assert flat["duration"] == "for 6 months"
        assert flat["previous_treatment"] == [
            {"modality": "CBT", "response": "improved"}
        ]
        assert flat["severity"] == "moderate"

    def test_to_flat_dict_empty(self):
        r = CaseUnderstandingResult(raw_text="")
        flat = r.to_flat_dict()
        assert flat["age"] is None
        assert flat["symptoms"] == []
        assert flat["duration"] is None
        assert flat["previous_treatment"] == []


# ═══════════════════════════════════════════════════════════════
# Extractor tests
# ═══════════════════════════════════════════════════════════════

_MOCK_FULL_RESPONSE = """{
  "age": {"value": 27, "confidence": "high", "source_text": "27-year-old"},
  "gender": {"value": "male", "confidence": "high", "source_text": "27-year-old male"},
  "occupation": {"value": "software engineer", "confidence": "medium", "source_text": "works as a software engineer"},
  "presenting_concerns": [
    {"value": "depressed mood", "confidence": "high", "source_text": "feeling depressed for 2 weeks"},
    {"value": "loss of interest", "confidence": "high", "source_text": "lost interest in everything"}
  ],
  "symptoms": [
    {"value": "low mood", "confidence": "high", "source_text": "feeling depressed"},
    {"value": "anhedonia", "confidence": "high", "source_text": "lost interest in everything"},
    {"value": "fatigue", "confidence": "medium", "source_text": "tired all the time"}
  ],
  "emotional_indicators": [
    {"value": "anxious", "confidence": "medium", "source_text": "feels anxious"},
    {"value": "irritable", "confidence": "low", "source_text": "easily irritated"}
  ],
  "behavioural_indicators": [
    {"value": "social withdrawal", "confidence": "high", "source_text": "stopped seeing friends"},
    {"value": "reduced activity", "confidence": "medium", "source_text": "stays in bed"}
  ],
  "stressors": [
    {"value": "work pressure", "confidence": "high", "source_text": "high workload at job"},
    {"value": "relationship conflict", "confidence": "medium", "source_text": "arguing with partner"}
  ],
  "protective_factors": [
    {"value": "supportive partner", "confidence": "high", "source_text": "partner is supportive"}
  ],
  "risk_factors": [
    {"value": "family history of depression", "confidence": "medium", "source_text": "mother had depression"}
  ],
  "functional_impairment": {"value": "unable to work for past week", "confidence": "high", "source_text": "has not been able to go to work"},
  "social_context": {"value": "lives with partner, limited social network", "confidence": "medium", "source_text": "lives with partner"},
  "duration": {"value": 2, "unit": "weeks", "original_text": "for the past 2 weeks"},
  "previous_treatment": [
    {"modality": "CBT", "response": "partial improvement", "duration": "3 months", "original_text": "had CBT 2 years ago"}
  ],
  "severity": "moderate",
  "severity_rationale": "Multiple depressive symptoms with functional impairment but retains some protective factors"
}"""

_MOCK_MINIMAL_RESPONSE = """{
  "age": {"value": null, "confidence": "unknown", "source_text": null},
  "gender": {"value": null, "confidence": "unknown", "source_text": null},
  "occupation": {"value": null, "confidence": "unknown", "source_text": null},
  "presenting_concerns": [],
  "symptoms": [{"value": "anxiety", "confidence": "low", "source_text": "feeling anxious"}],
  "emotional_indicators": [],
  "behavioural_indicators": [],
  "stressors": [],
  "protective_factors": [],
  "risk_factors": [],
  "functional_impairment": {"value": null, "confidence": "unknown", "source_text": null},
  "social_context": {"value": null, "confidence": "unknown", "source_text": null},
  "duration": {"value": null, "unit": null, "original_text": null},
  "previous_treatment": [],
  "severity": "unspecified",
  "severity_rationale": null
}"""


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.generate = AsyncMock()
    return llm


@pytest.fixture
def extractor(mock_llm):
    return CaseUnderstandingExtractor(llm=mock_llm)


class TestExtractor:
    @pytest.mark.asyncio
    async def test_full_case_understanding(self, extractor, mock_llm):
        mock_llm.generate.return_value = _MOCK_FULL_RESPONSE

        result = await extractor.extract(
            "27-year-old male software engineer with depressed mood and anhedonia for 2 weeks."
        )

        assert result.demographic.age.value == 27
        assert result.demographic.gender.value == "male"
        assert result.demographic.occupation.value == "software engineer"
        assert len(result.clinical_presentation.presenting_concerns) == 2
        assert len(result.clinical_presentation.symptoms) == 3
        assert len(result.clinical_presentation.emotional_indicators) == 2
        assert len(result.clinical_presentation.behavioural_indicators) == 2
        assert len(result.contextual_factors.stressors) == 2
        assert len(result.contextual_factors.protective_factors) == 1
        assert len(result.contextual_factors.risk_factors) == 1
        assert result.contextual_factors.functional_impairment.value == "unable to work for past week"
        assert result.contextual_factors.social_context is not None
        assert result.clinical_presentation.duration.value == 2
        assert result.clinical_presentation.duration.unit == "weeks"
        assert len(result.treatment_history.previous_treatment) == 1
        assert result.treatment_history.previous_treatment[0].modality == "CBT"
        assert result.overall_severity.severity == Severity.MODERATE
        assert result.overall_severity.rationale is not None
        assert result.extraction_ms > 0

    @pytest.mark.asyncio
    async def test_minimal_input(self, extractor, mock_llm):
        mock_llm.generate.return_value = _MOCK_MINIMAL_RESPONSE

        result = await extractor.extract("I feel anxious.")

        assert result.demographic.age is None
        assert result.demographic.gender is None
        assert result.clinical_presentation.presenting_concerns == []
        assert len(result.clinical_presentation.symptoms) == 1
        assert result.clinical_presentation.symptoms[0].value == "anxiety"
        assert result.clinical_presentation.symptoms[0].confidence == ConfidenceRating.LOW
        assert result.clinical_presentation.duration is None
        assert result.contextual_factors.stressors == []
        assert result.treatment_history.previous_treatment == []
        assert result.overall_severity.severity == Severity.UNSPECIFIED

    @pytest.mark.asyncio
    async def test_flat_dict_output(self, extractor, mock_llm):
        mock_llm.generate.return_value = _MOCK_FULL_RESPONSE

        result = await extractor.extract("27-year-old male.")
        flat = result.to_flat_dict()

        assert flat["age"] == 27
        assert flat["gender"] == "male"
        assert flat["occupation"] == "software engineer"
        assert "depressed mood" in flat["presenting_concerns"]
        assert "low mood" in flat["symptoms"]
        assert "anxious" in flat["emotional_indicators"]
        assert "social withdrawal" in flat["behavioural_indicators"]
        assert "work pressure" in flat["stressors"]
        assert "supportive partner" in flat["protective_factors"]
        assert "family history of depression" in flat["risk_factors"]
        assert flat["functional_impairment"] == "unable to work for past week"
        assert flat["duration"] == "for the past 2 weeks"
        assert flat["previous_treatment"] == [
            {"modality": "CBT", "response": "partial improvement"}
        ]
        assert flat["severity"] == "moderate"

    @pytest.mark.asyncio
    async def test_empty_input(self, extractor, mock_llm):
        result = await extractor.extract("")
        assert result.demographic.age is None
        assert result.clinical_presentation.symptoms == []
        assert result.overall_severity.severity == Severity.UNSPECIFIED
        mock_llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_whitespace_input(self, extractor, mock_llm):
        result = await extractor.extract("   ")
        assert result.demographic.age is None
        mock_llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_llm_returns_malformed_json(self, extractor, mock_llm):
        mock_llm.generate.return_value = "this is not json"

        result = await extractor.extract("Patient has anxiety.")
        # Should return empty result instead of crashing
        assert result.demographic.age is None
        assert result.clinical_presentation.symptoms == []

    @pytest.mark.asyncio
    async def test_llm_returns_json_with_markdown_fence(self, extractor, mock_llm):
        mock_llm.generate.return_value = f"```json\n{_MOCK_FULL_RESPONSE}\n```"

        result = await extractor.extract("Patient text.")
        assert result.demographic.age.value == 27
        assert result.overall_severity.severity == Severity.MODERATE

    @pytest.mark.asyncio
    async def test_llm_failure_graceful_degradation(self, extractor, mock_llm):
        mock_llm.generate.side_effect = RuntimeError("Ollama unavailable")

        result = await extractor.extract("Patient has depression.")
        # Must return empty result, not crash
        assert isinstance(result, CaseUnderstandingResult)
        assert result.demographic.age is None
        assert result.clinical_presentation.symptoms == []
        assert result.raw_text == "Patient has depression."

    @pytest.mark.asyncio
    async def test_partial_data_with_missing_fields(self, extractor, mock_llm):
        partial = """{
          "age": {"value": 35, "confidence": "high", "source_text": "35-year-old"},
          "gender": {"value": null, "confidence": "unknown", "source_text": null},
          "occupation": {"value": null, "confidence": "unknown", "source_text": null},
          "presenting_concerns": [],
          "symptoms": [],
          "emotional_indicators": [],
          "behavioural_indicators": [],
          "stressors": [],
          "protective_factors": [],
          "risk_factors": [],
          "functional_impairment": {"value": null, "confidence": "unknown", "source_text": null},
          "social_context": {"value": null, "confidence": "unknown", "source_text": null},
          "duration": {"value": null, "unit": null, "original_text": null},
          "previous_treatment": [],
          "severity": "unspecified",
          "severity_rationale": null
        }"""
        mock_llm.generate.return_value = partial

        result = await extractor.extract("35-year-old patient.")
        assert result.demographic.age.value == 35
        assert result.demographic.gender is None
        assert result.demographic.occupation is None
        assert result.clinical_presentation.symptoms == []
        assert result.clinical_presentation.duration is None

    @pytest.mark.asyncio
    async def test_confidence_rating_coercion(self, extractor, mock_llm):
        response = """{
          "age": {"value": 25, "confidence": "very sure", "source_text": "25"},
          "gender": {"value": null, "confidence": "unknown", "source_text": null},
          "occupation": {"value": null, "confidence": "unknown", "source_text": null},
          "presenting_concerns": [],
          "symptoms": [],
          "emotional_indicators": [],
          "behavioural_indicators": [],
          "stressors": [],
          "protective_factors": [],
          "risk_factors": [],
          "functional_impairment": {"value": null, "confidence": "unknown", "source_text": null},
          "social_context": {"value": null, "confidence": "unknown", "source_text": null},
          "duration": {"value": null, "unit": null, "original_text": null},
          "previous_treatment": [],
          "severity": "unspecified",
          "severity_rationale": null
        }"""
        mock_llm.generate.return_value = response

        result = await extractor.extract("25-year-old.")
        # Invalid confidence "very sure" should fall back to UNKNOWN
        assert result.demographic.age.confidence == ConfidenceRating.UNKNOWN

    @pytest.mark.asyncio
    async def test_previous_treatment_list_handling(self, extractor, mock_llm):
        response = """{
          "age": {"value": null, "confidence": "unknown", "source_text": null},
          "gender": {"value": null, "confidence": "unknown", "source_text": null},
          "occupation": {"value": null, "confidence": "unknown", "source_text": null},
          "presenting_concerns": [],
          "symptoms": [],
          "emotional_indicators": [],
          "behavioural_indicators": [],
          "stressors": [],
          "protective_factors": [],
          "risk_factors": [],
          "functional_impairment": {"value": null, "confidence": "unknown", "source_text": null},
          "social_context": {"value": null, "confidence": "unknown", "source_text": null},
          "duration": {"value": null, "unit": null, "original_text": null},
          "previous_treatment": [
            {"modality": "CBT", "response": "improved", "duration": "12 weeks", "original_text": "completed 12-week CBT"},
            {"modality": "sertraline", "response": "no response", "duration": null, "original_text": "tried sertraline"}
          ],
          "severity": "mild",
          "severity_rationale": null
        }"""
        mock_llm.generate.return_value = response

        result = await extractor.extract("Had CBT and medication.")
        assert len(result.treatment_history.previous_treatment) == 2
        assert result.treatment_history.previous_treatment[1].modality == "sertraline"
        assert result.treatment_history.previous_treatment[1].response == "no response"

    @pytest.mark.asyncio
    async def test_severity_enum_fallback(self, extractor, mock_llm):
        response = """{
          "age": {"value": null, "confidence": "unknown", "source_text": null},
          "gender": {"value": null, "confidence": "unknown", "source_text": null},
          "occupation": {"value": null, "confidence": "unknown", "source_text": null},
          "presenting_concerns": [],
          "symptoms": [],
          "emotional_indicators": [],
          "behavioural_indicators": [],
          "stressors": [],
          "protective_factors": [],
          "risk_factors": [],
          "functional_impairment": {"value": null, "confidence": "unknown", "source_text": null},
          "social_context": {"value": null, "confidence": "unknown", "source_text": null},
          "duration": {"value": null, "unit": null, "original_text": null},
          "previous_treatment": [],
          "severity": "critical",
          "severity_rationale": null
        }"""
        mock_llm.generate.return_value = response

        result = await extractor.extract("Test.")
        # Invalid severity "critical" should fall back to UNSPECIFIED
        assert result.overall_severity.severity == Severity.UNSPECIFIED


# ═══════════════════════════════════════════════════════════════
# Extractor can be instantiated standalone (LangGraph agent use)
# ═══════════════════════════════════════════════════════════════

class TestExtractorStandalone:
    def test_can_be_instantiated_without_pipeline(self, mock_llm):
        extractor = CaseUnderstandingExtractor(llm=mock_llm)
        assert extractor is not None

    def test_result_is_valid_pydantic(self):
        r = CaseUnderstandingResult(raw_text="test")
        assert r.model_dump() is not None
