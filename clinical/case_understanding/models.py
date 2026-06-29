from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ConfidenceRating(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    UNSPECIFIED = "unspecified"


class Duration(BaseModel):
    value: int | None = Field(None, description="Numeric duration value")
    unit: str | None = Field(
        None,
        description="Unit of time: days, weeks, months, years",
    )
    original_text: str | None = Field(
        None,
        description="Raw text from which duration was extracted",
    )

    model_config = {"frozen": True}


class PreviousTreatment(BaseModel):
    modality: str | None = Field(None, description="Type of treatment, e.g. CBT, medication")
    response: str | None = Field(
        None,
        description="Reported response: improved, no change, worsened, unknown",
    )
    duration: str | None = Field(None, description="Duration of previous treatment")
    original_text: str | None = Field(None, description="Raw text mentioning treatment")

    model_config = {"frozen": True}


class ExtractedField(BaseModel):
    """Wrapper for an extracted field with confidence and provenance."""

    value: Any = Field(..., description="The extracted value")
    confidence: ConfidenceRating = Field(
        default=ConfidenceRating.UNKNOWN,
        description="Confidence in this extracted value",
    )
    source_text: str | None = Field(
        None,
        description="The input text snippet that supports this value",
    )

    model_config = {"frozen": True}


class DemographicInfo(BaseModel):
    age: ExtractedField | None = Field(None, description="Client age")
    gender: ExtractedField | None = Field(None, description="Client gender")
    occupation: ExtractedField | None = Field(None, description="Client occupation")

    model_config = {"frozen": True}


class ClinicalPresentation(BaseModel):
    presenting_concerns: list[ExtractedField] = Field(
        default_factory=list,
        description="Primary reasons for seeking help",
    )
    symptoms: list[ExtractedField] = Field(
        default_factory=list,
        description="Reported signs and symptoms",
    )
    emotional_indicators: list[ExtractedField] = Field(
        default_factory=list,
        description="Affective/emotional state observations",
    )
    behavioural_indicators: list[ExtractedField] = Field(
        default_factory=list,
        description="Observed or reported behaviours",
    )
    duration: Duration | None = Field(
        None,
        description="Duration of presenting concerns",
    )

    model_config = {"frozen": True}


class ContextualFactors(BaseModel):
    stressors: list[ExtractedField] = Field(
        default_factory=list,
        description="Identified psychosocial stressors",
    )
    protective_factors: list[ExtractedField] = Field(
        default_factory=list,
        description="Protective or resilience factors",
    )
    risk_factors: list[ExtractedField] = Field(
        default_factory=list,
        description="Risk factors (e.g. self-harm, substance use, isolation)",
    )
    functional_impairment: ExtractedField | None = Field(
        None,
        description="Impact on daily functioning",
    )
    social_context: ExtractedField | None = Field(
        None,
        description="Social support, relationships, living situation",
    )

    model_config = {"frozen": True}


class TreatmentHistory(BaseModel):
    previous_treatment: list[PreviousTreatment] = Field(
        default_factory=list,
        description="Past or current treatments mentioned",
    )

    model_config = {"frozen": True}


class OverallSeverity(BaseModel):
    severity: Severity = Field(
        default=Severity.UNSPECIFIED,
        description="Overall clinical severity assessment",
    )
    confidence: ConfidenceRating = Field(
        default=ConfidenceRating.UNKNOWN,
        description="Confidence in severity assessment",
    )
    rationale: str | None = Field(
        None,
        description="Brief rationale for severity rating",
    )

    model_config = {"frozen": True}


class CaseUnderstandingResult(BaseModel):
    """Complete structured output of the case understanding extraction.

    This is the top-level result that future LangGraph agents will receive
    as input. Every field is optional so agents can handle partial data
    gracefully.
    """

    demographic: DemographicInfo = Field(
        default_factory=DemographicInfo,
        description="Demographic information",
    )
    clinical_presentation: ClinicalPresentation = Field(
        default_factory=ClinicalPresentation,
        description="Clinical signs, symptoms, and presentation details",
    )
    contextual_factors: ContextualFactors = Field(
        default_factory=ContextualFactors,
        description="Environmental and psychosocial context",
    )
    treatment_history: TreatmentHistory = Field(
        default_factory=TreatmentHistory,
        description="Previous treatment information",
    )
    overall_severity: OverallSeverity = Field(
        default_factory=OverallSeverity,
        description="Overall clinical severity assessment",
    )
    raw_text: str = Field(
        ...,
        description="Original input text that was analysed",
    )
    extracted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this extraction was performed",
    )
    extraction_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Time taken for extraction in milliseconds",
    )

    def to_flat_dict(self) -> dict[str, Any]:
        """Flatten the nested structure into a single dict for easy consumption.

        LangGraph agents can use this to access fields without navigating
        the nested model hierarchy.
        """
        result: dict[str, Any] = {
            "age": self._unwrap(self.demographic.age),
            "gender": self._unwrap(self.demographic.gender),
            "occupation": self._unwrap(self.demographic.occupation),
            "presenting_concerns": [self._unwrap(c) for c in self.clinical_presentation.presenting_concerns],
            "symptoms": [self._unwrap(s) for s in self.clinical_presentation.symptoms],
            "emotional_indicators": [self._unwrap(e) for e in self.clinical_presentation.emotional_indicators],
            "behavioural_indicators": [self._unwrap(b) for b in self.clinical_presentation.behavioural_indicators],
            "stressors": [self._unwrap(s) for s in self.contextual_factors.stressors],
            "protective_factors": [self._unwrap(p) for p in self.contextual_factors.protective_factors],
            "risk_factors": [self._unwrap(r) for r in self.contextual_factors.risk_factors],
            "functional_impairment": self._unwrap(self.contextual_factors.functional_impairment),
            "social_context": self._unwrap(self.contextual_factors.social_context),
            "duration": self.clinical_presentation.duration.original_text if self.clinical_presentation.duration else None,
            "previous_treatment": [
                {"modality": t.modality, "response": t.response}
                for t in self.treatment_history.previous_treatment
            ],
            "severity": self.overall_severity.severity.value,
            "severity_confidence": self.overall_severity.confidence.value,
        }
        return result

    @staticmethod
    def _unwrap(field: ExtractedField | None) -> Any:
        """Extract the value from an ExtractedField, or None."""
        if field is None:
            return None
        return field.value

    model_config = {"frozen": True}
