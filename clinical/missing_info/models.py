from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class MissingInfoItem(BaseModel):
    """A specific piece of clinical information that is missing from the input."""

    info_gap: str = Field(
        ...,
        min_length=5,
        description="What information is missing (e.g. sleep quality, suicidal ideation, medical history)",
    )
    clinical_relevance: str = Field(
        ...,
        min_length=10,
        description="Why this information matters clinically — how it would affect formulation, risk assessment, or treatment planning",
    )
    suggested_questions: list[str] = Field(
        ...,
        min_length=1,
        description="Specific assessment questions the clinician could ask to gather this information",
    )

    model_config = {"frozen": True}


class MissingInfoResult(BaseModel):
    """Structured output of the missing information detection process.

    Identifies clinical information gaps in the provided text without
    generating conclusions or diagnoses.  Every field is optional so
    downstream consumers can handle partial data gracefully.
    """

    input_summary: str = Field(
        default="",
        description="Brief summary of the clinical information that was provided",
    )
    missing_information: list[MissingInfoItem] = Field(
        default_factory=list,
        description="Specific pieces of clinical information that appear to be missing",
    )
    overall_assessment: str = Field(
        default="",
        description="General assessment of the completeness of the clinical picture",
    )
    detected_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    detection_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Time taken for detection in milliseconds",
    )

    model_config = {"frozen": True}
