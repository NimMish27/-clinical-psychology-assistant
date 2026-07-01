from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class Formulation(BaseModel):
    """A single possible clinical formulation for the case.

    Each formulation offers one way of understanding the client's
    presentation. It is NOT a diagnosis — it is a hypothesis that
    organises the available information into a coherent clinical
    picture.
    """

    label: str = Field(
        ...,
        min_length=5,
        description="Short descriptive label for this formulation (e.g. 'Cognitive-behavioural understanding centred on avoidant coping')",
    )
    explanation: str = Field(
        ...,
        min_length=20,
        description="Detailed narrative explaining how the client's difficulties may have developed and are maintained",
    )
    supporting_symptoms: list[str] = Field(
        ...,
        min_length=1,
        description="Specific symptoms or observations from the case that support this formulation",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in this formulation given the available information (0=tentative, 1=strongly supported)",
    )

    model_config = {"frozen": True}


class ClinicalFormulationResult(BaseModel):
    """Structured output of the clinical formulation process.

    This is the top-level result that LangGraph agents and the clinical
    pipeline can consume.  Every field is optional so downstream consumers
    can handle partial data gracefully.

    The formulation module NEVER produces a diagnosis.  It generates
    hypotheses that help organise clinical thinking.
    """

    case_summary: str = Field(
        ...,
        min_length=20,
        description="Concise summary of the client's presentation, key concerns, and relevant context",
    )
    possible_formulations: list[Formulation] = Field(
        default_factory=list,
        description="One or more possible ways of understanding the clinical picture",
    )
    supporting_evidence: list[str] = Field(
        default_factory=list,
        description="Evidence from the case and literature that strengthens these formulations",
    )
    alternative_explanations: list[str] = Field(
        default_factory=list,
        description="Alternative interpretations that cannot be ruled out with current information",
    )
    missing_assessment_information: list[str] = Field(
        default_factory=list,
        description="Specific information that would strengthen or refute the formulations above",
    )
    caution: str = Field(
        default="",
        description="Clinical cautionary note — e.g. limitations, comorbidity considerations, cultural factors",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall confidence in the formulation (0=highly speculative, 1=well-supported by evidence)",
    )
    formulated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    formulation_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Time taken for formulation in milliseconds",
    )

    model_config = {"frozen": True}
