from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ClinicalInputType(str, Enum):
    SINGLE_STATEMENT = "single_statement"
    SYMPTOM_LIST = "symptom_list"
    CASE_STUDY = "case_study"


class ClinicalInput(BaseModel):
    raw_text: str = Field(
        ...,
        min_length=3,
        max_length=50000,
        description="Raw clinical input text",
    )
    input_type: ClinicalInputType | None = Field(
        None,
        description="Override auto-detection of input type",
    )

    model_config = {"frozen": True}


class CaseUnderstanding(BaseModel):
    input_type: ClinicalInputType
    summary: str = Field(..., description="One-paragraph summary of the case")
    key_topics: list[str] = Field(
        default_factory=list,
        description="Clinical topics identified (e.g. depression, CBT, comorbidity)",
    )
    clinical_context: str | None = Field(
        None,
        description="Relevant clinical context or setting (e.g. outpatient, emergency, primary care)",
    )

    model_config = {"frozen": True}


class ClinicalFeatures(BaseModel):
    symptoms: list[str] = Field(
        default_factory=list,
        description="Reported signs and symptoms",
    )
    diagnoses: list[str] = Field(
        default_factory=list,
        description="Mentioned or suspected diagnoses",
    )
    patient_history: list[str] = Field(
        default_factory=list,
        description="Relevant personal, medical, and psychiatric history",
    )
    family_history: list[str] = Field(
        default_factory=list,
        description="Family psychiatric or medical history",
    )
    risk_factors: list[str] = Field(
        default_factory=list,
        description="Risk factors identified (e.g. social isolation, substance use)",
    )
    protective_factors: list[str] = Field(
        default_factory=list,
        description="Protective or resilience factors",
    )
    treatment_history: list[str] = Field(
        default_factory=list,
        description="Past or current treatments and response",
    )
    other_relevant: list[str] = Field(
        default_factory=list,
        description="Other clinically relevant observations",
    )

    model_config = {"frozen": True}


class RetrievalQuery(BaseModel):
    query: str = Field(..., description="Search query for ChromaDB retrieval")
    weight: float = Field(
        default=1.0,
        ge=0.1,
        le=3.0,
        description="Relative importance weight for ranking fusion",
    )
    rationale: str = Field(
        ...,
        description="Clinical rationale for generating this query",
    )

    model_config = {"frozen": True}


class EvidenceSynthesis(BaseModel):
    synthesis: str = Field(
        ...,
        description="Integrated summary of retrieved evidence",
    )
    supporting_evidence: list[str] = Field(
        default_factory=list,
        description="Evidence that supports the clinical picture",
    )
    contradicting_evidence: list[str] = Field(
        default_factory=list,
        description="Evidence that contradicts or complicates",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in the synthesis (0=speculative, 1=well-supported)",
    )

    model_config = {"frozen": True}


class ClinicalResponse(BaseModel):
    analysis: str = Field(
        ...,
        description="Comprehensive clinical analysis",
    )
    formulation: str | None = Field(
        None,
        description="Case formulation integrating biopsychosocial factors",
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Evidence-based recommendations",
    )
    evidence_summary: str = Field(
        ...,
        description="Brief summary of evidence used",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall confidence in the response",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Limitations and caveats",
    )

    model_config = {"frozen": True}


class PipelineResult(BaseModel):
    """Full result returned by the clinical pipeline orchestrator."""
    input_type: ClinicalInputType
    understanding: CaseUnderstanding
    features: ClinicalFeatures
    queries: list[RetrievalQuery]
    evidence: EvidenceSynthesis
    response: ClinicalResponse
    elapsed_ms: float = Field(..., ge=0.0)

    model_config = {"frozen": True}


class PipelineStage(str, Enum):
    INPUT_PROCESSING = "input_processing"
    FEATURE_EXTRACTION = "feature_extraction"
    QUERY_GENERATION = "query_generation"
    RETRIEVAL = "retrieval"
    EVIDENCE_SYNTHESIS = "evidence_synthesis"
    RESPONSE_GENERATION = "response_generation"


class PipelineError(Exception):
    def __init__(
        self,
        stage: PipelineStage,
        message: str,
        cause: Exception | None = None,
    ):
        self.stage = stage
        self.message = message
        self.cause = cause
        super().__init__(f"[{stage.value}] {message}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "message": self.message,
            "cause": str(self.cause) if self.cause else None,
        }
