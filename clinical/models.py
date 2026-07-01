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
    key_findings: list[str] = Field(
        default_factory=list,
        description="Key findings from the evidence base",
    )
    common_themes: list[str] = Field(
        default_factory=list,
        description="Common themes across retrieved literature",
    )
    areas_of_agreement: list[str] = Field(
        default_factory=list,
        description="Areas of strong agreement in the evidence",
    )
    areas_of_uncertainty: list[str] = Field(
        default_factory=list,
        description="Areas where evidence is conflicting or uncertain",
    )
    practical_implications: list[str] = Field(
        default_factory=list,
        description="Practical implications for clinicians",
    )
    evidence_summary: str = Field(
        ...,
        description="Concise evidence summary without copying raw chunks",
    )

    model_config = {"frozen": True}


class Formulation(BaseModel):
    explanation: str = Field(
        ...,
        description="Detailed explanation of this formulation",
    )
    supporting_symptoms: list[str] = Field(
        default_factory=list,
        description="Symptoms from the case that support this formulation",
    )
    confidence_level: str = Field(
        ...,
        description="Confidence level of this formulation (e.g. High, Moderate, Low)",
    )

    model_config = {"frozen": True}


class ClinicalFormulation(BaseModel):
    case_summary: str = Field(
        ...,
        description="Comprehensive summary of the clinical case",
    )
    possible_formulations: list[Formulation] = Field(
        default_factory=list,
        description="List of possible clinical formulations",
    )
    supporting_evidence: list[str] = Field(
        default_factory=list,
        description="Evidence supporting the formulations",
    )
    alternative_explanations: list[str] = Field(
        default_factory=list,
        description="Alternative explanations or differential considerations without diagnosing",
    )
    missing_assessment_information: list[str] = Field(
        default_factory=list,
        description="Information missing that would help clarify the formulations",
    )

    model_config = {"frozen": True}


class PipelineResult(BaseModel):
    """Full result returned by the clinical pipeline orchestrator."""
    input_type: ClinicalInputType
    understanding: CaseUnderstanding
    features: ClinicalFeatures
    queries: list[RetrievalQuery]
    evidence: EvidenceSynthesis
    formulation: ClinicalFormulation
    elapsed_ms: float = Field(..., ge=0.0)

    model_config = {"frozen": True}


class PipelineStage(str, Enum):
    INPUT_PROCESSING = "input_processing"
    FEATURE_EXTRACTION = "feature_extraction"
    QUERY_GENERATION = "query_generation"
    RETRIEVAL = "retrieval"
    EVIDENCE_SYNTHESIS = "evidence_synthesis"
    FORMULATION_GENERATION = "formulation_generation"


class ClinicalResponse(BaseModel):
    """Final clinical response generated by the pipeline."""

    analysis: str = Field(
        ...,
        description="Comprehensive clinical analysis text",
    )
    formulation: str | None = Field(
        None,
        description="Clinical formulation if generated",
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Clinical recommendations",
    )
    evidence_summary: str = Field(
        ...,
        description="Summary of evidence used",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in the response",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Limitations of the analysis",
    )

    model_config = {"frozen": True}


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
