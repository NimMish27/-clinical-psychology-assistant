"""
clinical/framework/models.py — Data models for framework metadata, analysis lifecycle,
and structured output types.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RetrievalStrategy(str, Enum):
    """How a framework retrieves evidence from its literature collection."""

    HYBRID = "hybrid"
    SEMANTIC_ONLY = "semantic_only"
    KEYWORD_ONLY = "keyword_only"
    NONE = "none"


class InterventionStyle(BaseModel):
    """Describes the intervention approach for a therapeutic framework."""

    primary_focus: str = Field(
        description="The central therapeutic mechanism (e.g. 'acceptance and mindfulness', 'cognitive restructuring')",
    )
    techniques: list[str] = Field(
        description="Common techniques used in this framework",
    )
    session_structure: str = Field(
        description="Typical session format or structure",
    )
    therapeutic_stance: str = Field(
        description="The therapist's role and attitude",
    )


class PromptTemplates(BaseModel):
    """
    Maps lifecycle methods to YAML template filenames within the
    plugin's ``prompts/`` directory.

    Each field defaults to ``<method_name>.yaml``.  Plugins may
    override individual fields to point to custom filenames.
    """

    analyze_case: str = Field(
        default="analyze_case.yaml",
        description="Template filename for ClinicalFramework.analyze_case()",
    )
    retrieve_evidence: str = Field(
        default="retrieve_evidence.yaml",
        description="Template filename for ClinicalFramework.retrieve_evidence()",
    )
    case_conceptualization: str = Field(
        default="case_conceptualization.yaml",
        description="Template filename for ClinicalFramework.generate_case_conceptualization()",
    )
    therapeutic_focus: str = Field(
        default="therapeutic_focus.yaml",
        description="Template filename for ClinicalFramework.generate_therapeutic_focus()",
    )
    intervention_plan: str = Field(
        default="intervention_plan.yaml",
        description="Template filename for ClinicalFramework.generate_intervention_plan()",
    )
    session_objectives: str = Field(
        default="session_objectives.yaml",
        description="Template filename for ClinicalFramework.generate_session_objectives()",
    )
    followup_questions: str = Field(
        default="followup_questions.yaml",
        description="Template filename for ClinicalFramework.generate_followup_questions()",
    )
    limitations: str = Field(
        default="limitations.yaml",
        description="Template filename for ClinicalFramework.generate_limitations()",
    )

    def filename_for(self, method: str) -> str:
        """Return the configured filename for a lifecycle method name."""
        return getattr(self, method, f"{method}.yaml")


class FrameworkMetadata(BaseModel):
    """
    Immutable metadata describing a therapeutic framework plugin.

    This is the canonical identity of a framework — used by the registry,
    the API, and the LangGraph orchestrator to understand what a plugin
    provides and how to invoke it.
    """

    id: str = Field(
        min_length=2,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Unique slug identifier (e.g. 'act', 'cbt', 'dbt')",
    )
    display_name: str = Field(
        min_length=2,
        max_length=128,
        description="Human-readable name (e.g. 'Acceptance and Commitment Therapy')",
    )
    version: str = Field(
        default="1.0.0",
        pattern=r"^\d+\.\d+\.\d+$",
        description="Semantic version of the plugin",
    )
    description: str = Field(
        min_length=10,
        max_length=2000,
        description="One-paragraph summary of the framework",
    )
    supported_use_cases: list[str] = Field(
        min_length=1,
        description="Clinical scenarios this framework is suited for",
    )
    literature_collection: str | None = Field(
        default=None,
        description="ChromaDB collection name for framework-specific literature; None if no dedicated collection",
    )
    retrieval_strategy: RetrievalStrategy = Field(
        default=RetrievalStrategy.HYBRID,
        description="How evidence is retrieved from the literature collection",
    )
    prompt_templates: PromptTemplates = Field(
        default_factory=PromptTemplates,
        description="Paths to prompt template files within the plugin package",
    )
    intervention_style: InterventionStyle = Field(
        description="Description of the intervention approach",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Arbitrary tags for filtering and categorization",
    )
    citations: list[str] = Field(
        default_factory=list,
        description="Seminal references for this framework",
    )

    class Config:
        frozen = True


class FrameworkAnalysisResult(BaseModel):
    """Output of running a framework plugin's analysis on a clinical case.

    Returned by ``FrameworkPlugin.analyze()`` and collected in
    ``GraphState.framework_results``.
    """

    slug: str = Field(description="Framework slug identifier")
    display_name: str = Field(description="Human-readable framework name")
    guidance: str = Field(
        default="",
        description="Framework-specific clinical guidance / narrative output",
    )
    applicable: bool = Field(
        default=True,
        description="Whether the framework was applicable to the case",
    )
    error: str | None = Field(
        default=None,
        description="Error message if analysis failed",
    )


# ═══════════════════════════════════════════════════════════════
# Framework lifecycle models
# ═══════════════════════════════════════════════════════════════

class FrameworkCaseInput(BaseModel):
    """Input payload for every lifecycle method on ClinicalFramework.

    Carries all clinical data a framework needs to produce its output.
    Fields are intentionally broad so each method can select what it
    requires without being coupled to the full pipeline state.
    """

    case_understanding: dict = Field(
        default_factory=dict,
        description="Flat dict from CaseUnderstandingResult.to_flat_dict()",
    )
    formulation: dict | None = Field(
        default=None,
        description="Optional ClinicalFormulationResult.model_dump()",
    )
    evidence_summary: str | None = Field(
        default=None,
        description="Optional evidence synthesis text (overall_summary)",
    )
    raw_text: str = Field(
        default="",
        description="Original client narrative text",
    )
    session_id: str | None = Field(
        default=None,
        description="Optional session identifier for context continuity",
    )

    model_config = {"frozen": True}


class CaseAnalysis(BaseModel):
    """Output of ClinicalFramework.analyze_case()."""

    key_features: list[str] = Field(
        description="Key clinical features identified through the framework lens",
    )
    presenting_patterns: list[str] = Field(
        description="Patterns in the presentation highlighted by this framework",
    )
    framework_observations: list[str] = Field(
        description="Framework-specific observations not captured by generic assessment",
    )
    severity_indicators: list[str] = Field(
        description="Indicators that inform severity from this framework's perspective",
    )
    strengths_and_resources: list[str] = Field(
        description="Client strengths and resources the framework leverages",
    )
    analysis_summary: str = Field(
        min_length=10,
        description="Narrative summary of the case analysis",
    )


class RetrievedEvidence(BaseModel):
    """A single evidence item retrieved by retrieve_evidence()."""

    source: str = Field(description="Source citation or document name")
    content: str = Field(
        min_length=10,
        description="Summary of the evidence content",
    )
    relevance: str = Field(
        min_length=10,
        description="How this evidence applies to the current case",
    )
    quality_rating: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional methodological quality rating (0-1)",
    )


class EvidenceResult(BaseModel):
    """Output of ClinicalFramework.retrieve_evidence()."""

    retrieved: list[RetrievedEvidence] = Field(
        description="Individual evidence items retrieved",
    )
    synthesis: str = Field(
        min_length=10,
        description="Integrated synthesis of the evidence base",
    )
    key_findings: list[str] = Field(
        description="Principal findings from the evidence",
    )
    evidence_gaps: list[str] = Field(
        description="Areas where evidence is lacking or inconclusive",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Overall confidence in the evidence base (0-1)",
    )


class CaseConceptualization(BaseModel):
    """Output of ClinicalFramework.generate_case_conceptualization()."""

    framework_formulation: str = Field(
        min_length=20,
        description="Narrative formulation from the framework's theoretical perspective",
    )
    maintaining_factors: list[str] = Field(
        description="Factors that maintain the client's difficulties per the framework",
    )
    core_mechanisms: list[str] = Field(
        description="Core psychological mechanisms implicated by the framework",
    )
    developmental_factors: list[str] = Field(
        description="Developmental and historical factors relevant to the framework",
    )
    conceptualization_summary: str = Field(
        min_length=10,
        description="Brief summary of the conceptualization",
    )


class TherapeuticFocusResult(BaseModel):
    """Output of ClinicalFramework.generate_therapeutic_focus()."""

    primary_focus_areas: list[str] = Field(
        min_length=1,
        description="Primary targets for therapeutic work",
    )
    secondary_focus_areas: list[str] = Field(
        description="Secondary or downstream focus areas",
    )
    staging_and_sequencing: str = Field(
        min_length=10,
        description="Recommended order and timing of focus areas",
    )
    focus_rationale: str = Field(
        min_length=10,
        description="Clinical rationale for the chosen focus areas",
    )


class InterventionPlanResult(BaseModel):
    """Output of ClinicalFramework.generate_intervention_plan()."""

    interventions: list[dict[str, str]] = Field(
        description="Specific interventions with technique and rationale",
    )
    provisional_strategy: str = Field(
        min_length=20,
        description="Overall intervention strategy narrative",
    )
    estimated_duration: str | None = Field(
        default=None,
        description="Estimated number of sessions or timeframe",
    )
    expected_outcomes: list[str] = Field(
        description="Expected therapeutic outcomes",
    )


class SessionObjectivesResult(BaseModel):
    """Output of ClinicalFramework.generate_session_objectives()."""

    immediate_objectives: list[str] = Field(
        description="Objectives for the upcoming session",
    )
    short_term_goals: list[str] = Field(
        min_length=1,
        description="Goals for the next 3-6 sessions",
    )
    long_term_goals: list[str] = Field(
        description="Goals for the overall course of therapy",
    )
    progress_indicators: list[str] = Field(
        description="Observable indicators that progress is being made",
    )


class FollowupQuestionsResult(BaseModel):
    """Output of ClinicalFramework.generate_followup_questions()."""

    assessment_questions: list[str] = Field(
        description="Questions to gather further assessment information",
    )
    therapeutic_probes: list[str] = Field(
        description="Therapeutic probing questions to deepen insight",
    )
    outcome_monitoring: list[str] = Field(
        description="Questions to monitor treatment progress and outcomes",
    )
    engagement_questions: list[str] = Field(
        description="Questions to strengthen therapeutic engagement and alliance",
    )


class LimitationsResult(BaseModel):
    """Output of ClinicalFramework.generate_limitations()."""

    contraindications: list[str] = Field(
        description="Clinical presentations where this framework is contraindicated",
    )
    precautions: list[str] = Field(
        description="Precautions to observe when applying this framework",
    )
    alternative_frameworks: list[str] = Field(
        description="Alternative frameworks that may be more appropriate",
    )
    caveats: list[str] = Field(
        description="General caveats and limitations of this framework",
    )


class FrameworkOutput(BaseModel):
    """Complete aggregated output of a ClinicalFramework's full analysis pipeline.

    Returned by ``ClinicalFramework.run_full_analysis()``.  Each field is
    optional so partial results are preserved even if an intermediate
    lifecycle method fails.
    """

    metadata: FrameworkMetadata = Field(description="Framework identifying metadata")
    case_analysis: CaseAnalysis | None = Field(
        default=None,
        description="Output of analyze_case()",
    )
    evidence: EvidenceResult | None = Field(
        default=None,
        description="Output of retrieve_evidence()",
    )
    conceptualization: CaseConceptualization | None = Field(
        default=None,
        description="Output of generate_case_conceptualization()",
    )
    therapeutic_focus: TherapeuticFocusResult | None = Field(
        default=None,
        description="Output of generate_therapeutic_focus()",
    )
    intervention_plan: InterventionPlanResult | None = Field(
        default=None,
        description="Output of generate_intervention_plan()",
    )
    session_objectives: SessionObjectivesResult | None = Field(
        default=None,
        description="Output of generate_session_objectives()",
    )
    followup_questions: FollowupQuestionsResult | None = Field(
        default=None,
        description="Output of generate_followup_questions()",
    )
    limitations: LimitationsResult | None = Field(
        default=None,
        description="Output of generate_limitations()",
    )
    processing_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Total wall-clock time for the full analysis pipeline",
    )
    error: str | None = Field(
        default=None,
        description="Top-level error message if the pipeline failed",
    )
