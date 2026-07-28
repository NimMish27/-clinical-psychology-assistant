"""
clinical/framework/__init__.py — Framework plugin system root.
"""
from clinical.framework.registry import FrameworkRegistry
from clinical.framework.base import ClinicalFramework, FrameworkPlugin
from clinical.framework.models import (
    CaseAnalysis,
    CaseConceptualization,
    EvidenceResult,
    FollowupQuestionsResult,
    FrameworkAnalysisResult,
    FrameworkCaseInput,
    FrameworkMetadata,
    FrameworkOutput,
    InterventionPlanResult,
    InterventionStyle,
    LimitationsResult,
    PromptTemplates,
    RetrievedEvidence,
    RetrievalStrategy,
    SessionObjectivesResult,
    TherapeuticFocusResult,
)

# Prompt management exports
from clinical.framework.prompts.manager import PromptManager
from clinical.framework.prompts.models import PromptTemplate, PromptCollection, PromptVariable
from clinical.framework.prompts.renderer import render_template, render_prompt_template
from clinical.framework.prompts.validator import (
    validate_yaml_structure,
    validate_template_syntax,
    validate_prompt_template,
    validate_variable_coverage,
    TemplateValidationError,
)

__all__ = [
    "FrameworkRegistry",
    "FrameworkPlugin",
    "ClinicalFramework",
    "FrameworkMetadata",
    "FrameworkAnalysisResult",
    "FrameworkCaseInput",
    "FrameworkOutput",
    "CaseAnalysis",
    "EvidenceResult",
    "RetrievedEvidence",
    "CaseConceptualization",
    "TherapeuticFocusResult",
    "InterventionPlanResult",
    "SessionObjectivesResult",
    "FollowupQuestionsResult",
    "LimitationsResult",
    "RetrievalStrategy",
    "InterventionStyle",
    "PromptTemplates",
    # Prompt management
    "PromptManager",
    "PromptTemplate",
    "PromptCollection",
    "PromptVariable",
    "render_template",
    "render_prompt_template",
    "validate_yaml_structure",
    "validate_template_syntax",
    "validate_prompt_template",
    "validate_variable_coverage",
    "TemplateValidationError",
]
