from __future__ import annotations

from typing import Annotated, Any, Optional

from typing_extensions import TypedDict

from clinical.case_understanding.models import CaseUnderstandingResult
from clinical.evidence_synthesis.models import EvidenceSynthesisResult
from clinical.formulation.models import ClinicalFormulationResult
from clinical.frameworks.models import FrameworkComparisonResult, FrameworkResult
from clinical.missing_info.models import MissingInfoResult
from clinical.query_generation.models import QueryGenerationResult
from clinical.response_generation.models import ClinicalResponseResult
from clinical.safety_validation.models import SafetyValidationResult
from clinical.session_planning.models import SessionPlanResult
from clinical.therapeutic_planning.models import TherapeuticPlanResult
from rag.retriever import RetrievedChunk


def _merge_errors(a: dict[str, str], b: dict[str, str]) -> dict[str, str]:
    """Accumulate errors across nodes — later entries do not overwrite earlier."""
    merged = dict(a)
    merged.update(b)
    return merged


class GraphState(TypedDict, total=False):
    """State flowing through the LangGraph clinical pipeline.

    Each key is populated by the corresponding node as it executes.
    Fields without a reducer annotation are **replaced** each time a
    node returns them.  ``errors`` uses a custom reducer that
    accumulates entries across nodes so no error is silently lost.
    """

    # ── Input ────────────────────────────────────────────────
    text: str
    session_id: Optional[str]
    enabled_frameworks: Optional[list[str]]
    selected_framework: Optional[str]
    workflow_id: Optional[str]
    awaiting_selection: Optional[bool]
    reported_diagnosis: Optional[str]
    patient_metadata: Optional[dict[str, Any]]

    # ── Module outputs (set by nodes, consumed by downstream) ─
    understanding: Optional[CaseUnderstandingResult]
    queries_result: Optional[QueryGenerationResult]
    retrieved_chunks: Optional[list[RetrievedChunk]]
    evidence: Optional[EvidenceSynthesisResult]
    framework_results: Optional[dict[str, FrameworkResult]]
    framework_comparison: Optional[FrameworkComparisonResult]
    formulation: Optional[ClinicalFormulationResult]
    missing_info: Optional[MissingInfoResult]
    plan: Optional[TherapeuticPlanResult]
    session_plan: Optional[SessionPlanResult]
    response: Optional[ClinicalResponseResult]
    safety_report: Optional[SafetyValidationResult]

    # ── Runtime metadata ─────────────────────────────────────
    errors: Annotated[dict[str, str], _merge_errors]
