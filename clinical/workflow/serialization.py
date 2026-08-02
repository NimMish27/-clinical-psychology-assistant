from __future__ import annotations

import dataclasses
import json
from typing import Any

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

# ── Model field registry ──────────────────────────────────────
# Maps GraphState keys carrying a single Pydantic model to their class.
# ``model_dump(mode="json")`` round-trips via ``model_validate`` so
# datetimes / nested models survive JSON persistence.

_MODEL_FIELDS: dict[str, type] = {
    "understanding": CaseUnderstandingResult,
    "queries_result": QueryGenerationResult,
    "evidence": EvidenceSynthesisResult,
    "framework_comparison": FrameworkComparisonResult,
    "formulation": ClinicalFormulationResult,
    "missing_info": MissingInfoResult,
    "plan": TherapeuticPlanResult,
    "session_plan": SessionPlanResult,
    "response": ClinicalResponseResult,
    "safety_report": SafetyValidationResult,
}


def state_to_jsonable(state: dict[str, Any]) -> dict[str, Any]:
    """Convert a GraphState dict into a JSON-serialisable structure."""
    out: dict[str, Any] = {}
    for key, value in state.items():
        if value is None:
            out[key] = None
        elif key in _MODEL_FIELDS:
            out[key] = value.model_dump(mode="json")
        elif key == "framework_results":
            out[key] = {
                name: res.model_dump(mode="json")
                for name, res in value.items()
            }
        elif key == "retrieved_chunks":
            out[key] = [dataclasses.asdict(c) for c in value]
        else:
            out[key] = value
    return out


def state_from_jsonable(payload: dict[str, Any]) -> dict[str, Any]:
    """Rebuild a GraphState dict from a JSON-serialisable structure."""
    state: dict[str, Any] = {}
    for key, value in payload.items():
        if value is None:
            state[key] = None
        elif key in _MODEL_FIELDS:
            state[key] = _MODEL_FIELDS[key].model_validate(value)
        elif key == "framework_results":
            state[key] = {
                name: FrameworkResult.model_validate(res)
                for name, res in value.items()
            }
        elif key == "retrieved_chunks":
            state[key] = [RetrievedChunk(**c) for c in value]
        else:
            state[key] = value
    return state


def state_to_json(state: dict[str, Any]) -> str:
    """Serialise a GraphState dict to a compact JSON string."""
    return json.dumps(
        state_to_jsonable(state),
        sort_keys=True,
        separators=(",", ":"),
    )


def state_from_json(data: str) -> dict[str, Any]:
    """Deserialise a GraphState dict from a JSON string."""
    return state_from_jsonable(json.loads(data))
