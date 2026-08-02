from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, status

from api.dependencies import (
    ArchiveStoreDep,
    ClinicalGraphDep,
    RequestIdDep,
    WorkflowStoreDep,
)
from api.schemas.models import (
    ClinicalAnalyzeRequest,
    ClinicalAnalyzeResponse,
    FrameworkSelectRequest,
)
from app_logging.logger import get_logger

_log = get_logger(__name__)

router = APIRouter(prefix="/api/v1/clinical", tags=["Clinical"])


def _detect_input_type(text: str) -> str:
    text_lower = text.strip()
    if not text_lower:
        return "case_study"
    lines = [l.strip() for l in text_lower.split("\n") if l.strip()]
    bullet_keys = sum(
        1 for l in lines
        if l.startswith("-") or l.startswith("*") or (l and l[0].isdigit() and ". " in l[:5])
    )
    if bullet_keys >= 3 and all(len(l) < 200 for l in lines):
        return "symptom_list"
    if len(text_lower) < 200:
        return "single_statement"
    return "case_study"


def _model_dump(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return {}


def _to_response(
    state: dict[str, Any],
    *,
    input_type: str,
    elapsed_ms: float,
    status_: str,
    workflow_id: str | None = None,
) -> ClinicalAnalyzeResponse:
    """Convert a graph state into the public API response model."""
    framework_results = state.get("framework_results") or {}
    return ClinicalAnalyzeResponse(
        input_type=input_type,
        status=status_,
        workflow_id=workflow_id,
        understanding=_model_dump(state.get("understanding")),
        queries=_model_dump(state.get("queries_result")).get("queries", []),
        evidence=_model_dump(state.get("evidence")),
        framework_analyses=[_model_dump(r) for r in framework_results.values()],
        framework_comparison=_model_dump(state.get("framework_comparison")),
        formulation=_model_dump(state.get("formulation")),
        missing_info=_model_dump(state.get("missing_info")),
        therapeutic_planning=_model_dump(state.get("plan")),
        session_planning=_model_dump(state.get("session_plan")),
        response=_model_dump(state.get("response")),
        safety_report=_model_dump(state.get("safety_report")),
        reported_diagnosis=state.get("reported_diagnosis"),
        patient_metadata=state.get("patient_metadata"),
        errors=state.get("errors", {}),
        elapsed_ms=round(elapsed_ms, 2),
    )


@router.post(
    "/analyze",
    response_model=ClinicalAnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Full clinical case analysis pipeline (LangGraph)",
    description=(
        "Runs the 8-stage LangGraph clinical analysis pipeline: case understanding, "
        "query generation, evidence retrieval, evidence synthesis, clinical formulation, "
        "missing info detection, therapeutic planning, and response generation. Supports "
        "single client statements, symptom lists, and full case studies. When "
        "``resumable`` is true, execution pauses after framework comparison and returns "
        "``status=awaiting_selection`` plus a ``workflow_id`` for POST /framework/select."
    ),
    responses={
        200: {"description": "Clinical analysis complete"},
        400: {"description": "Invalid input"},
        422: {"description": "Validation error"},
        503: {"description": "Pipeline dependency unavailable"},
    },
)
async def clinical_analyze(
    body: ClinicalAnalyzeRequest,
    run_pipeline: ClinicalGraphDep,
    request_id: RequestIdDep,
    workflow_store: WorkflowStoreDep,
) -> ClinicalAnalyzeResponse:
    t_total = time.perf_counter()

    input_type = body.input_type or _detect_input_type(body.text)

    _log.info(
        "clinical.analyze.request",
        request_id=request_id,
        text_length=len(body.text),
        input_type=input_type,
        resumable=body.resumable,
    )

    workflow_id: str | None = None
    try:
        if body.resumable:
            workflow_id = uuid.uuid4().hex
            state: dict[str, Any] = await run_pipeline(
                body.text,
                workflow_id=workflow_id,
                reported_diagnosis=body.reported_diagnosis,
                patient_metadata=body.patient_metadata,
            )
        else:
            state = await run_pipeline(
                body.text,
                reported_diagnosis=body.reported_diagnosis,
                patient_metadata=body.patient_metadata,
            )
    except Exception as exc:
        _log.error(
            "clinical.analyze.graph_error",
            request_id=request_id,
            error=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Clinical analysis failed: {exc}",
        ) from exc

    elapsed_ms = (time.perf_counter() - t_total) * 1000

    awaiting_selection = bool(state.get("awaiting_selection"))
    if awaiting_selection and workflow_id:
        workflow_store.save(workflow_id, dict(state))

    # Convert state to response, handling Optional model outputs
    errors = state.get("errors", {})

    _log.info(
        "clinical.analyze.complete",
        request_id=request_id,
        input_type=input_type,
        error_count=len(errors),
        elapsed_ms=round(elapsed_ms, 2),
        status="awaiting_selection" if awaiting_selection else "completed",
    )

    return _to_response(
        state,
        input_type=input_type,
        elapsed_ms=elapsed_ms,
        status_="awaiting_selection" if awaiting_selection else "completed",
        workflow_id=workflow_id if awaiting_selection else None,
    )


@router.post(
    "/framework/select",
    response_model=ClinicalAnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Resume a paused workflow with a selected framework",
    description=(
        "Resumes a workflow paused after framework comparison. The psychologist "
        "chooses one framework; downstream stages then continue using only that "
        "framework. The pending workflow is deleted on success."
    ),
    responses={
        200: {"description": "Workflow resumed and completed"},
        404: {"description": "Workflow not found or expired"},
        422: {"description": "Selected framework not in the workflow's enabled set"},
        503: {"description": "Pipeline dependency unavailable"},
    },
)
async def framework_select(
    body: FrameworkSelectRequest,
    run_pipeline: ClinicalGraphDep,
    request_id: RequestIdDep,
    workflow_store: WorkflowStoreDep,
    archive_store: ArchiveStoreDep,
) -> ClinicalAnalyzeResponse:
    t_total = time.perf_counter()

    # Atomically claim the pending workflow so only one caller can resume
    # it — a concurrent second resume attempt gets a 404.
    pending = workflow_store.claim(body.workflow_id)
    if pending is None:
        _log.warning(
            "clinical.framework_select.not_found",
            request_id=request_id,
            workflow_id=body.workflow_id,
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found or expired.",
        )

    enabled = pending.get("enabled_frameworks") or list(
        (pending.get("framework_results") or {}).keys()
    )
    if body.selected_framework not in enabled:
        _log.warning(
            "clinical.framework_select.invalid_framework",
            request_id=request_id,
            workflow_id=body.workflow_id,
            selected_framework=body.selected_framework,
            enabled=enabled,
        )
        # Restore the pending workflow so the psychologist can retry with
        # a valid framework.
        workflow_store.save(body.workflow_id, pending)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"selected_framework '{body.selected_framework}' is not in "
                f"this workflow's enabled frameworks: {enabled}"
            ),
        )

    pending["selected_framework"] = body.selected_framework
    pending["awaiting_selection"] = False

    try:
        state = await run_pipeline(initial_state=pending)
    except Exception as exc:
        _log.error(
            "clinical.framework_select.graph_error",
            request_id=request_id,
            workflow_id=body.workflow_id,
            error=str(exc),
        )
        # Restore the pending workflow so the psychologist can retry.
        workflow_store.save(body.workflow_id, pending)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Workflow resume failed: {exc}",
        ) from exc

    # Archive the completed workflow for future sessions ("future memory").
    try:
        archive_store.archive(body.workflow_id, body.selected_framework, state)
    except Exception as exc:
        _log.error(
            "clinical.framework_select.archive_failed",
            request_id=request_id,
            workflow_id=body.workflow_id,
            error=str(exc),
        )

    elapsed_ms = (time.perf_counter() - t_total) * 1000
    _log.info(
        "clinical.framework_select.complete",
        request_id=request_id,
        workflow_id=body.workflow_id,
        selected_framework=body.selected_framework,
        error_count=len(state.get("errors", {})),
        elapsed_ms=round(elapsed_ms, 2),
    )

    return _to_response(
        state,
        input_type="case_study",
        elapsed_ms=elapsed_ms,
        status_="completed",
        workflow_id=body.workflow_id,
    )
