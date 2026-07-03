from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, HTTPException, status

from api.dependencies import (
    ClinicalGraphDep,
    RequestIdDep,
)
from api.schemas.models import ClinicalAnalyzeRequest, ClinicalAnalyzeResponse
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


@router.post(
    "/analyze",
    response_model=ClinicalAnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Full clinical case analysis pipeline (LangGraph)",
    description=(
        "Runs the 8-stage LangGraph clinical analysis pipeline: case understanding, "
        "query generation, evidence retrieval, evidence synthesis, clinical formulation, "
        "missing info detection, therapeutic planning, and response generation. Supports "
        "single client statements, symptom lists, and full case studies."
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
) -> ClinicalAnalyzeResponse:
    t_total = time.perf_counter()

    input_type = body.input_type or _detect_input_type(body.text)

    _log.info(
        "clinical.analyze.request",
        request_id=request_id,
        text_length=len(body.text),
        input_type=input_type,
    )

    try:
        state: dict[str, Any] = await run_pipeline(body.text)
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

    # Convert state to response, handling Optional model outputs
    errors = state.get("errors", {})

    _log.info(
        "clinical.analyze.complete",
        request_id=request_id,
        input_type=input_type,
        error_count=len(errors),
        elapsed_ms=round(elapsed_ms, 2),
    )

    return ClinicalAnalyzeResponse(
        input_type=input_type,
        understanding=_model_dump(state.get("understanding")),
        queries=_model_dump(state.get("queries_result")).get("queries", []),
        evidence=_model_dump(state.get("evidence")),
        formulation=_model_dump(state.get("formulation")),
        missing_info=_model_dump(state.get("missing_info")),
        therapeutic_planning=_model_dump(state.get("plan")),
        response=_model_dump(state.get("response")),
        errors=errors,
        elapsed_ms=round(elapsed_ms, 2),
    )
