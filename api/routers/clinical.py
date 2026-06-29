from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, status

from api.dependencies import (
    ClinicalPipelineDep,
    RequestIdDep,
)
from api.schemas.models import ClinicalAnalyzeRequest, ClinicalAnalyzeResponse
from clinical.models import ClinicalInput, PipelineError
from app_logging.logger import get_logger

_log = get_logger(__name__)

router = APIRouter(prefix="/api/v1/clinical", tags=["Clinical"])


@router.post(
    "/analyze",
    response_model=ClinicalAnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Full clinical case analysis pipeline",
    description=(
        "Runs the 7-stage clinical analysis pipeline: input classification, "
        "feature extraction, retrieval query generation, evidence retrieval, "
        "evidence synthesis, and clinical response generation. Supports single "
        "client statements, symptom lists, and full case studies."
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
    pipeline: ClinicalPipelineDep,
    request_id: RequestIdDep,
) -> ClinicalAnalyzeResponse:
    t_total = time.perf_counter()

    _log.info(
        "clinical.analyze.request",
        request_id=request_id,
        text_length=len(body.text),
        input_type=body.input_type,
    )

    try:
        result = await pipeline.run(
            ClinicalInput(
                raw_text=body.text,
                input_type=body.input_type,
            )
        )
    except PipelineError as exc:
        _log.error(
            "clinical.analyze.pipeline_error",
            request_id=request_id,
            stage=exc.stage.value,
            error=exc.message,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Pipeline failed at {exc.stage.value}: {exc.message}",
        ) from exc
    except Exception as exc:
        _log.error(
            "clinical.analyze.unhandled_error",
            request_id=request_id,
            error=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Clinical analysis pipeline encountered an unexpected error.",
        ) from exc

    elapsed_ms = (time.perf_counter() - t_total) * 1000
    _log.info(
        "clinical.analyze.complete",
        request_id=request_id,
        input_type=result.input_type.value,
        query_count=len(result.queries),
        elapsed_ms=round(elapsed_ms, 2),
    )

    return ClinicalAnalyzeResponse(
        input_type=result.input_type.value,
        understanding=result.understanding.model_dump(),
        features=result.features.model_dump(),
        queries=[q.model_dump() for q in result.queries],
        evidence=result.evidence.model_dump(),
        response=result.response.model_dump(),
        elapsed_ms=result.elapsed_ms,
    )
