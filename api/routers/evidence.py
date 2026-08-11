"""
api/routers/evidence.py
───────────────────────
POST /evidence/search — retrieval-only literature search for the
Evidence tabs and Evidence Explorer.

Unlike ``/chat`` this endpoint never calls an LLM. It returns structured
evidence items (plus the framework evidence bundle — synthesis, key
findings, gaps, confidence — when a curated framework is selected) so the
frontend can render evidence cards, confidence rings, and quality meters.

Source of truth for retrieval stays in the retrievers:
    framework path  → ``clinical.retrieval`` (CBT / ACT / Humanistic)
    general path    → ``rag.retriever.Retriever`` (clinical knowledge base)
    user documents  → ``rag.vector_store.VectorStore`` on ``user_documents``
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, status

from api.dependencies import RequestIdDep, RetrieverDep, SettingsDep
from api.schemas.models import (
    EvidenceItem,
    EvidenceSearchRequest,
    EvidenceSearchResponse,
)
from app_logging.logger import get_logger

if TYPE_CHECKING:
    from clinical.retrieval.act_retriever import ACTEvidenceFilter, ACTEvidenceResult
    from clinical.retrieval.models import (
        CBTEvidenceFilter,
        CBTEvidenceResult,
        HumanisticEvidenceFilter,
        HumanisticEvidenceResult,
    )
    from rag.retriever import RetrievalResult, RetrievedChunk

_log = get_logger(__name__)

router = APIRouter(prefix="/evidence", tags=["Evidence"])


def _build_framework_filter(
    framework: str,
    body: EvidenceSearchRequest,
) -> CBTEvidenceFilter | ACTEvidenceFilter | HumanisticEvidenceFilter:
    """Construct the typed metadata filter for a curated framework."""
    from clinical.retrieval.act_retriever import ACTEvidenceFilter
    from clinical.retrieval.models import CBTEvidenceFilter, HumanisticEvidenceFilter

    cls = {
        "cbt": CBTEvidenceFilter,
        "act": ACTEvidenceFilter,
        "humanistic": HumanisticEvidenceFilter,
    }[framework]

    return cls(
        framework=framework,
        chapter=body.chapter,
        topic=body.topic,
        intervention=body.intervention,
        symptom=body.symptom,
        population=body.population,
    )


def _build_generic_metadata_filter(body: EvidenceSearchRequest) -> dict[str, str] | None:
    """Flat metadata filter dict for the general retriever path."""
    filt: dict[str, str] = {}
    for field in ("chapter", "topic", "intervention", "symptom", "population"):
        value = getattr(body, field)
        if value:
            filt[field] = value
    return filt or None


def _framework_items_to_evidence(
    result: ACTEvidenceResult | CBTEvidenceResult | HumanisticEvidenceResult,
    *,
    is_user_document: bool = False,
) -> list[EvidenceItem]:
    """Normalize framework evidence items (CBT/ACT/Humanistic) to the API shape."""
    items = []
    for item in result.retrieved:
        items.append(
            EvidenceItem(
                source=item.source,
                content=item.content,
                quality_rating=item.quality_rating,
                framework=getattr(item, "framework", None),
                chapter=getattr(item, "chapter", None),
                topic=getattr(item, "topic", None),
                intervention=getattr(item, "intervention", None),
                symptom=getattr(item, "symptom", None),
                population=getattr(item, "population", None),
                page=item.page,
                chunk_id=item.chunk_id,
                is_user_document=is_user_document,
            )
        )
    return items


def _generic_items_to_evidence(
    retrieval_result: RetrievalResult,
) -> list[EvidenceItem]:
    """Convert generic retriever chunks to the API evidence shape."""
    items = []
    for chunk in retrieval_result.chunks:
        meta = chunk.metadata or {}
        items.append(
            EvidenceItem(
                source=chunk.source,
                content=chunk.text,
                quality_rating=chunk.score,
                framework=meta.get("framework"),
                chapter=meta.get("chapter"),
                topic=meta.get("topic"),
                intervention=meta.get("intervention"),
                symptom=meta.get("symptom"),
                population=meta.get("population"),
                page=chunk.page,
                chunk_id=chunk.chunk_id,
                is_user_document=False,
            )
        )
    return items


def _user_chunks_to_evidence(
    chunks: list[RetrievedChunk],
) -> list[EvidenceItem]:
    """Convert user-document chunks to the API evidence shape (tagged)."""
    items = []
    for chunk in chunks:
        items.append(
            EvidenceItem(
                source=chunk.source,
                content=chunk.text,
                quality_rating=chunk.score,
                framework=None,
                chapter=None,
                topic=None,
                intervention=None,
                symptom=None,
                population=None,
                page=chunk.page,
                chunk_id=chunk.chunk_id,
                is_user_document=True,
            )
        )
    return items


def _merge_user_items(
    items: list[EvidenceItem],
    user_items: list[EvidenceItem],
    n_results: int,
) -> list[EvidenceItem]:
    """Dedupe by chunk_id, re-rank by descending quality_rating, truncate."""
    seen: dict[str, EvidenceItem] = {}
    for item in list(items) + list(user_items):
        key = item.chunk_id or f"{item.source}:{item.page}"
        existing = seen.get(key)
        if existing is None or (item.quality_rating or 0.0) > (existing.quality_rating or 0.0):
            seen[key] = item

    sorted_items = sorted(
        seen.values(),
        key=lambda i: i.quality_rating or 0.0,
        reverse=True,
    )[:n_results]
    return sorted_items


@router.post(
    "/search",
    response_model=EvidenceSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Search clinical literature (retrieval-only)",
    description=(
        "Embeds the query and retrieves the most relevant evidence from a "
        "curated framework knowledge base (cbt/act/humanistic) or the general "
        "clinical knowledge base. Optionally merges and reranks results from "
        "the user's uploaded documents. Returns structured evidence items "
        "without any LLM generation."
    ),
    responses={
        200: {"description": "Evidence search completed"},
        400: {"description": "Invalid query"},
        503: {"description": "Retriever unavailable"},
    },
)
async def search_evidence(
    body: EvidenceSearchRequest,
    retriever: RetrieverDep,
    settings: SettingsDep,
    request_id: RequestIdDep,
) -> EvidenceSearchResponse:
    t_start = time.perf_counter()

    _log.info(
        "evidence.search",
        request_id=request_id,
        query_length=len(body.query),
        framework=body.framework,
        n_results=body.n_results,
        include_user_documents=body.include_user_documents,
        filters=[
            f
            for f in ("chapter", "topic", "intervention", "symptom", "population")
            if getattr(body, f)
        ],
    )

    try:
        if body.framework is not None:
            from api.routers.chat import _get_framework_retriever

            framework_retriever = _get_framework_retriever(body.framework)
            result = await framework_retriever.retrieve_async(
                body.query,
                filters=_build_framework_filter(body.framework, body),
            )
            items = _framework_items_to_evidence(result)
            synthesis: str | None = result.synthesis
            key_findings = list(result.key_findings)
            evidence_gaps = list(result.evidence_gaps)
            confidence: float | None = result.confidence
            total_candidates = result.total_candidates
            elapsed_ms = result.elapsed_ms
        else:
            retrieval_result = await retriever.aretrieve(
                body.query,
                n_results=body.n_results,
                metadata_filter=_build_generic_metadata_filter(body),
            )
            items = _generic_items_to_evidence(retrieval_result)
            synthesis = None
            key_findings = []
            evidence_gaps = []
            confidence = None
            total_candidates = retrieval_result.total_candidates
            elapsed_ms = retrieval_result.elapsed_ms
    except Exception as exc:
        _log.error("evidence.search_failed", request_id=request_id, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Evidence search failed: {exc}",
        ) from exc

    user_documents_returned = 0
    if body.include_user_documents:
        try:
            from api.routers.chat import _retrieve_user_documents

            user_chunks = await asyncio.to_thread(
                _retrieve_user_documents,
                body.query,
                settings,
                body.n_results,
            )
            if user_chunks:
                items = _merge_user_items(
                    items,
                    _user_chunks_to_evidence(user_chunks),
                    body.n_results,
                )
                user_documents_returned = sum(1 for i in items if i.is_user_document)
        except Exception as exc:
            _log.warning(
                "evidence.user_documents_failed",
                request_id=request_id,
                error=str(exc),
            )

    elapsed_total = (time.perf_counter() - t_start) * 1000

    _log.info(
        "evidence.complete",
        request_id=request_id,
        returned=len(items),
        user_documents_returned=user_documents_returned,
        total_candidates=total_candidates,
        elapsed_ms=round(elapsed_total, 2),
    )

    return EvidenceSearchResponse(
        query=body.query,
        framework=body.framework,
        sources=items,
        synthesis=synthesis,
        key_findings=key_findings,
        evidence_gaps=evidence_gaps,
        confidence=confidence,
        total_candidates=total_candidates,
        elapsed_ms=round(elapsed_ms, 2),
        user_documents_included=body.include_user_documents,
        user_documents_returned=user_documents_returned,
    )
