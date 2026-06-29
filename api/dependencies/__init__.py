"""
api/dependencies/__init__.py
─────────────────────────────
FastAPI dependency injection providers.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from app_logging.logger import get_logger

_log = get_logger(__name__)


@lru_cache(maxsize=1)
def _cached_settings():
    from config.settings import get_settings
    return get_settings()


def get_settings_dep():
    """Provide the application Settings singleton."""
    return _cached_settings()


SettingsDep = Annotated[object, Depends(get_settings_dep)]


def get_retriever_dep():
    """Provide the process-wide Retriever singleton."""
    try:
        from rag.retriever import get_retriever
        return get_retriever()
    except Exception as exc:
        _log.error("deps.retriever_unavailable", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retriever service is unavailable.",
        ) from exc


RetrieverDep = Annotated[object, Depends(get_retriever_dep)]


def get_vector_store_dep():
    """Provide the process-wide VectorStore singleton."""
    try:
        from rag.vector_store import get_vector_store
        return get_vector_store()
    except Exception as exc:
        _log.error("deps.vector_store_unavailable", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store (ChromaDB) is unavailable.",
        ) from exc


VectorStoreDep = Annotated[object, Depends(get_vector_store_dep)]


def get_pdf_loader_dep():
    """Provide a stateless PDFLoader instance."""
    try:
        from ingestion.loaders.pdf_loader import PDFLoader
        return PDFLoader()
    except Exception as exc:
        _log.error("deps.pdf_loader_unavailable", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PDF loader unavailable.",
        ) from exc


PDFLoaderDep = Annotated[object, Depends(get_pdf_loader_dep)]


def get_chunker_dep():
    """Provide a Chunker instance configured from settings."""
    try:
        settings = _cached_settings()
        from ingestion.processors.chunker import Chunker
        return Chunker(
            chunk_size=settings.rag.chunk_size,
            chunk_overlap=settings.rag.chunk_overlap,
        )
    except Exception as exc:
        _log.error("deps.chunker_unavailable", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chunker unavailable.",
        ) from exc


ChunkerDep = Annotated[object, Depends(get_chunker_dep)]


def get_request_id(request: Request) -> str:
    """Extract the request ID injected by RequestLoggingMiddleware."""
    return getattr(request.state, "request_id", "unknown")


RequestIdDep = Annotated[str, Depends(get_request_id)]


def get_clinical_pipeline_dep():
    """Provide the process-wide ClinicalPipeline singleton."""
    try:
        from clinical.pipeline import ClinicalPipeline
        pipeline = ClinicalPipeline(
            retriever=get_retriever_dep(),
        )
        return pipeline
    except Exception as exc:
        _log.error("deps.clinical_pipeline_unavailable", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Clinical analysis pipeline is unavailable.",
        ) from exc


ClinicalPipelineDep = Annotated[object, Depends(get_clinical_pipeline_dep)]
