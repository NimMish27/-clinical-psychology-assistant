"""
api/routers/ingest.py
──────────────────────
POST /ingest — Upload and ingest PDF documents into ChromaDB.

Flow per file:
  UploadFile \u2192 temp disk write \u2192 PDFLoader \u2192 Chunker \u2192 embed_documents
             \u2192 VectorStore.add_documents \u2192 FileIngestResult

Design decisions:
  - Files are written to a temp directory and deleted after processing.
    No permanent copies are kept server-side (clinical confidentiality).
  - Each file is processed sequentially to control memory usage.
    Future: parallel processing with bounded semaphore.
  - Partial success is reported — one failing file does not abort others.
  - The entire pipeline runs in a thread-pool executor so the async
    event loop is never blocked by CPU/IO-heavy operations.
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from api.dependencies import (
    ChunkerDep,
    PDFLoaderDep,
    RequestIdDep,
    SettingsDep,
    VectorStoreDep,
)
from api.schemas.models import (
    FileIngestResult,
    IngestResponse,
    IngestStatus,
)
from app_logging.logger import audit_logger, get_logger

_log = get_logger(__name__)

router = APIRouter(prefix="/ingest", tags=["Ingestion"])

_MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024
_ALLOWED_CONTENT_TYPES = {"application/pdf", "application/octet-stream"}


def _ingest_one_file(
    file_path: Path,
    filename: str,
    loader,
    chunker,
    vector_store,
    settings,
) -> FileIngestResult:
    from rag.embeddings import embed_documents

    # ── Extract ───────────────────────────────────────────────────────────────
    try:
        extraction = loader.load(file_path)
    except Exception as exc:
        _log.error("ingest.extract_failed", filename=filename, error=str(exc))
        return FileIngestResult(
            filename=filename,
            status=IngestStatus.FAILED,
            error=f"PDF extraction failed: {exc}",
        )

    pages_extracted = extraction.extracted_pages

    if pages_extracted == 0:
        return FileIngestResult(
            filename=filename,
            status=IngestStatus.FAILED,
            pages_extracted=0,
            error="No text could be extracted from this PDF.",
        )

    # ── Chunk ─────────────────────────────────────────────────────────────────
    try:
        chunking = chunker.chunk_document(extraction)
    except Exception as exc:
        _log.error("ingest.chunk_failed", filename=filename, error=str(exc))
        return FileIngestResult(
            filename=filename,
            status=IngestStatus.FAILED,
            pages_extracted=pages_extracted,
            error=f"Chunking failed: {exc}",
        )

    chunks_created = chunking.total_chunks
    if chunks_created == 0:
        return FileIngestResult(
            filename=filename,
            status=IngestStatus.FAILED,
            pages_extracted=pages_extracted,
            chunks_created=0,
            error="No chunks produced — document may be image-only or too short.",
        )

    # ── Embed ─────────────────────────────────────────────────────────────────
    try:
        texts   = [c.text   for c in chunking.chunks]
        sources = [c.source for c in chunking.chunks]
        pages   = [c.page   for c in chunking.chunks]

        embedding_result = embed_documents(
            texts,
            sources=sources,
            pages=pages,
            batch_size=settings.embedding.batch_size,
        )
    except Exception as exc:
        _log.error("ingest.embed_failed", filename=filename, error=str(exc))
        return FileIngestResult(
            filename=filename,
            status=IngestStatus.FAILED,
            pages_extracted=pages_extracted,
            chunks_created=chunks_created,
            error=f"Embedding failed: {exc}",
        )

    chunks_embedded = embedding_result.total_embedded

    # ── Store ─────────────────────────────────────────────────────────────────
    try:
        chunks_batch = chunking.to_chromadb_batch()
        emb_batch    = embedding_result.to_chromadb_batch()

        insert_result = vector_store.add_documents(
            ids=chunks_batch["ids"],
            documents=chunks_batch["documents"],
            embeddings=emb_batch["embeddings"],
            metadatas=chunks_batch["metadatas"],
        )
    except Exception as exc:
        _log.error("ingest.store_failed", filename=filename, error=str(exc))
        return FileIngestResult(
            filename=filename,
            status=IngestStatus.FAILED,
            pages_extracted=pages_extracted,
            chunks_created=chunks_created,
            chunks_embedded=chunks_embedded,
            error=f"Vector store insert failed: {exc}",
        )

    chunks_stored = insert_result.total_inserted
    file_status = (
        IngestStatus.SUCCESS if insert_result.total_failed == 0
        else IngestStatus.PARTIAL
    )

    audit_logger.log_document_ingested(
        doc_id=filename,
        source_type="pdf",
        chunk_count=chunks_stored,
    )

    return FileIngestResult(
        filename=filename,
        status=file_status,
        pages_extracted=pages_extracted,
        chunks_created=chunks_created,
        chunks_embedded=chunks_embedded,
        chunks_stored=chunks_stored,
    )


@router.post(
    "",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest PDF documents into the knowledge base",
    description=(
        "Upload one or more PDF files. Each file is extracted, chunked, "
        "embedded with BAAI/bge-large-en-v1.5, and stored in ChromaDB. "
        "Files are not retained server-side after processing."
    ),
    responses={
        200: {"description": "Ingestion complete (check status field for partial failures)"},
        400: {"description": "No valid PDF files provided"},
        413: {"description": "One or more files exceed the 50 MB size limit"},
        503: {"description": "Ingestion pipeline unavailable"},
    },
)
async def ingest(
    loader: PDFLoaderDep,
    chunker: ChunkerDep,
    vector_store: VectorStoreDep,
    settings: SettingsDep,
    request_id: RequestIdDep,
    files: list[UploadFile] = File(..., description="PDF files to ingest"),
) -> IngestResponse:
    t_total = time.perf_counter()

    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided.",
        )

    _log.info(
        "ingest.request",
        request_id=request_id,
        file_count=len(files),
        filenames=[f.filename for f in files],
    )

    for upload in files:
        content_type = (upload.content_type or "").lower()
        if content_type not in _ALLOWED_CONTENT_TYPES and not (
            upload.filename or ""
        ).lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"'{upload.filename}' is not a PDF file.",
            )

    file_results: list[FileIngestResult] = []
    loop = asyncio.get_event_loop()

    with tempfile.TemporaryDirectory(prefix="cpa_ingest_") as tmp_dir:
        for upload in files:
            filename = upload.filename or "unknown.pdf"
            tmp_path = Path(tmp_dir) / filename

            try:
                content = await upload.read()
                if len(content) > _MAX_FILE_SIZE_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"'{filename}' exceeds the 50 MB file size limit.",
                    )
                tmp_path.write_bytes(content)
            except HTTPException:
                raise
            except Exception as exc:
                _log.error(
                    "ingest.upload_write_failed",
                    filename=filename,
                    error=str(exc),
                )
                file_results.append(FileIngestResult(
                    filename=filename,
                    status=IngestStatus.FAILED,
                    error=f"Failed to write upload: {exc}",
                ))
                continue

            _log.info(
                "ingest.processing_file",
                request_id=request_id,
                filename=filename,
                size_bytes=len(content),
            )

            result = await loop.run_in_executor(
                None,
                _ingest_one_file,
                tmp_path,
                filename,
                loader,
                chunker,
                vector_store,
                settings,
            )
            file_results.append(result)

    succeeded   = sum(1 for r in file_results if r.status == IngestStatus.SUCCESS)
    failed      = sum(1 for r in file_results if r.status == IngestStatus.FAILED)
    partial     = sum(1 for r in file_results if r.status == IngestStatus.PARTIAL)
    total_chunks = sum(r.chunks_stored for r in file_results)
    elapsed_ms  = (time.perf_counter() - t_total) * 1000

    if succeeded + partial > 0 and failed > 0:
        overall = IngestStatus.PARTIAL
    elif failed == len(file_results):
        overall = IngestStatus.FAILED
    else:
        overall = IngestStatus.SUCCESS

    _log.info(
        "ingest.complete",
        request_id=request_id,
        total_files=len(files),
        succeeded=succeeded,
        partial=partial,
        failed=failed,
        total_chunks=total_chunks,
        elapsed_ms=round(elapsed_ms, 2),
    )

    return IngestResponse(
        status=overall,
        total_files=len(files),
        succeeded=succeeded + partial,
        failed=failed,
        total_chunks=total_chunks,
        files=file_results,
        elapsed_ms=round(elapsed_ms, 2),
        collection=settings.chroma.collection_name,
    )
