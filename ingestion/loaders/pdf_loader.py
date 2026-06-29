"""
ingestion/loaders/pdf_loader.py
────────────────────────────────
Production-ready PDF text extraction using PyMuPDF (fitz).

Design principles:
  - Single Responsibility: this module only extracts text. Chunking,
    embedding, and storage are handled downstream.
  - Fail gracefully: corrupted pages are skipped and logged; the pipeline
    continues rather than aborting an entire batch for one bad page.
  - Structured output: always returns a PDFExtractionResult — callers
    never parse exception messages to understand what happened.
  - No side effects: the loader is stateless and does not write to disk.
  - Fully typed: all public interfaces carry type annotations.

Usage (single file):
    from ingestion.loaders.pdf_loader import PDFLoader

    loader = PDFLoader()
    result = loader.load("data/raw/cbt_manual.pdf")

    for page_dict in result.to_page_dicts():
        print(page_dict)  # {"page": 1, "text": "...", "source": "cbt_manual.pdf"}

Usage (batch):
    results = loader.load_batch(["doc1.pdf", "doc2.pdf"], skip_errors=True)

Usage (async — for FastAPI endpoints):
    result = await loader.aload("data/raw/cbt_manual.pdf")
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING

from ingestion.loaders.exceptions import (
    PDFCorruptedError,
    PDFEmptyError,
    PDFFileNotFoundError,
    PDFPageExtractionError,
    PDFPasswordError,
    PDFPermissionError,
)
from ingestion.loaders.models import (
    DocumentStatus,
    ExtractionStatus,
    PDFExtractionResult,
    PDFMetadata,
    PageRecord,
)

# Lazy import guard: fitz (PyMuPDF) is a heavy C extension.
# We import at module level for type hints only and do the real import
# inside methods so tests can mock it easily.
try:
    import fitz  # PyMuPDF
    _FITZ_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FITZ_AVAILABLE = False
    fitz = None  # type: ignore[assignment]

if TYPE_CHECKING:
    import fitz as fitz_types  # noqa: F401 — used for type hints only


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_fitz() -> None:
    """Raise a clear ImportError if PyMuPDF is not installed."""
    if not _FITZ_AVAILABLE:
        raise ImportError(
            "PyMuPDF (fitz) is required for PDF loading. "
            "Install it with: pip install pymupdf"
        )


def _validate_path(path: Path) -> None:
    """
    Pre-flight path checks before handing off to PyMuPDF.

    Raises:
        PDFFileNotFoundError: path does not exist or is not a file.
        PDFPermissionError:   file exists but cannot be read.
    """
    if not path.exists() or not path.is_file():
        raise PDFFileNotFoundError(path=path)
    try:
        path.open("rb").close()
    except PermissionError as exc:
        raise PDFPermissionError(path=path, cause=exc) from exc


def _open_document(path: Path) -> "fitz_types.Document":
    """
    Open a PDF document with PyMuPDF.

    Tries normal open first; if that fails, attempts PyMuPDF's built-in
    repair mode before raising PDFCorruptedError.

    Raises:
        PDFPasswordError:  document is encrypted.
        PDFCorruptedError: document cannot be parsed even after repair.
    """
    try:
        doc: fitz_types.Document = fitz.open(str(path))  # type: ignore[union-attr]
    except fitz.FileDataError as exc:  # type: ignore[union-attr]
        # Attempt lightweight repair (rebuilds xref table)
        try:
            doc = fitz.open(str(path), filetype="pdf")  # type: ignore[union-attr]
            if doc.needs_pass:
                raise PDFPasswordError(path=path) from exc
        except Exception as repair_exc:
            raise PDFCorruptedError(
                path=path, repair_attempted=True, cause=repair_exc
            ) from exc
    except Exception as exc:
        raise PDFCorruptedError(path=path, repair_attempted=False, cause=exc) from exc

    if doc.needs_pass:
        doc.close()
        raise PDFPasswordError(path=path)

    return doc


def _extract_metadata(doc: "fitz_types.Document", page_count: int) -> PDFMetadata:
    """
    Extract document-level metadata from the PDF info dictionary.
    All fields are treated as optional — metadata is unreliable in the wild.
    """
    try:
        info: dict = doc.metadata or {}
    except Exception:
        info = {}

    def _safe(key: str) -> str | None:
        val = info.get(key)
        return str(val).strip() or None if val else None

    return PDFMetadata(
        title=_safe("title"),
        author=_safe("author"),
        subject=_safe("subject"),
        creator=_safe("creator"),
        producer=_safe("producer"),
        creation_date=_safe("creationDate"),
        modification_date=_safe("modDate"),
        page_count=page_count,
    )


def _extract_page_text(
    doc: "fitz_types.Document",
    page_index: int,
    source_name: str,
) -> PageRecord:
    """
    Extract text from a single page.

    Uses PyMuPDF's "text" extraction mode which returns plain UTF-8 text.
    The "blocks" and "dict" modes are richer but much slower — use them
    in a separate layout-preserving loader if needed.

    Args:
        doc:         Open PyMuPDF document.
        page_index:  0-based internal index.
        source_name: Filename for the PageRecord.source field.

    Returns:
        PageRecord with status SUCCESS, EMPTY, or SKIPPED.
    """
    page_number = page_index + 1  # convert to 1-based for humans

    try:
        page: fitz_types.Page = doc[page_index]  # type: ignore[index]
        raw_text: str = page.get_text("text")  # type: ignore[attr-defined]
        cleaned_text = _clean_text(raw_text)

        return PageRecord(
            page=page_number,
            text=cleaned_text,
            source=source_name,
            status=ExtractionStatus.SUCCESS,
        )

    except Exception as exc:
        # Per-page errors are non-fatal — we record and skip.
        return PageRecord(
            page=page_number,
            text="",
            source=source_name,
            status=ExtractionStatus.SKIPPED,
            error=f"{type(exc).__name__}: {exc}",
        )


def _clean_text(raw: str) -> str:
    """
    Lightweight text cleaning applied to every extracted page.

    Operations (in order):
      1. Normalise line endings to \\n
      2. Collapse runs of 3+ blank lines to a single blank line
         (preserves paragraph breaks but removes excessive whitespace)
      3. Strip leading/trailing whitespace from the whole page

    We intentionally do NOT:
      - Remove hyphens (clinical abbreviations like "CBT-I" must be preserved)
      - Strip single blank lines (they often separate paragraphs)
      - Lower-case (downstream embeddings handle that)
    """
    if not raw:
        return ""

    # Normalise Windows line endings
    text = raw.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse 3+ consecutive newlines → 2 (one blank line)
    import re
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

class PDFLoader:
    """
    Stateless PDF text extractor backed by PyMuPDF.

    Thread-safety: instances are stateless; the same instance can be used
    across threads or async tasks without synchronisation.

    Args:
        skip_empty_pages: If True (default), pages with no extractable text
            are included in the result with status EMPTY. Set False to omit
            them entirely from result.pages.
        min_text_length:  Pages whose cleaned text is shorter than this
            threshold are treated as empty (catches pages with only a few
            stray characters from artifacts). Default: 10.
    """

    def __init__(
        self,
        *,
        skip_empty_pages: bool = False,
        min_text_length: int = 10,
    ) -> None:
        self._skip_empty_pages = skip_empty_pages
        self._min_text_length = min_text_length

        # Import guard — fail fast at construction time, not mid-batch
        _require_fitz()

        from app_logging.logger import get_logger
        self._log = get_logger(__name__)

    # ── Core synchronous loader ───────────────────────────────────────────────

    def load(self, path: str | Path) -> PDFExtractionResult:
        """
        Extract text from all pages of a PDF file.

        Args:
            path: Path to the PDF file (str or Path).

        Returns:
            PDFExtractionResult with per-page PageRecords and aggregate stats.

        Raises:
            PDFFileNotFoundError: File does not exist.
            PDFPermissionError:   File cannot be read.
            PDFPasswordError:     File is password-protected.
            PDFCorruptedError:    File cannot be parsed by PyMuPDF.
            PDFEmptyError:        File opened but has zero pages.

        Note:
            Per-page errors (PDFPageExtractionError equivalent) do NOT raise —
            they are recorded in the result as SKIPPED pages. The result's
            status will be PARTIAL in that case.
        """
        resolved = Path(path).resolve()
        source_name = resolved.name
        t_start = time.perf_counter()

        self._log.info(
            "pdf_loader.start",
            source=source_name,
            path=str(resolved),
        )

        # ── Pre-flight ────────────────────────────────────────────────────────
        _validate_path(resolved)

        # ── Open document ─────────────────────────────────────────────────────
        doc = _open_document(resolved)

        try:
            page_count = len(doc)

            if page_count == 0:
                raise PDFEmptyError(path=resolved)

            self._log.info(
                "pdf_loader.opened",
                source=source_name,
                page_count=page_count,
            )

            # ── Extract metadata ───────────────────────────────────────────────
            metadata = _extract_metadata(doc, page_count)

            # ── Extract pages ──────────────────────────────────────────────────
            pages: list[PageRecord] = []
            skipped_errors: list[str] = []

            for i in range(page_count):
                record = _extract_page_text(doc, i, source_name)

                # Apply min_text_length threshold
                if (
                    record.status == ExtractionStatus.SUCCESS
                    and record.char_count < self._min_text_length
                ):
                    record = PageRecord(
                        page=record.page,
                        text="",
                        source=record.source,
                        status=ExtractionStatus.EMPTY,
                    )

                if record.status == ExtractionStatus.SKIPPED:
                    err_detail = record.error or "unknown error"
                    skipped_errors.append(f"page {record.page}: {err_detail}")
                    self._log.warning(
                        "pdf_loader.page_skipped",
                        source=source_name,
                        page=record.page,
                        reason=err_detail,
                    )

                if self._skip_empty_pages and record.status == ExtractionStatus.EMPTY:
                    continue

                pages.append(record)

        finally:
            doc.close()

        # ── Determine overall status ───────────────────────────────────────────
        n_skipped = sum(1 for p in pages if p.status == ExtractionStatus.SKIPPED)
        if n_skipped == 0:
            doc_status = DocumentStatus.SUCCESS
        elif n_skipped < page_count:
            doc_status = DocumentStatus.PARTIAL
        else:
            doc_status = DocumentStatus.FAILED

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        result = PDFExtractionResult(
            source=source_name,
            status=doc_status,
            pages=pages,
            metadata=metadata,
            extraction_time_ms=round(elapsed_ms, 2),
            errors=skipped_errors,
        )

        self._log.info(
            "pdf_loader.complete",
            source=source_name,
            status=doc_status,
            total_pages=result.total_pages,
            extracted_pages=result.extracted_pages,
            skipped_pages=result.skipped_pages,
            empty_pages=result.empty_pages,
            total_chars=result.total_chars,
            elapsed_ms=round(elapsed_ms, 2),
        )

        return result

    # ── Batch loader ──────────────────────────────────────────────────────────

    def load_batch(
        self,
        paths: list[str | Path],
        *,
        skip_errors: bool = True,
    ) -> list[PDFExtractionResult]:
        """
        Extract text from multiple PDF files.

        Args:
            paths:       List of PDF file paths.
            skip_errors: If True (default), document-level errors
                (not found, corrupted, etc.) are logged and skipped
                rather than raising and aborting the whole batch.
                If False, the first error raises immediately.

        Returns:
            List of PDFExtractionResult in the same order as paths.
            Failed documents are represented by a result with
            status=FAILED (when skip_errors=True).
        """
        results: list[PDFExtractionResult] = []
        total = len(paths)

        self._log.info("pdf_loader.batch_start", total_files=total)

        for idx, path in enumerate(paths, start=1):
            self._log.debug(
                "pdf_loader.batch_progress",
                file_index=idx,
                total=total,
                path=str(path),
            )
            try:
                result = self.load(path)
                results.append(result)
            except Exception as exc:
                if not skip_errors:
                    raise
                source_name = Path(path).name
                self._log.error(
                    "pdf_loader.batch_file_failed",
                    source=source_name,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                # Represent the failure as a FAILED result so the caller
                # has a consistent return type.
                results.append(
                    PDFExtractionResult(
                        source=source_name,
                        status=DocumentStatus.FAILED,
                        pages=[],
                        errors=[f"{type(exc).__name__}: {exc}"],
                    )
                )

        n_success = sum(1 for r in results if r.status == DocumentStatus.SUCCESS)
        n_partial = sum(1 for r in results if r.status == DocumentStatus.PARTIAL)
        n_failed  = sum(1 for r in results if r.status == DocumentStatus.FAILED)

        self._log.info(
            "pdf_loader.batch_complete",
            total=total,
            success=n_success,
            partial=n_partial,
            failed=n_failed,
        )

        return results

    # ── Async interface ───────────────────────────────────────────────────────

    async def aload(self, path: str | Path) -> PDFExtractionResult:
        """
        Async wrapper around load() for use in FastAPI endpoints.

        Runs the synchronous PyMuPDF extraction in a thread-pool executor
        so it does not block the event loop.

        Args:
            path: Path to the PDF file.

        Returns:
            PDFExtractionResult (same as load()).
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.load, path)

    async def aload_batch(
        self,
        paths: list[str | Path],
        *,
        skip_errors: bool = True,
        max_concurrency: int = 4,
    ) -> list[PDFExtractionResult]:
        """
        Async batch loader with bounded concurrency.

        Runs up to max_concurrency extractions in parallel using a semaphore,
        preventing memory spikes when processing large batches.

        Args:
            paths:           List of PDF file paths.
            skip_errors:     Passed through to the underlying load call.
            max_concurrency: Maximum simultaneous extractions. Default: 4.

        Returns:
            List of PDFExtractionResult in the same order as paths.
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _load_one(path: str | Path) -> PDFExtractionResult:
            async with semaphore:
                try:
                    return await self.aload(path)
                except Exception as exc:
                    if not skip_errors:
                        raise
                    source_name = Path(path).name
                    self._log.error(
                        "pdf_loader.async_batch_file_failed",
                        source=source_name,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                    return PDFExtractionResult(
                        source=source_name,
                        status=DocumentStatus.FAILED,
                        pages=[],
                        errors=[f"{type(exc).__name__}: {exc}"],
                    )

        return list(await asyncio.gather(*(_load_one(p) for p in paths)))
