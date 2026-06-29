"""
ingestion/loaders/exceptions.py
────────────────────────────────
Exception hierarchy for the PDF ingestion pipeline.

All exceptions carry structured context (file path, page number, etc.)
so callers can make informed decisions — retry, skip, or escalate —
without parsing string messages.

Hierarchy:
    PDFLoaderError              ← base for all PDF errors
    ├── PDFFileNotFoundError    ← path does not exist
    ├── PDFPermissionError      ← OS-level read permission denied
    ├── PDFPasswordError        ← document requires a password
    ├── PDFCorruptedError       ← file is structurally invalid / unreadable
    ├── PDFEmptyError           ← opened OK but contains zero pages
    └── PDFPageExtractionError  ← specific page failed during text extraction
"""

from __future__ import annotations

from pathlib import Path


# ── Base ──────────────────────────────────────────────────────────────────────

class PDFLoaderError(Exception):
    """
    Base class for all PDF loader exceptions.

    Attributes:
        path:    Absolute path of the file being processed.
        message: Human-readable description.
        cause:   The original exception that triggered this one, if any.
    """

    def __init__(
        self,
        message: str,
        path: Path | str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        # Store path as str to preserve the caller's original slash style.
        # On Windows, Path() normalises forward slashes to backslashes which
        # breaks assertions like assert "/data/doc.pdf" in str(exc).
        self._path_raw: str | None = str(path) if path is not None else None
        self.path: Path | None = Path(path) if path else None
        self.cause: BaseException | None = cause

    def __str__(self) -> str:
        base = super().__str__()
        path_hint = f" [file: {self._path_raw}]" if self._path_raw else ""
        cause_hint = f" [caused by: {type(self.cause).__name__}: {self.cause}]" if self.cause else ""
        return f"{base}{path_hint}{cause_hint}"

    def to_dict(self) -> dict[str, str | None]:
        """Serialisable representation for API error responses."""
        return {
            "error_type": type(self).__name__,
            "message": str(super().__str__()),
            "path": str(self.path) if self.path else None,
            "cause": str(self.cause) if self.cause else None,
        }


# ── Concrete exceptions ───────────────────────────────────────────────────────

class PDFFileNotFoundError(PDFLoaderError):
    """
    The specified PDF path does not exist or is not a file.

    Example:
        raise PDFFileNotFoundError(path="/data/raw/missing.pdf")
    """

    def __init__(self, path: Path | str, cause: BaseException | None = None) -> None:
        super().__init__(
            message=f"PDF file not found: '{path}'",
            path=path,
            cause=cause,
        )


class PDFPermissionError(PDFLoaderError):
    """
    The process does not have OS-level read permission for the file.

    Example:
        raise PDFPermissionError(path="/data/restricted.pdf")
    """

    def __init__(self, path: Path | str, cause: BaseException | None = None) -> None:
        super().__init__(
            message=f"Permission denied reading PDF: '{path}'",
            path=path,
            cause=cause,
        )


class PDFPasswordError(PDFLoaderError):
    """
    The PDF is encrypted and requires a password to open.

    Attributes:
        path: File that requires authentication.

    Example:
        raise PDFPasswordError(path="/data/protected.pdf")
    """

    def __init__(self, path: Path | str, cause: BaseException | None = None) -> None:
        super().__init__(
            message=f"PDF is password-protected and cannot be read without credentials: '{path}'",
            path=path,
            cause=cause,
        )


class PDFCorruptedError(PDFLoaderError):
    """
    The file exists but PyMuPDF cannot parse its structure.

    This covers: truncated files, invalid cross-reference tables,
    non-PDF content with a .pdf extension, and unrecoverable
    internal stream errors.

    Attributes:
        path:            The corrupt file.
        repair_attempted: Whether PyMuPDF's repair mode was tried.

    Example:
        raise PDFCorruptedError(path="/data/bad.pdf", repair_attempted=True)
    """

    def __init__(
        self,
        path: Path | str,
        repair_attempted: bool = False,
        cause: BaseException | None = None,
    ) -> None:
        repair_note = " (repair mode also failed)" if repair_attempted else ""
        super().__init__(
            message=f"PDF is corrupted or not a valid PDF{repair_note}: '{path}'",
            path=path,
            cause=cause,
        )
        self.repair_attempted = repair_attempted


class PDFEmptyError(PDFLoaderError):
    """
    The PDF opened successfully but contains zero pages.

    This is distinct from a corrupted file — the document is structurally
    valid but has no content to extract.

    Example:
        raise PDFEmptyError(path="/data/blank.pdf")
    """

    def __init__(self, path: Path | str, cause: BaseException | None = None) -> None:
        super().__init__(
            message=f"PDF opened successfully but contains no pages: '{path}'",
            path=path,
            cause=cause,
        )


class PDFPageExtractionError(PDFLoaderError):
    """
    Text extraction failed for a specific page.

    The document as a whole is valid; only this page is problematic
    (e.g. a corrupted content stream on an individual page, a rendering
    error, or an image-only page with no text layer).

    Attributes:
        page_number: 1-based page index that failed.
        recoverable: If True, the pipeline can skip this page and continue.

    Example:
        raise PDFPageExtractionError(path="doc.pdf", page_number=7, recoverable=True)
    """

    def __init__(
        self,
        path: Path | str,
        page_number: int,
        recoverable: bool = True,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            message=(
                f"Failed to extract text from page {page_number} of '{path}' "
                f"({'recoverable — page skipped' if recoverable else 'unrecoverable'})"
            ),
            path=path,
            cause=cause,
        )
        self.page_number = page_number
        self.recoverable = recoverable

    def to_dict(self) -> dict[str, str | int | bool | None]:  # type: ignore[override]
        base = super().to_dict()
        return {**base, "page_number": self.page_number, "recoverable": self.recoverable}
