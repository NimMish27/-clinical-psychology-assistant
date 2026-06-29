from __future__ import annotations

from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, model_validator


class DocumentStatus(str, Enum):
    """Overall document processing outcome."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class ExtractionStatus(str, Enum):
    """Outcome of text extraction for a single page."""
    SUCCESS = "success"
    EMPTY = "empty"
    SKIPPED = "skipped"


class PDFMetadata(BaseModel):
    """Document-level metadata parsed from PDF info dictionary."""
    title: str | None = None
    author: str | None = None
    subject: str | None = None
    creator: str | None = None
    producer: str | None = None
    creation_date: str | None = None
    modification_date: str | None = None
    page_count: int

    model_config = {"frozen": True}


class PageRecord(BaseModel):
    """
    Structured record representing text extracted from a single page.
    """
    page: int = Field(..., ge=1)
    text: str
    source: str
    status: ExtractionStatus = ExtractionStatus.SUCCESS
    error: str | None = None
    char_count: int = 0

    @field_validator("source")
    @classmethod
    def filename_only(cls, v: str) -> str:
        """Store only the filename component, never a full path."""
        return Path(v).name

    @model_validator(mode="after")
    def compute_fields(self) -> PageRecord:
        # compute char_count
        object.__setattr__(self, "char_count", len(self.text))
        if self.status == ExtractionStatus.SUCCESS and not self.text:
            object.__setattr__(self, "status", ExtractionStatus.EMPTY)
        return self

    def is_usable(self) -> bool:
        """Returns True if the page has extractable text and did not fail/skip."""
        return self.status == ExtractionStatus.SUCCESS and bool(self.text)

    model_config = {"frozen": True}


class PDFExtractionResult(BaseModel):
    """
    Aggregated result of text extraction for an entire document.
    """
    source: str
    status: DocumentStatus
    pages: list[PageRecord] = Field(default_factory=list)
    metadata: PDFMetadata | None = None
    errors: list[str] = Field(default_factory=list)

    @property
    def total_pages(self) -> int:
        return len(self.pages)

    @property
    def extracted_pages(self) -> int:
        return sum(1 for p in self.pages if p.status == ExtractionStatus.SUCCESS)

    @property
    def empty_pages(self) -> int:
        return sum(1 for p in self.pages if p.status == ExtractionStatus.EMPTY)

    @property
    def skipped_pages(self) -> int:
        return sum(1 for p in self.pages if p.status == ExtractionStatus.SKIPPED)

    @property
    def total_chars(self) -> int:
        return sum(p.char_count for p in self.pages)

    def usable_pages(self) -> list[PageRecord]:
        return [p for p in self.pages if p.is_usable()]

    def to_page_dicts(self) -> list[dict]:
        return [
            {
                "page": p.page,
                "text": p.text,
                "source": p.source,
            }
            for p in self.usable_pages()
        ]

    model_config = {"frozen": False}
