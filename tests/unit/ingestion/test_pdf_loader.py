"""
tests/unit/ingestion/test_pdf_loader.py
────────────────────────────────────────
Unit tests for the PDF extraction module.

All tests use mocks — no real PDFs or PyMuPDF installation required.
This keeps the test suite fast and CI-friendly.

Run:
    pytest tests/unit/ingestion/test_pdf_loader.py -v
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from ingestion.loaders.exceptions import (
    PDFCorruptedError,
    PDFEmptyError,
    PDFFileNotFoundError,
    PDFPasswordError,
    PDFPermissionError,
    PDFPageExtractionError,
    PDFLoaderError,
)
from ingestion.loaders.models import (
    DocumentStatus,
    ExtractionStatus,
    PDFExtractionResult,
    PDFMetadata,
    PageRecord,
)


# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def mock_fitz_page():
    """A mock PyMuPDF page that returns controllable text."""
    page = MagicMock()
    page.get_text.return_value = "Sample clinical text for page.\n\nSecond paragraph."
    return page


@pytest.fixture()
def mock_fitz_doc(mock_fitz_page):
    """A mock PyMuPDF document with 3 pages."""
    doc = MagicMock()
    doc.__len__ = MagicMock(return_value=3)
    doc.__getitem__ = MagicMock(return_value=mock_fitz_page)
    doc.needs_pass = False
    doc.metadata = {
        "title": "CBT Manual",
        "author": "Dr. Beck",
        "subject": "Cognitive Behavioural Therapy",
        "creator": "Word",
        "producer": "Adobe",
        "creationDate": "D:20230101",
        "modDate": "D:20231201",
    }
    return doc


@pytest.fixture()
def real_path(tmp_path):
    """A real (empty) file on disk so _validate_path passes."""
    p = tmp_path / "cbt_manual.pdf"
    p.write_bytes(b"%PDF-1.4 fake content")
    return p


# ────────────────────────────────────────────────────────────────────────────
# Exception model tests
# ────────────────────────────────────────────────────────────────────────────

class TestExceptions:
    def test_base_exception_str(self):
        exc = PDFLoaderError("Something went wrong", path="/data/doc.pdf")
        assert "Something went wrong" in str(exc)
        assert "/data/doc.pdf" in str(exc)

    def test_base_exception_to_dict(self):
        exc = PDFLoaderError("oops", path="/data/doc.pdf")
        d = exc.to_dict()
        assert d["error_type"] == "PDFLoaderError"
        assert d["message"] == "oops"
        assert "doc.pdf" in d["path"]

    def test_file_not_found(self):
        exc = PDFFileNotFoundError(path="/missing.pdf")
        assert "not found" in str(exc)
        assert exc.path == Path("/missing.pdf")

    def test_password_error(self):
        exc = PDFPasswordError(path="/secure.pdf")
        assert "password" in str(exc).lower()

    def test_corrupted_with_repair(self):
        exc = PDFCorruptedError(path="/bad.pdf", repair_attempted=True)
        assert "repair mode also failed" in str(exc)
        assert exc.repair_attempted is True

    def test_corrupted_without_repair(self):
        exc = PDFCorruptedError(path="/bad.pdf", repair_attempted=False)
        assert "repair mode" not in str(exc)

    def test_empty_error(self):
        exc = PDFEmptyError(path="/blank.pdf")
        assert "no pages" in str(exc).lower()

    def test_page_extraction_error(self):
        exc = PDFPageExtractionError(path="doc.pdf", page_number=5, recoverable=True)
        assert exc.page_number == 5
        assert exc.recoverable is True
        d = exc.to_dict()
        assert d["page_number"] == 5
        assert d["recoverable"] is True

    def test_page_extraction_error_unrecoverable(self):
        exc = PDFPageExtractionError(path="doc.pdf", page_number=2, recoverable=False)
        assert "unrecoverable" in str(exc)

    def test_cause_chaining(self):
        original = ValueError("disk error")
        exc = PDFCorruptedError(path="/bad.pdf", cause=original)
        assert exc.cause is original
        assert "ValueError" in str(exc)


# ────────────────────────────────────────────────────────────────────────────
# PageRecord model tests
# ────────────────────────────────────────────────────────────────────────────

class TestPageRecord:
    def test_char_count_auto_computed(self):
        r = PageRecord(page=1, text="hello world", source="doc.pdf")
        assert r.char_count == 11

    def test_empty_text_sets_empty_status(self):
        r = PageRecord(page=1, text="", source="doc.pdf", status=ExtractionStatus.SUCCESS)
        assert r.status == ExtractionStatus.EMPTY
        assert r.char_count == 0

    def test_source_strips_path(self):
        r = PageRecord(page=1, text="x" * 50, source="/data/raw/cbt_manual.pdf")
        assert r.source == "cbt_manual.pdf"

    def test_is_usable_with_content(self):
        r = PageRecord(page=1, text="Clinical content here", source="doc.pdf")
        assert r.is_usable() is True

    def test_is_usable_empty(self):
        r = PageRecord(page=1, text="", source="doc.pdf")
        assert r.is_usable() is False

    def test_is_usable_skipped(self):
        r = PageRecord(
            page=1, text="", source="doc.pdf",
            status=ExtractionStatus.SKIPPED, error="fitz error"
        )
        assert r.is_usable() is False

    def test_frozen(self):
        r = PageRecord(page=1, text="text", source="doc.pdf")
        with pytest.raises(Exception):  # ValidationError or TypeError
            r.page = 2  # type: ignore


# ────────────────────────────────────────────────────────────────────────────
# PDFExtractionResult model tests
# ────────────────────────────────────────────────────────────────────────────

class TestPDFExtractionResult:
    def _make_result(self, pages):
        return PDFExtractionResult(
            source="test.pdf",
            status=DocumentStatus.SUCCESS,
            pages=pages,
        )

    def test_aggregates_computed(self):
        pages = [
            PageRecord(page=1, text="A" * 100, source="test.pdf"),
            PageRecord(page=2, text="", source="test.pdf"),
            PageRecord(page=3, text="B" * 50, source="test.pdf"),
        ]
        result = self._make_result(pages)
        assert result.total_pages == 3
        assert result.extracted_pages == 2
        assert result.empty_pages == 1
        assert result.total_chars == 150

    def test_to_page_dicts_only_usable(self):
        pages = [
            PageRecord(page=1, text="Clinical text on page one.", source="cbt.pdf"),
            PageRecord(page=2, text="", source="cbt.pdf"),  # empty
        ]
        result = self._make_result(pages)
        dicts = result.to_page_dicts()
        assert len(dicts) == 1
        assert dicts[0] == {
            "page": 1,
            "text": "Clinical text on page one.",
            "source": "cbt.pdf",
        }

    def test_to_page_dicts_format(self):
        pages = [PageRecord(page=1, text="text", source="manual.pdf")]
        result = self._make_result(pages)
        d = result.to_page_dicts()[0]
        assert set(d.keys()) == {"page", "text", "source"}

    def test_usable_pages(self):
        pages = [
            PageRecord(page=1, text="good content here", source="doc.pdf"),
            PageRecord(page=2, text="", source="doc.pdf"),
        ]
        result = self._make_result(pages)
        usable = result.usable_pages()
        assert len(usable) == 1
        assert usable[0].page == 1


# ────────────────────────────────────────────────────────────────────────────
# PDFLoader tests (mocked fitz)
# ────────────────────────────────────────────────────────────────────────────

class TestPDFLoader:
    """Tests for PDFLoader using mocked PyMuPDF."""

    def _make_loader(self):
        """Instantiate loader with fitz availability patched."""
        with patch("ingestion.loaders.pdf_loader._FITZ_AVAILABLE", True):
            with patch("ingestion.loaders.pdf_loader._require_fitz"):
                from ingestion.loaders.pdf_loader import PDFLoader
                with patch.object(PDFLoader, "__init__", lambda self, **kw: None):
                    loader = PDFLoader.__new__(PDFLoader)
                    loader._skip_empty_pages = False
                    loader._min_text_length = 10
                    # Attach a real logger mock
                    log = MagicMock()
                    log.info = MagicMock()
                    log.warning = MagicMock()
                    log.error = MagicMock()
                    log.debug = MagicMock()
                    loader._log = log
                    return loader

    @patch("ingestion.loaders.pdf_loader._validate_path")
    @patch("ingestion.loaders.pdf_loader._open_document")
    def test_load_success(self, mock_open, mock_validate, mock_fitz_doc, real_path):
        """Happy path: 3 pages all extracted successfully."""
        mock_open.return_value = mock_fitz_doc
        mock_fitz_doc.__len__ = MagicMock(return_value=3)
        mock_fitz_doc.__getitem__ = MagicMock(
            side_effect=lambda i: MagicMock(
                get_text=MagicMock(return_value=f"Page {i+1} content with sufficient text.")
            )
        )

        loader = self._make_loader()
        result = loader.load(real_path)

        assert result.status == DocumentStatus.SUCCESS
        assert result.total_pages == 3
        assert result.extracted_pages == 3
        assert result.skipped_pages == 0
        mock_fitz_doc.close.assert_called_once()

    @patch("ingestion.loaders.pdf_loader._validate_path")
    @patch("ingestion.loaders.pdf_loader._open_document")
    def test_load_one_page_fails(self, mock_open, mock_validate, real_path):
        """One page raises — result is PARTIAL, other pages succeed."""
        doc = MagicMock()
        doc.__len__ = MagicMock(return_value=3)
        doc.needs_pass = False
        doc.metadata = {}

        def side_effect(i):
            page = MagicMock()
            if i == 1:
                page.get_text.side_effect = RuntimeError("stream error on page 2")
            else:
                page.get_text.return_value = f"Good text on page {i+1} with content."
            return page

        doc.__getitem__ = MagicMock(side_effect=side_effect)
        mock_open.return_value = doc

        loader = self._make_loader()
        result = loader.load(real_path)

        assert result.status == DocumentStatus.PARTIAL
        assert result.skipped_pages == 1
        assert result.extracted_pages == 2
        assert len(result.errors) == 1
        assert "page 2" in result.errors[0]

    @patch("ingestion.loaders.pdf_loader._validate_path")
    def test_load_file_not_found_raises(self, mock_validate, real_path):
        """PDFFileNotFoundError propagates from _validate_path."""
        mock_validate.side_effect = PDFFileNotFoundError(path=real_path)
        loader = self._make_loader()

        with pytest.raises(PDFFileNotFoundError):
            loader.load(real_path)

    @patch("ingestion.loaders.pdf_loader._validate_path")
    @patch("ingestion.loaders.pdf_loader._open_document")
    def test_load_password_protected(self, mock_open, mock_validate, real_path):
        """PDFPasswordError propagates from _open_document."""
        mock_open.side_effect = PDFPasswordError(path=real_path)
        loader = self._make_loader()

        with pytest.raises(PDFPasswordError):
            loader.load(real_path)

    @patch("ingestion.loaders.pdf_loader._validate_path")
    @patch("ingestion.loaders.pdf_loader._open_document")
    def test_load_corrupted(self, mock_open, mock_validate, real_path):
        """PDFCorruptedError propagates from _open_document."""
        mock_open.side_effect = PDFCorruptedError(path=real_path, repair_attempted=True)
        loader = self._make_loader()

        with pytest.raises(PDFCorruptedError) as exc_info:
            loader.load(real_path)
        assert exc_info.value.repair_attempted is True

    @patch("ingestion.loaders.pdf_loader._validate_path")
    @patch("ingestion.loaders.pdf_loader._open_document")
    def test_load_empty_pdf(self, mock_open, mock_validate, real_path):
        """PDFEmptyError raised when document has 0 pages."""
        doc = MagicMock()
        doc.__len__ = MagicMock(return_value=0)
        doc.needs_pass = False
        mock_open.return_value = doc

        loader = self._make_loader()

        with pytest.raises(PDFEmptyError):
            loader.load(real_path)
        doc.close.assert_called_once()

    @patch("ingestion.loaders.pdf_loader._validate_path")
    @patch("ingestion.loaders.pdf_loader._open_document")
    def test_doc_always_closed_on_error(self, mock_open, mock_validate, real_path):
        """Document is closed even when extraction raises mid-way."""
        doc = MagicMock()
        doc.__len__ = MagicMock(return_value=2)
        doc.needs_pass = False
        doc.metadata = {}
        # Make page access raise after opening
        doc.__getitem__ = MagicMock(side_effect=RuntimeError("boom"))
        mock_open.return_value = doc

        loader = self._make_loader()
        # Load should not raise (page errors are recoverable)
        result = loader.load(real_path)

        doc.close.assert_called_once()
        assert result.status in (DocumentStatus.PARTIAL, DocumentStatus.FAILED)

    @patch("ingestion.loaders.pdf_loader._validate_path")
    @patch("ingestion.loaders.pdf_loader._open_document")
    def test_min_text_length_threshold(self, mock_open, mock_validate, real_path):
        """Pages with fewer chars than min_text_length are treated as EMPTY."""
        doc = MagicMock()
        doc.__len__ = MagicMock(return_value=2)
        doc.needs_pass = False
        doc.metadata = {}

        def side_effect(i):
            page = MagicMock()
            page.get_text.return_value = "Hi" if i == 0 else "Full page of clinical content."
            return page

        doc.__getitem__ = MagicMock(side_effect=side_effect)
        mock_open.return_value = doc

        loader = self._make_loader()
        loader._min_text_length = 10
        result = loader.load(real_path)

        statuses = {p.page: p.status for p in result.pages}
        assert statuses[1] == ExtractionStatus.EMPTY   # "Hi" is only 2 chars
        assert statuses[2] == ExtractionStatus.SUCCESS  # full text passes

    # ── Batch tests ─────────────────────────────────────────────────────────

    @patch("ingestion.loaders.pdf_loader._validate_path")
    @patch("ingestion.loaders.pdf_loader._open_document")
    def test_batch_skip_errors(self, mock_open, mock_validate, real_path):
        """Batch continues past a failed file when skip_errors=True."""
        good_doc = MagicMock()
        good_doc.__len__ = MagicMock(return_value=1)
        good_doc.needs_pass = False
        good_doc.metadata = {}
        good_page = MagicMock()
        good_page.get_text.return_value = "Good clinical text on this page."
        good_doc.__getitem__ = MagicMock(return_value=good_page)

        call_count = 0
        def open_side_effect(path):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise PDFCorruptedError(path=path)
            return good_doc

        mock_open.side_effect = open_side_effect

        loader = self._make_loader()
        results = loader.load_batch([real_path, real_path], skip_errors=True)

        assert len(results) == 2
        assert results[0].status == DocumentStatus.FAILED
        assert results[1].status == DocumentStatus.SUCCESS

    @patch("ingestion.loaders.pdf_loader._validate_path")
    @patch("ingestion.loaders.pdf_loader._open_document")
    def test_batch_raises_on_error_when_skip_false(self, mock_open, mock_validate, real_path):
        """Batch raises immediately when skip_errors=False."""
        mock_open.side_effect = PDFCorruptedError(path=real_path)
        loader = self._make_loader()

        with pytest.raises(PDFCorruptedError):
            loader.load_batch([real_path], skip_errors=False)

    # ── Async tests ──────────────────────────────────────────────────────────

    @patch("ingestion.loaders.pdf_loader._validate_path")
    @patch("ingestion.loaders.pdf_loader._open_document")
    def test_aload_returns_result(self, mock_open, mock_validate, mock_fitz_doc, real_path):
        """aload() returns same result as load()."""
        mock_fitz_doc.__len__ = MagicMock(return_value=1)
        mock_fitz_doc.__getitem__ = MagicMock(
            return_value=MagicMock(
                get_text=MagicMock(return_value="Async extracted clinical content.")
            )
        )
        mock_open.return_value = mock_fitz_doc

        loader = self._make_loader()
        result = asyncio.run(loader.aload(real_path))

        assert isinstance(result, PDFExtractionResult)
        assert result.status == DocumentStatus.SUCCESS


# ────────────────────────────────────────────────────────────────────────────
# Text cleaning tests
# ────────────────────────────────────────────────────────────────────────────

class TestCleanText:
    def _clean(self, raw: str) -> str:
        from ingestion.loaders.pdf_loader import _clean_text
        return _clean_text(raw)

    def test_empty_string(self):
        assert self._clean("") == ""

    def test_strips_leading_trailing(self):
        assert self._clean("  hello  ") == "hello"

    def test_normalises_windows_line_endings(self):
        result = self._clean("line1\r\nline2\r\nline3")
        assert "\r" not in result
        assert result == "line1\nline2\nline3"

    def test_collapses_excessive_blank_lines(self):
        raw = "para1\n\n\n\n\npara2"
        result = self._clean(raw)
        assert "\n\n\n" not in result
        assert "para1" in result
        assert "para2" in result

    def test_preserves_single_blank_lines(self):
        raw = "para1\n\npara2"
        result = self._clean(raw)
        assert result == "para1\n\npara2"

    def test_preserves_hyphens(self):
        """Clinical abbreviations like CBT-I must not be altered."""
        result = self._clean("CBT-I is effective for insomnia.")
        assert "CBT-I" in result

    def test_preserves_clinical_content(self):
        raw = "DSM-5 Criterion A: Depressed mood.\n\nCriterion B: Anhedonia."
        result = self._clean(raw)
        assert "DSM-5" in result
        assert "Criterion A" in result
        assert "Criterion B" in result
