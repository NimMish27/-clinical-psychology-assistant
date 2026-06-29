"""
tests/unit/ingestion/test_chunker.py
──────────────────────────────────────
Unit tests for the clinical document chunker.

All tests use mocks or in-memory fixtures — no LangChain installation
or real PDFs required in CI. The splitter itself is mocked so tests
focus on orchestration, ID generation, model validation, and edge-case
handling rather than re-testing LangChain internals.

Run:
    pytest tests/unit/ingestion/test_chunker.py -v
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ingestion.processors.models import Chunk, ChunkingResult, ChunkingStatus


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_page(page: int, text: str, source: str = "DSM5.pdf") -> dict:
    return {"page": page, "text": text, "source": source}


def _make_chunker(**kwargs):
    """
    Build a Chunker with the LangChain splitter mocked out.
    Passes kwargs to Chunker.__init__ (chunk_size, chunk_overlap, etc.)
    """
    with patch("ingestion.processors.chunker._LC_AVAILABLE", True):
        with patch("ingestion.processors.chunker.RecursiveCharacterTextSplitter") as MockSplitter:
            instance = MockSplitter.return_value
            # No side_effect so that each test controls return_value independently.
            # When a test sets mock_splitter.split_text.return_value, it takes effect.
            instance.split_text = MagicMock(return_value=[])
            from ingestion.processors.chunker import Chunker
            chunker = Chunker(**kwargs)
            # Replace the real splitter with the mock instance for all future calls
            chunker._splitter = instance
            return chunker, instance


# ── Chunk model tests ─────────────────────────────────────────────────────────

class TestChunkModel:
    def test_to_dict_format(self):
        c = Chunk(
            chunk_id="DSM5__p0012__c0003",
            text="Criterion A: Depressed mood most of the day.",
            page=12,
            source="DSM5.pdf",
            chunk_index=3,
        )
        d = c.to_dict()
        assert set(d.keys()) == {"chunk_id", "text", "page", "source"}
        assert d["chunk_id"] == "DSM5__p0012__c0003"
        assert d["page"] == 12
        assert d["source"] == "DSM5.pdf"

    def test_char_and_word_count_auto_computed(self):
        text = "Major depressive episode criteria."
        c = Chunk(chunk_id="x__p0001__c0000", text=text, page=1, source="doc.pdf", chunk_index=0)
        assert c.char_count == len(text)
        assert c.word_count == 4

    def test_source_strips_path(self):
        c = Chunk(
            chunk_id="doc__p0001__c0000",
            text="Some clinical text here to pass the minimum length check.",
            page=1,
            source="/data/raw/subdir/DSM5.pdf",
            chunk_index=0,
        )
        assert c.source == "DSM5.pdf"

    def test_text_stripped(self):
        c = Chunk(
            chunk_id="doc__p0001__c0000",
            text="  whitespace padded text content for the chunk  ",
            page=1,
            source="doc.pdf",
            chunk_index=0,
        )
        assert c.text == "whitespace padded text content for the chunk"

    def test_empty_text_raises(self):
        with pytest.raises(Exception):  # Pydantic ValidationError
            Chunk(chunk_id="x", text="", page=1, source="doc.pdf", chunk_index=0)

    def test_whitespace_only_text_raises(self):
        with pytest.raises(Exception):
            Chunk(chunk_id="x", text="   ", page=1, source="doc.pdf", chunk_index=0)

    def test_frozen(self):
        c = Chunk(chunk_id="x__p0001__c0000", text="clinical text sample here", page=1, source="doc.pdf", chunk_index=0)
        with pytest.raises(Exception):
            c.page = 2  # type: ignore

    def test_to_chromadb_dict(self):
        c = Chunk(
            chunk_id="DSM5__p0001__c0000",
            text="Clinical criteria text content goes here for testing.",
            page=1,
            source="DSM5.pdf",
            chunk_index=0,
        )
        d = c.to_chromadb_dict()
        assert d["id"] == "DSM5__p0001__c0000"
        assert d["document"] == c.text
        assert d["metadata"]["source"] == "DSM5.pdf"
        assert d["metadata"]["page"] == 1
        assert d["metadata"]["chunk_index"] == 0
        assert "char_count" in d["metadata"]
        assert "word_count" in d["metadata"]

    def test_page_must_be_positive(self):
        with pytest.raises(Exception):
            Chunk(chunk_id="x", text="text", page=0, source="doc.pdf", chunk_index=0)


# ── ChunkingResult model tests ────────────────────────────────────────────────

class TestChunkingResult:
    def _make_chunk(self, page: int, idx: int, text: str = "A" * 80) -> Chunk:
        return Chunk(
            chunk_id=f"doc__p{page:04d}__c{idx:04d}",
            text=text,
            page=page,
            source="doc.pdf",
            chunk_index=idx,
        )

    def test_aggregates_computed(self):
        chunks = [
            self._make_chunk(1, 0, "A" * 100),
            self._make_chunk(1, 1, "B" * 80),
            self._make_chunk(2, 0, "C" * 120),
        ]
        result = ChunkingResult(source="doc.pdf", status=ChunkingStatus.SUCCESS, chunks=chunks)
        assert result.total_chunks == 3
        assert result.total_chars == 300

    def test_to_dicts_format(self):
        chunks = [self._make_chunk(1, 0, "Clinical chunk text here to validate output.")]
        result = ChunkingResult(source="DSM5.pdf", status=ChunkingStatus.SUCCESS, chunks=chunks)
        dicts = result.to_dicts()
        assert len(dicts) == 1
        assert set(dicts[0].keys()) == {"chunk_id", "text", "page", "source"}

    def test_to_chromadb_batch_structure(self):
        chunks = [self._make_chunk(1, i, "X" * 80) for i in range(3)]
        result = ChunkingResult(source="doc.pdf", status=ChunkingStatus.SUCCESS, chunks=chunks)
        batch = result.to_chromadb_batch()
        assert set(batch.keys()) == {"ids", "documents", "metadatas"}
        assert len(batch["ids"]) == 3
        assert len(batch["documents"]) == 3
        assert len(batch["metadatas"]) == 3

    def test_empty_chromadb_batch(self):
        result = ChunkingResult(source="doc.pdf", status=ChunkingStatus.EMPTY)
        batch = result.to_chromadb_batch()
        assert batch == {"ids": [], "documents": [], "metadatas": []}


# ── Chunk ID generation tests ─────────────────────────────────────────────────

class TestChunkIdGeneration:
    def _make_id(self, source: str, page: int, idx: int) -> str:
        from ingestion.processors.chunker import _make_chunk_id
        return _make_chunk_id(source, page, idx)

    def test_format(self):
        cid = self._make_id("DSM5.pdf", 12, 3)
        assert cid == "DSM5__p0012__c0003"

    def test_page_zero_padded(self):
        cid = self._make_id("manual.pdf", 1, 0)
        assert "__p0001__" in cid

    def test_large_page_number(self):
        cid = self._make_id("doc.pdf", 9999, 0)
        assert "__p9999__" in cid

    def test_source_with_path_stripped(self):
        cid = self._make_id("/data/raw/CBT_Manual.pdf", 1, 0)
        assert cid.startswith("CBT_Manual__")

    def test_spaces_in_source_sanitised(self):
        cid = self._make_id("CBT Manual 2024.pdf", 1, 0)
        assert " " not in cid

    def test_special_chars_sanitised(self):
        cid = self._make_id("DSM-5 (2013).pdf", 1, 0)
        assert "(" not in cid
        assert ")" not in cid

    def test_deterministic(self):
        id1 = self._make_id("DSM5.pdf", 5, 2)
        id2 = self._make_id("DSM5.pdf", 5, 2)
        assert id1 == id2

    def test_different_pages_different_ids(self):
        assert self._make_id("doc.pdf", 1, 0) != self._make_id("doc.pdf", 2, 0)

    def test_different_indices_different_ids(self):
        assert self._make_id("doc.pdf", 1, 0) != self._make_id("doc.pdf", 1, 1)

    def test_different_sources_different_ids(self):
        assert self._make_id("DSM5.pdf", 1, 0) != self._make_id("ICD11.pdf", 1, 0)


# ── Preprocessor tests ────────────────────────────────────────────────────────

class TestPreprocess:
    def _pre(self, text: str) -> str:
        from ingestion.processors.chunker import Chunker
        return Chunker._preprocess(text)

    def test_empty(self):
        assert self._pre("") == ""

    def test_strips_outer_whitespace(self):
        assert self._pre("  hello  ") == "hello"

    def test_normalises_crlf(self):
        result = self._pre("line1\r\nline2")
        assert "\r" not in result

    def test_removes_replacement_char(self):
        result = self._pre("text\ufffdwith\ufffdartefacts")
        assert "\ufffd" not in result

    def test_collapses_excessive_blank_lines(self):
        result = self._pre("para1\n\n\n\n\npara2")
        assert "\n\n\n" not in result

    def test_preserves_single_blank_line(self):
        result = self._pre("para1\n\npara2")
        assert result == "para1\n\npara2"

    def test_removes_standalone_page_numbers(self):
        text = "Content before.\n\n42\n\nContent after."
        result = self._pre(text)
        assert "\n42\n" not in result
        assert "Content before." in result
        assert "Content after." in result

    def test_preserves_clinical_hyphens(self):
        result = self._pre("CBT-I is effective for insomnia.\nDSM-5 criteria apply.")
        assert "CBT-I" in result
        assert "DSM-5" in result

    def test_preserves_clinical_punctuation(self):
        result = self._pre("Criterion A (1): Depressed mood; Criterion A (2): Anhedonia.")
        assert "Criterion A (1)" in result
        assert "Criterion A (2)" in result


# ── Chunker initialisation tests ──────────────────────────────────────────────

class TestChunkerInit:
    def test_invalid_overlap_greater_than_size(self):
        with patch("ingestion.processors.chunker._LC_AVAILABLE", True), \
             patch("ingestion.processors.chunker.RecursiveCharacterTextSplitter"):
            from ingestion.processors.chunker import Chunker
            with pytest.raises(ValueError, match="chunk_size"):
                Chunker(chunk_size=100, chunk_overlap=200)

    def test_invalid_min_chunk_chars(self):
        with patch("ingestion.processors.chunker._LC_AVAILABLE", True), \
             patch("ingestion.processors.chunker.RecursiveCharacterTextSplitter"):
            from ingestion.processors.chunker import Chunker
            with pytest.raises(ValueError, match="min_chunk_chars"):
                Chunker(min_chunk_chars=0)

    def test_missing_langchain_raises(self):
        with patch("ingestion.processors.chunker._LC_AVAILABLE", False):
            from ingestion.processors.chunker import Chunker
            with pytest.raises(ImportError, match="LangChain"):
                Chunker()

    def test_config_property(self):
        chunker, _ = _make_chunker(chunk_size=800, chunk_overlap=150)
        cfg = chunker.config
        assert cfg["chunk_size"] == 800
        assert cfg["chunk_overlap"] == 150

    def test_repr(self):
        chunker, _ = _make_chunker(chunk_size=800, chunk_overlap=150)
        assert "800" in repr(chunker)
        assert "150" in repr(chunker)


# ── chunk_pages tests ─────────────────────────────────────────────────────────

class TestChunkPages:
    def test_happy_path_produces_chunks(self):
        chunker, mock_splitter = _make_chunker()
        long_text = "Clinical psychology content. " * 30
        mock_splitter.split_text.return_value = [
            long_text[:len(long_text)//2],
            long_text[len(long_text)//2:],
        ]
        pages = [_make_page(1, long_text, "DSM5.pdf")]
        result = chunker.chunk_pages(pages)

        assert result.status == ChunkingStatus.SUCCESS
        assert result.total_chunks == 2
        assert result.source == "DSM5.pdf"

    def test_empty_pages_list_returns_empty(self):
        chunker, _ = _make_chunker()
        result = chunker.chunk_pages([])
        assert result.status == ChunkingStatus.EMPTY
        assert result.total_chunks == 0

    def test_page_below_min_length_skipped(self):
        chunker, mock_splitter = _make_chunker()
        # Text shorter than _MIN_PAGE_TEXT_LENGTH (30 chars)
        pages = [_make_page(1, "Short.", "doc.pdf")]
        result = chunker.chunk_pages(pages)

        assert result.skipped_pages == 1
        assert result.total_chunks == 0
        mock_splitter.split_text.assert_not_called()

    def test_multiple_pages_chunked(self):
        chunker, mock_splitter = _make_chunker()
        long = "Clinical content here. " * 20
        mock_splitter.split_text.return_value = [long[:len(long)//2], long[len(long)//2:]]
        pages = [
            _make_page(1, long, "DSM5.pdf"),
            _make_page(2, long, "DSM5.pdf"),
            _make_page(3, long, "DSM5.pdf"),
        ]
        result = chunker.chunk_pages(pages)
        assert result.total_pages_in == 3
        assert result.total_chunks == 6  # 2 chunks per page

    def test_chunk_ids_are_unique(self):
        chunker, mock_splitter = _make_chunker()
        long = "Unique clinical text chunk content here. " * 20
        mock_splitter.split_text.return_value = [long[:len(long)//2], long[len(long)//2:]]
        pages = [
            _make_page(1, long, "DSM5.pdf"),
            _make_page(2, long, "DSM5.pdf"),
        ]
        result = chunker.chunk_pages(pages)
        ids = [c.chunk_id for c in result.chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_chunk_ids_contain_source_and_page(self):
        chunker, mock_splitter = _make_chunker()
        long = "Major depressive disorder clinical criteria text. " * 15
        mock_splitter.split_text.return_value = [long]
        pages = [_make_page(7, long, "DSM5.pdf")]
        result = chunker.chunk_pages(pages)

        assert len(result.chunks) == 1
        assert "DSM5" in result.chunks[0].chunk_id
        assert "p0007" in result.chunks[0].chunk_id

    def test_page_metadata_on_chunks(self):
        chunker, mock_splitter = _make_chunker()
        text = "Criterion A: Significant weight loss when not dieting. " * 12
        mock_splitter.split_text.return_value = [text]
        pages = [_make_page(42, text, "DSM5.pdf")]
        result = chunker.chunk_pages(pages)

        chunk = result.chunks[0]
        assert chunk.page == 42
        assert chunk.source == "DSM5.pdf"

    def test_splitter_error_on_page_records_as_skipped(self):
        chunker, mock_splitter = _make_chunker()
        mock_splitter.split_text.side_effect = RuntimeError("stream error")
        long_text = "Long enough text to pass the minimum length threshold for splitting. " * 3
        pages = [_make_page(1, long_text, "doc.pdf")]
        result = chunker.chunk_pages(pages)

        assert result.skipped_pages == 1
        assert result.total_chunks == 0
        assert len(result.errors) == 1
        assert "page 1" in result.errors[0]

    def test_chunks_below_min_char_discarded(self):
        chunker, mock_splitter = _make_chunker(min_chunk_chars=100)
        # Splitter returns one tiny chunk, one good chunk
        mock_splitter.split_text.return_value = [
            "Too short.",                      # < 100 chars → discarded
            "A" * 150,                         # ≥ 100 chars → kept
        ]
        long_text = "X" * 300
        pages = [_make_page(1, long_text, "doc.pdf")]
        result = chunker.chunk_pages(pages)

        assert result.total_chunks == 1
        assert result.chunks[0].char_count == 150

    def test_source_hint_overrides_page_source(self):
        chunker, mock_splitter = _make_chunker()
        text = "Sufficient text content for the chunking operation here. " * 5
        mock_splitter.split_text.return_value = [text]
        pages = [{"page": 1, "text": text, "source": "original.pdf"}]
        result = chunker.chunk_pages(pages, source_hint="override.pdf")

        assert result.source == "override.pdf"

    def test_config_stored_in_chunks(self):
        chunker, mock_splitter = _make_chunker(chunk_size=600, chunk_overlap=100)
        text = "Text content for testing chunk configuration storage. " * 10
        mock_splitter.split_text.return_value = [text]
        pages = [_make_page(1, text, "doc.pdf")]
        result = chunker.chunk_pages(pages)

        chunk = result.chunks[0]
        assert chunk.chunk_size_config == 600
        assert chunk.overlap_config == 100


# ── chunk_document integration tests ─────────────────────────────────────────

class TestChunkDocument:
    def _make_extraction(self, pages_text: list[tuple[int, str]], source: str = "DSM5.pdf"):
        """Build a mock PDFExtractionResult."""
        mock_extraction = MagicMock()
        mock_extraction.source = source

        mock_pages = []
        for page_num, text in pages_text:
            p = MagicMock()
            p.page = page_num
            p.text = text
            p.source = source
            mock_pages.append(p)

        mock_extraction.usable_pages = MagicMock(return_value=mock_pages)
        return mock_extraction

    def test_delegates_to_chunk_pages(self):
        chunker, mock_splitter = _make_chunker()
        text = "Major depressive disorder diagnostic criteria content. " * 15
        mock_splitter.split_text.return_value = [text[:len(text)//2], text[len(text)//2:]]

        extraction = self._make_extraction([(1, text), (2, text)])
        result = chunker.chunk_document(extraction)

        assert result.status == ChunkingStatus.SUCCESS
        assert result.total_pages_in == 2
        assert result.total_chunks == 4

    def test_no_usable_pages_returns_empty(self):
        chunker, _ = _make_chunker()
        extraction = self._make_extraction([])
        result = chunker.chunk_document(extraction)
        assert result.status == ChunkingStatus.EMPTY

    def test_source_preserved(self):
        chunker, mock_splitter = _make_chunker()
        text = "CBT treatment protocol for anxiety disorders. " * 15
        mock_splitter.split_text.return_value = [text]
        extraction = self._make_extraction([(1, text)], source="CBT_Protocols.pdf")
        result = chunker.chunk_document(extraction)
        assert result.source == "CBT_Protocols.pdf"


# ── chunk_text tests ──────────────────────────────────────────────────────────

class TestChunkText:
    def test_returns_list_of_chunks(self):
        chunker, mock_splitter = _make_chunker()
        text = "Ad hoc text for direct chunking via the chunk_text method. " * 10
        mock_splitter.split_text.return_value = [text[:len(text)//2], text[len(text)//2:]]

        chunks = chunker.chunk_text(text, source="notes.pdf", page=3)
        assert len(chunks) == 2
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.page == 3 for c in chunks)

    def test_empty_text_returns_empty_list(self):
        chunker, _ = _make_chunker()
        chunks = chunker.chunk_text("", source="doc.pdf")
        assert chunks == []

    def test_source_in_chunk_ids(self):
        chunker, mock_splitter = _make_chunker()
        text = "Session note content for testing chunk text method. " * 8
        mock_splitter.split_text.return_value = [text]
        chunks = chunker.chunk_text(text, source="session_notes.pdf")
        assert "session_notes" in chunks[0].chunk_id


# ── chunk_batch tests ─────────────────────────────────────────────────────────

class TestChunkBatch:
    def _make_extraction(self, source: str, text: str):
        mock_e = MagicMock()
        mock_e.source = source
        p = MagicMock()
        p.page = 1
        p.text = text
        p.source = source
        mock_e.usable_pages = MagicMock(return_value=[p])
        return mock_e

    def test_processes_all_documents(self):
        chunker, mock_splitter = _make_chunker()
        text = "Clinical content for batch processing tests. " * 15
        mock_splitter.split_text.return_value = [text]
        extractions = [
            self._make_extraction("doc1.pdf", text),
            self._make_extraction("doc2.pdf", text),
            self._make_extraction("doc3.pdf", text),
        ]
        results = chunker.chunk_batch(extractions)
        assert len(results) == 3
        assert all(r.status == ChunkingStatus.SUCCESS for r in results)

    def test_skip_errors_true_continues(self):
        chunker, mock_splitter = _make_chunker()
        text = "Content for skip errors batch test processing. " * 15
        call_count = [0]

        def split_side_effect(t):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("splitter crash")
            return [t]

        mock_splitter.split_text.side_effect = split_side_effect
        extractions = [
            self._make_extraction("bad.pdf", text),
            self._make_extraction("good.pdf", text),
        ]
        results = chunker.chunk_batch(extractions, skip_errors=True)
        assert len(results) == 2
        # First may be EMPTY/FAILED, second should succeed
        assert results[1].status == ChunkingStatus.SUCCESS

    def test_skip_errors_false_raises(self):
        chunker, _ = _make_chunker()
        # Extraction that raises on usable_pages()
        bad = MagicMock()
        bad.source = "bad.pdf"
        bad.usable_pages = MagicMock(side_effect=RuntimeError("boom"))

        with pytest.raises(RuntimeError, match="boom"):
            chunker.chunk_batch([bad], skip_errors=False)


# ── Async tests ───────────────────────────────────────────────────────────────

class TestAsyncChunker:
    def test_achunk_document_returns_result(self):
        chunker, mock_splitter = _make_chunker()
        text = "Async chunking test with sufficient clinical content. " * 15
        mock_splitter.split_text.return_value = [text]

        mock_e = MagicMock()
        mock_e.source = "DSM5.pdf"
        p = MagicMock()
        p.page = 1; p.text = text; p.source = "DSM5.pdf"
        mock_e.usable_pages = MagicMock(return_value=[p])

        result = asyncio.run(chunker.achunk_document(mock_e))
        assert isinstance(result, ChunkingResult)
        assert result.status == ChunkingStatus.SUCCESS

    def test_achunk_pages_returns_result(self):
        chunker, mock_splitter = _make_chunker()
        text = "Async page chunking test content for clinical documents. " * 10
        mock_splitter.split_text.return_value = [text]
        pages = [_make_page(1, text)]
        result = asyncio.run(chunker.achunk_pages(pages, source_hint="test.pdf"))
        assert isinstance(result, ChunkingResult)


# ── ChromaDB integration shape tests ─────────────────────────────────────────

class TestChromaDBOutput:
    """Verify output shapes are correct for ChromaDB ingestion."""

    def test_batch_ids_are_strings(self):
        chunks = [
            Chunk(chunk_id=f"doc__p0001__c{i:04d}", text="A" * 80,
                  page=1, source="doc.pdf", chunk_index=i)
            for i in range(5)
        ]
        result = ChunkingResult(source="doc.pdf", status=ChunkingStatus.SUCCESS, chunks=chunks)
        batch = result.to_chromadb_batch()
        assert all(isinstance(i, str) for i in batch["ids"])

    def test_batch_documents_are_strings(self):
        chunks = [
            Chunk(chunk_id=f"doc__p0001__c{i:04d}", text="B" * 80,
                  page=1, source="doc.pdf", chunk_index=i)
            for i in range(3)
        ]
        result = ChunkingResult(source="doc.pdf", status=ChunkingStatus.SUCCESS, chunks=chunks)
        batch = result.to_chromadb_batch()
        assert all(isinstance(d, str) for d in batch["documents"])

    def test_batch_metadatas_are_dicts(self):
        chunks = [
            Chunk(chunk_id=f"doc__p0001__c{i:04d}", text="C" * 80,
                  page=1, source="doc.pdf", chunk_index=i)
            for i in range(3)
        ]
        result = ChunkingResult(source="doc.pdf", status=ChunkingStatus.SUCCESS, chunks=chunks)
        batch = result.to_chromadb_batch()
        assert all(isinstance(m, dict) for m in batch["metadatas"])

    def test_batch_lengths_consistent(self):
        chunks = [
            Chunk(chunk_id=f"doc__p0001__c{i:04d}", text="D" * 80,
                  page=1, source="doc.pdf", chunk_index=i)
            for i in range(7)
        ]
        result = ChunkingResult(source="doc.pdf", status=ChunkingStatus.SUCCESS, chunks=chunks)
        batch = result.to_chromadb_batch()
        assert len(batch["ids"]) == len(batch["documents"]) == len(batch["metadatas"]) == 7

    def test_metadata_contains_required_fields(self):
        c = Chunk(chunk_id="doc__p0001__c0000", text="E" * 80, page=1, source="doc.pdf", chunk_index=0)
        meta = c.to_chromadb_dict()["metadata"]
        required = {"source", "page", "chunk_index", "char_count", "word_count"}
        assert required.issubset(set(meta.keys()))
