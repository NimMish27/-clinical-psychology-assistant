"""
ingestion/processors/chunker.py
────────────────────────────────
Clinical document chunker using LangChain's RecursiveCharacterTextSplitter.

Design decisions:
  ┌─────────────────────────────────────────────────────────────────────┐
  │ Why RecursiveCharacterTextSplitter?                                 │
  │                                                                     │
  │ Clinical texts (DSM-5, CBT manuals, ICD-11) are structured with    │
  │ headings, numbered criteria, and multi-sentence paragraphs.         │
  │ Recursive splitting tries separators in order:                      │
  │   \\n\\n → \\n → ". " → " " → ""                                    │
  │ This preserves paragraph boundaries first, then sentence            │
  │ boundaries, and only falls back to word or character splits as a    │
  │ last resort — keeping clinical criteria intact as a unit.           │
  │                                                                     │
  │ A fixed TokenTextSplitter would silently break "Criterion A: (1)"  │
  │ mid-sentence; semantic splitters add GPU overhead not justified     │
  │ at ingestion time.                                                   │
  └─────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────┐
  │ Why chunk_size=800, overlap=150?                                    │
  │                                                                     │
  │ • BAAI/bge-large-en-v1.5 has a 512-token context window.           │
  │   800 chars ≈ 150–180 tokens — well within the model limit and     │
  │   leaves room for the query prefix added at retrieval time.         │
  │ • 150-char overlap (≈ 1–2 sentences) ensures a sentence that spans │
  │   a chunk boundary appears in full in at least one chunk.           │
  │ • Both values are configurable — override in .env or at runtime.   │
  └─────────────────────────────────────────────────────────────────────┘

Usage (from PageRecord list produced by PDFLoader):
    from ingestion.processors.chunker import Chunker
    from ingestion.loaders import PDFLoader

    loader  = PDFLoader()
    chunker = Chunker()

    extraction = loader.load("data/raw/DSM5.pdf")
    result     = chunker.chunk_document(extraction)

    for chunk_dict in result.to_dicts():
        print(chunk_dict)
        # {"chunk_id": "DSM5__p0012__c0003", "text": "...", "page": 12, "source": "DSM5.pdf"}

Usage (from raw page dicts — e.g. from an external source):
    pages = [{"page": 1, "text": "...", "source": "manual.pdf"}, ...]
    result = chunker.chunk_pages(pages)

Usage (async — for FastAPI endpoints):
    result = await chunker.achunk_document(extraction)
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
from pathlib import Path
from typing import Sequence

from ingestion.processors.models import Chunk, ChunkingResult, ChunkingStatus

# LangChain import — guarded so the rest of the module loads cleanly
# even in environments where langchain is not installed (e.g. type-check runs).
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    _LC_AVAILABLE = True
except ImportError:
    try:
        # Older langchain versions bundle the splitter here
        from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore[no-redef]
        _LC_AVAILABLE = True
    except ImportError:
        _LC_AVAILABLE = False
        RecursiveCharacterTextSplitter = None  # type: ignore[assignment,misc]

from app_logging.logger import get_logger

_log = get_logger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────────

# Default separators ordered from coarsest to finest.
# Clinical texts prioritise paragraph and sentence boundaries.
_CLINICAL_SEPARATORS: list[str] = [
    "\n\n",    # paragraph break (highest priority)
    "\n",      # line break
    ". ",      # sentence boundary
    "? ",      # question (rare but present in interview guides)
    "! ",      # exclamation
    "; ",      # clause boundary (common in DSM criteria lists)
    ", ",      # phrase boundary
    " ",       # word boundary
    "",        # character boundary (last resort)
]

# Minimum text length (chars) to bother chunking a page.
# Pages shorter than this are almost certainly headers, footers, or
# image captions with no clinical substance.
_MIN_PAGE_TEXT_LENGTH: int = 30


# ── Chunk ID generation ────────────────────────────────────────────────────────

def _make_chunk_id(source: str, page: int, chunk_index: int) -> str:
    """
    Build a deterministic, human-readable chunk ID.

    Format:  ``{source_stem}__p{page:04d}__c{chunk_index:04d}``
    Example: ``DSM5__p0012__c0003``

    The stem is sanitised (spaces and punctuation → underscores) so the
    ID is safe for use as a ChromaDB document ID, a filename, and a URL
    path segment without escaping.

    Deterministic property: given the same (source, page, chunk_index)
    triple the ID is always the same, enabling idempotent upserts into
    ChromaDB.
    """
    stem = Path(source).stem
    # Sanitise: keep alphanumeric and hyphens, replace everything else
    stem_clean = re.sub(r"[^a-zA-Z0-9\-]", "_", stem)
    # Collapse multiple underscores
    stem_clean = re.sub(r"_+", "_", stem_clean).strip("_")
    return f"{stem_clean}__p{page:04d}__c{chunk_index:04d}"


def _make_content_hash(text: str, source: str, page: int) -> str:
    """
    SHA-256 content hash for deduplication (first 12 hex chars).
    Not used as the primary ID but stored in metadata for auditing.
    """
    payload = f"{source}:{page}:{text}"
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


# ── Main class ────────────────────────────────────────────────────────────────

class Chunker:
    """
    Splits clinical document pages into overlapping text chunks.

    The class is stateless beyond its configuration — the same instance
    can process many documents concurrently without synchronisation.

    Args:
        chunk_size:  Target character count per chunk. Default: 800.
                     Must be > overlap.
        chunk_overlap: Characters of overlap between consecutive chunks.
                       Default: 150.
        separators:  Custom separator list. Defaults to clinical-optimised
                     list (paragraph → sentence → word → char).
        min_chunk_chars: Chunks shorter than this are discarded as noise.
                         Default: 50.
        strip_whitespace: Whether to strip leading/trailing whitespace from
                          each chunk before creating the Chunk object.
                          Default: True.

    Raises:
        ImportError:  If LangChain is not installed.
        ValueError:   If chunk_size ≤ chunk_overlap.
    """

    def __init__(
        self,
        *,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        separators: list[str] | None = None,
        min_chunk_chars: int = 50,
        strip_whitespace: bool = True,
    ) -> None:
        if not _LC_AVAILABLE:
            raise ImportError(
                "LangChain is required for chunking. "
                "Install it with: pip install langchain-text-splitters"
            )
        if chunk_size <= chunk_overlap:
            raise ValueError(
                f"chunk_size ({chunk_size}) must be greater than "
                f"chunk_overlap ({chunk_overlap})."
            )
        if min_chunk_chars < 1:
            raise ValueError("min_chunk_chars must be at least 1.")

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._min_chunk_chars = min_chunk_chars
        self._strip_whitespace = strip_whitespace

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or _CLINICAL_SEPARATORS,
            length_function=len,
            is_separator_regex=False,
            keep_separator=False,
        )

        _log.info(
            "chunker.initialised",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_chars=min_chunk_chars,
        )

    # ── Public synchronous interface ──────────────────────────────────────────

    def chunk_document(self, extraction_result: object) -> ChunkingResult:
        """
        Chunk a complete ``PDFExtractionResult`` (from PDFLoader).

        Accepts the PDFExtractionResult produced by PDFLoader and iterates
        over its usable pages. This is the primary entry point for the
        ingestion pipeline.

        Args:
            extraction_result: A ``PDFExtractionResult`` instance. Typed as
                ``object`` to avoid a hard import of the loader models here —
                duck-typed against ``.source``, ``.usable_pages()``.

        Returns:
            ChunkingResult with all chunks and aggregate statistics.
        """
        # Duck-type the extraction result to avoid circular imports
        source: str = getattr(extraction_result, "source", "unknown.pdf")
        usable_pages = getattr(extraction_result, "usable_pages", lambda: [])()

        page_dicts = [
            {"page": p.page, "text": p.text, "source": p.source}
            for p in usable_pages
        ]

        return self.chunk_pages(page_dicts, source_hint=source)

    def chunk_pages(
        self,
        pages: Sequence[dict],
        *,
        source_hint: str | None = None,
    ) -> ChunkingResult:
        """
        Chunk a sequence of page dictionaries.

        Each dict must have: ``page`` (int, 1-based), ``text`` (str),
        ``source`` (str).

        This is the lower-level entry point — useful when page data comes
        from a source other than PDFLoader (e.g. a DOCX loader, a test
        fixture, or a database).

        Args:
            pages:       Iterable of ``{"page": N, "text": "...", "source": "..."}``
                         dicts. Pages with empty text are silently skipped.
            source_hint: Override the source filename used in logging.
                         If None, inferred from the first page dict.

        Returns:
            ChunkingResult containing all produced Chunk objects.
        """
        if not pages:
            source = source_hint or "unknown.pdf"
            _log.warning("chunker.no_pages", source=source)
            return ChunkingResult(
                source=source,
                status=ChunkingStatus.EMPTY,
                total_pages_in=0,
            )

        source = source_hint or pages[0].get("source", "unknown.pdf")
        t_start = time.perf_counter()

        _log.info(
            "chunker.start",
            source=source,
            page_count=len(pages),
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )

        all_chunks: list[Chunk] = []
        skipped_pages: int = 0
        errors: list[str] = []

        for page_dict in pages:
            page_num: int = page_dict.get("page", 0)
            raw_text: str = page_dict.get("text", "")
            page_source: str = page_dict.get("source", source)

            chunks, skip, error = self._chunk_page(
                text=raw_text,
                page=page_num,
                source=page_source,
            )

            if skip:
                skipped_pages += 1
            if error:
                errors.append(error)

            all_chunks.extend(chunks)

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        status = (
            ChunkingStatus.SUCCESS if all_chunks
            else ChunkingStatus.EMPTY
        )

        result = ChunkingResult(
            source=source,
            status=status,
            chunks=all_chunks,
            total_pages_in=len(pages),
            skipped_pages=skipped_pages,
            errors=errors,
        )

        _log.info(
            "chunker.complete",
            source=source,
            status=status,
            total_pages_in=len(pages),
            total_chunks=result.total_chunks,
            skipped_pages=skipped_pages,
            total_chars=result.total_chars,
            elapsed_ms=round(elapsed_ms, 2),
        )

        return result

    def chunk_text(
        self,
        text: str,
        *,
        source: str,
        page: int = 1,
    ) -> list[Chunk]:
        """
        Chunk a single raw text string.

        Convenience method for ad-hoc chunking — e.g. splitting a
        web-scraped article or a note typed directly into the system.

        Args:
            text:   The raw text to split.
            source: Source identifier (used in chunk_id and metadata).
            page:   Page number to assign to all produced chunks. Default: 1.

        Returns:
            List of Chunk objects. Empty list if text is below minimum length.
        """
        chunks, _, _ = self._chunk_page(text=text, page=page, source=source)
        return chunks

    def chunk_batch(
        self,
        extractions: list[object],
        *,
        skip_errors: bool = True,
    ) -> list[ChunkingResult]:
        """
        Chunk multiple PDFExtractionResult objects in sequence.

        Args:
            extractions:  List of PDFExtractionResult objects.
            skip_errors:  If True, errors on individual documents are logged
                          and a FAILED ChunkingResult is appended. If False,
                          the first error raises immediately.

        Returns:
            List of ChunkingResult in the same order as extractions.
        """
        results: list[ChunkingResult] = []
        total = len(extractions)

        _log.info("chunker.batch_start", total_documents=total)

        for idx, extraction in enumerate(extractions, start=1):
            source = getattr(extraction, "source", f"document_{idx}")
            _log.debug("chunker.batch_progress", index=idx, total=total, source=source)

            try:
                results.append(self.chunk_document(extraction))
            except Exception as exc:
                if not skip_errors:
                    raise
                _log.error(
                    "chunker.batch_document_failed",
                    source=source,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                results.append(
                    ChunkingResult(
                        source=source,
                        status=ChunkingStatus.FAILED,
                        errors=[f"{type(exc).__name__}: {exc}"],
                    )
                )

        n_ok   = sum(1 for r in results if r.status == ChunkingStatus.SUCCESS)
        n_fail = sum(1 for r in results if r.status == ChunkingStatus.FAILED)
        _log.info(
            "chunker.batch_complete",
            total=total,
            success=n_ok,
            failed=n_fail,
            total_chunks=sum(r.total_chunks for r in results),
        )

        return results

    # ── Async interface ───────────────────────────────────────────────────────

    async def achunk_document(self, extraction_result: object) -> ChunkingResult:
        """
        Async wrapper around chunk_document() for FastAPI endpoints.

        Chunking is CPU-bound (string splitting), so we offload to a
        thread-pool executor to keep the event loop responsive.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.chunk_document, extraction_result)

    async def achunk_pages(
        self,
        pages: list[dict],
        *,
        source_hint: str | None = None,
    ) -> ChunkingResult:
        """Async wrapper around chunk_pages()."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.chunk_pages(pages, source_hint=source_hint),
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _chunk_page(
        self,
        text: str,
        page: int,
        source: str,
    ) -> tuple[list[Chunk], bool, str | None]:
        """
        Split a single page's text into Chunk objects.

        Args:
            text:   Raw page text.
            page:   1-based page number.
            source: Source filename.

        Returns:
            (chunks, was_skipped, error_message_or_None)
        """
        cleaned = self._preprocess(text)

        if len(cleaned) < _MIN_PAGE_TEXT_LENGTH:
            _log.debug(
                "chunker.page_skipped",
                source=source,
                page=page,
                reason="below minimum text length",
                char_count=len(cleaned),
                threshold=_MIN_PAGE_TEXT_LENGTH,
            )
            return [], True, None

        try:
            raw_chunks: list[str] = self._splitter.split_text(cleaned)
        except Exception as exc:
            msg = f"page {page}: splitter error — {type(exc).__name__}: {exc}"
            _log.error(
                "chunker.page_split_error",
                source=source,
                page=page,
                error=str(exc),
            )
            return [], True, msg

        chunks: list[Chunk] = []
        for idx, chunk_text in enumerate(raw_chunks):
            if self._strip_whitespace:
                chunk_text = chunk_text.strip()

            if len(chunk_text) < self._min_chunk_chars:
                _log.debug(
                    "chunker.chunk_discarded",
                    source=source,
                    page=page,
                    chunk_index=idx,
                    reason="below min_chunk_chars",
                    char_count=len(chunk_text),
                )
                continue

            chunk = Chunk(
                chunk_id=_make_chunk_id(source, page, idx),
                text=chunk_text,
                page=page,
                source=source,
                chunk_index=idx,
                chunk_size_config=self._chunk_size,
                overlap_config=self._chunk_overlap,
            )
            chunks.append(chunk)

        _log.debug(
            "chunker.page_done",
            source=source,
            page=page,
            raw_chunks=len(raw_chunks),
            kept_chunks=len(chunks),
        )

        return chunks, False, None

    @staticmethod
    def _preprocess(text: str) -> str:
        """
        Normalise raw page text before splitting.

        Steps (order matters):
          1. Strip leading/trailing whitespace.
          2. Normalise line endings to \\n.
          3. Remove Unicode replacement character (\\ufffd) — common PDF
             artefact from encoding issues in scanned clinical docs.
          4. Collapse runs of 3+ blank lines to a single blank line.
             Preserves intentional paragraph separation.
          5. Remove lines that are clearly page artefacts: pure page
             numbers, running headers that are ≤ 4 words and all-caps
             or purely numeric.

        We do NOT:
          - Lower-case (embeddings handle normalisation)
          - Remove hyphens (CBT-I, DSM-5, ICD-10 must be preserved)
          - Remove punctuation (clinical criteria rely on it)
        """
        if not text:
            return ""

        # 1. Strip outer whitespace
        text = text.strip()

        # 2. Normalise line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # 3. Remove replacement character
        text = text.replace("\ufffd", "")

        # 4. Collapse excessive blank lines (3+ → 2)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 5. Remove isolated page-number lines (standalone integers)
        text = re.sub(r"(?m)^\s*\d{1,4}\s*$", "", text)

        # 6. Final blank-line collapse after removals
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    # ── Configuration introspection ───────────────────────────────────────────

    @property
    def chunk_size(self) -> int:
        """Configured chunk size in characters."""
        return self._chunk_size

    @property
    def chunk_overlap(self) -> int:
        """Configured overlap in characters."""
        return self._chunk_overlap

    @property
    def config(self) -> dict:
        """
        Return the chunker's configuration as a plain dict.
        Useful for logging and for storing provenance in the vector store.
        """
        return {
            "chunk_size": self._chunk_size,
            "chunk_overlap": self._chunk_overlap,
            "min_chunk_chars": self._min_chunk_chars,
            "strip_whitespace": self._strip_whitespace,
        }

    def __repr__(self) -> str:
        return (
            f"Chunker(chunk_size={self._chunk_size}, "
            f"chunk_overlap={self._chunk_overlap})"
        )
