"""
ingestion/processors/models.py
────────────────────────────────
Pydantic models for chunked document output.

These sit one layer above PageRecord (from loaders/models.py) in the
ingestion pipeline:

    PDF file
      └─► PDFLoader  ──► PDFExtractionResult  (list of PageRecord)
            └─► Chunker   ──► ChunkingResult      (list of Chunk)
                  └─► VectorStore (ChromaDB)

Keeping chunk models separate from loader models means:
  - The vector store and RAG layer depend only on this contract.
  - Chunking strategy can change (size, splitter type) without affecting
    the loader or the downstream RAG interface.
  - FastAPI can expose Chunk directly as a response schema.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Enums ─────────────────────────────────────────────────────────────────────

class ChunkingStatus(str, Enum):
    """Overall chunking outcome for a document."""
    SUCCESS = "success"    # at least one chunk produced
    EMPTY   = "empty"      # document had no usable text
    FAILED  = "failed"     # unrecoverable error during chunking


# ── Core chunk model ──────────────────────────────────────────────────────────

class Chunk(BaseModel):
    """
    A single text chunk ready for embedding and vector storage.

    This is the canonical unit consumed by the embedding layer and
    ChromaDB. Every field required for retrieval, display, and audit
    is embedded in the chunk itself — no separate lookup needed.

    Attributes:
        chunk_id:   Globally unique identifier for this chunk.
                    Format: ``{source_stem}__p{page:04d}__c{index:04d}``
                    e.g.  ``DSM5__p0012__c0003``
                    Deterministic — re-running ingestion on the same
                    source produces the same IDs, enabling idempotent upserts.

        text:       The chunk's text content. Never empty.

        page:       1-based page number from the originating PDF.
                    For multi-page chunks this is the page where the chunk
                    *starts*.

        source:     Filename of the originating PDF (e.g. ``DSM5.pdf``).
                    Full paths are not stored — only the filename — to avoid
                    leaking server filesystem structure.

        chunk_index: 0-based position of this chunk within the page.
                     Combined with ``page``, uniquely identifies the chunk's
                     position in the source document.

        char_count: Character count of ``text``. Auto-computed.

        word_count: Approximate word count. Auto-computed.

        chunk_size_config:  The chunk_size used when this chunk was created.
                            Stored so the vector collection can be audited
                            for consistency.

        overlap_config:     The chunk_overlap used. Same rationale.
    """

    chunk_id: str = Field(
        ...,
        description="Deterministic unique identifier for this chunk",
        examples=["DSM5__p0012__c0003"],
    )
    text: str = Field(..., min_length=1, description="Chunk text content")
    page: int = Field(..., ge=1, description="1-based source page number")
    source: str = Field(..., description="Source PDF filename")

    # Position metadata
    chunk_index: int = Field(
        ..., ge=0, description="0-based index of chunk within its source page"
    )

    # Auto-computed
    char_count: int = Field(default=0, ge=0)
    word_count: int = Field(default=0, ge=0)

    # Provenance — which splitter config produced this chunk
    chunk_size_config: int = Field(default=800, ge=1)
    overlap_config: int = Field(default=150, ge=0)

    @model_validator(mode="after")
    def compute_counts(self) -> "Chunk":
        object.__setattr__(self, "char_count", len(self.text))
        object.__setattr__(self, "word_count", len(self.text.split()))
        return self

    @field_validator("source")
    @classmethod
    def filename_only(cls, v: str) -> str:
        """Store only the filename component, never a full path."""
        from pathlib import Path
        return Path(v).name

    @field_validator("text")
    @classmethod
    def strip_text(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Chunk text must not be empty or whitespace-only")
        return stripped

    def to_chromadb_dict(self) -> dict:
        """
        Serialise to the three-part format expected by ChromaDB's
        ``collection.add()`` / ``collection.upsert()`` API.

        Returns:
            {
                "id":        chunk_id,
                "document":  text,
                "metadata":  {source, page, chunk_index, ...}
            }
        """
        return {
            "id": self.chunk_id,
            "document": self.text,
            "metadata": {
                "source": self.source,
                "page": self.page,
                "chunk_index": self.chunk_index,
                "char_count": self.char_count,
                "word_count": self.word_count,
                "chunk_size_config": self.chunk_size_config,
                "overlap_config": self.overlap_config,
            },
        }

    def to_dict(self) -> dict:
        """
        Canonical output format as specified in the project requirements.

        Returns:
            {"chunk_id": "...", "text": "...", "page": 12, "source": "DSM5.pdf"}
        """
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "page": self.page,
            "source": self.source,
        }

    model_config = {"frozen": True}


# ── Document-level chunking result ────────────────────────────────────────────

class ChunkingResult(BaseModel):
    """
    Complete chunking result for one source document.

    Produced by ``Chunker.chunk_document()`` and consumed by the
    ingestion pipeline before being handed to the vector store.

    Attributes:
        source:          Filename of the source PDF.
        status:          Overall chunking outcome.
        chunks:          All chunks produced. Empty on FAILED/EMPTY.
        total_pages_in:  How many pages were fed into the chunker.
        total_chunks:    Total chunks produced. Auto-computed.
        total_chars:     Sum of all chunk character counts. Auto-computed.
        skipped_pages:   Pages that had no usable text (not chunked).
        chunked_at:      UTC timestamp.
        errors:          Any non-fatal errors encountered.
    """

    source: str
    status: ChunkingStatus
    chunks: list[Chunk] = Field(default_factory=list)

    total_pages_in: int = Field(default=0, ge=0)
    total_chunks: int = Field(default=0, ge=0)
    total_chars: int = Field(default=0, ge=0)
    skipped_pages: int = Field(default=0, ge=0)

    chunked_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    errors: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def compute_aggregates(self) -> "ChunkingResult":
        self.total_chunks = len(self.chunks)
        self.total_chars = sum(c.char_count for c in self.chunks)
        return self

    def to_chromadb_batch(self) -> dict[str, list]:
        """
        Flatten all chunks into the batched format for a single
        ``collection.add()`` / ``collection.upsert()`` call.

        Returns:
            {
                "ids":        [...],
                "documents":  [...],
                "metadatas":  [...]
            }
        """
        if not self.chunks:
            return {"ids": [], "documents": [], "metadatas": []}

        ids, documents, metadatas = [], [], []
        for c in self.chunks:
            d = c.to_chromadb_dict()
            ids.append(d["id"])
            documents.append(d["document"])
            metadatas.append(d["metadata"])

        return {"ids": ids, "documents": documents, "metadatas": metadatas}

    def to_dicts(self) -> list[dict]:
        """Return the canonical list-of-dicts output format."""
        return [c.to_dict() for c in self.chunks]

    model_config = {"frozen": False}
