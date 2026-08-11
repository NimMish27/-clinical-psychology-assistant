"""
api/schemas/models.py
──────────────────────
Pydantic request and response models for the Clinical Psychology Assistant API.

All public API contracts are defined here — no raw dicts cross the API
boundary. This enables:
  - Automatic OpenAPI / Swagger docs generation
  - Request validation before handlers run
  - Consistent error payloads across all endpoints

Schema versioning convention:
  If a breaking change is needed, add a V2 suffix and keep V1 for
  backwards compatibility. Never mutate an existing schema in a way
  that breaks existing clients.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# Curated clinical frameworks with their own knowledge bases.
FrameworkName = Literal["cbt", "act", "humanistic"]


# ── Shared sub-models ──────────────────────────────────────────────────────────

class SourceReference(BaseModel):
    """
    A single source chunk referenced in a chat answer.

    Returned in the ``sources`` list of every ChatResponse so the
    clinician can trace the answer back to its originating document.

    Attributes:
        text:   The exact chunk text that contributed to the answer.
        source: PDF filename (e.g. ``"DSM5.pdf"``).
        page:   1-based page number in the source document.
        score:  Cosine similarity score in [0, 1].
    """
    text: str = Field(..., description="Chunk text used to generate the answer")
    source: str = Field(..., description="Source PDF filename")
    page: int = Field(..., ge=1, description="1-based page number")
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score [0, 1]")

    model_config = {"frozen": True}


class ErrorDetail(BaseModel):
    """Structured error payload returned on 4xx / 5xx responses."""
    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error description")
    details: dict[str, Any] | None = Field(None, description="Optional structured context")

    model_config = {"frozen": True}


# ── /chat ─────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """
    Request body for ``POST /chat``.

    Attributes:
        query:          The clinician's natural language question.
        session_id:     Optional session identifier for conversation continuity.
                        If omitted, each request is stateless.
        n_sources:      Number of source chunks to retrieve and include.
                        Defaults to RAG_RERANK_TOP_K from settings (4).
        source_filter:  Restrict retrieval to a specific source document.
                        Example: ``"DSM5.pdf"``
        page_range:     Restrict retrieval to a page range ``[start, end]``.
        framework:      Restrict retrieval to a curated clinical framework's
                        knowledge base (``cbt``, ``act``, ``humanistic``).
                        Defaults to the general clinical knowledge base.
        include_user_documents: When True, also retrieve from the user's
                        uploaded documents collection and merge + rerank the
                        results alongside the curated knowledge.
        stream:         Reserved for future streaming support. Currently ignored.
    """
    query: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Clinical question to answer",
        examples=["What are the DSM-5 diagnostic criteria for Major Depressive Disorder?"],
    )
    session_id: str | None = Field(
        None,
        max_length=128,
        description="Optional session ID for conversation continuity",
        examples=["sess-dr-jones-20241201"],
    )
    n_sources: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Number of source chunks to retrieve",
    )
    source_filter: str | None = Field(
        None,
        description="Restrict retrieval to this source filename",
        examples=["DSM5.pdf"],
    )
    page_range: list[int] | None = Field(
        None,
        min_length=2,
        max_length=2,
        description="Inclusive [start, end] page range for retrieval",
        examples=[[50, 100]],
    )
    framework: FrameworkName | None = Field(
        None,
        description="Curated clinical framework to scope retrieval to",
        examples=["act"],
    )
    include_user_documents: bool = Field(
        False,
        description="Merge and rerank results from the user's uploaded documents",
    )
    stream: bool = Field(
        False,
        description="Reserved for future streaming support",
    )

    @field_validator("query")
    @classmethod
    def query_not_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be empty or whitespace-only.")
        return v.strip()

    @field_validator("page_range")
    @classmethod
    def validate_page_range(cls, v: list[int] | None) -> list[int] | None:
        if v is not None:
            if v[0] < 1:
                raise ValueError("page_range start must be \u2265 1.")
            if v[0] > v[1]:
                raise ValueError("page_range start must be \u2264 end.")
        return v


class ChatResponse(BaseModel):
    """
    Response body for ``POST /chat``.

    Attributes:
        answer:         Generated clinical answer from the LLM.
        sources:        Source chunks used to generate the answer,
                        sorted by descending relevance score.
        session_id:     Echoed from the request, or None if stateless.
        model:          Ollama model name used for generation.
        retrieval_ms:   Time spent in vector search (ms).
        generation_ms:  Time spent in LLM generation (ms).
        total_ms:       Total request processing time (ms).
        created_at:     UTC timestamp of the response.
    """
    answer: str = Field(..., description="Generated clinical answer")
    sources: list[SourceReference] = Field(
        default_factory=list,
        description="Source chunks used, sorted by relevance",
    )
    session_id: str | None = Field(None)
    model: str = Field(..., description="LLM model used for generation")
    retrieval_ms: float = Field(..., ge=0.0)
    generation_ms: float = Field(..., ge=0.0)
    total_ms: float = Field(..., ge=0.0)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    model_config = {"frozen": True}


# ── /ingest ───────────────────────────────────────────────────────────────────

class IngestStatus(str, Enum):
    """Outcome of an ingestion job."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED  = "failed"


class FileIngestResult(BaseModel):
    """Per-file result within an ingestion response."""
    filename: str
    status: IngestStatus
    pages_extracted: int = Field(default=0, ge=0)
    chunks_created: int = Field(default=0, ge=0)
    chunks_embedded: int = Field(default=0, ge=0)
    chunks_stored: int = Field(default=0, ge=0)
    error: str | None = None

    model_config = {"frozen": True}


class IngestResponse(BaseModel):
    """
    Response body for ``POST /ingest``.

    Attributes:
        status:         Overall job status.
        total_files:    Number of files submitted.
        succeeded:      Files fully processed.
        failed:         Files that encountered errors.
        total_chunks:   Total chunks stored across all files.
        files:          Per-file breakdown.
        elapsed_ms:     Total ingestion time (ms).
        collection:     ChromaDB collection that was populated.
    """
    status: IngestStatus
    total_files: int = Field(..., ge=0)
    succeeded: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    total_chunks: int = Field(default=0, ge=0)
    files: list[FileIngestResult] = Field(default_factory=list)
    elapsed_ms: float = Field(..., ge=0.0)
    collection: str = Field(..., description="ChromaDB collection populated")

    model_config = {"frozen": True}


# ── /health ───────────────────────────────────────────────────────────────────

class ServiceStatus(str, Enum):
    """Status of an individual service dependency."""
    OK       = "ok"
    DEGRADED = "degraded"
    DOWN     = "down"


class DependencyHealth(BaseModel):
    """Health status for a single service dependency."""
    status: ServiceStatus
    latency_ms: float | None = None
    detail: str | None = None

    model_config = {"frozen": True}


class HealthResponse(BaseModel):
    """
    Response body for ``GET /health``.

    HTTP status code reflects overall health:
      200 \u2192 all dependencies OK
      206 \u2192 at least one dependency degraded (PARTIAL_CONTENT)
      503 \u2192 critical dependency down (SERVICE_UNAVAILABLE)

    Attributes:
        status:       Overall system health.
        version:      Application version from settings.
        environment:  Deployment environment (development / staging / production).
        uptime_s:     Seconds since the API server started.
        dependencies: Health of each external dependency.
        checked_at:   UTC timestamp of the health check.
    """
    status: ServiceStatus
    version: str
    environment: str
    uptime_s: float
    dependencies: dict[str, DependencyHealth] = Field(default_factory=dict)
    checked_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    model_config = {"frozen": True}


# ── /clinical/analyze ─────────────────────────────────────────────────────────

class ClinicalAnalyzeRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=3,
        max_length=50000,
        description="Clinical input text — single statement, symptom list, or case study",
    )
    input_type: str | None = Field(
        None,
        description='Override: "single_statement", "symptom_list", "case_study"',
    )


class ClinicalAnalyzeResponse(BaseModel):
    input_type: str = Field(default="", description="Detected input type")
    understanding: dict[str, Any] = Field(
        default_factory=dict,
        description="Case understanding extraction result",
    )
    queries: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Generated retrieval queries",
    )
    evidence: dict[str, Any] = Field(
        default_factory=dict,
        description="Evidence synthesis result",
    )
    formulation: dict[str, Any] = Field(
        default_factory=dict,
        description="Clinical formulation result",
    )
    missing_info: dict[str, Any] = Field(
        default_factory=dict,
        description="Missing information detection result",
    )
    therapeutic_planning: dict[str, Any] = Field(
        default_factory=dict,
        description="Therapeutic planning result",
    )
    response: dict[str, Any] = Field(
        default_factory=dict,
        description="Final clinical response markdown",
    )
    safety_report: dict[str, Any] = Field(
        default_factory=dict,
        description="Safety validation report with issues found and revision status",
    )
    errors: dict[str, str] = Field(
        default_factory=dict,
        description="Node-level errors keyed by stage name",
    )
    elapsed_ms: float = Field(default=0.0, ge=0.0)

    model_config = {"frozen": True}


# ── /kb ───────────────────────────────────────────────────────────────────────

class KbCollectionInfo(BaseModel):
    """Status of one curated clinical framework's knowledge base."""

    framework: str = Field(..., description="Framework slug, e.g. 'act'")
    built: bool = Field(False, description="True when a built collection is active")
    collection: str | None = Field(None, description="Active ChromaDB collection name")
    index_version: str | None = Field(None, description="Index version, e.g. '2026-08'")
    chunk_count: int | None = Field(None, ge=0, description="Stored chunk count")
    built_at: str | None = Field(None, description="ISO timestamp of the last build")
    books: list[str] = Field(default_factory=list, description="Curated book slugs")


class KbCollectionsResponse(BaseModel):
    """Response body for ``GET /kb/collections``."""

    collections: list[KbCollectionInfo]


class KbRebuildRequest(BaseModel):
    """Request body for ``POST /kb/rebuild``."""

    framework: FrameworkName | None = Field(
        None,
        description="Restrict the rebuild to one framework. None rebuilds all curated sources.",
    )


class KbRebuildResponse(BaseModel):
    """Response body for ``POST /kb/rebuild``."""

    status: Literal["ok", "failed"] = Field(
        ...,
        description="ok when the build+index succeeded, failed when validation errored",
    )
    built_at: str | None = None
    source_files: list[str] = Field(default_factory=list)
    parents: int = 0
    children: int = 0
    total_tokens: int = 0
    errors: list[str] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)
    collections: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Active framework collections after indexing",
    )


class KbDocumentSummary(BaseModel):
    """One uploaded document in a user collection."""

    source: str = Field(..., description="Source filename")
    chunks: int = Field(0, ge=0, description="Number of stored chunks")
    user: str | None = Field(None, description="Uploader tag")
    project: str | None = Field(None, description="Project tag")
    document_type: str | None = Field(None, description="Document type tag")
    upload_date: str | None = Field(None, description="Upload date (ISO)")


class KbDocumentsResponse(BaseModel):
    """Response body for ``GET /kb/documents``."""

    collection: str = Field(..., description="Collection that was listed")
    sources: list[KbDocumentSummary] = Field(default_factory=list)


class KbDeleteResponse(BaseModel):
    """Response body for ``DELETE /kb/documents``."""

    collection: str = Field(..., description="Collection the source was removed from")
    source: str = Field(..., description="Source filename removed")
    deleted: int = Field(0, ge=0, description="Number of chunks deleted")


# ── /evidence/search ──────────────────────────────────────────────────────────

class EvidenceSearchRequest(BaseModel):
    """Request body for ``POST /evidence/search``.

    Retrieval-only literature search (no LLM generation). When ``framework``
    is set, search is scoped to that curated knowledge base and the result
    carries the framework evidence bundle (synthesis, key findings, gaps,
    confidence). Otherwise the general clinical knowledge base is searched.
    """

    query: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Natural-language literature search query",
        examples=["Acceptance and values work for chronic pain"],
    )
    framework: FrameworkName | None = Field(
        None,
        description="Curated clinical framework to scope the search to",
        examples=["act"],
    )
    include_user_documents: bool = Field(
        False,
        description="Merge and rerank results from the user's uploaded documents",
    )
    n_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of evidence items to return",
    )
    chapter: str | None = Field(None, max_length=128, description="Chapter filter")
    topic: str | None = Field(None, max_length=128, description="Topic filter")
    intervention: str | None = Field(None, max_length=128, description="Intervention filter")
    symptom: str | None = Field(None, max_length=128, description="Symptom filter")
    population: str | None = Field(None, max_length=128, description="Population filter")

    @field_validator("query")
    @classmethod
    def query_not_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be empty or whitespace-only.")
        return v.strip()


class EvidenceItem(BaseModel):
    """One retrieved evidence item in an evidence search response."""

    source: str = Field(..., description="Source document name")
    content: str = Field(..., description="Evidence text content")
    quality_rating: float | None = Field(None, ge=0.0, le=1.0, description="Relevance score [0, 1]")
    framework: str | None = Field(None, description="Framework the item belongs to")
    chapter: str | None = Field(None, description="Chapter within the source document")
    topic: str | None = Field(None, description="Clinical topic")
    intervention: str | None = Field(None, description="Intervention described")
    symptom: str | None = Field(None, description="Symptom or presentation")
    population: str | None = Field(None, description="Target population")
    page: int = Field(0, description="Page number in the source document")
    chunk_id: str = Field("", description="Chunk identifier in the vector store")
    is_user_document: bool = Field(
        False,
        description="True when the item came from the user's uploaded documents",
    )


class EvidenceSearchResponse(BaseModel):
    """Response body for ``POST /evidence/search``."""

    query: str = Field(..., description="Echoed search query")
    framework: str | None = Field(None, description="Framework the search was scoped to")
    sources: list[EvidenceItem] = Field(
        default_factory=list,
        description="Retrieved evidence, sorted by descending relevance",
    )
    synthesis: str | None = Field(
        None,
        description="Integrated evidence summary (framework searches only)",
    )
    key_findings: list[str] = Field(default_factory=list, description="Principal findings")
    evidence_gaps: list[str] = Field(
        default_factory=list,
        description="Areas where evidence is lacking",
    )
    confidence: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Overall confidence in the evidence set",
    )
    total_candidates: int = Field(0, description="Candidates considered before ranking")
    elapsed_ms: float = Field(0.0, description="Wall-clock retrieval time (ms)")
    user_documents_included: bool = Field(False, description="Whether user documents were merged")
    user_documents_returned: int = Field(
        0,
        description="How many user-document items survived ranking",
    )
