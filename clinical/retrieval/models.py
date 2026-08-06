"""
clinical/retrieval/models.py — Structured evidence output models for clinical retrieval.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CBTEvidenceFilter(BaseModel):
    """Typed metadata filter for CBT evidence retrieval.

    All filters are AND-ed together. Multi-value fields accept a list
    for ``$in`` matching (e.g., ``topic=["depression", "anxiety"]``).
    When a field is ``None``, no filter is applied for that field.
    """

    framework: str = Field(
        default="cbt",
        description="Framework slug to filter by",
    )
    chapter: str | list[str] | None = Field(
        default=None,
        description="Chapter or section within the source document",
    )
    topic: str | list[str] | None = Field(
        default=None,
        description="Clinical topic or subject area",
    )
    intervention: str | list[str] | None = Field(
        default=None,
        description="Therapeutic intervention or technique",
    )
    symptom: str | list[str] | None = Field(
        default=None,
        description="Symptom or clinical presentation",
    )
    population: str | list[str] | None = Field(
        default=None,
        description="Target population (e.g., adults, adolescents, veterans)",
    )

    def to_metadata_filter(self) -> dict | None:
        """Convert to the metadata_filter dict expected by the retriever layer.

        Returns ``None`` when no fields beyond ``framework`` are populated.
        """
        parts: dict = {}
        for field_name in ("framework", "chapter", "topic", "intervention", "symptom", "population"):
            val = getattr(self, field_name, None)
            if val is None:
                continue
            if isinstance(val, list):
                parts[field_name] = {"$in": val}
            else:
                parts[field_name] = val
        return parts or None


class CBTEvidenceItem(BaseModel):
    """A single structured evidence item retrieved from CBT literature.

    Extends ``RetrievedEvidence`` with metadata fields for transparency
    about where the evidence came from and what it covers.
    """

    source: str = Field(
        description="Source document name (PDF filename or citation)",
    )
    content: str = Field(
        min_length=10,
        description="The evidence text content",
    )
    relevance: str = Field(
        min_length=10,
        description="How this evidence applies to the clinical case or query",
    )
    quality_rating: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Relevance / similarity score from retrieval pipeline (0-1)",
    )
    framework: str = Field(
        default="cbt",
        description="Framework this evidence belongs to",
    )
    chapter: str | None = Field(
        default=None,
        description="Chapter or section within the source document",
    )
    topic: str | None = Field(
        default=None,
        description="Clinical topic this evidence addresses",
    )
    intervention: str | None = Field(
        default=None,
        description="Therapeutic intervention described",
    )
    symptom: str | None = Field(
        default=None,
        description="Symptom or presentation this evidence relates to",
    )
    population: str | None = Field(
        default=None,
        description="Target population for this evidence",
    )
    page: int = Field(
        default=0,
        description="Page number in the source document",
    )
    chunk_id: str = Field(
        default="",
        description="Unique identifier in the ChromaDB collection",
    )


class CBTEvidenceResult(BaseModel):
    """Complete result of a CBT evidence retrieval operation.

    Returned by ``CBTRetriever.retrieve()`` and compatible with the
    ``EvidenceResult`` model from ``clinical.framework.models``.
    """

    query: str = Field(
        description="The original search query",
    )
    retrieved: list[CBTEvidenceItem] = Field(
        description="Evidence items retrieved, sorted by relevance",
    )
    synthesis: str = Field(
        min_length=10,
        description="Integrated summary of the retrieved evidence",
    )
    key_findings: list[str] = Field(
        description="Principal findings from the evidence",
    )
    evidence_gaps: list[str] = Field(
        description="Areas where evidence is lacking or inconclusive",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Overall confidence in the evidence set (0-1)",
    )
    total_candidates: int = Field(
        default=0,
        description="Total candidates before threshold filtering",
    )
    elapsed_ms: float = Field(
        default=0.0,
        description="Wall-clock time for the retrieval operation",
    )

    def to_framework_evidence_result(self):
        """Convert to the ``EvidenceResult`` model used by framework plugins."""
        from clinical.framework.models import EvidenceResult, RetrievedEvidence

        return EvidenceResult(
            retrieved=[
                RetrievedEvidence(
                    source=item.source,
                    content=item.content,
                    relevance=item.relevance,
                    quality_rating=item.quality_rating,
                )
                for item in self.retrieved
            ],
            synthesis=self.synthesis,
            key_findings=self.key_findings,
            evidence_gaps=self.evidence_gaps,
            confidence=self.confidence,
        )


class HumanisticEvidenceFilter(BaseModel):
    """Typed metadata filter for Humanistic evidence retrieval.

    Defaults to the ``"humanistic"`` framework slug.
    """

    framework: str = Field(
        default="humanistic",
        description="Framework slug to filter by",
    )
    chapter: str | list[str] | None = Field(
        default=None,
        description="Chapter or section within the source document",
    )
    topic: str | list[str] | None = Field(
        default=None,
        description="Clinical topic or subject area",
    )
    intervention: str | list[str] | None = Field(
        default=None,
        description="Therapeutic intervention or technique",
    )
    symptom: str | list[str] | None = Field(
        default=None,
        description="Symptom or clinical presentation",
    )
    population: str | list[str] | None = Field(
        default=None,
        description="Target population (e.g., adults, adolescents, veterans)",
    )


class HumanisticEvidenceItem(BaseModel):
    """A single structured evidence item from Humanistic literature."""

    source: str = Field(description="Source document name (PDF filename or citation)")
    content: str = Field(min_length=10, description="The evidence text content")
    relevance: str = Field(min_length=10, description="How this evidence applies to the clinical case")
    quality_rating: float | None = Field(default=None, ge=0.0, le=1.0, description="Relevance score from retrieval pipeline")
    framework: str = Field(default="humanistic", description="Framework this evidence belongs to")
    chapter: str | None = Field(default=None, description="Chapter or section within the source document")
    topic: str | None = Field(default=None, description="Clinical topic this evidence addresses")
    intervention: str | None = Field(default=None, description="Therapeutic intervention described")
    symptom: str | None = Field(default=None, description="Symptom or presentation this evidence relates to")
    population: str | None = Field(default=None, description="Target population for this evidence")
    page: int = Field(default=0, description="Page number in the source document")
    chunk_id: str = Field(default="", description="Unique identifier in the ChromaDB collection")


class HumanisticEvidenceResult(BaseModel):
    """Complete result of a Humanistic evidence retrieval operation."""

    query: str = Field(description="The original search query")
    retrieved: list[HumanisticEvidenceItem] = Field(description="Evidence items retrieved, sorted by relevance")
    synthesis: str = Field(min_length=10, description="Integrated summary of the retrieved evidence")
    key_findings: list[str] = Field(description="Principal findings from the evidence")
    evidence_gaps: list[str] = Field(description="Areas where evidence is lacking or inconclusive")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Overall confidence in the evidence set")
    total_candidates: int = Field(default=0, description="Total candidates before threshold filtering")
    elapsed_ms: float = Field(default=0.0, description="Wall-clock time for the retrieval operation")

    def to_framework_evidence_result(self):
        """Convert to ``EvidenceResult`` used by framework plugins."""
        from clinical.framework.models import EvidenceResult, RetrievedEvidence

        return EvidenceResult(
            retrieved=[
                RetrievedEvidence(
                    source=item.source,
                    content=item.content,
                    relevance=item.relevance,
                    quality_rating=item.quality_rating,
                )
                for item in self.retrieved
            ],
            synthesis=self.synthesis,
            key_findings=self.key_findings,
            evidence_gaps=self.evidence_gaps,
            confidence=self.confidence,
        )
