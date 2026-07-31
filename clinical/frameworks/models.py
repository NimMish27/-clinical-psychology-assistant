from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class FrameworkContext(BaseModel):
    """Typed context passed to every framework plugin for analysis.

    Contains all data a framework may need to produce its analysis.
    The ``extra`` dict allows future extension without interface changes.
    """

    understanding: Any = Field(
        ..., description="CaseUnderstandingResult — structured case features",
    )
    evidence: Any | None = Field(
        None, description="EvidenceSynthesisResult — shared evidence synthesis",
    )
    raw_text: str = Field(
        default="", description="Original unprocessed case text",
    )
    retrieved_chunks: list[Any] = Field(
        default_factory=list,
        description="Retrieved evidence chunks from the RAG pipeline",
    )
    session_id: str | None = Field(
        None, description="Optional session identifier for multi-turn context",
    )
    llm: Any | None = Field(
        None, exclude=True,
        description="LLMService instance for LLM generation calls",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Extensibility field for future context data",
    )

    model_config = {"frozen": True}


class FrameworkResult(BaseModel):
    framework_name: str = Field(
        ..., min_length=1, description="Short name (e.g. 'CBT', 'ACT')",
    )
    framework_description: str = Field(
        ..., min_length=10, description="One-sentence description of the framework",
    )
    conceptualization: str = Field(
        ..., min_length=20,
        description=(
            "How this framework conceptualises the client's presentation — "
            "hypotheses about development and maintenance"
        ),
    )
    key_concepts: list[str] = Field(
        default_factory=list,
        description="Key theoretical concepts from this framework applied to the case",
    )
    therapeutic_approaches: list[str] = Field(
        default_factory=list,
        description="Specific therapeutic strategies or techniques this framework suggests",
    )
    strengths: list[str] = Field(
        default_factory=list,
        description="Strengths of applying this framework to this particular case",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Limitations, contraindications, or cautions for this framework with this case",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description=(
            "How well this framework fits the available information "
            "(0=tentative, 1=strong fit)"
        ),
    )
    evidence_alignment: str = Field(
        default="",
        description=(
            "How the retrieved evidence aligns with or challenges "
            "this framework's perspective"
        ),
    )
    analysed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    analysis_ms: float = Field(
        default=0.0, ge=0.0,
        description="Time taken for this framework analysis in milliseconds",
    )

    model_config = {"frozen": True}


class FrameworkMetadata(BaseModel):
    """Framework-level metadata for discovery, selection, and evaluation.

    Reusable by the framework evaluation harness, the comparison engine,
    a future frontend, and human-in-the-loop framework selection. The
    registry exposes this via ``get_metadata()`` / ``list_metadata()``.
    """

    id: str = Field(
        ..., min_length=1, description="Stable machine-readable identifier",
    )
    display_name: str = Field(
        ..., min_length=1, description="Human-readable framework name",
    )
    description: str = Field(
        ..., min_length=10, description="One-sentence description of the framework",
    )
    version: str = Field(
        default="0.1.0", description="Plugin version (semver)",
    )
    supported_features: list[str] = Field(
        default_factory=list,
        description="Capabilities or techniques this framework supports",
    )
    primary_use_cases: list[str] = Field(
        default_factory=list,
        description="Presentations or client groups this framework suits best",
    )
    benchmark_categories: list[str] = Field(
        default_factory=list,
        description="Benchmark case categories this framework is expected to apply to",
    )

    model_config = {"frozen": True}


class FrameworkComparisonRow(BaseModel):
    dimension: str = Field(
        ..., min_length=3,
        description="Dimension being compared (e.g. 'Conceptualisation', 'Key Concepts')",
    )
    entries: dict[str, str] = Field(
        ...,
        description="Mapping of framework_name -> text for this dimension",
    )

    model_config = {"frozen": True}


class FrameworkComparisonResult(BaseModel):
    rows: list[FrameworkComparisonRow] = Field(
        default_factory=list,
        description="Side-by-side comparison across frameworks",
    )
    convergence_areas: list[str] = Field(
        default_factory=list,
        description="Areas where frameworks converge or agree",
    )
    divergence_areas: list[str] = Field(
        default_factory=list,
        description="Areas where frameworks offer different perspectives",
    )
    recommendation: str = Field(
        default="",
        description=(
            "Evidence-informed suggestion for which framework(s) may be "
            "most relevant — always presented as clinician consideration, "
            "never a prescription"
        ),
    )
    compared_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    model_config = {"frozen": True}
