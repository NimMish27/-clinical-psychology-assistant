from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class Finding(BaseModel):
    """A single key finding drawn from the retrieved evidence."""

    statement: str = Field(
        ...,
        min_length=5,
        description="Concise statement of the finding",
    )
    supporting_sources: list[str] = Field(
        ...,
        min_length=1,
        description="Source citations supporting this finding (e.g. 'DSM5.pdf, p.12')",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in this finding (0=speculative, 1=well-established)",
    )

    model_config = {"frozen": True}


class Theme(BaseModel):
    """A common theme that emerges across the retrieved evidence."""

    name: str = Field(
        ...,
        min_length=3,
        description="Short label for the theme",
    )
    description: str = Field(
        ...,
        min_length=5,
        description="Explanation of how this theme manifests in the evidence",
    )
    prevalence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How consistently this theme appears across sources",
    )

    model_config = {"frozen": True}


class Agreement(BaseModel):
    """An area where multiple sources converge."""

    topic: str = Field(
        ...,
        min_length=3,
        description="The topic of agreement",
    )
    consensus: str = Field(
        ...,
        min_length=5,
        description="What the sources agree on",
    )
    supporting_sources: list[str] = Field(
        ...,
        min_length=1,
        description="Sources that support this consensus",
    )

    model_config = {"frozen": True}


class Uncertainty(BaseModel):
    """An area where evidence is limited, conflicting, or absent."""

    topic: str = Field(
        ...,
        min_length=3,
        description="The topic of uncertainty",
    )
    description: str = Field(
        ...,
        min_length=5,
        description="Nature of the uncertainty or gap",
    )
    implications: str = Field(
        ...,
        min_length=5,
        description="How this uncertainty affects clinical decision-making",
    )

    model_config = {"frozen": True}


class Implication(BaseModel):
    """A practical action or consideration for the clinician."""

    recommendation: str = Field(
        ...,
        min_length=5,
        description="Actionable recommendation based on the evidence",
    )
    strength: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How strongly the evidence supports this implication",
    )
    source: str | None = Field(
        None,
        description="Key source or guideline underpinning this implication",
    )

    model_config = {"frozen": True}


class EvidenceSynthesisResult(BaseModel):
    """Structured output of the evidence synthesis process.

    This is the top-level result that LangGraph agents and the clinical
    pipeline can consume.  Every field is optional so downstream consumers
    can handle partial data gracefully.
    """

    key_findings: list[Finding] = Field(
        default_factory=list,
        description="Key evidence findings relevant to the case",
    )
    common_themes: list[Theme] = Field(
        default_factory=list,
        description="Common themes emerging across the evidence",
    )
    areas_of_agreement: list[Agreement] = Field(
        default_factory=list,
        description="Topics where multiple sources converge",
    )
    areas_of_uncertainty: list[Uncertainty] = Field(
        default_factory=list,
        description="Gaps, conflicts, or limitations in the evidence",
    )
    practical_implications: list[Implication] = Field(
        default_factory=list,
        description="Actionable recommendations for clinical practice",
    )
    overall_summary: str = Field(
        default="",
        description="Concise paragraph summarising the entire evidence synthesis",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall confidence in this synthesis (0=speculative, 1=well-supported)",
    )
    chunks_analysed: int = Field(
        default=0,
        ge=0,
        description="Number of retrieved chunks that were analysed",
    )
    synthesised_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    synthesis_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Time taken for synthesis in milliseconds",
    )

    model_config = {"frozen": True}
