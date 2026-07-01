from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class ClinicalResponseResult(BaseModel):
    """The final markdown-formatted clinical response.

    This is the terminal output of the clinical analysis pipeline —
    a complete, ready-to-present report synthesising all previous
    module outputs into a structured markdown document.
    """

    markdown: str = Field(
        ...,
        min_length=50,
        description="Full markdown-formatted clinical response with all 11 sections",
    )
    sections_generated: int = Field(
        default=0,
        ge=0,
        le=11,
        description="Number of sections successfully generated (0-11)",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall confidence in the response based on completeness of input data",
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    generation_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Time taken for generation in milliseconds",
    )

    model_config = {"frozen": True}
