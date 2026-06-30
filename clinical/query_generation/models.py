from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class QueryCategory(str, Enum):
    DIAGNOSTIC = "diagnostic"
    TREATMENT = "treatment"
    PHENOMENOLOGY = "phenomenology"
    CONTEXTUAL = "contextual"
    ASSESSMENT = "assessment"
    RISK = "risk"


class OptimizedQuery(BaseModel):
    """A single optimized retrieval query with metadata."""

    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="The search query string for vector retrieval",
    )
    category: QueryCategory = Field(
        ...,
        description="Category of knowledge this query targets",
    )
    weight: float = Field(
        default=1.0,
        ge=0.1,
        le=3.0,
        description="Relative importance for fusion scoring across queries",
    )
    rationale: str = Field(
        ...,
        min_length=5,
        description="Why this query improves retrieval for this case",
    )
    expansion_of: str | None = Field(
        None,
        description="Source concept this query expands upon, if applicable",
    )

    model_config = {"frozen": True}


class QueryGenerationResult(BaseModel):
    """Complete output of the retrieval query generation process."""

    queries: list[OptimizedQuery] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Generated retrieval queries sorted by descending weight",
    )
    raw_text_summary: str | None = Field(
        None,
        description="Brief summary of what the queries target",
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    generation_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Time taken to generate queries in milliseconds",
    )

    def to_query_strings(self) -> list[str]:
        """Return just the query strings, sorted by weight descending."""
        return [q.query for q in sorted(self.queries, key=lambda x: x.weight, reverse=True)]

    def to_weighted_queries(self) -> list[dict[str, Any]]:
        """Return query- weight pairs for fusion."""
        return [
            {"query": q.query, "weight": q.weight}
            for q in sorted(self.queries, key=lambda x: x.weight, reverse=True)
        ]

    model_config = {"frozen": True}
