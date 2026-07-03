from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class SafetyIssueCategory(str, Enum):
    UNSUPPORTED_DIAGNOSIS = "unsupported_diagnosis"
    HALLUCINATION = "hallucination"
    MISSING_CITATION = "missing_citation"
    OVERCONFIDENT_LANGUAGE = "overconfident_language"
    ETHICAL_CONCERN = "ethical_concern"
    UNSAFE_RECOMMENDATION = "unsafe_recommendation"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SafetyIssue(BaseModel):
    category: SafetyIssueCategory = Field(
        ..., description="Category of the safety issue",
    )
    severity: Severity = Field(
        ..., description="How severe this issue is",
    )
    location: str = Field(
        ..., description="Section or location in the response where the issue was found",
    )
    excerpt: str = Field(
        ..., max_length=200, description="Text snippet containing the issue",
    )
    explanation: str = Field(
        ..., min_length=5, description="Why this is a concern",
    )
    suggestion: str = Field(
        ..., min_length=5, description="How to fix or mitigate the issue",
    )

    model_config = {"frozen": True}


class SafetyValidationResult(BaseModel):
    markdown: str = Field(
        ..., description="The validated (and possibly revised) markdown response",
    )
    original_markdown: str = Field(
        ..., description="The original unmodified markdown for comparison",
    )
    issues: list[SafetyIssue] = Field(
        default_factory=list,
        description="Issues found during validation, empty if clean",
    )
    was_revised: bool = Field(
        default=False,
        description="Whether the response was modified during validation",
    )
    revision_summary: str = Field(
        default="",
        description="Summary of what was changed, if revised",
    )
    overall_verdict: str = Field(
        ...,
        description='One of: "clean", "minor_issues", "needs_review", "unsafe"',
    )
    validated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    validation_ms: float = Field(
        default=0.0, ge=0.0,
    )
