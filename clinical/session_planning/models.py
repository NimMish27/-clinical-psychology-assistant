from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class SessionObjective(BaseModel):
    """A single objective for a proposed therapy session."""

    title: str = Field(
        ...,
        min_length=5,
        description="Short descriptive title for the session (e.g. 'Introducing thought records')",
    )
    focus: str = Field(
        ...,
        min_length=10,
        description="What the session aims to explore or accomplish, grounded in the formulation and plan",
    )
    suggested_activities: list[str] = Field(
        default_factory=list,
        description="Concrete in-session activities or techniques for clinician consideration",
    )
    homework: str | None = Field(
        None,
        description="Suggested between-session task or reflection for clinician consideration",
    )
    expected_duration_minutes: int | None = Field(
        None,
        ge=0,
        description="Indicative session length in minutes, if known",
    )

    model_config = {"frozen": True}


class SessionPlanResult(BaseModel):
    """Structured output of the session planning process.

    Proposes a sequence of therapy sessions aligned with the selected
    therapeutic framework. All items are suggestions for clinician
    consideration — the psychologist always makes the final call.
    """

    disclaimer: str = Field(
        ...,
        min_length=20,
        description="Clinical disclaimer emphasising that session proposals require clinical judgement",
    )
    session_count: int = Field(
        default=0,
        ge=0,
        description="Number of proposed sessions in the plan",
    )
    cadence: str | None = Field(
        None,
        description="Suggested session cadence (e.g. 'weekly', 'twice weekly')",
    )
    objectives: list[SessionObjective] = Field(
        default_factory=list,
        description="Ordered list of proposed session objectives",
    )
    review_points: list[str] = Field(
        default_factory=list,
        description="Points where progress should be reviewed and the plan adapted",
    )
    planned_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )
    planning_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Time taken for session planning in milliseconds",
    )

    model_config = {"frozen": True}
