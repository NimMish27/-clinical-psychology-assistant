from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class TherapeuticFocus(BaseModel):
    """A suggested area of therapeutic focus."""

    area: str = Field(
        ...,
        min_length=5,
        description="The domain or theme to focus on (e.g. cognitive patterns, emotional regulation)",
    )
    rationale: str = Field(
        ...,
        min_length=10,
        description="Why this focus area is relevant given the formulation and evidence",
    )

    model_config = {"frozen": True}


class TreatmentGoal(BaseModel):
    """A collaboratively defined treatment goal."""

    goal: str = Field(
        ...,
        min_length=10,
        description="Specific, measurable, and realistic goal statement",
    )
    suggested_measurement: str | None = Field(
        None,
        description="How progress might be tracked (e.g. PHQ-9 score, behavioural record)",
    )
    indicative_timeframe: str | None = Field(
        None,
        description="Suggested review period (e.g. 4 weeks, 12 sessions)",
    )

    model_config = {"frozen": True}


class CBTStrategy(BaseModel):
    """A CBT technique suggested for clinician consideration."""

    technique: str = Field(
        ...,
        min_length=5,
        description="The CBT technique (e.g. thought record, behavioural experiment)",
    )
    rationale: str = Field(
        ...,
        min_length=10,
        description="How this technique maps to the formulation",
    )
    application: str | None = Field(
        None,
        description="How the technique might be introduced and used with this client",
    )

    model_config = {"frozen": True}


class ACTStrategy(BaseModel):
    """An ACT process suggested for clinician consideration."""

    process: str = Field(
        ...,
        min_length=5,
        description="The ACT process (e.g. acceptance, defusion, values clarification)",
    )
    rationale: str = Field(
        ...,
        min_length=10,
        description="Why this process is relevant given the formulation",
    )
    application: str | None = Field(
        None,
        description="How this process might be explored with the client",
    )

    model_config = {"frozen": True}


class DBTStrategy(BaseModel):
    """A DBT skill or module suggested for clinician consideration."""

    skill: str = Field(
        ...,
        min_length=5,
        description="The DBT skill or module (e.g. distress tolerance, interpersonal effectiveness)",
    )
    rationale: str = Field(
        ...,
        min_length=10,
        description="Why this skill area is relevant given the presentation",
    )
    application: str | None = Field(
        None,
        description="How the skill might be introduced and practised",
    )

    model_config = {"frozen": True}


class PsychEducationSuggestion(BaseModel):
    """A psychoeducation topic suggested for clinician consideration."""

    topic: str = Field(
        ...,
        min_length=5,
        description="The topic for psychoeducation (e.g. anxiety cycle, sleep hygiene)",
    )
    key_points: list[str] = Field(
        ...,
        min_length=1,
        description="Key messages to convey",
    )

    model_config = {"frozen": True}


class BehaviouralActivationSuggestion(BaseModel):
    """A behavioural activation suggestion for clinician consideration."""

    activity_domain: str = Field(
        ...,
        min_length=5,
        description="Area of activity (e.g. social, physical, occupational)",
    )
    suggested_activities: list[str] = Field(
        ...,
        min_length=1,
        description="Specific activities to explore with the client",
    )
    rationale: str = Field(
        ...,
        min_length=10,
        description="How these activities relate to the formulation and goals",
    )

    model_config = {"frozen": True}


class SelfCompassionStrategy(BaseModel):
    """A self-compassion practice suggested for clinician consideration."""

    practice: str = Field(
        ...,
        min_length=5,
        description="The self-compassion practice (e.g. compassionate letter, soothing rhythm breathing)",
    )
    rationale: str = Field(
        ...,
        min_length=10,
        description="Why this practice may be helpful given the formulation",
    )

    model_config = {"frozen": True}


class HomeworkIdea(BaseModel):
    """A between-session activity for clinician consideration."""

    activity: str = Field(
        ...,
        min_length=10,
        description="Description of the suggested between-session activity",
    )
    purpose: str = Field(
        ...,
        min_length=10,
        description="The therapeutic purpose this activity serves",
    )
    frequency: str | None = Field(
        None,
        description="Suggested frequency (e.g. daily, 3 times per week)",
    )

    model_config = {"frozen": True}


class InterventionDirection(BaseModel):
    """A broad therapeutic direction for clinician consideration."""

    area: str = Field(
        ...,
        min_length=5,
        description="The intervention domain (e.g. cognitive restructuring, emotional regulation, behavioural activation)",
    )
    suggested_approaches: list[str] = Field(
        ...,
        min_length=1,
        description="Specific approaches or modalities to consider",
    )
    rationale: str = Field(
        ...,
        min_length=10,
        description="How this direction follows from the formulation and evidence",
    )

    model_config = {"frozen": True}


class TherapeuticPlanResult(BaseModel):
    """Structured output of the therapeutic planning process.

    This is the top-level result that LangGraph agents and the clinical
    pipeline can consume.  Every field is optional so downstream consumers
    can handle partial data gracefully.

    All output is presented as evidence-informed suggestions for clinician
    consideration — NOT prescriptions or treatment mandates.
    """

    disclaimer: str = Field(
        ...,
        min_length=20,
        description="Clinical disclaimer emphasising that all suggestions require clinical judgement",
    )
    therapeutic_focus: list[TherapeuticFocus] = Field(
        default_factory=list,
        description="Suggested areas of therapeutic focus based on formulation",
    )
    treatment_goals: list[TreatmentGoal] = Field(
        default_factory=list,
        description="Example treatment goals for collaborative discussion",
    )
    intervention_directions: list[InterventionDirection] = Field(
        default_factory=list,
        description="Broad therapeutic directions to consider",
    )
    cbt_strategies: list[CBTStrategy] = Field(
        default_factory=list,
        description="CBT techniques suggested for clinician consideration",
    )
    act_strategies: list[ACTStrategy] = Field(
        default_factory=list,
        description="ACT processes suggested for clinician consideration",
    )
    dbt_strategies: list[DBTStrategy] = Field(
        default_factory=list,
        description="DBT skills suggested for clinician consideration",
    )
    psychoeducation_suggestions: list[PsychEducationSuggestion] = Field(
        default_factory=list,
        description="Psychoeducation topics for clinician consideration",
    )
    behavioural_activation_suggestions: list[BehaviouralActivationSuggestion] = Field(
        default_factory=list,
        description="Behavioural activation suggestions for clinician consideration",
    )
    self_compassion_strategies: list[SelfCompassionStrategy] = Field(
        default_factory=list,
        description="Self-compassion practices for clinician consideration",
    )
    homework_ideas: list[HomeworkIdea] = Field(
        default_factory=list,
        description="Between-session activity ideas for clinician consideration",
    )
    planned_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    planning_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Time taken for planning in milliseconds",
    )

    model_config = {"frozen": True}
