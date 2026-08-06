"""
clinical/framework/dbt/plugin.py — Dialectical Behavior Therapy framework plugin.
"""

from clinical.framework.base import ClinicalFramework
from clinical.framework.models import (
    FrameworkMetadata,
    InterventionStyle,
    RetrievalStrategy,
)
from clinical.framework.registry import register_framework
from clinical.framework.utils import _safe_topics


@register_framework
class DBTPlugin(ClinicalFramework):

    @classmethod
    def metadata(cls) -> FrameworkMetadata:
        return FrameworkMetadata(
            id="dbt",
            display_name="Dialectical Behavior Therapy",
            version="1.0.0",
            description=(
                "Dialectical Behavior Therapy (DBT) is a comprehensive cognitive-behavioural "
                "treatment originally developed for borderline personality disorder. It balances "
                "change-oriented strategies (problem-solving, skills training) with acceptance-based "
                "strategies (validation, mindfulness). DBT organises treatment around five "
                "modes: individual therapy, group skills training, phone coaching, therapist "
                "consultation team, and ancillary treatment. Skills are taught in four modules: "
                "mindfulness, distress tolerance, emotion regulation, and interpersonal "
                "effectiveness."
            ),
            supported_use_cases=[
                "Emotion dysregulation and intense affect",
                "Self-harm and suicidal behaviour",
                "Borderline personality patterns",
                "Chronic interpersonal difficulties",
                "Impulse control problems",
                "Eating disorders with dysregulation",
                "Treatment-resistant depression and anxiety",
            ],
            literature_collection="dbt_literature",
            retrieval_strategy=RetrievalStrategy.HYBRID,
            intervention_style=InterventionStyle(
                primary_focus="Balancing acceptance and change through skills training and validation",
                techniques=[
                    "Mindfulness skills (wise mind, what/how skills)",
                    "Distress tolerance and crisis survival strategies",
                    "Emotion regulation skills",
                    "Interpersonal effectiveness training",
                    "Validation strategies and dialectical strategies",
                ],
                session_structure=(
                    "Highly structured; individual sessions follow a diary card review, "
                    "skills coaching, and chain analysis of problem behaviours; "
                    "group sessions follow a rotating skills curriculum"
                ),
                therapeutic_stance=(
                    "Active, directive, and validating; therapist balances acceptance of the "
                    "client as they are while pushing for change; therapist participates in "
                    "a consultation team to maintain treatment fidelity"
                ),
            ),
            tags=["dbt", "dialectical", "skills", "emotion regulation", "validation", "self-harm"],
            citations=[
                "Linehan, M. M. (1993). Cognitive-Behavioral Treatment of Borderline Personality Disorder.",
                "Linehan, M. M. (2015). DBT Skills Training Manual (2nd ed.).",
            ],
        )

    @classmethod
    def is_applicable(cls, case_understanding: dict) -> bool:
        topics = _safe_topics(case_understanding)
        keywords = {
            "emotion", "dysregulation", "self-harm", "suicidal", "impulse",
            "borderline", "personality", "crisis", "intense", "affect",
            "instability", "relationship", "anger", "emptiness",
            "dissociation", "urge", "overwhelm",
        }
        return bool(keywords & topics)
