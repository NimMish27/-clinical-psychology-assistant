"""
clinical/framework/general/plugin.py — General Psychotherapy framework plugin.
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
class GeneralPsychotherapyPlugin(ClinicalFramework):
    """
    Catch-all framework for any clinical case.

    ``is_applicable`` always returns True, making this the default framework
    when no other specialised framework matches. It provides a balanced,
    integrative clinical perspective.
    """

    @classmethod
    def metadata(cls) -> FrameworkMetadata:
        return FrameworkMetadata(
            id="general",
            display_name="General Psychotherapy",
            version="1.0.0",
            description=(
                "General psychotherapy provides an integrative, evidence-informed clinical "
                "perspective that draws on common factors across therapeutic modalities. "
                "It emphasises the therapeutic alliance, case conceptualisation, "
                "psychoeducation, and collaborative treatment planning. This framework "
                "serves as a default when no specialised framework is indicated, ensuring "
                "every case receives structured clinical reasoning."
            ),
            supported_use_cases=[
                "Any clinical presentation requiring structured case conceptualisation",
                "Cases where no single specialised framework is clearly indicated",
                "Initial assessment and treatment planning",
                "Brief intervention and support",
                "Integration of multiple treatment approaches",
                "Monitoring therapeutic progress and outcomes",
            ],
            literature_collection=None,
            retrieval_strategy=RetrievalStrategy.NONE,
            intervention_style=InterventionStyle(
                primary_focus="Integrative case conceptualisation and common-factors therapy",
                techniques=[
                    "Active listening and empathic reflection",
                    "Psychoeducation and collaborative formulation",
                    "Goal setting and treatment planning",
                    "Progress monitoring and outcome assessment",
                    "Relapse prevention and maintenance planning",
                ],
                session_structure=(
                    "Flexible and responsive to client needs; typically includes check-in, "
                    "agenda setting, therapeutic work, summary, and between-session planning"
                ),
                therapeutic_stance=(
                    "Warm, collaborative, and flexible; therapist draws on multiple orientations "
                    "as indicated by the client's presentation and preferences"
                ),
            ),
            tags=["general", "integrative", "common-factors", "psychotherapy", "eclectic"],
            citations=[
                "Wampold, B. E. (2001). The Great Psychotherapy Debate.",
                "Norcross, J. C., & Lambert, M. J. (2019). Psychotherapy relationships that work (3rd ed.).",
            ],
        )

    @classmethod
    def is_applicable(cls, case_understanding: dict) -> bool:
        return True
