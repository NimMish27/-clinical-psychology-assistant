"""
clinical/framework/alliance_rupture/plugin.py — Alliance Rupture and Repair framework plugin.
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
class AllianceRupturePlugin(ClinicalFramework):

    @classmethod
    def metadata(cls) -> FrameworkMetadata:
        return FrameworkMetadata(
            id="alliance_rupture",
            display_name="Alliance Rupture and Repair",
            version="1.0.0",
            description=(
                "Alliance rupture and repair focuses on identifying and resolving "
                "breakdowns in the therapeutic alliance. Ruptures can be withdrawal-based "
                "(client disengages, avoids, or complies passively) or confrontation-based "
                "(client expresses anger, dissatisfaction, or challenges the therapist). "
                "Repair involves metacommunication about the relational process, validating "
                "the client's experience, and collaboratively renegotiating the goals and "
                "tasks of therapy."
            ),
            supported_use_cases=[
                "Therapy process difficulties and impasses",
                "Client disengagement or dropout risk",
                "Interpersonal patterns re-enacted in therapy",
                "Negative reactions to therapy or therapist",
                "Enhancing therapeutic alliance in early sessions",
                "Cross-cultural ruptures and misunderstandings",
            ],
            literature_collection="alliance_rupture_literature",
            retrieval_strategy=RetrievalStrategy.HYBRID,
            intervention_style=InterventionStyle(
                primary_focus="Identifying and repairing therapeutic alliance ruptures",
                techniques=[
                    "Metacommunication about the therapeutic relationship",
                    "Directly inviting feedback about the therapy process",
                    "Validating client's experience of the rupture",
                    "Collaborative renegotiation of goals and tasks",
                    "Exploring relational patterns underlying the rupture",
                ],
                session_structure=(
                    "Can be integrated into any therapeutic orientation; typically involves "
                    "a metacommunicative turn when a rupture marker is observed"
                ),
                therapeutic_stance=(
                    "Transparent, collaborative, non-defensive; therapist takes responsibility "
                    "for their contribution to the rupture and models repair"
                ),
            ),
            tags=["alliance", "rupture", "repair", "therapeutic relationship", "process"],
            citations=[
                "Safran, J. D., & Muran, J. C. (2000). Negotiating the Therapeutic Alliance.",
                "Eubanks, C. F., Muran, J. C., & Safran, J. D. (2018). Alliance rupture repair: A meta-analysis.",
            ],
        )

    @classmethod
    def is_applicable(cls, case_understanding: dict) -> bool:
        topics = _safe_topics(case_understanding)
        keywords = {
            "alliance", "rupture", "therapeutic", "relationship", "engagement",
            "dropout", "disengagement", "resistance", "defensiveness",
            "therapy", "process", "impasse", "negative", "reaction",
            "compliance", "withdrawal", "confrontation",
        }
        return bool(keywords & topics)
