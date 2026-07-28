"""
clinical/framework/mi/plugin.py — Motivational Interviewing framework plugin.
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
class MIPlugin(ClinicalFramework):

    @classmethod
    def metadata(cls) -> FrameworkMetadata:
        return FrameworkMetadata(
            id="mi",
            display_name="Motivational Interviewing",
            version="1.0.0",
            description=(
                "Motivational Interviewing (MI) is a collaborative, person-centred counselling "
                "style designed to strengthen personal motivation for and commitment to a specific "
                "goal by eliciting and exploring the person's own reasons for change. MI is "
                "grounded in four foundational processes: engaging (building the working alliance), "
                "focusing (developing a clear direction), evoking (eliciting change talk), and "
                "planning (developing commitment to change). It uses core skills (OARS: open "
                "questions, affirmations, reflections, summaries) and strategically responds to "
                "change talk and sustain talk."
            ),
            supported_use_cases=[
                "Substance use and addictive behaviours",
                "Health behaviour change (diet, exercise, medication adherence)",
                "Readiness for therapy engagement",
                "Ambivalence about treatment",
                "Behavioural risk reduction",
                "Chronic disease management",
            ],
            literature_collection="mi_literature",
            retrieval_strategy=RetrievalStrategy.HYBRID,
            intervention_style=InterventionStyle(
                primary_focus="Eliciting and strengthening intrinsic motivation for change",
                techniques=[
                    "Open questions about change",
                    "Affirmations of strengths and efforts",
                    "Reflective listening (simple and complex)",
                    "Summarising to consolidate change talk",
                    "Decisional balance and exploring values",
                ],
                session_structure=(
                    "Goal-focused and time-sensitive; sessions follow the four processes "
                    "(engage, focus, evoke, plan) but flexibly revisit earlier processes "
                    "if resistance or ambivalence arises"
                ),
                therapeutic_stance=(
                    "Collaborative, evocative, and honouring of client autonomy; therapist "
                    "actively avoids the 'righting reflex' and trusts the client's expertise "
                    "about their own life"
                ),
            ),
            tags=["mi", "motivational", "change", "ambivalence", "substance use", "behaviour change"],
            citations=[
                "Miller, W. R., & Rollnick, S. (2013). Motivational Interviewing (3rd ed.).",
                "Miller, W. R., & Rose, G. S. (2009). Toward a theory of motivational interviewing.",
            ],
        )

    @classmethod
    def is_applicable(cls, case_understanding: dict) -> bool:
        topics = _safe_topics(case_understanding)
        keywords = {
            "motivation", "ambivalence", "substance", "alcohol", "drug",
            "addiction", "change", "readiness", "engagement", "adherence",
            "risk", "behaviour", "lifestyle", "commitment",
        }
        return bool(keywords & topics)
