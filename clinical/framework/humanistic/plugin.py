"""
clinical/framework/humanistic/plugin.py — Humanistic Therapy framework plugin.

Production-ready overrides for all 8 lifecycle methods with:
- Humanistic-specific clinical content (self-concept, congruence, UPR, empathy, authenticity)
- "Never diagnose" enforcement at prompt and post-processing level
- Vector store retrieval with metadata filtering for evidence
- Graceful fallback to LLM on retrieval failure
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from clinical.framework.base import ClinicalFramework
from clinical.framework.models import (
    CaseAnalysis,
    CaseConceptualization,
    EvidenceResult,
    FollowupQuestionsResult,
    FrameworkMetadata,
    InterventionStyle,
    InterventionPlanResult,
    LimitationsResult,
    RetrievalStrategy,
    RetrievedEvidence,
    SessionObjectivesResult,
    TherapeuticFocusResult,
)
from clinical.framework.registry import register_framework
from clinical.framework.utils import _safe_topics
from app_logging.logger import get_logger

if TYPE_CHECKING:
    from clinical.llm import LLMService

_log = get_logger(__name__)

# ── Never-diagnose helpers (same pattern as CBT/ACT plugins) ──────────

_DIAGNOSTIC_PATTERNS: list[re.Pattern] = [
    re.compile(
        r"(?i)\b(major depressive disorder|clinical depression|mdd|"
        r"generalized anxiety disorder|generalised anxiety disorder|gad|"
        r"social anxiety disorder|sad|social phobia|"
        r"panic disorder|agoraphobia|"
        r"obsessive-compulsive disorder|obsessive compulsive disorder|ocd|"
        r"post-traumatic stress disorder|posttraumatic stress disorder|ptsd|"
        r"bipolar (i|ii|disorder)|"
        r"borderline personality disorder|bpd|"
        r"eating disorder|anorexia nervosa|bulimia nervosa|binge-eating disorder|"
        r"attention deficit hyperactivity disorder|adhd|"
        r"autism spectrum disorder|asd)\b"
    ),
]


def _strip_diagnoses(text: str) -> str:
    for pat in _DIAGNOSTIC_PATTERNS:
        text = pat.sub(
            lambda m: _diagnosis_replacement(m.group(0)),
            text,
        )
    return text


def _diagnosis_replacement(match: str) -> str:
    mapping: dict[str, str] = {
        "major depressive disorder": "persistent low mood and reduced functioning",
        "clinical depression": "clinically significant low mood",
        "mdd": "persistent low mood",
        "generalized anxiety disorder": "persistent and excessive worry",
        "generalised anxiety disorder": "persistent and excessive worry",
        "gad": "persistent worry patterns",
        "social anxiety disorder": "marked anxiety in social situations",
        "sad": "elevated social anxiety",
        "social phobia": "elevated social anxiety",
        "panic disorder": "recurrent unexpected intense fear episodes",
        "agoraphobia": "anxiety about being in situations from which escape is difficult",
        "obsessive-compulsive disorder": "recurrent intrusive thoughts and repetitive behaviours",
        "obsessive compulsive disorder": "recurrent intrusive thoughts and repetitive behaviours",
        "ocd": "intrusive thoughts and compulsive behaviours",
        "post-traumatic stress disorder": "persistent distress following traumatic experience",
        "posttraumatic stress disorder": "persistent distress following traumatic experience",
        "ptsd": "trauma-related distress",
        "bipolar i": "alternating periods of elevated and low mood",
        "bipolar ii": "alternating periods of hypomanic and low mood",
        "bipolar disorder": "alternating periods of elevated and low mood",
        "borderline personality disorder": "significant emotional and interpersonal instability",
        "bpd": "significant emotional and interpersonal instability",
        "eating disorder": "disrupted eating patterns and body image concerns",
        "anorexia nervosa": "restricted eating with weight concerns",
        "bulimia nervosa": "binge-eating followed by compensatory behaviours",
        "binge-eating disorder": "recurrent binge-eating episodes",
        "attention deficit hyperactivity disorder": "difficulties with attention and impulse regulation",
        "adhd": "attentional and regulation difficulties",
        "autism spectrum disorder": "differences in social communication and sensory processing",
        "asd": "neurodevelopmental differences in social communication",
    }
    return mapping.get(match.lower(), match)


def _strip_diagnoses_from_result[T: Any](result: T) -> T:
    """Recursively strip diagnostic labels from all string fields in a Pydantic model."""
    if not hasattr(result, "model_fields"):
        return result
    for field_name, field_info in result.model_fields.items():
        val = getattr(result, field_name, None)
        if val is None:
            continue
        if isinstance(val, str):
            cleaned = _strip_diagnoses(val)
            if cleaned != val:
                setattr(result, field_name, cleaned)
        elif isinstance(val, list):
            cleaned_list: list[Any] = []
            for item in val:
                if isinstance(item, str):
                    cleaned_list.append(_strip_diagnoses(item))
                elif isinstance(item, dict):
                    cleaned_dict: dict[str, Any] = {}
                    for k, v in item.items():
                        if isinstance(v, str):
                            cleaned_dict[k] = _strip_diagnoses(v)
                        else:
                            cleaned_dict[k] = v
                    cleaned_list.append(cleaned_dict)
                else:
                    cleaned_list.append(item)
            setattr(result, field_name, cleaned_list)
        elif isinstance(val, dict):
            cleaned_dict: dict[str, Any] = {}
            for k, v in val.items():
                if isinstance(v, str):
                    cleaned_dict[k] = _strip_diagnoses(v)
                else:
                    cleaned_dict[k] = v
            setattr(result, field_name, cleaned_dict)
    return result


@register_framework
class HumanisticPlugin(ClinicalFramework):

    @classmethod
    def metadata(cls) -> FrameworkMetadata:
        return FrameworkMetadata(
            id="humanistic",
            display_name="Humanistic Therapy",
            version="2.0.0",
            description=(
                "Humanistic therapy emphasises the client's inherent capacity for "
                "self-actualisation and personal growth. It prioritises the therapeutic "
                "relationship, unconditional positive regard, empathy, and genuineness. "
                "Rather than pathologising symptoms, it views distress as a result of "
                "incongruence between the client's real self and ideal self, or blocked "
                "growth due to conditions of worth imposed by others. "
                "Humanistic therapy encompasses person-centred (Rogers), Gestalt (Perls), "
                "and experiential (Gendlin) traditions, all united by a phenomenological "
                "stance — understanding the client's subjective lived experience from "
                "their own internal frame of reference. "
                "The therapy relationship is seen as the primary vehicle for change, "
                "not a delivery system for techniques. "
                "Core processes include: developing congruence (alignment between "
                "self-concept and experience), reducing conditions of worth, strengthening "
                "unconditional positive self-regard, deepening organismic trusting, and "
                "freeing the actualising tendency."
            ),
            supported_use_cases=[
                "Low self-esteem and self-worth issues",
                "Existential concerns and meaning-making",
                "Relationship difficulties and interpersonal patterns",
                "Personal growth and self-actualisation",
                "Adjustment to life transitions",
                "Mild to moderate anxiety and depression",
                "Identity exploration and self-discovery",
                "Shame and self-criticism",
            ],
            literature_collection="humanistic_literature",
            retrieval_strategy=RetrievalStrategy.HYBRID,
            intervention_style=InterventionStyle(
                primary_focus=(
                    "Self-actualisation, congruence, unconditional positive regard, "
                    "empathic understanding, and authentic relational contact"
                ),
                techniques=[
                    "Active listening and empathic responding",
                    "Unconditional positive regard and non-judgmental acceptance",
                    "Therapist congruence and genuine presence",
                    "Focusing and experiential exploration (Gendlin)",
                    "Gestalt two-chair dialogue for self-splits",
                    "Empty chair for unfinished business",
                    "Phenomenological inquiry and here-and-now process",
                    "Evocative unfolding of implicit meaning",
                ],
                session_structure=(
                    "Client-led with minimal structure; therapist follows the client's "
                    "moment-to-moment experience; emphasis on here-and-now relational process; "
                    "sessions are emergent rather than agenda-driven"
                ),
                therapeutic_stance=(
                    "Authentic, empathic, non-directive; therapist is a fellow human being "
                    "rather than an expert diagnostician; the therapist's way of being is "
                    "the core intervention; trust in the client's actualising tendency"
                ),
            ),
            tags=["humanistic", "person-centered", "gestalt", "experiential",
                  "phenomenological", "rogerian", "client-centred"],
            citations=[
                "Rogers, C. R. (1957). The necessary and sufficient conditions of therapeutic personality change.",
                "Rogers, C. R. (1961). On Becoming a Person.",
                "Perls, F., Hefferline, R., & Goodman, P. (1951). Gestalt Therapy.",
                "Gendlin, E. T. (1978). Focusing.",
            ],
        )

    @classmethod
    def is_applicable(cls, case_understanding: dict) -> bool:
        topics = _safe_topics(case_understanding)
        keywords = {
            "self-esteem", "self-worth", "identity", "meaning", "purpose",
            "relationship", "existential", "personal", "growth",
            "incongruence", "worth", "life", "transition",
            "self", "acceptance", "shame", "guilt", "authenticity",
            "belonging", "loneliness", "isolation", "fulfilment",
            "potential", "direction", "purpose",
        }
        return bool(keywords & topics)

    # ── Helper: build a query string from case input for retrieval ──────────

    @staticmethod
    def _build_search_query(inp: Any) -> str:
        parts: list[str] = []
        if hasattr(inp, "case_understanding") and isinstance(inp.case_understanding, dict):
            for val in inp.case_understanding.values():
                if isinstance(val, str) and len(val) > 20:
                    parts.append(val)
        if hasattr(inp, "raw_text") and inp.raw_text:
            parts.append(inp.raw_text)
        if parts:
            return " ".join(parts[:3])
        return "Humanistic therapy evidence for clinical case"

    # ── Helper: evidence retrieval with HumanisticRetriever ────────────────

    async def _try_vector_retrieval(
        self,
        inp: Any,
        max_results: int = 5,
    ) -> EvidenceResult | None:
        """Attempt to retrieve humanistic evidence using the HumanisticRetriever."""
        try:
            from clinical.retrieval.humanistic_retriever import (
                HumanisticEvidenceFilter,
                HumanisticRetriever,
            )

            query_text = self._build_search_query(inp)
            if not query_text.strip():
                return None

            retriever = HumanisticRetriever(n_results=max_results)
            result = retriever.retrieve(
                query=query_text,
                filters=HumanisticEvidenceFilter(framework="humanistic"),
            )

            if not result.retrieved:
                return None

            return result.to_framework_evidence_result()

        except Exception as exc:
            _log.warning(
                "humanistic.vector_retrieval_fallback",
                error=str(exc),
            )
            return None

    # ── 1. analyze_case ─────────────────────────────────────────────────────

    async def analyze_case(
        self,
        inp: Any,
        llm: LLMService | None = None,
    ) -> CaseAnalysis:
        extra = (
            "Analyse this case through the humanistic therapy lens. Identify:\n"
            "- Self-concept: how does the client perceive themselves? What self-narratives "
            "and self-structures are operating?\n"
            "- Incongruence: where does the client's actual experience not match their "
            "self-concept? What aspects of experience are being distorted or denied?\n"
            "- Conditions of worth: what conditional positive regard has the client "
            "internalised? What must they believe they must be to be worthy of love?\n"
            "- Congruence vs. incongruence: how aligned is the client's inner experience "
            "with their outer expression? Where are they presenting a façade?\n"
            "- Self-regard: is the client's self-regard conditional or unconditional? "
            "How harsh or accepting are they toward themselves?\n"
            "- Organismic valuing: is the client in touch with their own inner knowing, "
            "or are they relying on introjected values and external shoulds?\n"
            "- Blocks to self-actualisation: what is preventing the client from growing "
            "in directions that are genuinely their own?\n"
            "- Client strengths humanistic therapy can leverage: capacity for "
            "self-reflection, openness, willingness to connect, resilience\n\n"
            "IMPORTANT: Never assign diagnostic labels (e.g., 'major depressive disorder', 'GAD'). "
            "Describe patterns, behaviours, and experiential processes using descriptive "
            "clinical language grounded in the person-centred tradition."
        )
        try:
            result = await self._llm_generate_typed(
                CaseAnalysis, inp, extra, llm, template_name="analyze_case",
            )
            return _strip_diagnoses_from_result(result)
        except Exception as exc:
            _log.warning("humanistic.analyze_case_failed", error=str(exc))
            return CaseAnalysis(
                key_features=[],
                presenting_patterns=[],
                framework_observations=[],
                severity_indicators=[],
                strengths_and_resources=[],
                analysis_summary=f"Analysis unavailable: {exc}",
            )

    # ── 2. retrieve_evidence ────────────────────────────────────────────────

    async def retrieve_evidence(
        self,
        inp: Any,
        llm: LLMService | None = None,
    ) -> EvidenceResult:
        vector_result = await self._try_vector_retrieval(inp)
        if vector_result is not None:
            return vector_result

        extra = (
            "As a Humanistic therapy practitioner, synthesise evidence relevant to this case.\n"
            "Consider:\n"
            "- Rogers' necessary and sufficient conditions: meta-analyses showing empathy, "
            "UPR, and congruence predict therapeutic outcome\n"
            "- Common factors research: the therapeutic relationship accounts for the "
            "largest share of outcome variance (Lambert, Wampold, Norcross)\n"
            "- Process-experiential and emotion-focused therapy (EFT) outcome studies\n"
            "- Self-concept and self-discrepancy theory research (Higgins, Markus)\n"
            "- Self-compassion research and its relationship to UPR (Neff, Gilbert)\n"
            "- The organismic valuing process: growth-oriented humanistic neuroscience\n"
            "- Meta-analyses of person-centred and experiential therapies\n"
            "- Gestalt therapy process research: awareness, contact, field theory\n"
            "- Focusing (Gendlin) outcome research\n"
            "- Measures: Barrett-Lennard Relationship Inventory, SEI, SCL-90\n\n"
            "IMPORTANT: Never assign diagnostic labels. "
            "Describe patterns and processes using descriptive clinical language only."
        )
        try:
            result = await self._llm_generate_typed(
                EvidenceResult, inp, extra, llm, template_name="retrieve_evidence",
            )
            return _strip_diagnoses_from_result(result)
        except Exception as exc:
            _log.warning("humanistic.retrieve_evidence_failed", error=str(exc))
            return EvidenceResult(
                retrieved=[],
                synthesis=f"Evidence retrieval unavailable: {exc}",
                key_findings=[],
                evidence_gaps=[],
                confidence=0.0,
            )

    # ── 3. generate_case_conceptualization ──────────────────────────────────

    async def generate_case_conceptualization(
        self,
        inp: Any,
        llm: LLMService | None = None,
    ) -> CaseConceptualization:
        extra = (
            "Generate a comprehensive humanistic case conceptualisation using Rogers' self-theory.\n\n"
            "Structure the conceptualisation around:\n"
            "1. **Self-concept and self-structure**: What self-narratives, self-evaluations, "
            "and identity constructs are present? Describe the relationship between the "
            "real self (actualising tendency) and the ideal self (introjected conditions of worth).\n"
            "2. **Conditions of worth**: What conditional positive regard has the client "
            "internalised from significant others, culture, or society? What must they "
            "believe they must be or do to be worthy of love, acceptance, or respect? "
            "Examples: 'I must be perfect', 'I must not show weakness', "
            "'My feelings are wrong or shameful'.\n"
            "3. **Incongruence**: Where does the client's actual organismic experience "
            "clash with their self-concept? How does this create vulnerability, anxiety, "
            "or defensiveness (subception, distortion, denial)?\n"
            "4. **Self-regard**: What is the client's level of positive self-regard? "
            "Is it conditional (dependent on meeting conditions of worth) or unconditional? "
            "How does it affect their willingness to fully experience their inner world?\n"
            "5. **Organismic trusting**: Has the client lost trust in their own inner "
            "experiencing as a guide for living? Do they rely on external authorities, "
            "rules, or shoulds?\n"
            "6. **Maintaining cycles**: How do conditions of worth, incongruence, and "
            "defensiveness interact to maintain the client's distress and block growth?\n"
            "7. **Self-actualisation tendency**: What is the client's natural growth "
            "direction, and what is blocking it?\n\n"
            "IMPORTANT: Never assign diagnostic labels. "
            "Describe patterns, behaviours, and experiential processes using the "
            "language of self-theory — conditions of worth, incongruence, self-regard, "
            "organismic valuing, and the actualising tendency."
        )
        try:
            result = await self._llm_generate_typed(
                CaseConceptualization, inp, extra, llm,
                template_name="case_conceptualization",
            )
            return _strip_diagnoses_from_result(result)
        except Exception as exc:
            _log.warning("humanistic.conceptualization_failed", error=str(exc))
            return CaseConceptualization(
                framework_formulation=f"Conceptualization unavailable: {exc}",
                maintaining_factors=[],
                core_mechanisms=[],
                developmental_factors=[],
                conceptualization_summary=f"Unable to generate: {exc}",
            )

    # ── 4. generate_therapeutic_focus ───────────────────────────────────────

    async def generate_therapeutic_focus(
        self,
        inp: Any,
        llm: LLMService | None = None,
    ) -> TherapeuticFocusResult:
        extra = (
            "Identify primary therapeutic focus areas from a humanistic therapy perspective.\n"
            "Focus on Rogers' necessary and sufficient conditions:\n"
            "- **Psychological contact**: ensuring there is a genuine meeting between "
            "therapist and client as two persons\n"
            "- **Client incongruence**: creating conditions for the client to safely "
            "contact and integrate previously denied or distorted experience\n"
            "- **Therapist congruence**: being real, transparent, and genuine — not "
            "hiding behind a professional role\n"
            "- **Unconditional positive regard**: offering deep acceptance and prizing "
            "of the whole person, beyond evaluation or judgment\n"
            "- **Empathic understanding**: striving to grasp the client's internal frame "
            "of reference moment by moment\n"
            "- **Client perception**: ensuring the client experiences being understood "
            "and accepted\n\n"
            "Consider also:\n"
            "- Supporting the actualising tendency: trusting the client's innate growth direction\n"
            "- Phenomenological inquiry: exploring the client's lived experience directly\n"
            "- Experiential processing (focusing): helping the client contact their felt sense\n"
            "- Relational depth: working at the level of I-Thou relating (Buber)\n"
            "- Gestalt contact and awareness: what is the client avoiding contact with?\n"
            "- Working with self-splits and polarities (two-chair work)\n\n"
            "IMPORTANT: Never assign diagnostic labels. "
            "Describe relational and experiential processes using humanistic therapy language."
        )
        try:
            result = await self._llm_generate_typed(
                TherapeuticFocusResult, inp, extra, llm,
                template_name="therapeutic_focus",
            )
            return _strip_diagnoses_from_result(result)
        except Exception as exc:
            _log.warning("humanistic.therapeutic_focus_failed", error=str(exc))
            return TherapeuticFocusResult(
                primary_focus_areas=["Unable to determine focus"],
                secondary_focus_areas=[],
                staging_and_sequencing=f"Staging unavailable: {exc}",
                focus_rationale=f"Rationale unavailable: {exc}",
            )

    # ── 5. generate_intervention_plan ───────────────────────────────────────

    async def generate_intervention_plan(
        self,
        inp: Any,
        llm: LLMService | None = None,
    ) -> InterventionPlanResult:
        extra = (
            "Generate a detailed humanistic therapy intervention plan.\n\n"
            "Humanistic 'interventions' are primarily relational rather than technical. "
            "The core intervention is the therapist's way of being:\n\n"
            "**Relational foundations (the six conditions):**\n"
            "- **Empathic responding**: reflect the client's internal frame of reference. "
            "'It sounds like you're saying...', 'Let me see if I understand...', "
            "'What I'm hearing is...'\n"
            "- **Unconditional positive regard**: communicate deep acceptance without "
            "evaluation, even for difficult or shameful material. "
            "'I accept all of you, even this part that you struggle with.'\n"
            "- **Congruence/genuineness**: be real, transparent, and authentic. "
            "Share your own experience when it serves the process. "
            "'I notice something happening in the room right now...'\n"
            "- **Present-moment relational process**: notice and name what is happening "
            "between you and the client in the here-and-now\n\n"
            "**Experiential interventions:**\n"
            "- **Focusing (Gendlin)**: help the client turn inward to the 'felt sense'. "
            "'Take a moment and notice what you feel in your body when you say that.'\n"
            "- **Process-directive responding**: guide attention to emerging experience "
            "without directing content. 'What's happening right now as you say that?'\n"
            "- **Gestalt two-chair dialogue**: for splits within the self (critic vs. "
            "experiencing self, 'top dog' vs. 'underdog')\n"
            "- **Empty chair work**: for unfinished business with significant others\n"
            "- **Evocative unfolding**: explore puzzling or problematic reactions to "
            "uncover their implicit meaning structure\n\n"
            "**Relational awareness interventions:**\n"
            "- Name relational patterns as they emerge in the therapeutic dyad\n"
            "- Explore how contact and awareness are interrupted (projection, introjection, "
            "retroflection, deflection, confluence in Gestalt terms)\n"
            "- Support the client's organismic valuing process: "
            "'What does your own inner knowing tell you about this?'\n\n"
            "Techniques serve the relationship, not the other way around.\n\n"
            "IMPORTANT: Never assign diagnostic labels. "
            "Describe interventions using relational and experiential language only."
        )
        try:
            result = await self._llm_generate_typed(
                InterventionPlanResult, inp, extra, llm,
                template_name="intervention_plan",
            )
            return _strip_diagnoses_from_result(result)
        except Exception as exc:
            _log.warning("humanistic.intervention_plan_failed", error=str(exc))
            return InterventionPlanResult(
                interventions=[],
                provisional_strategy=f"Plan unavailable: {exc}",
                expected_outcomes=[],
            )

    # ── 6. generate_session_objectives ──────────────────────────────────────

    async def generate_session_objectives(
        self,
        inp: Any,
        llm: LLMService | None = None,
    ) -> SessionObjectivesResult:
        extra = (
            "Define humanistic therapy session-level objectives and goals.\n"
            "Session structure is emergent, client-led, and relational. "
            "Objectives are direction-pointers, not fixed targets:\n\n"
            "**Immediate session objectives:**\n"
            "- Deepen empathic attunement to the client's internal frame of reference\n"
            "- Help the client feel heard, understood, and accepted at a deep level\n"
            "- Create a safe relational space for contact with avoided experience\n"
            "- Invite the client to notice their present-moment felt sense without judgment\n"
            "- Follow the client's lead: what is most alive for them right now?\n"
            "- Name and accept whatever emerges without fixing or changing it\n"
            "- Be fully present and genuine with the client\n\n"
            "**Short-term goals (early sessions):**\n"
            "- Client experiences being truly heard and understood\n"
            "- Client begins to relax defences and share more vulnerably\n"
            "- Client makes contact with previously denied or distorted experience\n"
            "- Increased trust in the therapeutic relationship\n"
            "- Reduction in conditions of worth: client begins to accept parts of "
            "self that were previously rejected\n"
            "- Client begins to extend the therapist's UPR to themselves\n"
            "- Client can identify and stay with felt senses without rushing to interpret\n\n"
            "**Long-term goals:**\n"
            "- Increased congruence: self-concept aligns more with actual experiencing\n"
            "- Greater unconditional positive self-regard\n"
            "- Increased organismic trusting: reliance on inner knowing over external shoulds\n"
            "- Reduced defensiveness: openness to new experience without distortion\n"
            "- The actualising tendency is freed: growth in directions that are genuinely the client's own\n"
            "- Continued growth and actualisation beyond therapy\n\n"
            "**Progress indicators:**\n"
            "- More authentic, less defensive language about self\n"
            "- Client reports feeling more alive, present, and whole\n"
            "- Spontaneous vulnerable sharing without prompting\n"
            "- Greater self-acceptance and self-compassion\n"
            "- Choices based on organismic valuing rather than conditions of worth\n"
            "- Reduced rigidity in self-narratives; fluidity in self-description\n\n"
            "IMPORTANT: Never assign diagnostic labels. "
            "Describe objectives using the language of self-theory and relational process."
        )
        try:
            result = await self._llm_generate_typed(
                SessionObjectivesResult, inp, extra, llm,
                template_name="session_objectives",
            )
            return _strip_diagnoses_from_result(result)
        except Exception as exc:
            _log.warning("humanistic.session_objectives_failed", error=str(exc))
            return SessionObjectivesResult(
                immediate_objectives=[],
                short_term_goals=["Unable to generate goals"],
                long_term_goals=[],
                progress_indicators=[],
            )

    # ── 7. generate_followup_questions ──────────────────────────────────────

    async def generate_followup_questions(
        self,
        inp: Any,
        llm: LLMService | None = None,
    ) -> FollowupQuestionsResult:
        extra = (
            "Suggest humanistic therapy follow-up questions for assessment and therapy.\n\n"
            "Frame all questions in a curious, empathic, non-directive manner. "
            "The goal is to deepen exploration, not extract information.\n\n"
            "**Self-concept and self-regard:**\n"
            "- 'How would you describe yourself — who are you at your core?'\n"
            "- 'What parts of yourself do you accept, and what parts do you struggle with?'\n"
            "- 'What would it feel like to accept yourself fully, just as you are right now?'\n"
            "- 'When you look at yourself, whose voice do you hear evaluating you?'\n\n"
            "**Conditions of worth:**\n"
            "- 'What did you learn growing up about what makes a person worthy of love?'\n"
            "- 'What conditions do you place on your own self-acceptance?'\n"
            "- 'What would happen if you stopped trying to meet those conditions?'\n"
            "- 'What would it mean to trust your own inner sense of what is right for you?'\n\n"
            "**Incongruence exploration:**\n"
            "- 'I hear [A] and also [B]. Can we explore that gap together?'\n"
            "- 'What happens inside you when your experience doesn't match who you think you should be?'\n"
            "- 'Where in your life do you feel most like you're pretending or holding back?'\n\n"
            "**Empathic probes:**\n"
            "- 'Help me understand what that experience is like from the inside.'\n"
            "- 'When you say [client's word], what does that actually feel like for you?'\n"
            "- 'Take a moment and just be with that. What do you notice in your body?'\n"
            "- 'I notice you [tensed up / looked away / went quiet]. What's happening right now?'\n\n"
            "**Organismic valuing:**\n"
            "- 'What does your own inner sense tell you about this, setting aside what others expect?'\n"
            "- 'If you could trust your own knowing completely, what would change?'\n\n"
            "**Growth and actualisation:**\n"
            "- 'If you could be more fully yourself, what would that look like?'\n"
            "- 'What direction does your life want to move in, if you let it?'\n"
            "- 'What would it mean to live a life that is truly your own?'\n\n"
            "**Engagement:**\n"
            "- 'How does it feel to be speaking about this with me today?'\n"
            "- 'What is it like for you to be in this room right now?'\n"
            "- 'What has it been like for you to be truly heard in this way?'\n\n"
            "IMPORTANT: Never assign diagnostic labels. "
            "Frame all questions in a collaborative, empathic, experiential manner "
            "consistent with person-centred and humanistic principles."
        )
        try:
            result = await self._llm_generate_typed(
                FollowupQuestionsResult, inp, extra, llm,
                template_name="followup_questions",
            )
            return _strip_diagnoses_from_result(result)
        except Exception as exc:
            _log.warning("humanistic.followup_questions_failed", error=str(exc))
            return FollowupQuestionsResult(
                assessment_questions=[],
                therapeutic_probes=[],
                outcome_monitoring=[],
                engagement_questions=[],
            )

    # ── 8. generate_limitations ─────────────────────────────────────────────

    async def generate_limitations(
        self,
        inp: Any,
        llm: LLMService | None = None,
    ) -> LimitationsResult:
        extra = (
            "Describe limitations and contraindications of applying humanistic therapy to this case.\n\n"
            "**Situations where humanistic therapy may be less suitable:**\n"
            "- Severe and persistent distress requiring structured symptom management "
            "(intense intrusive re-experiencing, disabling compulsive patterns, "
            "severe substance withdrawal)\n"
            "- High acute risk: active suicidal or self-harming behaviour requiring "
            "crisis protocols and behavioural stabilisation\n"
            "- Cognitive or communication barriers: significantly limited capacity for "
            "self-reflection, verbal processing, or relational contact (severe cognitive "
            "impairment, acute psychosis, advanced dementia)\n"
            "- Mandated or involuntary treatment: non-directive stance is challenging "
            "when the client has not chosen therapy\n"
            "- Clear need for directive intervention: specific phobias needing structured "
            "exposure, or specific skill deficits requiring psychoeducation\n"
            "- Litigation or disability assessment contexts: humanistic acceptance "
            "conflicts with adversarial or evaluative demands\n\n"
            "**Precautions:**\n"
            "- Intellectualisation risk: clients may use humanistic concepts as avoidance"
            " rather than genuine experiential contact\n"
            "- Cultural sensitivity: individualism, autonomy, and verbal expression may "
            "not align with all cultural backgrounds; interdependence and non-verbal "
            "processing may be primary\n"
            "- Therapist self-care: deep empathic attunement carries risk of empathy "
            "fatigue and vicarious distress\n"
            "- Non-directive stances ≠ passive; they require active, engaged presence\n"
            "- Some clients need structure; consider gradual introduction of more "
            "directive elements if needed\n\n"
            "**Alternative frameworks:**\n"
            "- **CBT**: structured, goal-oriented; complements humanistic therapy with "
            "specific symptom reduction and skills training\n"
            "- **ACT**: process-based approach with explicit values and defusion; "
            "more structured than pure person-centred\n"
            "- **DBT**: significant emotional dysregulation and self-harming behaviour "
            "requiring structured skill-building\n"
            "- **EMDR**: trauma-specific processing when identifiable memories drive distress\n"
            "- **Psychodynamic**: insight into unconscious attachment patterns\n"
            "- **Systemic/family therapy**: relational dynamics as primary context\n\n"
            "IMPORTANT: Never assign diagnostic labels. "
            "Describe limitations and alternatives using descriptive clinical and relational language."
        )
        try:
            result = await self._llm_generate_typed(
                LimitationsResult, inp, extra, llm, template_name="limitations",
            )
            return _strip_diagnoses_from_result(result)
        except Exception as exc:
            _log.warning("humanistic.limitations_failed", error=str(exc))
            return LimitationsResult(
                contraindications=[],
                precautions=[],
                alternative_frameworks=[],
                caveats=[f"Limitations unavailable: {exc}"],
            )
