"""
clinical/framework/act/plugin.py — Acceptance and Commitment Therapy framework plugin.

Production-ready overrides for all 8 lifecycle methods with:
- ACT-specific clinical content in every method
- "Never diagnose" enforcement at prompt and post-processing level
- Vector store retrieval with metadata filtering for evidence
- Graceful fallback to LLM on retrieval failure
- Full hexaflex coverage (acceptance, defusion, present moment, self-as-context, values, committed action)
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

# ── Never-diagnose helpers (same patterns as CBT plugin) ──────────────

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
class ACTPlugin(ClinicalFramework):

    @classmethod
    def metadata(cls) -> FrameworkMetadata:
        return FrameworkMetadata(
            id="act",
            display_name="Acceptance and Commitment Therapy",
            version="2.0.0",
            description=(
                "Acceptance and Commitment Therapy (ACT) is a third-wave behavioural "
                "therapy that uses acceptance, mindfulness, and values-based action to "
                "increase psychological flexibility. ACT views psychological distress as "
                "rooted in experiential avoidance and cognitive fusion — the tendency to "
                "get entangled with thoughts and avoid uncomfortable inner experiences. "
                "Treatment helps clients defuse from unhelpful thoughts, accept difficult "
                "feelings, connect to the present moment, clarify personal values, and "
                "commit to values-driven action. ACT is process-based rather than "
                "protocol-driven, targeting the six core processes of the hexaflex: "
                "acceptance, defusion, self-as-context, contact with the present moment, "
                "values, and committed action."
            ),
            supported_use_cases=[
                "Chronic pain and illness",
                "Anxiety (GAD, social anxiety, panic)",
                "Depression and low mood",
                "Chronic stress and burnout",
                "Obsessive-compulsive patterns",
                "Substance use and addictive behaviours",
                "Life transitions and meaning-making",
            ],
            literature_collection="act_literature",
            retrieval_strategy=RetrievalStrategy.HYBRID,
            intervention_style=InterventionStyle(
                primary_focus=(
                    "Psychological flexibility through acceptance, defusion, "
                    "values-based action, and present-moment awareness"
                ),
                techniques=[
                    "Cognitive defusion exercises (leaves on a stream, name the story)",
                    "Acceptance and willingness interventions (creative hopelessness, making room)",
                    "Mindfulness and present-moment awareness",
                    "Values clarification and committed action planning",
                    "Self-as-context work (observing self, the chessboard)",
                    "Metaphors for psychological flexibility (passengers on the bus, tug of war)",
                ],
                session_structure=(
                    "Semi-structured; sessions typically begin with a brief mindfulness or "
                    "centring exercise, followed by experiential work on one or more ACT "
                    "processes, and end with committed action planning"
                ),
                therapeutic_stance=(
                    "Active, collaborative, experiential; the therapist models acceptance "
                    "and defusion, fosters creative hopelessness to help the client see "
                    "what hasn't worked, and works alongside the client as a fellow human "
                    "being, not an expert who fixes problems"
                ),
            ),
            tags=["act", "acceptance", "mindfulness", "values", "defusion", "third-wave", "psychological flexibility", "experiential"],
            citations=[
                "Hayes, S. C., Strosahl, K. D., & Wilson, K. G. (2011). Acceptance and Commitment Therapy: The Process and Practice of Mindful Change (2nd ed.).",
                "Luoma, J. B., Hayes, S. C., & Walser, R. D. (2017). Learning ACT: An Acceptance and Commitment Therapy Skills-Training Manual for Therapists (2nd ed.).",
                "Hayes, S. C., & Smith, S. (2005). Get Out of Your Mind and Into Your Life.",
            ],
        )

    @classmethod
    def is_applicable(cls, case_understanding: dict) -> bool:
        topics = _safe_topics(case_understanding)
        keywords = {
            "anxiety", "depression", "stress", "burnout", "chronic",
            "pain", "illness", "obsessive", "compulsive", "substance",
            "addiction", "avoidance", "fusion", "values", "meaning",
            "uncertainty", "worry", "rumination", "experiential",
            "acceptance", "mindfulness", "flexibility",
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
        return "ACT evidence for clinical case"

    # ── Helper: ACT evidence retrieval with ACTRetriever ───────────────────

    async def _try_vector_retrieval(
        self,
        inp: Any,
        max_results: int = 5,
    ) -> EvidenceResult | None:
        """Attempt to retrieve ACT evidence using the ACTRetriever.

        Uses hybrid retrieval (dense + BM25 + RRF fusion) with cross-encoder
        reranking, always scoped to ACT literature via ``framework="act"``
        metadata filtering.  Returns ``None`` to signal the caller should
        fall back to LLM-based evidence generation.
        """
        try:
            from clinical.retrieval.act_retriever import (
                ACTEvidenceFilter,
                ACTRetriever,
            )

            query_text = self._build_search_query(inp)
            if not query_text.strip():
                return None

            retriever = ACTRetriever(n_results=max_results)
            result = retriever.retrieve(
                query=query_text,
                filters=ACTEvidenceFilter(framework="act"),
            )

            if not result.retrieved:
                return None

            return result.to_framework_evidence_result()

        except Exception as exc:
            _log.warning(
                "act.vector_retrieval_fallback",
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
            "Analyse this case through the ACT lens of psychological flexibility. Identify:\n"
            "- Experiential avoidance: what private experiences is the client avoiding or struggling to control?\n"
            "- Cognitive fusion: what thoughts, stories, or rules is the client entangled with?\n"
            "- Contact with present moment: is the client stuck in past (rumination) or future (worry)?\n"
            "- Self-as-content: what conceptualised self-stories are limiting flexibility? (e.g., 'I'm broken', 'I'm an anxious person')\n"
            "- Values disconnection: what valued directions have been lost due to avoidance and fusion?\n"
            "- Committed action barriers: where is the client stuck in inaction, impulsivity, or rigid persistence?\n"
            "- Client strengths ACT can leverage: openness, willingness to try new things, values awareness\n\n"
            "IMPORTANT: Never assign diagnostic labels (e.g., 'major depressive disorder', 'GAD'). "
            "Describe patterns, behaviours, and cognitive processes using descriptive clinical language only. "
            "Focus on the function of behaviour, not its form."
        )
        try:
            result = await self._llm_generate_typed(
                CaseAnalysis, inp, extra, llm, template_name="analyze_case",
            )
            return _strip_diagnoses_from_result(result)
        except Exception as exc:
            _log.warning("act.analyze_case_failed", error=str(exc))
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
        # Try vector store retrieval first with metadata filtering
        vector_result = await self._try_vector_retrieval(inp)
        if vector_result is not None:
            return vector_result

        # Fall back to LLM-based evidence generation
        extra = (
            "As an ACT practitioner, synthesise evidence relevant to this case.\n"
            "Consider:\n"
            "- Meta-analyses and RCTs of ACT for the identified problem areas\n"
            "- Process-based evidence: psychological flexibility as a transdiagnostic mechanism\n"
            "- ACT models: RFT underpinnings, Psychological Flexibility Model, hexaflex\n"
            "- Specific protocols: ACT for depression, ACT for anxiety, ACT for chronic pain, "
            "ACT for OCD patterns, ACT for substance use, ACT for burnout\n"
            "- Process measures: AAQ-II, VQ, CFQ, CompACT\n"
            "- Comparisons: ACT vs. CBT, mechanisms of change in ACT\n\n"
            "IMPORTANT: Never assign diagnostic labels. "
            "Describe patterns and processes using descriptive clinical language only. "
            "Focus on function over form."
        )
        try:
            result = await self._llm_generate_typed(
                EvidenceResult, inp, extra, llm, template_name="retrieve_evidence",
            )
            return _strip_diagnoses_from_result(result)
        except Exception as exc:
            _log.warning("act.retrieve_evidence_failed", error=str(exc))
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
            "Generate a comprehensive ACT case conceptualisation.\n"
            "Structure around the psychological (in)flexibility model (hexaflex):\n"
            "1. Experiential avoidance: what control strategies does the client use and at what cost?\n"
            "2. Cognitive fusion: what thoughts, beliefs, rules, and self-stories is the client fused with?\n"
            "3. Conceptualised self: what rigid self-narratives limit flexibility?\n"
            "4. Dominance of past/future: where is attention stuck in rumination or worry?\n"
            "5. Disconnection from values: what valued directions have been abandoned or obscured?\n"
            "6. Inaction, impulsivity, or avoidant persistence: where is the client stuck?\n"
            "7. Maintaining cycles: how do these six processes interact to perpetuate suffering?\n"
            "8. Contact with the present moment: capacity for flexible attention to here-and-now\n\n"
            "IMPORTANT: Never assign diagnostic labels. "
            "Describe patterns, behaviours, and cognitive processes using descriptive clinical language only. "
            "Focus on the function of behaviour, not its form."
        )
        try:
            result = await self._llm_generate_typed(
                CaseConceptualization, inp, extra, llm,
                template_name="case_conceptualization",
            )
            return _strip_diagnoses_from_result(result)
        except Exception as exc:
            _log.warning("act.conceptualization_failed", error=str(exc))
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
            "Identify primary therapeutic focus areas from an ACT perspective.\n"
            "Consider the six core ACT processes:\n"
            "- Creative hopelessness: helping the client contact the costs of their control agenda\n"
            "- Acceptance and willingness: making room for difficult private experiences in service of values\n"
            "- Cognitive defusion: unhooking from the content of thoughts (I notice I'm having the thought that...)\n"
            "- Contact with the present moment: building flexible, intentional attention (not formal meditation)\n"
            "- Self-as-context: contacting the observing self that is distinct from thoughts, feelings, and stories\n"
            "- Values clarification: exploring what truly matters to the client (not what they 'should' care about)\n"
            "- Committed action: building patterns of values-consistent behaviour\n"
            "- Staging: what to address first based on the client's current suffering and willingness\n\n"
            "IMPORTANT: Never assign diagnostic labels. "
            "Describe patterns, behaviours, and cognitive processes using descriptive clinical language only."
        )
        try:
            result = await self._llm_generate_typed(
                TherapeuticFocusResult, inp, extra, llm,
                template_name="therapeutic_focus",
            )
            return _strip_diagnoses_from_result(result)
        except Exception as exc:
            _log.warning("act.therapeutic_focus_failed", error=str(exc))
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
            "Generate a detailed ACT intervention plan.\n\n"
            "Acceptance and willingness interventions:\n"
            "- Creative hopelessness: explore what the client has tried and what it has cost\n"
            "- 'Tug of war with a monster' — the goal is to drop the rope, not win the fight\n"
            "- 'Man in the hole' — the digging (controlling) is the problem, not the solution\n"
            "- 'Two scales' — feeling better vs. living better; ACT focuses on living better\n"
            "- Willingness: choosing to make room for difficult experiences in service of values\n"
            "- Body-based acceptance: breathing into areas of tension, making space for sensations\n\n"
            "Defusion interventions:\n"
            "- 'I'm having the thought that...' — unhooking from thought content\n"
            "- 'Passengers on the bus' — driving the bus in a valued direction despite passengers\n"
            "- 'Leaves on a stream' — visualising thoughts as leaves floating downstream\n"
            "- 'Thank your mind' — acknowledging the mind's activity without buying into it\n"
            "- Sing the thought / say it in a funny voice / repeat it rapidly\n"
            "- Label the process: noticing when the 'I can't' story shows up again\n\n"
            "Present-moment interventions:\n"
            "- Mindful breathing with anchoring to the breath\n"
            "- Five senses grounding exercises\n"
            "- 'Just noticing' — observing present-moment experience without evaluation\n\n"
            "Self-as-context interventions:\n"
            "- 'The chessboard' metaphor — you are the board, not the pieces\n"
            "- 'Sky and weather' — thoughts are weather; you are the sky\n"
            "- The observing self exercise — noticing the 'you' that has been aware across time\n\n"
            "Values clarification:\n"
            "- Values card sort or exploration: relationships, work, health, leisure, community, spirituality\n"
            "- 'What do you want your life to stand for?'\n"
            "- 'If pain were no object, what would you do?'\n"
            "- Distinguish values (ongoing directions) from goals (achievable outcomes)\n\n"
            "Committed action:\n"
            "- Specific behavioural commitments (what, when, where, with whom)\n"
            "- 'What is one small step you could take this week?'\n"
            "- Anticipate fusion/avoidance barriers and plan responses\n"
            "- 'Are you willing to have that feeling while doing this thing that matters?'\n\n"
            "IMPORTANT: Never assign diagnostic labels. "
            "Describe patterns, behaviours, and cognitive processes using descriptive clinical language only. "
            "Focus on function over form."
        )
        try:
            result = await self._llm_generate_typed(
                InterventionPlanResult, inp, extra, llm,
                template_name="intervention_plan",
            )
            return _strip_diagnoses_from_result(result)
        except Exception as exc:
            _log.warning("act.intervention_plan_failed", error=str(exc))
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
            "Define ACT session-level objectives and goals.\n"
            "ACT session structure: brief centring exercise -> bridge from previous session -> "
            "collaborative focus-setting -> experiential working part -> committed action planning -> optional closing\n\n"
            "Immediate objectives:\n"
            "- Wake up from fusion with a specific thought or story\n"
            "- Make experiential contact with the costs of avoidance\n"
            "- Willingly make room for one difficult feeling or sensation\n"
            "- Clarify what matters in one life domain\n"
            "- Take one small committed action step\n"
            "- Strengthen present-moment awareness\n\n"
            "Short-term goals (3-6 sessions):\n"
            "- Distinguish having a thought from buying into a thought\n"
            "- Identify control strategies and contact their workability\n"
            "- Demonstrate willingness to have difficult inner experiences while doing valued activities\n"
            "- Articulate personal values in at least two life domains\n"
            "- Take committed actions consistent with identified values\n"
            "- Notice when attention has drifted and return flexibly\n\n"
            "Long-term goals:\n"
            "- Build broader, more flexible behavioural repertoire\n"
            "- Spontaneously defuse from unhelpful thoughts without instructions\n"
            "- Make consistently values-consistent choices despite difficult inner experiences\n"
            "- Reduced dominance of conceptualised self (self-stories no longer drive behaviour)\n"
            "- Valued living (not symptom reduction) is the primary outcome metric\n\n"
            "IMPORTANT: Never assign diagnostic labels. "
            "Describe patterns, behaviours, and cognitive processes using descriptive clinical language only. "
            "Focus on building psychological flexibility, not on symptom reduction."
        )
        try:
            result = await self._llm_generate_typed(
                SessionObjectivesResult, inp, extra, llm,
                template_name="session_objectives",
            )
            return _strip_diagnoses_from_result(result)
        except Exception as exc:
            _log.warning("act.session_objectives_failed", error=str(exc))
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
            "Suggest ACT follow-up questions for assessment and therapy.\n\n"
            "Creative hopelessness / control assessment:\n"
            "- 'What have you tried to get rid of or control these difficult feelings?'\n"
            "- 'How has that been working for you — in the long run?'\n"
            "- 'What has it cost you to keep fighting these experiences?'\n"
            "- 'What would you be doing if you weren't struggling so much?'\n\n"
            "Defusion probes:\n"
            "- 'When that thought shows up, what does it tell you to do?'\n"
            "- 'Can you notice that thought as just a thought, separate from the situation?'\n"
            "- 'What happens when you add a noticing phrase like I notice I am having the thought that before the thought?\n"
            "- 'What's the story your mind is telling you about this?'\n\n"
            "Acceptance and willingness:\n"
            "- 'What feeling are you most unwilling to have right now?'\n"
            "- 'If you could make room for that feeling, just for a moment, what would that be like?'\n"
            "- 'What would you be willing to feel in order to do something that matters to you?'\n"
            "- 'Can you find a way to breathe into where this shows up in your body?'\n\n"
            "Present-moment awareness:\n"
            "- 'Right now, what are you noticing in your body, in your mind, around you?'\n"
            "- 'Can you just describe what you notice without judging it?'\n"
            "- 'Where in your body do you feel this most strongly?'\n\n"
            "Values exploration:\n"
            "- 'What matters most to you in your life?'\n"
            "- 'If you could choose what your life is about, what would you choose?'\n"
            "- 'What would you do if pain were no object?'\n"
            "- 'What kind of [partner/parent/friend] do you want to be?'\n\n"
            "Committed action:\n"
            "- 'What is one small step you could take this week in that valued direction?'\n"
            "- 'What might get in the way? What are you willing to have show up?'\n"
            "- 'How will you know you've taken that step?'\n\n"
            "Frame all questions in a collaborative, curious, experiential manner.\n\n"
            "IMPORTANT: Never assign diagnostic labels. "
            "Describe patterns, behaviours, and cognitive processes using descriptive clinical language only."
        )
        try:
            result = await self._llm_generate_typed(
                FollowupQuestionsResult, inp, extra, llm,
                template_name="followup_questions",
            )
            return _strip_diagnoses_from_result(result)
        except Exception as exc:
            _log.warning("act.followup_questions_failed", error=str(exc))
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
            "Describe limitations and contraindications of applying ACT to this case.\n"
            "Contraindications / cautionary indicators:\n"
            "- Active psychosis, mania, or severe cognitive impairment limiting experiential work\n"
            "- High acute risk requiring crisis intervention before ACT\n"
            "- Inability to engage with metaphorical or experiential language\n"
            "- Clients seeking direct symptom reduction rather than process-based work\n"
            "- Mandated treatment where willing engagement may be limited\n\n"
            "Precautions:\n"
            "- Risk of intellectualisation (client learns ACT language but avoids experiential contact)\n"
            "- Cultural adaptation of metaphors and exercises required\n"
            "- ACT can feel counterintuitive for clients expecting problem-solving approaches\n"
            "- Titrate contact with avoided private experiences carefully\n"
            "- Distinguish therapist values from client values in values work\n\n"
            "Alternative frameworks:\n"
            "- CBT: structured, symptom-focused, and psychoeducational approach\n"
            "- DBT: significant emotional dysregulation and self-harming behaviour\n"
            "- CFT: high shame and self-criticism as primary barriers\n"
            "- Schema therapy: pervasive early maladaptive schemas\n"
            "- ERP/Exposure-based CBT: specific phobias or OCD with habituation protocols\n"
            "- Psychodynamic: insight into unconscious processes\n"
            "- Systemic: family or relational dynamics are primary\n\n"
            "IMPORTANT: Never assign diagnostic labels. "
            "Describe patterns, behaviours, and cognitive processes using descriptive clinical language only."
        )
        try:
            result = await self._llm_generate_typed(
                LimitationsResult, inp, extra, llm, template_name="limitations",
            )
            return _strip_diagnoses_from_result(result)
        except Exception as exc:
            _log.warning("act.limitations_failed", error=str(exc))
            return LimitationsResult(
                contraindications=[],
                precautions=[],
                alternative_frameworks=[],
                caveats=[f"Limitations unavailable: {exc}"],
            )
