"""
clinical/framework/cbt/plugin.py — Cognitive Behavioral Therapy framework plugin.

Production-ready overrides for all 8 lifecycle methods with:
- CBT-specific clinical content in every method
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
from clinical.framework.prompts.manager import PromptManager
from clinical.framework.registry import register_framework
from clinical.framework.utils import _safe_topics
from app_logging.logger import get_logger

if TYPE_CHECKING:
    from clinical.llm import LLMService

_log = get_logger(__name__)

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
class CBTPlugin(ClinicalFramework):

    @classmethod
    def metadata(cls) -> FrameworkMetadata:
        return FrameworkMetadata(
            id="cbt",
            display_name="Cognitive Behavioral Therapy",
            version="2.0.0",
            description=(
                "Cognitive Behavioral Therapy (CBT) is a structured, time-limited psychotherapy "
                "that targets the relationships between thoughts, feelings, and behaviours. "
                "It is based on the cognitive model: distorted or unhelpful thinking patterns "
                "influence emotions and behaviours, and modifying these patterns leads to "
                "symptom improvement. CBT is problem-focused and goal-oriented, using both "
                "cognitive techniques (cognitive restructuring, behavioural experiments) and "
                "behavioural techniques (exposure, activity scheduling, skills training)."
            ),
            supported_use_cases=[
                "Depressive disorders",
                "Anxiety disorders (GAD, panic, social anxiety, specific phobias)",
                "Obsessive-compulsive disorder",
                "Post-traumatic stress disorder",
                "Eating disorders",
                "Insomnia and sleep difficulties",
                "Chronic pain management",
                "Anger management",
            ],
            literature_collection="cbt_literature",
            retrieval_strategy=RetrievalStrategy.HYBRID,
            intervention_style=InterventionStyle(
                primary_focus="Identifying and modifying unhelpful thinking patterns and behaviours",
                techniques=[
                    "Cognitive restructuring and thought records",
                    "Behavioural experiments and hypothesis testing",
                    "Exposure therapy and response prevention",
                    "Activity scheduling and behavioural activation",
                    "Socratic questioning and guided discovery",
                ],
                session_structure=(
                    "Highly structured; each session follows a consistent agenda: mood check, "
                    "bridge from previous session, agenda setting, review of homework, session "
                    "content, new homework assignment, and summary with feedback"
                ),
                therapeutic_stance=(
                    "Active, directive, collaborative; therapist acts as a teacher-coach, "
                    "providing psychoeducation and guiding the client to become their own therapist"
                ),
            ),
            tags=["cbt", "cognitive", "behavioural", "structured", "time-limited", "evidence-based"],
            citations=[
                "Beck, A. T. (1979). Cognitive Therapy and the Emotional Disorders.",
                "Beck, J. S. (2021). Cognitive Behavior Therapy: Basics and Beyond (3rd ed.).",
            ],
        )

    @classmethod
    def is_applicable(cls, case_understanding: dict) -> bool:
        topics = _safe_topics(case_understanding)
        keywords = {
            "anxiety", "depression", "panic", "phobia", "obsessive",
            "compulsive", "trauma", "ptsd", "eating", "insomnia",
            "anger", "worry", "rumination", "avoidance", "safety",
            "thought", "cognition", "belief", "behaviour", "habit",
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
        return "CBT evidence for clinical case"

    # ── Helper: CBT evidence retrieval with CBTRetriever ───────────────────

    async def _try_vector_retrieval(
        self,
        inp: Any,
        max_results: int = 5,
    ) -> EvidenceResult | None:
        """Attempt to retrieve CBT evidence using the CBTRetriever.

        Uses hybrid retrieval (dense + BM25 + RRF fusion) with cross-encoder
        reranking, always scoped to CBT literature via ``framework="cbt"``
        metadata filtering.  Returns ``None`` to signal the caller should
        fall back to LLM-based evidence generation.
        """
        try:
            from clinical.retrieval import CBTRetriever, CBTEvidenceFilter

            query_text = self._build_search_query(inp)
            if not query_text.strip():
                return None

            retriever = CBTRetriever(n_results=max_results)
            result = retriever.retrieve(
                query=query_text,
                filters=CBTEvidenceFilter(framework="cbt"),
            )

            if not result.retrieved:
                return None

            return result.to_framework_evidence_result()

        except Exception as exc:
            _log.warning(
                "cbt.vector_retrieval_fallback",
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
            "Analyse this case through the CBT lens. Identify:\n"
            "- Negative automatic thoughts (NATs) and their content\n"
            "- Cognitive distortions (catastrophising, black-and-white thinking, mind reading, "
            "personalisation, overgeneralisation, emotional reasoning, should-statements)\n"
            "- Core beliefs about self, others, and the world\n"
            "- Intermediate beliefs (rules, attitudes, assumptions)\n"
            "- The cognitive triad (self, world/others, future)\n"
            "- Safety behaviours, avoidance, and behavioural patterns\n"
            "- Client strengths CBT can leverage (motivation, insight, homework willingness)\n\n"
            "IMPORTANT: Never assign diagnostic labels (e.g., 'major depressive disorder', 'GAD'). "
            "Describe patterns, behaviours, and cognitive processes using descriptive clinical language only."
        )
        try:
            result = await self._llm_generate_typed(
                CaseAnalysis, inp, extra, llm, template_name="analyze_case",
            )
            return _strip_diagnoses_from_result(result)  # type: ignore[return-value]
        except Exception as exc:
            _log.warning("cbt.analyze_case_failed", error=str(exc))
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
            "As a CBT practitioner, synthesise evidence relevant to this case.\n"
            "Consider:\n"
            "- Meta-analyses and RCTs of CBT for the identified problem areas\n"
            "- Cognitive models (Beck, Clark, Salkovskis, Wells)\n"
            "- Behavioural mechanisms (behavioural activation, exposure, habituation, inhibitory learning)\n"
            "- Specific protocols (cognitive restructuring, thought records, behavioural experiments)\n"
            "- Relapse prevention and maintenance strategies\n\n"
            "IMPORTANT: Never assign diagnostic labels. "
            "Describe patterns, behaviours, and cognitive processes using descriptive clinical language only."
        )
        try:
            result = await self._llm_generate_typed(
                EvidenceResult, inp, extra, llm, template_name="retrieve_evidence",
            )
            return _strip_diagnoses_from_result(result)  # type: ignore[return-value]
        except Exception as exc:
            _log.warning("cbt.retrieve_evidence_failed", error=str(exc))
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
            "Generate a comprehensive CBT case conceptualisation.\n"
            "Structure around the cognitive-behavioural model:\n"
            "1. Core beliefs / schemas (self, others, world)\n"
            "2. Intermediate beliefs (rules, attitudes, conditional assumptions)\n"
            "3. Triggering situations / activating events\n"
            "4. Negative automatic thoughts (NATs)\n"
            "5. Cognitive distortions\n"
            "6. Emotional consequences\n"
            "7. Behavioural consequences (avoidance, safety behaviours, withdrawal)\n"
            "8. Maintaining cycles (how components interact to perpetuate difficulties)\n"
            "9. Developmental origins (early experiences forming core beliefs)\n\n"
            "IMPORTANT: Never assign diagnostic labels. "
            "Describe patterns, behaviours, and cognitive processes using descriptive clinical language only."
        )
        try:
            result = await self._llm_generate_typed(
                CaseConceptualization, inp, extra, llm,
                template_name="case_conceptualization",
            )
            return _strip_diagnoses_from_result(result)  # type: ignore[return-value]
        except Exception as exc:
            _log.warning("cbt.conceptualization_failed", error=str(exc))
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
            "Identify primary therapeutic focus areas from a CBT perspective.\n"
            "Consider:\n"
            "- Cognitive interventions: restructuring NATs, modifying intermediate beliefs, "
            "schema change, Socratic questioning, thought records\n"
            "- Behavioural interventions: behavioural activation, exposure, response prevention, "
            "behavioural experiments, activity scheduling\n"
            "- Emotional focus: emotional regulation, identifying/labelling emotions\n"
            "- Relapse prevention: high-risk situations, coping strategies, maintenance planning\n"
            "- Staging: what to address first, what requires stabilisation, what builds foundation\n\n"
            "IMPORTANT: Never assign diagnostic labels. "
            "Describe patterns, behaviours, and cognitive processes using descriptive clinical language only."
        )
        try:
            result = await self._llm_generate_typed(
                TherapeuticFocusResult, inp, extra, llm,
                template_name="therapeutic_focus",
            )
            return _strip_diagnoses_from_result(result)  # type: ignore[return-value]
        except Exception as exc:
            _log.warning("cbt.therapeutic_focus_failed", error=str(exc))
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
            "Generate a detailed CBT intervention plan.\n"
            "Cognitive techniques:\n"
            "- Daily thought record (situation -> mood -> automatic thought -> evidence for/against -> alternative thought -> re-rate mood)\n"
            "- Cognitive restructuring via Socratic questioning\n"
            "- Pie-chart and continuum techniques for black-and-white thinking\n"
            "- Decatastrophising / 'what if' technique\n"
            "- Core belief worksheet (old -> evidence for/against -> new balanced belief)\n"
            "- Behavioural experiment log (prediction -> experiment -> outcome -> alternative)\n\n"
            "Behavioural techniques:\n"
            "- Activity scheduling with graded task assignment\n"
            "- Behavioural activation hierarchy\n"
            "- Exposure hierarchy with response prevention\n"
            "- Behavioural experiments for hypothesis testing\n"
            "- Relaxation strategies (diaphragmatic breathing, PMR)\n\n"
            "Relapse prevention:\n"
            "- Identifying early warning signs and high-risk situations\n"
            "- Coping strategy rehearsal\n"
            "- Wellness and maintenance planning\n\n"
            "For each intervention specify technique, rationale, implementation approach, "
            "homework assignment, and expected mechanism of change.\n\n"
            "IMPORTANT: Never assign diagnostic labels. "
            "Describe patterns, behaviours, and cognitive processes using descriptive clinical language only."
        )
        try:
            result = await self._llm_generate_typed(
                InterventionPlanResult, inp, extra, llm,
                template_name="intervention_plan",
            )
            return _strip_diagnoses_from_result(result)  # type: ignore[return-value]
        except Exception as exc:
            _log.warning("cbt.intervention_plan_failed", error=str(exc))
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
            "Define CBT session-level objectives and goals.\n"
            "Standard session structure: mood check -> bridge -> agenda setting -> working part -> homework -> summary/feedback\n\n"
            "Immediate objectives:\n"
            "- Psychoeducation on the cognitive model (thoughts-feelings-behaviours link)\n"
            "- Identifying specific automatic thoughts for in-session work\n"
            "- Specific behavioural homework assignment\n"
            "- Skill acquisition (thought record, activity scheduling, exposure)\n\n"
            "Short-term goals (3-6 sessions):\n"
            "- Ability to identify, challenge, and modify NATs\n"
            "- Increased behavioural activation, reduced avoidance\n"
            "- Progress on exposure hierarchy\n"
            "- Improved emotional regulation, awareness of cognitive distortions\n\n"
            "Long-term goals:\n"
            "- Schema / core belief modification\n"
            "- Relapse prevention capabilities\n"
            "- Generalisation of skills across life domains\n"
            "- Client becomes their own therapist\n\n"
            "IMPORTANT: Never assign diagnostic labels. "
            "Describe patterns, behaviours, and cognitive processes using descriptive clinical language only."
        )
        try:
            result = await self._llm_generate_typed(
                SessionObjectivesResult, inp, extra, llm,
                template_name="session_objectives",
            )
            return _strip_diagnoses_from_result(result)  # type: ignore[return-value]
        except Exception as exc:
            _log.warning("cbt.session_objectives_failed", error=str(exc))
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
            "Suggest CBT follow-up questions for assessment and therapy.\n\n"
            "Assessment / cognitive exploration:\n"
            "- 'When you feel [emotion], what thoughts run through your mind?'\n"
            "- 'What does that say about you? About others? About your future?'\n"
            "- 'On a scale of 0-100, how much do you believe that thought right now?'\n\n"
            "Socratic questioning probes:\n"
            "- 'What is the evidence for and against this thought?'\n"
            "- 'Is there an alternative way of looking at this?'\n"
            "- 'What would you tell a close friend who had this thought?'\n"
            "- 'What are the advantages and disadvantages of thinking this way?'\n\n"
            "Behavioural activation / exposure probes:\n"
            "- 'What activities have you stopped doing that you used to enjoy?'\n"
            "- 'What have you been avoiding that you would like to face?'\n"
            "- 'How does avoidance affect your mood in the long run?'\n\n"
            "Homework / skill generalisation:\n"
            "- 'What would be helpful to practise between sessions?'\n"
            "- 'What might get in the way? How could we plan for that?'\n\n"
            "Engagement:\n"
            "- 'Does the way we are working together make sense to you?'\n"
            "- 'What has been the most useful part of our work together?'\n\n"
            "Frame questions in a collaborative, curious, non-judgemental manner.\n\n"
            "IMPORTANT: Never assign diagnostic labels. "
            "Describe patterns, behaviours, and cognitive processes using descriptive clinical language only."
        )
        try:
            result = await self._llm_generate_typed(
                FollowupQuestionsResult, inp, extra, llm,
                template_name="followup_questions",
            )
            return _strip_diagnoses_from_result(result)  # type: ignore[return-value]
        except Exception as exc:
            _log.warning("cbt.followup_questions_failed", error=str(exc))
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
            "Describe limitations and contraindications of applying CBT to this case.\n"
            "Contraindications / cautionary indicators:\n"
            "- Active psychosis, severe dissociation, or cognitive impairment limiting structured work\n"
            "- High risk requiring crisis intervention\n"
            "- Severe relational difficulties requiring extended alliance-building\n"
            "- Primary systemic/contextual drivers\n"
            "- Severe trauma requiring stabilisation before CBT techniques\n\n"
            "Precautions:\n"
            "- Risk of intellectualisation (client learns concepts but avoids emotional processing)\n"
            "- Pace cognitive restructuring to avoid overwhelming the client\n"
            "- Do not invalidate emotions while challenging thoughts\n"
            "- Need for cultural adaptation\n"
            "- Monitor homework non-compliance as engagement signal\n\n"
            "Alternative frameworks:\n"
            "- DBT: significant emotional dysregulation or self-harm\n"
            "- ACT: high experiential avoidance and cognitive fusion\n"
            "- CFT: high self-criticism or shame\n"
            "- Schema therapy: pervasive treatment-resistant core beliefs\n"
            "- Psychodynamic: insight-oriented work\n"
            "- CPT/PE: trauma-focused work\n"
            "- Systemic: family/relational dynamics\n\n"
            "IMPORTANT: Never assign diagnostic labels. "
            "Describe patterns, behaviours, and cognitive processes using descriptive clinical language only."
        )
        try:
            result = await self._llm_generate_typed(
                LimitationsResult, inp, extra, llm, template_name="limitations",
            )
            return _strip_diagnoses_from_result(result)  # type: ignore[return-value]
        except Exception as exc:
            _log.warning("cbt.limitations_failed", error=str(exc))
            return LimitationsResult(
                contraindications=[],
                precautions=[],
                alternative_frameworks=[],
                caveats=[f"Limitations unavailable: {exc}"],
            )
