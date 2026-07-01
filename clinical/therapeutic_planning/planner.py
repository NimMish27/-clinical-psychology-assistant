from __future__ import annotations

import json
import time
from typing import Any

from clinical.therapeutic_planning.models import (
    ACTStrategy,
    BehaviouralActivationSuggestion,
    CBTStrategy,
    DBTStrategy,
    HomeworkIdea,
    InterventionDirection,
    PsychEducationSuggestion,
    SelfCompassionStrategy,
    TherapeuticFocus,
    TherapeuticPlanResult,
    TreatmentGoal,
)
from clinical.llm import LLMService
from app_logging.logger import get_logger

_log = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a senior therapeutic planning advisor supporting a clinical \
psychologist. You will receive a clinical formulation and an evidence \
summary for a client.

Your task is to produce a structured therapeutic plan. You must NOT \
prescribe treatment. Present everything as **evidence-informed \
suggestions for clinician consideration**. The clinician retains \
full clinical responsibility for all decisions.

Use careful, collaborative language. Phrases such as "the clinician \
may wish to consider", "one option is to explore", and "this might \
be particularly relevant if" are appropriate.

Produce the following sections:

1. DISCLAIMER — A brief statement that all suggestions require \
   clinical judgement and are not treatment prescriptions. (1-2 sentences)

2. THERAPEUTIC FOCUS — 2-4 broad areas of focus that follow from \
   the formulation. Each must include:
   - area: domain or theme (e.g. cognitive patterns, emotional regulation)
   - rationale: why this area is relevant

3. TREATMENT GOALS — 2-4 example goals for collaborative discussion. \
   Each must include:
   - goal: specific, measurable, realistic statement
   - suggested_measurement: how progress might be tracked (optional)
   - indicative_timeframe: e.g. "4 weeks", "12 sessions" (optional)

4. INTERVENTION DIRECTIONS — 2-4 broad therapeutic directions. Each \
   must include:
   - area: intervention domain
   - suggested_approaches: specific approaches to consider
   - rationale: how this follows from the formulation

5. CBT STRATEGIES — 1-3 CBT techniques for consideration. Each must \
   include:
   - technique: the technique name
   - rationale: how it maps to the formulation
   - application: how it might be introduced (optional)

6. ACT STRATEGIES — 1-3 ACT processes for consideration. Each must \
   include:
   - process: the ACT process (e.g. acceptance, defusion, values)
   - rationale: why this process is relevant
   - application: how to explore it (optional)

7. DBT STRATEGIES — 1-3 DBT skills for consideration. Each must \
   include:
   - skill: the skill or module
   - rationale: why relevant given the presentation
   - application: how to introduce it (optional)

8. PSYCHOEDUCATION SUGGESTIONS — 1-3 topics. Each must include:
   - topic: the topic
   - key_points: list of key messages to convey

9. BEHAVIOURAL ACTIVATION SUGGESTIONS — 1-3 activity domains. \
   Each must include:
   - activity_domain: area (e.g. social, physical, occupational)
   - suggested_activities: specific activities to explore
   - rationale: how these relate to the formulation

10. SELF-COMPASSION STRATEGIES — 1-3 practices. Each must include:
    - practice: the practice name
    - rationale: why it may be helpful

11. HOMEWORK IDEAS — 1-3 between-session activities. Each must \
    include:
    - activity: description
    - purpose: therapeutic purpose
    - frequency: suggested frequency (optional)

Rules:
- Never use imperative language ("do X", "prescribe Y"). Use \
  tentative suggestions ("the clinician may consider", "one option is").
- Base all suggestions on the specific formulation and evidence provided.
- If information is limited, acknowledge this and suggest accordingly.
- Empty sections (empty arrays) are acceptable if no suggestions apply.

Respond EXACTLY in the JSON format below. No markdown fences. \
No text outside the JSON. Use empty lists for sections with no items.

{
  "disclaimer": "The following suggestions are evidence-informed and intended for clinician consideration only. They do not constitute a treatment prescription and should be adapted based on clinical judgement and client preferences.",
  "therapeutic_focus": [
    {
      "area": "Cognitive patterns and self-critical thinking",
      "rationale": "The formulation identifies perfectionist beliefs as a key maintaining factor. Addressing these beliefs may reduce the cycle of overwork and withdrawal."
    }
  ],
  "treatment_goals": [
    {
      "goal": "Reduce frequency of social avoidance from daily to twice per week over 8 sessions",
      "suggested_measurement": "Behavioural record",
      "indicative_timeframe": "8 sessions"
    }
  ],
  "intervention_directions": [
    {
      "area": "Cognitive restructuring",
      "suggested_approaches": ["Socratic questioning", "behavioural experiments", "thought records"],
      "rationale": "Challenging perfectionist and fear-of-judgement beliefs may reduce avoidance and improve mood."
    }
  ],
  "cbt_strategies": [
    {
      "technique": "Thought record",
      "rationale": "Helps the client identify and evaluate automatic thoughts about social situations",
      "application": "Introduce using a recent social situation, focus on the thought-feeling-behaviour link"
    }
  ],
  "act_strategies": [
    {
      "process": "Cognitive defusion",
      "rationale": "The client appears fused with self-critical thoughts. Defusion may reduce their behavioural impact.",
      "application": "Begin with 'I notice I am having the thought that...' phrasing"
    }
  ],
  "dbt_strategies": [
    {
      "skill": "Distress tolerance – TIPP skills",
      "rationale": "Client reports intense emotional reactions to perceived criticism",
      "application": "Practise TIPP in session before applying to real-world situations"
    }
  ],
  "psychoeducation_suggestions": [
    {
      "topic": "The anxiety cycle",
      "key_points": ["Avoidance maintains anxiety in the long term", "Short-term relief reinforces avoidance behaviour"]
    }
  ],
  "behavioural_activation_suggestions": [
    {
      "activity_domain": "Social connection",
      "suggested_activities": ["Coffee with a friend", "Joining a low-pressure group activity"],
      "rationale": "Graded social exposure may reduce avoidance and increase positive reinforcement"
    }
  ],
  "self_compassion_strategies": [
    {
      "practice": "Compassionate letter writing",
      "rationale": "May help the client develop a kinder internal voice, counteracting self-critical patterns"
    }
  ],
  "homework_ideas": [
    {
      "activity": "Complete a thought record for one social situation between sessions",
      "purpose": "Build awareness of automatic thoughts and their emotional impact",
      "frequency": "Once before next session"
    }
  ]
}
"""


class TherapeuticPlanner:
    """Generate a structured therapeutic plan from formulation and evidence.

    Takes the output of the Clinical Formulation and Evidence Synthesis
    modules and produces a structured therapeutic plan with suggestions
    across 10 domains.  The planner NEVER prescribes treatment — all
    output is presented as evidence-informed suggestions for clinician
    consideration.

    Designed to be reusable by LangGraph agents — no pipeline dependency.

    Usage::

        planner = TherapeuticPlanner(llm_service)
        plan = await planner.plan(formulation_result, evidence_result)
        for goal in plan.treatment_goals:
            print(goal.goal)
    """

    def __init__(self, llm: LLMService):
        self._llm = llm

    async def plan(
        self,
        formulation: str | None = None,
        *,
        evidence_summary: str | None = None,
        case_summary: str | None = None,
        formulations_text: list[str] | None = None,
        supporting_evidence: list[str] | None = None,
        alternative_explanations: list[str] | None = None,
        missing_information: list[str] | None = None,
        caution: str | None = None,
        key_findings: list[str] | None = None,
        evidence_themes: list[str] | None = None,
    ) -> TherapeuticPlanResult:
        """Generate a therapeutic plan from formulation and evidence.

        Args:
            formulation:       Full text of the ClinicalFormulationResult
                               (model_dump/json) or any string combining
                               the formulation information.
            evidence_summary:  Full text of the EvidenceSynthesisResult
                               or a text summary of the evidence.
            case_summary:      Case summary from the formulation.
            formulations_text: List of formulation label texts.
            supporting_evidence: Supporting evidence from formulation.
            alternative_explanations: Alternative explanations from formulation.
            missing_information: Missing assessment info from formulation.
            caution:           Caution note from formulation.
            key_findings:      Key findings from evidence synthesis.
            evidence_themes:   Common themes from evidence synthesis.

        Returns:
            TherapeuticPlanResult with structured plan sections.
        """
        t_start = time.perf_counter()

        prompt = self._build_prompt(
            formulation=formulation,
            evidence_summary=evidence_summary,
            case_summary=case_summary,
            formulations_text=formulations_text,
            supporting_evidence=supporting_evidence,
            alternative_explanations=alternative_explanations,
            missing_information=missing_information,
            caution=caution,
            key_findings=key_findings,
            evidence_themes=evidence_themes,
        )

        if not self._has_content(prompt):
            _log.warning("therapeutic_planning.empty_input")
            return TherapeuticPlanResult(
                disclaimer="No formulation or evidence was provided. A therapeutic plan cannot be generated without clinical information.",
            )

        try:
            raw = await self._llm.generate(
                prompt=prompt,
                system_prompt=_SYSTEM_PROMPT,
            )
            data = self._parse_response(raw)
            result = self._build_result(data)
        except Exception as exc:
            _log.error(
                "therapeutic_planning.planning_failed",
                error=str(exc),
            )
            result = TherapeuticPlanResult(
                disclaimer="Therapeutic planning could not be completed automatically. The formulation and evidence are available for manual review.",
            )

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        object.__setattr__(result, "planning_ms", round(elapsed_ms, 2))
        return result

    # ── Content check ─────────────────────────────────────

    @staticmethod
    def _has_content(prompt: str) -> bool:
        """Check whether the prompt has substantive content beyond section headers."""
        stripped = prompt.strip()
        if not stripped:
            return False
        # If only the header and no data fields were populated, it's empty
        lines = [l for l in stripped.split("\n") if l.strip() and not l.strip().startswith("##")]
        return len(lines) > 0

    # ── Prompt building ────────────────────────────────────

    def _build_prompt(
        self,
        formulation: str | None,
        evidence_summary: str | None,
        case_summary: str | None,
        formulations_text: list[str] | None,
        supporting_evidence: list[str] | None,
        alternative_explanations: list[str] | None,
        missing_information: list[str] | None,
        caution: str | None,
        key_findings: list[str] | None,
        evidence_themes: list[str] | None,
    ) -> str:
        parts: list[str] = []

        if formulation:
            parts.append(f"## Clinical Formulation\n{formulation}\n")

        if case_summary:
            parts.append(f"## Case Summary\n{case_summary}\n")

        if formulations_text:
            parts.append("## Possible Formulations\n" + "\n".join(f"- {f}" for f in formulations_text) + "\n")

        if supporting_evidence:
            parts.append("## Supporting Evidence\n" + "\n".join(f"- {e}" for e in supporting_evidence) + "\n")

        if alternative_explanations:
            parts.append("## Alternative Explanations\n" + "\n".join(f"- {a}" for a in alternative_explanations) + "\n")

        if missing_information:
            parts.append("## Missing Assessment Information\n" + "\n".join(f"- {m}" for m in missing_information) + "\n")

        if caution:
            parts.append(f"## Caution\n{caution}\n")

        if evidence_summary:
            parts.append(f"## Evidence Summary\n{evidence_summary}\n")

        if key_findings:
            parts.append("## Key Findings\n" + "\n".join(f"- {k}" for k in key_findings) + "\n")

        if evidence_themes:
            parts.append("## Evidence Themes\n" + "\n".join(f"- {t}" for t in evidence_themes) + "\n")

        return "\n".join(parts)

    # ── Response parsing ───────────────────────────────────

    def _parse_response(self, raw: str) -> dict[str, Any]:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()

        start = cleaned.index("{")
        end = cleaned.rindex("}")
        return json.loads(cleaned[start : end + 1])

    # ── Result building ─────────────────────────────────────

    def _build_result(self, data: dict[str, Any]) -> TherapeuticPlanResult:
        return TherapeuticPlanResult(
            disclaimer=str(data.get("disclaimer", "")),
            therapeutic_focus=self._build_focus(data.get("therapeutic_focus", [])),
            treatment_goals=self._build_goals(data.get("treatment_goals", [])),
            intervention_directions=self._build_directions(data.get("intervention_directions", [])),
            cbt_strategies=self._build_cbt(data.get("cbt_strategies", [])),
            act_strategies=self._build_act(data.get("act_strategies", [])),
            dbt_strategies=self._build_dbt(data.get("dbt_strategies", [])),
            psychoeducation_suggestions=self._build_psychoeducation(data.get("psychoeducation_suggestions", [])),
            behavioural_activation_suggestions=self._build_ba(data.get("behavioural_activation_suggestions", [])),
            self_compassion_strategies=self._build_self_compassion(data.get("self_compassion_strategies", [])),
            homework_ideas=self._build_homework(data.get("homework_ideas", [])),
        )

    def _build_focus(self, raw: list[Any]) -> list[TherapeuticFocus]:
        result: list[TherapeuticFocus] = []
        for item in raw:
            if not isinstance(item, dict) or not item.get("area"):
                continue
            try:
                result.append(TherapeuticFocus(
                    area=str(item["area"]).strip(),
                    rationale=str(item.get("rationale", "")).strip(),
                ))
            except (ValueError, TypeError) as exc:
                _log.warning("therapeutic_planning.invalid_focus", error=str(exc))
        return result

    def _build_goals(self, raw: list[Any]) -> list[TreatmentGoal]:
        result: list[TreatmentGoal] = []
        for item in raw:
            if not isinstance(item, dict) or not item.get("goal"):
                continue
            try:
                result.append(TreatmentGoal(
                    goal=str(item["goal"]).strip(),
                    suggested_measurement=str(item["suggested_measurement"]).strip() if item.get("suggested_measurement") else None,
                    indicative_timeframe=str(item["indicative_timeframe"]).strip() if item.get("indicative_timeframe") else None,
                ))
            except (ValueError, TypeError) as exc:
                _log.warning("therapeutic_planning.invalid_goal", error=str(exc))
        return result

    def _build_directions(self, raw: list[Any]) -> list[InterventionDirection]:
        result: list[InterventionDirection] = []
        for item in raw:
            if not isinstance(item, dict) or not item.get("area"):
                continue
            try:
                result.append(InterventionDirection(
                    area=str(item["area"]).strip(),
                    suggested_approaches=[str(s) for s in item.get("suggested_approaches", [])],
                    rationale=str(item.get("rationale", "")).strip(),
                ))
            except (ValueError, TypeError) as exc:
                _log.warning("therapeutic_planning.invalid_direction", error=str(exc))
        return result

    def _build_cbt(self, raw: list[Any]) -> list[CBTStrategy]:
        result: list[CBTStrategy] = []
        for item in raw:
            if not isinstance(item, dict) or not item.get("technique"):
                continue
            try:
                result.append(CBTStrategy(
                    technique=str(item["technique"]).strip(),
                    rationale=str(item.get("rationale", "")).strip(),
                    application=str(item["application"]).strip() if item.get("application") else None,
                ))
            except (ValueError, TypeError) as exc:
                _log.warning("therapeutic_planning.invalid_cbt", error=str(exc))
        return result

    def _build_act(self, raw: list[Any]) -> list[ACTStrategy]:
        result: list[ACTStrategy] = []
        for item in raw:
            if not isinstance(item, dict) or not item.get("process"):
                continue
            try:
                result.append(ACTStrategy(
                    process=str(item["process"]).strip(),
                    rationale=str(item.get("rationale", "")).strip(),
                    application=str(item["application"]).strip() if item.get("application") else None,
                ))
            except (ValueError, TypeError) as exc:
                _log.warning("therapeutic_planning.invalid_act", error=str(exc))
        return result

    def _build_dbt(self, raw: list[Any]) -> list[DBTStrategy]:
        result: list[DBTStrategy] = []
        for item in raw:
            if not isinstance(item, dict) or not item.get("skill"):
                continue
            try:
                result.append(DBTStrategy(
                    skill=str(item["skill"]).strip(),
                    rationale=str(item.get("rationale", "")).strip(),
                    application=str(item["application"]).strip() if item.get("application") else None,
                ))
            except (ValueError, TypeError) as exc:
                _log.warning("therapeutic_planning.invalid_dbt", error=str(exc))
        return result

    def _build_psychoeducation(self, raw: list[Any]) -> list[PsychEducationSuggestion]:
        result: list[PsychEducationSuggestion] = []
        for item in raw:
            if not isinstance(item, dict) or not item.get("topic"):
                continue
            try:
                result.append(PsychEducationSuggestion(
                    topic=str(item["topic"]).strip(),
                    key_points=[str(s) for s in item.get("key_points", [])],
                ))
            except (ValueError, TypeError) as exc:
                _log.warning("therapeutic_planning.invalid_psychoeducation", error=str(exc))
        return result

    def _build_ba(self, raw: list[Any]) -> list[BehaviouralActivationSuggestion]:
        result: list[BehaviouralActivationSuggestion] = []
        for item in raw:
            if not isinstance(item, dict) or not item.get("activity_domain"):
                continue
            try:
                result.append(BehaviouralActivationSuggestion(
                    activity_domain=str(item["activity_domain"]).strip(),
                    suggested_activities=[str(s) for s in item.get("suggested_activities", [])],
                    rationale=str(item.get("rationale", "")).strip(),
                ))
            except (ValueError, TypeError) as exc:
                _log.warning("therapeutic_planning.invalid_ba", error=str(exc))
        return result

    def _build_self_compassion(self, raw: list[Any]) -> list[SelfCompassionStrategy]:
        result: list[SelfCompassionStrategy] = []
        for item in raw:
            if not isinstance(item, dict) or not item.get("practice"):
                continue
            try:
                result.append(SelfCompassionStrategy(
                    practice=str(item["practice"]).strip(),
                    rationale=str(item.get("rationale", "")).strip(),
                ))
            except (ValueError, TypeError) as exc:
                _log.warning("therapeutic_planning.invalid_self_compassion", error=str(exc))
        return result

    def _build_homework(self, raw: list[Any]) -> list[HomeworkIdea]:
        result: list[HomeworkIdea] = []
        for item in raw:
            if not isinstance(item, dict) or not item.get("activity"):
                continue
            try:
                result.append(HomeworkIdea(
                    activity=str(item["activity"]).strip(),
                    purpose=str(item.get("purpose", "")).strip(),
                    frequency=str(item["frequency"]).strip() if item.get("frequency") else None,
                ))
            except (ValueError, TypeError) as exc:
                _log.warning("therapeutic_planning.invalid_homework", error=str(exc))
        return result
