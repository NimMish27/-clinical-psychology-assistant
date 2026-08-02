from __future__ import annotations

import json
import time
from typing import Any

from app_logging.logger import get_logger
from clinical.llm import LLMService
from clinical.session_planning.models import SessionObjective, SessionPlanResult

_log = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a session planning advisor supporting a clinical psychologist. \
You will receive a clinical formulation, a therapeutic plan, and the \
therapeutic framework the psychologist has selected.

Your task is to propose an ordered sequence of therapy sessions \
that operationalises the plan within the selected framework. You must \
NOT prescribe treatment. Present everything as **evidence-informed \
suggestions for clinician consideration**. The clinician retains \
full clinical responsibility.

Use careful, collaborative language. Phrases such as "the clinician \
may wish to structure", "one option is", and "this session could" \
are appropriate.

Produce the following sections:

1. DISCLAIMER — A brief statement that the session proposals are \
   suggestions and must be adapted to the client and clinician judgement.

2. SESSION COUNT — Total number of proposed sessions.

3. CADENCE — Suggested session frequency (e.g. "weekly", "twice weekly").

4. OBJECTIVES — An ordered list of session objectives. Each must \
   include:
   - title: short descriptive title
   - focus: what the session aims to explore or accomplish
   - suggested_activities: specific in-session activities/techniques
   - homework: suggested between-session task (optional)
   - expected_duration_minutes: indicative length (optional)

5. REVIEW POINTS — Where progress should be reviewed and the plan \
   adapted (e.g. after session 4, when specific goals are met).

Rules:
- Anchor each session to the selected framework's concepts and the \
  therapeutic plan's intervention directions.
- Never use imperative language. Use tentative suggestions.
- If information is limited, acknowledge this and keep sessions flexible.
- Empty sections (empty arrays) are acceptable if nothing applies.

Respond EXACTLY in the JSON format below. No markdown fences. \
No text outside the JSON. Use empty lists for sections with no items.

{
  "disclaimer": "The following session proposals are evidence-informed suggestions for clinician consideration. They should be adapted to the client's presentation and preferences, and the clinician retains full clinical responsibility.",
  "session_count": 8,
  "cadence": "weekly",
  "objectives": [
    {
      "title": "Engagement and shared formulation",
      "focus": "Establish a collaborative therapeutic relationship and present the formulation within the selected framework.",
      "suggested_activities": [
        "Review the case formulation together",
        "Discuss expectations and goals"
      ],
      "homework": "Reflect on which formulation element resonates most",
      "expected_duration_minutes": 50
    }
  ],
  "review_points": [
    "After session 4, review progress against treatment goals",
    "If the client reports deterioration, pause and reassess"
  ]
}
"""


class SessionPlanner:
    """Generate a structured session plan from formulation and treatment plan.

    Takes the output of the Clinical Formulation and Therapeutic Planning
    modules (plus the psychologist-selected framework) and proposes an
    ordered sequence of sessions. All output is presented as suggestions
    for clinician consideration.

    Designed to be reusable by LangGraph agents — no pipeline dependency.
    """

    def __init__(self, llm: LLMService):
        self._llm = llm

    async def plan(
        self,
        formulation: str | None = None,
        *,
        plan_text: str | None = None,
        case_summary: str | None = None,
        selected_framework: str | None = None,
        framework_lens: str | None = None,
        therapeutic_focus: list[str] | None = None,
        treatment_goals: list[str] | None = None,
        caution: str | None = None,
    ) -> SessionPlanResult:
        """Generate a session plan from formulation and treatment plan.

        Args:
            formulation:       Full text of the ClinicalFormulationResult.
            plan_text:         Full text of the TherapeuticPlanResult.
            case_summary:      Case summary from the formulation.
            selected_framework: Framework chosen by the psychologist.
            framework_lens:    Formatted description of the selected
                               framework's analysis of this case.
            therapeutic_focus: Suggested focus areas from the plan.
            treatment_goals:   Suggested treatment goals from the plan.
            caution:           Caution note from the formulation.

        Returns:
            SessionPlanResult with structured session objectives.
        """
        t_start = time.perf_counter()

        prompt = self._build_prompt(
            formulation=formulation,
            plan_text=plan_text,
            case_summary=case_summary,
            selected_framework=selected_framework,
            framework_lens=framework_lens,
            therapeutic_focus=therapeutic_focus,
            treatment_goals=treatment_goals,
            caution=caution,
        )

        if not self._has_content(prompt):
            _log.warning("session_planning.empty_input")
            return SessionPlanResult(
                disclaimer="No formulation or treatment plan was provided. A session plan cannot be generated without clinical information.",
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
                "session_planning.planning_failed",
                error=str(exc),
            )
            result = SessionPlanResult(
                disclaimer="Session planning could not be completed automatically. The formulation and treatment plan are available for manual review.",
            )

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        object.__setattr__(result, "planning_ms", round(elapsed_ms, 2))
        return result

    # ── Content check ─────────────────────────────────────

    @staticmethod
    def _has_content(prompt: str) -> bool:
        stripped = prompt.strip()
        if not stripped:
            return False
        lines = [l for l in stripped.split("\n") if l.strip() and not l.strip().startswith("##")]
        return len(lines) > 0

    # ── Prompt building ────────────────────────────────────

    def _build_prompt(
        self,
        formulation: str | None,
        plan_text: str | None,
        case_summary: str | None,
        selected_framework: str | None,
        framework_lens: str | None,
        therapeutic_focus: list[str] | None,
        treatment_goals: list[str] | None,
        caution: str | None,
    ) -> str:
        parts: list[str] = []

        if selected_framework:
            parts.append(f"## Selected Framework\n{selected_framework}\n")

        if framework_lens:
            parts.append(f"## Selected Framework Analysis\n{framework_lens}\n")

        if case_summary:
            parts.append(f"## Case Summary\n{case_summary}\n")

        if formulation:
            parts.append(f"## Clinical Formulation\n{formulation}\n")

        if plan_text:
            parts.append(f"## Therapeutic Plan\n{plan_text}\n")

        if therapeutic_focus:
            parts.append("## Therapeutic Focus\n" + "\n".join(f"- {f}" for f in therapeutic_focus) + "\n")

        if treatment_goals:
            parts.append("## Treatment Goals\n" + "\n".join(f"- {g}" for g in treatment_goals) + "\n")

        if caution:
            parts.append(f"## Caution\n{caution}\n")

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

    def _build_result(self, data: dict[str, Any]) -> SessionPlanResult:
        objectives = self._build_objectives(data.get("objectives", []))
        return SessionPlanResult(
            disclaimer=str(data.get("disclaimer", "")),
            session_count=self._clamp_count(data.get("session_count", len(objectives))),
            cadence=str(data.get("cadence", "")) or None,
            objectives=objectives,
            review_points=[str(s) for s in data.get("review_points", []) if s],
        )

    def _build_objectives(self, raw: list[Any]) -> list[SessionObjective]:
        result: list[SessionObjective] = []
        for item in raw:
            if not isinstance(item, dict) or not item.get("title"):
                continue
            try:
                result.append(SessionObjective(
                    title=str(item["title"]).strip(),
                    focus=str(item.get("focus", "")).strip(),
                    suggested_activities=[str(s) for s in item.get("suggested_activities", []) if s],
                    homework=str(item["homework"]).strip() if item.get("homework") else None,
                    expected_duration_minutes=self._clamp_duration(item.get("expected_duration_minutes")),
                ))
            except (ValueError, TypeError) as exc:
                _log.warning("session_planning.invalid_objective", error=str(exc))
        return result

    @staticmethod
    def _clamp_count(v: Any) -> int:
        try:
            return max(0, int(v))
        except (ValueError, TypeError):
            return 0

    @staticmethod
    def _clamp_duration(v: Any) -> int | None:
        try:
            return max(0, int(v))
        except (ValueError, TypeError):
            return None
