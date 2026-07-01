from __future__ import annotations

import json
import time
from typing import Any

from clinical.response_generation.models import ClinicalResponseResult
from clinical.llm import LLMService
from app_logging.logger import get_logger

_log = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a clinical report writer in a psychological therapy service. You \
will receive structured data from several clinical analysis modules. Your \
task is to compose a complete, ready-to-present clinical report in \
markdown format.

The report must contain ALL of the following sections in order. If a section \
has no data, state that clearly rather than omitting it.

---

1. CASE SUMMARY
A concise paragraph summarising the client's presentation, context, and \
reason for seeking help. Write in plain, respectful language.

2. PRESENTING CONCERNS
Bullet-point list of the specific concerns the client reported. Include \
duration and context where available.

3. OBSERVED SYMPTOMS
Bullet-point list of symptoms that were identified from the case \
information. Organise by domain (e.g. mood, anxiety, behavioural, \
cognitive, physical) if multiple symptoms are present.

4. CLINICAL FORMULATION
A narrative formulation — NOT a diagnosis. Describe how the client's \
difficulties may have developed and are maintained. Include the confidence \
level of the formulation. Use tentative language throughout.

5. POSSIBLE DIFFERENTIAL CONSIDERATIONS
Alternative ways of understanding the presentation that cannot be ruled \
out with the current information. List each as a brief statement. \
Never present these as alternative diagnoses.

6. MISSING INFORMATION
Key clinical information gaps that would strengthen or clarify the \
formulation. For each gap, briefly note why it matters.

7. EVIDENCE SUMMARY
A synthesis of relevant evidence from the literature that supports or \
contextualises the formulation. Include key findings and practical \
implications. Cite sources where available.

8. THERAPEUTIC FOCUS
Suggested broad areas for therapeutic work, based on the formulation \
and evidence. Present as suggestions for clinician consideration.

9. SUGGESTED INTERVENTION DIRECTIONS
Specific therapeutic approaches and strategies the clinician may \
consider. Organise by modality if relevant (CBT, ACT, DBT, BA, etc.). \
Present as evidence-informed suggestions — never as prescriptions.

10. REFERENCES
Any sources, guidelines, or evidence cited in the report. List in a \
simple format. If none were provided, state "No specific references \
were provided with the source data."

11. CONFIDENCE LEVEL
A brief statement about the overall confidence in this report given \
the completeness and quality of the available information. Include \
a numerical confidence rating (0.0-1.0).

---

Rules:
- Use professional, respectful, and tentative clinical language.
- NEVER state a diagnosis.
- Use markdown formatting: ## for section headers, - for bullet lists, \
  **bold** for emphasis where appropriate.
- Write in full clinical sentences. Avoid bullet-spam — use paragraphs \
  where narrative is appropriate.
- If data for a section is empty or absent, write a brief note such as \
  "No differential considerations were identified from the available \
  information."
- The final document should be 2-4 pages when printed.

Respond EXACTLY in the JSON format below. No markdown fences. \
No text outside the JSON.

{
  "markdown": "## 1. CASE SUMMARY\\n\\n...",
  "sections_generated": 11,
  "confidence": 0.7
}
"""


class ResponseGenerator:
    """Generate a complete markdown clinical report from all module outputs.

    Takes the outputs of the Case Understanding, Clinical Formulation,
    Evidence Synthesis, Therapeutic Planning, and Missing Information
    modules and composes a structured markdown report with 11 sections.

    Designed to be reusable by LangGraph agents — no pipeline dependency.

    Usage::

        generator = ResponseGenerator(llm_service)
        response = await generator.generate(
            case_summary=formulation_result.case_summary,
            presenting_concerns=...,
            ...
        )
        print(response.markdown)
    """

    def __init__(self, llm: LLMService):
        self._llm = llm

    async def generate(
        self,
        case_summary: str | None = None,
        presenting_concerns: list[str] | None = None,
        observed_symptoms: list[str] | None = None,
        formulation_text: str | None = None,
        formulation_confidence: float | None = None,
        differential_considerations: list[str] | None = None,
        missing_information: str | None = None,
        evidence_summary: str | None = None,
        evidence_findings: list[str] | None = None,
        therapeutic_focus: list[str] | None = None,
        intervention_directions: str | None = None,
        cbt_strategies: list[str] | None = None,
        act_strategies: list[str] | None = None,
        dbt_strategies: list[str] | None = None,
        references: list[str] | None = None,
        caution: str | None = None,
    ) -> ClinicalResponseResult:
        """Generate a full markdown clinical report.

        Args:
            case_summary:             Case summary text.
            presenting_concerns:      List of presenting concerns.
            observed_symptoms:        List of observed symptoms.
            formulation_text:         Full formulation narrative.
            formulation_confidence:   Formulation confidence (0-1).
            differential_considerations: List of differential considerations.
            missing_information:      Missing info text or summary.
            evidence_summary:         Evidence synthesis summary.
            evidence_findings:        Key findings from evidence.
            therapeutic_focus:        List of therapeutic focus areas.
            intervention_directions:  Intervention directions text.
            cbt_strategies:           CBT strategies list.
            act_strategies:           ACT strategies list.
            dbt_strategies:           DBT strategies list.
            references:               Reference list.
            caution:                  Caution note from formulation.

        Returns:
            ClinicalResponseResult with the markdown report.
        """
        t_start = time.perf_counter()

        prompt = self._build_prompt(
            case_summary=case_summary,
            presenting_concerns=presenting_concerns,
            observed_symptoms=observed_symptoms,
            formulation_text=formulation_text,
            formulation_confidence=formulation_confidence,
            differential_considerations=differential_considerations,
            missing_information=missing_information,
            evidence_summary=evidence_summary,
            evidence_findings=evidence_findings,
            therapeutic_focus=therapeutic_focus,
            intervention_directions=intervention_directions,
            cbt_strategies=cbt_strategies,
            act_strategies=act_strategies,
            dbt_strategies=dbt_strategies,
            references=references,
            caution=caution,
        )

        if not self._has_content(prompt):
            _log.warning("response_generation.empty_input")
            return ClinicalResponseResult(
                markdown="## Clinical Report\n\nNo clinical data was provided. A report cannot be generated without input from the analysis modules.",
                sections_generated=0,
                confidence=0.0,
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
                "response_generation.generation_failed",
                error=str(exc),
            )
            result = ClinicalResponseResult(
                markdown="## Clinical Report\n\nThe clinical report could not be generated automatically. Individual module outputs are available for manual review and composition.",
                sections_generated=0,
                confidence=0.0,
            )

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        object.__setattr__(result, "generation_ms", round(elapsed_ms, 2))
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
        case_summary: str | None,
        presenting_concerns: list[str] | None,
        observed_symptoms: list[str] | None,
        formulation_text: str | None,
        formulation_confidence: float | None,
        differential_considerations: list[str] | None,
        missing_information: str | None,
        evidence_summary: str | None,
        evidence_findings: list[str] | None,
        therapeutic_focus: list[str] | None,
        intervention_directions: str | None,
        cbt_strategies: list[str] | None,
        act_strategies: list[str] | None,
        dbt_strategies: list[str] | None,
        references: list[str] | None,
        caution: str | None,
    ) -> str:
        parts: list[str] = []

        if case_summary:
            parts.append(f"## Case Summary\n{case_summary}\n")
        if presenting_concerns:
            parts.append("## Presenting Concerns\n" + "\n".join(f"- {c}" for c in presenting_concerns) + "\n")
        if observed_symptoms:
            parts.append("## Observed Symptoms\n" + "\n".join(f"- {s}" for s in observed_symptoms) + "\n")
        if formulation_text:
            parts.append(f"## Clinical Formulation\n{formulation_text}\n")
        if formulation_confidence is not None:
            parts.append(f"## Formulation Confidence\n{formulation_confidence}\n")
        if differential_considerations:
            parts.append("## Differential Considerations\n" + "\n".join(f"- {d}" for d in differential_considerations) + "\n")
        if caution:
            parts.append(f"## Caution\n{caution}\n")
        if missing_information:
            parts.append(f"## Missing Information\n{missing_information}\n")
        if evidence_summary:
            parts.append(f"## Evidence Summary\n{evidence_summary}\n")
        if evidence_findings:
            parts.append("## Key Evidence Findings\n" + "\n".join(f"- {f}" for f in evidence_findings) + "\n")
        if therapeutic_focus:
            parts.append("## Therapeutic Focus\n" + "\n".join(f"- {f}" for f in therapeutic_focus) + "\n")
        if intervention_directions:
            parts.append(f"## Intervention Directions\n{intervention_directions}\n")
        if cbt_strategies:
            parts.append("## CBT Strategies\n" + "\n".join(f"- {c}" for c in cbt_strategies) + "\n")
        if act_strategies:
            parts.append("## ACT Strategies\n" + "\n".join(f"- {a}" for a in act_strategies) + "\n")
        if dbt_strategies:
            parts.append("## DBT Strategies\n" + "\n".join(f"- {d}" for d in dbt_strategies) + "\n")
        if references:
            parts.append("## References\n" + "\n".join(f"- {r}" for r in references) + "\n")

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

    def _build_result(self, data: dict[str, Any]) -> ClinicalResponseResult:
        return ClinicalResponseResult(
            markdown=str(data.get("markdown", "")),
            sections_generated=self._clamp_int(data.get("sections_generated", 0), 0, 11),
            confidence=self._clamp_float(data.get("confidence", 0.0)),
        )

    @staticmethod
    def _clamp_int(v: Any, lo: int, hi: int) -> int:
        try:
            return max(lo, min(hi, int(v)))
        except (ValueError, TypeError):
            return lo

    @staticmethod
    def _clamp_float(v: Any) -> float:
        try:
            return max(0.0, min(1.0, float(v)))
        except (ValueError, TypeError):
            return 0.0
