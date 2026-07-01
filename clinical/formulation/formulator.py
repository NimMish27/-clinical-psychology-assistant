from __future__ import annotations

import json
import time
from typing import Any

from clinical.formulation.models import ClinicalFormulationResult, Formulation
from clinical.llm import LLMService
from app_logging.logger import get_logger

_log = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a clinical formulation specialist in psychological therapy services. \
You will receive structured case information about a client.

Your task is to produce a clinical formulation — a hypothesis about how the \
client's difficulties may have developed and are being maintained. You must \
NEVER produce a diagnosis. Formulations are not diagnoses; they are \
individualised hypotheses that guide intervention.

Use careful, tentative clinical language throughout. Phrases such as \
"may reflect", "this might suggest", "it is possible that", and \
"one way of understanding this is" are appropriate.

Produce the following sections:

1. CASE SUMMARY — A concise paragraph (3-5 sentences) describing the \
   client's presentation, key concerns, and relevant contextual factors.

2. POSSIBLE FORMULATIONS — One or more ways of understanding the clinical \
   picture. Each formulation must include:
   - label: short descriptive label
   - explanation: detailed narrative of how difficulties developed and \
     are maintained, referencing specific case information
   - supporting_symptoms: list of specific symptoms/observations from the \
     case that fit this formulation
   - confidence: 0.0-1.0 reflecting how well the available information \
     supports this particular formulation

3. SUPPORTING EVIDENCE — Specific observations from the case and any \
   relevant clinical knowledge that strengthens these formulations. \
   List items as brief statements.

4. ALTERNATIVE EXPLANATIONS — Other interpretations that cannot be ruled \
   out given the current information. List each as a brief statement.

5. MISSING ASSESSMENT INFORMATION — Specific details that would help to \
   strengthen or refute the formulations above (e.g. developmental \
   history, trauma history, medical investigations, standardised measures).

6. CAUTION — A clinical cautionary note. Consider comorbidity, cultural \
   factors, scope of practice, and limitations of the available information.

7. CONFIDENCE — Overall confidence (0.0-1.0) in the formulations given \
   the quality and completeness of the available information.

Rules:
- NEVER state a diagnosis or use diagnostic codes.
- Use tentative language throughout. Avoid definitive statements.
- Ground formulations in the specific case information provided.
- If information is limited, say so explicitly and rate confidence accordingly.
- Be conservative with confidence ratings: 0.3-0.5 for very limited \
  information, 0.5-0.7 for moderate information, 0.7-0.9 for well-supported.

Respond EXACTLY in the JSON format below. No markdown fences. \
No text outside the JSON. Use empty lists for categories with no items.

{
  "case_summary": "Concise summary of the client's presentation and context.",
  "possible_formulations": [
    {
      "label": "Cognitive-behavioural understanding centred on avoidant coping",
      "explanation": "This formulation considers how the client's early experiences may have shaped core beliefs about themselves and others, leading to specific patterns of thinking and behaviour that maintain their current difficulties. The client's reported avoidance of social situations may reflect an attempt to manage anxiety, which in the short term provides relief but in the long term prevents disconfirmatory experiences.",
      "supporting_symptoms": [
        "Reports avoiding social gatherings for the past 6 months",
        "Describes racing thoughts before anticipated social events",
        "Endorses beliefs about being judged negatively by others"
      ],
      "confidence": 0.7
    }
  ],
  "supporting_evidence": [
    "Client reports onset following a significant life stressor",
    "Pattern of avoidance is consistent across multiple settings",
    "Mood and anxiety symptoms appear interconnected"
  ],
  "alternative_explanations": [
    "Physical health conditions may account for some reported symptoms and would require medical investigation",
    "Substance use or medication side effects cannot be fully ruled out as contributing factors",
    "Cultural factors may influence how distress is expressed and experienced"
  ],
  "missing_assessment_information": [
    "Developmental history including early attachment experiences",
    "Trauma history and previous significant adverse events",
    "Standardised measures of anxiety and depression (e.g. GAD-7, PHQ-9)",
    "Medical assessment to rule out organic causes"
  ],
  "caution": "This formulation is based on the available clinical information and should be considered tentative. It does not replace a comprehensive clinical assessment. Cultural, social, and medical factors should be considered alongside this psychological understanding.",
  "confidence": 0.65
}
"""


class ClinicalFormulator:
    """Generate clinical formulations from structured case information.

    Takes the output of the Case Understanding module (and optionally the
    Evidence Synthesis module) and produces a structured clinical formulation
    — a hypothesis about how the client's difficulties developed and are
    maintained. The formulator NEVER produces a diagnosis.

    Designed to be reusable by LangGraph agents — no pipeline dependency.

    Usage::

        formulator = ClinicalFormulator(llm_service)
        result = await formulator.formulate(case_data)
        for f in result.possible_formulations:
            print(f.label, f.confidence)
    """

    def __init__(self, llm: LLMService):
        self._llm = llm

    async def formulate(
        self,
        case_data: dict[str, Any] | None = None,
        *,
        case_summary: str | None = None,
        symptoms: list[str] | None = None,
        contextual_factors: list[str] | None = None,
        risk_factors: list[str] | None = None,
        protective_factors: list[str] | None = None,
        duration: str | None = None,
        previous_treatment: str | None = None,
        evidence_synthesis: str | None = None,
    ) -> ClinicalFormulationResult:
        """Generate a clinical formulation from case information.

        Args:
            case_data:       Optional flat dict from ``CaseUnderstandingResult.to_flat_dict()``.
                             When provided, individual keyword arguments are ignored.
            case_summary:    Free-text summary of the client's presentation.
            symptoms:        Reported signs and symptoms.
            contextual_factors: Psychosocial stressors and context.
            risk_factors:    Identified risk factors.
            protective_factors: Protective or resilience factors.
            duration:        Duration of presenting concerns.
            previous_treatment: Previous treatment history.
            evidence_synthesis: Free-text evidence synthesis summary.

        Returns:
            ClinicalFormulationResult with structured formulation fields.
        """
        t_start = time.perf_counter()

        if case_data is not None:
            prompt = self._build_prompt_from_dict(case_data, evidence_synthesis)
        else:
            prompt = self._build_prompt_from_kwargs(
                case_summary=case_summary,
                symptoms=symptoms,
                contextual_factors=contextual_factors,
                risk_factors=risk_factors,
                protective_factors=protective_factors,
                duration=duration,
                previous_treatment=previous_treatment,
                evidence_synthesis=evidence_synthesis,
            )

        # Check whether any substantive case content was provided
        has_content = bool(
            case_data
            or case_summary
            or symptoms
            or contextual_factors
            or risk_factors
            or protective_factors
            or duration
            or previous_treatment
            or evidence_synthesis
        )
        if not has_content:
            _log.warning("clinical_formulation.empty_input")
            return ClinicalFormulationResult(
                case_summary="No case information was provided for formulation. Formulation cannot proceed without case information.",
                caution="Formulation cannot proceed without case information.",
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
                "clinical_formulation.formulation_failed",
                error=str(exc),
            )
            result = ClinicalFormulationResult(
                case_summary="Clinical formulation failed. The case information is available for manual review.",
                caution="Formulation could not be completed automatically.",
            )

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        object.__setattr__(result, "formulation_ms", round(elapsed_ms, 2))
        return result

    # ── Prompt building ────────────────────────────────────

    def _build_prompt_from_dict(
        self,
        data: dict[str, Any],
        evidence_synthesis: str | None,
    ) -> str:
        parts: list[str] = ["## Case Information\n"]

        def add(label: str, key: str) -> None:
            val = data.get(key)
            if val and (isinstance(val, list) and len(val) > 0) or (isinstance(val, str) and val.strip()):
                parts.append(f"### {label}\n{val}\n")

        add("Age / Gender / Occupation", "age")
        add("Gender", "gender")
        add("Occupation", "occupation")
        add("Presenting Concerns", "presenting_concerns")
        add("Symptoms", "symptoms")
        add("Emotional Indicators", "emotional_indicators")
        add("Behavioural Indicators", "behavioural_indicators")
        add("Stressors", "stressors")
        add("Protective Factors", "protective_factors")
        add("Risk Factors", "risk_factors")
        add("Functional Impairment", "functional_impairment")
        add("Social Context", "social_context")
        add("Duration", "duration")
        add("Previous Treatment", "previous_treatment")
        add("Severity", "severity")

        if evidence_synthesis:
            parts.append(f"### Evidence Synthesis\n{evidence_synthesis}\n")

        return "\n".join(parts)

    def _build_prompt_from_kwargs(
        self,
        case_summary: str | None,
        symptoms: list[str] | None,
        contextual_factors: list[str] | None,
        risk_factors: list[str] | None,
        protective_factors: list[str] | None,
        duration: str | None,
        previous_treatment: str | None,
        evidence_synthesis: str | None,
    ) -> str:
        parts: list[str] = ["## Case Information\n"]

        if case_summary:
            parts.append(f"### Case Summary\n{case_summary}\n")
        if symptoms:
            parts.append(f"### Symptoms\n{chr(10).join('- ' + s for s in symptoms)}\n")
        if contextual_factors:
            parts.append(f"### Contextual Factors\n{chr(10).join('- ' + c for c in contextual_factors)}\n")
        if risk_factors:
            parts.append(f"### Risk Factors\n{chr(10).join('- ' + r for r in risk_factors)}\n")
        if protective_factors:
            parts.append(f"### Protective Factors\n{chr(10).join('- ' + p for p in protective_factors)}\n")
        if duration:
            parts.append(f"### Duration\n{duration}\n")
        if previous_treatment:
            parts.append(f"### Previous Treatment\n{previous_treatment}\n")
        if evidence_synthesis:
            parts.append(f"### Evidence Synthesis\n{evidence_synthesis}\n")

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

    def _build_result(self, data: dict[str, Any]) -> ClinicalFormulationResult:
        return ClinicalFormulationResult(
            case_summary=str(data.get("case_summary", "")),
            possible_formulations=self._build_formulations(data.get("possible_formulations", [])),
            supporting_evidence=[str(s) for s in data.get("supporting_evidence", [])],
            alternative_explanations=[str(s) for s in data.get("alternative_explanations", [])],
            missing_assessment_information=[str(s) for s in data.get("missing_assessment_information", [])],
            caution=str(data.get("caution", "")),
            confidence=self._clamp_confidence(data.get("confidence", 0.0)),
        )

    def _build_formulations(self, raw: list[Any]) -> list[Formulation]:
        result: list[Formulation] = []
        for item in raw:
            if not isinstance(item, dict) or not item.get("label"):
                continue
            try:
                result.append(Formulation(
                    label=str(item["label"]).strip(),
                    explanation=str(item.get("explanation", "")).strip(),
                    supporting_symptoms=[str(s) for s in item.get("supporting_symptoms", [])],
                    confidence=self._clamp_confidence(item.get("confidence", 0.0)),
                ))
            except (ValueError, TypeError) as exc:
                _log.warning("clinical_formulation.invalid_formulation", error=str(exc))
        return result

    @staticmethod
    def _clamp_confidence(v: Any) -> float:
        try:
            return max(0.0, min(1.0, float(v)))
        except (ValueError, TypeError):
            return 0.0
