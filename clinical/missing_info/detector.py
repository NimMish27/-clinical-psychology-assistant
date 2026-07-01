from __future__ import annotations

import json
import time
from typing import Any

from clinical.missing_info.models import MissingInfoItem, MissingInfoResult
from clinical.llm import LLMService
from app_logging.logger import get_logger

_log = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a clinical intake advisor. You will receive a client's statement \
or a set of clinical notes.

Your task is to identify specific clinical information that is MISSING \
from the provided text. Do NOT generate conclusions, diagnoses, or \
formulations. Only identify what you cannot determine from the given \
information.

For each missing piece of information, provide:

1. INFO GAP — What specific information is absent (e.g. "sleep quality", \
   "duration of symptoms", "suicidal ideation").

2. CLINICAL RELEVANCE — Why this missing information matters clinically. \
   How would it affect risk assessment, formulation, or treatment planning?

3. SUGGESTED QUESTIONS — 1-3 specific questions a clinician could ask to \
   gather this information. Use natural, conversational language.

Examples of common clinical information gaps:
- Sleep quality and duration
- Appetite and weight changes
- Duration and onset of symptoms
- Mood fluctuations and triggers
- Medical history and medications
- Family psychiatric history
- Substance use (alcohol, drugs, caffeine)
- Suicidal ideation, plans, or intent
- Self-harm behaviours
- Functional impairment (work, social, daily activities)
- Previous treatment and response
- Social support network
- Trauma history
- Developmental history
- Cultural and spiritual factors
- Strengths and coping strategies
- Motivation for treatment
- Risk to others

Rules:
- NEVER diagnose or suggest diagnoses.
- NEVER make clinical inferences beyond identifying gaps.
- Be thorough — if the text says "I feel sad", note that duration, \
  severity, context, and associated features are all missing.
- Prioritise safety-relevant gaps (suicidal ideation, self-harm, risk \
  to others) when the text hints at distress.
- Present questions in plain, empathetic language a clinician might use.

Respond EXACTLY in the JSON format below. No markdown fences. \
No text outside the JSON. Use empty arrays for sections with no items.

{
  "input_summary": "Brief description of what clinical information was provided (2-3 sentences).",
  "missing_information": [
    {
      "info_gap": "Sleep quality and duration",
      "clinical_relevance": "Sleep disturbance is a transdiagnostic factor that can worsen mood, anxiety, and cognitive function. It also guides treatment choices (e.g. CBT-i vs medication).",
      "suggested_questions": [
        "Can you tell me about your sleep — how many hours do you typically get?",
        "Do you have trouble falling asleep, staying asleep, or waking too early?",
        "How does your sleep affect how you feel during the day?"
      ]
    },
    {
      "info_gap": "Suicidal ideation and self-harm risk",
      "clinical_relevance": "Any expression of distress warrants a safety assessment. Identifying suicidal ideation is essential for risk management and treatment planning.",
      "suggested_questions": [
        "Have you had any thoughts about hurting yourself or ending your life?",
        "Have you ever acted on these thoughts?",
        "What supports do you have in place if you feel unsafe?"
      ]
    }
  ],
  "overall_assessment": "The provided information gives a初步 indication of low mood but lacks essential details on duration, severity, sleep, appetite, risk, medical history, and functional impact. A comprehensive assessment is needed before any clinical decisions can be made."
}
"""


class MissingInfoDetector:
    """Identify clinical information gaps in client statements or notes.

    Takes a client's description of their difficulties and identifies what
    clinically relevant information is still missing.  The detector does
    NOT generate conclusions, diagnoses, or formulations — it only reports
    gaps.

    Designed to be reusable by LangGraph agents — no pipeline dependency.

    Usage::

        detector = MissingInfoDetector(llm_service)
        result = await detector.detect("I wake up tired every day.")
        for item in result.missing_information:
            print(item.info_gap)
            for q in item.suggested_questions:
                print(f"  - {q}")
    """

    def __init__(self, llm: LLMService):
        self._llm = llm

    async def detect(
        self,
        text: str,
        *,
        context: str | None = None,
    ) -> MissingInfoResult:
        """Identify missing clinical information in the given text.

        Args:
            text:    The client's statement or clinical notes to analyse.
            context: Optional additional context (e.g. case summary,
                     formulation highlights) to inform gap detection.

        Returns:
            MissingInfoResult with identified gaps and suggested questions.
        """
        t_start = time.perf_counter()

        if not text or not text.strip():
            _log.warning("missing_info.empty_input")
            return MissingInfoResult(
                input_summary="No input was provided.",
                overall_assessment="Cannot identify missing information from empty input.",
            )

        try:
            prompt = self._build_prompt(text, context=context)
            raw = await self._llm.generate(
                prompt=prompt,
                system_prompt=_SYSTEM_PROMPT,
            )
            data = self._parse_response(raw)
            result = self._build_result(data, text)
        except Exception as exc:
            _log.error(
                "missing_info.detection_failed",
                error=str(exc),
            )
            result = MissingInfoResult(
                input_summary=str(text)[:200],
                overall_assessment="Missing information detection could not be completed automatically.",
            )

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        object.__setattr__(result, "detection_ms", round(elapsed_ms, 2))
        return result

    # ── Prompt building ────────────────────────────────────

    def _build_prompt(self, text: str, *, context: str | None) -> str:
        parts: list[str] = [
            f"## Client Statement / Clinical Notes\n{text.strip()}\n",
        ]
        if context:
            parts.append(f"## Additional Context\n{context.strip()}\n")
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

    def _build_result(self, data: dict[str, Any], text: str) -> MissingInfoResult:
        return MissingInfoResult(
            input_summary=str(data.get("input_summary", text[:200])),
            missing_information=self._build_items(data.get("missing_information", [])),
            overall_assessment=str(data.get("overall_assessment", "")),
        )

    def _build_items(self, raw: list[Any]) -> list[MissingInfoItem]:
        result: list[MissingInfoItem] = []
        for item in raw:
            if not isinstance(item, dict) or not item.get("info_gap"):
                continue
            try:
                result.append(MissingInfoItem(
                    info_gap=str(item["info_gap"]).strip(),
                    clinical_relevance=str(item.get("clinical_relevance", "")).strip(),
                    suggested_questions=[str(q) for q in item.get("suggested_questions", [])],
                ))
            except (ValueError, TypeError) as exc:
                _log.warning("missing_info.invalid_item", error=str(exc))
        return result
