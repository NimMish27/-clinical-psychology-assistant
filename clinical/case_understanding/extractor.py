from __future__ import annotations

import json
import time
from typing import Any

from clinical.case_understanding.models import (
    CaseUnderstandingResult,
    ClinicalPresentation,
    ConfidenceRating,
    ContextualFactors,
    DemographicInfo,
    Duration,
    ExtractedField,
    OverallSeverity,
    PreviousTreatment,
    Severity,
    TreatmentHistory,
)
from clinical.llm import LLMService
from app_logging.logger import get_logger

_log = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a clinical intake specialist. Extract structured case information from \
the therapist notes below.

Analyse the text and extract the following fields. For every field, provide:
- The extracted value (use null if not mentioned)
- A confidence rating: "high", "medium", "low", or "unknown"
- The exact source text snippet that supports this value (or null)

Fields to extract:
1. age — client age (number or null)
2. gender — client gender (string or null)
3. occupation — client occupation (string or null)
4. presenting_concerns — list of reasons the client is seeking help
5. symptoms — list of reported signs and symptoms
6. emotional_indicators — affective/emotional state (e.g. anxious, irritable, euthymic)
7. behavioural_indicators — observed or reported behaviours (e.g. withdrawal, restlessness)
8. stressors — list of psychosocial stressors
9. protective_factors — factors that support resilience
10. risk_factors — risk indicators (self-harm, substance use, isolation, etc.)
11. functional_impairment — how symptoms affect daily functioning (string or null)
12. social_context — relationships, living situation, support system (string or null)
13. duration — duration of concerns (object with value, unit, original_text, or null)
14. previous_treatment — list of previous treatments (each with modality, response, duration, original_text)
15. severity — overall severity: "mild", "moderate", "severe", or "unspecified"
16. severity_rationale — brief justification for the severity rating

Respond EXACTLY in the JSON format below. Do not include markdown fences or \
any text outside the JSON object. Every field is required in the JSON, \
use null or empty lists as appropriate.

{
  "age": {"value": null, "confidence": "unknown", "source_text": null},
  "gender": {"value": null, "confidence": "unknown", "source_text": null},
  "occupation": {"value": null, "confidence": "unknown", "source_text": null},
  "presenting_concerns": [],
  "symptoms": [],
  "emotional_indicators": [],
  "behavioural_indicators": [],
  "stressors": [],
  "protective_factors": [],
  "risk_factors": [],
  "functional_impairment": {"value": null, "confidence": "unknown", "source_text": null},
  "social_context": {"value": null, "confidence": "unknown", "source_text": null},
  "duration": {"value": null, "unit": null, "original_text": null},
  "previous_treatment": [],
  "severity": "unspecified",
  "severity_rationale": null
}

Each list item must be: {"value": "...", "confidence": "medium", "source_text": "..."}
Each previous_treatment item must be: {"modality": "...", "response": "...", "duration": "...", "original_text": "..."}
"""


class CaseUnderstandingExtractor:
    """Extract structured case understanding from free-text clinical notes.

    This class is designed to be reusable by future LangGraph agents.
    It has no dependency on the pipeline orchestrator or any specific
    workflow — just the LLM service and the models.

    Usage::

        extractor = CaseUnderstandingExtractor(llm_service)
        result = await extractor.extract(therapist_notes)
        result.to_flat_dict()  # simple dict for agent consumption
    """

    def __init__(self, llm: LLMService):
        self._llm = llm

    async def extract(
        self,
        text: str,
    ) -> CaseUnderstandingResult:
        if not text or not text.strip():
            _log.warning("case_understanding.empty_input")
            return self._empty_result(text)

        t_start = time.perf_counter()
        try:
            raw = await self._llm.generate(
                prompt=text,
                system_prompt=_SYSTEM_PROMPT,
            )
            data = self._parse_response(raw)
            result = self._build_result(text, data)
        except Exception as exc:
            _log.error(
                "case_understanding.extraction_failed",
                error=str(exc),
                text_length=len(text),
            )
            result = self._empty_result(text)

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        object.__setattr__(result, "extraction_ms", round(elapsed_ms, 2))
        return result

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

    def _build_result(
        self,
        text: str,
        data: dict[str, Any],
    ) -> CaseUnderstandingResult:
        return CaseUnderstandingResult(
            demographic=self._build_demographic(data),
            clinical_presentation=self._build_clinical_presentation(data),
            contextual_factors=self._build_contextual_factors(data),
            treatment_history=self._build_treatment_history(data),
            overall_severity=self._build_severity(data),
            raw_text=text,
        )

    def _build_demographic(self, data: dict[str, Any]) -> DemographicInfo:
        return DemographicInfo(
            age=self._extracted_field(data, "age"),
            gender=self._extracted_field(data, "gender"),
            occupation=self._extracted_field(data, "occupation"),
        )

    def _build_clinical_presentation(self, data: dict[str, Any]) -> ClinicalPresentation:
        return ClinicalPresentation(
            presenting_concerns=self._extracted_list(data, "presenting_concerns"),
            symptoms=self._extracted_list(data, "symptoms"),
            emotional_indicators=self._extracted_list(data, "emotional_indicators"),
            behavioural_indicators=self._extracted_list(data, "behavioural_indicators"),
            duration=self._build_duration(data.get("duration")),
        )

    def _build_contextual_factors(self, data: dict[str, Any]) -> ContextualFactors:
        return ContextualFactors(
            stressors=self._extracted_list(data, "stressors"),
            protective_factors=self._extracted_list(data, "protective_factors"),
            risk_factors=self._extracted_list(data, "risk_factors"),
            functional_impairment=self._extracted_field(data, "functional_impairment"),
            social_context=self._extracted_field(data, "social_context"),
        )

    def _build_treatment_history(self, data: dict[str, Any]) -> TreatmentHistory:
        raw = data.get("previous_treatment", [])
        if not raw or not isinstance(raw, list):
            return TreatmentHistory()
        treatments = []
        for item in raw:
            if isinstance(item, dict):
                treatments.append(
                    PreviousTreatment(
                        modality=item.get("modality"),
                        response=item.get("response"),
                        duration=item.get("duration"),
                        original_text=item.get("original_text"),
                    )
                )
        return TreatmentHistory(previous_treatment=treatments)

    def _build_severity(self, data: dict[str, Any]) -> OverallSeverity:
        raw = data.get("severity", "unspecified")
        try:
            severity = Severity(raw)
        except ValueError:
            severity = Severity.UNSPECIFIED

        conf_raw = data.get("severity_confidence", "unknown")
        try:
            confidence = ConfidenceRating(conf_raw)
        except ValueError:
            confidence = ConfidenceRating.UNKNOWN

        return OverallSeverity(
            severity=severity,
            confidence=confidence,
            rationale=data.get("severity_rationale"),
        )

    def _build_duration(self, raw: Any) -> Duration | None:
        if not raw or not isinstance(raw, dict):
            return None
        value = raw.get("value")
        unit = raw.get("unit")
        original_text = raw.get("original_text")
        if value is None and unit is None and original_text is None:
            return None
        return Duration(
            value=value,
            unit=unit,
            original_text=original_text,
        )

    def _extracted_field(
        self,
        data: dict[str, Any],
        key: str,
    ) -> ExtractedField | None:
        raw = data.get(key)
        if not raw or not isinstance(raw, dict):
            return None
        if raw.get("value") is None:
            return None

        conf_raw = raw.get("confidence", "unknown")
        try:
            confidence = ConfidenceRating(conf_raw)
        except ValueError:
            confidence = ConfidenceRating.UNKNOWN

        return ExtractedField(
            value=raw["value"],
            confidence=confidence,
            source_text=raw.get("source_text"),
        )

    def _extracted_list(
        self,
        data: dict[str, Any],
        key: str,
    ) -> list[ExtractedField]:
        raw = data.get(key, [])
        if not raw or not isinstance(raw, list):
            return []
        result = []
        for item in raw:
            if not isinstance(item, dict) or item.get("value") is None:
                continue
            conf_raw = item.get("confidence", "unknown")
            try:
                confidence = ConfidenceRating(conf_raw)
            except ValueError:
                confidence = ConfidenceRating.UNKNOWN
            result.append(
                ExtractedField(
                    value=item["value"],
                    confidence=confidence,
                    source_text=item.get("source_text"),
                )
            )
        return result

    def _empty_result(self, text: str) -> CaseUnderstandingResult:
        return CaseUnderstandingResult(
            raw_text=text,
        )
