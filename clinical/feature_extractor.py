from __future__ import annotations

import json

from clinical.llm import LLMService
from clinical.models import (
    CaseUnderstanding,
    ClinicalFeatures,
    PipelineError,
    PipelineStage,
)
from app_logging.logger import get_logger

_log = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a clinical psychology diagnostician. Extract structured clinical \
features from the case understanding below.

Identify:
- symptoms: specific signs and symptoms reported
- diagnoses: any mentioned or suspected diagnoses
- patient_history: relevant personal, medical, or psychiatric history
- family_history: family psychiatric or medical history
- risk_factors: suicide risk, substance use, social isolation, etc.
- protective_factors: social support, treatment engagement, resilience
- treatment_history: past or current treatments and response
- other_relevant: any other clinically relevant observations

Respond EXACTLY in this JSON format — use empty lists for missing categories:
{"symptoms": [...], "diagnoses": [...], "patient_history": [...], \
"family_history": [...], "risk_factors": [...], "protective_factors": [...], \
"treatment_history": [...], "other_relevant": [...]}
"""


class FeatureExtractor:
    def __init__(self, llm: LLMService):
        self._llm = llm

    async def extract(self, case: CaseUnderstanding) -> ClinicalFeatures:
        try:
            raw = await self._llm.generate(
                prompt=self._build_prompt(case),
                system_prompt=_SYSTEM_PROMPT,
            )
            data = self._parse(raw)
            return ClinicalFeatures(**data)
        except Exception as exc:
            raise PipelineError(
                stage=PipelineStage.FEATURE_EXTRACTION,
                message=f"Feature extraction failed: {exc}",
                cause=exc,
            ) from exc

    def _build_prompt(self, case: CaseUnderstanding) -> str:
        parts = [f"Input type: {case.input_type.value}"]
        if case.summary:
            parts.append(f"Summary: {case.summary}")
        if case.key_topics:
            parts.append(f"Key topics: {', '.join(case.key_topics)}")
        if case.clinical_context:
            parts.append(f"Context: {case.clinical_context}")
        return "\n".join(parts)

    def _parse(self, raw: str) -> dict:
        start = raw.index("{")
        end = raw.rindex("}")
        return json.loads(raw[start : end + 1])
