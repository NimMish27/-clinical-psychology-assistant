from __future__ import annotations

import json

from clinical.llm import LLMService
from clinical.models import (
    CaseUnderstanding,
    ClinicalFeatures,
    ClinicalResponse,
    EvidenceSynthesis,
    PipelineError,
    PipelineStage,
)
from app_logging.logger import get_logger

_log = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a senior clinical psychologist providing a case analysis. Based on \
the case understanding, clinical features, and synthesised evidence below, \
generate a comprehensive clinical response.

Include:
1. A clinical analysis synthesising all available information.
2. A case formulation integrating biopsychosocial factors (if sufficient \
evidence exists, otherwise null).
3. 3-5 evidence-based recommendations.
4. A brief summary of the evidence used.
5. A confidence score (0.0-1.0).
6. Limitations and caveats (e.g. "No retrieved evidence on comorbid \
conditions", "Diagnosis requires in-person assessment").

Be precise, clinical, and cautious. Do not diagnose definitively — use \
formulation language. Note when evidence is absent.

Respond EXACTLY in this JSON format:
{"analysis": "...", "formulation": "...", "recommendations": ["..."], \
"evidence_summary": "...", "confidence": 0.0, "limitations": ["..."]}

If a formulation cannot be made, set "formulation" to null.
"""


class ResponseGenerator:
    def __init__(self, llm: LLMService):
        self._llm = llm

    async def generate(
        self,
        evidence: EvidenceSynthesis,
        case: CaseUnderstanding,
        features: ClinicalFeatures,
    ) -> ClinicalResponse:
        try:
            raw = await self._llm.generate(
                prompt=self._build_prompt(evidence, case, features),
                system_prompt=_SYSTEM_PROMPT,
            )
            data = self._parse(raw)
            return ClinicalResponse(**data)
        except Exception as exc:
            raise PipelineError(
                stage=PipelineStage.RESPONSE_GENERATION,
                message=f"Response generation failed: {exc}",
                cause=exc,
            ) from exc

    def _build_prompt(
        self,
        evidence: EvidenceSynthesis,
        case: CaseUnderstanding,
        features: ClinicalFeatures,
    ) -> str:
        parts = [
            f"## Case Understanding\nInput type: {case.input_type.value}",
        ]
        if case.summary:
            parts.append(f"Summary: {case.summary}")

        parts.append("\n## Clinical Features")
        for field in features.model_fields_set:
            values = getattr(features, field)
            if values:
                parts.append(f"{field}: {', '.join(values)}")

        parts.append(f"\n## Evidence Synthesis\n{evidence.synthesis}")
        if evidence.supporting_evidence:
            parts.append("\nSupporting evidence:")
            for e in evidence.supporting_evidence:
                parts.append(f"- {e}")
        if evidence.contradicting_evidence:
            parts.append("\nContradicting evidence:")
            for e in evidence.contradicting_evidence:
                parts.append(f"- {e}")
        parts.append(f"\nEvidence confidence: {evidence.confidence}")
        return "\n".join(parts)

    def _parse(self, raw: str) -> dict:
        start = raw.index("{")
        end = raw.rindex("}")
        return json.loads(raw[start : end + 1])
