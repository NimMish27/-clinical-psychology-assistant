from __future__ import annotations

import json

from clinical.llm import LLMService
from clinical.models import (
    CaseUnderstanding,
    ClinicalFeatures,
    ClinicalFormulation,
    EvidenceSynthesis,
    PipelineError,
    PipelineStage,
)
from app_logging.logger import get_logger

_log = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a senior clinical psychologist formulating a case. Based on the \
case understanding, clinical features, and synthesised evidence below, \
generate a comprehensive clinical formulation.

NEVER diagnose definitively. Use careful clinical language and differential considerations.

Produce:
1. Case Summary: A comprehensive summary of the clinical case.
2. Possible Formulations: A list of formulations (each with explanation, supporting symptoms, and confidence level [High/Moderate/Low]).
3. Supporting Evidence: List of evidence points supporting these formulations.
4. Alternative Explanations: Alternative explanations or differential considerations (do not diagnose).
5. Missing Assessment Information: What information is missing that would help clarify the formulations.

Respond EXACTLY in this JSON format:
{"case_summary": "...", "possible_formulations": [{"explanation": "...", "supporting_symptoms": ["..."], "confidence_level": "..."}], \
"supporting_evidence": ["..."], "alternative_explanations": ["..."], "missing_assessment_information": ["..."]}
"""


class FormulationGenerator:
    def __init__(self, llm: LLMService):
        self._llm = llm

    async def generate(
        self,
        evidence: EvidenceSynthesis,
        case: CaseUnderstanding,
        features: ClinicalFeatures,
    ) -> ClinicalFormulation:
        try:
            raw = await self._llm.generate(
                prompt=self._build_prompt(evidence, case, features),
                system_prompt=_SYSTEM_PROMPT,
            )
            data = self._parse(raw)
            data.setdefault("case_summary", data.pop("analysis", "No case summary available."))
            return ClinicalFormulation(**data)
        except Exception as exc:
            raise PipelineError(
                stage=PipelineStage.FORMULATION_GENERATION,
                message=f"Formulation generation failed: {exc}",
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

        parts.append(f"\n## Evidence Synthesis\n{evidence.evidence_summary}")
        if evidence.key_findings:
            parts.append("\nKey Findings:")
            for e in evidence.key_findings:
                parts.append(f"- {e}")
        if evidence.practical_implications:
            parts.append("\nPractical Implications:")
            for e in evidence.practical_implications:
                parts.append(f"- {e}")
                
        return "\n".join(parts)

    def _parse(self, raw: str) -> dict:
        start = raw.index("{")
        end = raw.rindex("}")
        return json.loads(raw[start : end + 1])
