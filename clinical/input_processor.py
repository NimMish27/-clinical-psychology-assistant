from __future__ import annotations

from clinical.llm import LLMService
from clinical.models import (
    CaseUnderstanding,
    ClinicalInput,
    ClinicalInputType,
    PipelineError,
    PipelineStage,
)
from app_logging.logger import get_logger

_log = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a clinical psychology intake specialist. Analyse the following input \
and determine:
1. Input type — is this a single client statement, a list of symptoms, or a \
full case study?
2. Provide a one-paragraph clinical summary.
3. List the key clinical topics (e.g. "depression", "CBT", "anxiety", "trauma").
4. Identify the clinical context if apparent (e.g. "outpatient clinic", \
"emergency assessment", "therapy session note").

Respond EXACTLY in this JSON format:
{"input_type": "single_statement|symptom_list|case_study", "summary": "...", \
"key_topics": ["..."], "clinical_context": "..."}
"""


class InputProcessor:
    def __init__(self, llm: LLMService):
        self._llm = llm

    async def process(self, inp: ClinicalInput) -> CaseUnderstanding:
        try:
            if inp.input_type:
                return CaseUnderstanding(
                    input_type=inp.input_type,
                    summary="",
                    key_topics=[],
                    clinical_context=None,
                )

            raw = await self._llm.generate(
                prompt=inp.raw_text,
                system_prompt=_SYSTEM_PROMPT,
            )
            parsed = self._parse(raw)
            return CaseUnderstanding(**parsed)
        except Exception as exc:
            raise PipelineError(
                stage=PipelineStage.INPUT_PROCESSING,
                message=f"Failed to process input: {exc}",
                cause=exc,
            ) from exc

    def _parse(self, raw: str) -> dict:
        import json
        start = raw.index("{")
        end = raw.rindex("}")
        parsed = json.loads(raw[start : end + 1])
        parsed["input_type"] = ClinicalInputType(parsed["input_type"])
        return parsed
