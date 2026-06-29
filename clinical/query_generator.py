from __future__ import annotations

import json

from clinical.llm import LLMService
from clinical.models import (
    CaseUnderstanding,
    ClinicalFeatures,
    PipelineError,
    PipelineStage,
    RetrievalQuery,
)
from app_logging.logger import get_logger

_log = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a clinical information retrieval specialist. Given the clinical \
features extracted from a case, generate 3-6 targeted search queries to \
find the most relevant clinical evidence in a knowledge base of clinical \
documents (DSM-5, ICD-11, treatment guidelines, research papers).

Each query should:
- Be a natural language question or statement
- Target specific clinical knowledge (diagnostic criteria, treatment \
efficacy, differential diagnosis, risk factors, prognosis)
- Include a weight (0.1-3.0) indicating importance — higher weight for \
the most critical queries
- Include a brief rationale explaining why this query matters

Focus on evidence that would help a clinician formulate a diagnosis, \
treatment plan, or case conceptualisation.

Respond EXACTLY in this JSON format:
{"queries": [{"query": "...", "weight": 1.0, "rationale": "..."}, ...]}
"""


class QueryGenerator:
    def __init__(self, llm: LLMService):
        self._llm = llm

    async def generate(
        self,
        features: ClinicalFeatures,
        case: CaseUnderstanding,
    ) -> list[RetrievalQuery]:
        try:
            raw = await self._llm.generate(
                prompt=self._build_prompt(features, case),
                system_prompt=_SYSTEM_PROMPT,
            )
            data = self._parse(raw)
            queries = [RetrievalQuery(**q) for q in data["queries"]]
            if not queries:
                queries = [self._fallback_query(features)]
            return queries
        except Exception as exc:
            _log.warning(
                "query_generator.fallback",
                error=str(exc),
            )
            return [self._fallback_query(features)]

    def _build_prompt(
        self,
        features: ClinicalFeatures,
        case: CaseUnderstanding,
    ) -> str:
        lines = [f"Input type: {case.input_type.value}"]
        if case.summary:
            lines.append(f"Case summary: {case.summary}")
        lines.append("")
        lines.append("Clinical features:")
        for field in features.model_fields_set:
            values = getattr(features, field)
            if values:
                lines.append(f"  {field}: {', '.join(values)}")
        return "\n".join(lines)

    def _parse(self, raw: str) -> dict:
        start = raw.index("{")
        end = raw.rindex("}")
        return json.loads(raw[start : end + 1])

    def _fallback_query(self, features: ClinicalFeatures) -> RetrievalQuery:
        terms = []
        if features.diagnoses:
            terms.extend(features.diagnoses[:2])
        if features.symptoms:
            terms.extend(features.symptoms[:2])
        query = " ".join(terms) if terms else "clinical assessment guidelines"
        return RetrievalQuery(
            query=query,
            weight=1.0,
            rationale="Fallback query from extracted terms.",
        )
