from __future__ import annotations

import json

from clinical.llm import LLMService
from clinical.models import (
    ClinicalFeatures,
    EvidenceSynthesis,
    PipelineError,
    PipelineStage,
    RetrievalQuery,
)
from rag.retriever import RetrievedChunk
from app_logging.logger import get_logger

_log = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a clinical evidence synthesis specialist. Review the retrieved \
evidence chunks alongside the clinical features extracted from a case.

Produce:
1. Key findings from the evidence.
2. Common themes across the retrieved literature.
3. Areas of agreement in the evidence.
4. Areas of uncertainty or conflicting evidence.
5. Practical implications for clinicians.
6. A concise evidence summary (DO NOT copy the raw chunks, just summarize).

Focus on diagnostic criteria, treatment efficacy, risk assessment, and \
clinical guidelines. Note any gaps in the evidence base.

Respond EXACTLY in this JSON format:
{"key_findings": ["..."], "common_themes": ["..."], \
"areas_of_agreement": ["..."], "areas_of_uncertainty": ["..."], \
"practical_implications": ["..."], "evidence_summary": "..."}
"""


class EvidenceSynthesizer:
    def __init__(self, llm: LLMService):
        self._llm = llm

    async def synthesize(
        self,
        chunks: list[RetrievedChunk],
        queries: list[RetrievalQuery],
        features: ClinicalFeatures,
    ) -> EvidenceSynthesis:
        try:
            if not chunks:
                return EvidenceSynthesis(
                    key_findings=[],
                    common_themes=[],
                    areas_of_agreement=[],
                    areas_of_uncertainty=["No relevant clinical evidence was found in the knowledge base."],
                    practical_implications=[],
                    evidence_summary="No evidence available.",
                )

            raw = await self._llm.generate(
                prompt=self._build_prompt(chunks, queries, features),
                system_prompt=_SYSTEM_PROMPT,
            )
            data = self._parse(raw)
            data.setdefault("evidence_summary", data.pop("synthesis", "No evidence summary available."))
            return EvidenceSynthesis(**data)
        except Exception as exc:
            raise PipelineError(
                stage=PipelineStage.EVIDENCE_SYNTHESIS,
                message=f"Evidence synthesis failed: {exc}",
                cause=exc,
            ) from exc

    def _build_prompt(
        self,
        chunks: list[RetrievedChunk],
        queries: list[RetrievalQuery],
        features: ClinicalFeatures,
    ) -> str:
        parts = ["## Clinical Features"]
        for field in features.model_fields_set:
            values = getattr(features, field)
            if values:
                parts.append(f"{field}: {', '.join(values)}")

        parts.append("\n## Retrieval Queries Used")
        for q in queries:
            parts.append(f"- {q.query} (weight={q.weight})")

        parts.append(f"\n## Retrieved Evidence ({len(chunks)} chunks)")
        for i, c in enumerate(chunks, 1):
            source = f"[{c.source}, p.{c.page}, score={c.score:.2f}]"
            parts.append(f"{i}. {source} {c.text[:300]}")
        return "\n".join(parts)

    def _parse(self, raw: str) -> dict:
        start = raw.index("{")
        end = raw.rindex("}")
        return json.loads(raw[start : end + 1])
