from __future__ import annotations

import json
import time
from typing import Any

from clinical.evidence_synthesis.models import (
    Agreement,
    EvidenceSynthesisResult,
    Finding,
    Implication,
    Theme,
    Uncertainty,
)
from clinical.llm import LLMService
from app_logging.logger import get_logger

_log = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are an evidence synthesis specialist in clinical psychology. You will \
receive a set of retrieved document chunks relevant to a clinical case.

Your task is to produce a structured evidence synthesis. Do NOT copy or \
paraphrase the chunks. Instead, synthesise across them to identify:

1. KEY FINDINGS — The most important evidence-based conclusions relevant \
   to the case. Each finding must cite the source(s) that support it \
   (e.g. "DSM5.pdf, p.12"). Include a confidence rating (0.0-1.0).

2. COMMON THEMES — Recurring topics or patterns that appear across \
   multiple sources. Describe each theme and estimate its prevalence \
   (0.0-1.0 = how consistently it appears across sources).

3. AREAS OF AGREEMENT — Topics where multiple sources converge on the \
   same conclusion. List what they agree on and which sources agree.

4. AREAS OF UNCERTAINTY — Gaps, conflicting findings, or limitations \
   in the evidence base. Describe how each uncertainty affects \
   clinical decision-making.

5. PRACTICAL IMPLICATIONS — Actionable recommendations for the \
   clinician that follow from the evidence. Rate the strength of \
   each implication (0.0-1.0).

6. OVERALL SUMMARY — A single concise paragraph (3-5 sentences) that \
   captures the essence of what the evidence says.

7. CONFIDENCE — Overall confidence in this synthesis (0.0-1.0). \
   Consider: number and quality of sources, consistency across sources, \
   recency, and direct relevance to the case.

Rules:
- Every finding and agreement MUST cite specific sources by filename and page.
- Never copy chunks verbatim. Paraphrase and synthesise.
- If evidence is sparse or inconsistent, say so clearly.
- Be conservative with confidence ratings — prefer 0.6-0.8 for moderately \
  supported claims, 0.3-0.5 for tentative ones.

Respond EXACTLY in the JSON format below. No markdown fences. \
No text outside the JSON. Use empty lists for categories with no findings.

{
  "key_findings": [
    {
      "statement": "Concise finding statement",
      "supporting_sources": ["Source.pdf, p.12"],
      "confidence": 0.85
    }
  ],
  "common_themes": [
    {
      "name": "Theme label",
      "description": "How this theme appears across evidence",
      "prevalence": 0.8
    }
  ],
  "areas_of_agreement": [
    {
      "topic": "Topic label",
      "consensus": "What sources agree on",
      "supporting_sources": ["SourceA.pdf", "SourceB.pdf"]
    }
  ],
  "areas_of_uncertainty": [
    {
      "topic": "Topic label",
      "description": "Nature of gap or conflict",
      "implications": "How this affects clinical decisions"
    }
  ],
  "practical_implications": [
    {
      "recommendation": "Actionable recommendation",
      "strength": 0.75,
      "source": "Key guideline or source"
    }
  ],
  "overall_summary": "Concise paragraph synthesising all findings.",
  "confidence": 0.75
}
"""


class EvidenceSynthesizer:
    """Synthesise retrieved evidence into a structured clinical summary.

    Takes chunks retrieved by the RAG pipeline and produces a concise,
    structured evidence synthesis that clinicians can act on without
    reading the raw chunks.

    Designed to be reusable by LangGraph agents — no pipeline dependency.

    Usage::

        synthesizer = EvidenceSynthesizer(llm_service)
        result = await synthesizer.synthesise(chunks, query="CBT for GAD")
        for finding in result.key_findings:
            print(finding.statement)
    """

    def __init__(self, llm: LLMService):
        self._llm = llm

    async def synthesise(
        self,
        chunks: list[Any],
        *,
        query: str | None = None,
        case_context: str | None = None,
    ) -> EvidenceSynthesisResult:
        """Synthesise evidence from retrieved chunks.

        Args:
            chunks:        List of retrieved chunk objects. Each must have
                           ``text``, ``source``, ``page``, and ``score``
                           attributes (like ``RetrievedChunk``), or be a
                           dict with those keys.
            query:         Optional original query for context.
            case_context:  Optional case description for context.

        Returns:
            EvidenceSynthesisResult with structured synthesis fields.
        """
        t_start = time.perf_counter()
        if not chunks:
            _log.warning("evidence_synthesis.empty_chunks")
            return EvidenceSynthesisResult(
                overall_summary="No evidence chunks were provided for synthesis.",
                chunks_analysed=0,
            )

        try:
            raw = await self._llm.generate(
                prompt=self._build_prompt(chunks, query=query, case_context=case_context),
                system_prompt=_SYSTEM_PROMPT,
            )
            data = self._parse_response(raw)
            result = self._build_result(data, len(chunks))
        except Exception as exc:
            _log.error(
                "evidence_synthesis.synthesis_failed",
                error=str(exc),
                chunks=len(chunks),
            )
            result = EvidenceSynthesisResult(
                overall_summary="Evidence synthesis failed. The retrieved chunks are available for manual review.",
                chunks_analysed=len(chunks),
            )

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        object.__setattr__(result, "synthesis_ms", round(elapsed_ms, 2))
        return result

    def _build_prompt(
        self,
        chunks: list[Any],
        *,
        query: str | None,
        case_context: str | None,
    ) -> str:
        parts: list[str] = []

        if query:
            parts.append(f"## Clinical Query\n{query}\n")
        if case_context:
            parts.append(f"## Case Context\n{case_context}\n")

        parts.append(f"## Retrieved Evidence ({len(chunks)} chunks)\n")
        for i, c in enumerate(chunks, 1):
            text, source, page, score = self._unpack_chunk(c)
            header = f"### Chunk {i} — {source}, p.{page} (similarity: {score:.2f})"
            # Truncate very long chunks to keep prompt size manageable
            truncated = text[:800] if len(text) > 800 else text
            parts.append(f"{header}\n{truncated}\n")

        return "\n".join(parts)

    def _unpack_chunk(self, chunk: Any) -> tuple[str, str, int, float]:
        """Extract text, source, page, score from a chunk regardless of type."""
        if hasattr(chunk, "text"):
            text = chunk.text
            source = getattr(chunk, "source", "unknown")
            page = getattr(chunk, "page", 0)
            score = getattr(chunk, "score", 0.0)
        elif isinstance(chunk, dict):
            text = chunk.get("text", "")
            source = chunk.get("source", "unknown")
            page = chunk.get("page", 0)
            score = chunk.get("score", 0.0)
        else:
            text = str(chunk)
            source = "unknown"
            page = 0
            score = 0.0
        return str(text), str(source), int(page), float(score)

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
        data: dict[str, Any],
        chunk_count: int,
    ) -> EvidenceSynthesisResult:
        return EvidenceSynthesisResult(
            key_findings=self._build_findings(data.get("key_findings", [])),
            common_themes=self._build_themes(data.get("common_themes", [])),
            areas_of_agreement=self._build_agreements(data.get("areas_of_agreement", [])),
            areas_of_uncertainty=self._build_uncertainties(data.get("areas_of_uncertainty", [])),
            practical_implications=self._build_implications(data.get("practical_implications", [])),
            overall_summary=str(data.get("overall_summary", "")),
            confidence=self._clamp_confidence(data.get("confidence", 0.0)),
            chunks_analysed=chunk_count,
        )

    def _build_findings(self, raw: list[Any]) -> list[Finding]:
        result: list[Finding] = []
        for item in raw:
            if not isinstance(item, dict) or not item.get("statement"):
                continue
            try:
                result.append(Finding(
                    statement=str(item["statement"]).strip(),
                    supporting_sources=[str(s) for s in item.get("supporting_sources", [])],
                    confidence=self._clamp_confidence(item.get("confidence", 0.0)),
                ))
            except (ValueError, TypeError) as exc:
                _log.warning("evidence_synthesis.invalid_finding", error=str(exc))
        return result

    def _build_themes(self, raw: list[Any]) -> list[Theme]:
        result: list[Theme] = []
        for item in raw:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            try:
                result.append(Theme(
                    name=str(item["name"]).strip(),
                    description=str(item.get("description", "")).strip(),
                    prevalence=self._clamp_confidence(item.get("prevalence", 0.0)),
                ))
            except (ValueError, TypeError) as exc:
                _log.warning("evidence_synthesis.invalid_theme", error=str(exc))
        return result

    def _build_agreements(self, raw: list[Any]) -> list[Agreement]:
        result: list[Agreement] = []
        for item in raw:
            if not isinstance(item, dict) or not item.get("topic"):
                continue
            try:
                result.append(Agreement(
                    topic=str(item["topic"]).strip(),
                    consensus=str(item.get("consensus", "")).strip(),
                    supporting_sources=[str(s) for s in item.get("supporting_sources", [])],
                ))
            except (ValueError, TypeError) as exc:
                _log.warning("evidence_synthesis.invalid_agreement", error=str(exc))
        return result

    def _build_uncertainties(self, raw: list[Any]) -> list[Uncertainty]:
        result: list[Uncertainty] = []
        for item in raw:
            if not isinstance(item, dict) or not item.get("topic"):
                continue
            try:
                result.append(Uncertainty(
                    topic=str(item["topic"]).strip(),
                    description=str(item.get("description", "")).strip(),
                    implications=str(item.get("implications", "")).strip(),
                ))
            except (ValueError, TypeError) as exc:
                _log.warning("evidence_synthesis.invalid_uncertainty", error=str(exc))
        return result

    def _build_implications(self, raw: list[Any]) -> list[Implication]:
        result: list[Implication] = []
        for item in raw:
            if not isinstance(item, dict) or not item.get("recommendation"):
                continue
            try:
                result.append(Implication(
                    recommendation=str(item["recommendation"]).strip(),
                    strength=self._clamp_confidence(item.get("strength", 0.0)),
                    source=str(item["source"]).strip() if item.get("source") else None,
                ))
            except (ValueError, TypeError) as exc:
                _log.warning("evidence_synthesis.invalid_implication", error=str(exc))
        return result

    @staticmethod
    def _clamp_confidence(v: Any) -> float:
        try:
            return max(0.0, min(1.0, float(v)))
        except (ValueError, TypeError):
            return 0.0
