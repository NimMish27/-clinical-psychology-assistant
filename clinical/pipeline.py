from __future__ import annotations

import time

from clinical.evidence_synthesizer import EvidenceSynthesizer
from clinical.feature_extractor import FeatureExtractor
from clinical.formulation_generator import FormulationGenerator
from clinical.input_processor import InputProcessor
from clinical.llm import LLMService
from clinical.models import (
    ClinicalInput,
    PipelineError,
    PipelineResult,
    PipelineStage,
)
from clinical.query_generator import QueryGenerator
from rag.retriever import Retriever
from app_logging.logger import get_logger

_log = get_logger(__name__)


class ClinicalPipeline:
    """Orchestrator for the full clinical case analysis pipeline.

    Stages:
        Input → CaseUnderstanding → ClinicalFeatures → RetrievalQueries
        → RetrievedChunks → EvidenceSynthesis → ClinicalFormulation
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMService | None = None,
    ):
        self._retriever = retriever
        llm = llm or LLMService()
        self._input_processor = InputProcessor(llm)
        self._feature_extractor = FeatureExtractor(llm)
        self._query_generator = QueryGenerator(llm)
        self._evidence_synthesizer = EvidenceSynthesizer(llm)
        self._formulation_generator = FormulationGenerator(llm)

    async def run(self, inp: ClinicalInput) -> PipelineResult:
        t_start = time.perf_counter()

        understanding = await self._input_processor.process(inp)
        features = await self._feature_extractor.extract(understanding)
        queries = await self._query_generator.generate(features, understanding)

        all_chunks = []
        for q in queries:
            try:
                result = await self._retriever.aretrieve(
                    q.query,
                    n_results=3,
                )
                all_chunks.extend(result.chunks)
            except Exception as exc:
                _log.warning(
                    "pipeline.retrieval_failed",
                    query=q.query,
                    error=str(exc),
                )

        all_chunks = self._deduplicate(all_chunks)
        evidence = await self._evidence_synthesizer.synthesize(
            all_chunks, queries, features,
        )
        formulation = await self._formulation_generator.generate(
            evidence, understanding, features,
        )

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        return PipelineResult(
            input_type=understanding.input_type,
            understanding=understanding,
            features=features,
            queries=queries,
            evidence=evidence,
            formulation=formulation,
            elapsed_ms=round(elapsed_ms, 2),
        )

    def _deduplicate(self, chunks: list) -> list:
        seen = set()
        unique = []
        for c in chunks:
            key = (c.chunk_id, c.source, c.page)
            if key not in seen:
                seen.add(key)
                unique.append(c)
        return unique
