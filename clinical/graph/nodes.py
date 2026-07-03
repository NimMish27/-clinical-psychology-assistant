from __future__ import annotations

import time
import traceback
from typing import Any

from clinical.case_understanding import CaseUnderstandingExtractor
from clinical.evidence_synthesis import EvidenceSynthesizer
from clinical.formulation import ClinicalFormulator
from clinical.graph.state import GraphState
from clinical.llm import LLMService, get_llm_service
from clinical.missing_info import MissingInfoDetector
from clinical.query_generation import RetrievalQueryGenerator
from clinical.response_generation import ResponseGenerator
from clinical.therapeutic_planning import TherapeuticPlanner
from rag.retriever import Retriever, RetrievedChunk, get_retriever
from app_logging.logger import get_logger

_log = get_logger(__name__)


# ── Helpers ──────────────────────────────────────────────────

def _sort_and_deduplicate(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Remove duplicates by (chunk_id, source, page) and sort by score descending."""
    seen: set[tuple[str, str, int]] = set()
    unique: list[RetrievedChunk] = []
    for c in chunks:
        key = (c.chunk_id, c.source, c.page)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    unique.sort(key=lambda x: -x.score)
    return unique


# ── Node factories (lazy singletons) ─────────────────────────

_ex: CaseUnderstandingExtractor | None = None  # noqa: N816
_qg: RetrievalQueryGenerator | None = None  # noqa: N816
_rt: Retriever | None = None  # noqa: N816
_es: EvidenceSynthesizer | None = None  # noqa: N816
_cf: ClinicalFormulator | None = None  # noqa: N816
_mi: MissingInfoDetector | None = None  # noqa: N816
_tp: TherapeuticPlanner | None = None  # noqa: N816
_rg: ResponseGenerator | None = None  # noqa: N816


def _get_case_understanding(llm: Any) -> CaseUnderstandingExtractor:
    global _ex
    if _ex is None:
        _ex = CaseUnderstandingExtractor(llm=llm)
    return _ex


def _get_query_generator(llm: Any) -> RetrievalQueryGenerator:
    global _qg
    if _qg is None:
        _qg = RetrievalQueryGenerator(llm=llm)
    return _qg


def _get_retriever() -> Retriever:
    global _rt
    if _rt is None:
        _rt = get_retriever()
    return _rt


def _get_evidence_synthesizer(llm: Any) -> EvidenceSynthesizer:
    global _es
    if _es is None:
        _es = EvidenceSynthesizer(llm=llm)
    return _es


def _get_formulator(llm: Any) -> ClinicalFormulator:
    global _cf
    if _cf is None:
        _cf = ClinicalFormulator(llm=llm)
    return _cf


def _get_missing_info_detector(llm: Any) -> MissingInfoDetector:
    global _mi
    if _mi is None:
        _mi = MissingInfoDetector(llm=llm)
    return _mi


def _get_therapeutic_planner(llm: Any) -> TherapeuticPlanner:
    global _tp
    if _tp is None:
        _tp = TherapeuticPlanner(llm=llm)
    return _tp


def _get_response_generator(llm: Any) -> ResponseGenerator:
    global _rg
    if _rg is None:
        _rg = ResponseGenerator(llm=llm)
    return _rg


# ── Node implementations ─────────────────────────────────────

async def node_case_understanding(state: GraphState) -> dict[str, Any]:
    """Extract structured case understanding from raw text."""
    text = state.get("text", "")
    if not text:
        return {"errors": {"case_understanding": "No input text provided"}}

    try:
        llm = get_llm_service()
        extractor = _get_case_understanding(llm)
        result = await extractor.extract(text)
        return {"understanding": result}
    except Exception as exc:
        _log.error("graph.case_understanding_failed", error=str(exc))
        return {"errors": {"case_understanding": f"{exc}"}}


async def node_query_generation(state: GraphState) -> dict[str, Any]:
    """Generate retrieval queries from case understanding."""
    understanding = state.get("understanding")
    if not understanding:
        return {"errors": {"query_generation": "No case understanding available"}}

    try:
        llm = get_llm_service()
        gen = _get_query_generator(llm)
        result = await gen.generate(understanding)
        return {"queries_result": result}
    except Exception as exc:
        _log.error("graph.query_generation_failed", error=str(exc))
        return {"errors": {"query_generation": f"{exc}"}}


async def node_retrieval(state: GraphState) -> dict[str, Any]:
    """Retrieve relevant chunks for each generated query."""
    queries_result = state.get("queries_result")
    if not queries_result:
        return {"errors": {"retrieval": "No queries available for retrieval"}}

    query_strings = queries_result.to_query_strings()
    retriever = _get_retriever()
    all_chunks: list[RetrievedChunk] = []

    for q in query_strings:
        try:
            result = await retriever.aretrieve(q, n_results=3)
            all_chunks.extend(result.chunks)
        except Exception as exc:
            _log.warning("graph.retrieval_query_failed", query=q, error=str(exc))

    all_chunks = _sort_and_deduplicate(all_chunks)
    return {"retrieved_chunks": all_chunks}


async def node_evidence_synthesis(state: GraphState) -> dict[str, Any]:
    """Synthesise evidence from retrieved chunks."""
    chunks = state.get("retrieved_chunks")
    if not chunks:
        return {"errors": {"evidence_synthesis": "No retrieved chunks available"}}

    try:
        understanding = state.get("understanding")
        case_text = understanding.model_dump_json() if understanding else None
        query_text = ""
        qr = state.get("queries_result")
        if qr:
            query_text = " ".join(qr.to_query_strings())

        llm = get_llm_service()
        synth = _get_evidence_synthesizer(llm)
        result = await synth.synthesise(
            chunks,
            query=query_text or None,
            case_context=case_text,
        )
        return {"evidence": result}
    except Exception as exc:
        _log.error("graph.evidence_synthesis_failed", error=str(exc))
        return {"errors": {"evidence_synthesis": f"{exc}"}}


async def node_clinical_formulation(state: GraphState) -> dict[str, Any]:
    """Generate clinical formulation from understanding and evidence."""
    understanding = state.get("understanding")
    if not understanding:
        return {"errors": {"clinical_formulation": "No case understanding available"}}

    try:
        case_data = understanding.to_flat_dict()
        evidence = state.get("evidence")
        evidence_str = evidence.overall_summary if evidence else None

        llm = get_llm_service()
        formulator = _get_formulator(llm)
        result = await formulator.formulate(
            case_data=case_data,
            evidence_synthesis=evidence_str,
        )
        return {"formulation": result}
    except Exception as exc:
        _log.error("graph.clinical_formulation_failed", error=str(exc))
        return {"errors": {"clinical_formulation": f"{exc}"}}


async def node_missing_info(state: GraphState) -> dict[str, Any]:
    """Identify missing clinical information."""
    text = state.get("text", "")
    if not text:
        return {"errors": {"missing_info": "No input text provided"}}

    try:
        understanding = state.get("understanding")
        context = understanding.model_dump_json() if understanding else None

        llm = get_llm_service()
        detector = _get_missing_info_detector(llm)
        result = await detector.detect(text=text, context=context)
        return {"missing_info": result}
    except Exception as exc:
        _log.error("graph.missing_info_failed", error=str(exc))
        return {"errors": {"missing_info": f"{exc}"}}


async def node_therapeutic_planning(state: GraphState) -> dict[str, Any]:
    """Generate therapeutic plan from formulation and evidence."""
    formulation = state.get("formulation")
    if not formulation:
        return {"errors": {"therapeutic_planning": "No formulation available"}}

    try:
        formulation_text = formulation.model_dump_json()
        evidence = state.get("evidence")
        evidence_text = evidence.overall_summary if evidence else None

        llm = get_llm_service()
        planner = _get_therapeutic_planner(llm)
        result = await planner.plan(
            formulation=formulation_text,
            evidence_summary=evidence_text,
            case_summary=formulation.case_summary,
            formulations_text=[f.explanation for f in formulation.possible_formulations],
            supporting_evidence=formulation.supporting_evidence,
            alternative_explanations=formulation.alternative_explanations,
            missing_information=formulation.missing_assessment_information,
            caution=formulation.caution,
            evidence_themes=[t.description for t in evidence.common_themes] if evidence else None,
        )
        return {"plan": result}
    except Exception as exc:
        _log.error("graph.therapeutic_planning_failed", error=str(exc))
        return {"errors": {"therapeutic_planning": f"{exc}"}}


async def node_response_generation(state: GraphState) -> dict[str, Any]:
    """Compose the final markdown clinical report."""
    understanding = state.get("understanding")
    if not understanding:
        return {"errors": {"response_generation": "No data available for response"}}

    try:
        formulation = state.get("formulation")
        evidence = state.get("evidence")
        plan = state.get("plan")
        missing_info = state.get("missing_info")

        llm = get_llm_service()
        generator = _get_response_generator(llm)

        # Prepare inputs from all upstream modules
        presenting = []
        symptoms = []
        if understanding:
            presenting = [
                str(c.value) for c in understanding.clinical_presentation.presenting_concerns
                if c.value is not None
            ]
            symptoms = [
                str(s.value) for s in understanding.clinical_presentation.symptoms
                if s.value is not None
            ]

        differentials = []
        formulation_text = None
        formulation_confidence = None
        caution_text = None
        if formulation:
            formulation_text = formulation.case_summary
            formulation_confidence = formulation.confidence
            differentials = formulation.alternative_explanations
            caution_text = formulation.caution

        evidence_summary = None
        evidence_findings: list[str] | None = None
        if evidence:
            evidence_summary = evidence.overall_summary
            evidence_findings = [f.description for f in evidence.key_findings]

        missing_text = None
        if missing_info and missing_info.missing_information:
            items = [
                f"**{item.info_gap}** — {item.clinical_relevance}"
                for item in missing_info.missing_information[:5]
            ]
            missing_text = "\n\n".join(items) + f"\n\n*Overall: {missing_info.overall_assessment}*"

        focus_areas: list[str] | None = None
        intervention_text: str | None = None
        cbt: list[str] | None = None
        act: list[str] | None = None
        dbt: list[str] | None = None
        refs: list[str] | None = None
        if plan:
            focus_areas = [f.area for f in plan.therapeutic_focus]
            intervention_text = "\n".join(
                f"- **{d.modality}**: {d.description}" for d in plan.intervention_directions
            )
            cbt = [f"{s.technique}: {s.rationale}" for s in plan.cbt_strategies]
            act = [f"{s.technique}: {s.rationale}" for s in plan.act_strategies]
            dbt = [f"{s.technique}: {s.rationale}" for s in plan.dbt_strategies]
            refs = plan.references if hasattr(plan, "references") else []

        result = await generator.generate(
            case_summary=formulation_text,
            presenting_concerns=presenting or None,
            observed_symptoms=symptoms or None,
            formulation_text=formulation_text,
            formulation_confidence=formulation_confidence,
            differential_considerations=differentials or None,
            missing_information=missing_text,
            evidence_summary=evidence_summary,
            evidence_findings=evidence_findings or None,
            therapeutic_focus=focus_areas or None,
            intervention_directions=intervention_text,
            cbt_strategies=cbt or None,
            act_strategies=act or None,
            dbt_strategies=dbt or None,
            references=refs or None,
            caution=caution_text,
        )
        return {"response": result}
    except Exception as exc:
        _log.error("graph.response_generation_failed", error=str(exc))
        return {"errors": {"response_generation": f"{exc}"}}
