from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clinical.graph.state import GraphState, _merge_errors


class TestGraphState:
    def test_initial_state(self):
        state: GraphState = {
            "text": "Client presents with low mood.",
            "session_id": None,
            "errors": {},
        }
        assert state["text"] == "Client presents with low mood."
        assert state["errors"] == {}

    def test_merge_errors_accumulates(self):
        a = {"err1": "first"}
        b = {"err2": "second"}
        merged = _merge_errors(a, b)
        assert merged == {"err1": "first", "err2": "second"}

    def test_merge_errors_later_overwrites(self):
        a = {"err1": "first"}
        b = {"err1": "replacement"}
        merged = _merge_errors(a, b)
        assert merged == {"err1": "replacement"}

    def test_merge_errors_empty_base(self):
        merged = _merge_errors({}, {"err1": "val"})
        assert merged == {"err1": "val"}

    def test_merge_errors_empty_update(self):
        merged = _merge_errors({"err1": "val"}, {})
        assert merged == {"err1": "val"}


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

_MOCK_UNDERSTANDING = MagicMock()
_MOCK_UNDERSTANDING.to_flat_dict.return_value = {"age": 34, "gender": "female"}
_MOCK_UNDERSTANDING.model_dump_json.return_value = '{"age": 34}'
_MOCK_UNDERSTANDING.clinical_presentation.presenting_concerns = []
_MOCK_UNDERSTANDING.clinical_presentation.symptoms = []

_MOCK_QUERIES_RESULT = MagicMock()
_MOCK_QUERIES_RESULT.to_query_strings.return_value = ["depression CBT", "anxiety treatment"]

_MOCK_RETRIEVED_CHUNKS = [
    MagicMock(chunk_id="c1", source="doc1.pdf", page=1, score=0.9, text="Evidence text 1"),
    MagicMock(chunk_id="c2", source="doc2.pdf", page=2, score=0.8, text="Evidence text 2"),
]

_MOCK_EVIDENCE = MagicMock()
_MOCK_EVIDENCE.overall_summary = "Evidence summary text"
_MOCK_EVIDENCE.key_findings = [MagicMock(description="Finding 1")]
_MOCK_EVIDENCE.common_themes = [MagicMock(description="Theme 1")]

_MOCK_FORMULATION = MagicMock()
_MOCK_FORMULATION.case_summary = "Case summary text"
_MOCK_FORMULATION.confidence = 0.7
_MOCK_FORMULATION.alternative_explanations = ["GAD", "Adjustment"]
_MOCK_FORMULATION.caution = "Risk assessment needed"
_MOCK_FORMULATION.supporting_evidence = ["Evidence 1"]
_MOCK_FORMULATION.missing_assessment_information = ["Trauma history"]
_MOCK_FORMULATION.possible_formulations = [MagicMock(explanation="Formulation 1")]
_MOCK_FORMULATION.model_dump_json.return_value = '{"case_summary": "test"}'

_MOCK_MISSING_INFO = MagicMock()
_MOCK_MISSING_INFO.missing_information = [
    MagicMock(info_gap="Trauma history", clinical_relevance="Important for formulation"),
]
_MOCK_MISSING_INFO.overall_assessment = "Moderate gaps"

_MOCK_PLAN = MagicMock()
_MOCK_PLAN.therapeutic_focus = [MagicMock(area="Cognitive patterns")]
_MOCK_PLAN.intervention_directions = [MagicMock(modality="CBT", description="Cognitive restructuring")]
_MOCK_PLAN.cbt_strategies = [MagicMock(technique="Thought record", rationale="Identify thoughts")]
_MOCK_PLAN.act_strategies = [MagicMock(technique="Defusion", rationale="Reduce fusion")]
_MOCK_PLAN.dbt_strategies = [MagicMock(technique="Mindfulness", rationale="Stay present")]
_MOCK_PLAN.references = ["Beck (1976)"]

_MOCK_RESPONSE = MagicMock()
_MOCK_RESPONSE.markdown = "## 1. CASE SUMMARY\n\nFull report..."
_MOCK_RESPONSE.sections_generated = 11
_MOCK_RESPONSE.confidence = 0.7

_BASE_STATE: GraphState = {
    "text": "34-year-old female with low mood.",
    "session_id": None,
    "understanding": _MOCK_UNDERSTANDING,
    "queries_result": _MOCK_QUERIES_RESULT,
    "retrieved_chunks": _MOCK_RETRIEVED_CHUNKS,
    "evidence": _MOCK_EVIDENCE,
    "formulation": _MOCK_FORMULATION,
    "missing_info": _MOCK_MISSING_INFO,
    "plan": _MOCK_PLAN,
    "response": _MOCK_RESPONSE,
    "errors": {},
}


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.generate = AsyncMock()
    return llm


# ═══════════════════════════════════════════════════════════════
# Node tests
# ═══════════════════════════════════════════════════════════════

class TestNodeCaseUnderstanding:
    @pytest.mark.asyncio
    async def test_extracts_understanding(self):
        from clinical.graph.nodes import node_case_understanding

        mock_extractor = MagicMock()
        mock_extractor.extract = AsyncMock(return_value=_MOCK_UNDERSTANDING)

        with patch("clinical.graph.nodes._get_case_understanding", return_value=mock_extractor):
            state: GraphState = {"text": "34-year-old female with low mood.", "errors": {}}
            result = await node_case_understanding(state)
            assert "understanding" in result
            assert result["understanding"] is _MOCK_UNDERSTANDING

    @pytest.mark.asyncio
    async def test_empty_text_returns_error(self):
        from clinical.graph.nodes import node_case_understanding

        state: GraphState = {"text": "", "errors": {}}
        result = await node_case_understanding(state)
        assert "errors" in result
        assert "No input text" in result["errors"]["case_understanding"]

    @pytest.mark.asyncio
    async def test_extractor_failure_returns_error(self):
        from clinical.graph.nodes import node_case_understanding

        mock_extractor = MagicMock()
        mock_extractor.extract = AsyncMock(side_effect=RuntimeError("LLM unavailable"))

        with patch("clinical.graph.nodes._get_case_understanding", return_value=mock_extractor):
            state: GraphState = {"text": "34-year-old female.", "errors": {}}
            result = await node_case_understanding(state)
            assert "errors" in result
            assert "LLM unavailable" in result["errors"]["case_understanding"]


class TestNodeQueryGeneration:
    @pytest.mark.asyncio
    async def test_generates_queries(self):
        from clinical.graph.nodes import node_query_generation

        mock_gen = MagicMock()
        mock_gen.generate = AsyncMock(return_value=_MOCK_QUERIES_RESULT)

        with patch("clinical.graph.nodes._get_query_generator", return_value=mock_gen):
            state = dict(_BASE_STATE)
            state["errors"] = {}
            result = await node_query_generation(state)
            assert "queries_result" in result

    @pytest.mark.asyncio
    async def test_no_understanding_returns_error(self):
        from clinical.graph.nodes import node_query_generation

        state: GraphState = {"text": "test", "errors": {}}
        result = await node_query_generation(state)
        assert "errors" in result


class TestNodeRetrieval:
    @pytest.mark.asyncio
    async def test_retrieves_chunks(self):
        from clinical.graph.nodes import node_retrieval

        mock_retriever = MagicMock()
        mock_retriever.aretrieve = AsyncMock()
        mock_retriever.aretrieve.return_value.chunks = _MOCK_RETRIEVED_CHUNKS

        with patch("clinical.graph.nodes._get_retriever", return_value=mock_retriever):
            state = dict(_BASE_STATE)
            state["errors"] = {}
            result = await node_retrieval(state)
            assert "retrieved_chunks" in result
            assert len(result["retrieved_chunks"]) == 2

    @pytest.mark.asyncio
    async def test_no_queries_returns_error(self):
        from clinical.graph.nodes import node_retrieval

        state: GraphState = {"text": "test", "errors": {}}
        result = await node_retrieval(state)
        assert "errors" in result

    @pytest.mark.asyncio
    async def test_handles_retrieval_failure(self):
        from clinical.graph.nodes import node_retrieval

        mock_retriever = MagicMock()
        mock_retriever.aretrieve = AsyncMock(side_effect=RuntimeError("DB down"))

        with patch("clinical.graph.nodes._get_retriever", return_value=mock_retriever):
            state = dict(_BASE_STATE)
            state["errors"] = {}
            result = await node_retrieval(state)
            assert "retrieved_chunks" in result
            assert len(result["retrieved_chunks"]) == 0  # graceful degradation


class TestNodeEvidenceSynthesis:
    @pytest.mark.asyncio
    async def test_synthesises_evidence(self):
        from clinical.graph.nodes import node_evidence_synthesis

        mock_synth = MagicMock()
        mock_synth.synthesise = AsyncMock(return_value=_MOCK_EVIDENCE)

        with patch("clinical.graph.nodes._get_evidence_synthesizer", return_value=mock_synth):
            state = dict(_BASE_STATE)
            state["errors"] = {}
            result = await node_evidence_synthesis(state)
            assert "evidence" in result
            assert result["evidence"] is _MOCK_EVIDENCE

    @pytest.mark.asyncio
    async def test_no_chunks_returns_error(self):
        from clinical.graph.nodes import node_evidence_synthesis

        state: GraphState = {"text": "test", "errors": {}}
        result = await node_evidence_synthesis(state)
        assert "errors" in result


class TestNodeClinicalFormulation:
    @pytest.mark.asyncio
    async def test_formulates(self):
        from clinical.graph.nodes import node_clinical_formulation

        mock_form = MagicMock()
        mock_form.formulate = AsyncMock(return_value=_MOCK_FORMULATION)

        with patch("clinical.graph.nodes._get_formulator", return_value=mock_form):
            state = dict(_BASE_STATE)
            state["errors"] = {}
            result = await node_clinical_formulation(state)
            assert "formulation" in result

    @pytest.mark.asyncio
    async def test_no_understanding_returns_error(self):
        from clinical.graph.nodes import node_clinical_formulation

        state: GraphState = {"text": "test", "errors": {}}
        result = await node_clinical_formulation(state)
        assert "errors" in result


class TestNodeMissingInfo:
    @pytest.mark.asyncio
    async def test_detects_missing_info(self):
        from clinical.graph.nodes import node_missing_info

        mock_det = MagicMock()
        mock_det.detect = AsyncMock(return_value=_MOCK_MISSING_INFO)

        with patch("clinical.graph.nodes._get_missing_info_detector", return_value=mock_det):
            state: GraphState = {"text": "Client presents with anxiety.", "errors": {}}
            result = await node_missing_info(state)
            assert "missing_info" in result

    @pytest.mark.asyncio
    async def test_empty_text_returns_error(self):
        from clinical.graph.nodes import node_missing_info

        state: GraphState = {"text": "", "errors": {}}
        result = await node_missing_info(state)
        assert "errors" in result


class TestNodeTherapeuticPlanning:
    @pytest.mark.asyncio
    async def test_plans(self):
        from clinical.graph.nodes import node_therapeutic_planning

        mock_planner = MagicMock()
        mock_planner.plan = AsyncMock(return_value=_MOCK_PLAN)

        with patch("clinical.graph.nodes._get_therapeutic_planner", return_value=mock_planner):
            state = dict(_BASE_STATE)
            state["errors"] = {}
            result = await node_therapeutic_planning(state)
            assert "plan" in result

    @pytest.mark.asyncio
    async def test_no_formulation_returns_error(self):
        from clinical.graph.nodes import node_therapeutic_planning

        state: GraphState = {"text": "test", "errors": {}}
        result = await node_therapeutic_planning(state)
        assert "errors" in result


class TestNodeResponseGeneration:
    @pytest.mark.asyncio
    async def test_generates_response(self):
        from clinical.graph.nodes import node_response_generation

        mock_resp = MagicMock()
        mock_resp.generate = AsyncMock(return_value=_MOCK_RESPONSE)

        with patch("clinical.graph.nodes._get_response_generator", return_value=mock_resp):
            state = dict(_BASE_STATE)
            state["errors"] = {}
            result = await node_response_generation(state)
            assert "response" in result

    @pytest.mark.asyncio
    async def test_no_understanding_returns_error(self):
        from clinical.graph.nodes import node_response_generation

        state: GraphState = {"text": "test", "errors": {}}
        result = await node_response_generation(state)
        assert "errors" in result


# ═══════════════════════════════════════════════════════════════
# Graph builder tests
# ═══════════════════════════════════════════════════════════════

class TestBuildGraph:
    def test_build_graph_returns_state_graph(self):
        from clinical.graph.graph import build_graph

        builder = build_graph()
        assert builder is not None

    def test_get_graph_returns_singleton(self):
        from clinical.graph.graph import get_graph

        g1 = get_graph()
        g2 = get_graph()
        assert g1 is g2

    def test_compiled_graph_runs(self):
        from clinical.graph.graph import build_graph

        builder = build_graph()
        compiled = builder.compile()
        assert compiled is not None

    @pytest.mark.asyncio
    async def test_run_pipeline_happy_path(self):
        mock_extractor = MagicMock()
        mock_extractor.extract = AsyncMock(return_value=_MOCK_UNDERSTANDING)
        mock_gen = MagicMock()
        mock_gen.generate = AsyncMock(return_value=_MOCK_QUERIES_RESULT)
        mock_ret = MagicMock()
        mock_ret.aretrieve = AsyncMock(return_value=MagicMock(chunks=_MOCK_RETRIEVED_CHUNKS))
        mock_synth = MagicMock()
        mock_synth.synthesise = AsyncMock(return_value=_MOCK_EVIDENCE)
        mock_form = MagicMock()
        mock_form.formulate = AsyncMock(return_value=_MOCK_FORMULATION)
        mock_det = MagicMock()
        mock_det.detect = AsyncMock(return_value=_MOCK_MISSING_INFO)
        mock_planner = MagicMock()
        mock_planner.plan = AsyncMock(return_value=_MOCK_PLAN)
        mock_resp = MagicMock()
        mock_resp.generate = AsyncMock(return_value=_MOCK_RESPONSE)

        with (
            patch("clinical.graph.nodes._get_case_understanding", return_value=mock_extractor),
            patch("clinical.graph.nodes._get_query_generator", return_value=mock_gen),
            patch("clinical.graph.nodes._get_retriever", return_value=mock_ret),
            patch("clinical.graph.nodes._get_evidence_synthesizer", return_value=mock_synth),
            patch("clinical.graph.nodes._get_formulator", return_value=mock_form),
            patch("clinical.graph.nodes._get_missing_info_detector", return_value=mock_det),
            patch("clinical.graph.nodes._get_therapeutic_planner", return_value=mock_planner),
            patch("clinical.graph.nodes._get_response_generator", return_value=mock_resp),
        ):
            from clinical.graph.graph import run_pipeline

            result = await run_pipeline("34-year-old female with low mood.")
            assert "response" in result
            assert result["response"] is _MOCK_RESPONSE
            assert "understanding" in result
            assert "evidence" in result
            assert "formulation" in result
            assert "plan" in result


class TestDeduplicate:
    def test_deduplicates_by_chunk_id_source_page(self):
        from clinical.graph.nodes import _sort_and_deduplicate

        chunks = [
            MagicMock(chunk_id="c1", source="a.pdf", page=1, score=0.9),
            MagicMock(chunk_id="c1", source="a.pdf", page=1, score=0.9),
            MagicMock(chunk_id="c2", source="a.pdf", page=1, score=0.8),
        ]
        result = _sort_and_deduplicate(chunks)
        assert len(result) == 2

    def test_sorts_by_score_descending(self):
        from clinical.graph.nodes import _sort_and_deduplicate

        chunks = [
            MagicMock(chunk_id="c2", source="a.pdf", page=1, score=0.5),
            MagicMock(chunk_id="c1", source="a.pdf", page=1, score=0.9),
        ]
        result = _sort_and_deduplicate(chunks)
        assert result[0].score == 0.9
        assert result[1].score == 0.5

    def test_empty_list(self):
        from clinical.graph.nodes import _sort_and_deduplicate

        result = _sort_and_deduplicate([])
        assert result == []
