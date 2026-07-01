from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clinical.models import (
    CaseUnderstanding,
    ClinicalFeatures,
    ClinicalInput,
    ClinicalInputType,
    ClinicalResponse,
    EvidenceSynthesis,
    PipelineError,
    PipelineStage,
    RetrievalQuery,
)
from clinical.pipeline import ClinicalPipeline


# ═══════════════════════════════════════════════════════════════
# Model unit tests
# ═══════════════════════════════════════════════════════════════

class TestClinicalInput:
    def test_valid_input(self):
        inp = ClinicalInput(raw_text="Patient reports feeling sad.")
        assert len(inp.raw_text) > 0

    def test_too_short_raises(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ClinicalInput(raw_text="ab")

    def test_too_long_raises(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ClinicalInput(raw_text="x" * 50001)

    def test_input_type_override(self):
        inp = ClinicalInput(
            raw_text="Some text",
            input_type=ClinicalInputType.CASE_STUDY,
        )
        assert inp.input_type == ClinicalInputType.CASE_STUDY


class TestCaseUnderstanding:
    def test_minimal(self):
        c = CaseUnderstanding(
            input_type=ClinicalInputType.SINGLE_STATEMENT,
            summary="Test",
            key_topics=["anxiety"],
        )
        assert c.input_type == ClinicalInputType.SINGLE_STATEMENT
        assert c.summary == "Test"

    def test_defaults(self):
        c = CaseUnderstanding(
            input_type=ClinicalInputType.SYMPTOM_LIST,
            summary="",
        )
        assert c.key_topics == []
        assert c.clinical_context is None


class TestClinicalFeatures:
    def test_all_empty_by_default(self):
        f = ClinicalFeatures()
        assert f.symptoms == []
        assert f.diagnoses == []
        assert f.patient_history == []

    def test_with_values(self):
        f = ClinicalFeatures(
            symptoms=["sadness", "fatigue"],
            diagnoses=["MDD"],
        )
        assert "sadness" in f.symptoms
        assert "MDD" in f.diagnoses


class TestRetrievalQuery:
    def test_valid(self):
        q = RetrievalQuery(
            query="DSM-5 criteria for MDD",
            weight=1.5,
            rationale="Core diagnostic question",
        )
        assert q.weight == 1.5

    def test_weight_validation(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            RetrievalQuery(query="test", weight=0.0, rationale="x")
        with pytest.raises(ValidationError):
            RetrievalQuery(query="test", weight=3.5, rationale="x")


class TestEvidenceSynthesis:
    def test_valid(self):
        e = EvidenceSynthesis(
            key_findings=["MDD diagnosis supported by evidence"],
            common_themes=["Depression"],
            areas_of_agreement=["DSM-5 criteria met"],
            areas_of_uncertainty=[],
            practical_implications=["Consider CBT"],
            evidence_summary="Evidence supports MDD diagnosis.",
        )
        assert e.evidence_summary
        assert e.key_findings == ["MDD diagnosis supported by evidence"]

    def test_all_fields_default_to_empty_list(self):
        e = EvidenceSynthesis(
            key_findings=[],
            common_themes=[],
            areas_of_agreement=[],
            areas_of_uncertainty=[],
            practical_implications=[],
            evidence_summary="No evidence available.",
        )
        assert e.key_findings == []
        assert e.evidence_summary == "No evidence available."


class TestClinicalResponse:
    def test_minimal(self):
        r = ClinicalResponse(
            analysis="Patient presents with...",
            evidence_summary="Based on 3 sources",
            confidence=0.7,
        )
        assert r.formulation is None
        assert r.recommendations == []

    def test_full(self):
        r = ClinicalResponse(
            analysis="Analysis text",
            formulation="Biopsychosocial formulation",
            recommendations=["CBT", "Monitor mood"],
            evidence_summary="Evidence from DSM-5",
            confidence=0.85,
            limitations=["No family history available"],
        )
        assert len(r.recommendations) == 2


# ═══════════════════════════════════════════════════════════════
# Pipeline orchestrator
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def mock_retriever():
    r = MagicMock()

    async def aretrieve(query, n_results=3, **kw):
        from rag.retriever import RetrievedChunk
        return MagicMock(
            chunks=[
                RetrievedChunk(
                    text=f"Evidence for: {query}",
                    source="DSM5.pdf",
                    page=10,
                    score=0.92,
                    chunk_id=f"query_{i}",
                    rank=i + 1,
                    metadata={"source": "DSM5.pdf", "page": 10},
                )
                for i in range(2)
            ],
            found=True,
            top=MagicMock(text=f"Evidence for: {query}"),
        )

    r.aretrieve = aretrieve
    return r


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.generate = AsyncMock()
    return llm


class TestClinicalPipeline:
    @pytest.mark.asyncio
    async def test_full_pipeline_happy_path(self, mock_retriever, mock_llm):
        mock_llm.generate.side_effect = [
            # Input processor
            '{"input_type": "case_study", "summary": "A case of depression", '
            '"key_topics": ["depression", "CBT"], "clinical_context": "outpatient"}',
            # Feature extractor
            '{"symptoms": ["sadness", "anhedonia"], "diagnoses": ["MDD"], '
            '"patient_history": ["previous episode"], "family_history": [], '
            '"risk_factors": ["social isolation"], "protective_factors": [], '
            '"treatment_history": [], "other_relevant": []}',
            # Query generator
            '{"queries": [{"query": "DSM-5 MDD criteria", "weight": 2.0, '
            '"rationale": "Confirm diagnosis"}, '
            '{"query": "CBT for depression efficacy", "weight": 1.5, '
            '"rationale": "Treatment planning"}]}',
            # Evidence synthesizer
            '{"synthesis": "Evidence supports MDD", "supporting_evidence": '
            '["Criterion A", "Criterion B"], "contradicting_evidence": [], '
            '"confidence": 0.85}',
            # Response generator
            '{"analysis": "Patient meets MDD criteria", "formulation": null, '
            '"recommendations": ["CBT"], "evidence_summary": "DSM-5 criteria met", '
            '"confidence": 0.8, "limitations": ["Telehealth assessment"]}',
        ]

        pipeline = ClinicalPipeline(
            retriever=mock_retriever,
            llm=mock_llm,
        )
        result = await pipeline.run(
            ClinicalInput(raw_text="Patient has been feeling sad for 2 weeks.")
        )

        assert result.input_type == ClinicalInputType.CASE_STUDY
        assert len(result.features.symptoms) == 2
        assert len(result.queries) == 2
        assert "MDD" in result.formulation.case_summary
        assert result.elapsed_ms > 0
        assert mock_llm.generate.call_count == 5

    @pytest.mark.asyncio
    async def test_pipeline_with_single_statement(self, mock_retriever, mock_llm):
        mock_llm.generate.side_effect = [
            '{"input_type": "single_statement", "summary": "Brief statement", '
            '"key_topics": ["anxiety"], "clinical_context": null}',
            '{"symptoms": ["nervousness"], "diagnoses": [], '
            '"patient_history": [], "family_history": [], '
            '"risk_factors": [], "protective_factors": [], '
            '"treatment_history": [], "other_relevant": []}',
            '{"queries": [{"query": "anxiety treatment guidelines", '
            '"weight": 1.0, "rationale": "General guidance"}]}',
            '{"synthesis": "General anxiety evidence", "supporting_evidence": '
            '["Guidelines"], "contradicting_evidence": [], "confidence": 0.6}',
            '{"analysis": "Consider GAD screening", "formulation": null, '
            '"recommendations": ["PHQ-9", "GAD-7"], "evidence_summary": "Brief", '
            '"confidence": 0.5, "limitations": ["Limited info"]}',
        ]

        pipeline = ClinicalPipeline(
            retriever=mock_retriever,
            llm=mock_llm,
        )
        result = await pipeline.run(
            ClinicalInput(raw_text="I feel nervous all the time.")
        )

        assert result.input_type == ClinicalInputType.SINGLE_STATEMENT
        assert result.queries[0].weight == 1.0

    @pytest.mark.asyncio
    async def test_deduplication(self, mock_retriever, mock_llm):
        mock_llm.generate.side_effect = [
            '{"input_type": "case_study", "summary": "Test", '
            '"key_topics": ["test"], "clinical_context": null}',
            '{"symptoms": ["pain"], "diagnoses": [], '
            '"patient_history": [], "family_history": [], '
            '"risk_factors": [], "protective_factors": [], '
            '"treatment_history": [], "other_relevant": []}',
            '{"queries": [{"query": "pain management", "weight": 1.0, '
            '"rationale": "Main query"}, '
            '{"query": "pain guidelines", "weight": 0.5, '
            '"rationale": "Secondary"}]}',
            '{"synthesis": "Pain evidence", "supporting_evidence": ["E1"], '
            '"contradicting_evidence": [], "confidence": 0.7}',
            '{"analysis": "Pain assessment needed", "formulation": null, '
            '"recommendations": ["Assessment"], "evidence_summary": "E1", '
            '"confidence": 0.6, "limitations": []}',
        ]

        pipeline = ClinicalPipeline(
            retriever=mock_retriever,
            llm=mock_llm,
        )
        result = await pipeline.run(
            ClinicalInput(raw_text="Patient reports chronic pain.")
        )

        assert "Pain" in result.evidence.evidence_summary
        assert result.formulation.case_summary

    @pytest.mark.asyncio
    async def test_retrieval_failure_does_not_crash_pipeline(self, mock_llm):
        """When retrieval fails, the pipeline returns a degraded response."""
        mock_llm.generate.side_effect = [
            '{"input_type": "single_statement", "summary": "Test", '
            '"key_topics": [], "clinical_context": null}',
            '{"symptoms": [], "diagnoses": [], '
            '"patient_history": [], "family_history": [], '
            '"risk_factors": [], "protective_factors": [], '
            '"treatment_history": [], "other_relevant": []}',
            '{"queries": [{"query": "test query", "weight": 1.0, '
            '"rationale": "Fallback"}]}',
            '{"analysis": "No analysis possible", "formulation": null, '
            '"recommendations": [], "evidence_summary": "None", '
            '"confidence": 0.1, "limitations": ["No evidence"]}',
        ]

        failing_retriever = MagicMock()
        async def fail(*a, **kw):
            raise RuntimeError("ChromaDB unavailable")
        failing_retriever.aretrieve = fail

        pipeline = ClinicalPipeline(
            retriever=failing_retriever,
            llm=mock_llm,
        )
        result = await pipeline.run(
            ClinicalInput(raw_text="Test input")
        )

        assert result.formulation is not None
        assert "No evidence" in result.evidence.evidence_summary

    @pytest.mark.asyncio
    async def test_input_type_override_bypassed_llm(self, mock_retriever, mock_llm):
        mock_llm.generate.side_effect = [
            '{"symptoms": [], "diagnoses": [], '
            '"patient_history": [], "family_history": [], '
            '"risk_factors": [], "protective_factors": [], '
            '"treatment_history": [], "other_relevant": []}',
            '{"queries": [{"query": "depression", "weight": 1.0, '
            '"rationale": "Test"}]}',
            '{"synthesis": "None", "supporting_evidence": [], '
            '"contradicting_evidence": [], "confidence": 0.0}',
            '{"analysis": "Test", "formulation": null, '
            '"recommendations": [], "evidence_summary": "None", '
            '"confidence": 0.0, "limitations": []}',
        ]

        pipeline = ClinicalPipeline(
            retriever=mock_retriever,
            llm=mock_llm,
        )
        result = await pipeline.run(
            ClinicalInput(
                raw_text="Some text",
                input_type=ClinicalInputType.SYMPTOM_LIST,
            )
        )

        assert result.input_type == ClinicalInputType.SYMPTOM_LIST
        assert mock_llm.generate.call_count == 4  # input_processor skipped

    @pytest.mark.asyncio
    async def test_input_processor_parsing_error_raises(self, mock_retriever):
        llm = MagicMock()
        llm.generate = AsyncMock(return_value="not valid json")

        pipeline = ClinicalPipeline(
            retriever=mock_retriever,
            llm=llm,
        )
        with pytest.raises(PipelineError) as exc_info:
            await pipeline.run(
                ClinicalInput(raw_text="Some text")
            )
        assert exc_info.value.stage == PipelineStage.INPUT_PROCESSING


# ═══════════════════════════════════════════════════════════════
# API endpoint integration
# ═══════════════════════════════════════════════════════════════

class TestClinicalAPI:
    def _mock_pipeline(self):
        from clinical.models import (
            CaseUnderstanding, ClinicalFeatures, ClinicalFormulation,
            ClinicalInputType, EvidenceSynthesis, Formulation,
            PipelineResult, RetrievalQuery,
        )
        return PipelineResult(
            input_type=ClinicalInputType.SINGLE_STATEMENT,
            understanding=CaseUnderstanding(
                input_type=ClinicalInputType.SINGLE_STATEMENT,
                summary="Test summary",
                key_topics=["anxiety"],
                clinical_context=None,
            ),
            features=ClinicalFeatures(symptoms=["anxiety"]),
            queries=[RetrievalQuery(
                query="test",
                weight=1.0,
                rationale="test",
            )],
            evidence=EvidenceSynthesis(
                key_findings=[],
                common_themes=[],
                areas_of_agreement=[],
                areas_of_uncertainty=[],
                practical_implications=[],
                evidence_summary="Test evidence",
            ),
            formulation=ClinicalFormulation(
                case_summary="Test case summary with enough characters.",
                possible_formulations=[
                    Formulation(
                        explanation="Test explanation with enough characters to pass validation.",
                        supporting_symptoms=["anxiety"],
                        confidence_level="Moderate",
                    ),
                ],
                supporting_evidence=[],
                alternative_explanations=[],
                missing_assessment_information=[],
            ),
            elapsed_ms=10.0,
        )

    def test_clinical_analyze_endpoint_registered(self, client):
        test_client, app = client
        mock_pipeline = MagicMock()
        mock_pipeline.run = AsyncMock(return_value=self._mock_pipeline())
        from api.dependencies import get_clinical_pipeline_dep
        app.dependency_overrides[get_clinical_pipeline_dep] = lambda: mock_pipeline
        resp = test_client.post(
            "/api/v1/clinical/analyze",
            json={"text": "Patient feels anxious."},
        )
        app.dependency_overrides.clear()
        assert resp.status_code == 200
        data = resp.json()
        assert data["input_type"] == "single_statement"
        assert "case_summary" in data["formulation"]

    def test_422_empty_text(self, client):
        test_client, app = client
        resp = test_client.post(
            "/api/v1/clinical/analyze",
            json={"text": ""},
        )
        assert resp.status_code == 422

    def test_422_text_too_short(self, client):
        test_client, app = client
        resp = test_client.post(
            "/api/v1/clinical/analyze",
            json={"text": "ab"},
        )
        assert resp.status_code == 422

    def test_with_valid_input_type_override(self, client):
        test_client, app = client
        mock_pipeline = MagicMock()
        mock_pipeline.run = AsyncMock(return_value=self._mock_pipeline())
        from api.dependencies import get_clinical_pipeline_dep
        app.dependency_overrides[get_clinical_pipeline_dep] = lambda: mock_pipeline
        resp = test_client.post(
            "/api/v1/clinical/analyze",
            json={"text": "Patient reports sadness, anhedonia, fatigue.",
                  "input_type": "symptom_list"},
        )
        app.dependency_overrides.clear()
        assert resp.status_code == 200


@pytest.fixture(scope="module")
def client(mock_settings):
    with patch("config.settings.get_settings", return_value=mock_settings), \
         patch("api.dependencies._cached_settings", return_value=mock_settings), \
         patch("api.dependencies.get_retriever_dep") as mock_retriever_dep, \
         patch("app_logging.logger.setup_logging"), \
         patch("rag.embeddings.load_embedding_model"), \
         patch("rag.vector_store.get_vector_store") as mock_vs_factory:

        mock_vs = MagicMock()
        mock_vs.create_collection.return_value = MagicMock(
            name="test_collection", document_count=42
        )
        mock_vs.get_collection_info.return_value = MagicMock(
            name="test_collection", document_count=42
        )
        mock_vs_factory.return_value = mock_vs

        mock_retriever_dep.return_value = MagicMock()

        from api.main import create_app
        app = create_app(settings=mock_settings)

        from fastapi.testclient import TestClient
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c, app


@pytest.fixture(scope="module")
def mock_settings():
    cfg = MagicMock()
    cfg.app.name = "CPA Test"
    cfg.app.version = "0.1.0"
    cfg.app.env = "development"
    cfg.app.allowed_origins = ["http://localhost:3000"]
    cfg.app.debug = True
    cfg.llm.ollama_model = "llama3.1:8b"
    cfg.llm.ollama_base_url = "http://localhost:11434"
    cfg.llm.temperature = 0.1
    cfg.llm.top_k = 40
    cfg.llm.top_p = 0.9
    cfg.llm.max_tokens = 2048
    cfg.chroma.collection_name = "test_collection"
    cfg.chroma.persist_dir = MagicMock()
    cfg.chroma.persist_dir.__truediv__ = lambda s, x: MagicMock()
    cfg.embedding.model_name = "BAAI/bge-large-en-v1.5"
    cfg.embedding.batch_size = 32
    cfg.rag.chunk_size = 800
    cfg.rag.chunk_overlap = 150
    cfg.rag.top_k = 5
    cfg.rag.similarity_threshold = 0.35
    cfg.logging.level = "INFO"
    cfg.logging.format = "console"
    cfg.logging.file = MagicMock()
    cfg.logging.rotation = "10 MB"
    cfg.logging.retention = "30 days"
    cfg.is_production = MagicMock(return_value=False)
    cfg.server.host = "0.0.0.0"
    cfg.server.port = 8000
    cfg.server.workers = 1
    cfg.server.reload = True
    return cfg
