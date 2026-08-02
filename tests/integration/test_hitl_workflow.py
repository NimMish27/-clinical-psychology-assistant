"""
tests/integration/test_hitl_workflow.py
───────────────────────────────────────
End-to-end API tests for the human-in-the-loop workflow:

  1. POST /api/v1/clinical/analyze with resumable=true pauses after
     framework comparison and persists the pending state.
  2. POST /api/v1/clinical/framework/select resumes the workflow using
     only the chosen framework.
  3. Error paths (unknown workflow, invalid framework).
  4. Backward compatibility of the legacy (non-resumable) endpoint.

The real LangGraph pipeline runs with its node factories mocked (no
Ollama / ChromaDB). The WorkflowStore is backed by a temporary SQLite
file.
"""

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_archive_store_dep, get_workflow_store_dep
from clinical.case_understanding.models import (
    CaseUnderstandingResult,
    ClinicalPresentation,
    ConfidenceRating,
    ContextualFactors,
    Duration,
    ExtractedField,
    OverallSeverity,
    Severity,
)
from clinical.evidence_synthesis.models import (
    Agreement,
    EvidenceSynthesisResult,
    Finding,
    Theme,
)
from clinical.formulation.models import ClinicalFormulationResult, Formulation
from clinical.frameworks.models import FrameworkResult
from clinical.missing_info.models import MissingInfoItem, MissingInfoResult
from clinical.query_generation.models import OptimizedQuery, QueryCategory, QueryGenerationResult
from clinical.response_generation.models import ClinicalResponseResult
from clinical.safety_validation.models import SafetyValidationResult
from clinical.session_planning.models import SessionObjective, SessionPlanResult
from clinical.therapeutic_planning.models import (
    ACTStrategy,
    CBTStrategy,
    DBTStrategy,
    InterventionDirection,
    TherapeuticFocus,
    TherapeuticPlanResult,
)
from clinical.workflow.archive import ArchiveStore
from clinical.workflow.store import WorkflowStore
from rag.retriever import RetrievedChunk

TEXT = "34-year-old female presenting with low mood and reduced motivation over six weeks."


# ═══════════════════════════════════════════════════════════════
# Fixtures & mock model builders
# ═══════════════════════════════════════════════════════════════

def _mock_settings(tmp_path):
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
    cfg.workflow.db_path = tmp_path / "workflows.db"
    cfg.workflow.ttl_seconds = 3600
    return cfg


@pytest.fixture
def client(tmp_path):
    mock_settings = _mock_settings(tmp_path)
    store = WorkflowStore(db_path=tmp_path / "workflows.db", ttl_seconds=3600)
    archive = ArchiveStore(db_path=tmp_path / "workflows_archive.db", retention_days=365)

    with (
        patch("config.settings.get_settings", return_value=mock_settings),
        patch("api.dependencies._cached_settings", return_value=mock_settings),
        patch("app_logging.logger.setup_logging"),
        patch("rag.embeddings.load_embedding_model"),
        patch("rag.vector_store.get_vector_store") as mock_vs_factory,
    ):
        mock_vs = MagicMock()
        mock_vs.create_collection.return_value = MagicMock(
            name="test_collection", document_count=42
        )
        mock_vs.get_collection_info.return_value = MagicMock(
            name="test_collection", document_count=42
        )
        mock_vs_factory.return_value = mock_vs

        from api.main import create_app

        app = create_app(settings=mock_settings)
        app.dependency_overrides[get_workflow_store_dep] = lambda: store
        app.dependency_overrides[get_archive_store_dep] = lambda: archive

        with TestClient(app, raise_server_exceptions=False) as c:
            yield c, store, archive


def _understanding() -> CaseUnderstandingResult:
    return CaseUnderstandingResult(
        clinical_presentation=ClinicalPresentation(
            presenting_concerns=[
                ExtractedField(value="low mood", confidence=ConfidenceRating.MEDIUM)
            ],
            symptoms=[
                ExtractedField(value="reduced motivation", confidence=ConfidenceRating.MEDIUM)
            ],
            duration=Duration(value=6, unit="weeks", original_text="six weeks"),
        ),
        contextual_factors=ContextualFactors(
            stressors=[
                ExtractedField(value="work pressure", confidence=ConfidenceRating.MEDIUM)
            ],
        ),
        overall_severity=OverallSeverity(
            severity=Severity.MODERATE,
            confidence=ConfidenceRating.MEDIUM,
            rationale="Low mood with functional impact",
        ),
        raw_text=TEXT,
    )


def _queries() -> QueryGenerationResult:
    return QueryGenerationResult(
        queries=[
            OptimizedQuery(
                query="low mood evidence-based treatment",
                category=QueryCategory.TREATMENT,
                weight=1.0,
                rationale="Targets evidence-based treatment for low mood",
            ),
        ],
        raw_text_summary="Treatment for low mood",
    )


def _chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            text="Cognitive behavioural therapy is effective for low mood.",
            source="DSM5.pdf",
            page=12,
            score=0.88,
            chunk_id="DSM5__p0012__c0001",
            rank=1,
            metadata={"source": "DSM5.pdf", "page": 12},
        ),
        RetrievedChunk(
            text="Behavioural activation reduces depressive symptoms.",
            source="NICE_Guidelines.pdf",
            page=4,
            score=0.82,
            chunk_id="NICE__p0004__c0001",
            rank=2,
            metadata={"source": "NICE_Guidelines.pdf", "page": 4},
        ),
    ]


def _evidence() -> EvidenceSynthesisResult:
    return EvidenceSynthesisResult(
        key_findings=[
            Finding(
                statement="CBT shows strong evidence for low mood",
                supporting_sources=["DSM5.pdf, p.12"],
                confidence=0.8,
            )
        ],
        common_themes=[
            Theme(
                name="Cognitive patterns",
                description="Negative thoughts maintain low mood",
                prevalence=0.8,
            )
        ],
        areas_of_agreement=[
            Agreement(
                topic="CBT efficacy",
                consensus="CBT is an effective first-line approach",
                supporting_sources=["DSM5.pdf"],
            )
        ],
        overall_summary="Evidence supports CBT as an effective approach for low mood.",
        confidence=0.8,
        chunks_analysed=2,
    )


def _framework_results() -> dict[str, FrameworkResult]:
    out = {}
    for name in ("CBT", "ACT", "DBT", "Humanistic"):
        out[name] = FrameworkResult(
            framework_name=name,
            framework_description=f"{name} framework applied to this case",
            conceptualization=f"Conceptualisation through the {name} lens",
            key_concepts=[f"{name} core concept"],
            therapeutic_approaches=[f"{name} technique"],
            strengths=["Good fit for this presentation"],
            limitations=["Further assessment recommended"],
            confidence=0.7 if name != "DBT" else 0.8,
            evidence_alignment="Aligns with retrieved evidence",
        )
    return out


def _formulation() -> ClinicalFormulationResult:
    return ClinicalFormulationResult(
        case_summary=(
            "A 34-year-old female presenting with low mood and reduced "
            "motivation over the past six weeks."
        ),
        possible_formulations=[
            Formulation(
                label="Cognitive-behavioural formulation",
                explanation=(
                    "Negative automatic thoughts about work performance and "
                    "reduced activity may be maintaining the low mood."
                ),
                supporting_symptoms=["low mood", "reduced motivation"],
                confidence=0.7,
            )
        ],
        supporting_evidence=["Evidence supports the cognitive model of low mood"],
        alternative_explanations=["Adjustment reaction to work pressure"],
        missing_assessment_information=["Trauma history", "Sleep quality"],
        caution="Risk assessment recommended given functional impairment",
        confidence=0.7,
    )


def _missing_info() -> MissingInfoResult:
    return MissingInfoResult(
        input_summary="Low mood with reduced motivation",
        missing_information=[
            MissingInfoItem(
                info_gap="Sleep quality",
                clinical_relevance="Sleep disturbance affects mood and treatment planning",
                suggested_questions=["How has your sleep been recently?"],
            )
        ],
        overall_assessment="Moderate gaps in the clinical picture",
    )


def _plan() -> TherapeuticPlanResult:
    return TherapeuticPlanResult(
        disclaimer=(
            "All suggestions are evidence-informed and require the "
            "clinician's professional judgement."
        ),
        therapeutic_focus=[
            TherapeuticFocus(
                area="Cognitive patterns",
                rationale="Negative thoughts maintain the low mood",
            )
        ],
        intervention_directions=[
            InterventionDirection(
                area="Cognitive restructuring",
                suggested_approaches=["CBT"],
                rationale="Targets maintaining cognitions",
            )
        ],
        cbt_strategies=[
            CBTStrategy(
                technique="Thought record",
                rationale="Identifies and tests automatic thoughts",
            )
        ],
        act_strategies=[
            ACTStrategy(
                process="Defusion",
                rationale="Reduces fusion with unhelpful thoughts",
            )
        ],
        dbt_strategies=[
            DBTStrategy(
                skill="Mindfulness",
                rationale="Supports emotional regulation",
            )
        ],
    )


def _response() -> ClinicalResponseResult:
    return ClinicalResponseResult(
        markdown=(
            "## 1. Case Summary\n\n"
            "A 34-year-old female presents with low mood and reduced "
            "motivation over the past six weeks.\n\n"
            "## 2. Formulation\n\n"
            "Cognitive-behavioural formulation is suggested.\n\n"
        ),
        sections_generated=11,
        confidence=0.7,
    )


def _session_plan() -> SessionPlanResult:
    return SessionPlanResult(
        disclaimer=(
            "Session proposals are suggestions for clinician consideration "
            "and must be adapted to the client."
        ),
        session_count=8,
        cadence="weekly",
        objectives=[
            SessionObjective(
                title="Engagement and shared formulation",
                focus="Establish the therapeutic relationship within the DBT lens",
                suggested_activities=["Review formulation", "Set goals"],
                homework="Reflect on formulation",
                expected_duration_minutes=50,
            )
        ],
        review_points=["Review after session 4"],
    )


def _safety_report() -> SafetyValidationResult:
    return SafetyValidationResult(
        markdown=_response().markdown,
        original_markdown=_response().markdown,
        issues=[],
        was_revised=False,
        revision_summary="",
        overall_verdict="clean",
    )


# ═══════════════════════════════════════════════════════════════
# Patch helpers
# ═══════════════════════════════════════════════════════════════

def _patch_llm():
    return patch("clinical.graph.nodes.get_llm_service", return_value=MagicMock())


def _upstream_mocks():
    """Mock the upstream node factories needed up to framework comparison."""
    extractor = MagicMock()
    extractor.extract = AsyncMock(return_value=_understanding())
    gen = MagicMock()
    gen.generate = AsyncMock(return_value=_queries())
    retriever = MagicMock()
    retriever.aretrieve = AsyncMock(return_value=MagicMock(chunks=_chunks()))
    synth = MagicMock()
    synth.synthesise = AsyncMock(return_value=_evidence())
    router = MagicMock()
    router.route = AsyncMock(return_value=_framework_results())
    return (
        patch("clinical.graph.nodes._get_case_understanding", return_value=extractor),
        patch("clinical.graph.nodes._get_query_generator", return_value=gen),
        patch("clinical.graph.nodes._get_retriever", return_value=retriever),
        patch("clinical.graph.nodes._get_evidence_synthesizer", return_value=synth),
        patch("clinical.graph.nodes.ParallelFrameworkRouter", return_value=router),
    )


def _downstream_mocks():
    """Mock the downstream node factories needed to complete a resume."""
    formulator = MagicMock()
    formulator.formulate = AsyncMock(return_value=_formulation())
    detector = MagicMock()
    detector.detect = AsyncMock(return_value=_missing_info())
    planner = MagicMock()
    planner.plan = AsyncMock(return_value=_plan())
    session_planner = MagicMock()
    session_planner.plan = AsyncMock(return_value=_session_plan())
    generator = MagicMock()
    generator.generate = AsyncMock(return_value=_response())
    validator = MagicMock()
    validator.validate = AsyncMock(return_value=_safety_report())
    return (
        patch("clinical.graph.nodes._get_formulator", return_value=formulator),
        patch("clinical.graph.nodes._get_missing_info_detector", return_value=detector),
        patch("clinical.graph.nodes._get_therapeutic_planner", return_value=planner),
        patch("clinical.graph.nodes._get_session_planner", return_value=session_planner),
        patch("clinical.graph.nodes._get_response_generator", return_value=generator),
        patch("clinical.graph.nodes._get_safety_validator", return_value=validator),
    )


# ═══════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════

def _pause_via_api(test_client) -> str:
    """Run a resumable analyze through the API and return its workflow_id."""
    with ExitStack() as stack:
        stack.enter_context(_patch_llm())
        for p in _upstream_mocks():
            stack.enter_context(p)
        resp = test_client.post(
            "/api/v1/clinical/analyze",
            json={"text": TEXT, "resumable": True},
        )
    assert resp.status_code == 200
    return resp.json()["workflow_id"]


class TestPause:
    def test_resumable_analyze_pauses_after_comparison(self, client):
        test_client, store, _ = client
        with ExitStack() as stack:
            stack.enter_context(_patch_llm())
            for p in _upstream_mocks():
                stack.enter_context(p)
            resp = test_client.post(
                "/api/v1/clinical/analyze",
                json={"text": TEXT, "resumable": True},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "awaiting_selection"
        assert data["workflow_id"]
        assert data["framework_comparison"]["rows"]
        assert data["framework_comparison"]["recommendation"]
        assert data["response"] == {}

        persisted = store.load(data["workflow_id"])
        assert persisted is not None
        assert persisted["awaiting_selection"] is True
        assert set(persisted["enabled_frameworks"]) == {
            "CBT", "ACT", "DBT", "Humanistic"
        }
        assert set(persisted["framework_results"].keys()) == {
            "CBT", "ACT", "DBT", "Humanistic"
        }

    def test_non_resumable_analyze_completes(self, client):
        test_client, store, _ = client
        with ExitStack() as stack:
            stack.enter_context(_patch_llm())
            for p in _upstream_mocks():
                stack.enter_context(p)
            for p in _downstream_mocks():
                stack.enter_context(p)
            resp = test_client.post(
                "/api/v1/clinical/analyze",
                json={"text": TEXT},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["workflow_id"] is None
        assert data["response"]["markdown"]


class TestSelect:
    def test_select_resumes_with_chosen_framework(self, client):
        test_client, store, archive = client
        workflow_id = _pause_via_api(test_client)

        with ExitStack() as stack:
            stack.enter_context(_patch_llm())
            for p in _downstream_mocks():
                stack.enter_context(p)
            resp = test_client.post(
                "/api/v1/clinical/framework/select",
                json={"workflow_id": workflow_id, "selected_framework": "DBT"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["workflow_id"] == workflow_id
        assert data["formulation"]
        assert data["missing_info"]
        assert data["therapeutic_planning"]
        assert data["session_planning"]["session_count"] == 8
        assert data["safety_report"]["overall_verdict"] == "clean"
        assert "DBT" in data["response"]["markdown"]
        assert "Framework Comparison" in data["response"]["markdown"]

        rows = data["framework_comparison"]["rows"]
        assert rows, "expected a comparison after resume"
        for row in rows:
            assert set(row["entries"].keys()) == {"DBT"}, (
                "resume must continue using only the selected framework"
            )

        assert store.load(workflow_id) is None, "pending workflow should be deleted"
        archived = archive.load(workflow_id)
        assert archived is not None, "completed workflow should be archived"
        assert archived["selected_framework"] == "DBT"

    def test_select_unknown_workflow_returns_404(self, client):
        test_client, _, _ = client
        resp = test_client.post(
            "/api/v1/clinical/framework/select",
            json={"workflow_id": "missing-id", "selected_framework": "DBT"},
        )
        assert resp.status_code == 404

    def test_select_invalid_framework_returns_422(self, client):
        test_client, store, _ = client
        workflow_id = _pause_via_api(test_client)
        resp = test_client.post(
            "/api/v1/clinical/framework/select",
            json={"workflow_id": workflow_id, "selected_framework": "NOPE"},
        )
        assert resp.status_code == 422
        # The pending workflow must be restored so the psychologist can retry.
        assert store.load(workflow_id) is not None
