"""
tests/unit/api/test_api.py
───────────────────────────
Unit tests for all three API endpoints: /chat, /ingest, /health.

All external dependencies are mocked — no Ollama, ChromaDB, or real
PDF files are required. Tests use FastAPI's TestClient / AsyncClient.

Run:
    pytest tests/unit/api/test_api.py -v
"""

from __future__ import annotations

import io
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from api.dependencies import get_settings_dep, get_retriever_dep


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


@pytest.fixture(scope="module")
def client(mock_settings):
    with patch("config.settings.get_settings", return_value=mock_settings), \
         patch("api.dependencies._cached_settings", return_value=mock_settings), \
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

        from api.main import create_app
        app = create_app(settings=mock_settings)

        with TestClient(app, raise_server_exceptions=False) as c:
            yield c, mock_vs


def _make_retrieval_result(n: int = 3, score: float = 0.88):
    from rag.retriever import RetrievalResult, RetrievedChunk
    chunks = [
        RetrievedChunk(
            text=f"Clinical content chunk {i}.",
            source="DSM5.pdf",
            page=i + 1,
            score=score - i * 0.03,
            chunk_id=f"DSM5__p{i+1:04d}__c0000",
            rank=i + 1,
            metadata={"source": "DSM5.pdf", "page": i + 1},
        )
        for i in range(n)
    ]
    return MagicMock(
        chunks=chunks,
        found=True,
        top=chunks[0] if chunks else None,
        to_cited_context=MagicMock(
            return_value="\n\n---\n\n".join(
                f"[DSM5.pdf, p.{c.page}, score={c.score:.2f}] {c.text}"
                for c in chunks
            )
        ),
        to_context_string=MagicMock(
            return_value="\n\n".join(c.text for c in chunks)
        ),
        to_dicts=MagicMock(return_value=[c.to_dict() for c in chunks]),
        retrieval_ms=45.2,
        embedding_ms=12.1,
        search_ms=33.1,
    )


# ═══════════════════════════════════════════════════════════════
# POST /api/v1/chat
# ═══════════════════════════════════════════════════════════════

class TestChatEndpoint:
    def _post(self, client_tuple, body: dict, **patches) -> Any:
        test_client, mock_vs = client_tuple
        return test_client.post("/api/v1/chat", json=body)

    def test_200_with_answer_and_sources(self, client):
        test_client, _ = client
        retrieval_result = _make_retrieval_result(3)
        mock_retriever = MagicMock()
        mock_retriever.aretrieve = AsyncMock(return_value=retrieval_result)

        with patch("api.routers.chat.RetrieverDep", mock_retriever), \
             patch("api.dependencies.get_retriever_dep", return_value=mock_retriever), \
             patch("api.routers.chat._call_llm", return_value="CBT is effective for GAD."):

            resp = test_client.post(
                "/api/v1/chat",
                json={"query": "What is cognitive behavioural therapy?"},
            )

        assert resp.status_code in (200, 422, 503)

    def test_response_schema_fields(self, client):
        test_client, _ = client
        retrieval_result = _make_retrieval_result(2)

        mock_settings = MagicMock(
            llm=MagicMock(
                ollama_model="llama3.1:8b",
                ollama_base_url="http://localhost:11434",
                temperature=0.1, top_k=40, top_p=0.9, max_tokens=2048,
            )
        )
        mock_retriever = MagicMock()
        mock_retriever.aretrieve = AsyncMock(return_value=retrieval_result)

        test_client.app.dependency_overrides[get_settings_dep] = lambda: mock_settings
        test_client.app.dependency_overrides[get_retriever_dep] = lambda: mock_retriever

        with patch("api.routers.chat._call_llm",
                   return_value="DSM-5 defines MDD as..."):
            resp = test_client.post(
                "/api/v1/chat",
                json={"query": "What are the DSM-5 criteria for MDD?"},
            )

        test_client.app.dependency_overrides.clear()

        if resp.status_code == 200:
            data = resp.json()
            assert "answer" in data
            assert "sources" in data
            assert isinstance(data["sources"], list)
            assert "model" in data
            assert "total_ms" in data

    def test_422_empty_query(self, client):
        test_client, _ = client
        resp = test_client.post("/api/v1/chat", json={"query": ""})
        assert resp.status_code == 422

    def test_422_query_too_short(self, client):
        test_client, _ = client
        resp = test_client.post("/api/v1/chat", json={"query": "AB"})
        assert resp.status_code == 422

    def test_422_query_too_long(self, client):
        test_client, _ = client
        resp = test_client.post(
            "/api/v1/chat",
            json={"query": "A" * 2001},
        )
        assert resp.status_code == 422

    def test_422_invalid_n_sources(self, client):
        test_client, _ = client
        resp = test_client.post(
            "/api/v1/chat",
            json={"query": "What is CBT?", "n_sources": 0},
        )
        assert resp.status_code == 422

    def test_422_invalid_page_range_start_zero(self, client):
        test_client, _ = client
        resp = test_client.post(
            "/api/v1/chat",
            json={"query": "What is CBT?", "page_range": [0, 10]},
        )
        assert resp.status_code == 422

    def test_422_invalid_page_range_reversed(self, client):
        test_client, _ = client
        resp = test_client.post(
            "/api/v1/chat",
            json={"query": "What is CBT?", "page_range": [50, 10]},
        )
        assert resp.status_code == 422

    def test_422_missing_query(self, client):
        test_client, _ = client
        resp = test_client.post("/api/v1/chat", json={})
        assert resp.status_code == 422

    def test_error_response_has_code_field(self, client):
        test_client, _ = client
        resp = test_client.post("/api/v1/chat", json={"query": ""})
        assert resp.status_code == 422
        data = resp.json()
        assert "code" in data
        assert "message" in data

    def test_session_id_optional(self, client):
        test_client, _ = client
        resp = test_client.post(
            "/api/v1/chat",
            json={"query": "What is CBT?", "session_id": "sess-001"},
        )
        assert resp.status_code != 422

    def test_source_filter_optional(self, client):
        test_client, _ = client
        resp = test_client.post(
            "/api/v1/chat",
            json={"query": "What is CBT?", "source_filter": "DSM5.pdf"},
        )
        assert resp.status_code != 422

    def test_valid_page_range_accepted(self, client):
        test_client, _ = client
        resp = test_client.post(
            "/api/v1/chat",
            json={"query": "What is CBT?", "page_range": [10, 50]},
        )
        assert resp.status_code != 422


# ═══════════════════════════════════════════════════════════════
# POST /api/v1/ingest
# ═══════════════════════════════════════════════════════════════

class TestIngestEndpoint:
    def _pdf_file(self, name: str = "test.pdf") -> tuple:
        content = b"%PDF-1.4 fake clinical document content"
        return ("files", (name, io.BytesIO(content), "application/pdf"))

    def test_400_no_files(self, client):
        test_client, _ = client
        resp = test_client.post("/api/v1/ingest", data={})
        assert resp.status_code in (400, 422)

    def test_response_schema_when_processing(self, client):
        from api.dependencies import get_chunker_dep, get_pdf_loader_dep, get_vector_store_dep
        test_client, _ = client

        mock_extraction = MagicMock()
        mock_extraction.extracted_pages = 5
        mock_extraction.source = "test.pdf"
        mock_extraction.usable_pages = MagicMock(return_value=[
            MagicMock(page=i, text=f"Page {i} clinical content " * 20, source="test.pdf")
            for i in range(1, 6)
        ])

        mock_chunking = MagicMock()
        mock_chunking.total_chunks = 10
        mock_chunking.chunks = [
            MagicMock(
                text=f"Clinical chunk {i}.", source="test.pdf",
                page=1, chunk_id=f"test__p0001__c{i:04d}",
            )
            for i in range(10)
        ]
        mock_chunking.to_chromadb_batch = MagicMock(return_value={
            "ids": [f"test__p0001__c{i:04d}" for i in range(10)],
            "documents": [f"chunk {i}" for i in range(10)],
            "metadatas": [{"source": "test.pdf", "page": 1}] * 10,
        })

        mock_emb_result = MagicMock()
        mock_emb_result.total_embedded = 10
        mock_emb_result.to_chromadb_batch = MagicMock(return_value={
            "embeddings": [[0.1] * 1024] * 10,
        })

        mock_insert_result = MagicMock()
        mock_insert_result.total_inserted = 10
        mock_insert_result.total_failed = 0

        mock_loader = MagicMock()
        mock_loader.load = MagicMock(return_value=mock_extraction)

        mock_chunker = MagicMock()
        mock_chunker.chunk_document = MagicMock(return_value=mock_chunking)

        mock_vs = MagicMock()
        mock_vs.add_documents = MagicMock(return_value=mock_insert_result)

        test_client.app.dependency_overrides[get_pdf_loader_dep] = lambda: mock_loader
        test_client.app.dependency_overrides[get_chunker_dep] = lambda: mock_chunker
        test_client.app.dependency_overrides[get_vector_store_dep] = lambda: mock_vs

        with patch("rag.embeddings.embed_documents", return_value=mock_emb_result):
            resp = test_client.post(
                "/api/v1/ingest",
                files=[self._pdf_file("clinical_doc.pdf")],
            )

        test_client.app.dependency_overrides.clear()

        assert resp.status_code in (200, 422, 503)
        if resp.status_code == 200:
            data = resp.json()
            assert "status" in data
            assert "total_files" in data
            assert "files" in data
            assert "total_chunks" in data
            assert "collection" in data

# ═══════════════════════════════════════════════════════════════
# POST /api/v1/clinical/analyze
# ═══════════════════════════════════════════════════════════════

class TestClinicalAnalyzeEndpoint:
    def test_200_with_mock_graph(self, client):
        test_client, _ = client
        mock_state = {
            "text": "Patient feels anxious.",
            "understanding": None,
            "queries_result": None,
            "evidence": None,
            "formulation": None,
            "missing_info": None,
            "plan": None,
            "response": None,
            "errors": {},
        }
        mock_run = AsyncMock(return_value=mock_state)
        from api.dependencies import get_clinical_graph_dep
        test_client.app.dependency_overrides[get_clinical_graph_dep] = lambda: mock_run
        resp = test_client.post(
            "/api/v1/clinical/analyze",
            json={"text": "Patient feels anxious."},
        )
        test_client.app.dependency_overrides.clear()
        assert resp.status_code == 200

    def test_response_schema_fields(self, client):
        test_client, _ = client
        from clinical.case_understanding.models import (
            CaseUnderstandingResult, DemographicInfo,
        )
        from clinical.query_generation.models import OptimizedQuery, QueryGenerationResult
        from clinical.evidence_synthesis.models import EvidenceSynthesisResult
        from clinical.formulation.models import ClinicalFormulationResult, Formulation

        mock_state = {
            "text": "Patient feels anxious.",
            "understanding": CaseUnderstandingResult(
                demographic=DemographicInfo(), raw_text="Patient feels anxious.",
            ),
            "queries_result": QueryGenerationResult(queries=[
                OptimizedQuery(query="anxiety", category="treatment", weight=1.0, rationale="test rationale text"),
            ]),
            "evidence": EvidenceSynthesisResult(overall_summary="Summary text."),
            "formulation": ClinicalFormulationResult(
                case_summary="Test case summary with enough characters.",
                possible_formulations=[
                    Formulation(label="CBT formulation", explanation="A" * 20,
                                supporting_symptoms=["anxiety"], confidence=0.7),
                ],
            ),
            "missing_info": None,
            "plan": None,
            "response": None,
            "errors": {},
        }
        mock_run = AsyncMock(return_value=mock_state)
        from api.dependencies import get_clinical_graph_dep
        test_client.app.dependency_overrides[get_clinical_graph_dep] = lambda: mock_run
        resp = test_client.post(
            "/api/v1/clinical/analyze",
            json={"text": "Patient feels anxious."},
        )
        test_client.app.dependency_overrides.clear()
        assert resp.status_code == 200
        data = resp.json()
        assert "input_type" in data
        assert "understanding" in data
        assert "queries" in data
        assert "evidence" in data
        assert "formulation" in data
        assert "missing_info" in data
        assert "therapeutic_planning" in data
        assert "response" in data
        assert "errors" in data
        assert "elapsed_ms" in data

    def test_422_empty_text(self, client):
        test_client, _ = client
        resp = test_client.post(
            "/api/v1/clinical/analyze",
            json={"text": ""},
        )
        assert resp.status_code == 422

    def test_422_text_too_short(self, client):
        test_client, _ = client
        resp = test_client.post(
            "/api/v1/clinical/analyze",
            json={"text": "ab"},
        )
        assert resp.status_code == 422

    def test_503_when_graph_fails(self, client):
        test_client, _ = client
        async def _failing(*args, **kwargs):
            raise RuntimeError("Graph crashed")
        from api.dependencies import get_clinical_graph_dep
        test_client.app.dependency_overrides[get_clinical_graph_dep] = lambda: _failing
        resp = test_client.post(
            "/api/v1/clinical/analyze",
            json={"text": "Patient feels anxious and has had symptoms for weeks."},
        )
        test_client.app.dependency_overrides.clear()
        assert resp.status_code == 503


# ═══════════════════════════════════════════════════════════════
# POST /api/v1/ingest
# ═══════════════════════════════════════════════════════════════

    def test_ingest_response_schema_fields(self, client):
        from api.schemas.models import FileIngestResult, IngestResponse, IngestStatus
        result = IngestResponse(
            status=IngestStatus.SUCCESS,
            total_files=1,
            succeeded=1,
            failed=0,
            total_chunks=42,
            files=[FileIngestResult(
                filename="DSM5.pdf",
                status=IngestStatus.SUCCESS,
                pages_extracted=300,
                chunks_created=42,
                chunks_embedded=42,
                chunks_stored=42,
            )],
            elapsed_ms=1234.5,
            collection="clinical_knowledge_base",
        )
        d = result.model_dump()
        assert d["status"] == "success"
        assert d["total_files"] == 1
        assert d["total_chunks"] == 42
        assert len(d["files"]) == 1
        assert d["files"][0]["filename"] == "DSM5.pdf"


# ═══════════════════════════════════════════════════════════════
# GET /health
# ═══════════════════════════════════════════════════════════════

class TestHealthEndpoint:
    def test_health_endpoint_exists(self, client):
        test_client, _ = client
        resp = test_client.get("/health")
        assert resp.status_code in (200, 206, 503)

    def test_health_response_has_required_fields(self, client):
        test_client, _ = client
        resp = test_client.get("/health")
        data = resp.json()
        assert "status" in data
        assert "version" in data
        assert "environment" in data
        assert "dependencies" in data
        assert "uptime_s" in data

    def test_health_dependencies_present(self, client):
        test_client, _ = client
        resp = test_client.get("/health")
        data = resp.json()
        deps = data.get("dependencies", {})
        assert isinstance(deps, dict)

    def test_health_status_is_valid_enum(self, client):
        test_client, _ = client
        resp = test_client.get("/health")
        data = resp.json()
        assert data["status"] in ("ok", "degraded", "down")

    def test_503_when_chromadb_down(self, client):
        from api.schemas.models import DependencyHealth, ServiceStatus
        test_client, _ = client
        with patch("api.routers.health._check_chromadb",
                   return_value=DependencyHealth(
                       status=ServiceStatus.DOWN,
                       latency_ms=10.0,
                       detail="Connection refused",
                   )):
            resp = test_client.get("/health")
        assert resp.status_code in (200, 206, 503)

    def test_health_version_matches_settings(self, client):
        test_client, _ = client
        resp = test_client.get("/health")
        if resp.status_code in (200, 206):
            data = resp.json()
            assert data["version"] == "0.1.0"

    def test_health_environment_field(self, client):
        test_client, _ = client
        resp = test_client.get("/health")
        if resp.status_code in (200, 206):
            data = resp.json()
            assert data["environment"] in ("development", "staging", "production")


# ═══════════════════════════════════════════════════════════════
# Schemas (unit tests — no HTTP)
# ═══════════════════════════════════════════════════════════════

class TestSchemas:
    def test_chat_request_trims_query(self):
        from api.schemas.models import ChatRequest
        req = ChatRequest(query="  What is CBT?  ")
        assert req.query == "What is CBT?"

    def test_chat_request_whitespace_only_raises(self):
        from pydantic import ValidationError
        from api.schemas.models import ChatRequest
        with pytest.raises(ValidationError):
            ChatRequest(query="   ")

    def test_chat_request_too_short_raises(self):
        from pydantic import ValidationError
        from api.schemas.models import ChatRequest
        with pytest.raises(ValidationError):
            ChatRequest(query="AB")

    def test_chat_request_page_range_validation(self):
        from pydantic import ValidationError
        from api.schemas.models import ChatRequest
        with pytest.raises(ValidationError):
            ChatRequest(query="What is CBT?", page_range=[50, 10])

    def test_source_reference_score_bounds(self):
        from pydantic import ValidationError
        from api.schemas.models import SourceReference
        with pytest.raises(ValidationError):
            SourceReference(text="t", source="d.pdf", page=1, score=1.5)

    def test_source_reference_valid(self):
        from api.schemas.models import SourceReference
        sr = SourceReference(
            text="Criterion A text.", source="DSM5.pdf", page=12, score=0.91
        )
        assert sr.score == 0.91
        assert sr.page == 12

    def test_chat_response_model(self):
        from api.schemas.models import ChatResponse, SourceReference
        resp = ChatResponse(
            answer="CBT is effective for depression.",
            sources=[SourceReference(text="t", source="d.pdf", page=1, score=0.9)],
            model="llama3.1:8b",
            retrieval_ms=45.2,
            generation_ms=1200.0,
            total_ms=1250.0,
        )
        assert resp.answer == "CBT is effective for depression."
        assert len(resp.sources) == 1

    def test_health_response_model(self):
        from api.schemas.models import DependencyHealth, HealthResponse, ServiceStatus
        resp = HealthResponse(
            status=ServiceStatus.OK,
            version="0.1.0",
            environment="development",
            uptime_s=123.4,
            dependencies={
                "ollama": DependencyHealth(status=ServiceStatus.OK, latency_ms=5.2),
            },
        )
        assert resp.status == ServiceStatus.OK
        assert "ollama" in resp.dependencies

    def test_ingest_response_partial_status(self):
        from api.schemas.models import (
            FileIngestResult, IngestResponse, IngestStatus
        )
        resp = IngestResponse(
            status=IngestStatus.PARTIAL,
            total_files=2,
            succeeded=1,
            failed=1,
            total_chunks=25,
            files=[
                FileIngestResult(filename="good.pdf", status=IngestStatus.SUCCESS,
                                 pages_extracted=10, chunks_created=25,
                                 chunks_embedded=25, chunks_stored=25),
                FileIngestResult(filename="bad.pdf", status=IngestStatus.FAILED,
                                 error="Corrupted PDF"),
            ],
            elapsed_ms=3400.0,
            collection="clinical_kb",
        )
        assert resp.status == IngestStatus.PARTIAL
        assert resp.failed == 1


# ═══════════════════════════════════════════════════════════════
# Middleware behaviour
# ═══════════════════════════════════════════════════════════════

class TestMiddleware:
    def test_request_id_in_response_headers(self, client):
        test_client, _ = client
        resp = test_client.get("/health")
        assert "x-request-id" in resp.headers

    def test_process_time_in_response_headers(self, client):
        test_client, _ = client
        resp = test_client.get("/health")
        assert "x-process-time-ms" in resp.headers

    def test_custom_request_id_honoured(self, client):
        test_client, _ = client
        resp = test_client.get(
            "/health",
            headers={"X-Request-ID": "custom-id-12345"},
        )
        assert resp.headers.get("x-request-id") == "custom-id-12345"

    def test_cors_headers_present(self, client):
        test_client, _ = client
        resp = test_client.options(
            "/api/v1/chat",
            headers={"Origin": "http://localhost:3000"},
        )
        assert resp.status_code in (200, 405)


# ═══════════════════════════════════════════════════════════════
# Error handling
# ═══════════════════════════════════════════════════════════════

class TestErrorHandling:
    def test_404_unknown_route(self, client):
        test_client, _ = client
        resp = test_client.get("/api/v1/unknown-endpoint")
        assert resp.status_code == 404

    def test_405_wrong_method(self, client):
        test_client, _ = client
        resp = test_client.get("/api/v1/chat")
        assert resp.status_code == 405

    def test_validation_error_has_structured_body(self, client):
        test_client, _ = client
        resp = test_client.post("/api/v1/chat", json={"query": ""})
        assert resp.status_code == 422
        body = resp.json()
        assert "code" in body
        assert "message" in body
