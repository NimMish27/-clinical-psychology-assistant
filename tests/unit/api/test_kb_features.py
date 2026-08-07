"""
tests/unit/api/test_kb_features.py
───────────────────────────────────
Unit tests for the knowledge-base feature set:
  - /chat with a curated ``framework`` selector
  - /chat with ``include_user_documents`` merge
  - /ingest with a target ``collection`` + tagging metadata
  - /kb/collections, /kb/rebuild, /kb/documents (list + delete)

All external dependencies are mocked — no Ollama, ChromaDB, embeddings,
or real PDFs are required.
"""

from __future__ import annotations

import io
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


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
        mock_vs.collection_name = "test_collection"
        mock_vs_factory.return_value = mock_vs

        from api.main import create_app
        app = create_app(settings=mock_settings)

        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


def _act_evidence_result(n: int = 2) -> Any:
    from clinical.retrieval.act_retriever import ACTEvidenceItem, ACTEvidenceResult

    items = [
        ACTEvidenceItem(
            source=f"hayes2011.pdf",
            content=f"Acceptance and commitment therapy content {i}.",
            relevance="Directly applies to psychological flexibility.",
            quality_rating=0.9 - i * 0.05,
            framework="act",
            chapter="5",
            topic="psychological-flexibility",
            page=i + 1,
            chunk_id=f"act_chunk_{i}",
        )
        for i in range(n)
    ]
    return ACTEvidenceResult(
        query="acceptance and values",
        retrieved=items,
        synthesis="Evidence supports ACT interventions.",
        key_findings=["Psychological flexibility matters"],
        evidence_gaps=["More longitudinal studies"],
        confidence=0.7,
        total_candidates=10,
        elapsed_ms=12.3,
    )


# ═══════════════════════════════════════════════════════════════
# /chat — framework selector + include_user_documents
# ═══════════════════════════════════════════════════════════════

class TestChatFramework:
    def test_framework_accepted(self):
        from api.schemas.models import ChatRequest
        req = ChatRequest(query="What is acceptance?", framework="act")
        assert req.framework == "act"

    def test_invalid_framework_rejected(self):
        from api.schemas.models import ChatRequest
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            ChatRequest(query="What is acceptance?", framework="dbt")

    def test_include_user_documents_defaults_false(self):
        from api.schemas.models import ChatRequest
        req = ChatRequest(query="What is CBT?")
        assert req.include_user_documents is False

    def test_framework_retrieval_returns_sources(self, client):
        mock_fw = MagicMock()
        mock_fw.retrieve_async = AsyncMock(return_value=_act_evidence_result(2))

        with patch("api.routers.chat._get_framework_retriever", return_value=mock_fw), \
             patch("api.routers.chat._call_llm", return_value="ACT helps."):
            resp = client.post(
                "/api/v1/chat",
                json={"query": "What is acceptance?", "framework": "act"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["sources"]) == 2
        assert data["sources"][0]["source"] == "hayes2011.pdf"
        mock_fw.retrieve_async.assert_awaited_once_with("What is acceptance?")

    def test_include_user_documents_merges(self, client):
        from rag.retriever import RetrievedChunk

        user_chunk = RetrievedChunk(
            text="User document clinical note.",
            source="my-note.pdf",
            page=1,
            score=0.99,
            chunk_id="user_1",
            rank=1,
            metadata={"source": "my-note.pdf", "user": "dr-smith"},
        )

        with patch("api.routers.chat._get_framework_retriever") as fw_patch, \
             patch(
                 "api.routers.chat._retrieve_user_documents",
                 return_value=[user_chunk],
             ), \
             patch("api.routers.chat._call_llm", return_value="Merged answer."):
            fw_patch.return_value.retrieve_async = AsyncMock(
                return_value=_act_evidence_result(1)
            )
            resp = client.post(
                "/api/v1/chat",
                json={
                    "query": "What is acceptance?",
                    "framework": "act",
                    "include_user_documents": True,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["sources"][0]["source"] == "my-note.pdf"

    def test_include_user_documents_generic_path(self, client):
        from api.dependencies import get_retriever_dep
        from rag.retriever import RetrievedChunk

        user_chunk = RetrievedChunk(
            text="User document clinical note.",
            source="my-note.pdf",
            page=1,
            score=0.95,
            chunk_id="user_1",
            rank=1,
            metadata={"source": "my-note.pdf"},
        )

        mock_retriever = MagicMock()
        mock_retriever.aretrieve = AsyncMock(
            return_value=MagicMock(
                chunks=[],
                found=False,
                to_cited_context=MagicMock(return_value="none"),
            )
        )
        client.app.dependency_overrides[get_retriever_dep] = lambda: mock_retriever
        try:
            with patch("api.routers.chat._retrieve_user_documents",
                       return_value=[user_chunk]), \
                 patch("api.routers.chat._call_llm", return_value="Merged answer."):
                resp = client.post(
                    "/api/v1/chat",
                    json={"query": "What is CBT?", "include_user_documents": True},
                )
        finally:
            client.app.dependency_overrides.clear()

        assert resp.status_code == 200
        assert resp.json()["sources"][0]["source"] == "my-note.pdf"


# ═══════════════════════════════════════════════════════════════
# /ingest — target collection + tagging metadata
# ═══════════════════════════════════════════════════════════════

class TestIngestCollection:
    def test_ingest_to_user_collection_with_metadata(self, client):
        from api.schemas.models import FileIngestResult, IngestStatus

        captured: dict[str, Any] = {}

        def fake_ingest_one_file(
            file_path, filename, loader, chunker, vector_store, settings, extra_metadata
        ):
            captured["store"] = vector_store
            captured["metadata"] = extra_metadata
            return FileIngestResult(
                filename=filename,
                status=IngestStatus.SUCCESS,
                pages_extracted=2,
                chunks_created=4,
                chunks_embedded=4,
                chunks_stored=4,
            )

        mock_store = MagicMock()
        mock_store.collection_name = "user_documents"

        with patch("api.routers.ingest._ingest_one_file",
                   side_effect=fake_ingest_one_file), \
             patch("rag.vector_store.VectorStore", return_value=mock_store):
            resp = client.post(
                "/api/v1/ingest",
                files=[
                    ("files", ("note.pdf", io.BytesIO(b"%PDF fake"), "application/pdf"))
                ],
                data={
                    "collection": "user_documents",
                    "user": "dr-smith",
                    "project": "clinic-a",
                    "document_type": "case-note",
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["collection"] == "user_documents"
        assert data["total_chunks"] == 4
        assert captured["metadata"]["user"] == "dr-smith"
        assert captured["metadata"]["project"] == "clinic-a"
        assert captured["metadata"]["document_type"] == "case-note"
        assert "upload_date" in captured["metadata"]

    def test_ingest_default_collection_preserved(self, client, mock_settings):
        from api.schemas.models import FileIngestResult, IngestStatus

        captured: dict[str, Any] = {}

        def fake_ingest_one_file(
            file_path, filename, loader, chunker, vector_store, settings, extra_metadata
        ):
            captured["store"] = vector_store
            return FileIngestResult(
                filename=filename,
                status=IngestStatus.SUCCESS,
                pages_extracted=1,
                chunks_created=2,
                chunks_embedded=2,
                chunks_stored=2,
            )

        with patch("api.routers.ingest._ingest_one_file",
                   side_effect=fake_ingest_one_file):
            resp = client.post(
                "/api/v1/ingest",
                files=[
                    ("files", ("doc.pdf", io.BytesIO(b"%PDF fake"), "application/pdf"))
                ],
            )

        assert resp.status_code == 200
        assert resp.json()["collection"] == "test_collection"
        assert captured["store"].collection_name == "test_collection"


# ═══════════════════════════════════════════════════════════════
# /kb — collections, rebuild, documents
# ═══════════════════════════════════════════════════════════════

class TestKbRouter:
    def test_collections_lists_manifest(self, client):
        from knowledge_pipeline.models import BuildResult

        manifest = {
            "act": {
                "active": True,
                "books": ["hayes-2011-acceptance-commitment-therapy-2nd"],
                "built_at": "2026-08-06T12:18:08+00:00",
                "chunk_count": 4,
                "collection": "kb_act_2026-08",
                "index_version": "2026-08",
            }
        }
        with patch("api.routers.kb._load_manifest_lock", return_value=manifest):
            resp = client.get("/api/v1/kb/collections")

        assert resp.status_code == 200
        data = resp.json()["collections"]
        by_name = {c["framework"]: c for c in data}
        assert by_name["act"]["built"] is True
        assert by_name["act"]["collection"] == "kb_act_2026-08"
        assert by_name["act"]["chunk_count"] == 4
        assert by_name["cbt"]["built"] is False
        assert by_name["humanistic"]["built"] is False

    def test_rebuild_success(self, client):
        from knowledge_pipeline.models import BuildResult

        build = BuildResult(
            built_at="2026-08-07T00:00:00+00:00",
            source_files=["05-creative-hopelessness.md"],
            parents=4,
            children=4,
            total_tokens=1070,
            errors=[],
            skipped=[],
        )
        lock = {
            "act": {
                "collection": "kb_act_2026-08",
                "chunk_count": 4,
                "index_version": "2026-08",
            }
        }
        with patch(
            "knowledge_pipeline.builder.build_knowledge_base",
            return_value=build,
        ), patch(
            "knowledge_pipeline.indexer.index_chunks",
            return_value=lock,
        ) as index_mock:
            resp = client.post("/api/v1/kb/rebuild", json={"framework": "act"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["children"] == 4
        assert data["collections"][0]["collection"] == "kb_act_2026-08"
        index_mock.assert_called_once_with(force_embed=False)

    def test_rebuild_failed_build_skips_index(self, client):
        from knowledge_pipeline.models import BuildResult

        build = BuildResult(
            built_at="2026-08-07T00:00:00+00:00",
            errors=["citation unresolved: bib:missing"],
            skipped=["hayes-2011"],
        )
        with patch(
            "knowledge_pipeline.builder.build_knowledge_base",
            return_value=build,
        ), patch(
            "knowledge_pipeline.indexer.index_chunks",
        ) as index_mock:
            resp = client.post("/api/v1/kb/rebuild", json={})

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "failed"
        assert data["errors"] == ["citation unresolved: bib:missing"]
        index_mock.assert_not_called()

    def test_documents_empty_when_collection_missing(self, client):
        mock_store = MagicMock()
        mock_store.collection_exists.return_value = False

        with patch("rag.vector_store.VectorStore", return_value=mock_store):
            resp = client.get("/api/v1/kb/documents")

        assert resp.status_code == 200
        assert resp.json() == {"collection": "user_documents", "sources": []}
        mock_store.list_sources.assert_not_called()

    def test_documents_list(self, client):
        mock_store = MagicMock()
        mock_store.collection_exists.return_value = True
        mock_store.list_sources.return_value = [
            {
                "source": "note.pdf",
                "chunks": 4,
                "user": "dr-smith",
                "project": None,
                "document_type": "case-note",
                "upload_date": "2026-08-07",
            }
        ]

        with patch("rag.vector_store.VectorStore", return_value=mock_store):
            resp = client.get(
                "/api/v1/kb/documents",
                params={"collection": "user_documents"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["collection"] == "user_documents"
        assert data["sources"][0]["source"] == "note.pdf"
        assert data["sources"][0]["chunks"] == 4

    def test_document_delete(self, client):
        mock_store = MagicMock()
        mock_store.collection_exists.return_value = True
        mock_store.delete_by_source.return_value = 4

        with patch("rag.vector_store.VectorStore", return_value=mock_store):
            resp = client.delete(
                "/api/v1/kb/documents",
                params={"collection": "user_documents", "source": "note.pdf"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data == {"collection": "user_documents", "source": "note.pdf", "deleted": 4}
        mock_store.delete_by_source.assert_called_once_with("note.pdf")
