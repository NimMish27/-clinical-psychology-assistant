"""
tests/unit/clinical/test_workflow.py
────────────────────────────────────
Unit tests for the human-in-the-loop workflow: the framework-selection
gate node, the graph wiring around it, the SQLite WorkflowStore, and the
GraphState <-> JSON serialization round-trip.

Run:
    pytest tests/unit/clinical/test_workflow.py -v
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from clinical.frameworks.models import FrameworkComparisonResult, FrameworkResult
from clinical.graph.graph import (
    RESUME_SEQUENCE,
    SEQUENCE,
    _after_framework_selection,
    build_graph,
    build_resume_graph,
)
from clinical.graph.nodes import node_framework_selection
from clinical.workflow.archive import ArchiveStore
from clinical.workflow.serialization import state_from_json, state_to_json
from clinical.workflow.store import WorkflowStore
from rag.retriever import RetrievedChunk

if TYPE_CHECKING:
    from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _fw(name: str, confidence: float = 0.7) -> FrameworkResult:
    return FrameworkResult(
        framework_name=name,
        framework_description=f"{name} framework for this case",
        conceptualization=f"Conceptualisation through the {name} lens",
        key_concepts=[f"{name} concept"],
        therapeutic_approaches=[f"{name} technique"],
        strengths=["Fit for this presentation"],
        limitations=["Needs more data"],
        confidence=confidence,
        evidence_alignment="Aligns with retrieved evidence",
    )


def _results(*names: str) -> dict[str, FrameworkResult]:
    return {name: _fw(name) for name in names}


# ═══════════════════════════════════════════════════════════════
# node_framework_selection
# ═══════════════════════════════════════════════════════════════

class TestNodeFrameworkSelection:
    @pytest.mark.asyncio
    async def test_pause_when_workflow_id_without_selection(self):
        result = await node_framework_selection({
            "workflow_id": "wf-123",
            "text": "case",
            "errors": {},
        })
        assert result == {"awaiting_selection": True}

    @pytest.mark.asyncio
    async def test_resume_prunes_to_selected_framework(self):
        state = {
            "workflow_id": "wf-123",
            "selected_framework": "DBT",
            "enabled_frameworks": ["CBT", "ACT", "DBT", "Humanistic"],
            "framework_results": _results("CBT", "ACT", "DBT", "Humanistic"),
        }
        result = await node_framework_selection(state)
        assert result["awaiting_selection"] is False
        assert result["enabled_frameworks"] == ["DBT"]
        assert list(result["framework_results"].keys()) == ["DBT"]
        assert result["framework_results"]["DBT"].framework_name == "DBT"
        assert isinstance(result["framework_comparison"], FrameworkComparisonResult)

    @pytest.mark.asyncio
    async def test_resume_recomputes_single_framework_comparison(self):
        results = _results("CBT", "DBT")
        result = await node_framework_selection({
            "workflow_id": "wf-123",
            "selected_framework": "CBT",
            "framework_results": results,
        })
        comparison = result["framework_comparison"]
        assert comparison.recommendation
        assert "CBT" in comparison.recommendation

    @pytest.mark.asyncio
    async def test_resume_missing_result_returns_empty_gracefully(self):
        result = await node_framework_selection({
            "workflow_id": "wf-123",
            "selected_framework": "NOPE",
            "framework_results": _results("CBT"),
        })
        assert result["awaiting_selection"] is False
        assert result["framework_results"] == {}
        assert isinstance(result["framework_comparison"], FrameworkComparisonResult)

    @pytest.mark.asyncio
    async def test_legacy_run_passes_through(self):
        result = await node_framework_selection({
            "text": "case",
            "errors": {},
        })
        assert result == {"awaiting_selection": False}


# ═══════════════════════════════════════════════════════════════
# Routing
# ═══════════════════════════════════════════════════════════════

class TestAfterFrameworkSelection:
    def test_pause_when_awaiting(self):
        assert _after_framework_selection({"awaiting_selection": True}) == "pause"

    def test_continue_when_not_awaiting(self):
        assert _after_framework_selection({"awaiting_selection": False}) == "continue"
        assert _after_framework_selection({}) == "continue"


# ═══════════════════════════════════════════════════════════════
# Graph wiring
# ═══════════════════════════════════════════════════════════════

class TestGraphWiring:
    def test_framework_selection_is_in_sequence(self):
        assert "framework_selection" in SEQUENCE
        assert SEQUENCE.index("framework_comparison") < SEQUENCE.index(
            "framework_selection"
        )
        assert SEQUENCE.index("framework_selection") < SEQUENCE.index(
            "clinical_formulation"
        )

    def test_full_graph_compiles(self):
        compiled = build_graph().compile()
        assert compiled is not None

    def test_resume_graph_compiles(self):
        compiled = build_resume_graph().compile()
        assert compiled is not None

    def test_resume_sequence_contains_only_downstream_nodes(self):
        assert RESUME_SEQUENCE[0] == "framework_selection"
        assert RESUME_SEQUENCE[-1] == "safety_validation"
        assert "case_understanding" not in RESUME_SEQUENCE
        assert "framework_comparison" not in RESUME_SEQUENCE

    def test_session_planning_in_both_sequences(self):
        assert "session_planning" in SEQUENCE
        assert "session_planning" in RESUME_SEQUENCE
        assert SEQUENCE.index("therapeutic_planning") < SEQUENCE.index("session_planning") < SEQUENCE.index("response_generation")
        assert RESUME_SEQUENCE.index("therapeutic_planning") < RESUME_SEQUENCE.index("session_planning") < RESUME_SEQUENCE.index("response_generation")


# ═══════════════════════════════════════════════════════════════
# Serialization
# ═══════════════════════════════════════════════════════════════

class TestSerialization:
    def test_round_trip_full_state(self):
        state = {
            "text": "34-year-old female with low mood.",
            "session_id": "sess-1",
            "workflow_id": "wf-123",
            "awaiting_selection": True,
            "enabled_frameworks": ["CBT", "ACT", "DBT", "Humanistic"],
            "errors": {"retrieval": "boom"},
            "framework_results": _results("CBT", "DBT"),
            "retrieved_chunks": [
                RetrievedChunk(
                    text="chunk text",
                    source="DSM5.pdf",
                    page=2,
                    score=0.9,
                    chunk_id="c1",
                    rank=1,
                    metadata={"source": "DSM5.pdf", "page": 2},
                )
            ],
        }
        restored = state_from_json(state_to_json(state))
        assert restored["text"] == state["text"]
        assert restored["workflow_id"] == "wf-123"
        assert restored["awaiting_selection"] is True
        assert restored["errors"] == {"retrieval": "boom"}
        assert restored["enabled_frameworks"] == ["CBT", "ACT", "DBT", "Humanistic"]
        assert isinstance(restored["framework_results"]["CBT"], FrameworkResult)
        assert isinstance(restored["retrieved_chunks"][0], RetrievedChunk)

    def test_round_trip_handles_none(self):
        state = {"text": "case", "formulation": None, "errors": {}}
        restored = state_from_json(state_to_json(state))
        assert restored["text"] == "case"
        assert restored["formulation"] is None


# ═══════════════════════════════════════════════════════════════
# WorkflowStore
# ═══════════════════════════════════════════════════════════════

class TestWorkflowStore:
    @pytest.fixture
    def store(self, tmp_path: Path) -> WorkflowStore:
        return WorkflowStore(db_path=tmp_path / "workflows.db", ttl_seconds=3600)

    def test_save_and_load(self, store: WorkflowStore):
        store.save("wf-1", {"text": "case", "errors": {}})
        loaded = store.load("wf-1")
        assert loaded is not None
        assert loaded["text"] == "case"

    def test_load_missing_returns_none(self, store: WorkflowStore):
        assert store.load("does-not-exist") is None

    def test_delete(self, store: WorkflowStore):
        store.save("wf-1", {"text": "case", "errors": {}})
        store.delete("wf-1")
        assert store.load("wf-1") is None

    def test_overwrite_same_id(self, store: WorkflowStore):
        store.save("wf-1", {"text": "first", "errors": {}})
        store.save("wf-1", {"text": "second", "errors": {}})
        assert store.load("wf-1")["text"] == "second"

    def test_expired_returns_none(self, tmp_path: Path):
        store = WorkflowStore(db_path=tmp_path / "wf.db", ttl_seconds=3600)
        store.save("wf-1", {"text": "case", "errors": {}})
        # Force expiry by rewriting the stored timestamp into the past.
        past = datetime.now(UTC) - timedelta(hours=2)
        from clinical.workflow.store import sqlite3
        with sqlite3.connect(store._db_path) as conn:
            conn.execute(
                "UPDATE pending_workflows SET expires_at = ? WHERE workflow_id = ?",
                (past.isoformat(), "wf-1"),
            )
        assert store.load("wf-1") is None

    def test_prune_expired(self, tmp_path: Path):
        store = WorkflowStore(db_path=tmp_path / "wf.db", ttl_seconds=3600)
        store.save("wf-live", {"text": "case", "errors": {}})
        store.save("wf-old", {"text": "case", "errors": {}})
        past = datetime.now(UTC) - timedelta(hours=2)
        from clinical.workflow.store import sqlite3
        with sqlite3.connect(store._db_path) as conn:
            conn.execute(
                "UPDATE pending_workflows SET expires_at = ? WHERE workflow_id = ?",
                (past.isoformat(), "wf-old"),
            )
        assert store.prune_expired() == 1
        assert store.load("wf-live") is not None

    def test_claim_returns_and_removes(self, store: WorkflowStore):
        store.save("wf-1", {"text": "case", "awaiting_selection": True, "errors": {}})
        claimed = store.claim("wf-1")
        assert claimed is not None
        assert claimed["awaiting_selection"] is True
        assert store.load("wf-1") is None

    def test_claim_missing_returns_none(self, store: WorkflowStore):
        assert store.claim("nope") is None

    def test_claim_only_succeeds_once(self, store: WorkflowStore):
        store.save("wf-1", {"text": "case", "errors": {}})
        first = store.claim("wf-1")
        second = store.claim("wf-1")
        assert first is not None
        assert second is None

    def test_claim_expired_returns_none(self, tmp_path: Path):
        store = WorkflowStore(db_path=tmp_path / "wf.db", ttl_seconds=3600)
        store.save("wf-1", {"text": "case", "errors": {}})
        past = datetime.now(UTC) - timedelta(hours=2)
        from clinical.workflow.store import sqlite3
        with sqlite3.connect(store._db_path) as conn:
            conn.execute(
                "UPDATE pending_workflows SET expires_at = ? WHERE workflow_id = ?",
                (past.isoformat(), "wf-1"),
            )
        assert store.claim("wf-1") is None


class TestArchiveStore:
    @pytest.fixture
    def archive(self, tmp_path: Path) -> ArchiveStore:
        return ArchiveStore(db_path=tmp_path / "archive.db", retention_days=365)

    def test_archive_and_load(self, archive: ArchiveStore):
        archive.archive(
            "wf-1",
            "DBT",
            {"text": "case", "selected_framework": "DBT", "errors": {}},
        )
        loaded = archive.load("wf-1")
        assert loaded is not None
        assert loaded["selected_framework"] == "DBT"

    def test_load_missing_returns_none(self, archive: ArchiveStore):
        assert archive.load("nope") is None

    def test_list_recent(self, archive: ArchiveStore):
        archive.archive("wf-1", "CBT", {"errors": {}})
        archive.archive("wf-2", "ACT", {"errors": {}})
        recent = archive.list_recent()
        assert len(recent) == 2
        assert {r["selected_framework"] for r in recent} == {"CBT", "ACT"}

    def test_delete(self, archive: ArchiveStore):
        archive.archive("wf-1", "CBT", {"errors": {}})
        archive.delete("wf-1")
        assert archive.load("wf-1") is None

    def test_prune_expired(self, tmp_path: Path):
        archive = ArchiveStore(db_path=tmp_path / "a.db", retention_days=365)
        archive.archive("wf-live", "CBT", {"errors": {}})
        archive.archive("wf-old", "ACT", {"errors": {}})
        past = datetime.now(UTC) - timedelta(days=400)
        from clinical.workflow.archive import sqlite3
        with sqlite3.connect(archive._db_path) as conn:
            conn.execute(
                "UPDATE completed_workflows SET expires_at = ? WHERE workflow_id = ?",
                (past.isoformat(), "wf-old"),
            )
        assert archive.prune_expired() == 1
        assert archive.load("wf-live") is not None

    def test_archive_round_trips_full_state(self, archive: ArchiveStore):
        state = {
            "text": "case",
            "selected_framework": "DBT",
            "enabled_frameworks": ["DBT"],
            "framework_results": _results("DBT"),
            "errors": {},
        }
        archive.archive("wf-1", "DBT", state)
        loaded = archive.load("wf-1")
        assert isinstance(loaded["framework_results"]["DBT"], FrameworkResult)
