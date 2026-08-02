from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from app_logging.logger import get_logger
from clinical.workflow.serialization import state_from_json, state_to_json

if TYPE_CHECKING:
    from pathlib import Path

_log = get_logger(__name__)


class WorkflowStore:
    """SQLite-backed store for paused (human-in-the-loop) workflows.

    Persists the full intermediate ``GraphState`` of a resumable run
    keyed by ``workflow_id`` so the pipeline can be resumed later from
    the framework-selection pause point. Rows expire after ``ttl``.
    """

    def __init__(self, db_path: Path, ttl_seconds: int = 86400) -> None:
        self._db_path = db_path
        self._ttl_seconds = int(ttl_seconds)
        self._init_db()

    # ── Connection helpers ────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pending_workflows (
                    workflow_id  TEXT PRIMARY KEY,
                    pending_state TEXT NOT NULL,
                    created_at   TEXT NOT NULL,
                    expires_at   TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_workflow_expires
                ON pending_workflows(expires_at)
                """
            )

    # ── Public API ────────────────────────────────────────────

    def save(self, workflow_id: str, state: dict[str, Any]) -> None:
        """Persist a paused workflow state."""
        now = datetime.now(UTC)
        expires = now + timedelta(seconds=self._ttl_seconds)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO pending_workflows
                    (workflow_id, pending_state, created_at, expires_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    workflow_id,
                    state_to_json(state),
                    now.isoformat(),
                    expires.isoformat(),
                ),
            )
        _log.info(
            "workflow_store.saved",
            workflow_id=workflow_id,
            expires_at=expires.isoformat(),
        )

    def load(self, workflow_id: str) -> dict[str, Any] | None:
        """Load a paused workflow state, or None if missing / expired."""
        now = datetime.now(UTC)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT pending_state FROM pending_workflows
                WHERE workflow_id = ? AND expires_at > ?
                """,
                (workflow_id, now.isoformat()),
            ).fetchone()
        if row is None:
            return None
        return state_from_json(row["pending_state"])

    def claim(self, workflow_id: str) -> dict[str, Any] | None:
        """Atomically load and remove a paused workflow.

        Uses a single ``DELETE ... RETURNING`` statement so only one
        caller can claim a given workflow — a concurrent second resume
        attempt sees the row as already gone. Returns the state, or None
        if the workflow is missing / expired / already claimed.
        """
        now = datetime.now(UTC)
        with self._connect() as conn:
            row = conn.execute(
                """
                DELETE FROM pending_workflows
                WHERE workflow_id = ? AND expires_at > ?
                RETURNING pending_state
                """,
                (workflow_id, now.isoformat()),
            ).fetchone()
        if row is None:
            return None
        _log.info("workflow_store.claimed", workflow_id=workflow_id)
        return state_from_json(row["pending_state"])

    def delete(self, workflow_id: str) -> None:
        """Remove a paused workflow (e.g. after successful resume)."""
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM pending_workflows WHERE workflow_id = ?",
                (workflow_id,),
            )
        _log.info("workflow_store.deleted", workflow_id=workflow_id)

    def prune_expired(self) -> int:
        """Delete all expired rows; returns the number removed."""
        now = datetime.now(UTC)
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM pending_workflows WHERE expires_at <= ?",
                (now.isoformat(),),
            )
            removed = cur.rowcount
        if removed:
            _log.info("workflow_store.pruned", removed=removed)
        return removed

    @property
    def ttl_seconds(self) -> int:
        return self._ttl_seconds


# ── Module-level singleton ─────────────────────────────────────

_store: WorkflowStore | None = None


def get_workflow_store() -> WorkflowStore:
    """Return the process-wide WorkflowStore singleton.

    Configured from ``WorkflowSettings`` (db path + TTL).
    """
    global _store
    if _store is None:
        from config.settings import get_settings

        settings = get_settings()
        _store = WorkflowStore(
            db_path=settings.workflow.db_path,
            ttl_seconds=settings.workflow.ttl_seconds,
        )
    return _store
