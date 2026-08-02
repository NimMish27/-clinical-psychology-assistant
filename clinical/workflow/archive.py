from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from app_logging.logger import get_logger
from clinical.workflow.serialization import state_from_json, state_to_json

if TYPE_CHECKING:
    from pathlib import Path

_log = get_logger(__name__)


class ArchiveStore:
    """SQLite-backed store for completed (archived) clinical workflows.

    Provides the "future memory" of the system: after a resumable
    workflow is completed, the psychologist-chosen framework and the key
    outputs are archived so later sessions can reference what was
    previously decided. Kept separate from the pending-workflow store.

    Rows are purged after ``retention_days``.
    """

    def __init__(self, db_path: Path, retention_days: int = 365) -> None:
        self._db_path = db_path
        self._retention_days = int(retention_days)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS completed_workflows (
                    workflow_id       TEXT PRIMARY KEY,
                    selected_framework TEXT NOT NULL,
                    state             TEXT NOT NULL,
                    created_at        TEXT NOT NULL,
                    expires_at        TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_archive_expires
                ON completed_workflows(expires_at)
                """
            )

    def archive(
        self,
        workflow_id: str,
        selected_framework: str,
        state: dict[str, Any],
    ) -> None:
        """Persist a completed workflow's chosen framework and outputs."""
        now = datetime.now(UTC)
        expires = now + timedelta(days=self._retention_days)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO completed_workflows
                    (workflow_id, selected_framework, state, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    workflow_id,
                    selected_framework,
                    state_to_json(state),
                    now.isoformat(),
                    expires.isoformat(),
                ),
            )
        _log.info(
            "archive_store.archived",
            workflow_id=workflow_id,
            selected_framework=selected_framework,
            expires_at=expires.isoformat(),
        )

    def load(self, workflow_id: str) -> dict[str, Any] | None:
        """Load an archived completed workflow, or None if missing / expired."""
        now = datetime.now(UTC)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT state FROM completed_workflows
                WHERE workflow_id = ? AND expires_at > ?
                """,
                (workflow_id, now.isoformat()),
            ).fetchone()
        if row is None:
            return None
        return state_from_json(row["state"])

    def list_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return the most recently archived workflows (metadata only)."""
        limit = max(0, min(int(limit), 200))
        now = datetime.now(UTC)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT workflow_id, selected_framework, created_at
                FROM completed_workflows
                WHERE expires_at > ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (now.isoformat(), limit),
            ).fetchall()
        return [
            {
                "workflow_id": row["workflow_id"],
                "selected_framework": row["selected_framework"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def delete(self, workflow_id: str) -> None:
        """Remove an archived workflow."""
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM completed_workflows WHERE workflow_id = ?",
                (workflow_id,),
            )
        _log.info("archive_store.deleted", workflow_id=workflow_id)

    def prune_expired(self) -> int:
        """Delete all expired rows; returns the number removed."""
        now = datetime.now(UTC)
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM completed_workflows WHERE expires_at <= ?",
                (now.isoformat(),),
            )
            removed = cur.rowcount
        if removed:
            _log.info("archive_store.pruned", removed=removed)
        return removed

    @property
    def retention_days(self) -> int:
        return self._retention_days


# ── Module-level singleton ─────────────────────────────────────

_archive: ArchiveStore | None = None


def get_archive_store() -> ArchiveStore:
    """Return the process-wide ArchiveStore singleton.

    Configured from ``WorkflowSettings`` (archive db path + retention).
    """
    global _archive
    if _archive is None:
        from config.settings import get_settings

        settings = get_settings()
        _archive = ArchiveStore(
            db_path=settings.workflow.archive_db_path,
            retention_days=settings.workflow.archive_retention_days,
        )
    return _archive
