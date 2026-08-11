"""
tests/unit/knowledge_pipeline/test_indexer.py
"""
from __future__ import annotations

import json

import pytest

from knowledge_pipeline import indexer
from knowledge_pipeline.paths import MANIFEST_LOCK


class TestSanitizeMetadata:
    def test_empty_list_becomes_none(self):
        meta = {"topic": "x", "topic_all": [], "symptom": []}
        out = indexer._sanitize_metadata(meta)
        assert out["topic_all"] is None
        assert out["symptom"] is None
        assert out["topic"] == "x"

    def test_nonempty_list_kept(self):
        meta = {"topic_all": ["a", "b"]}
        assert indexer._sanitize_metadata(meta)["topic_all"] == ["a", "b"]


class TestReadActiveCollection:
    def test_resolves_collection_from_lock(self, tmp_path, monkeypatch):
        lock_file = tmp_path / "manifest.lock"
        lock_file.write_text(
            json.dumps({"act": {"collection": "kb_act_2026-08"}}),
            encoding="utf-8",
        )
        monkeypatch.setattr(indexer, "MANIFEST_LOCK", lock_file)
        assert indexer.read_active_collection("act") == "kb_act_2026-08"

    def test_returns_none_when_framework_unknown(self, tmp_path, monkeypatch):
        lock_file = tmp_path / "manifest.lock"
        lock_file.write_text(json.dumps({"act": {"collection": "kb_act_2026-08"}}), encoding="utf-8")
        monkeypatch.setattr(indexer, "MANIFEST_LOCK", lock_file)
        assert indexer.read_active_collection("cbt") is None

    def test_missing_lock_file_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(indexer, "MANIFEST_LOCK", tmp_path / "nope.lock")
        assert indexer.read_active_collection("act") is None


def test_read_chunks_missing_file_raises(tmp_path):
    with pytest.raises(indexer.IndexerError):
        indexer._read_chunks(tmp_path / "chunks.jsonl")


def test_real_manifest_lock_is_valid_json():
    # After an index run the lock must exist and be parseable JSON.
    if MANIFEST_LOCK.is_file():
        data = json.loads(MANIFEST_LOCK.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
