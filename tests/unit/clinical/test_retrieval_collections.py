"""Tests for framework-aware Chroma collection resolution."""

from __future__ import annotations

import builtins
import json
from typing import TYPE_CHECKING

import pytest

import knowledge_pipeline.indexer as indexer_mod
from clinical.retrieval import collections
from clinical.retrieval.collections import (
    get_framework_vector_store,
    resolve_framework_collection,
    resolve_framework_store,
)

if TYPE_CHECKING:
    from pathlib import Path


class _FakeVectorStore:
    """Stand-in for ``rag.vector_store.VectorStore`` that records construction."""

    created: list[_FakeVectorStore] = []

    def __init__(
        self,
        collection_name: str | None = None,
        persist_dir: Path | None = None,
        *,
        exists: bool = True,
    ) -> None:
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self._exists = exists
        _FakeVectorStore.created.append(self)

    def collection_exists(self) -> bool:
        return self._exists


@pytest.fixture(autouse=True)
def _reset_fake_store() -> None:
    _FakeVectorStore.created = []
    yield
    _FakeVectorStore.created = []


def _write_lock(path: Path, entries: dict[str, object]) -> None:
    path.write_text(json.dumps(entries), encoding="utf-8")


def _default_store() -> _FakeVectorStore:
    return _FakeVectorStore(collection_name="clinical_knowledge_base")


class TestResolveFrameworkCollection:
    def test_returns_active_collection_from_lock(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        lock = tmp_path / "manifest.lock"
        _write_lock(lock, {"act": {"collection": "kb_act_2026-08", "active": True}})
        monkeypatch.setattr(indexer_mod, "MANIFEST_LOCK", lock)

        assert resolve_framework_collection("act") == "kb_act_2026-08"

    def test_returns_none_for_unlisted_framework(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        lock = tmp_path / "manifest.lock"
        _write_lock(lock, {"act": {"collection": "kb_act_2026-08"}})
        monkeypatch.setattr(indexer_mod, "MANIFEST_LOCK", lock)

        assert resolve_framework_collection("cbt") is None

    def test_returns_none_when_lock_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(indexer_mod, "MANIFEST_LOCK", tmp_path / "missing.json")

        assert resolve_framework_collection("act") is None

    def test_returns_none_when_lock_corrupt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        lock = tmp_path / "manifest.lock"
        lock.write_text("not json {", encoding="utf-8")
        monkeypatch.setattr(indexer_mod, "MANIFEST_LOCK", lock)

        assert resolve_framework_collection("act") is None

    def test_returns_none_when_pipeline_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real_import = builtins.__import__

        def fake_import(
            name: str, *args: object, **kwargs: object
        ) -> object:
            if name == "knowledge_pipeline" or name.startswith("knowledge_pipeline."):
                raise ImportError("pipeline not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        assert resolve_framework_collection("act") is None


class TestResolveFrameworkStore:
    def test_uses_active_collection_when_it_exists(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            collections, "resolve_framework_collection", lambda fw: "kb_act_2026-08"
        )
        monkeypatch.setattr(collections, "VectorStore", _FakeVectorStore)
        default = _default_store()
        monkeypatch.setattr(collections, "get_vector_store", lambda: default)

        store, name = resolve_framework_store("act")

        assert name == "kb_act_2026-08"
        assert store.collection_name == "kb_act_2026-08"

    def test_falls_back_when_aliased_collection_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            collections, "resolve_framework_collection", lambda fw: "kb_act_2026-08"
        )
        monkeypatch.setattr(
            collections,
            "VectorStore",
            lambda *args: _FakeVectorStore(*args, exists=False),
        )
        default = _default_store()
        monkeypatch.setattr(collections, "get_vector_store", lambda: default)

        store, name = resolve_framework_store("act")

        assert name is None
        assert store is default

    def test_falls_back_when_no_alias(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(collections, "resolve_framework_collection", lambda fw: None)
        default = _default_store()
        monkeypatch.setattr(collections, "get_vector_store", lambda: default)

        store, name = resolve_framework_store("humanistic")

        assert name is None
        assert store is default

    def test_falls_back_when_resolution_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(fw: str) -> str | None:
            raise RuntimeError("chroma down")

        monkeypatch.setattr(collections, "resolve_framework_collection", boom)
        default = _default_store()
        monkeypatch.setattr(collections, "get_vector_store", lambda: default)

        store, name = resolve_framework_store("act")

        assert name is None
        assert store is default


class TestGetFrameworkVectorStore:
    def test_returns_active_store(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        active = _FakeVectorStore(collection_name="kb_act_2026-08")
        monkeypatch.setattr(
            collections,
            "resolve_framework_store",
            lambda fw: (active, "kb_act_2026-08"),
        )

        assert get_framework_vector_store("act") is active

    def test_returns_default_when_resolved_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        default = _default_store()
        monkeypatch.setattr(
            collections, "resolve_framework_store", lambda fw: (default, None)
        )

        assert get_framework_vector_store("cbt") is default
