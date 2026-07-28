"""
tests/unit/clinical/test_framework_registry.py — Tests for the Framework Registry system.
"""

from __future__ import annotations

import pytest
from clinical.framework.models import (
    FrameworkMetadata,
    InterventionStyle,
    RetrievalStrategy,
)
from clinical.framework.registry import (
    DuplicateFrameworkError,
    FrameworkNotFoundError,
    FrameworkRegistry,
    register_framework,
)
from clinical.framework.base import FrameworkPlugin


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_registry():
    """Reset the registry singleton before each test."""
    FrameworkRegistry._instance = None
    yield
    FrameworkRegistry._instance = None


def _make_meta(
    id_: str = "test_framework",
    display_name: str = "Test Framework",
    description: str = "A test framework for unit testing purposes.",
) -> FrameworkMetadata:
    return FrameworkMetadata(
        id=id_,
        display_name=display_name,
        description=description,
        supported_use_cases=["Testing", "Validation"],
        intervention_style=InterventionStyle(
            primary_focus="Testing focus",
            techniques=["Test technique"],
            session_structure="Test structure",
            therapeutic_stance="Test stance",
        ),
    )


def _make_plugin_cls(
    id_: str = "test_framework",
    display_name: str = "Test Framework",
    applicable: bool = True,
):
    meta = _make_meta(id_=id_, display_name=display_name)

    class TestPlugin(FrameworkPlugin):
        @classmethod
        def metadata(cls) -> FrameworkMetadata:
            return meta

        @classmethod
        def is_applicable(cls, case_understanding: dict) -> bool:
            return applicable

    return TestPlugin


# ── Test: FrameworkMetadata model ───────────────────────────────────────────


class TestFrameworkMetadata:
    def test_minimal_metadata(self):
        meta = _make_meta()
        assert meta.id == "test_framework"
        assert meta.version == "1.0.0"
        assert meta.retrieval_strategy == RetrievalStrategy.HYBRID
        assert isinstance(meta.intervention_style, InterventionStyle)

    def test_id_validation(self):
        with pytest.raises(Exception):
            _make_meta(id_="Invalid-ID!")

    def test_version_validation(self):
        with pytest.raises(Exception):
            FrameworkMetadata(
                id="valid",
                display_name="Valid",
                description="A valid description for the test case here.",
                version="bad",
                supported_use_cases=["Testing"],
                intervention_style=InterventionStyle(
                    primary_focus="Focus",
                    techniques=["Tech"],
                    session_structure="Struct",
                    therapeutic_stance="Stance",
                ),
            )

    def test_frozen(self):
        meta = _make_meta()
        with pytest.raises(Exception):
            meta.id = "changed"  # type: ignore[misc]


# ── Test: Singleton behavior ────────────────────────────────────────────────


class TestRegistrySingleton:
    def test_get_instance_returns_singleton(self):
        r1 = FrameworkRegistry.get_instance()
        r2 = FrameworkRegistry.get_instance()
        assert r1 is r2

    def test_singleton_reset(self):
        r1 = FrameworkRegistry.get_instance()
        FrameworkRegistry._instance = None
        r2 = FrameworkRegistry.get_instance()
        assert r1 is not r2


# ── Test: Registration ──────────────────────────────────────────────────────


class TestRegistryRegistration:
    def test_register_plugin(self):
        registry = FrameworkRegistry.get_instance()
        cls = _make_plugin_cls("test_reg")
        registry.register(cls)
        assert "test_reg" in registry
        assert registry.get("test_reg") is cls

    def test_register_duplicate_raises(self):
        registry = FrameworkRegistry.get_instance()
        cls1 = _make_plugin_cls("duplicate")
        cls2 = _make_plugin_cls("duplicate")
        registry.register(cls1)
        with pytest.raises(DuplicateFrameworkError):
            registry.register(cls2)

    def test_register_same_class_idempotent(self):
        registry = FrameworkRegistry.get_instance()
        cls = _make_plugin_cls("idempotent")
        registry.register(cls)
        before = registry.list_slugs()
        registry.register(cls)
        after = registry.list_slugs()
        assert before == after

    def test_decorator_registers(self):
        registry = FrameworkRegistry.get_instance()

        @register_framework
        class DecoratedPlugin(FrameworkPlugin):
            @classmethod
            def metadata(cls) -> FrameworkMetadata:
                return _make_meta(id_="decorated", display_name="Decorated")

            @classmethod
            def is_applicable(cls, case_understanding: dict) -> bool:
                return True

        assert "decorated" in registry


# ── Test: Get / Get metadata ────────────────────────────────────────────────


class TestRegistryAccessors:
    def test_get_returns_class(self):
        registry = FrameworkRegistry.get_instance()
        cls = _make_plugin_cls("get_test")
        registry.register(cls)
        assert registry.get("get_test") is cls

    def test_get_unknown_raises(self):
        registry = FrameworkRegistry.get_instance()
        with pytest.raises(FrameworkNotFoundError):
            registry.get("nonexistent")

    def test_get_metadata(self):
        registry = FrameworkRegistry.get_instance()
        meta = _make_meta(id_="meta_test", display_name="Meta Test")
        cls = _make_plugin_cls("meta_test", display_name="Meta Test")
        registry.register(cls)
        retrieved = registry.get_metadata("meta_test")
        assert retrieved.id == "meta_test"
        assert retrieved.display_name == "Meta Test"

    def test_list_metadata_ordered(self):
        registry = FrameworkRegistry.get_instance()
        registry.discover()
        slugs = [m.id for m in registry.list_metadata()]
        assert slugs == sorted(slugs), f"Not sorted: {slugs}"

    def test_len_reflects_count(self):
        registry = FrameworkRegistry.get_instance()
        registry.discover()
        n_before = len(registry)
        registry.register(_make_plugin_cls("additional_test"))
        n_after = len(registry)
        assert n_after == n_before + 1


# ── Test: Enable / Disable ──────────────────────────────────────────────────


class TestRegistryEnableDisable:
    def test_default_enabled(self):
        registry = FrameworkRegistry.get_instance()
        cls = _make_plugin_cls("enabled_test")
        registry.register(cls)
        assert registry.is_enabled("enabled_test") is True

    def test_disable(self):
        registry = FrameworkRegistry.get_instance()
        registry.register(_make_plugin_cls("disable_test"))
        registry.disable("disable_test")
        assert registry.is_enabled("disable_test") is False

    def test_enable(self):
        registry = FrameworkRegistry.get_instance()
        registry.register(_make_plugin_cls("enable_test"))
        registry.disable("enable_test")
        registry.enable("enable_test")
        assert registry.is_enabled("enable_test") is True

    def test_disable_unknown_raises(self):
        registry = FrameworkRegistry.get_instance()
        with pytest.raises(FrameworkNotFoundError):
            registry.disable("nonexistent")

    def test_list_enabled_metadata_excludes_disabled(self):
        registry = FrameworkRegistry.get_instance()
        registry.register(_make_plugin_cls("enabled_one"))
        registry.register(_make_plugin_cls("disabled_one"))
        registry.disable("disabled_one")
        slugs = [m.id for m in registry.list_enabled_metadata()]
        assert "enabled_one" in slugs
        assert "disabled_one" not in slugs


# ── Test: Applicability ─────────────────────────────────────────────────────


class TestRegistryApplicability:
    def test_applicable_for_returns_matching(self):
        registry = FrameworkRegistry.get_instance()

        cls_matches = _make_plugin_cls("matches", applicable=True)
        cls_skips = _make_plugin_cls("skips", applicable=False)

        registry.register(cls_matches)
        registry.register(cls_skips)

        result = registry.applicable_for({"presenting_concerns": "test"})
        slugs = [slug for slug, _ in result]
        assert "matches" in slugs
        assert "skips" not in slugs

    def test_applicable_for_skips_disabled(self):
        registry = FrameworkRegistry.get_instance()
        registry.discover()
        registry.register(_make_plugin_cls("disabled_but_applicable", applicable=True))
        registry.disable("disabled_but_applicable")
        # general plugin always returns True; disable it too for clean count
        registry.disable("general")
        result = registry.applicable_for({"presenting_concerns": "test"})
        assert len(result) == 0

    def test_applicable_for_includes_disabled_when_flag_false(self):
        registry = FrameworkRegistry.get_instance()
        registry.discover()
        registry.register(_make_plugin_cls("disabled_but_applicable", applicable=True))
        registry.disable("disabled_but_applicable")
        result = registry.applicable_for(
            {"presenting_concerns": "test"},
            only_enabled=False,
        )
        slugs = [slug for slug, _ in result]
        assert "disabled_but_applicable" in slugs


# ── Test: Auto-discovery ────────────────────────────────────────────────────


class TestRegistryDiscovery:
    def test_discover_finds_all_plugins(self):
        registry = FrameworkRegistry.get_instance()
        registry.discover()
        slugs = registry.list_slugs()
        expected = {"humanistic", "alliance_rupture", "act", "dbt", "mi", "cbt", "general"}
        for slug in expected:
            assert slug in slugs, f"Missing plugin: {slug}"
        assert len(slugs) >= 7

    def test_discover_returns_plugin_classes(self):
        registry = FrameworkRegistry.get_instance()
        found = registry.discover()
        plugin_ids = {cls.metadata().id for cls in found}
        assert "humanistic" in plugin_ids
        assert "act" in plugin_ids
        assert "cbt" in plugin_ids

    def test_is_applicable_methods_return_bool(self):
        registry = FrameworkRegistry.get_instance()
        registry.discover()
        for cls in registry._plugins.values():
            result = cls.is_applicable({})
            assert isinstance(result, bool)


# ── Test: Abstract base class ───────────────────────────────────────────────


class TestFrameworkPluginABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            FrameworkPlugin()  # type: ignore[abstract]

    def test_subclass_must_implement_abstract_methods(self):
        # In Python 3.12+ the class definition succeeds but instantiation fails
        class IncompletePlugin(FrameworkPlugin):  # type: ignore[abstract]
            pass

        with pytest.raises(TypeError):
            IncompletePlugin()


# ── Test: Repr ──────────────────────────────────────────────────────────────


class TestRegistryRepr:
    def test_repr(self):
        registry = FrameworkRegistry.get_instance()
        registry.discover()
        text = repr(registry)
        assert "FrameworkRegistry" in text
        assert "registered" in text
