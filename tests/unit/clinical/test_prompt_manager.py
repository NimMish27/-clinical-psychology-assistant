"""
tests/unit/clinical/test_prompt_manager.py — Tests for the Prompt Management System.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from clinical.framework.prompts.models import PromptTemplate, PromptCollection, PromptVariable
from clinical.framework.prompts.manager import PromptManager
from clinical.framework.prompts.renderer import render_template, render_prompt_template
from clinical.framework.prompts.validator import (
    validate_yaml_structure,
    validate_template_syntax,
    validate_prompt_template,
    validate_variable_coverage,
    TemplateValidationError,
)


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def tmp_prompts_dir(tmp_path: Path) -> Path:
    """Create a temporary prompts directory with sample templates."""
    fw_dir = tmp_path / "test_fw"
    fw_dir.mkdir(parents=True, exist_ok=True)
    (fw_dir / "plugin.py").write_text("# plugin", encoding="utf-8")
    prompts = fw_dir / "prompts"
    prompts.mkdir(parents=True, exist_ok=True)

    analyze = prompts / "analyze_case.yaml"
    analyze.write_text("""\
version: "1.0.0"
description: "Test — analyse a clinical case"
variables:
  - name: system_header
    description: "System header"
    required: true
  - name: case_text
    description: "Case data"
    required: true
templates:
  en: |
    {{ system_header }}

    As a test practitioner, analyse this case.

    {{ case_text }}

    Return JSON.
""", encoding="utf-8")

    plan = prompts / "intervention_plan.yaml"
    plan.write_text("""\
version: "2.0.0"
description: "Test — intervention plan"
variables:
  - name: system_header
    description: "System header"
    required: true
  - name: case_text
    description: "Case data"
    required: false
templates:
  en: |
    {{ system_header }}
    Plan: {{ case_text }}
  es: |
    {{ system_header }}
    Plan: {{ case_text }}
""", encoding="utf-8")

    return tmp_path


@pytest.fixture
def reset_manager() -> None:
    """Reset PromptManager singleton state before/after tests."""
    PromptManager.reset()
    yield
    PromptManager.reset()


# ═══════════════════════════════════════════════════════════════════════
# PromptTemplate model
# ═══════════════════════════════════════════════════════════════════════

class TestPromptTemplate:
    def test_basic_creation(self) -> None:
        tpl = PromptTemplate(
            name="analyze_case",
            version="1.0.0",
            description="Test",
            templates={"en": "Hello {{ name }}"},
            framework_slug="test",
        )
        assert tpl.name == "analyze_case"
        assert tpl.version == "1.0.0"
        assert tpl.templates["en"] == "Hello {{ name }}"

    def test_locales_property(self) -> None:
        tpl = PromptTemplate(
            name="test",
            version="1.0.0",
            templates={"en": "a", "es": "b", "fr": "c"},
            framework_slug="test",
        )
        assert sorted(tpl.locales) == sorted(["en", "es", "fr"])

    def test_en_fallback_on_missing(self) -> None:
        """If no 'en' in templates, the model_validator adds an empty one."""
        tpl = PromptTemplate(
            name="test",
            version="1.0.0",
            templates={"es": "Hola"},
            framework_slug="test",
        )
        assert "en" in tpl.templates
        assert tpl.templates["en"] == ""

    def test_missing_variables(self) -> None:
        tpl = PromptTemplate(
            name="test",
            version="1.0.0",
            variables=[
                PromptVariable(name="name", required=True),
                PromptVariable(name="age", required=True),
                PromptVariable(name="optional_field", required=False),
            ],
            templates={"en": "{{ name }} is {{ age }}"},
            framework_slug="test",
        )
        assert tpl.missing_variables({"name": "Alice"}) == ["age"]
        assert tpl.missing_variables({"name": "Alice", "age": "30"}) == []


# ═══════════════════════════════════════════════════════════════════════
# PromptCollection model
# ═══════════════════════════════════════════════════════════════════════

class TestPromptCollection:
    def test_empty_collection(self) -> None:
        col = PromptCollection(slug="test")
        assert col.names() == []
        assert col.get("missing") is None
        assert col.has("anything") is False

    def test_with_templates(self) -> None:
        t1 = PromptTemplate(name="a", version="1.0", templates={"en": ""}, framework_slug="test")
        t2 = PromptTemplate(name="b", version="1.0", templates={"en": ""}, framework_slug="test")
        col = PromptCollection(slug="test", templates={"a": t1, "b": t2})
        assert col.names() == ["a", "b"]
        assert col.get("a") is t1
        assert col.has("b") is True
        assert len(col.all()) == 2


# ═══════════════════════════════════════════════════════════════════════
# Renderer
# ═══════════════════════════════════════════════════════════════════════

class TestRenderer:
    def test_render_simple(self) -> None:
        result = render_template("Hello {{ name }}!", {"name": "World"})
        assert result == "Hello World!"

    def test_render_with_multiline(self) -> None:
        tpl = "Line1\n{{ var }}\nLine3"
        result = render_template(tpl, {"var": "Line2"})
        assert result == "Line1\nLine2\nLine3"

    def test_render_missing_variable(self) -> None:
        """Missing variables produce a placeholder when PROMPT_STRICT is off."""
        result = render_template("Hello {{ name }}!", {})
        assert "__UNDEFINED__" in result

    def test_render_via_prompt_template(self) -> None:
        tpl = PromptTemplate(
            name="greeting",
            version="1.0",
            templates={"en": "Hi {{ name }}!", "es": "Hola {{ name }}!"},
            framework_slug="test",
            variables=[PromptVariable(name="name", required=True)],
        )
        result = render_prompt_template(tpl, {"name": "Alice"}, locale="en")
        assert result == "Hi Alice!"

        result_es = render_prompt_template(tpl, {"name": "Alice"}, locale="es")
        assert result_es == "Hola Alice!"

    def test_render_fallback_to_en(self) -> None:
        tpl = PromptTemplate(
            name="test",
            version="1.0",
            templates={"en": "Hello {{ name }}!"},
            framework_slug="test",
        )
        result = render_prompt_template(tpl, {"name": "Bob"}, locale="fr")
        assert result == "Hello Bob!"

    def test_render_no_content_raises(self) -> None:
        tpl = PromptTemplate(
            name="empty",
            version="1.0",
            templates={},
            framework_slug="test",
        )
        with pytest.raises(ValueError, match="no content"):
            render_prompt_template(tpl, {}, locale="en")


# ═══════════════════════════════════════════════════════════════════════
# Validator
# ═══════════════════════════════════════════════════════════════════════

class TestValidator:
    def test_validate_valid_yaml(self) -> None:
        data = {
            "version": "1.0.0",
            "description": "Test",
            "templates": {"en": "Hello {{ name }}"},
            "variables": [{"name": "name", "required": True}],
        }
        assert validate_yaml_structure(data) == []

    def test_validate_missing_version(self) -> None:
        data = {"templates": {"en": "Hello"}}
        errors = validate_yaml_structure(data)
        assert any("version" in e for e in errors)

    def test_validate_missing_templates(self) -> None:
        data = {"version": "1.0.0"}
        errors = validate_yaml_structure(data)
        assert any("templates" in e for e in errors)

    def test_validate_empty_template_content(self) -> None:
        data = {"version": "1.0.0", "templates": {"en": ""}}
        errors = validate_yaml_structure(data)
        assert any("empty" in e for e in errors)

    def test_validate_template_syntax_valid(self) -> None:
        assert validate_template_syntax("Hello {{ name }}!") == []

    def test_validate_template_syntax_invalid(self) -> None:
        errors = validate_template_syntax("Hello {{ name }!")
        assert len(errors) > 0

    def test_validate_prompt_template_full(self) -> None:
        data = {
            "version": "1.0.0",
            "templates": {"en": "Hello {{ name }}"},
        }
        assert validate_prompt_template(data) == []

    def test_validate_variable_coverage(self) -> None:
        """Test that variables used in template body are declared."""
        tpl = PromptTemplate(
            name="test",
            version="1.0",
            templates={"en": "{{ declared_var }} and {{ undeclared_var }}"},
            framework_slug="test",
            variables=[PromptVariable(name="declared_var", required=True)],
        )
        errors = validate_variable_coverage(tpl, {"declared_var"})
        assert any("undeclared" in e for e in errors)
        assert "undeclared_var" in errors[0]


# ═══════════════════════════════════════════════════════════════════════
# PromptManager
# ═══════════════════════════════════════════════════════════════════════

class TestPromptManager:
    def test_for_framework_creates_instance(self, reset_manager) -> None:
        mgr = PromptManager.for_framework("cbt")
        assert mgr is not None
        assert isinstance(mgr, PromptManager)

    def test_for_framework_is_singleton(self, reset_manager) -> None:
        mgr1 = PromptManager.for_framework("cbt")
        mgr2 = PromptManager.for_framework("cbt")
        assert mgr1 is mgr2

    def test_different_frameworks_different_managers(self, reset_manager) -> None:
        mgr1 = PromptManager.for_framework("cbt")
        mgr2 = PromptManager.for_framework("act")
        assert mgr1 is not mgr2

    def test_load_framework_from_temp_dir(self, reset_manager, tmp_prompts_dir) -> None:
        PromptManager.set_base_dir(tmp_prompts_dir)
        mgr = PromptManager.for_framework("test_fw")
        col = mgr.load_framework("test_fw")
        assert col.slug == "test_fw"
        assert col.has("analyze_case")
        assert col.has("intervention_plan")
        assert col.file_count == 2

    def test_load_framework_no_prompts_dir(self, reset_manager) -> None:
        """Framework without prompts/ dir returns empty collection."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "empty_fw" / "plugin.py").parent.mkdir(parents=True, exist_ok=True)
            (base / "empty_fw" / "plugin.py").write_text("# plugin")
            PromptManager.set_base_dir(base)
            mgr = PromptManager.for_framework("empty_fw")
            col = mgr.load_framework("empty_fw")
            assert col.file_count == 0
            assert col.names() == []

    def test_get_template(self, reset_manager, tmp_prompts_dir) -> None:
        PromptManager.set_base_dir(tmp_prompts_dir)
        mgr = PromptManager.for_framework("test_fw")
        tpl = mgr.get_template("test_fw", "analyze_case")
        assert tpl is not None
        assert tpl.name == "analyze_case"
        assert tpl.version == "1.0.0"
        assert tpl.framework_slug == "test_fw"

    def test_get_template_missing(self, reset_manager, tmp_prompts_dir) -> None:
        PromptManager.set_base_dir(tmp_prompts_dir)
        mgr = PromptManager.for_framework("test_fw")
        assert mgr.get_template("test_fw", "nonexistent") is None

    def test_has_template(self, reset_manager, tmp_prompts_dir) -> None:
        PromptManager.set_base_dir(tmp_prompts_dir)
        mgr = PromptManager.for_framework("test_fw")
        assert mgr.has_template("test_fw", "analyze_case") is True
        assert mgr.has_template("test_fw", "missing") is False

    def test_list_templates(self, reset_manager, tmp_prompts_dir) -> None:
        PromptManager.set_base_dir(tmp_prompts_dir)
        mgr = PromptManager.for_framework("test_fw")
        names = mgr.list_templates("test_fw")
        assert "analyze_case" in names
        assert "intervention_plan" in names
        assert len(names) == 2

    def test_render_success(self, reset_manager, tmp_prompts_dir) -> None:
        PromptManager.set_base_dir(tmp_prompts_dir)
        mgr = PromptManager.for_framework("test_fw")
        result = mgr.render("test_fw", "analyze_case", {
            "system_header": "SYSTEM",
            "case_text": "CASE DATA",
        })
        assert result is not None
        assert "SYSTEM" in result
        assert "CASE DATA" in result

    def test_render_missing_template(self, reset_manager, tmp_prompts_dir) -> None:
        PromptManager.set_base_dir(tmp_prompts_dir)
        mgr = PromptManager.for_framework("test_fw")
        result = mgr.render("test_fw", "nonexistent", {"x": "y"})
        assert result is None

    def test_render_different_locale(self, reset_manager, tmp_prompts_dir) -> None:
        PromptManager.set_base_dir(tmp_prompts_dir)
        mgr = PromptManager.for_framework("test_fw")
        result = mgr.render("test_fw", "intervention_plan", {
            "system_header": "SYS",
            "case_text": "DATA",
        }, locale="es")
        assert result is not None
        assert "SYS" in result

    def test_reload(self, reset_manager, tmp_prompts_dir) -> None:
        PromptManager.set_base_dir(tmp_prompts_dir)
        mgr = PromptManager.for_framework("test_fw")
        col1 = mgr.load_framework("test_fw")
        assert col1.file_count == 2

        # Add a new template file
        new_tpl = tmp_prompts_dir / "test_fw" / "prompts" / "new.yaml"
        new_tpl.write_text("version: '1.0'\ntemplates:\n  en: 'New: {{ x }}'\n", encoding="utf-8")

        mgr.reload("test_fw")
        col2 = mgr.load_framework("test_fw")
        assert col2.file_count == 3
        assert col2.has("new")

    def test_validate_all_valid(self, reset_manager, tmp_prompts_dir) -> None:
        PromptManager.set_base_dir(tmp_prompts_dir)
        mgr = PromptManager.for_framework("test_fw")
        mgr.load_framework("test_fw")
        errors = mgr.validate_all("test_fw")
        assert errors == {}

    def test_discover_framework_slugs(self, reset_manager, tmp_prompts_dir) -> None:
        PromptManager.set_base_dir(tmp_prompts_dir)
        mgr = PromptManager.for_framework("test_fw")
        slugs = mgr.discover_framework_slugs()
        assert "test_fw" in slugs

    @pytest.mark.parametrize("method", [
        "analyze_case", "retrieve_evidence", "case_conceptualization",
        "therapeutic_focus", "intervention_plan", "session_objectives",
        "followup_questions", "limitations",
    ])
    def test_template_exists_for_all_methods(self, method) -> None:
        """Verify that all 8 lifecycle templates exist for each framework."""
        PromptManager.reset()
        mgr = PromptManager.global_manager()
        for slug in ["act", "cbt", "dbt", "mi", "humanistic", "alliance_rupture", "general"]:
            tpl = mgr.get_template(slug, method)
            assert tpl is not None, f"Missing template '{method}' for framework '{slug}'"
            assert tpl.version >= "1.0.0", f"Template '{method}' for '{slug}' has old version {tpl.version}"

    def test_stats(self, reset_manager, tmp_prompts_dir) -> None:
        PromptManager.set_base_dir(tmp_prompts_dir)
        mgr = PromptManager.for_framework("test_fw")
        mgr.load_framework("test_fw")
        stats = mgr.stats("test_fw")
        assert stats["total_templates"] == 2
        assert stats["per_framework"]["test_fw"] == 2

    def test_load_single_file_invalid_yaml(self, reset_manager, tmp_prompts_dir) -> None:
        """Invalid YAML should be skipped gracefully."""
        bad_file = tmp_prompts_dir / "test_fw" / "prompts" / "bad.yaml"
        bad_file.write_text("{invalid: yaml: unclosed", encoding="utf-8")
        PromptManager.set_base_dir(tmp_prompts_dir)
        mgr = PromptManager.for_framework("test_fw")
        col = mgr.load_framework("test_fw")
        # bad file should be skipped, valid files still loaded
        assert col.file_count == 3  # 2 original + 1 bad (counted)
        assert col.has("analyze_case")
        assert col.has("intervention_plan")
        assert not col.has("bad")
