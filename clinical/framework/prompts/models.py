"""
clinical/framework/prompts/models.py — Prompt template data models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator


class PromptVariable(BaseModel):
    """A single variable accepted by a prompt template."""

    name: str = Field(description="Variable name used in the Jinja2 template")
    description: str = Field(default="", description="Human-readable description")
    required: bool = Field(default=True, description="Whether the variable must be provided")
    default: str | None = Field(default=None, description="Default value if not supplied")


class PromptTemplate(BaseModel):
    """
    A single prompt template loaded from a YAML file.

    Each template corresponds to one lifecycle method
    (e.g. ``analyze_case``, ``intervention_plan``) and stores
    versioned Jinja2 content in one or more locales.
    """

    name: str = Field(description="Template name (filename stem)")
    version: str = Field(description="Semantic version of this template")
    description: str = Field(default="", description="Purpose of this template")
    variables: list[PromptVariable] = Field(default_factory=list, description="Expected variables")
    templates: dict[str, str] = Field(description="Locale → Jinja2 template body")

    framework_slug: str = Field(description="Owning framework slug (e.g. 'cbt')")
    file_path: str = Field(default="", description="Absolute path to the YAML source file")
    loaded_at: float = Field(default=0.0, description="Unix timestamp when loaded")

    @model_validator(mode="after")
    def _ensure_en_fallback(self) -> "PromptTemplate":
        if "en" not in self.templates:
            self.templates["en"] = ""
        return self

    def render(self, variables: dict[str, Any], locale: str = "en") -> str:
        """Render the Jinja2 template with the given variables.

        Falls back to ``en`` if the requested locale is not available.
        """
        from clinical.framework.prompts.renderer import render_template
        return render_template(self, variables, locale)

    def missing_variables(self, variables: dict[str, Any]) -> list[str]:
        """Return names of required variables not present in *variables*."""
        return [v.name for v in self.variables if v.required and v.name not in variables]

    @property
    def locales(self) -> list[str]:
        return list(self.templates.keys())


class PromptCollection(BaseModel):
    """
    Collection of all prompt templates for a single framework.

    Loaded from a ``prompts/`` directory on disk.  Provides
    dict-style access by template name.
    """

    slug: str = Field(description="Framework slug (e.g. 'cbt')")
    templates: dict[str, PromptTemplate] = Field(
        default_factory=dict,
        description="Template name → PromptTemplate",
    )
    loaded_at: float = Field(default=0.0, description="Unix timestamp when collection was loaded")
    file_count: int = Field(default=0, description="Number of YAML files loaded")

    def get(self, name: str, default: Any = None) -> PromptTemplate | None:
        return self.templates.get(name, default)

    def all(self) -> list[PromptTemplate]:
        return list(self.templates.values())

    def names(self) -> list[str]:
        return list(self.templates.keys())

    def has(self, name: str) -> bool:
        return name in self.templates
