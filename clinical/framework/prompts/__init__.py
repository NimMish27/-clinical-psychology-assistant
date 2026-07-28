"""
clinical/framework/prompts/__init__.py — Prompt management system.

Provides a thread-safe PromptManager that loads Jinja2 prompt templates
from YAML files in each framework's ``prompts/`` directory, validates
them, and renders them with provided variables.
"""

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

__all__ = [
    "PromptManager",
    "PromptTemplate",
    "PromptCollection",
    "PromptVariable",
    "render_template",
    "render_prompt_template",
    "validate_yaml_structure",
    "validate_template_syntax",
    "validate_prompt_template",
    "validate_variable_coverage",
    "TemplateValidationError",
]
