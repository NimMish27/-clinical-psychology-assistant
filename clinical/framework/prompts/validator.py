"""
clinical/framework/prompts/validator.py — Template validation logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, BaseLoader, TemplateSyntaxError

from clinical.framework.prompts.models import PromptTemplate, PromptVariable
from app_logging.logger import get_logger

_log = get_logger(__name__)

_env = Environment(loader=BaseLoader(), autoescape=False)


class TemplateValidationError(Exception):
    """Raised when a prompt template fails validation."""

    def __init__(self, message: str, file_path: str | None = None) -> None:
        self.file_path = file_path
        super().__init__(message)


ValidationResult = tuple[bool, list[str]]


def validate_yaml_structure(data: Any, file_path: str = "") -> list[str]:
    """Validate the top-level structure of a parsed YAML template.

    Checks for required fields and correct types.  Returns a list of
    error messages (empty = valid).
    """
    errors: list[str] = []

    if not isinstance(data, dict):
        errors.append(f"Expected a dict at top level, got {type(data).__name__}")
        return errors

    if "version" not in data:
        errors.append("Missing required field: 'version'")
    elif not isinstance(data["version"], str):
        errors.append("'version' must be a string")

    if "templates" not in data:
        errors.append("Missing required field: 'templates'")
    elif not isinstance(data["templates"], dict):
        errors.append("'templates' must be a dict (locale → string)")
    else:
        for locale, content in data["templates"].items():
            if not isinstance(content, str):
                errors.append(f"Content for locale '{locale}' must be a string")
            if not content.strip():
                errors.append(f"Template content for locale '{locale}' is empty")

    if "variables" in data:
        if not isinstance(data["variables"], list):
            errors.append("'variables' must be a list")
        else:
            for i, var in enumerate(data["variables"]):
                if not isinstance(var, dict):
                    errors.append(f"variables[{i}] must be a dict")
                    continue
                if "name" not in var:
                    errors.append(f"variables[{i}] missing required field 'name'")
                elif not isinstance(var["name"], str):
                    errors.append(f"variables[{i}]['name'] must be a string")

    return errors


def validate_template_syntax(template_body: str, name: str = "<unknown>") -> list[str]:
    """Validate that a Jinja2 template body is syntactically valid."""
    errors: list[str] = []
    try:
        _env.parse(template_body)
    except TemplateSyntaxError as exc:
        errors.append(f"Jinja2 syntax error at line {exc.lineno}: {exc.message}")
    return errors


def validate_prompt_template(data: dict, file_path: str = "") -> list[str]:
    """Run all validations on a parsed template."""
    errors: list[str] = []
    errors.extend(validate_yaml_structure(data, file_path))

    templates_dict = data.get("templates", {})
    if isinstance(templates_dict, dict):
        for locale, body in templates_dict.items():
            if isinstance(body, str):
                errors.extend(
                    validate_template_syntax(body, f"{Path(file_path).name}[{locale}]")
                )

    return errors


def validate_variable_coverage(
    template: PromptTemplate,
    variable_names: set[str],
) -> list[str]:
    """Check that all variables used in the template body are declared."""
    errors: list[str] = []
    declared = {v.name for v in template.variables}

    for locale, body in template.templates.items():
        try:
            ast = _env.parse(body)
        except TemplateSyntaxError:
            continue  # already caught by validate_template_syntax

        from jinja2.meta import find_undeclared_variables
        used = find_undeclared_variables(ast)
        undeclared = used - declared
        if undeclared:
            errors.append(
                f"Locale '{locale}': variables used but not declared: "
                f"{', '.join(sorted(undeclared))}"
            )

    return errors
