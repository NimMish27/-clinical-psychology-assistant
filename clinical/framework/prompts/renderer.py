"""
clinical/framework/prompts/renderer.py — Jinja2 template rendering.
"""

from __future__ import annotations

from typing import Any

from jinja2 import Environment, BaseLoader, TemplateNotFound, TemplateSyntaxError, Undefined, UndefinedError

from app_logging.logger import get_logger

_log = get_logger(__name__)

_UNDEFINED_PLACEHOLDER = "__UNDEFINED__"


class _LenientUndefined(Undefined):
    """
    When ``PROMPT_STRICT=1`` is set, raises on undefined variables
    (like ``jinja2.StrictUndefined``).  Otherwise silently returns a
    placeholder so templates never crash at runtime.
    """

    def __str__(self) -> str:
        import os
        if os.getenv("PROMPT_STRICT", "").lower() in ("1", "true", "yes"):
            raise UndefinedError(
                f"Template variable '{self._undefined_name}' is undefined",
            )
        return _UNDEFINED_PLACEHOLDER


_env = Environment(
    loader=BaseLoader(),
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=False,
    undefined=_LenientUndefined,
)


def render_template(
    template_source: str,
    variables: dict[str, Any],
) -> str:
    """Render a Jinja2 template string with the given variables.

    Args:
        template_source: Raw Jinja2 template string.
        variables: Dict of variable names to values.

    Returns:
        Rendered string (``__UNDEFINED__`` placeholders for missing variables
        unless ``PROMPT_STRICT`` is set).

    Raises:
        TemplateSyntaxError: If the template has syntax errors.
    """
    tpl = _env.from_string(template_source)
    rendered = tpl.render(**variables)
    return rendered


def render_prompt_template(
    template: "PromptTemplate",  # noqa: F821 — forward ref, imported at call site
    variables: dict[str, Any],
    locale: str = "en",
) -> str:
    """Render a ``PromptTemplate`` object for the given locale.

    Falls back to ``en`` if the requested locale is not available.
    """
    from clinical.framework.prompts.models import PromptTemplate

    if not isinstance(template, PromptTemplate):
        raise TypeError(f"Expected PromptTemplate, got {type(template).__name__}")

    body = template.templates.get(locale) or template.templates.get("en")
    if not body:
        raise ValueError(
            f"Template '{template.name}' has no content for locale "
            f"'{locale}' and no 'en' fallback"
        )

    missing = template.missing_variables(variables)
    if missing:
        _log.warning(
            "prompt.missing_variables",
            template=template.name,
            framework=template.framework_slug,
            missing=missing,
        )

    return render_template(body, variables)
