"""
clinical/framework/prompts/manager.py — Central prompt template manager.

Provides a thread-safe singleton per framework slug that discovers,
loads, caches, validates, and renders Jinja2 prompt templates from
each framework's ``prompts/`` directory.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from threading import RLock
from typing import Any, ClassVar
from collections import OrderedDict

import yaml

from clinical.framework.prompts.models import PromptTemplate, PromptCollection, PromptVariable
from clinical.framework.prompts.renderer import render_prompt_template
from clinical.framework.prompts.validator import (
    validate_prompt_template,
    TemplateValidationError,
    ValidationResult,
)
from app_logging.logger import get_logger

_log = get_logger(__name__)

_LIFECYCLE_METHODS: tuple[str, ...] = (
    "analyze_case",
    "retrieve_evidence",
    "case_conceptualization",
    "therapeutic_focus",
    "intervention_plan",
    "session_objectives",
    "followup_questions",
    "limitations",
)

_LIFECYCLE_NAMES: dict[str, str] = {
    "analyze_case": "Case analysis",
    "retrieve_evidence": "Evidence retrieval",
    "case_conceptualization": "Case conceptualisation",
    "therapeutic_focus": "Therapeutic focus",
    "intervention_plan": "Intervention plan",
    "session_objectives": "Session objectives",
    "followup_questions": "Follow-up questions",
    "limitations": "Limitations & contraindications",
}


class PromptManager:
    """
    Thread-safe manager for loading, caching, and rendering prompt templates.

    Usage::

        mgr = PromptManager.for_framework("cbt")
        template = mgr.get_template("analyze_case")
        rendered = mgr.render("analyze_case", variables={"system_header": ...})

    Templates are cached in memory after the first load.  Use
    ``reload()`` to force a re-read from disk.
    """

    _instances: ClassVar[dict[str, "PromptManager"]] = {}
    _lock: ClassVar[RLock] = RLock()
    _BASE_DIR: ClassVar[Path | None] = None

    def __init__(self, base_dir: Path) -> None:
        """Create a manager for frameworks under *base_dir*.

        Normally created via ``for_framework()``, not directly.
        """
        self._base_dir: Path = base_dir.resolve()
        self._collections: dict[str, PromptCollection] = OrderedDict()
        self._cache_lock: RLock = RLock()

    # ── Factory ────────────────────────────────────────────────────────────

    @classmethod
    def for_framework(cls, slug: str) -> "PromptManager":
        """Return (or create) the manager for a single framework.

        The prompts directory is resolved relative to the
        framework's own package directory
        (``clinical/framework/{slug}/prompts/``).
        """
        root = cls._resolve_base_dir()
        with cls._lock:
            if slug not in cls._instances:
                cls._instances[slug] = cls(root)
            return cls._instances[slug]

    @classmethod
    def global_manager(cls) -> "PromptManager":
        """Return a manager that sees all frameworks under the base directory."""
        root = cls._resolve_base_dir()
        with cls._lock:
            if "*" not in cls._instances:
                cls._instances["*"] = cls(root)
            return cls._instances["*"]

    @classmethod
    def _resolve_base_dir(cls) -> Path:
        if cls._BASE_DIR is not None:
            return cls._BASE_DIR.resolve()
        # Default: clinical/framework/  (parent of individual plugin dirs)
        return Path(__file__).resolve().parent.parent

    @classmethod
    def set_base_dir(cls, path: str | Path) -> None:
        """Override the base directory (useful in tests)."""
        cls._BASE_DIR = Path(path).resolve()

    @classmethod
    def reset(cls) -> None:
        """Clear all cached instances (useful in tests)."""
        with cls._lock:
            cls._instances.clear()
            cls._BASE_DIR = None

    # ── Discovery ──────────────────────────────────────────────────────────

    @staticmethod
    def _is_framework_dir(path: Path) -> bool:
        """Check if a directory contains a ``plugin.py`` and a ``prompts/`` dir."""
        return (
            path.is_dir()
            and (path / "plugin.py").exists()
            and (path / "prompts").is_dir()
        )

    def discover_framework_slugs(self) -> list[str]:
        """Return slugs for all frameworks that have a ``prompts/`` directory."""
        slugs: list[str] = []
        for entry in sorted(self._base_dir.iterdir()):
            if self._is_framework_dir(entry):
                slugs.append(entry.name)
        return slugs

    # ── Loading ────────────────────────────────────────────────────────────

    def load_framework(self, slug: str) -> PromptCollection:
        """Load all YAML templates from ``{slug}/prompts/``.

        Results are cached.  Use ``reload(slug)`` to force a re-read.
        """
        with self._cache_lock:
            if slug in self._collections:
                return self._collections[slug]

            prompts_dir = self._base_dir / slug / "prompts"
            if not prompts_dir.is_dir():
                _log.warning("prompt.no_prompts_dir", framework=slug, dir=str(prompts_dir))
                col = PromptCollection(slug=slug)
                self._collections[slug] = col
                return col

            yaml_files = sorted(prompts_dir.glob("*.yaml"))
            templates: dict[str, PromptTemplate] = {}

            for fpath in yaml_files:
                tpl = self._load_single_file(slug, fpath)
                if tpl is not None:
                    templates[tpl.name] = tpl

            col = PromptCollection(
                slug=slug,
                templates=templates,
                loaded_at=time.time(),
                file_count=len(yaml_files),
            )
            self._collections[slug] = col

            available = list(templates.keys())
            missing = [m for m in _LIFECYCLE_METHODS if m not in templates]
            if missing:
                _log.info(
                    "prompt.partial_coverage",
                    framework=slug,
                    available=available,
                    missing=missing,
                )
            else:
                _log.info(
                    "prompt.full_coverage",
                    framework=slug,
                    count=len(templates),
                    methods=available,
                )

            return col

    def _load_single_file(
        self,
        slug: str,
        fpath: Path,
    ) -> PromptTemplate | None:
        """Load and validate a single YAML template file."""
        name = fpath.stem
        try:
            raw = fpath.read_text(encoding="utf-8")
            data = yaml.safe_load(raw)

            if not isinstance(data, dict):
                _log.error("prompt.invalid_yaml", file=str(fpath), framework=slug)
                return None

            errors = validate_prompt_template(data, str(fpath))
            if errors:
                for err in errors:
                    _log.error("prompt.validation_error", file=str(fpath), error=err)
                return None

            variables_list = data.get("variables", [])
            variables = [
                PromptVariable(**v) if isinstance(v, dict) else PromptVariable(name=str(v))
                for v in variables_list
            ]

            return PromptTemplate(
                name=name,
                version=data["version"],
                description=data.get("description", ""),
                variables=variables,
                templates=data["templates"],
                framework_slug=slug,
                file_path=str(fpath.resolve()),
                loaded_at=time.time(),
            )

        except yaml.YAMLError as exc:
            _log.error("prompt.yaml_parse_error", file=str(fpath), error=str(exc))
            return None
        except Exception as exc:
            _log.error("prompt.load_error", file=str(fpath), error=str(exc))
            return None

    # ── Access ─────────────────────────────────────────────────────────────

    def get_collection(self, slug: str) -> PromptCollection:
        """Return the cached collection (loads if needed)."""
        return self.load_framework(slug)

    def get_template(self, slug: str, template_name: str) -> PromptTemplate | None:
        """Get a single template by framework slug and template name."""
        col = self.load_framework(slug)
        return col.get(template_name)

    def has_template(self, slug: str, template_name: str) -> bool:
        """Check if a template exists for the given framework and name."""
        col = self.load_framework(slug)
        return col.has(template_name)

    def list_templates(self, slug: str) -> list[str]:
        """Return names of all loaded templates for a framework."""
        col = self.load_framework(slug)
        return col.names()

    def list_frameworks(self) -> list[str]:
        """Return slugs of all frameworks that have prompts loaded."""
        with self._cache_lock:
            return list(self._collections.keys())

    # ── Rendering ──────────────────────────────────────────────────────────

    def render(
        self,
        slug: str,
        template_name: str,
        variables: dict[str, Any],
        locale: str = "en",
    ) -> str | None:
        """Render a prompt template for the given framework.

        Args:
            slug: Framework slug (e.g. ``"cbt"``).
            template_name: Template filename stem (e.g. ``"analyze_case"``).
            variables: Variables to inject into the Jinja2 template.
            locale: Target locale (falls back to ``en``).

        Returns:
            Rendered string, or ``None`` if the template does not exist.
        """
        tpl = self.get_template(slug, template_name)
        if tpl is None:
            return None

        try:
            return render_prompt_template(tpl, variables, locale)
        except Exception as exc:
            _log.error(
                "prompt.render_error",
                framework=slug,
                template=template_name,
                error=str(exc),
            )
            return None

    # ── Maintenance ────────────────────────────────────────────────────────

    def reload(self, slug: str | None = None) -> None:
        """Force re-load of templates from disk.

        Args:
            slug: Reload a specific framework, or ``None`` to reload all.
        """
        with self._cache_lock:
            if slug is not None:
                self._collections.pop(slug, None)
            else:
                self._collections.clear()

        if slug is not None:
            self.load_framework(slug)
        else:
            for found_slug in self.discover_framework_slugs():
                self.load_framework(found_slug)

    def validate_all(self, slug: str | None = None) -> dict[str, list[str]]:
        """Validate all templates for a framework (or all frameworks).

        Returns:
            Dict mapping ``"framework_slug/template_name"`` to a list of
            validation error strings.  Empty dict means all valid.
        """
        results: dict[str, list[str]] = {}
        slugs = [slug] if slug else self.discover_framework_slugs()
        for s in slugs:
            col = self.load_framework(s)
            for name, tpl in col.templates.items():
                import yaml
                try:
                    raw_data = yaml.safe_load(Path(tpl.file_path).read_text(encoding="utf-8"))
                except Exception as exc:
                    results[f"{s}/{name}"] = [f"Cannot re-read: {exc}"]
                    continue
                from clinical.framework.prompts.validator import (
                    validate_prompt_template,
                    validate_variable_coverage,
                )
                errors = validate_prompt_template(raw_data, tpl.file_path)
                variable_names = {v.name for v in tpl.variables}
                errors.extend(validate_variable_coverage(tpl, variable_names))
                if errors:
                    results[f"{s}/{name}"] = errors
        return results

    def stats(self, slug: str | None = None) -> dict[str, Any]:
        """Return statistics about loaded templates."""
        slugs = [slug] if slug else self.list_frameworks()
        total_files = 0
        total_size = 0
        framework_counts: dict[str, int] = {}

        for s in slugs:
            col = self.get_collection(s)
            count = len(col.templates)
            framework_counts[s] = count
            total_files += count
            for tpl in col.templates.values():
                for locale, body in tpl.templates.items():
                    total_size += len(body.encode("utf-8"))

        return {
            "frameworks": len(slugs),
            "total_templates": total_files,
            "total_bytes": total_size,
            "per_framework": framework_counts,
        }
