from __future__ import annotations

from app_logging.logger import get_logger
from clinical.frameworks.act.plugin import ACTFramework
from clinical.frameworks.base import BaseFramework
from clinical.frameworks.cbt.plugin import CBTFramework
from clinical.frameworks.dbt.plugin import DBTFramework
from clinical.frameworks.humanistic.plugin import HumanisticFramework
from clinical.frameworks.models import FrameworkMetadata

_log = get_logger(__name__)


class FrameworkRegistry:
    """Discovers and manages registered therapeutic framework plugins.

    Frameworks can be registered at import time (by importing the plugin
    module) or explicitly via ``register()``.  The registry provides a
    single source of truth for which frameworks are available to the
    parallel router.

    Usage::

        registry = FrameworkRegistry()
        for fw in registry.get_enabled():
            result = await fw.analyze(understanding, evidence, llm)
    """

    def __init__(self) -> None:
        self._frameworks: dict[str, BaseFramework] = {}

    def register(self, framework: BaseFramework) -> None:
        """Register a framework plugin by its ``.name`` property."""
        if framework.name in self._frameworks:
            _log.warning(
                "framework_registry.duplicate",
                name=framework.name,
            )
        self._frameworks[framework.name] = framework
        _log.info("framework_registry.registered", name=framework.name)

    def get(self, name: str) -> BaseFramework | None:
        """Retrieve a framework by name, or None if not found."""
        return self._frameworks.get(name)

    def get_enabled(
        self,
        enabled_names: list[str] | None = None,
    ) -> list[BaseFramework]:
        """Return all registered frameworks, optionally filtered.

        Args:
            enabled_names: If provided, only return frameworks whose
                           ``.name`` appears in this list.

        Returns:
            List of framework instances (in registration order).
        """
        if enabled_names is not None:
            return [
                fw for name, fw in self._frameworks.items()
                if name in enabled_names
            ]
        return list(self._frameworks.values())

    def get_metadata(self, name: str) -> FrameworkMetadata | None:
        """Return metadata for a registered framework by name, or None."""
        fw = self._frameworks.get(name)
        return fw.metadata if fw else None

    def list_metadata(
        self,
        enabled_names: list[str] | None = None,
    ) -> list[FrameworkMetadata]:
        """Return metadata for all (optionally filtered) frameworks.

        Useful for the evaluation harness, a future frontend, the
        comparison engine, and human-in-the-loop framework selection.
        """
        return [fw.metadata for fw in self.get_enabled(enabled_names)]


# ── Module-level singleton ─────────────────────────────────────

_registry: FrameworkRegistry | None = None


def get_framework_registry() -> FrameworkRegistry:
    """Return the singleton FrameworkRegistry, creating it if needed.

    On first call, automatically registers the built-in frameworks
    (CBT, ACT, DBT, Humanistic).  Additional frameworks can be registered
    later via ``get_framework_registry().register(...)``.
    """
    global _registry
    if _registry is None:
        _registry = FrameworkRegistry()
        _registry.register(CBTFramework())
        _registry.register(ACTFramework())
        _registry.register(DBTFramework())
        _registry.register(HumanisticFramework())
    return _registry
