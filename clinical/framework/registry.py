"""
clinical/framework/registry.py — FrameworkRegistry singleton.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from app_logging.logger import get_logger

if TYPE_CHECKING:
    from clinical.framework.base import FrameworkPlugin
    from clinical.framework.models import FrameworkMetadata

_log = get_logger(__name__)


class FrameworkRegistryError(Exception):
    """Base exception for registry operations."""


class DuplicateFrameworkError(FrameworkRegistryError):
    """Raised when a framework slug is registered more than once."""


class FrameworkNotFoundError(FrameworkRegistryError):
    """Raised when querying a slug that does not exist in the registry."""


class FrameworkDisabledError(FrameworkRegistryError):
    """Raised when attempting to use a disabled framework."""


def register_framework(plugin_cls: type[FrameworkPlugin]):
    """
    Decorator that automatically registers a framework plugin class.

    Usage::

        @register_framework
        class ACTPlugin(FrameworkPlugin):
            ...
    """
    FrameworkRegistry.get_instance().register(plugin_cls)
    return plugin_cls


class FrameworkRegistry:
    """
    Singleton responsible for discovering, registering, and managing
    therapeutic framework plugins.

    Responsibilities:
        * Auto-discovery of plugins via package scanning
        * Explicit registration (decorator or direct call)
        * Enable / disable per-plugin at runtime
        * Metadata queries and filtering
        * Dynamic loading for lazy instantiation

    Thread‑safe: all mutations are protected by a re-entrant lock.
    """

    _instance: FrameworkRegistry | None = None
    _lock = threading.RLock()

    def __init__(self) -> None:
        if not getattr(self, "_initialised", False):
            self._plugins: dict[str, type[FrameworkPlugin]] = {}
            self._enabled: dict[str, bool] = {}
            self._discovered = False
            self._initialised = True

    # ── Singleton access ────────────────────────────────────────────────────

    @classmethod
    def get_instance(cls) -> FrameworkRegistry:
        """Return the process-wide FrameworkRegistry singleton."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ── Discovery ────────────────────────────────────────────────────────────

    def discover(self) -> list[type[FrameworkPlugin]]:
        """
        Scan all sub-packages of ``clinical.framework`` for ``plugin.py``
        modules containing ``FrameworkPlugin`` subclasses.

        Called automatically on first access but safe to call multiple times.
        """
        with self._lock:
            if self._discovered:
                return list(self._plugins.values())

            from clinical.framework.loader import find_plugin_classes

            found = find_plugin_classes()
            count_before = len(self._plugins)
            for cls in found:
                self._register(cls)
            self._discovered = True

            _log.info(
                "registry.discovery_complete",
                discovered=len(found),
                new=len(self._plugins) - count_before,
                total=len(self._plugins),
            )
            return found

    # ── Registration ────────────────────────────────────────────────────────

    def register(self, plugin_cls: type[FrameworkPlugin]) -> None:
        """Register a single plugin class. Idempotent — re-registration of the
        same slug is a no-op if the class is identical, otherwise raises."""
        with self._lock:
            self._register(plugin_cls)

    def _register(self, plugin_cls: type[FrameworkPlugin]) -> None:
        meta = plugin_cls.metadata()
        slug = meta.id

        existing = self._plugins.get(slug)
        if existing is plugin_cls:
            return  # re-registration of same class → no-op

        if existing is not None:
            raise DuplicateFrameworkError(
                f"Framework slug '{slug}' is already registered by "
                f"'{existing.__module__}.{existing.__qualname__}'. "
                f"Cannot register '{plugin_cls.__module__}.{plugin_cls.__qualname__}'.",
            )

        self._plugins[slug] = plugin_cls
        self._enabled.setdefault(slug, True)

        _log.info(
            "registry.registered",
            slug=slug,
            name=meta.display_name,
            module=plugin_cls.__module__,
        )

    # ── Accessors ───────────────────────────────────────────────────────────

    def get(self, slug: str) -> type[FrameworkPlugin]:
        """
        Return the plugin class for the given slug.

        Raises:
            FrameworkNotFoundError: If the slug is not registered.
        """
        self.discover()
        cls = self._plugins.get(slug)
        if cls is None:
            raise FrameworkNotFoundError(
                f"Framework '{slug}' is not registered. "
                f"Available: {', '.join(sorted(self._plugins))}",
            )
        return cls

    def get_metadata(self, slug: str) -> FrameworkMetadata:
        """Return the metadata for a registered framework."""
        return self.get(slug).metadata()

    def list_metadata(self) -> list[FrameworkMetadata]:
        """
        Return metadata for all registered frameworks, ordered by slug.
        """
        self.discover()
        return [cls.metadata() for cls in self._plugins.values()]

    def list_enabled_metadata(self) -> list[FrameworkMetadata]:
        """Return metadata for all enabled frameworks."""
        self.discover()
        return [
            cls.metadata()
            for slug, cls in self._plugins.items()
            if self._enabled.get(slug, True)
        ]

    def list_slugs(self) -> list[str]:
        """Return sorted list of registered slug identifiers."""
        self.discover()
        return sorted(self._plugins)

    # ── Enable / Disable ────────────────────────────────────────────────────

    def enable(self, slug: str) -> None:
        """Enable a registered framework."""
        self.get(slug)  # raises if not found
        with self._lock:
            self._enabled[slug] = True
        _log.info("registry.enabled", slug=slug)

    def disable(self, slug: str) -> None:
        """Disable a registered framework."""
        self.get(slug)
        with self._lock:
            self._enabled[slug] = False
        _log.info("registry.disabled", slug=slug)

    def is_enabled(self, slug: str) -> bool:
        """Check whether a framework is enabled."""
        self.get(slug)
        return self._enabled.get(slug, True)

    # ── Applicability ──────────────────────────────────────────────────────

    def applicable_for(
        self,
        case_understanding: dict,
        *,
        only_enabled: bool = True,
    ) -> list[tuple[str, type[FrameworkPlugin]]]:
        """
        Return ``(slug, plugin_class)`` pairs for frameworks that report
        themselves as applicable to the given case understanding.

        Args:
            case_understanding: Flat dict from ``CaseUnderstanding.to_flat_dict()``.
            only_enabled: If True (default), skip disabled frameworks.

        Returns:
            List of ``(slug, plugin_class)`` tuples.
        """
        self.discover()
        result: list[tuple[str, type[FrameworkPlugin]]] = []
        for slug, cls in self._plugins.items():
            if only_enabled and not self.is_enabled(slug):
                continue
            try:
                if cls.is_applicable(case_understanding):
                    result.append((slug, cls))
            except Exception as exc:
                _log.warning(
                    "registry.applicability_failed",
                    slug=slug,
                    error=str(exc),
                )
        return result

    # ── Introspection ───────────────────────────────────────────────────────

    def __len__(self) -> int:
        self.discover()
        return len(self._plugins)

    def __contains__(self, slug: str) -> bool:
        self.discover()
        return slug in self._plugins

    def __repr__(self) -> str:
        self.discover()
        enabled = sum(1 for s in self._plugins if self._enabled.get(s, True))
        return (
            f"FrameworkRegistry({len(self._plugins)} registered, "
            f"{enabled} enabled)"
        )
