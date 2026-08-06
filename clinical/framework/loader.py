"""
clinical/framework/loader.py — Dynamic module loading for framework plugins.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import sys
from pathlib import Path
from types import ModuleType

from app_logging.logger import get_logger

_log = get_logger(__name__)


def import_submodules(package_path: str) -> dict[str, ModuleType]:
    """
    Import all submodules under a dotted package path.

    Args:
        package_path: Dotted path to the package (e.g. ``"clinical.framework"``).

    Returns:
        Dict mapping submodule name → imported module.
    """
    package = importlib.import_module(package_path)
    results: dict[str, ModuleType] = {}

    for _, name, is_pkg in pkgutil.walk_packages(
        package.__path__,  # type: ignore[arg-type]
        prefix=f"{package_path}.",
    ):
        try:
            if is_pkg:
                results[name] = importlib.import_module(name)
            else:
                results[name] = importlib.import_module(name)
        except Exception as exc:
            _log.warning("loader.import_failed", module=name, error=str(exc))

    return results


def find_plugin_classes(
    base_package: str = "clinical.framework",
    plugin_submodule: str = "plugin",
) -> list[type]:
    """
    Scan all sub-packages of *base_package* for a module named *plugin_submodule*
    and collect all classes defined within that inherit from ``FrameworkPlugin``.

    This is the core auto-discovery mechanism used by the registry.

    Args:
        base_package: The parent package to scan.
        plugin_submodule: Name of the module to look for in each sub-package.

    Returns:
        List of plugin classes (not instances).
    """
    from clinical.framework.base import FrameworkPlugin

    discovered: list[type] = []
    modules = import_submodules(base_package)

    for mod_name, mod in modules.items():
        if not mod_name.endswith(f".{plugin_submodule}"):
            continue
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, FrameworkPlugin)
                and attr is not FrameworkPlugin
                and not inspect.isabstract(attr)
            ):
                _log.info(
                    "loader.discovered_plugin",
                    module=mod_name,
                    plugin=attr.__name__,
                )
                discovered.append(attr)

    return discovered


def load_plugin_class(module_path: str, class_name: str) -> type:
    """
    Import a specific plugin class by its fully qualified path.

    Args:
        module_path: Dotted module path (e.g. ``"clinical.framework.act.plugin"``).
        class_name: Name of the plugin class (e.g. ``"ACTPlugin"``).

    Returns:
        The plugin class.

    Raises:
        ImportError: If the module or class cannot be loaded.
    """
    module = importlib.import_module(module_path)
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(
            f"Class '{class_name}' not found in module '{module_path}'",
        ) from exc
