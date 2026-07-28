"""
api/routers/frameworks.py — Framework metadata API endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter

from app_logging.logger import get_logger

_log = get_logger(__name__)

router = APIRouter(prefix="/api/v1/frameworks", tags=["Frameworks"])


@router.get("")
async def list_frameworks():
    """Return metadata for all registered framework plugins."""
    from clinical.framework.registry import FrameworkRegistry

    registry = FrameworkRegistry.get_instance()
    registry.discover()
    all_meta = registry.list_metadata()
    return {
        "frameworks": [m.model_dump(mode="json") for m in all_meta],
        "count": len(all_meta),
    }


@router.get("/{slug}")
async def get_framework(slug: str):
    """Return metadata for a single framework plugin."""
    from clinical.framework.registry import FrameworkRegistry

    registry = FrameworkRegistry.get_instance()
    registry.discover()
    meta = registry.get_metadata(slug)
    return meta.model_dump(mode="json")


@router.post("/{slug}/enable")
async def enable_framework(slug: str):
    """Enable a framework plugin."""
    from clinical.framework.registry import FrameworkRegistry

    registry = FrameworkRegistry.get_instance()
    registry.discover()
    registry.enable(slug)
    return {"status": "enabled", "slug": slug}


@router.post("/{slug}/disable")
async def disable_framework(slug: str):
    """Disable a framework plugin."""
    from clinical.framework.registry import FrameworkRegistry

    registry = FrameworkRegistry.get_instance()
    registry.discover()
    registry.disable(slug)
    return {"status": "disabled", "slug": slug}
