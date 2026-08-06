"""
clinical/framework/utils.py — Shared utilities for framework plugins.
"""

from __future__ import annotations


def _safe_topics(case_understanding: dict) -> set[str]:
    """
    Extract a set of lowercase topic-like strings from a case understanding dict.

    Checks several common keys that may contain presentation keywords:
    ``presenting_concerns``, ``symptoms``, ``key_themes``, ``risk_factors``,
    ``contextual_stressors``, and any key ending in ``_topics`` or ``_themes``.

    This is intentionally permissive — ``is_applicable()`` methods should
    return True *too often* rather than too rarely, so the orchestrator has
    more clinical material to work with.
    """
    topics: set[str] = set()

    candidate_keys = [
        "presenting_concerns",
        "symptoms",
        "key_themes",
        "risk_factors",
        "contextual_stressors",
    ]

    for key in candidate_keys:
        val = case_understanding.get(key)
        _add_text(topics, val)

    # Catch-all for any key that looks topic-like
    for key, val in case_understanding.items():
        if not isinstance(val, str):
            continue
        if key.endswith("_topics") or key.endswith("_themes") or key.endswith("_description"):
            _add_text(topics, val)

    # Also check a free-text 'presentation' or 'clinical_text' if present
    for key in ("presentation", "clinical_text", "case_summary"):
        val = case_understanding.get(key)
        _add_text(topics, val)

    return topics


def _add_text(topics: set[str], val: object) -> None:
    if isinstance(val, str):
        for token in val.lower().replace(",", " ").replace(";", " ").split():
            token = token.strip(".,!?()[]{}'\"")
            if len(token) > 3:
                topics.add(token)
    elif isinstance(val, list):
        for item in val:
            _add_text(topics, item)
