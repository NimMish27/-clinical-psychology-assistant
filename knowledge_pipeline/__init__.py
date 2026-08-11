"""
knowledge_pipeline
══════════════════
Production knowledge-base build pipeline.

Turns curated Markdown sources + book manifests + citations into validated,
structure-aware parent/child chunks, then embeds and stores them in ChromaDB
collections named ``kb_{framework}_{index_version}``.

Pipeline stages (see scripts/build_knowledge_base.py):

    curated sources + manifests
        → loader        (load schema, manifests, sources, citations)
        → validator     (schema + vocab + citation resolution gate)
        → chunker       (structure-aware parent-child splitting)
        → builder       (chunks.jsonl / parents.jsonl / hash registry)
        → indexer       (embed → ChromaDB + manifest.lock alias)
"""

from knowledge_pipeline.models import (
    BuildResult,
    ChildChunk,
    ParentUnit,
    SourceDocument,
    SourceSection,
)

__all__ = [
    "BuildResult",
    "ChildChunk",
    "ParentUnit",
    "SourceDocument",
    "SourceSection",
]
