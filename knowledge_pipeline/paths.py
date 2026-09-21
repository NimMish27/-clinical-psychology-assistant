"""
knowledge_pipeline/paths.py
════════════════════════════
Canonical paths for the knowledge base. Derived from the project root so the
pipeline works without touching ``config/settings``.

Directory layout
────────────────
    knowledge/
      rules/schema.yaml                  canonical metadata schema
      rules/chunking-rules.yaml          chunking rulebook (this pipeline)
      manifests/{framework}/{book}.yaml  one manifest per book+edition
      sources/{framework}/{book}/NN-slug.md   curated chapters
      citations/bibliography.bib         canonical citation store
      built/                             derived artifacts (gitignored)
      index/manifest.lock                alias → active collection
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

KNOWLEDGE_ROOT = PROJECT_ROOT / "knowledge"

# Rules
SCHEMA_PATH = KNOWLEDGE_ROOT / "rules" / "schema.yaml"
CHUNKING_RULES_PATH = KNOWLEDGE_ROOT / "rules" / "chunking-rules.yaml"

# Content
MANIFESTS_DIR = KNOWLEDGE_ROOT / "manifests"
SOURCES_DIR = KNOWLEDGE_ROOT / "sources"
CITATIONS_PATH = KNOWLEDGE_ROOT / "citations" / "bibliography.bib"

# Derived / index (gitignored, rebuildable)
BUILT_DIR = KNOWLEDGE_ROOT / "built"
INDEX_DIR = KNOWLEDGE_ROOT / "index"

CHUNKS_JSONL = BUILT_DIR / "chunks.jsonl"
PARENTS_JSONL = BUILT_DIR / "parents.jsonl"
HASH_REGISTRY = BUILT_DIR / "hash_registry.json"
MANIFEST_LOCK = INDEX_DIR / "manifest.lock"

# ChromaDB storage (shared with the application)
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "chroma"

# Source file naming: NN-slug.md where NN is the zero-padded chapter number.
SOURCE_FILE_PATTERN = "{chapter:02d}-{slug}.md"


def ensure_derived_dirs() -> None:
    """Create the derived-artifact directories if they do not exist."""
    BUILT_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
