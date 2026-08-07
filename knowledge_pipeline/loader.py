"""
knowledge_pipeline/loader.py
════════════════════════════
Loads the knowledge base's schema, chunking rules, manifests, curated source
files, and citations into Python structures.

No validation happens here — that is the validator's job. This module only
reads and parses.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import yaml

from app_logging.logger import get_logger
from knowledge_pipeline.models import SourceDocument, SourceSection
from knowledge_pipeline.paths import (
    CHUNKING_RULES_PATH,
    CITATIONS_PATH,
    MANIFESTS_DIR,
    SCHEMA_PATH,
    SOURCES_DIR,
)

if TYPE_CHECKING:
    from pathlib import Path

_log = get_logger(__name__)

# ── Markdown heading levels ────────────────────────────────────────────────────
_H1 = re.compile(r"^#\s+(.+?)\s*$")
_H2 = re.compile(r"^##\s+(.+?)\s*$")
_H3 = re.compile(r"^###\s+(.+?)\s*$")

_FRONT_MATTER_DELIM = re.compile(r"^---\s*$", re.MULTILINE)

# Lightweight BibTeX entry start: "@kind{key," at line start.
_BIB_ENTRY = re.compile(r"^@\w+\{\s*([^,\s]+)\s*,", re.MULTILINE)


def _slugify(text: str) -> str:
    """Convert arbitrary text to a kebab-case slug safe for chunk_id.

    Example: "The Control Agenda" → "the-control-agenda".
    """
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


def _dedupe_slug(slug: str, used: set[str]) -> str:
    """Return a chapter-unique slug, appending -2, -3, ... on collision."""
    candidate, n = slug, 1
    while candidate in used:
        n += 1
        candidate = f"{slug}-{n}"
    used.add(candidate)
    return candidate


# ── YAML helpers ───────────────────────────────────────────────────────────────

def load_yaml(path: Path) -> dict:
    """Load a YAML file, raising a descriptive error on failure."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot parse YAML at '{path}': {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping at '{path}', got {type(data).__name__}.")
    return data


# ── Schema / rules ─────────────────────────────────────────────────────────────

def load_schema() -> dict:
    """Load knowledge/rules/schema.yaml."""
    return load_yaml(SCHEMA_PATH)


def load_chunking_rules() -> dict:
    """Load knowledge/rules/chunking-rules.yaml."""
    return load_yaml(CHUNKING_RULES_PATH)


# ── Manifests ──────────────────────────────────────────────────────────────────

def discover_manifests(framework: str | None = None) -> list[Path]:
    """Return sorted manifest file paths, optionally filtered by framework."""
    if framework:
        base = MANIFESTS_DIR / framework
        if not base.is_dir():
            _log.warning("loader.manifests_missing", dir=str(base))
            return []
        return sorted(base.glob("*.yaml"))
    if not MANIFESTS_DIR.is_dir():
        _log.warning("loader.manifests_missing", dir=str(MANIFESTS_DIR))
        return []
    return sorted(MANIFESTS_DIR.rglob("*.yaml"))


def load_manifest(path: Path) -> dict:
    """Load and parse one book manifest."""
    return load_yaml(path)


def resolve_source_path(manifest: dict) -> Path | None:
    """Resolve a manifest's curated chapter directory under sources/.

    Returns None (and logs) when the directory does not exist yet.
    """
    framework = str(manifest.get("framework", ""))
    book_slug = str(manifest.get("book_slug", ""))
    chapter_dir = SOURCES_DIR / framework / book_slug
    if not chapter_dir.is_dir():
        _log.warning("loader.source_dir_missing", dir=str(chapter_dir))
        return None
    return chapter_dir


def chapter_source_files(chapter_dir: Path) -> list[Path]:
    """Return sorted curated Markdown files in a book's source directory."""
    return sorted(chapter_dir.glob("*.md"))


# ── Curated source files ───────────────────────────────────────────────────────

def _split_front_matter(text: str) -> tuple[dict | None, str]:
    """Split a Markdown file into (front_matter_dict, body).

    Returns (None, body) when no front-matter block is present.
    """
    matches = list(_FRONT_MATTER_DELIM.finditer(text))
    if len(matches) < 2:
        return None, text
    start = matches[0].end()
    end = matches[1].start()
    fm_block = text[start:end]
    body = text[matches[1].end():]
    try:
        fm = yaml.safe_load(fm_block)
    except yaml.YAMLError:
        fm = None
    return (fm if isinstance(fm, dict) else None), body


def parse_markdown_sections(body: str) -> list[SourceSection]:
    """Split a Markdown body into sections delimited by ``##`` headings.

    Content before the first ``##`` becomes an ``is_intro`` section. Within a
    section, ``###`` headings are preserved verbatim in the section text; the
    chunker uses them later as child-candidate boundaries.
    """
    sections: list[SourceSection] = []
    current_title: str | None = None
    current_lines: list[str] = []
    is_intro = True
    used_slugs: set[str] = set()

    def flush() -> None:
        nonlocal current_title, current_lines, is_intro
        if current_title is None and not current_lines:
            return
        text = "\n".join(current_lines).strip()
        if not text:
            current_title, current_lines = None, []
            return
        title = current_title or "Introduction"
        slug = _dedupe_slug(_slugify(title), used_slugs)
        sections.append(
            SourceSection(
                slug=slug,
                title=title,
                text=text,
                order=len(sections),
                is_intro=is_intro,
            )
        )
        current_title, current_lines = None, []
        is_intro = False

    for raw_line in body.splitlines():
        if not raw_line.strip():
            current_lines.append("")
            continue
        m2 = _H2.match(raw_line)
        if m2:
            flush()
            current_title = m2.group(1).strip()
            current_lines.append(raw_line.rstrip())
            continue
        current_lines.append(raw_line.rstrip())

    flush()
    return sections


def load_source_file(path: Path) -> SourceDocument:
    """Parse one curated chapter Markdown file into a SourceDocument."""
    text = path.read_text(encoding="utf-8")
    front_matter, body = _split_front_matter(text)
    sections = parse_markdown_sections(body)
    return SourceDocument(
        path=str(path),
        front_matter=front_matter or {},
        sections=sections,
    )


# ── Citations ──────────────────────────────────────────────────────────────────

def load_citations(path: Path | None = None) -> dict[str, str]:
    """Extract BibTeX keys from the bibliography into a {key: entry_text} map.

    Dependency-free: a lightweight regex parse is sufficient for the
    resolution gate (we only need to know the keys exist). Full BibTeX
    expansion is out of scope for the pipeline.
    """
    bib_path = path or CITATIONS_PATH
    if not bib_path.is_file():
        _log.warning("loader.citations_missing", path=str(bib_path))
        return {}
    text = bib_path.read_text(encoding="utf-8")
    entries: dict[str, str] = {}
    for match in _BIB_ENTRY.finditer(text):
        key = match.group(1)
        entries[key] = key  # key → key; the entry text is not needed here.
    _log.info("loader.citations_loaded", path=str(bib_path), count=len(entries))
    return entries
