"""
knowledge_pipeline/validator.py
════════════════════════════════
Schema validation gate for the knowledge base.

Validates curated source front-matter and book manifests against
``knowledge/rules/schema.yaml`` (field types, enums, regexes, controlled
vocabularies) and enforces the citation-resolution gate: every ``bib:...``
referenced anywhere must exist in ``knowledge/citations/bibliography.bib``
or the build fails.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from app_logging.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable

_log = get_logger(__name__)


class ValidationError(RuntimeError):
    """Raised when the validation gate fails."""

    def __init__(self, errors: Iterable[str]) -> None:
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


def _static_fields(schema: dict, kind: str | None = None) -> list[dict]:
    fields: list[dict] = []
    for name, spec in schema.get("static_fields", {}).items():
        entry = {"name": name, **spec}
        if kind is None or entry.get("type") == kind:
            fields.append(entry)
    return fields


# ── Controlled vocabulary helpers ──────────────────────────────────────────────

def _vocab_for(field: dict, schema: dict) -> list[str] | None:
    vocab_name = field.get("vocab")
    if not vocab_name:
        return None
    vocab = schema.get("vocabularies", {}).get(vocab_name, {})
    return vocab.get("common")


# ── Front-matter validation ────────────────────────────────────────────────────

def validate_front_matter(front_matter: dict, schema: dict) -> list[str]:
    """Validate one source file's front-matter. Returns a list of errors."""
    errors: list[str] = []
    fields = _static_fields(schema)

    for field in fields:
        name: str = field["name"]
        value = front_matter.get(name)
        present = value is not None and value != []

        if field.get("required") and not present:
            errors.append(f"{name}: required but missing")
            continue
        if not present:
            continue

        ftype = field.get("type")
        fname_label = f"front_matter.{name}"

        if ftype == "int" and (not isinstance(value, int) or isinstance(value, bool)):
            errors.append(f"{fname_label}: expected int, got {type(value).__name__}")
        elif ftype == "string" and not isinstance(value, str):
            errors.append(f"{fname_label}: expected string, got {type(value).__name__}")
        elif ftype in ("enum", "list[controlled]"):
            _validate_enum_or_list(value, field, schema, fname_label, errors)
        elif ftype == "list[string]" and (
            not isinstance(value, list) or not all(isinstance(v, str) for v in value)
        ):
            errors.append(f"{fname_label}: expected list of strings")

        pattern = field.get("pattern")
        if pattern:
            if isinstance(value, str) and not re.match(pattern, value):
                errors.append(f"{fname_label}: does not match pattern {pattern!r}: {value!r}")
            elif isinstance(value, list):
                for v in value:
                    if isinstance(v, str) and not re.match(pattern, v):
                        msg = f"{fname_label}: item {v!r} does not match pattern {pattern!r}"
                        errors.append(msg)

    return errors


def _validate_enum_or_list(
    value: object,
    field: dict,
    schema: dict,
    label: str,
    errors: list[str],
) -> None:
    """Validate enum values or controlled-vocab lists."""
    allowed = field.get("values") if field.get("type") == "enum" else _vocab_for(field, schema)

    items = value if isinstance(value, list) else [value]
    if field.get("type") == "enum" and isinstance(value, list):
        errors.append(f"{label}: enum fields must be a scalar, got a list")
        return

    for item in items:
        if allowed and item not in allowed:
            errors.append(f"{label}: value {item!r} not in allowed set")


# ── Manifest validation ────────────────────────────────────────────────────────

def validate_manifest(manifest: dict, schema: dict) -> list[str]:
    """Validate one book manifest. Returns a list of errors."""
    errors: list[str] = []

    if manifest.get("manifest_version") != 1:
        errors.append("manifest_version: expected 1")

    frameworks = schema.get("frameworks", [])
    if manifest.get("framework") not in frameworks:
        errors.append(f"framework: {manifest.get('framework')!r} not in {frameworks}")

    book_slug = manifest.get("book_slug")
    if not isinstance(book_slug, str) or not re.match(r"^[a-z0-9]+(-[a-z0-9]+)*$", book_slug or ""):
        errors.append(f"book_slug: {book_slug!r} is not a valid slug")

    required_keys = (
        "book_title",
        "authors",
        "edition",
        "publication_year",
        "language",
        "index_version",
    )
    for key in required_keys:
        if manifest.get(key) in (None, ""):
            errors.append(f"{key}: required but missing")

    chapters = manifest.get("chapters")
    if not isinstance(chapters, list) or not chapters:
        errors.append("chapters: expected a non-empty list")
    else:
        seen_numbers: set[int] = set()
        for i, ch in enumerate(chapters):
            if not isinstance(ch, dict):
                errors.append(f"chapters[{i}]: expected a mapping")
                continue
            num = ch.get("number")
            if num in seen_numbers:
                errors.append(f"chapters: duplicate chapter number {num}")
            seen_numbers.add(num)
            for key in ("number", "slug", "title"):
                if ch.get(key) in (None, ""):
                    errors.append(f"chapters[{i}]: {key} required but missing")

    return errors


# ── Citation resolution gate ───────────────────────────────────────────────────

def check_citation_ids(ids: list[str] | None, citations: dict[str, str]) -> list[str]:
    """Return the citation ids that do NOT resolve in the bibliography."""
    unresolved: list[str] = []
    for cid in ids or []:
        key = cid.removeprefix("bib:")
        if key not in citations:
            unresolved.append(cid)
    return unresolved


# ── Aggregate gate ─────────────────────────────────────────────────────────────

_BOOK_IDENTITY_FIELDS = (
    "book_title",
    "authors",
    "edition",
    "publication_year",
    "publisher",
    "isbn",
    "language",
)


def validate_book(
    manifest: dict,
    schema: dict,
    citations: dict[str, str],
    front_matters: list[dict],
) -> None:
    """Validate a full book (manifest + its chapter front-matters). Raises.

    Chapter front-matter is validated against the schema *merged* with the
    manifest's book-identity fields (book_title, authors, edition, ...) so
    chapters need not repeat them. Consistency between the front-matter's
    framework / book_slug / chapter and the manifest is also enforced.

    Raises:
        ValidationError: when any check fails, with all collected errors.
    """
    errors: list[str] = []
    errors += validate_manifest(manifest, schema)

    manifest_cites = manifest.get("citation_ids", [])
    errors += [
        f"manifest citation unresolved: {cid}"
        for cid in check_citation_ids(manifest_cites, citations)
    ]

    book_identity = {key: manifest.get(key) for key in _BOOK_IDENTITY_FIELDS}
    chapters_by_number = {
        ch.get("number"): ch
        for ch in manifest.get("chapters", [])
        if isinstance(ch, dict)
    }

    for fm in front_matters:
        effective_fm = {**book_identity, **fm}
        errors += validate_front_matter(effective_fm, schema)

        if fm.get("framework") and fm.get("framework") != manifest.get("framework"):
            errors.append(
                f"front_matter.framework {fm.get('framework')!r} != "
                f"manifest.framework {manifest.get('framework')!r}"
            )
        if fm.get("book_slug") and fm.get("book_slug") != manifest.get("book_slug"):
            errors.append(
                f"front_matter.book_slug {fm.get('book_slug')!r} != "
                f"manifest.book_slug {manifest.get('book_slug')!r}"
            )
        chapter = chapters_by_number.get(fm.get("chapter"))
        if chapter is None:
            errors.append(f"front_matter.chapter {fm.get('chapter')!r} not in manifest chapters")
        else:
            manifest_title = chapter.get("title")
            if fm.get("chapter_title") and manifest_title and fm["chapter_title"] != manifest_title:
                errors.append(
                    f"front_matter.chapter_title {fm.get('chapter_title')!r} != "
                    f"manifest chapter {fm.get('chapter')} title {manifest_title!r}"
                )

        fm_cites = fm.get("citation_ids", [])
        errors += [
            f"front_matter citation unresolved: {cid}"
            for cid in check_citation_ids(fm_cites, citations)
        ]

    if errors:
        raise ValidationError(errors)
    _log.info("validator.book_ok", book=manifest.get("book_slug"), sources=len(front_matters))
