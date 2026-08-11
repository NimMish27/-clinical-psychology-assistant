"""
tests/unit/knowledge_pipeline/test_validator.py
"""
from __future__ import annotations

import pytest

from knowledge_pipeline.validator import (
    ValidationError,
    check_citation_ids,
    validate_book,
    validate_front_matter,
    validate_manifest,
)


@pytest.fixture(scope="module")
def schema():
    from knowledge_pipeline.loader import load_schema

    return load_schema()


def _valid_fm() -> dict:
    return {
        "framework": "act",
        "book_slug": "hayes-2011-acceptance-commitment-therapy-2nd",
        "book_title": "ACT",
        "authors": ["Hayes, S. C."],
        "edition": 2,
        "publication_year": 2011,
        "language": "en",
        "chapter": 5,
        "chapter_title": "Creative Hopelessness",
        "content_type": "prose",
        "citation_ids": ["bib:hayes2011"],
        "topic": ["psychological-flexibility"],
    }


def _valid_manifest() -> dict:
    return {
        "manifest_version": 1,
        "framework": "act",
        "book_slug": "hayes-2011-acceptance-commitment-therapy-2nd",
        "book_title": "ACT",
        "authors": ["Hayes, S. C."],
        "edition": 2,
        "publication_year": 2011,
        "language": "en",
        "index_version": "2026-08",
        "citation_ids": ["bib:hayes2011"],
        "chapters": [{"number": 5, "slug": "creative-hopelessness", "title": "Creative Hopelessness"}],
    }


class TestValidateFrontMatter:
    def test_valid_passes(self, schema):
        assert validate_front_matter(_valid_fm(), schema) == []

    def test_missing_required(self, schema):
        fm = _valid_fm()
        del fm["content_type"]
        errors = validate_front_matter(fm, schema)
        assert any("content_type" in e for e in errors)

    def test_enum_violation(self, schema):
        fm = _valid_fm()
        fm["framework"] = "not-a-framework"
        assert any("not-a-framework" in e for e in validate_front_matter(fm, schema))

    def test_vocab_violation(self, schema):
        fm = _valid_fm()
        fm["topic"] = ["totally-made-up-topic"]
        assert any("totally-made-up-topic" in e for e in validate_front_matter(fm, schema))

    def test_citation_pattern(self, schema):
        fm = _valid_fm()
        fm["citation_ids"] = ["https://example.com/not-a-bib"]
        assert any("citation_ids" in e for e in validate_front_matter(fm, schema))


class TestValidateManifest:
    def test_valid(self, schema):
        assert validate_manifest(_valid_manifest(), schema) == []

    def test_bad_framework(self, schema):
        m = _valid_manifest()
        m["framework"] = "nope"
        assert any("framework" in e for e in validate_manifest(m, schema))

    def test_missing_chapters(self, schema):
        m = _valid_manifest()
        m.pop("chapters")
        assert any("chapters" in e for e in validate_manifest(m, schema))

    def test_duplicate_chapter_number(self, schema):
        m = _valid_manifest()
        m["chapters"] = [
            {"number": 5, "slug": "a", "title": "A"},
            {"number": 5, "slug": "b", "title": "B"},
        ]
        assert any("duplicate" in e for e in validate_manifest(m, schema))


class TestCitationGate:
    def test_resolved(self):
        assert check_citation_ids(["bib:hayes2011"], {"hayes2011": "hayes2011"}) == []

    def test_unresolved(self):
        assert check_citation_ids(["bib:missing"], {"hayes2011": "hayes2011"}) == ["bib:missing"]

    def test_manifest_key_without_prefix(self):
        assert check_citation_ids(["bib:x"], {"bib:x": "x"}) == ["bib:x"]


class TestValidateBook:
    def test_merges_book_identity(self, schema):
        # Chapter front-matter does NOT repeat book identity; manifest supplies it.
        fm = _valid_fm()
        for key in ("book_title", "authors", "edition", "publication_year", "language"):
            fm.pop(key, None)
        validate_book(_valid_manifest(), schema, {"hayes2011": "hayes2011"}, [fm])

    def test_framework_mismatch(self, schema):
        fm = _valid_fm()
        fm["framework"] = "cbt"
        with pytest.raises(ValidationError) as exc:
            validate_book(_valid_manifest(), schema, {"hayes2011": "hayes2011"}, [fm])
        assert any("framework" in e for e in exc.value.errors)

    def test_unresolved_citation_raises(self, schema):
        with pytest.raises(ValidationError) as exc:
            validate_book(_valid_manifest(), schema, {}, [_valid_fm()])
        assert any("unresolved" in e for e in exc.value.errors)
