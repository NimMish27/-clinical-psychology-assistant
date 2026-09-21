"""
tests/unit/knowledge_pipeline/test_loader.py
"""
from __future__ import annotations

import textwrap

from knowledge_pipeline import loader


def test_split_front_matter_returns_dict_and_body():
    text = textwrap.dedent(
        """\
        ---
        framework: act
        book_slug: demo-book
        ---
        # Chapter
        body text
        """
    )
    fm, body = loader._split_front_matter(text)
    assert fm == {"framework": "act", "book_slug": "demo-book"}
    assert "# Chapter" in body


def test_split_front_matter_absent():
    fm, body = loader._split_front_matter("no front matter here")
    assert fm is None
    assert body == "no front matter here"


def test_parse_markdown_sections_creates_intro_and_h2():
    body = textwrap.dedent(
        """\
        # Title
        intro paragraph line one
        intro paragraph line two

        ## The Control Agenda
        control text

        ### Subsection
        sub text

        ## Second Section
        second text
        """
    )
    sections = loader.parse_markdown_sections(body)
    assert [s.title for s in sections] == ["Introduction", "The Control Agenda", "Second Section"]
    assert sections[0].is_intro is True
    assert sections[1].is_intro is False
    assert sections[0].slug == "introduction"
    assert sections[1].slug == "the-control-agenda"


def test_parse_markdown_sections_dedupes_slugs():
    body = "## Intro\nx\n\n## Intro\nx\n"
    sections = loader.parse_markdown_sections(body)
    assert [s.slug for s in sections] == ["intro", "intro-2"]


def test_load_citations_extracts_keys(tmp_path):
    bib = tmp_path / "test.bib"
    bib.write_text(
        "@book{hayes2011,\n  title = {T}\n}\n@article{luoma2017,\n  title = {X}\n}\n",
        encoding="utf-8",
    )
    citations = loader.load_citations(bib)
    assert set(citations) == {"hayes2011", "luoma2017"}


def test_discover_manifests_filters_by_framework(tmp_path, monkeypatch):
    act_dir = tmp_path / "act"
    cbt_dir = tmp_path / "cbt"
    act_dir.mkdir(parents=True)
    cbt_dir.mkdir(parents=True)
    (act_dir / "a.yaml").write_text("framework: act\n", encoding="utf-8")
    (cbt_dir / "b.yaml").write_text("framework: cbt\n", encoding="utf-8")
    monkeypatch.setattr(loader, "MANIFESTS_DIR", tmp_path)

    act_manifests = loader.discover_manifests("act")
    assert [p.name for p in act_manifests] == ["a.yaml"]
    all_manifests = loader.discover_manifests()
    assert {p.name for p in all_manifests} == {"a.yaml", "b.yaml"}
