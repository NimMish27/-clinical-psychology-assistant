"""
tests/unit/knowledge_pipeline/test_chunker.py
"""
from __future__ import annotations

import textwrap

from knowledge_pipeline.chunker import (
    Chunker,
    Tokenizer,
    _content_hash,
    _split_sentences,
    _windows_from_sentences,
)
from knowledge_pipeline.loader import load_chunking_rules, load_schema
from knowledge_pipeline.models import SourceDocument, SourceSection


def _rules():
    return load_chunking_rules()


def _manifest() -> dict:
    return {
        "manifest_version": 1,
        "framework": "act",
        "book_slug": "hayes-2011-acceptance-commitment-therapy-2nd",
        "book_title": "Acceptance and Commitment Therapy",
        "authors": ["Hayes, S. C.", "Strosahl, K. D.", "Wilson, K. G."],
        "edition": 2,
        "publication_year": 2011,
        "publisher": "The Guilford Press",
        "language": "en",
        "index_version": "2026-08",
    }


def _source(text: str, content_type: str = "prose", topic: list | None = None) -> SourceDocument:
    return SourceDocument(
        path="fake/05-chapter.md",
        front_matter={
            "framework": "act",
            "book_slug": "hayes-2011-acceptance-commitment-therapy-2nd",
            "chapter": 5,
            "chapter_title": "Creative Hopelessness",
            "content_type": content_type,
            "citation_ids": ["bib:hayes2011"],
            "topic": topic or ["psychological-flexibility", "experiential-avoidance"],
            "intervention": ["creative-hopelessness"],
            "population": ["adults"],
        },
        sections=[SourceSection(slug="section-one", title="Section One", text=text, order=0)],
    )


def _chunk_source(source):
    chunker = Chunker(rules=_rules())
    return chunker.chunk_source(
        source=source,
        manifest=_manifest(),
        schema=load_schema(),
        source_hash="hash",
        embedding_model="BAAI/bge-large-en-v1.5",
    )


class TestTokenizer:
    def test_count_matches_encode(self):
        tok = Tokenizer()
        text = "The client struggles with the control agenda."
        assert tok.count(text) == len(tok.tokenize(text))

    def test_fallback_count(self):
        tok = Tokenizer(fallback_ratio=0.25)
        # Force fallback path by using a fake encoding that raises.
        tok._enc = None
        tok._uses_fallback = True
        assert tok.count("abcd") == 1
        assert tok.count("a" * 40) == 10


class TestChunkSource:
    def test_parent_child_ids(self):
        text = textwrap.dedent(
            """\
            ## Section One
            Paragraph A here.
            """
        )
        parents, children = _chunk_source(_source(text))
        assert len(parents) == 1
        assert len(children) == 1
        assert parents[0].parent_id == "hayes-2011-acceptance-commitment-therapy-2nd:5:section-one"
        assert children[0].chunk_id == "hayes-2011-acceptance-commitment-therapy-2nd:5:section-one:001"
        assert children[0].parent_id == parents[0].parent_id

    def test_metadata_primary_scalar_and_all_list(self):
        parents, children = _chunk_source(_source("## S\npara text."))
        meta = children[0].metadata
        assert meta["topic"] == "psychological-flexibility"
        assert meta["topic_all"] == ["psychological-flexibility", "experiential-avoidance"]
        assert meta["framework"] == "act"
        assert meta["chapter"] == 5
        assert meta["content_hash"] == _content_hash(children[0].text)

    def test_never_split_content_type_single_child(self):
        long_text = "## Table\n" + "\n".join(f"row {i} with some content to pad" for i in range(200))
        parents, children = _chunk_source(_source(long_text, content_type="table"))
        assert len(children) == 1

    def test_oversize_atom_splits_with_overlap(self):
        chunker = Chunker(rules=_rules())
        sentences = " ".join(f"Sentence number {i} about the control agenda and suffering." for i in range(60))
        text = f"## S\n{sentences}"
        source = _source(text)
        parents, children = chunker.chunk_source(
            source=source,
            manifest=_manifest(),
            schema=load_schema(),
            source_hash="h",
            embedding_model="m",
        )
        assert len(children) > 1
        max_t = int(_rules()["budgets"]["child"]["max_tokens"])
        for child in children:
            assert child.token_count <= max_t + 20
        # Overlap: the last sentence of child 1 recurs at the start of child 2.
        tail = _split_sentences(children[0].text)[-1].strip()
        assert tail in children[1].text

    def test_ancestry_format(self):
        parents, children = _chunk_source(_source("## S\npara"))
        assert children[0].metadata["ancestry"].startswith("Hayes, Strosahl & Wilson (2011) > Ch. 5 > ")


class TestWindowsFromSentences:
    def test_overlap_repeats_boundary_sentence(self):
        sent_token_lists = [[1] * 30 for _ in range(10)]  # 10 sentences of 30 tokens
        windows = _windows_from_sentences(sent_token_lists, max_t=100, overlap_t=50)
        assert len(windows) > 1
        assert windows[1][0] == windows[0][-1]

    def test_terminates_with_large_overlap(self):
        sent_token_lists = [[1] * 10 for _ in range(6)]
        windows = _windows_from_sentences(sent_token_lists, max_t=40, overlap_t=1000)
        assert windows  # must not loop forever
        assert windows[-1][-1] == sent_token_lists[-1]


class TestSplitSentences:
    def test_basic(self):
        assert len(_split_sentences("One. Two! Three? Four")) == 4
