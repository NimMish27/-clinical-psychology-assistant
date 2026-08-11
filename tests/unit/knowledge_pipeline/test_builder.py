"""
tests/unit/knowledge_pipeline/test_builder.py
"""
from __future__ import annotations

from knowledge_pipeline.builder import build_knowledge_base
from knowledge_pipeline.models import BuildResult


def test_build_real_worked_example_in_memory():
    result = build_knowledge_base(
        book_slug="hayes-2011-acceptance-commitment-therapy-2nd",
        write_outputs=False,
    )
    assert isinstance(result, BuildResult)
    assert result.ok, result.errors
    assert result.parents == 4
    assert result.children == 4
    assert len(result.source_files) == 1


def test_build_unknown_book_reports_error():
    result = build_knowledge_base(book_slug="does-not-exist", write_outputs=False)
    assert not result.ok
    assert result.errors


def test_build_all_frameworks_runs():
    result = build_knowledge_base(write_outputs=False)
    assert isinstance(result, BuildResult)
    # Only curated books build; the ACT worked example must be among them.
    assert result.parents >= 4
    assert result.children >= 4
