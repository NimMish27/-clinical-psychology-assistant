"""
knowledge_pipeline/models.py
════════════════════════════
Pydantic data models for the knowledge-base pipeline.

Data flow
─────────
    SourceDocument  (curated Markdown parsed into sections)
        → ParentUnit   (one per section, 1500–3000 token context unit)
            → ChildChunk  (200–400 token embedded retrieval unit)

The ``metadata`` dicts emitted by the chunker carry exactly the fields the
retrieval layer filters on (framework / chapter / topic / intervention /
symptom / population) plus the schema's derived fields.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SourceSection(BaseModel):
    """A structural unit (## heading) parsed from a curated Markdown file.

    Attributes:
        slug:     Kebab-case, chapter-unique identifier used in chunk_id.
        title:    Raw heading text.
        text:     Heading + body content for this section.
        order:    0-based position within the chapter.
        is_intro: True when the section is the content before the first ##.
    """

    slug: str
    title: str
    text: str
    order: int = Field(ge=0)
    is_intro: bool = False


class SourceDocument(BaseModel):
    """A parsed curated source file (one chapter).

    ``front_matter`` is the raw YAML dict from the file header; it is the
    chapter-level metadata validated against ``schema.yaml`` by the validator.
    """

    path: str
    front_matter: dict = Field(default_factory=dict)
    sections: list[SourceSection] = Field(default_factory=list)

    @property
    def framework(self) -> str:
        return str(self.front_matter.get("framework", ""))

    @property
    def book_slug(self) -> str:
        return str(self.front_matter.get("book_slug", ""))

    @property
    def chapter(self) -> int:
        return int(self.front_matter.get("chapter", 0))


class ParentUnit(BaseModel):
    """One section — the context-injection unit of a parent-child scheme."""

    parent_id: str
    book_slug: str
    chapter: int
    section_slug: str
    title: str
    text: str
    token_count: int = Field(ge=0)
    order: int = Field(ge=0)
    metadata: dict = Field(default_factory=dict)


class ChildChunk(BaseModel):
    """One embedded retrieval unit within a parent section."""

    chunk_id: str
    parent_id: str
    seq: int = Field(ge=1)
    text: str
    token_count: int = Field(ge=0)
    metadata: dict = Field(default_factory=dict)


class BuildResult(BaseModel):
    """Summary of one pipeline build run."""

    built_at: str
    source_files: list[str] = Field(default_factory=list)
    parents: int = 0
    children: int = 0
    total_chars: int = 0
    total_tokens: int = 0
    errors: list[str] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors
