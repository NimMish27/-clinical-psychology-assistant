"""
knowledge_pipeline/chunker.py
═════════════════════════════
Structure-aware parent-child chunking.

Turns a parsed ``SourceDocument`` (one chapter) into:

  * ParentUnit per ``##`` section (1500–3000 token context units), and
  * ChildChunk per embedded retrieval unit (200–400 tokens), split on
    ``###`` subsection boundaries then paragraphs then sentences.

Behaviour is driven by ``knowledge/rules/chunking-rules.yaml``.
"""

from __future__ import annotations

import hashlib
import re
import threading
from datetime import UTC, datetime
from typing import Any

from app_logging.logger import get_logger
from knowledge_pipeline.models import ChildChunk, ParentUnit, SourceDocument, SourceSection

_log = get_logger(__name__)


class Tokenizer:
    """Thread-safe token counter with a deterministic fallback.

    Uses tiktoken cl100k_base when available; otherwise approximates with
    ``chars * fallback_ratio`` (default 0.25 → 4 chars/token).
    """

    _lock = threading.Lock()
    _instance: Tokenizer | None = None

    def __init__(self, encoding_name: str = "cl100k_base", fallback_ratio: float = 0.25) -> None:
        self._encoding_name = encoding_name
        self._fallback_ratio = fallback_ratio
        self._enc = None
        self._uses_fallback = False
        try:
            import tiktoken

            self._enc = tiktoken.get_encoding(encoding_name)
        except Exception:
            self._enc = None
            self._uses_fallback = True

    @classmethod
    def get_instance(cls) -> Tokenizer:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._enc is not None:
            return len(self._enc.encode(text))
        return max(1, int(len(text) * self._fallback_ratio))

    def tokenize(self, text: str) -> list[int]:
        if self._enc is not None:
            return list(self._enc.encode(text))
        return [1] * self.count(text)

    def decode(self, tokens: list[int]) -> str:
        if self._enc is not None:
            return self._enc.decode(tokens)
        return ""

    @property
    def uses_fallback(self) -> bool:
        return self._uses_fallback

    @property
    def encoding_name(self) -> str:
        return self._encoding_name


# ── Metadata builders ───────────────────────────────────────────────────────────

def _primary_scalar(items: list[str] | None) -> str | None:
    """Collapse a controlled-vocab list to a single filterable scalar."""
    if not items:
        return None
    return str(items[0])


def _short_authors(authors: list[str]) -> str:
    """'Hayes, S. C.' + 'Strosahl, K. D.' → 'Hayes, Strosahl'."""
    surnames = [a.split(",")[0].strip() for a in authors]
    if len(surnames) > 1:
        return f"{', '.join(surnames[:-1])} & {surnames[-1]}"
    return surnames[0] if surnames else ""


class Chunker:
    """Chunks curated chapters into parent/child units per the rules file."""

    def __init__(self, rules: dict | None = None) -> None:
        if rules is None:
            from knowledge_pipeline.loader import load_chunking_rules

            rules = load_chunking_rules()
        self._rules = rules
        self._tok = Tokenizer.get_instance()
        budgets = rules.get("budgets", {})
        self._child_min = int(budgets["child"]["min_tokens"])
        self._child_max = int(budgets["child"]["max_tokens"])
        self._child_overlap = int(budgets["child"].get("overlap_tokens", 0))
        never_split = rules.get("never_split", {}).get("content_types", [])
        self._never_split = set(never_split)
        self._uses_fallback = self._tok.uses_fallback

        _log.info(
            "chunker.init",
            child_min=self._child_min,
            child_max=self._child_max,
            overlap=self._child_overlap,
            never_split=sorted(self._never_split),
            tokenizer=self._tok.encoding_name,
            fallback=self._uses_fallback,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def chunk_source(
        self,
        source: SourceDocument,
        manifest: dict,
        schema: dict,
        source_hash: str,
        embedding_model: str,
    ) -> tuple[list[ParentUnit], list[ChildChunk]]:
        """Chunk one chapter into (parents, children).

        Validation happens in the builder via ``validator.validate_book``;
        this method does not re-validate so it can be called after the gate.
        """
        parents: list[ParentUnit] = []
        children: list[ChildChunk] = []

        for section in source.sections:
            parent, section_children = self._chunk_section(
                section=section,
                source=source,
                manifest=manifest,
                source_hash=source_hash,
                embedding_model=embedding_model,
            )
            parents.append(parent)
            children.extend(section_children)

        return parents, children

    # ── Section → parent + children ─────────────────────────────────────────────

    def _chunk_section(
        self,
        section: SourceSection,
        source: SourceDocument,
        manifest: dict,
        source_hash: str,
        embedding_model: str,
    ) -> tuple[ParentUnit, list[ChildChunk]]:
        content_type = source.front_matter.get("content_type", "prose")
        section_title = section.title

        parent_id = f"{source.book_slug}:{source.chapter}:{section.slug}"
        ancestry = self._build_ancestry(source, manifest, section_title)

        base = self._base_metadata(
            source=source,
            manifest=manifest,
            section=section_title,
            source_hash=source_hash,
            embedding_model=embedding_model,
            ancestry=ancestry,
        )

        parent_meta = {**base, "parent_id": parent_id, "index_status": "validated"}
        parent = ParentUnit(
            parent_id=parent_id,
            book_slug=source.book_slug,
            chapter=source.chapter,
            section_slug=section.slug,
            title=section_title,
            text=section.text,
            token_count=self._tok.count(section.text),
            order=section.order,
            metadata=parent_meta,
        )

        never_split = content_type in self._never_split
        raw_children = [section.text] if never_split else self._split_section_text(section.text)

        children: list[ChildChunk] = []
        for seq, text in enumerate(raw_children, start=1):
            token_count = self._tok.count(text)
            chunk_id = f"{source.book_slug}:{source.chapter}:{section.slug}:{seq:03d}"
            meta = {
                **base,
                "chunk_id": chunk_id,
                "parent_id": parent_id,
                "seq": seq,
                "token_count": token_count,
                "content_hash": _content_hash(text),
                "index_status": "validated",
            }
            children.append(
                ChildChunk(
                    chunk_id=chunk_id,
                    parent_id=parent_id,
                    seq=seq,
                    text=text,
                    token_count=token_count,
                    metadata=meta,
                )
            )

        return parent, children

    # ── Text splitting ──────────────────────────────────────────────────────────

    def _split_section_text(self, section_text: str) -> list[str]:
        """Split a section's body into child texts respecting the token budget."""
        min_t, max_t = self._child_min, self._child_max
        atoms = self._make_atoms(section_text)

        normalized: list[str] = []
        for atom in atoms:
            if self._tok.count(atom) > max_t:
                normalized.extend(
                    self._split_sentences_to_windows(atom, max_t=max_t, min_t=min_t)
                )
            else:
                normalized.append(atom)
        return self._merge_atoms(normalized, min_t=min_t, max_t=max_t)

    def _make_atoms(self, section_text: str) -> list[str]:
        """Return atomic blocks: H3 subsections (kept whole) or paragraphs."""
        if re.search(r"(?m)^### ", section_text):
            blocks = re.split(r"(?m)^(?=### )", section_text)
            return [b.strip() for b in blocks if b.strip()]
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section_text) if p.strip()]
        return paragraphs

    def _merge_atoms(self, atoms: list[str], *, min_t: int, max_t: int) -> list[str]:
        """Merge atoms into children in [min_t, max_t]; undersized tail merges back."""
        children: list[str] = []
        buf: list[str] = []
        buf_tokens = 0

        for atom in atoms:
            atom_t = self._tok.count(atom)
            if buf and buf_tokens + atom_t + 2 > max_t:
                children.append("\n\n".join(buf))
                buf, buf_tokens = [atom], atom_t
            else:
                buf.append(atom)
                buf_tokens += atom_t

        if buf:
            children.append("\n\n".join(buf))

        # Merge undersized children forward (rule: merge with next sibling).
        merged: list[str] = []
        for child in children:
            if (
                merged
                and self._tok.count(merged[-1]) < min_t
                and self._tok.count(merged[-1]) + self._tok.count(child) + 2 <= max_t
            ):
                merged[-1] = f"{merged[-1]}\n\n{child}"
            else:
                merged.append(child)

        return merged

    def _split_sentences_to_windows(self, text: str, *, max_t: int, min_t: int) -> list[str]:
        """Split an oversize atom at sentence boundaries with token overlap."""
        sentences = [s.strip() for s in _split_sentences(text) if s.strip()]
        if not sentences:
            return [text]

        sent_token_lists = [self._tok.tokenize(s) for s in sentences]
        windows = _windows_from_sentences(
            sent_token_lists, max_t=max_t, overlap_t=self._child_overlap
        )
        out: list[str] = []
        for window in windows:
            # Decode each sentence separately and join with a space so the
            # reconstructed text keeps the original inter-sentence spacing.
            decoded = " ".join(
                self._tok.decode(st).strip() for st in window if self._tok.decode(st).strip()
            ).strip()
            if decoded:
                out.append(decoded)
        return out or [text]

    # ── Metadata helpers ────────────────────────────────────────────────────────

    def _build_ancestry(self, source: SourceDocument, manifest: dict, section: str) -> str:
        authors = manifest.get("authors", [])
        year = manifest.get("publication_year", "")
        prefix = (
            f"{_short_authors(authors)} ({year})"
            if authors
            else manifest.get("book_title", "")
        )
        return f"{prefix} > Ch. {source.chapter} > {section}"

    def _base_metadata(
        self,
        source: SourceDocument,
        manifest: dict,
        section: str,
        source_hash: str,
        embedding_model: str,
        ancestry: str,
    ) -> dict[str, Any]:
        fm = source.front_matter
        now = datetime.now(UTC).isoformat()

        meta: dict[str, Any] = {
            "framework": fm.get("framework") or manifest.get("framework"),
            "book_slug": fm.get("book_slug") or manifest.get("book_slug"),
            "book_title": fm.get("book_title") or manifest.get("book_title"),
            "authors": fm.get("authors") or manifest.get("authors", []),
            "edition": (
                fm.get("edition") if fm.get("edition") is not None else manifest.get("edition")
            ),
            "publication_year": (
                fm.get("publication_year")
                if fm.get("publication_year") is not None
                else manifest.get("publication_year")
            ),
            "publisher": fm.get("publisher") or manifest.get("publisher"),
            "isbn": fm.get("isbn") or manifest.get("isbn"),
            "language": fm.get("language") or manifest.get("language"),
            "chapter": fm.get("chapter"),
            "chapter_title": fm.get("chapter_title"),
            "section": section,
            "content_type": fm.get("content_type", "prose"),
            "citation_ids": fm.get("citation_ids", []),
            "confidence": fm.get("confidence", 1.0),
            # Filterable primary scalars for controlled-vocab fields.
            "topic": _primary_scalar(fm.get("topic")),
            "intervention": _primary_scalar(fm.get("intervention")),
            "symptom": _primary_scalar(fm.get("symptom")),
            "population": _primary_scalar(fm.get("population")),
            # Full sets preserved for audit / display.
            "topic_all": fm.get("topic", []),
            "intervention_all": fm.get("intervention", []),
            "symptom_all": fm.get("symptom", []),
            "population_all": fm.get("population", []),
            # Derived / provenance.
            "source": fm.get("book_slug", ""),
            "page": 0,
            "ancestry": ancestry,
            "source_hash": source_hash,
            "embedding_model": embedding_model,
            "index_version": manifest.get("index_version", ""),
            "created_at": now,
            "updated_at": now,
        }
        return meta


# ── Pure helpers ───────────────────────────────────────────────────────────────

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def _split_sentences(text: str) -> list[str]:
    """Split text at sentence boundaries, preserving abbreviations poorly is fine."""
    return [s for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]


def _content_hash(text: str) -> str:
    """sha256 of the chunk body (first 16 hex chars) for drift detection."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _windows_from_sentences(
    sent_token_lists: list[list[int]],
    *,
    max_t: int,
    overlap_t: int,
) -> list[list[list[int]]]:
    """Partition tokenised sentences into windows of ≤ max_t tokens with overlap."""
    windows: list[list[list[int]]] = []
    n = len(sent_token_lists)
    start = 0
    while start < n:
        end = start
        acc = 0
        while end < n and acc + len(sent_token_lists[end]) <= max_t:
            acc += len(sent_token_lists[end])
            end += 1
        if end == start:
            end = start + 1  # single sentence exceeds max_t alone — keep whole
        windows.append(sent_token_lists[start:end])
        if end >= n:
            break
        back = 0
        tail = 0
        for k in range(end - 1, start - 1, -1):
            token_len = len(sent_token_lists[k])
            if tail + token_len > overlap_t:
                break
            tail += token_len
            back += 1
        next_start = end - back
        start = next_start if next_start > start else end
    return windows
