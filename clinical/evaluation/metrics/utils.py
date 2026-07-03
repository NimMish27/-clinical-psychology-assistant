from __future__ import annotations

import re
from typing import Any


def normalize_text(text: str) -> str:
    """Strip punctuation, parentheses, leading dashes, collapse whitespace."""
    result = text.strip().lstrip("- ").strip()
    result = result.replace("(", "").replace(")", "").replace(",", "").replace(".", ":")
    result = result.replace(":", "").replace(";", "")
    return re.sub(r"\s+", " ", result).lower().strip()


def citation_matches_inline(
    inline_citation: str,
    normalized_refs: list[str],
    *,
    min_overlap: int = 2,
    min_ratio: float = 0.3,
) -> bool:
    """Check if an inline citation matches any reference using word overlap."""
    norm_cite = normalize_text(inline_citation)
    c_words = norm_cite.split()
    if not c_words:
        return False

    for ref in normalized_refs:
        r_words = ref.split()
        overlap = sum(1 for w in c_words if w in r_words)
        ratio = overlap / max(len(c_words), 1)
        if overlap >= min_overlap and ratio >= min_ratio:
            return True
    return False


def extract_sections(markdown: str) -> dict[str, str]:
    """Split markdown into sections by ## headers."""
    sections: dict[str, str] = {}
    current_header: str | None = None
    current_lines: list[str] = []
    header_re = re.compile(r"^##\s+\d*\.?\s*(.+)$", re.MULTILINE)

    for line in markdown.split("\n"):
        m = header_re.match(line)
        if m:
            if current_header:
                sections[current_header.strip().lower()] = "\n".join(current_lines).strip()
            current_header = m.group(1).strip()
            current_lines = []
        elif current_header:
            current_lines.append(line)

    if current_header:
        sections[current_header.strip().lower()] = "\n".join(current_lines).strip()
    return sections


def extract_references(markdown: str, ref_headers: set[str]) -> list[str]:
    """Extract and normalize the reference section content."""
    sections = extract_sections(markdown)
    ref_section_text = None
    for h in ref_headers:
        if h in sections:
            ref_section_text = sections[h]
            break

    refs: list[str] = []
    if ref_section_text:
        for line in ref_section_text.split("\n"):
            stripped = line.strip().lstrip("- ").strip()
            norm = normalize_text(stripped)
            if norm and len(norm) > 15:
                refs.append(norm)
    return refs
