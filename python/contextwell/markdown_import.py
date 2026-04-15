"""Markdown file parser for contextwell memory ingestion.

Splits a markdown document into chunks suitable for embedding and storage.
Supports header-based splitting (##/### boundaries), YAML front-matter for
metadata defaults, type heuristics from section titles, and paragraph-based
fallback chunking for files without headers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# Characters per chunk for paragraph-based fallback splitting.
_CHUNK_SIZE = 800
_CHUNK_OVERLAP = 100

# Regex patterns
_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_HEADER_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)

# Map section title keywords → memory type
_TYPE_HINTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bdecision\b", re.IGNORECASE), "decision"),
    (re.compile(r"\btodo\b|\bto.do\b|\btask\b", re.IGNORECASE), "todo"),
    (re.compile(r"\bcode\b|\bimplementation\b|\bsnippet\b", re.IGNORECASE), "code"),
    (re.compile(r"\bnote\b|\bchat\b|\bsummary\b|\bcompact\b", re.IGNORECASE), "chat"),
]


@dataclass
class MarkdownChunk:
    """A single chunk extracted from a markdown file, ready for embedding."""

    content: str
    type: str = "fact"
    tags: list[str] = field(default_factory=list)
    source: str = ""


def _parse_front_matter(text: str) -> tuple[dict[str, object], str]:
    """Extract YAML front-matter from the start of *text*.

    Returns ``(metadata_dict, body_without_front_matter)``.
    Only parses simple ``key: value`` pairs — avoids a heavy YAML dep.
    """
    match = _FRONT_MATTER_RE.match(text)
    if not match:
        return {}, text

    meta: dict[str, object] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            if key == "tags":
                # Simple inline list: "tags: [a, b, c]" or "tags: a, b"
                value = value.strip("[]")
                meta[key] = [t.strip() for t in value.split(",") if t.strip()]
            else:
                meta[key] = value
    return meta, text[match.end() :]


def _infer_type(title: str, type_hint: str) -> str:
    """Return a memory type inferred from a section title, falling back to *type_hint*."""
    for pattern, memory_type in _TYPE_HINTS:
        if pattern.search(title):
            return memory_type
    return type_hint


def _paragraph_chunks(text: str, chunk_size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> list[str]:
    """Split *text* into overlapping paragraph-aligned chunks.

    Tries to break on paragraph boundaries (blank lines). If a paragraph is
    longer than *chunk_size*, it is split at the nearest whitespace.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        # If a single paragraph exceeds chunk_size, hard-split it
        if len(para) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            for start in range(0, len(para), chunk_size - overlap):
                piece = para[start : start + chunk_size]
                if piece.strip():
                    chunks.append(piece.strip())
            continue

        if current and len(current) + len(para) + 2 > chunk_size:
            chunks.append(current.strip())
            # Carry overlap: last sentence(s) of previous chunk
            sentences = re.split(r"(?<=[.!?])\s+", current)
            carry = ""
            for s in reversed(sentences):
                if len(carry) + len(s) + 1 <= overlap:
                    carry = s + " " + carry
                else:
                    break
            current = carry.strip() + ("\n\n" if carry.strip() else "") + para
        else:
            current = (current + "\n\n" + para).strip() if current else para

    if current.strip():
        chunks.append(current.strip())

    return chunks


def parse(
    path: str | Path,
    *,
    default_tags: list[str] | None = None,
    default_type: str = "fact",
) -> list[MarkdownChunk]:
    """Parse a markdown file into a list of :class:`MarkdownChunk` objects.

    Strategy:
    1. Strip YAML front-matter; use it to override defaults.
    2. If ``##``/``###`` headers exist, split on them — each section is one chunk.
    3. If no headers, fall back to paragraph-based chunking (~800 chars each).

    Args:
        path: Absolute or relative path to the markdown file.
        default_tags: Base tags applied to every chunk.
        default_type: Fallback type when no heuristic matches.

    Returns:
        List of :class:`MarkdownChunk` objects, never empty (raises on empty file).
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    if not text.strip():
        msg = f"File is empty: {path}"
        raise ValueError(msg)

    tags = list(default_tags or [])
    memory_type = default_type
    source = str(path)

    # 1. Front-matter
    meta, body = _parse_front_matter(text)
    if "tags" in meta:
        raw = meta["tags"]
        extra = raw if isinstance(raw, list) else [str(raw)]
        tags = list({*tags, *extra})
    if "type" in meta:
        memory_type = str(meta["type"])

    # 2. Header-based splitting
    header_matches = list(_HEADER_RE.finditer(body))
    if header_matches:
        chunks: list[MarkdownChunk] = []
        for i, match in enumerate(header_matches):
            title = match.group(2).strip()
            start = match.start()
            end = header_matches[i + 1].start() if i + 1 < len(header_matches) else len(body)
            section_body = body[start:end].strip()
            if not section_body:
                continue
            section_type = _infer_type(title, memory_type)
            chunks.append(MarkdownChunk(content=section_body, type=section_type, tags=list(tags), source=source))
        if chunks:
            return chunks

    # 3. Paragraph-based fallback
    raw_chunks = _paragraph_chunks(body.strip())
    if not raw_chunks:
        raw_chunks = [body.strip()]
    return [MarkdownChunk(content=c, type=memory_type, tags=list(tags), source=source) for c in raw_chunks]
