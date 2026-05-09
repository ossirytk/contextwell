"""File parsers for contextwell memory ingestion — non-markdown formats.

Supported formats:
- ``.org``  — Org-mode (headline-based splitting on ``* ``/``** `` boundaries)
- ``.txt``  — Plain text (paragraph-based splitting, same logic as markdown fallback)
- Source code (``.py``, ``.js``, ``.ts``, ``.rs``, ``.go``, ``.java``, ``.c``,
  ``.cpp``, ``.cs``, ``.rb``, ``.kt``, ``.swift``) — regex function/class
  detection with character-size fallback chunking.

Usage::

    from contextwell.file_import import parse
    chunks = parse("/path/to/file.py", default_tags=["python"])
    # Each chunk is a MarkdownChunk-compatible namedtuple: content, type, tags, source
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# Characters per paragraph chunk (txt / code fallback).
_CHUNK_SIZE = 800
_CHUNK_OVERLAP = 100

# Extensions recognised as source code.
_SOURCE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".rs",
    ".go",
    ".java",
    ".c",
    ".cpp",
    ".cc",
    ".cxx",
    ".cs",
    ".rb",
    ".kt",
    ".swift",
    ".php",
    ".scala",
    ".lua",
    ".r",
    ".m",
    ".sh",
    ".bash",
}


@dataclass
class FileChunk:
    """A single chunk extracted from a file, ready for embedding."""

    content: str
    type: str = "code"
    tags: list[str] = field(default_factory=list)
    source: str = ""


# ---------------------------------------------------------------------------
# Org-mode parser
# ---------------------------------------------------------------------------

_ORG_HEADLINE_RE = re.compile(r"^(\*{1,3})\s+(.+)$", re.MULTILINE)

# Map Org TODO keywords to memory types.
_ORG_TYPE_HINTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bTODO\b|\bNEXT\b|\bTASK\b", re.IGNORECASE), "todo"),
    (re.compile(r"\bDECISION\b|\bADR\b", re.IGNORECASE), "decision"),
    (re.compile(r"\bNOTE\b|\bSUMMARY\b", re.IGNORECASE), "chat"),
    (re.compile(r"\bCODE\b|\bIMPL\b|\bSNIPPET\b", re.IGNORECASE), "code"),
]


def _infer_org_type(title: str, default: str) -> str:
    for pattern, memory_type in _ORG_TYPE_HINTS:
        if pattern.search(title):
            return memory_type
    return default


def _parse_org(text: str, *, default_tags: list[str], default_type: str, source: str) -> list[FileChunk]:
    """Split an Org-mode file on headline boundaries (* / ** / ***).

    Each headline and its body becomes one chunk. Preamble text before the
    first headline (if any) is included as a standalone chunk.
    """
    headline_matches = list(_ORG_HEADLINE_RE.finditer(text))
    if not headline_matches:
        # No headlines — treat as plain text.
        return _parse_txt(text, default_tags=default_tags, default_type=default_type, source=source)

    chunks: list[FileChunk] = []

    # Preamble before the first headline.
    preamble = text[: headline_matches[0].start()].strip()
    if preamble:
        chunks.append(FileChunk(content=preamble, type="fact", tags=list(default_tags), source=source))

    for i, match in enumerate(headline_matches):
        title = match.group(2).strip()
        start = match.start()
        end = headline_matches[i + 1].start() if i + 1 < len(headline_matches) else len(text)
        section_body = text[start:end].strip()
        if not section_body:
            continue
        memory_type = _infer_org_type(title, default_type)
        chunks.append(FileChunk(content=section_body, type=memory_type, tags=list(default_tags), source=source))

    return chunks or [FileChunk(content=text.strip(), type=default_type, tags=list(default_tags), source=source)]


# ---------------------------------------------------------------------------
# Plain-text parser
# ---------------------------------------------------------------------------


def _paragraph_chunks(text: str, chunk_size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> list[str]:
    """Split *text* into overlapping paragraph-aligned chunks."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    current = ""

    for para in paragraphs:
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


def _parse_txt(text: str, *, default_tags: list[str], default_type: str, source: str) -> list[FileChunk]:
    """Split a plain-text file into paragraph-aligned chunks."""
    raw_chunks = _paragraph_chunks(text.strip())
    if not raw_chunks:
        raw_chunks = [text.strip()]
    return [FileChunk(content=c, type=default_type, tags=list(default_tags), source=source) for c in raw_chunks]


# ---------------------------------------------------------------------------
# Source code parser
# ---------------------------------------------------------------------------

# Top-level function/class patterns per language family.
# Each pattern captures the leading line(s) of a definition.
_DEFINITION_PATTERNS: list[re.Pattern[str]] = [
    # Python: def / async def / class at column 0
    re.compile(r"^(?:async\s+)?def\s+\w+|^class\s+\w+", re.MULTILINE),
    # Rust: fn / pub fn / impl / struct / enum / trait at column 0
    re.compile(
        r"^(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+\w+|^(?:pub\s+)?(?:impl|struct|enum|trait)\b", re.MULTILINE
    ),
    # Go: func
    re.compile(r"^func\s+", re.MULTILINE),
    # Java/C#/Kotlin/Swift: typical method/class signatures
    re.compile(
        r"^(?:public|private|protected|internal|static|override|abstract|sealed)[\w\s<>]*(?:class|interface|fun|func|void|int|string|bool|var|let)\s+\w+",
        re.MULTILINE,
    ),
    # JavaScript/TypeScript: function / class / export function / export class / const f = () =>
    re.compile(
        r"^(?:export\s+)?(?:async\s+)?(?:function|class)\s+\w+|^(?:export\s+)?const\s+\w+\s*=\s*(?:async\s+)?\(",
        re.MULTILINE,
    ),
]


def _find_definition_boundaries(text: str) -> list[int]:
    """Return sorted start positions of top-level definitions found in *text*."""
    positions: set[int] = set()
    for pattern in _DEFINITION_PATTERNS:
        for m in pattern.finditer(text):
            positions.add(m.start())
    return sorted(positions)


def _parse_source(text: str, *, default_tags: list[str], source: str) -> list[FileChunk]:
    """Split source code into function/class-level chunks.

    Falls back to character-size chunking when no definition boundaries are found.
    """
    boundaries = _find_definition_boundaries(text)
    if not boundaries:
        # Fall back to plain character chunking.
        raw = _paragraph_chunks(text, chunk_size=_CHUNK_SIZE, overlap=_CHUNK_OVERLAP)
        if not raw:
            raw = [text.strip()]
        return [FileChunk(content=c, type="code", tags=list(default_tags), source=source) for c in raw if c.strip()]

    chunks: list[FileChunk] = []

    # Optional preamble before the first definition (imports, module-level code).
    preamble = text[: boundaries[0]].strip()
    if preamble:
        chunks.append(FileChunk(content=preamble, type="code", tags=list(default_tags), source=source))

    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
        section = text[start:end].strip()
        if not section:
            continue
        # Large sections are split further to stay within the chunk size.
        if len(section) > _CHUNK_SIZE * 3:
            sub_chunks = _paragraph_chunks(section, chunk_size=_CHUNK_SIZE, overlap=_CHUNK_OVERLAP)
            chunks.extend(
                FileChunk(content=sub, type="code", tags=list(default_tags), source=source)
                for sub in sub_chunks
                if sub.strip()
            )
        else:
            chunks.append(FileChunk(content=section, type="code", tags=list(default_tags), source=source))

    return chunks or [FileChunk(content=text.strip(), type="code", tags=list(default_tags), source=source)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse(
    path: str | Path,
    *,
    default_tags: list[str] | None = None,
    default_type: str = "fact",
) -> list[FileChunk]:
    """Parse a non-markdown file into a list of :class:`FileChunk` objects.

    Dispatches to the appropriate strategy based on the file extension:

    - ``.org``  — headline-based Org-mode splitting
    - Source code extensions — regex function/class detection
    - Everything else (incl. ``.txt``) — paragraph-based splitting

    Args:
        path: Absolute or relative path to the file.
        default_tags: Base tags applied to every chunk.
        default_type: Fallback memory type. Source code files default to
            ``"code"`` regardless of this value.

    Returns:
        List of :class:`FileChunk` objects. Never empty (raises on empty file).

    Raises:
        ValueError: When the file is empty.
        OSError: On read errors.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")

    if not text.strip():
        msg = f"File is empty: {path}"
        raise ValueError(msg)

    tags = list(default_tags or [])
    source = str(path)
    suffix = path.suffix.lower()

    if suffix == ".org":
        return _parse_org(text, default_tags=tags, default_type=default_type, source=source)
    if suffix in _SOURCE_EXTENSIONS:
        return _parse_source(text, default_tags=tags, source=source)
    # .txt and anything else
    return _parse_txt(text, default_tags=tags, default_type=default_type, source=source)
