"""Tests for the markdown_import parser."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from contextwell.markdown_import import MarkdownChunk, parse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Header-based splitting
# ---------------------------------------------------------------------------


def test_header_split_basic(tmp_path: Path) -> None:
    """Each ## section becomes a separate chunk."""
    md = _write(
        tmp_path,
        "plan.md",
        "## Introduction\nSome intro text.\n\n## Decision: use LanceDB\nWe chose LanceDB.\n",
    )
    chunks = parse(md)
    assert len(chunks) == 2
    assert "Introduction" in chunks[0].content
    assert "LanceDB" in chunks[1].content


def test_header_type_heuristic_decision(tmp_path: Path) -> None:
    """A section titled 'Decision: ...' is inferred as type='decision'."""
    md = _write(tmp_path, "d.md", "## Decision: Use Postgres\nBecause it scales.\n")
    chunks = parse(md)
    assert chunks[0].type == "decision"


def test_header_type_heuristic_todo(tmp_path: Path) -> None:
    """A section titled 'TODO' is inferred as type='todo'."""
    md = _write(tmp_path, "t.md", "## TODO: fix auth\nNeeds work.\n")
    chunks = parse(md)
    assert chunks[0].type == "todo"


def test_header_type_heuristic_code(tmp_path: Path) -> None:
    """A section titled 'Implementation' is inferred as type='code'."""
    md = _write(tmp_path, "c.md", "## Implementation\n```python\npass\n```\n")
    chunks = parse(md)
    assert chunks[0].type == "code"


def test_header_type_fallback(tmp_path: Path) -> None:
    """Unknown section title falls back to the provided default type."""
    md = _write(tmp_path, "f.md", "## Background\nSome context.\n")
    chunks = parse(md, default_type="chat")
    assert chunks[0].type == "chat"


def test_h3_headers_split(tmp_path: Path) -> None:
    """### headers also trigger splitting."""
    md = _write(
        tmp_path,
        "nested.md",
        "### Alpha\nContent A.\n### Beta\nContent B.\n",
    )
    chunks = parse(md)
    assert len(chunks) == 2


def test_h1_header_does_not_split(tmp_path: Path) -> None:
    """Top-level # headers do not trigger section splitting."""
    md = _write(tmp_path, "h1.md", "# Title\nBody line one.\nBody line two.\n")
    chunks = parse(md)
    assert len(chunks) == 1


def test_header_source_is_file_path(tmp_path: Path) -> None:
    """chunk.source is the string path of the file."""
    md = _write(tmp_path, "src.md", "## Section\nBody.\n")
    chunks = parse(md)
    assert chunks[0].source == str(md)


def test_tags_propagated_to_all_chunks(tmp_path: Path) -> None:
    """default_tags are applied to every chunk."""
    md = _write(tmp_path, "tagged.md", "## A\ntext\n## B\ntext\n")
    chunks = parse(md, default_tags=["arch", "decisions"])
    assert all("arch" in c.tags for c in chunks)
    assert all("decisions" in c.tags for c in chunks)


# ---------------------------------------------------------------------------
# YAML front-matter
# ---------------------------------------------------------------------------


def test_front_matter_type(tmp_path: Path) -> None:
    """Front-matter 'type:' overrides default_type."""
    md = _write(
        tmp_path,
        "fm.md",
        "---\ntype: decision\n---\n## Section\nBody.\n",
    )
    chunks = parse(md)
    assert chunks[0].type == "decision"


def test_front_matter_tags_merged(tmp_path: Path) -> None:
    """Front-matter tags are merged with default_tags."""
    md = _write(
        tmp_path,
        "fm_tags.md",
        "---\ntags: [frontend, css]\n---\n## Section\nBody.\n",
    )
    chunks = parse(md, default_tags=["project"])
    assert "frontend" in chunks[0].tags
    assert "project" in chunks[0].tags


def test_front_matter_stripped_from_content(tmp_path: Path) -> None:
    """The front-matter block does not appear in any chunk content."""
    md = _write(
        tmp_path,
        "strip.md",
        "---\ntype: fact\n---\n## Section\nReal content.\n",
    )
    chunks = parse(md)
    assert "---" not in chunks[0].content
    assert "type: fact" not in chunks[0].content


# ---------------------------------------------------------------------------
# Paragraph-based fallback
# ---------------------------------------------------------------------------


def test_flat_file_stored_as_single_chunk(tmp_path: Path) -> None:
    """A short flat file (no headers) becomes one chunk."""
    md = _write(tmp_path, "flat.md", "Just some plain text without any headers.\n")
    chunks = parse(md)
    assert len(chunks) == 1
    assert "plain text" in chunks[0].content


def test_flat_file_chunked_when_large(tmp_path: Path) -> None:
    """A flat file exceeding chunk size is split into multiple chunks."""
    # Build a file well over 800 chars with clear paragraph breaks
    paragraphs = [f"Paragraph {i}: " + ("word " * 30) for i in range(10)]
    content = "\n\n".join(paragraphs)
    md = _write(tmp_path, "large_flat.md", content)
    chunks = parse(md)
    assert len(chunks) > 1
    # Every chunk should be non-empty
    assert all(c.content.strip() for c in chunks)


def test_parse_raises_on_invalid_overlap() -> None:
    """Invalid paragraph overlap settings raise a clear error."""
    from contextwell import markdown_import  # noqa: PLC0415

    with pytest.raises(ValueError, match="overlap"):
        markdown_import._paragraph_chunks("plain text", chunk_size=10, overlap=10)  # noqa: SLF001


def test_empty_file_raises(tmp_path: Path) -> None:
    """An empty file raises ValueError."""
    md = _write(tmp_path, "empty.md", "   \n  \n")
    with pytest.raises(ValueError, match="empty"):
        parse(md)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_returns_markdown_chunk_instances(tmp_path: Path) -> None:
    """parse always returns a list of MarkdownChunk objects."""
    md = _write(tmp_path, "typed.md", "## Section\nContent.\n")
    chunks = parse(md)
    assert all(isinstance(c, MarkdownChunk) for c in chunks)
