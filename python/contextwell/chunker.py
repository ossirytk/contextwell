"""Word-based text chunker for long-form content.

Activated by setting ``CONTEXTWELL_CHUNKING=1``. When enabled, ``remember``
automatically splits content that exceeds ``CONTEXTWELL_CHUNK_SIZE`` words
(default 400) into overlapping chunks of the same size with
``CONTEXTWELL_CHUNK_OVERLAP`` words of overlap (default 50).

No external dependencies — splits on whitespace.
"""

from __future__ import annotations

import os


def _chunk_size() -> int:
    return int(os.getenv("CONTEXTWELL_CHUNK_SIZE", "400"))


def _chunk_overlap() -> int:
    return int(os.getenv("CONTEXTWELL_CHUNK_OVERLAP", "50"))


def chunking_enabled() -> bool:
    """Return True when ``CONTEXTWELL_CHUNKING=1`` is set."""
    return os.getenv("CONTEXTWELL_CHUNKING") == "1"


def chunk_text(
    text: str,
    max_words: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    """Split *text* into overlapping word-based chunks.

    Returns a single-element list when the word count is within *max_words*
    (i.e., no splitting needed). Each chunk is a plain string of space-
    joined words.

    Args:
        text: The text to split.
        max_words: Maximum words per chunk. Defaults to ``CONTEXTWELL_CHUNK_SIZE``.
        overlap: Number of words to repeat at the start of each subsequent
                 chunk. Defaults to ``CONTEXTWELL_CHUNK_OVERLAP``.
    """
    default_overlap = overlap is None
    if max_words is None:
        max_words = _chunk_size()

    if max_words < 1:
        msg = "max_words must be >= 1"
        raise ValueError(msg)
    if overlap is None:
        overlap = _chunk_overlap()
    if overlap < 0 or overlap >= max_words:
        if default_overlap and overlap >= max_words:
            overlap = max_words - 1
        else:
            msg = "overlap must satisfy 0 <= overlap < max_words"
            raise ValueError(msg)
    words = text.split()

    if len(words) <= max_words:
        return [text]

    chunks: list[str] = []
    step = max_words - overlap
    i = 0
    while i < len(words):
        chunk_words = words[i : i + max_words]
        chunks.append(" ".join(chunk_words))
        if i + max_words >= len(words):
            break
        i += step

    return chunks
