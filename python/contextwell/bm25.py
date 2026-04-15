"""BM25 sparse retrieval over contextwell memory rows.

Requires the optional ``rank-bm25`` dependency:
    pip install contextwell[hybrid]
    # or: uv add rank-bm25
"""

from __future__ import annotations


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def bm25_search(rows: list[dict], query: str, k: int) -> list[str]:
    """Return up to *k* memory IDs ranked by BM25 relevance to *query*.

    *rows* must each contain an ``"id"`` and a ``"content"`` key (the format
    returned by :func:`contextwell.store.scan`).
    Returns an empty list when *rows* is empty or *query* is blank.
    """
    if not rows or not query.strip():
        return []

    from rank_bm25 import BM25Okapi  # noqa: PLC0415

    corpus = [_tokenize(row["content"]) for row in rows]
    index = BM25Okapi(corpus)
    scores = index.get_scores(_tokenize(query))

    ranked = sorted(zip(scores, rows, strict=False), key=lambda x: x[0], reverse=True)
    return [row["id"] for _, row in ranked[:k]]
