"""Cross-encoder reranking using sentence-transformers CrossEncoder."""

from __future__ import annotations

import logging
import os

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_LOG = logging.getLogger(__name__)

_reranker = None  # module-level cache


def _reranker_model_name() -> str:
    return os.getenv("CONTEXTWELL_RERANK_MODEL", _DEFAULT_MODEL)


def _get_reranker():  # noqa: ANN202
    global _reranker  # noqa: PLW0603
    if _reranker is None:
        from sentence_transformers import CrossEncoder  # noqa: PLC0415

        _reranker = CrossEncoder(_reranker_model_name())
    return _reranker


def reset_reranker() -> None:
    """Clear the cached reranker (used in tests and after model changes)."""
    global _reranker  # noqa: PLW0603
    _reranker = None


def rerank(query: str, rows: list[dict], k: int) -> list[dict]:
    """Re-score *rows* using a cross-encoder and return the top-*k* by relevance.

    Pairs each row's ``content`` with *query* and passes them through the
    cross-encoder. Rows are sorted by descending score and limited to *k*.

    Falls back to returning the original *rows[:k]* if the cross-encoder
    cannot be loaded (e.g., sentence-transformers not installed).
    """
    if not rows or not query:
        return rows[:k]
    try:
        model = _get_reranker()
    except (ImportError, OSError):
        return rows[:k]
    try:
        pairs = [(query, str(row.get("content", ""))) for row in rows]
        # Handle both ndarray-like and plain list predictions across model versions.
        predictions = model.predict(pairs)
        scores: list[float] = predictions.tolist() if hasattr(predictions, "tolist") else list(predictions)
        ranked = sorted(zip(rows, scores, strict=True), key=lambda x: x[1], reverse=True)
        return [row for row, _ in ranked[:k]]
    except Exception:
        _LOG.exception("Unexpected reranker failure; falling back to original ranking.")
        return rows[:k]
