"""Embedding model wrapper — sentence-transformers with lazy load."""

from __future__ import annotations

_model = None
_MODEL_NAME = "all-MiniLM-L6-v2"


def _get_model():  # noqa: ANN202
    global _model  # noqa: PLW0603
    if _model is None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed(text: str) -> list[float]:
    """Return a normalised embedding vector for the given text."""
    model = _get_model()
    return model.encode(text, normalize_embeddings=True).tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Return normalised embedding vectors for a batch of texts."""
    model = _get_model()
    return model.encode(texts, normalize_embeddings=True).tolist()
