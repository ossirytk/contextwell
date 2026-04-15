"""Embedding model wrapper — sentence-transformers with lazy load.

The embedding provider and model are configurable via environment variables:

- ``CONTEXTWELL_EMBED_PROVIDER`` — ``"sentence-transformers"`` (default) or ``"openai"``
- ``CONTEXTWELL_EMBED_MODEL`` — model name; defaults to ``"all-MiniLM-L6-v2"``
  for sentence-transformers, or ``"text-embedding-3-small"`` for OpenAI
"""

from __future__ import annotations

import os

_model = None


def _provider() -> str:
    return os.getenv("CONTEXTWELL_EMBED_PROVIDER", "sentence-transformers")


def _model_name() -> str:
    """Return the configured embedding model name."""
    if _provider() == "openai":
        return os.getenv("CONTEXTWELL_EMBED_MODEL", "text-embedding-3-small")
    return os.getenv("CONTEXTWELL_EMBED_MODEL", "all-MiniLM-L6-v2")


def _get_model():  # noqa: ANN202
    global _model  # noqa: PLW0603
    if _model is None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        _model = SentenceTransformer(_model_name())
    return _model


def reset_model() -> None:
    """Clear the cached model, forcing a reload on the next embed call.

    Useful in tests or when ``CONTEXTWELL_EMBED_MODEL`` changes at runtime.
    """
    global _model  # noqa: PLW0603
    _model = None


def embed(text: str) -> list[float]:
    """Return a normalised embedding vector for the given text."""
    if _provider() == "openai":
        from contextwell.embedder_openai import embed as _embed  # noqa: PLC0415

        return _embed(text)
    model = _get_model()
    return model.encode(text, normalize_embeddings=True).tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Return normalised embedding vectors for a batch of texts."""
    if _provider() == "openai":
        from contextwell.embedder_openai import embed_batch as _embed_batch  # noqa: PLC0415

        return _embed_batch(texts)
    model = _get_model()
    return model.encode(texts, normalize_embeddings=True).tolist()
