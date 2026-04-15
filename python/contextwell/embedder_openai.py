"""OpenAI embedding adapter for contextwell.

Activated by setting ``CONTEXTWELL_EMBED_PROVIDER=openai``.
Requires the ``openai`` package (``pip install openai``) and a valid
``OPENAI_API_KEY`` environment variable.

Relevant environment variables:

- ``CONTEXTWELL_EMBED_PROVIDER=openai``   — enable this adapter
- ``OPENAI_API_KEY``                       — required
- ``CONTEXTWELL_EMBED_MODEL``             — model name (default: text-embedding-3-small)
- ``CONTEXTWELL_EMBED_DIM``               — must match the model output dimension
"""

from __future__ import annotations

import os


def _client():  # noqa: ANN202
    from openai import OpenAI  # noqa: PLC0415

    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def _model_name() -> str:
    return os.getenv("CONTEXTWELL_EMBED_MODEL", "text-embedding-3-small")


def embed(text: str) -> list[float]:
    """Return an embedding vector for *text* via the OpenAI Embeddings API."""
    response = _client().embeddings.create(input=[text], model=_model_name())
    return response.data[0].embedding


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Return embedding vectors for a batch of texts via the OpenAI Embeddings API."""
    response = _client().embeddings.create(input=texts, model=_model_name())
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
