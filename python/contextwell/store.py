"""LanceDB-backed memory store with hybrid search support."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from contextwell.schema import Memory

DB_PATH = Path.home() / ".contextwell" / "memories"


def _get_table():  # noqa: ANN202
    import lancedb  # noqa: PLC0415

    db = lancedb.connect(str(DB_PATH))
    if "memories" not in db.table_names():
        import pyarrow as pa  # noqa: PLC0415

        schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("content", pa.string()),
                pa.field("type", pa.string()),
                pa.field("scope", pa.string()),
                pa.field("project_id", pa.string()),
                pa.field("tags", pa.list_(pa.string())),
                pa.field("source", pa.string()),
                pa.field("created_at", pa.string()),
                pa.field("embedding", pa.list_(pa.float32(), 384)),
            ]
        )
        db.create_table("memories", schema=schema)
    return db.open_table("memories")


def store(memory: Memory) -> str:
    """Persist a memory to LanceDB. Returns the memory ID."""
    table = _get_table()
    table.add(
        [
            {
                "id": memory.id,
                "content": memory.content,
                "type": memory.type,
                "scope": memory.scope,
                "project_id": memory.project_id or "",
                "tags": memory.tags,
                "source": memory.source or "",
                "created_at": memory.created_at.isoformat(),
                "embedding": memory.embedding,
            }
        ]
    )
    return memory.id


def recall(
    embedding: list[float],
    scope: str = "",
    memory_type: str = "",
    k: int = 10,
) -> list[dict]:
    """Vector search with optional metadata filters. Returns top-k results."""
    table = _get_table()
    query = table.search(embedding).limit(k)
    if scope:
        query = query.where(f"scope = '{scope}'")
    if memory_type:
        query = query.where(f"type = '{memory_type}'")
    return query.to_list()


def forget(memory_id: str) -> bool:
    """Delete a memory by ID. Returns True if found and deleted."""
    table = _get_table()
    before = table.count_rows()
    table.delete(f"id = '{memory_id}'")
    return table.count_rows() < before
