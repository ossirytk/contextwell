"""LanceDB-backed memory store with hybrid search support."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lancedb.table import Table

    from contextwell.schema import Memory

DB_PATH = Path.home() / ".contextwell" / "memories"
_EMBEDDING_DIM = 384


def _escape_literal(value: str) -> str:
    """Escape quote and control chars for LanceDB filter literals."""
    return value.replace("'", "''").replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def _ensure_scalar_indexes(table: Table) -> None:
    """Create missing scalar indexes used by metadata filters."""
    indexed_columns = {column for index in table.list_indices() for column in index.columns}
    for column in ("scope", "type", "project_id"):
        if column not in indexed_columns:
            table.create_scalar_index(column)


def _where_clauses(
    scope: str = "",
    memory_type: str = "",
    project_id: str = "",
    tags: list[str] | None = None,
) -> list[str]:
    clauses = []
    if scope:
        clauses.append(f"scope = '{_escape_literal(scope)}'")
    if memory_type:
        clauses.append(f"type = '{_escape_literal(memory_type)}'")
    if project_id:
        clauses.append(f"project_id = '{_escape_literal(project_id)}'")
    if tags:
        # Any-match: memory must contain at least one of the requested tags.
        tag_conditions = " OR ".join(f"array_has(tags, '{_escape_literal(t)}')" for t in tags)
        clauses.append(f"({tag_conditions})")
    return clauses


def _get_db():  # noqa: ANN202
    import lancedb  # noqa: PLC0415

    DB_PATH.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(DB_PATH))


def _ensure_updated_at_column(table: Table) -> None:
    """Add the updated_at column to existing tables that pre-date this field."""
    if "updated_at" not in {f.name for f in table.schema}:
        table.add_columns({"updated_at": "''"})


def _get_table():  # noqa: ANN202
    import pyarrow as pa  # noqa: PLC0415

    db = _get_db()
    existing = db.list_tables().tables
    if "memories" not in existing:
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
                pa.field("updated_at", pa.string()),
                pa.field("embedding", pa.list_(pa.float32(), _EMBEDDING_DIM)),
            ]
        )
        tbl = db.create_table("memories", schema=schema)
        _ensure_scalar_indexes(tbl)
        return tbl
    tbl = db.open_table("memories")
    _ensure_scalar_indexes(tbl)
    _ensure_updated_at_column(tbl)
    return tbl


def _clean(row: dict) -> dict:
    """Strip LanceDB internal columns from a result row."""
    row.pop("_distance", None)
    row.pop("embedding", None)
    return row


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
    project_id: str = "",
    tags: list[str] | None = None,
    k: int = 10,
) -> list[dict]:
    """Vector search with optional metadata filters. Returns top-k results."""
    table = _get_table()

    clauses = _where_clauses(scope=scope, memory_type=memory_type, project_id=project_id, tags=tags)

    query = table.search(embedding).limit(k)
    if clauses:
        query = query.where(" AND ".join(clauses))

    return [_clean(row) for row in query.to_list()]


def scan(
    scope: str = "",
    memory_type: str = "",
    project_id: str = "",
    tags: list[str] | None = None,
    limit: int = 50,
) -> list[dict]:
    """Full-table scan with optional metadata filters. No vector required."""
    table = _get_table()

    clauses = _where_clauses(scope=scope, memory_type=memory_type, project_id=project_id, tags=tags)

    query = table.search().limit(limit)
    if clauses:
        query = query.where(" AND ".join(clauses))

    return [_clean(row) for row in query.to_list()]


def forget(memory_id: str) -> bool:
    """Delete a memory by ID. Returns True if found and deleted."""
    table = _get_table()
    before = table.count_rows()
    table.delete(f"id = '{_escape_literal(memory_id)}'")
    return table.count_rows() < before


def update(
    memory_id: str,
    content: str | None = None,
    memory_type: str | None = None,
    tags: list[str] | None = None,
    source: str | None = None,
    new_embedding: list[float] | None = None,
) -> bool:
    """Update fields of an existing memory in-place. Returns True if found and updated.

    Supports partial ID matching (first 8 chars). Re-embedding must be done by
    the caller and passed via *new_embedding* when *content* changes.
    """
    from datetime import UTC, datetime  # noqa: PLC0415

    table = _get_table()

    # Resolve full ID from partial match if needed.
    where = (
        f"id = '{_escape_literal(memory_id)}'"
        if len(memory_id) > 8  # noqa: PLR2004
        else f"id LIKE '{_escape_literal(memory_id)}%'"
    )

    values: dict[str, object] = {"updated_at": datetime.now(UTC).isoformat()}
    if content is not None:
        values["content"] = content
    if memory_type is not None:
        values["type"] = memory_type
    if tags is not None:
        values["tags"] = tags
    if source is not None:
        values["source"] = source
    if new_embedding is not None:
        values["embedding"] = new_embedding

    table.update(where=where, values=values)
    # Row count doesn't change on updates, so verify existence by re-querying.
    return bool(table.search().where(where).limit(1).to_list())
