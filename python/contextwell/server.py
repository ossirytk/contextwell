"""Contextwell MCP server — persistent semantic memory tools."""

from __future__ import annotations

from typing import Literal

from fastmcp import FastMCP

from contextwell.schema import Memory, MemoryScope, MemoryType

mcp = FastMCP(
    name="contextwell",
    instructions=(
        "Contextwell is a persistent semantic memory layer for Copilot. "
        "Use `remember` to store facts, decisions, code snippets, or notes. "
        "Use `recall` to search memory by meaning, not just keywords. "
        "Use `forget` to delete a memory by ID. "
        "Use `list_memories` to browse stored memories with filters."
    ),
)


@mcp.tool
def remember(
    content: str,
    type: MemoryType = "fact",  # noqa: A002
    scope: MemoryScope = "global",
    tags: list[str] | None = None,
    source: str = "",
) -> str:
    """Store a new memory.

    Args:
        content: The text to remember.
        type: Kind of memory — code, chat, decision, todo, or fact.
        scope: 'project' (tied to current project) or 'global'.
        tags: Optional labels for filtering later.
        source: Optional origin reference (file path, URL, conversation turn).
    """
    from contextwell.embedder import embed  # noqa: PLC0415
    from contextwell.store import store  # noqa: PLC0415

    memory = Memory(
        content=content,
        type=type,
        scope=scope,
        tags=tags or [],
        source=source or None,
    )
    memory.embedding = embed(content)
    memory_id = store(memory)
    return f"Remembered [{type}] #{memory_id[:8]}: {content[:80]}"


@mcp.tool
def recall(
    query: str,
    scope: Literal["project", "global", ""] = "",
    type: MemoryType | Literal[""] = "",  # noqa: A002
    k: int = 10,
) -> list[dict]:
    """Search memories by meaning using semantic similarity.

    Args:
        query: Natural language search query.
        scope: Filter by scope ('project' or 'global'). Empty means all.
        type: Filter by memory type. Empty means all types.
        k: Maximum number of results to return.
    """
    from contextwell.embedder import embed  # noqa: PLC0415
    from contextwell.store import recall as _recall  # noqa: PLC0415

    embedding = embed(query)
    results = _recall(embedding, scope=scope, memory_type=type, k=k)
    for r in results:
        r.pop("embedding", None)
    return results


@mcp.tool
def forget(memory_id: str) -> str:
    """Delete a memory by its ID.

    Args:
        memory_id: Full or partial ID of the memory to delete.
    """
    from contextwell.store import forget as _forget  # noqa: PLC0415

    deleted = _forget(memory_id)
    if deleted:
        return f"Memory {memory_id[:8]} deleted."
    return f"No memory found with ID {memory_id[:8]}."


@mcp.tool
def list_memories(
    scope: MemoryScope | Literal[""] = "",
    type: MemoryType | Literal[""] = "",  # noqa: A002
    limit: int = 50,
) -> list[dict]:
    """Browse stored memories with optional filters.

    Args:
        scope: Filter by scope. Empty means all.
        type: Filter by memory type. Empty means all.
        limit: Maximum number of results.
    """
    from contextwell.store import _get_table  # noqa: PLC0415

    table = _get_table()
    clauses = []
    if scope:
        clauses.append(f"scope = '{scope}'")
    if type:
        clauses.append(f"type = '{type}'")

    query = table.search().limit(limit)
    if clauses:
        query = query.where(" AND ".join(clauses))

    rows = query.to_list()
    for r in rows:
        r.pop("embedding", None)
    return rows


def run() -> None:
    mcp.run()


if __name__ == "__main__":
    run()
