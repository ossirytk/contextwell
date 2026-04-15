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
        "Use `list_memories` to browse stored memories with filters. "
        "For project-scoped memories, set scope='project' and contextwell will "
        "auto-detect the current git repository as the project context."
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
        scope: 'project' (tied to current git repo) or 'global'.
        tags: Optional labels for filtering later.
        source: Optional origin reference (file path, URL, conversation turn).
                For project-scoped memories, prefix with 'cwd:<path>' to set
                the working directory used for git root detection, e.g.
                'cwd:D:/myproject'. Otherwise the server CWD is used.
    """
    from contextwell.embedder import embed  # noqa: PLC0415
    from contextwell.project import detect_project_id  # noqa: PLC0415
    from contextwell.store import store  # noqa: PLC0415

    cwd: str | None = None
    clean_source: str | None = source or None
    if source and source.startswith("cwd:"):
        cwd, _, rest = source[4:].partition("|")
        clean_source = rest or None

    project_id = detect_project_id(cwd) if scope == "project" else None
    if scope == "project" and not project_id:
        msg = "Unable to detect project context for scope='project'."
        raise ValueError(msg)
    memory = Memory(
        content=content,
        type=type,
        scope=scope,
        project_id=project_id,
        tags=tags or [],
        source=clean_source,
    )
    memory.embedding = embed(content)
    memory_id = store(memory)
    scope_label = f"project:{project_id[:8]}" if project_id else scope
    return f"Remembered [{type}|{scope_label}] #{memory_id[:8]}: {content[:80]}"


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
               For 'project' scope, git root is auto-detected from server CWD.
        type: Filter by memory type. Empty means all types.
        k: Maximum number of results to return.
    """
    from contextwell.embedder import embed  # noqa: PLC0415
    from contextwell.project import detect_project_id  # noqa: PLC0415
    from contextwell.store import recall as _recall  # noqa: PLC0415

    embedding = embed(query)
    project_id = detect_project_id() if scope == "project" else ""
    if scope == "project" and not project_id:
        msg = "Unable to detect project context for scope='project'."
        raise ValueError(msg)
    return _recall(embedding, scope=scope, memory_type=type, project_id=project_id or "", k=k)


@mcp.tool
def forget(memory_id: str) -> str:
    """Delete a memory by its ID.

    Args:
        memory_id: Full or partial (first 8 chars) ID of the memory to delete.
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
               For 'project' scope, git root is auto-detected from server CWD.
        type: Filter by memory type. Empty means all.
        limit: Maximum number of results.
    """
    from contextwell.project import detect_project_id  # noqa: PLC0415
    from contextwell.store import scan  # noqa: PLC0415

    project_id = detect_project_id() if scope == "project" else ""
    if scope == "project" and not project_id:
        msg = "Unable to detect project context for scope='project'."
        raise ValueError(msg)
    return scan(scope=scope, memory_type=type, project_id=project_id or "", limit=limit)


def run() -> None:
    mcp.run()


if __name__ == "__main__":
    run()
