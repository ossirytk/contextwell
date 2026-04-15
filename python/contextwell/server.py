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


def _project_id_for_scope(scope: str, cwd: str | None = None, *, allow_source_hint: bool = False) -> str | None:
    """Resolve project_id for project scope, or None for non-project scopes."""
    if scope != "project":
        return None
    from contextwell.project import detect_project_id  # noqa: PLC0415

    project_id = detect_project_id(cwd)
    if not project_id:
        msg = "Unable to detect project context for scope='project'. Run from a git repository."
        if allow_source_hint:
            msg += " You can also pass source='cwd:<path>'."
        raise ValueError(msg)
    return project_id


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
    from contextwell.store import store  # noqa: PLC0415

    cwd: str | None = None
    clean_source: str | None = source or None
    if source and source.startswith("cwd:"):
        cwd, _, rest = source[4:].partition("|")
        clean_source = rest or None

    project_id = _project_id_for_scope(scope, cwd, allow_source_hint=True)
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
    tags: list[str] | None = None,
    k: int = 10,
) -> list[dict]:
    """Search memories by meaning using semantic similarity.

    Args:
        query: Natural language search query.
        scope: Filter by scope ('project' or 'global'). Empty means all.
               For 'project' scope, git root is auto-detected from server CWD.
        type: Filter by memory type. Empty means all types.
        tags: Only return memories that have at least one of these tags.
        k: Maximum number of results to return.
    """
    from contextwell.embedder import embed  # noqa: PLC0415
    from contextwell.store import recall as _recall  # noqa: PLC0415

    embedding = embed(query)
    project_id = _project_id_for_scope(scope) or ""
    return _recall(embedding, scope=scope, memory_type=type, project_id=project_id or "", tags=tags, k=k)


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
    tags: list[str] | None = None,
    limit: int = 50,
) -> list[dict]:
    """Browse stored memories with optional filters.

    Args:
        scope: Filter by scope. Empty means all.
               For 'project' scope, git root is auto-detected from server CWD.
        type: Filter by memory type. Empty means all.
        tags: Only return memories that have at least one of these tags.
        limit: Maximum number of results.
    """
    from contextwell.store import scan  # noqa: PLC0415

    project_id = _project_id_for_scope(scope) or ""
    return scan(scope=scope, memory_type=type, project_id=project_id or "", tags=tags, limit=limit)


@mcp.tool
@mcp.tool
def update(
    memory_id: str,
    content: str | None = None,
    type: MemoryType | None = None,  # noqa: A002
    tags: list[str] | None = None,
    source: str | None = None,
) -> str:
    """Update fields of an existing memory in-place.

    Only the fields you provide are changed; the rest are left untouched.
    The memory's original ID and created_at are preserved. updated_at is
    set automatically. Re-embedding is performed automatically if content changes.

    Args:
        memory_id: Full or partial (first 8 chars) ID of the memory to update.
        content: New text content. Triggers automatic re-embedding.
        type: New memory type (code, chat, decision, todo, or fact).
        tags: Replacement tag list (replaces existing tags entirely).
        source: New source reference.
    """
    from contextwell.store import update as _update  # noqa: PLC0415

    if all(v is None for v in (content, type, tags, source)):
        return "Nothing to update — provide at least one field to change."

    new_embedding: list[float] | None = None
    if content is not None:
        from contextwell.embedder import embed  # noqa: PLC0415

        new_embedding = embed(content)

    found = _update(
        memory_id,
        content=content,
        memory_type=type,
        tags=tags,
        source=source,
        new_embedding=new_embedding,
    )
    if found:
        return f"Memory {memory_id[:8]} updated."
    return f"No memory found with ID {memory_id[:8]}."


def remember_file(
    path: str,
    scope: MemoryScope = "global",
    tags: list[str] | None = None,
    type_hint: MemoryType = "fact",
    source: str = "",
) -> str:
    """Ingest a markdown file and store its sections as individual memories.

    Splits on ## / ### headers (each section becomes one memory). Falls back to
    paragraph-based chunking (~800 chars) for files without headers. YAML
    front-matter (---) sets default type, tags, and scope for the whole file.

    Args:
        path: Absolute or relative path to the markdown file to import.
        scope: 'project' (tied to current git repo) or 'global'.
               For project scope, prefix source with 'cwd:<path>' to set
               the working directory used for git root detection.
        tags: Additional tags applied to every memory created from this file.
        type_hint: Fallback memory type when no heuristic matches a section title.
                   Section titles containing 'decision', 'todo', 'code', etc. are
                   detected automatically.
        source: Optional origin hint. Supports 'cwd:<path>' prefix (same as
                remember) to set the working directory for project-scope detection.
    """
    from contextwell.embedder import embed  # noqa: PLC0415
    from contextwell.markdown_import import parse  # noqa: PLC0415
    from contextwell.store import store  # noqa: PLC0415

    cwd: str | None = None
    if source and source.startswith("cwd:"):
        cwd, _, source = source[4:].partition("|")
        source = source or ""

    project_id = _project_id_for_scope(scope, cwd, allow_source_hint=True)

    try:
        chunks = parse(path, default_tags=tags or [], default_type=type_hint)
    except (OSError, ValueError) as exc:
        return f"Error reading file: {exc}"

    ids: list[str] = []
    for chunk in chunks:
        memory = Memory(
            content=chunk.content,
            type=chunk.type,  # type: ignore[arg-type]
            scope=scope,
            project_id=project_id,
            tags=chunk.tags,
            source=source or chunk.source,
        )
        memory.embedding = embed(chunk.content)
        ids.append(store(memory))

    scope_label = f"project:{project_id[:8]}" if project_id else scope
    return (
        f"Imported {len(ids)} memor{'y' if len(ids) == 1 else 'ies'} "
        f"from '{path}' [{type_hint}|{scope_label}]: " + ", ".join(f"#{mid[:8]}" for mid in ids)
    )


def run() -> None:
    mcp.run()


if __name__ == "__main__":
    run()
