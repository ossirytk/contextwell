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
    allow_duplicate: bool = False,
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
        allow_duplicate: Skip the near-duplicate check and store regardless.
    """
    from contextwell.embedder import embed  # noqa: PLC0415
    from contextwell.store import check_duplicate, store  # noqa: PLC0415

    cwd: str | None = None
    clean_source: str | None = source or None
    if source and source.startswith("cwd:"):
        cwd, _, rest = source[4:].partition("|")
        clean_source = rest or None

    project_id = _project_id_for_scope(scope, cwd, allow_source_hint=True)
    embedding = embed(content)

    if not allow_duplicate:
        duplicate = check_duplicate(embedding)
        if duplicate:
            dup_id = duplicate["id"]
            snippet = duplicate["content"][:80]
            return (
                f"⚠ Near-duplicate detected (similarity ≥ 95%): "
                f"#{dup_id[:8]} — {snippet}. "
                f"Pass allow_duplicate=True to store anyway."
            )

    memory = Memory(
        content=content,
        type=type,
        scope=scope,
        project_id=project_id,
        tags=tags or [],
        source=clean_source,
    )
    memory.embedding = embedding
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
    rerank: bool = False,
) -> list[dict]:
    """Search memories by meaning using semantic similarity.

    Args:
        query: Natural language search query.
        scope: Filter by scope ('project' or 'global'). Empty means all.
               For 'project' scope, git root is auto-detected from server CWD.
        type: Filter by memory type. Empty means all types.
        tags: Only return memories that have at least one of these tags.
        k: Maximum number of results to return.
        rerank: When True, fetch up to k*3 candidates then re-score with a
                cross-encoder (cross-encoder/ms-marco-MiniLM-L-6-v2) for
                higher-precision ordering. Adds ~100-300 ms latency.
                Most beneficial when k > 5.
    """
    from contextwell.embedder import embed  # noqa: PLC0415
    from contextwell.store import recall as _recall  # noqa: PLC0415

    embedding = embed(query)
    project_id = _project_id_for_scope(scope) or ""
    return _recall(
        embedding,
        query=query,
        scope=scope,
        memory_type=type,
        project_id=project_id or "",
        tags=tags,
        k=k,
        rerank=rerank,
    )


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
    since: str = "",
    until: str = "",
) -> list[dict]:
    """Browse stored memories with optional filters.

    Args:
        scope: Filter by scope. Empty means all.
               For 'project' scope, git root is auto-detected from server CWD.
        type: Filter by memory type. Empty means all.
        tags: Only return memories that have at least one of these tags.
        limit: Maximum number of results.
        since: ISO 8601 date/datetime lower bound for created_at (inclusive).
               Examples: "2025-01-01", "2025-03-15T09:00:00".
        until: ISO 8601 date/datetime upper bound for created_at (inclusive).
               Examples: "2025-03-31", "2025-03-31T23:59:59".
    """
    from contextwell.store import scan  # noqa: PLC0415

    project_id = _project_id_for_scope(scope) or ""
    return scan(
        scope=scope,
        memory_type=type,
        project_id=project_id or "",
        tags=tags,
        limit=limit,
        since=since,
        until=until,
    )


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


@mcp.tool
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


@mcp.tool
def remember_batch(
    memories: list[dict],
    allow_duplicate: bool = False,
) -> str:
    """Store multiple memories in a single call using batched embedding.

    Each item in *memories* is a dict with the same fields as ``remember``:

    - ``content`` (str, required): The text to remember.
    - ``type`` (str, default "fact"): code, chat, decision, todo, or fact.
    - ``scope`` (str, default "global"): 'project' or 'global'.
    - ``tags`` (list[str], default []): Labels for filtering.
    - ``source`` (str, default ""): Origin reference. Supports the
      ``cwd:<path>`` prefix for project-scope CWD resolution.

    All embeddings are computed in one batched model call, making this
    significantly faster than calling ``remember`` N times for large sets.

    Args:
        memories: List of memory dicts (see fields above).
        allow_duplicate: When True, skip the near-duplicate check for every
                         item and store all memories unconditionally.

    Returns:
        A summary line with counts of stored vs skipped items plus their IDs.
    """
    from contextwell.embedder import embed_batch  # noqa: PLC0415, I001
    from contextwell.store import check_duplicate, store as _store  # noqa: PLC0415

    if not memories:
        return "No memories provided."

    # Parse and normalise each item before hitting the model.
    parsed: list[dict] = []
    for raw in memories:
        content = str(raw.get("content", "")).strip()
        if not content:
            continue
        source_raw = str(raw.get("source", ""))
        cwd: str | None = None
        clean_source: str | None = source_raw or None
        if source_raw.startswith("cwd:"):
            cwd, _, rest = source_raw[4:].partition("|")
            clean_source = rest or None
        scope: str = str(raw.get("scope", "global"))
        parsed.append(
            {
                "content": content,
                "type": str(raw.get("type", "fact")),
                "scope": scope,
                "tags": list(raw.get("tags") or []),
                "source": clean_source,
                "cwd": cwd,
                "project_id": _project_id_for_scope(scope, cwd, allow_source_hint=True) or "",
            }
        )

    if not parsed:
        return "No valid memories (all items were empty)."

    embeddings = embed_batch([item["content"] for item in parsed])

    stored_ids: list[str] = []
    skipped: list[str] = []

    for item, embedding in zip(parsed, embeddings, strict=True):
        if not allow_duplicate:
            dup = check_duplicate(embedding)
            if dup:
                skipped.append(f"#{dup['id'][:8]}")
                continue

        memory = Memory(
            content=item["content"],
            type=item["type"],  # type: ignore[arg-type]
            scope=item["scope"],  # type: ignore[arg-type]
            project_id=item["project_id"] or None,
            tags=item["tags"],
            source=item["source"],
        )
        memory.embedding = embedding
        stored_ids.append(_store(memory))

    n_stored = len(stored_ids)
    n_skipped = len(skipped)
    parts = [f"Stored {n_stored} memor{'y' if n_stored == 1 else 'ies'}"]
    if stored_ids:
        parts[0] += ": " + ", ".join(f"#{mid[:8]}" for mid in stored_ids)
    if n_skipped:
        parts.append(f"skipped {n_skipped} near-duplicate{'s' if n_skipped > 1 else ''}: " + ", ".join(skipped))
    return ". ".join(parts) + "."


@mcp.tool
def compress_memories(
    summary: str,
    type: MemoryType | Literal[""] = "",  # noqa: A002
    scope: MemoryScope | Literal[""] = "",
    threshold: float = 0.85,
    tags: list[str] | None = None,
    source: str = "",
) -> str:
    """Compress semantically similar memories into a single summary memory.

    Finds all stored memories whose cosine similarity to *summary* meets
    *threshold*, deletes them, and stores *summary* as a new memory whose
    ``parent_ids`` records the IDs of every compressed memory.

    Use this after a long session to condense redundant or overlapping
    memories — similar to Copilot's /compact but scoped to the memory store.

    Args:
        summary: The condensed text that replaces the cluster. Write this
                 yourself to capture the essential meaning of the memories
                 you want to collapse.
        type: Restrict the cluster search to this memory type.
        scope: Restrict the cluster search to this scope. For 'project',
               git root is auto-detected from server CWD.
        threshold: Minimum cosine similarity (0-1) to include a memory in
                   the cluster. Default 0.85 is intentionally conservative.
        tags: Tags applied to the new summary memory.
        source: Optional source reference for the new summary memory.
    """
    from contextwell.embedder import embed  # noqa: PLC0415
    from contextwell.store import compress as _compress  # noqa: PLC0415

    project_id = _project_id_for_scope(scope) or ""
    embedding = embed(summary)
    new_id, compressed_ids = _compress(
        summary_embedding=embedding,
        summary_content=summary,
        memory_type=type,
        scope=scope,
        project_id=project_id,
        threshold=threshold,
        tags=tags,
        source=source,
    )
    if not new_id:
        return (
            f"No cluster found — fewer than 2 memories have cosine similarity "
            f"≥ {threshold:.0%} to the provided summary. Nothing was changed."
        )
    scope_label = f"project:{project_id[:8]}" if project_id else (scope or "global")
    n = len(compressed_ids)
    short_ids = ", ".join(f"#{mid[:8]}" for mid in compressed_ids)
    return (
        f"Compressed {n} memor{'y' if n == 1 else 'ies'} [{scope_label}] "
        f"into #{new_id[:8]}. Replaced: {short_ids}."
    )


@mcp.tool
def export_memories(
    format: Literal["json", "markdown", "org"] = "json",  # noqa: A002
    scope: MemoryScope | Literal[""] = "",
    type: MemoryType | Literal[""] = "",  # noqa: A002
    tags: list[str] | None = None,
    since: str = "",
    until: str = "",
    path: str = "",
    limit: int = 1000,
) -> str:
    """Export memories as JSON, Markdown, or Org-mode.

    Scans the store with the given filters, then serialises the results in
    the requested format.  When *path* is provided the output is written to
    that file and a summary is returned; otherwise the full text is returned
    directly (suitable for small exports or piping into another tool).

    Args:
        format: Output format — 'json' (full fidelity, re-importable via
                remember_batch), 'markdown' (human-readable, grouped by
                type), or 'org' (Org-mode with PROPERTIES drawers).
        scope: Filter by scope. Empty means all.
               For 'project', git root is auto-detected from server CWD.
        type: Filter by memory type. Empty means all.
        tags: Only export memories that have at least one of these tags.
        since: ISO 8601 lower bound for created_at (inclusive).
        until: ISO 8601 upper bound for created_at (inclusive).
        path: File path to write the export to. When empty, the content is
              returned as a string.
        limit: Maximum number of memories to export (default 1000).
    """
    from contextwell.export import to_json, to_markdown, to_org  # noqa: PLC0415
    from contextwell.store import scan  # noqa: PLC0415

    project_id = _project_id_for_scope(scope) or ""
    scope_label = f"project:{project_id[:8]}" if project_id else (scope or "all")

    rows = scan(
        scope=scope,
        memory_type=type,
        project_id=project_id,
        tags=tags,
        limit=limit,
        since=since,
        until=until,
    )

    if not rows:
        return "No memories matched the given filters — nothing to export."

    if format == "markdown":
        content = to_markdown(rows, scope_label=scope_label)
    elif format == "org":
        content = to_org(rows, scope_label=scope_label)
    else:
        content = to_json(rows)

    if path:
        from pathlib import Path  # noqa: PLC0415

        out = Path(path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(content, encoding="utf-8")
        return (
            f"Exported {len(rows)} memor{'y' if len(rows) == 1 else 'ies'} "
            f"as {format} to {out}."
        )

    return content


def run() -> None:
    mcp.run()


if __name__ == "__main__":
    run()
