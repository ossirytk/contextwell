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


def _validate_expires_at(expires_at: str) -> str | None:
    """Return an error message if *expires_at* is not a valid ISO 8601 datetime, else None."""
    if not expires_at:
        return None
    from datetime import datetime  # noqa: PLC0415

    try:
        datetime.fromisoformat(expires_at)
    except ValueError:
        return f"expires_at must be a valid ISO 8601 datetime (e.g. '2026-12-31T23:59:59'), got: {expires_at!r}"
    else:
        return None


def _project_id_for_scope(
    scope: str,
    cwd: str | None = None,
    *,
    allow_source_hint: bool = False,
    scope_path: str = "",
) -> str | None:
    """Resolve project_id for project scope, or None for non-project scopes.

    When *scope_path* is provided and *scope* is ``'project'``, the project ID
    is derived directly from that path — no git repository is required.
    """
    if scope != "project":
        return None
    if scope_path:
        from contextwell.project import detect_project_id_from_path  # noqa: PLC0415

        return detect_project_id_from_path(scope_path)

    from contextwell.project import detect_project_id  # noqa: PLC0415

    project_id = detect_project_id(cwd)
    if not project_id:
        msg = "Unable to detect project context for scope='project'. Run from a git repository."
        if allow_source_hint:
            msg += " You can also pass source='cwd:<path>' or scope_path='<directory>'."
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
    expires_at: str = "",
    scope_path: str = "",
) -> str:
    """Store a new memory.

    When ``CONTEXTWELL_CHUNKING=1`` is set and *content* exceeds
    ``CONTEXTWELL_CHUNK_SIZE`` words (default 400), the content is
    automatically split into overlapping chunks and stored separately.
    Each chunk shares a ``chunk_of`` group ID so that ``recall`` can
    deduplicate them, showing only the best-matching chunk.

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
        expires_at: Optional ISO 8601 datetime string after which this memory
                    is silently excluded from recall and listing. Empty means
                    no expiry. Example: "2026-12-31T23:59:59Z".
        scope_path: When scope='project', use this directory path as the
                    project root instead of auto-detecting from git. Allows
                    project-scoped memories in non-git directories.
    """
    from contextwell.chunker import chunk_text, chunking_enabled  # noqa: PLC0415
    from contextwell.embedder import embed, embed_batch  # noqa: PLC0415
    from contextwell.store import check_duplicate, store  # noqa: PLC0415

    err = _validate_expires_at(expires_at)
    if err:
        return f"Error: {err}"

    cwd: str | None = None
    clean_source: str | None = source or None
    if source and source.startswith("cwd:"):
        cwd, _, rest = source[4:].partition("|")
        clean_source = rest or None

    project_id = _project_id_for_scope(scope, cwd, allow_source_hint=True, scope_path=scope_path)
    scope_label = f"project:{project_id[:8]}" if project_id else scope

    # --- Auto-chunking path ---
    if chunking_enabled():
        chunks = chunk_text(content)
        if len(chunks) > 1:
            from uuid import uuid4  # noqa: PLC0415

            group_id = str(uuid4())
            embeddings = embed_batch(chunks)
            if not allow_duplicate:
                for emb in embeddings:
                    duplicate = check_duplicate(emb)
                    if duplicate:
                        dup_id = duplicate["id"]
                        snippet = duplicate["content"][:80]
                        return (
                            f"⚠ Near-duplicate detected (similarity ≥ 95%): "
                            f"#{dup_id[:8]} — {snippet}. "
                            f"Pass allow_duplicate=True to store anyway."
                        )
            ids = []
            for chunk, emb in zip(chunks, embeddings, strict=True):
                m = Memory(
                    content=chunk,
                    type=type,
                    scope=scope,
                    project_id=project_id,
                    tags=tags or [],
                    source=clean_source,
                    chunk_of=group_id,
                    expires_at=expires_at or "",
                )
                m.embedding = emb
                ids.append(store(m))
            return (
                f"Chunked into {len(ids)} piece{'s' if len(ids) > 1 else ''} "
                f"[{type}|{scope_label}] group:{group_id[:8]}: " + ", ".join(f"#{mid[:8]}" for mid in ids)
            )

    # --- Normal (single-memory) path ---
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
        expires_at=expires_at or "",
    )
    memory.embedding = embedding
    memory_id = store(memory)
    return f"Remembered [{type}|{scope_label}] #{memory_id[:8]}: {content[:80]}"


@mcp.tool
def recall(  # noqa: PLR0913
    query: str,
    scope: Literal["project", "global", ""] = "",
    type: MemoryType | Literal[""] = "",  # noqa: A002
    tags: list[str] | None = None,
    k: int = 10,
    rerank: bool = False,
    since: str = "",
    until: str = "",
    include_score: bool = False,
    scope_path: str = "",
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
        since: ISO 8601 lower bound for created_at (inclusive).
               Examples: "2025-01-01", "2025-03-15T09:00:00".
        until: ISO 8601 upper bound for created_at (inclusive).
               Examples: "2025-03-31", "2025-03-31T23:59:59".
        include_score: When True, each result includes a ``"score"`` key
                       with the cosine similarity (0-1) for dense search, or
                       the RRF score for hybrid search.
        scope_path: When scope='project', derive the project ID from this
                    directory path instead of auto-detecting from git.
    """
    from contextwell.embedder import embed  # noqa: PLC0415
    from contextwell.store import recall as _recall  # noqa: PLC0415

    embedding = embed(query)
    project_id = _project_id_for_scope(scope, scope_path=scope_path) or ""
    return _recall(
        embedding,
        query=query,
        scope=scope,
        memory_type=type,
        project_id=project_id or "",
        tags=tags,
        k=k,
        rerank=rerank,
        since=since,
        until=until,
        include_score=include_score,
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
    scope_path: str = "",
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
        scope_path: When scope='project', derive the project ID from this
                    directory path instead of auto-detecting from git.
    """
    from contextwell.store import scan  # noqa: PLC0415

    project_id = _project_id_for_scope(scope, scope_path=scope_path) or ""
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
    scope_path: str = "",
) -> str:
    """Ingest a file and store its sections as individual memories.

    Supported file types:
    - Markdown (``.md``): splits on ## / ### headers (YAML front-matter supported).
    - Org-mode (``.org``): splits on ``*``/``**``/``***`` headlines.
    - Plain text (``.txt``): paragraph-based chunking (~800 chars).
    - Source code (``.py``, ``.rs``, ``.go``, ``.js``, ``.ts``, etc.): regex
      function/class detection with size-based fallback.

    Args:
        path: Absolute or relative path to the file to import.
        scope: 'project' (tied to current git repo) or 'global'.
               For project scope, prefix source with 'cwd:<path>' to set
               the working directory used for git root detection.
        tags: Additional tags applied to every memory created from this file.
        type_hint: Fallback memory type when no heuristic matches a section title.
                   Section titles containing 'decision', 'todo', 'code', etc. are
                   detected automatically.
        source: Optional origin hint. Supports 'cwd:<path>' prefix (same as
                remember) to set the working directory for project-scope detection.
        scope_path: When scope='project', derive the project ID from this
                    directory path instead of auto-detecting from git.
    """
    from pathlib import Path as _Path  # noqa: PLC0415

    from contextwell.embedder import embed  # noqa: PLC0415
    from contextwell.store import store  # noqa: PLC0415

    cwd: str | None = None
    if source and source.startswith("cwd:"):
        cwd, _, source = source[4:].partition("|")
        source = source or ""

    project_id = _project_id_for_scope(scope, cwd, allow_source_hint=True, scope_path=scope_path)

    suffix = _Path(path).suffix.lower()
    try:
        if suffix == ".md":
            from contextwell.markdown_import import parse  # noqa: PLC0415

            chunks = parse(path, default_tags=tags or [], default_type=type_hint)
        else:
            from contextwell.file_import import parse as _parse  # noqa: PLC0415

            chunks = _parse(path, default_tags=tags or [], default_type=type_hint)
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
    - ``expires_at`` (str, default ""): Optional ISO 8601 expiry datetime.
    - ``scope_path`` (str, default ""): Non-git project directory override.

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
        scope_path_raw: str = str(raw.get("scope_path", ""))
        expires_at_raw: str = str(raw.get("expires_at", ""))
        err = _validate_expires_at(expires_at_raw)
        if err:
            return f"Error in item {len(parsed)}: {err}"
        parsed.append(
            {
                "content": content,
                "type": str(raw.get("type", "fact")),
                "scope": scope,
                "tags": list(raw.get("tags") or []),
                "source": clean_source,
                "cwd": cwd,
                "expires_at": expires_at_raw,
                "project_id": _project_id_for_scope(scope, cwd, allow_source_hint=True, scope_path=scope_path_raw)
                or "",
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
            expires_at=item["expires_at"] or "",
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
    scope_path: str = "",
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
        scope_path: When scope='project', derive the project ID from this
                    directory path instead of auto-detecting from git.
    """
    from contextwell.embedder import embed  # noqa: PLC0415
    from contextwell.store import compress as _compress  # noqa: PLC0415

    project_id = _project_id_for_scope(scope, scope_path=scope_path) or ""
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
    return f"Compressed {n} memor{'y' if n == 1 else 'ies'} [{scope_label}] into #{new_id[:8]}. Replaced: {short_ids}."


@mcp.tool
def export_memories(  # noqa: PLR0913
    format: Literal["json", "markdown", "org"] = "json",  # noqa: A002
    scope: MemoryScope | Literal[""] = "",
    type: MemoryType | Literal[""] = "",  # noqa: A002
    tags: list[str] | None = None,
    since: str = "",
    until: str = "",
    path: str = "",
    limit: int = 1000,
    scope_path: str = "",
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
        scope_path: When scope='project', derive the project ID from this
                    directory path instead of auto-detecting from git.
    """
    from contextwell.export import to_json, to_markdown, to_org  # noqa: PLC0415
    from contextwell.store import scan  # noqa: PLC0415

    project_id = _project_id_for_scope(scope, scope_path=scope_path) or ""
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
        return f"Exported {len(rows)} memor{'y' if len(rows) == 1 else 'ies'} as {format} to {out}."

    return content


@mcp.tool
def memory_stats(stale_days: int = 0) -> dict:
    """Return a dashboard summary of the memory store.

    Provides situational awareness: how many memories are stored, how they
    break down by type and scope, the oldest and newest timestamps, and the
    approximate disk footprint of the store.

    Returns a dict with keys:
    - ``total``: total memory count
    - ``by_type``: mapping of type → count
    - ``by_scope``: mapping of scope → count
    - ``oldest``: ISO timestamp of the oldest memory (or ``""`` if empty)
    - ``newest``: ISO timestamp of the newest memory (or ``""`` if empty)
    - ``store_bytes``: disk usage in bytes of the LanceDB directory
    - ``stale_count``: (only when stale_days > 0) memories never updated and
      older than stale_days days

    Args:
        stale_days: When > 0, count memories that were never updated and have
                    a created_at older than this many days.
    """
    from contextwell.store import memory_stats as _stats  # noqa: PLC0415

    return _stats(stale_days=stale_days)


@mcp.tool
def purge_expired() -> str:
    """Permanently delete all memories whose expiry datetime has passed.

    Memories stored with an ``expires_at`` value (ISO 8601) that is now in
    the past are removed from the store. Memories with no expiry are unaffected.

    Returns a human-readable summary of how many memories were deleted.
    """
    from contextwell.store import purge_expired as _purge  # noqa: PLC0415

    n = _purge()
    return f"Purged {n} expired memor{'y' if n == 1 else 'ies'}."


@mcp.tool
def reembed_all(batch_size: int = 64) -> dict:
    """Re-embed all stored memories with the currently configured model.

    Use this after changing ``CONTEXTWELL_EMBED_MODEL`` to migrate existing
    memories to the new embedding space. All stored memories are re-embedded
    in batches and updated in-place; IDs and metadata are preserved.

    Args:
        batch_size: Number of memories to embed per batch (default 64).
                    Reduce if you hit memory limits with large models.

    Returns:
        A dict with keys ``total`` (memories inspected) and ``reembedded``
        (memories actually updated).
    """
    from contextwell.store import reembed_all as _reembed  # noqa: PLC0415

    return _reembed(batch_size=batch_size)


def run() -> None:
    mcp.run()


if __name__ == "__main__":
    run()
