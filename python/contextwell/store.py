"""LanceDB-backed memory store with hybrid search support."""

from __future__ import annotations

import os
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lancedb.table import Table

    from contextwell.schema import Memory

DB_PATH = Path.home() / ".contextwell" / "memories"


def _embedding_dim() -> int:
    """Return the configured embedding dimension (default 384).

    Override with ``CONTEXTWELL_EMBED_DIM`` to match a non-default model.
    Must match the dimension of the model set in ``CONTEXTWELL_EMBED_MODEL``.
    """
    return int(os.getenv("CONTEXTWELL_EMBED_DIM", "384"))


def _escape_literal(value: str) -> str:
    """Escape quote and control chars for LanceDB filter literals."""
    return value.replace("'", "''").replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def _ensure_scalar_indexes(table: Table) -> None:
    """Create missing scalar indexes used by metadata filters."""
    indexed_columns = {column for index in table.list_indices() for column in index.columns}
    for column in ("scope", "type", "project_id", "created_at"):
        if column not in indexed_columns:
            table.create_scalar_index(column)


def _normalize_date_bound(value: str, *, is_until: bool) -> str:
    """Normalize date-only bounds to full ISO datetimes for lexical filtering."""
    if "T" in value:
        return value
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        return value
    day_time = time.max if is_until else time.min
    return datetime.combine(parsed, day_time, tzinfo=UTC).isoformat()


def _where_clauses(
    scope: str = "",
    memory_type: str = "",
    project_id: str = "",
    tags: list[str] | None = None,
    since: str = "",
    until: str = "",
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
    if since:
        clauses.append(f"created_at >= '{_escape_literal(_normalize_date_bound(since, is_until=False))}'")
    if until:
        clauses.append(f"created_at <= '{_escape_literal(_normalize_date_bound(until, is_until=True))}'")
    return clauses


def _get_db():  # noqa: ANN202
    import lancedb  # noqa: PLC0415

    DB_PATH.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(DB_PATH))


def _ensure_updated_at_column(table: Table) -> None:
    """Add the updated_at column to existing tables that pre-date this field."""
    if "updated_at" not in {f.name for f in table.schema}:
        table.add_columns({"updated_at": "''"})


def _ensure_parent_ids_column(table: Table) -> None:
    """Add the parent_ids column to existing tables that pre-date this field."""
    if "parent_ids" not in {f.name for f in table.schema}:
        import pyarrow as pa  # noqa: PLC0415

        table.add_columns(pa.schema([pa.field("parent_ids", pa.list_(pa.string()))]))


def _ensure_chunk_of_column(table: Table) -> None:
    """Add the chunk_of column to existing tables that pre-date this field."""
    if "chunk_of" not in {f.name for f in table.schema}:
        table.add_columns({"chunk_of": "''"})


def _get_table():  # noqa: ANN202
    import pyarrow as pa  # noqa: PLC0415

    db = _get_db()
    existing = db.list_tables().tables
    dim = _embedding_dim()
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
                pa.field("parent_ids", pa.list_(pa.string())),
                pa.field("chunk_of", pa.string()),
                pa.field("embedding", pa.list_(pa.float32(), dim)),
            ]
        )
        tbl = db.create_table("memories", schema=schema)
        _ensure_scalar_indexes(tbl)
        return tbl
    tbl = db.open_table("memories")
    _ensure_scalar_indexes(tbl)
    _ensure_updated_at_column(tbl)
    _ensure_parent_ids_column(tbl)
    _ensure_chunk_of_column(tbl)
    _check_embedding_dim(tbl, dim)
    return tbl


def _check_embedding_dim(table: Table, expected: int) -> None:
    """Raise ValueError when the table's embedding dimension doesn't match *expected*."""
    emb_field = table.schema.field("embedding")
    actual: int = emb_field.type.list_size
    if actual != expected:
        msg = (
            f"Embedding dimension mismatch: the memories table was created with "
            f"dim={actual}, but CONTEXTWELL_EMBED_DIM={expected}. "
            f"Change CONTEXTWELL_EMBED_DIM to {actual}, or delete "
            f"{DB_PATH} to start fresh with the new dimension."
        )
        raise ValueError(msg)


def _clean(row: dict) -> dict:
    """Strip LanceDB internal columns from a result row."""
    row.pop("_distance", None)
    row.pop("embedding", None)
    if row.get("tags") is None:
        row["tags"] = []
    if row.get("parent_ids") is None:
        row["parent_ids"] = []
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
                "updated_at": "",
                "parent_ids": memory.parent_ids,
                "chunk_of": memory.chunk_of,
                "embedding": memory.embedding,
            }
        ]
    )
    return memory.id


def _recall_hybrid(
    table: Table,
    embedding: list[float],
    query: str,
    clauses: list[str],
    k: int,
) -> list[dict]:
    """Vector + BM25 search fused with RRF. Falls back to pure vector on ImportError."""
    try:
        from contextwell._core import search_candidates  # noqa: PLC0415
        from contextwell.bm25 import bm25_search  # noqa: PLC0415
    except ImportError:
        q = table.search(embedding).limit(k)
        if clauses:
            q = q.where(" AND ".join(clauses))
        return [_clean(row) for row in q.to_list()]

    corpus_q = table.search()
    if clauses:
        corpus_q = corpus_q.where(" AND ".join(clauses))
    corpus_rows = [_clean(row) for row in corpus_q.to_list()]
    if not corpus_rows:
        return []
    id_to_row = {row["id"]: row for row in corpus_rows}

    candidate_k = min(k * 3, len(corpus_rows))

    dense_q = table.search(embedding).limit(candidate_k)
    if clauses:
        dense_q = dense_q.where(" AND ".join(clauses))
    dense_ids = [row["id"] for row in dense_q.to_list()]

    try:
        sparse_ids = bm25_search(corpus_rows, query, candidate_k)
    except ImportError:
        return [id_to_row[mid] for mid in dense_ids if mid in id_to_row][:k]

    fused = search_candidates(dense_ids, sparse_ids)
    return [id_to_row[mid] for mid, _ in fused[:k] if mid in id_to_row]


def _dedup_chunks(rows: list[dict], k: int) -> list[dict]:
    """Remove duplicate chunks sharing the same ``chunk_of`` group.

    Results are assumed to be in descending relevance order.  The first
    (highest-scoring) chunk per group is kept; subsequent chunks from the
    same group are dropped.  Non-chunk rows (``chunk_of == ""``) pass through
    unchanged.  The list is truncated to *k* after deduplication.
    """
    seen: set[str] = set()
    out: list[dict] = []
    for row in rows:
        group = row.get("chunk_of") or ""
        if group:
            if group in seen:
                continue
            seen.add(group)
        out.append(row)
        if len(out) >= k:
            break
    return out


def recall(
    embedding: list[float],
    query: str = "",
    scope: str = "",
    memory_type: str = "",
    project_id: str = "",
    tags: list[str] | None = None,
    k: int = 10,
    rerank: bool = False,
) -> list[dict]:
    """Vector search with optional metadata filters. Returns top-k results.

    When the ``CONTEXTWELL_HYBRID`` environment variable is set to ``"1"``
    and *query* is provided, BM25 sparse retrieval is fused with vector
    search via Reciprocal Rank Fusion (requires the ``rank-bm25`` package).

    When *rerank* is ``True`` and *query* is provided, the initial candidate
    set is expanded to ``k * 3`` results and then re-scored by a cross-encoder
    (``cross-encoder/ms-marco-MiniLM-L-6-v2`` by default) for higher precision.

    Chunks sharing the same ``chunk_of`` group ID are deduplicated: only the
    highest-scoring chunk per group is kept.
    """
    table = _get_table()
    clauses = _where_clauses(scope=scope, memory_type=memory_type, project_id=project_id, tags=tags)

    # Expand candidate pool when chunking may produce duplicate groups or reranking.
    has_chunks = os.getenv("CONTEXTWELL_CHUNKING") == "1"
    candidate_k = max(k * 3, 20) if (rerank and query) or has_chunks else k

    if os.getenv("CONTEXTWELL_HYBRID") == "1" and query:
        results = _recall_hybrid(table, embedding, query, clauses, candidate_k)
    else:
        q = table.search(embedding).limit(candidate_k)
        if clauses:
            q = q.where(" AND ".join(clauses))
        results = [_clean(row) for row in q.to_list()]

    if rerank and query:
        from contextwell.reranker import rerank as _rerank  # noqa: PLC0415

        reranked = _rerank(query, results, len(results))
        return _dedup_chunks(reranked, k)

    # Deduplicate chunks from the same group, keeping the highest-scoring hit.
    results = _dedup_chunks(results, k)

    return results[:k]


def scan(
    scope: str = "",
    memory_type: str = "",
    project_id: str = "",
    tags: list[str] | None = None,
    limit: int = 50,
    since: str = "",
    until: str = "",
) -> list[dict]:
    """Full-table scan with optional metadata filters. No vector required."""
    table = _get_table()

    clauses = _where_clauses(
        scope=scope,
        memory_type=memory_type,
        project_id=project_id,
        tags=tags,
        since=since,
        until=until,
    )

    query = table.search().limit(limit)
    if clauses:
        query = query.where(" AND ".join(clauses))

    return [_clean(row) for row in query.to_list()]


def forget(memory_id: str) -> bool:
    """Delete a memory by ID. Returns True if found and deleted."""
    table = _get_table()
    resolved = _resolve_memory_id(table, memory_id)
    if not resolved:
        return False
    before = table.count_rows()
    table.delete(f"id = '{_escape_literal(resolved)}'")
    return table.count_rows() < before


def check_duplicate(
    embedding: list[float],
    threshold: float = 0.95,
) -> dict | None:
    """Return the most similar memory if cosine similarity meets threshold, else None.

    Uses cosine distance (distance = 1 - similarity). Returns the cleaned row dict
    of the best match when similarity >= threshold, or None otherwise.
    Returns None immediately when the store is empty.
    """
    table = _get_table()
    results = table.search(embedding).metric("cosine").limit(1).to_list()
    if not results:
        return None
    top = results[0]
    distance: float = max(0.0, top.get("_distance", 1.0))
    if distance < (1.0 - threshold):
        return _clean(top)
    return None


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
    table = _get_table()
    resolved = _resolve_memory_id(table, memory_id)
    if not resolved:
        return False
    where = f"id = '{_escape_literal(resolved)}'"

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


def _resolve_memory_id(table: Table, memory_id: str) -> str | None:
    """Resolve an ID or 8-char prefix to exactly one stored full ID."""
    escaped = _escape_literal(memory_id)
    if len(memory_id) > 8:  # noqa: PLR2004
        where = f"id = '{escaped}'"
        if table.search().where(where).limit(1).to_list():
            return memory_id
        return None

    matches = table.search().select(["id"]).where(f"id LIKE '{escaped}%'").limit(2).to_list()
    if len(matches) != 1:
        return None
    return str(matches[0]["id"])


def find_cluster(
    embedding: list[float],
    threshold: float = 0.85,
    scope: str = "",
    memory_type: str = "",
    project_id: str = "",
    k: int = 50,
) -> list[dict]:
    """Return memories whose cosine similarity to *embedding* meets *threshold*.

    Searches up to *k* candidates. Results are sorted by descending similarity
    and include only rows with similarity >= threshold.
    """
    table = _get_table()
    clauses = _where_clauses(scope=scope, memory_type=memory_type, project_id=project_id)
    q = table.search(embedding).metric("cosine").limit(k)
    if clauses:
        q = q.where(" AND ".join(clauses))
    results = []
    for row in q.to_list():
        distance: float = max(0.0, row.get("_distance", 1.0))
        if (1.0 - distance) >= threshold:
            results.append(_clean(row))
    return results


def compress(
    summary_embedding: list[float],
    summary_content: str,
    memory_type: str = "fact",
    scope: str = "global",
    project_id: str = "",
    threshold: float = 0.85,
    tags: list[str] | None = None,
    source: str = "",
) -> tuple[str, list[str]]:
    """Replace a cluster of similar memories with a single summary memory.

    Finds all memories whose cosine similarity to *summary_embedding* meets
    *threshold*, deletes them, and stores *summary_content* as a new memory
    whose ``parent_ids`` records the IDs of every compressed memory.

    Returns ``(new_memory_id, compressed_ids)``. If fewer than 2 memories are
    found in the cluster, nothing is changed and returns ``("", [])``.
    """
    from contextwell.schema import Memory as _Memory  # noqa: PLC0415

    cluster = find_cluster(
        summary_embedding,
        threshold=threshold,
        scope=scope,
        memory_type=memory_type,
        project_id=project_id,
    )
    if len(cluster) < 2:  # noqa: PLR2004
        return "", []

    source_ids = [row["id"] for row in cluster]
    for mid in source_ids:
        forget(mid)

    summary = _Memory(
        content=summary_content,
        type=memory_type,  # type: ignore[arg-type]
        scope=scope,  # type: ignore[arg-type]
        project_id=project_id or None,
        tags=tags or [],
        source=source or None,
        parent_ids=source_ids,
    )
    summary.embedding = summary_embedding
    new_id = store(summary)
    return new_id, source_ids


def memory_stats() -> dict:
    """Return an aggregated statistics dictionary for the memory store.

    Keys returned:
    - ``total``: total number of memories
    - ``by_type``: count per memory type
    - ``by_scope``: count per scope value
    - ``oldest``: ISO timestamp of the oldest ``created_at``, or ``""``
    - ``newest``: ISO timestamp of the newest ``created_at``, or ``""``
    - ``store_bytes``: disk usage of the LanceDB directory in bytes
    """
    table = _get_table()
    total = table.count_rows()

    by_type: dict[str, int] = {}
    by_scope: dict[str, int] = {}
    timestamps: list[str] = []

    page_size = 10_000
    offset = 0
    while offset < total:
        rows = table.search().select(["type", "scope", "created_at"]).limit(page_size).offset(offset).to_list()
        if not rows:
            break
        for row in rows:
            t = row.get("type") or "unknown"
            s = row.get("scope") or "unknown"
            by_type[t] = by_type.get(t, 0) + 1
            by_scope[s] = by_scope.get(s, 0) + 1
            ts = row.get("created_at") or ""
            if ts:
                timestamps.append(ts)
        offset += len(rows)

    timestamps.sort()
    store_bytes = sum(f.stat().st_size for f in DB_PATH.rglob("*") if f.is_file()) if DB_PATH.exists() else 0

    return {
        "total": total,
        "by_type": by_type,
        "by_scope": by_scope,
        "oldest": timestamps[0] if timestamps else "",
        "newest": timestamps[-1] if timestamps else "",
        "store_bytes": store_bytes,
    }
