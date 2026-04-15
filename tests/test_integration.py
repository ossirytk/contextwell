"""End-to-end integration test for contextwell embedding + store."""

import hashlib
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

import contextwell.embedder as embedder_module
import contextwell.store as store_module
from contextwell.bm25 import bm25_search
from contextwell.embedder import _model_name, reset_model
from contextwell.project import detect_project_id
from contextwell.schema import Memory
from contextwell.store import (
    _embedding_dim,
    check_duplicate,
    compress,
    find_cluster,
    forget,
    recall,
    scan,
    store,
    update,
)

_EMBEDDING_DIM = 384


def _test_embed(text: str) -> list[float]:
    """Return deterministic normalized test vectors to avoid model downloads in tests."""
    raw = hashlib.sha256(text.encode("utf-8")).digest()
    vector = [((raw[i % len(raw)] / 255.0) - 0.5) for i in range(_EMBEDDING_DIM)]
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def test_embedding_store_integration(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONTEXTWELL_TEST", "1")
    monkeypatch.setattr(store_module, "DB_PATH", tmp_path / "memories")

    vec = _test_embed("We chose LanceDB for hybrid search support")
    assert len(vec) == _EMBEDDING_DIM
    assert abs(sum(value**2 for value in vec) - 1.0) < 0.01

    m = Memory(content="We chose LanceDB over ChromaDB for hybrid search support", type="decision", scope="global")
    m.embedding = _test_embed(m.content)
    mid = store(m)
    assert len(mid) == 36

    m2 = Memory(content="Use all-MiniLM-L6-v2 as the default embedding model", type="decision", scope="global")
    m2.embedding = _test_embed(m2.content)
    store(m2)

    results = recall(_test_embed("database choice"), k=2)
    assert results
    assert "_distance" not in results[0]
    assert "embedding" not in results[0]

    rows = scan(memory_type="decision", limit=10)
    assert len(rows) == 2

    deleted = forget(mid)
    assert deleted
    rows_after = scan(limit=10)
    assert len(rows_after) == 1

    pid = detect_project_id(str(Path(__file__).parent))
    assert pid is not None
    assert len(pid) == 16


def test_tags_filter_scan(tmp_path, monkeypatch) -> None:
    """list_memories (scan) returns only memories matching at least one tag."""
    monkeypatch.setattr(store_module, "DB_PATH", tmp_path / "memories")

    m1 = Memory(content="Auth uses JWT tokens", type="decision", scope="global", tags=["auth", "security"])
    m1.embedding = _test_embed(m1.content)
    store(m1)

    m2 = Memory(content="Use Postgres for persistence", type="decision", scope="global", tags=["db", "postgres"])
    m2.embedding = _test_embed(m2.content)
    store(m2)

    m3 = Memory(content="Rate limiting on auth endpoints", type="fact", scope="global", tags=["auth", "api"])
    m3.embedding = _test_embed(m3.content)
    store(m3)

    # Single tag — should match m1 and m3 only
    rows = scan(tags=["auth"], limit=10)
    ids = {r["id"] for r in rows}
    assert m1.id in ids
    assert m3.id in ids
    assert m2.id not in ids

    # Multi-tag any-match — 'db' matches m2, 'security' matches m1
    rows = scan(tags=["db", "security"], limit=10)
    ids = {r["id"] for r in rows}
    assert m1.id in ids
    assert m2.id in ids
    assert m3.id not in ids

    # Tag with no matches — empty result
    rows = scan(tags=["nonexistent"], limit=10)
    assert rows == []


def test_tags_filter_recall(tmp_path, monkeypatch) -> None:
    """recall (vector search) respects the tags filter."""
    monkeypatch.setattr(store_module, "DB_PATH", tmp_path / "memories")

    m1 = Memory(content="Auth uses JWT tokens", type="decision", scope="global", tags=["auth"])
    m1.embedding = _test_embed(m1.content)
    store(m1)

    m2 = Memory(content="Auth uses session cookies as fallback", type="decision", scope="global", tags=["db"])
    m2.embedding = _test_embed(m2.content)
    store(m2)

    # Both are semantically close to the query, but the tag filter restricts to 'auth'
    results = recall(_test_embed("authentication approach"), tags=["auth"], k=5)
    ids = {r["id"] for r in results}
    assert m1.id in ids
    assert m2.id not in ids


def test_tags_filter_combined_with_type(tmp_path, monkeypatch) -> None:
    """Tags filter composes correctly with type filter."""
    monkeypatch.setattr(store_module, "DB_PATH", tmp_path / "memories")

    m1 = Memory(content="Use bcrypt for passwords", type="decision", scope="global", tags=["auth"])
    m1.embedding = _test_embed(m1.content)
    store(m1)

    m2 = Memory(content="Bcrypt snippet", type="code", scope="global", tags=["auth"])
    m2.embedding = _test_embed(m2.content)
    store(m2)

    rows = scan(memory_type="decision", tags=["auth"], limit=10)
    ids = {r["id"] for r in rows}
    assert m1.id in ids
    assert m2.id not in ids  # excluded by type filter


def test_update_content_and_reembed(tmp_path, monkeypatch) -> None:
    """update() replaces content and accepts a new embedding."""
    monkeypatch.setattr(store_module, "DB_PATH", tmp_path / "memories")

    m = Memory(content="Original content", type="fact", scope="global")
    m.embedding = _test_embed(m.content)
    store(m)

    new_content = "Updated content"
    new_embed = _test_embed(new_content)
    found = update(m.id, content=new_content, new_embedding=new_embed)
    assert found

    rows = scan(limit=10)
    assert len(rows) == 1
    assert rows[0]["content"] == new_content
    assert rows[0]["updated_at"] != ""


def test_update_tags_only(tmp_path, monkeypatch) -> None:
    """update() can change just tags without touching content or embedding."""
    monkeypatch.setattr(store_module, "DB_PATH", tmp_path / "memories")

    m = Memory(content="Some fact", type="fact", scope="global", tags=["old"])
    m.embedding = _test_embed(m.content)
    store(m)

    found = update(m.id, tags=["new", "tags"])
    assert found

    rows = scan(limit=10)
    assert rows[0]["tags"] == ["new", "tags"]
    assert rows[0]["content"] == "Some fact"  # unchanged


def test_update_partial_id(tmp_path, monkeypatch) -> None:
    """update() resolves memories by first 8 chars of ID."""
    monkeypatch.setattr(store_module, "DB_PATH", tmp_path / "memories")

    m = Memory(content="Partial ID test", type="fact", scope="global")
    m.embedding = _test_embed(m.content)
    store(m)

    found = update(m.id[:8], tags=["found-by-prefix"])
    assert found
    rows = scan(limit=10)
    assert "found-by-prefix" in rows[0]["tags"]


def test_update_not_found(tmp_path, monkeypatch) -> None:
    """update() returns False when no memory matches the given ID."""
    monkeypatch.setattr(store_module, "DB_PATH", tmp_path / "memories")

    m = Memory(content="Existing", type="fact", scope="global")
    m.embedding = _test_embed(m.content)
    store(m)

    found = update("00000000-0000-0000-0000-000000000000", content="ghost")
    assert not found


def test_update_preserves_created_at(tmp_path, monkeypatch) -> None:
    """update() sets updated_at but leaves created_at intact."""
    monkeypatch.setattr(store_module, "DB_PATH", tmp_path / "memories")

    m = Memory(content="Timestamp test", type="fact", scope="global")
    m.embedding = _test_embed(m.content)
    store(m)
    original_created_at = scan(limit=1)[0]["created_at"]

    update(m.id, tags=["touched"])

    row = scan(limit=1)[0]
    assert row["created_at"] == original_created_at
    assert row["updated_at"] != ""


# --- Item 4: Deduplication ---


def test_check_duplicate_found(tmp_path, monkeypatch) -> None:
    """check_duplicate returns the matching row when similarity >= threshold."""
    monkeypatch.setattr(store_module, "DB_PATH", tmp_path / "memories")

    m = Memory(content="Use JWT for auth", type="decision", scope="global")
    m.embedding = _test_embed(m.content)
    store(m)

    # Same embedding → cosine distance = 0 → similarity = 1.0 → duplicate
    result = check_duplicate(_test_embed(m.content), threshold=0.95)
    assert result is not None
    assert result["id"] == m.id
    assert result["content"] == m.content


def test_check_duplicate_not_found(tmp_path, monkeypatch) -> None:
    """check_duplicate returns None for content that is not similar enough."""
    monkeypatch.setattr(store_module, "DB_PATH", tmp_path / "memories")

    m = Memory(content="Use JWT for auth", type="decision", scope="global")
    m.embedding = _test_embed(m.content)
    store(m)

    # Different content → very different SHA-256-based vector → not a duplicate
    result = check_duplicate(_test_embed("Deploy to production every Friday"), threshold=0.95)
    assert result is None


def test_check_duplicate_empty_store(tmp_path, monkeypatch) -> None:
    """check_duplicate returns None when the store is empty (no search results)."""
    monkeypatch.setattr(store_module, "DB_PATH", tmp_path / "memories")

    result = check_duplicate(_test_embed("anything"), threshold=0.95)
    assert result is None


def test_check_duplicate_threshold_boundary(tmp_path, monkeypatch) -> None:
    """check_duplicate respects the threshold parameter (distance < 1 - threshold)."""
    monkeypatch.setattr(store_module, "DB_PATH", tmp_path / "memories")

    m = Memory(content="exact copy", type="fact", scope="global")
    m.embedding = _test_embed(m.content)
    store(m)

    # threshold=1.0 means distance < 0.0, which is never true — should not match
    result = check_duplicate(_test_embed(m.content), threshold=1.0)
    assert result is None

    # threshold=0.0 means any distance < 1.0 matches — identical vector must match
    result = check_duplicate(_test_embed(m.content), threshold=0.0)
    assert result is not None


# --- Item 5: Hybrid search (BM25 + Vector + RRF) ---


def test_bm25_search_ranks_by_keyword() -> None:
    """bm25_search returns IDs ranked with keyword-matching docs first."""

    rows = [
        {"id": "a", "content": "JWT is used for authentication and authorization"},
        {"id": "b", "content": "bcrypt hashes passwords securely"},
        {"id": "c", "content": "JWT tokens expire after a configurable period"},
        {"id": "d", "content": "Use environment variables for secrets"},
        {"id": "e", "content": "Redis caches session data"},
    ]
    results = bm25_search(rows, "JWT token authentication", k=3)
    assert "a" in results
    assert "c" in results


def test_bm25_search_empty_rows() -> None:
    """bm25_search returns [] for empty corpus."""

    assert bm25_search([], "anything", k=5) == []


def test_bm25_search_blank_query() -> None:
    """bm25_search returns [] for blank query."""

    rows = [{"id": "x", "content": "some content"}]
    assert bm25_search(rows, "   ", k=5) == []


def test_bm25_search_respects_k() -> None:
    """bm25_search returns at most k results."""

    rows = [{"id": str(i), "content": f"document {i} about authentication"} for i in range(10)]
    results = bm25_search(rows, "authentication", k=3)
    assert len(results) <= 3


def test_hybrid_recall_returns_results(tmp_path, monkeypatch) -> None:
    """Hybrid recall returns results when CONTEXTWELL_HYBRID=1."""
    monkeypatch.setattr(store_module, "DB_PATH", tmp_path / "memories")
    monkeypatch.setenv("CONTEXTWELL_HYBRID", "1")

    for text in [
        "JWT is used for auth",
        "bcrypt hashes passwords",
        "use environment variables for secrets",
        "Redis caches data",
        "deploy to production on Fridays",
    ]:
        m = Memory(content=text, type="fact", scope="global")
        m.embedding = _test_embed(text)
        store(m)

    embedding = _test_embed("JWT token authentication")
    results = recall(embedding, query="JWT token authentication", k=3)
    assert len(results) > 0
    assert all("content" in r for r in results)


def test_hybrid_recall_falls_back_without_env(tmp_path, monkeypatch) -> None:
    """Without CONTEXTWELL_HYBRID=1, recall uses pure vector search."""
    monkeypatch.setattr(store_module, "DB_PATH", tmp_path / "memories")
    monkeypatch.delenv("CONTEXTWELL_HYBRID", raising=False)

    m = Memory(content="Pure vector test", type="fact", scope="global")
    m.embedding = _test_embed(m.content)
    store(m)

    results = recall(_test_embed("Pure vector test"), query="Pure vector test", k=5)
    assert len(results) == 1
    assert results[0]["content"] == "Pure vector test"


# --- Item 6: Configurable embedding model ---


def test_model_name_default(monkeypatch) -> None:
    """Default model name is all-MiniLM-L6-v2 when no env var is set."""
    monkeypatch.delenv("CONTEXTWELL_EMBED_MODEL", raising=False)
    monkeypatch.delenv("CONTEXTWELL_EMBED_PROVIDER", raising=False)
    assert _model_name() == "BAAI/bge-small-en-v1.5"


def test_model_name_from_env(monkeypatch) -> None:
    """CONTEXTWELL_EMBED_MODEL overrides the default model name."""
    monkeypatch.setenv("CONTEXTWELL_EMBED_MODEL", "bge-small-en-v1.5")
    monkeypatch.delenv("CONTEXTWELL_EMBED_PROVIDER", raising=False)
    assert _model_name() == "bge-small-en-v1.5"


def test_model_name_openai_default(monkeypatch) -> None:
    """OpenAI provider defaults to text-embedding-3-small when no model is set."""
    monkeypatch.setenv("CONTEXTWELL_EMBED_PROVIDER", "openai")
    monkeypatch.delenv("CONTEXTWELL_EMBED_MODEL", raising=False)
    assert _model_name() == "text-embedding-3-small"


def test_embedding_dim_default(monkeypatch) -> None:
    """Default embedding dimension is 384."""
    monkeypatch.delenv("CONTEXTWELL_EMBED_DIM", raising=False)
    assert _embedding_dim() == 384


def test_embedding_dim_from_env(monkeypatch) -> None:
    """CONTEXTWELL_EMBED_DIM overrides the embedding dimension."""
    monkeypatch.setenv("CONTEXTWELL_EMBED_DIM", "768")
    assert _embedding_dim() == 768


def test_reset_model_clears_cache(monkeypatch) -> None:
    """reset_model() sets the cached model back to None."""
    monkeypatch.delenv("CONTEXTWELL_EMBED_PROVIDER", raising=False)
    embedder_module._model = object()  # noqa: SLF001  # inject a sentinel
    reset_model()
    assert embedder_module._model is None  # noqa: SLF001


def test_dimension_mismatch_raises(tmp_path, monkeypatch) -> None:
    """_get_table raises ValueError when the table dim doesn't match CONTEXTWELL_EMBED_DIM."""
    monkeypatch.setattr(store_module, "DB_PATH", tmp_path / "memories")

    # Create table with dim=8
    monkeypatch.setenv("CONTEXTWELL_EMBED_DIM", "8")
    m = Memory(content="dim test", type="fact", scope="global")
    m.embedding = [0.0] * 8
    store(m)

    # Re-open with a different dim — must raise
    monkeypatch.setenv("CONTEXTWELL_EMBED_DIM", "16")
    with pytest.raises(ValueError, match="Embedding dimension mismatch"):
        store_module._get_table()  # noqa: SLF001


# ---------------------------------------------------------------------------
# Date range filtering tests
# ---------------------------------------------------------------------------

def test_since_filter_excludes_older(tmp_path, monkeypatch) -> None:
    """scan with since= excludes memories with older created_at."""
    from datetime import UTC, datetime  # noqa: PLC0415

    monkeypatch.setattr(store_module, "DB_PATH", tmp_path / "memories")

    old = Memory(content="old memory", type="fact", scope="global")
    old.embedding = _test_embed("old memory")
    old.created_at = datetime(2024, 1, 1, tzinfo=UTC)
    store(old)

    new = Memory(content="new memory", type="fact", scope="global")
    new.embedding = _test_embed("new memory")
    new.created_at = datetime(2025, 6, 1, tzinfo=UTC)
    store(new)

    results = scan(since="2025-01-01")
    ids = {r["id"] for r in results}
    assert new.id in ids
    assert old.id not in ids


def test_until_filter_excludes_newer(tmp_path, monkeypatch) -> None:
    """scan with until= excludes memories with newer created_at."""
    from datetime import UTC, datetime  # noqa: PLC0415

    monkeypatch.setattr(store_module, "DB_PATH", tmp_path / "memories")

    old = Memory(content="old entry", type="fact", scope="global")
    old.embedding = _test_embed("old entry")
    old.created_at = datetime(2024, 3, 15, tzinfo=UTC)
    store(old)

    new = Memory(content="new entry", type="fact", scope="global")
    new.embedding = _test_embed("new entry")
    new.created_at = datetime(2025, 8, 1, tzinfo=UTC)
    store(new)

    results = scan(until="2024-12-31")
    ids = {r["id"] for r in results}
    assert old.id in ids
    assert new.id not in ids


def test_since_until_range(tmp_path, monkeypatch) -> None:
    """scan with both since= and until= returns only memories in the window."""
    from datetime import UTC, datetime  # noqa: PLC0415

    monkeypatch.setattr(store_module, "DB_PATH", tmp_path / "memories")

    dates = [
        datetime(2024, 12, 31, tzinfo=UTC),
        datetime(2025, 2, 15, tzinfo=UTC),
        datetime(2025, 6, 30, tzinfo=UTC),
    ]
    memories = []
    for i, dt in enumerate(dates):
        m = Memory(content=f"entry {i}", type="fact", scope="global")
        m.embedding = _test_embed(f"entry {i}")
        m.created_at = dt
        store(m)
        memories.append(m)

    results = scan(since="2025-01-01", until="2025-05-31")
    ids = {r["id"] for r in results}
    assert memories[1].id in ids       # 2025-02-15 is in range
    assert memories[0].id not in ids   # 2024-12-31 is before since
    assert memories[2].id not in ids   # 2025-06-30 is after until


# ---------------------------------------------------------------------------
# find_cluster / compress tests
# ---------------------------------------------------------------------------

def test_find_cluster_returns_similar(tmp_path, monkeypatch) -> None:
    """find_cluster returns memories whose cosine similarity meets the threshold."""
    monkeypatch.setattr(store_module, "DB_PATH", tmp_path / "memories")
    emb = _test_embed("similar memory")

    m1 = Memory(content="alpha", type="fact", scope="global")
    m1.embedding = emb
    m2 = Memory(content="beta", type="fact", scope="global")
    m2.embedding = emb
    store(m1)
    store(m2)

    cluster = find_cluster(emb, threshold=0.99)
    ids = {row["id"] for row in cluster}
    assert m1.id in ids
    assert m2.id in ids


def test_find_cluster_excludes_dissimilar(tmp_path, monkeypatch) -> None:
    """find_cluster does not return memories with low cosine similarity."""
    monkeypatch.setattr(store_module, "DB_PATH", tmp_path / "memories")
    emb_a = _test_embed("rust programming language")
    emb_b = _test_embed("zzz_completely_unrelated_xyz_999")

    m = Memory(content="rust", type="fact", scope="global")
    m.embedding = emb_b
    store(m)

    cluster = find_cluster(emb_a, threshold=0.99)
    ids = {row["id"] for row in cluster}
    assert m.id not in ids


def test_compress_replaces_cluster(tmp_path, monkeypatch) -> None:
    """compress() stores a summary memory with parent_ids set to the original IDs."""
    monkeypatch.setattr(store_module, "DB_PATH", tmp_path / "memories")
    emb = _test_embed("shared embedding vector")

    ids = []
    for i in range(3):
        m = Memory(content=f"source {i}", type="fact", scope="global")
        m.embedding = emb
        ids.append(store(m))

    new_id, compressed = compress(
        summary_embedding=emb,
        summary_content="consolidated summary",
        threshold=0.99,
    )
    assert new_id != ""
    assert set(compressed) == set(ids)

    # The new memory should have parent_ids recorded
    results = scan(scope="global")
    summary_rows = [r for r in results if r["id"] == new_id]
    assert len(summary_rows) == 1
    assert set(summary_rows[0]["parent_ids"]) == set(ids)


def test_compress_preserves_count(tmp_path, monkeypatch) -> None:
    """After compress(), total row count decreases by (N - 1)."""
    monkeypatch.setattr(store_module, "DB_PATH", tmp_path / "memories")
    emb = _test_embed("compression count test")

    for i in range(4):
        m = Memory(content=f"item {i}", type="fact", scope="global")
        m.embedding = emb
        store(m)

    before = len(scan(scope="global"))
    compress(summary_embedding=emb, summary_content="summary", threshold=0.99)
    after = len(scan(scope="global"))

    assert after == before - 3  # 4 removed, 1 added → net -3


def test_compress_too_few_returns_empty(tmp_path, monkeypatch) -> None:
    """compress() returns ('', []) when fewer than 2 memories match."""
    monkeypatch.setattr(store_module, "DB_PATH", tmp_path / "memories")
    emb_a = _test_embed("unique embedding for compress")
    emb_b = _test_embed("zzz_nothing_similar_at_all_999")

    m = Memory(content="lone ranger", type="fact", scope="global")
    m.embedding = emb_b
    store(m)

    new_id, compressed = compress(
        summary_embedding=emb_a,
        summary_content="should not be stored",
        threshold=0.99,
    )
    assert new_id == ""
    assert compressed == []
    # Original memory should still exist
    assert any(r["id"] == m.id for r in scan(scope="global"))


# ---------------------------------------------------------------------------
# Item 9: remember_batch tests
# ---------------------------------------------------------------------------

def _patch_embed_batch(monkeypatch) -> None:
    """Patch embed and embed_batch to use _test_embed (no model downloads)."""
    monkeypatch.setattr(
        "contextwell.embedder.embed",
        _test_embed,
    )
    monkeypatch.setattr(
        "contextwell.embedder.embed_batch",
        lambda texts: [_test_embed(t) for t in texts],
    )


def test_remember_batch_stores_all(tmp_path, monkeypatch) -> None:
    """remember_batch stores all provided memories."""
    import contextwell.server as server_module  # noqa: PLC0415

    monkeypatch.setattr(store_module, "DB_PATH", tmp_path / "memories")
    _patch_embed_batch(monkeypatch)

    items = [
        {"content": "batch item one", "type": "fact", "scope": "global"},
        {"content": "batch item two", "type": "code", "scope": "global"},
        {"content": "batch item three", "type": "decision", "scope": "global"},
    ]
    result = server_module.remember_batch(items)
    assert "Stored 3" in result

    rows = scan(scope="global")
    contents = {r["content"] for r in rows}
    assert {"batch item one", "batch item two", "batch item three"} <= contents


def test_remember_batch_empty_list(tmp_path, monkeypatch) -> None:
    """remember_batch with an empty list returns a friendly message."""
    import contextwell.server as server_module  # noqa: PLC0415

    monkeypatch.setattr(store_module, "DB_PATH", tmp_path / "memories")
    _patch_embed_batch(monkeypatch)

    result = server_module.remember_batch([])
    assert "No memories" in result


def test_remember_batch_skips_duplicates(tmp_path, monkeypatch) -> None:
    """remember_batch skips near-duplicates by default."""
    import contextwell.server as server_module  # noqa: PLC0415

    monkeypatch.setattr(store_module, "DB_PATH", tmp_path / "memories")
    _patch_embed_batch(monkeypatch)

    # Store one memory first
    m = Memory(content="original", type="fact", scope="global")
    m.embedding = _test_embed("original")
    store(m)

    # Batch with the same content — should be skipped
    result = server_module.remember_batch(
        [{"content": "original", "scope": "global"}]
    )
    assert "skipped" in result
    # Row count should still be 1
    assert len(scan(scope="global")) == 1


def test_remember_batch_allow_duplicate(tmp_path, monkeypatch) -> None:
    """remember_batch stores duplicates when allow_duplicate=True."""
    import contextwell.server as server_module  # noqa: PLC0415

    monkeypatch.setattr(store_module, "DB_PATH", tmp_path / "memories")
    _patch_embed_batch(monkeypatch)

    m = Memory(content="dupe candidate", type="fact", scope="global")
    m.embedding = _test_embed("dupe candidate")
    store(m)

    result = server_module.remember_batch(
        [{"content": "dupe candidate", "scope": "global"}],
        allow_duplicate=True,
    )
    assert "Stored 1" in result
    assert len(scan(scope="global")) == 2


def test_remember_batch_skips_empty_content(tmp_path, monkeypatch) -> None:
    """remember_batch ignores items with empty or missing content."""
    import contextwell.server as server_module  # noqa: PLC0415

    monkeypatch.setattr(store_module, "DB_PATH", tmp_path / "memories")
    _patch_embed_batch(monkeypatch)

    items = [
        {"content": "", "type": "fact"},
        {"type": "fact"},  # missing content key
        {"content": "valid item", "type": "fact", "scope": "global"},
    ]
    result = server_module.remember_batch(items)
    assert "Stored 1" in result
    rows = scan(scope="global")
    assert any(r["content"] == "valid item" for r in rows)


# ---------------------------------------------------------------------------
# Item 10: export_memories tests
# ---------------------------------------------------------------------------

def test_export_json(tmp_path, monkeypatch) -> None:
    """export_memories returns valid JSON with all expected fields."""
    import json as _json  # noqa: PLC0415

    import contextwell.server as server_module  # noqa: PLC0415

    monkeypatch.setattr(store_module, "DB_PATH", tmp_path / "memories")

    m = Memory(content="export test", type="fact", scope="global", tags=["x"])
    m.embedding = _test_embed("export test")
    store(m)

    result = server_module.export_memories(format="json")
    data = _json.loads(result)
    assert isinstance(data, list)
    assert len(data) == 1
    row = data[0]
    assert row["content"] == "export test"
    assert row["type"] == "fact"
    assert "x" in row["tags"]
    assert "embedding" not in row  # embeddings must be excluded


def test_export_markdown(tmp_path, monkeypatch) -> None:
    """export_memories returns well-formed Markdown."""
    import contextwell.server as server_module  # noqa: PLC0415

    monkeypatch.setattr(store_module, "DB_PATH", tmp_path / "memories")

    m = Memory(content="md export test", type="decision", scope="global")
    m.embedding = _test_embed("md export test")
    store(m)

    result = server_module.export_memories(format="markdown")
    assert result.startswith("# Contextwell Memory Export")
    assert "## decision" in result
    assert "md export test" in result


def test_export_org(tmp_path, monkeypatch) -> None:
    """export_memories returns well-formed Org-mode output."""
    import contextwell.server as server_module  # noqa: PLC0415

    monkeypatch.setattr(store_module, "DB_PATH", tmp_path / "memories")

    m = Memory(content="org export test", type="code", scope="global", tags=["rust"])
    m.embedding = _test_embed("org export test")
    store(m)

    result = server_module.export_memories(format="org")
    assert "#+TITLE: Contextwell Memory Export" in result
    assert "* code" in result
    assert "org export test" in result
    assert ":rust:" in result


def test_export_writes_file(tmp_path, monkeypatch) -> None:
    """export_memories writes to a file and returns a summary message."""
    import contextwell.server as server_module  # noqa: PLC0415

    monkeypatch.setattr(store_module, "DB_PATH", tmp_path / "memories")

    m = Memory(content="file export test", type="fact", scope="global")
    m.embedding = _test_embed("file export test")
    store(m)

    out_path = tmp_path / "export.json"
    result = server_module.export_memories(format="json", path=str(out_path))
    assert "Exported 1" in result
    assert out_path.exists()
    import json as _json  # noqa: PLC0415
    data = _json.loads(out_path.read_text())
    assert data[0]["content"] == "file export test"


def test_export_empty_store(tmp_path, monkeypatch) -> None:
    """export_memories returns a friendly message when no memories match."""
    import contextwell.server as server_module  # noqa: PLC0415

    monkeypatch.setattr(store_module, "DB_PATH", tmp_path / "memories")

    result = server_module.export_memories(format="json")
    assert "No memories" in result


def test_export_respects_filters(tmp_path, monkeypatch) -> None:
    """export_memories only exports memories that match the given filters."""
    import json as _json  # noqa: PLC0415

    import contextwell.server as server_module  # noqa: PLC0415

    monkeypatch.setattr(store_module, "DB_PATH", tmp_path / "memories")

    fact = Memory(content="a fact", type="fact", scope="global")
    fact.embedding = _test_embed("a fact")
    store(fact)

    code = Memory(content="some code", type="code", scope="global")
    code.embedding = _test_embed("some code")
    store(code)

    result = server_module.export_memories(format="json", type="code")
    data = _json.loads(result)
    assert len(data) == 1
    assert data[0]["type"] == "code"
