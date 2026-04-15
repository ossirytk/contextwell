"""End-to-end integration test for contextwell embedding + store."""

import hashlib
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

import contextwell.store as store_module
from contextwell.bm25 import bm25_search
from contextwell.project import detect_project_id
from contextwell.schema import Memory
from contextwell.store import check_duplicate, forget, recall, scan, store, update

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
