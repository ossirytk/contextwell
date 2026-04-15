"""End-to-end integration test for contextwell embedding + store."""

import hashlib
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

import contextwell.store as store_module
from contextwell.project import detect_project_id
from contextwell.schema import Memory
from contextwell.store import forget, recall, scan, store

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
