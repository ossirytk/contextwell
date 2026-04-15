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


def _fake_embed(text: str) -> list[float]:
    raw = hashlib.sha256(text.encode("utf-8")).digest()
    vector = [((raw[i % len(raw)] / 255.0) - 0.5) for i in range(384)]
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def test_embedding_store_integration(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONTEXTWELL_TEST", "1")
    monkeypatch.setattr(store_module, "DB_PATH", tmp_path / "memories")

    vec = _fake_embed("We chose LanceDB for hybrid search support")
    assert len(vec) == 384
    assert abs(sum(value**2 for value in vec) - 1.0) < 0.01

    m = Memory(content="We chose LanceDB over ChromaDB for hybrid search support", type="decision", scope="global")
    m.embedding = _fake_embed(m.content)
    mid = store(m)
    assert len(mid) == 36

    m2 = Memory(content="Use all-MiniLM-L6-v2 as the default embedding model", type="decision", scope="global")
    m2.embedding = _fake_embed(m2.content)
    store(m2)

    results = recall(_fake_embed("database choice"), k=2)
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
