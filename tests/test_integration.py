"""End-to-end integration test for contextwell embedding + store."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

import os
import tempfile

os.environ["CONTEXTWELL_TEST"] = "1"

# Point DB at a temp dir for this test
import contextwell.store as store_module

with tempfile.TemporaryDirectory() as tmp:
    store_module.DB_PATH = Path(tmp) / "memories"

    from contextwell.embedder import embed
    from contextwell.project import detect_project_id
    from contextwell.schema import Memory
    from contextwell.store import forget, recall, scan, store

    # Test 1: embed
    vec = embed("We chose LanceDB for hybrid search support")
    assert len(vec) == 384, f"Expected 384 dims, got {len(vec)}"
    assert abs(sum(x**2 for x in vec) - 1.0) < 0.01, "Expected normalized vector"
    print("✓ embed: 384-dim normalised vector")

    # Test 2: store
    m = Memory(content="We chose LanceDB over ChromaDB for hybrid search support", type="decision", scope="global")
    m.embedding = embed(m.content)
    mid = store(m)
    assert len(mid) == 36, f"Expected UUID, got {mid}"
    print(f"✓ store: ID={mid[:8]}")

    # Test 3: store second memory
    m2 = Memory(content="Use all-MiniLM-L6-v2 as the default embedding model", type="decision", scope="global")
    m2.embedding = embed(m2.content)
    store(m2)
    print("✓ store: second memory added")

    # Test 4: recall
    results = recall(embed("database choice"), k=2)
    assert len(results) > 0, "Expected at least one result"
    assert "_distance" not in results[0], "_distance should be stripped"
    assert "embedding" not in results[0], "embedding should be stripped"
    print(f"✓ recall: top result = {results[0]['content'][:60]!r}")

    # Test 5: scan (no vector)
    rows = scan(memory_type="decision", limit=10)
    assert len(rows) == 2, f"Expected 2 decisions, got {len(rows)}"
    print(f"✓ scan: {len(rows)} decision memories")

    # Test 6: forget
    deleted = forget(mid)
    assert deleted, "Expected deletion to succeed"
    rows_after = scan(limit=10)
    assert len(rows_after) == 1, f"Expected 1 memory after delete, got {len(rows_after)}"
    print("✓ forget: memory deleted")

    # Test 7: project detection
    pid = detect_project_id(str(Path(__file__).parent))
    assert pid is not None, "Expected project ID (contextwell is a git repo)"
    assert len(pid) == 16, f"Expected 16-char hash, got {len(pid)}"
    print(f"✓ project detect: ID={pid}")

    print("\nAll tests passed ✓")
