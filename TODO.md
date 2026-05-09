# Contextwell — Improvement Ideas

## Completed

The following items from the original TODO have been implemented:

- **PyO3 version cap (Python 3.14+)** — bumped to `pyo3 = "0.28"` in Cargo.toml; added `#[pyclass(from_py_object)]` to silence deprecation.
- **`recall` missing date filters** — `since`/`until` params added to `recall()` in store.py and exposed on the MCP tool.
- **Embedding model migration path** — `reembed_all` MCP tool re-embeds all stored memories in batches with the current model.
- **`memory_stats` stale memories flag** — `stale_days` param added; result includes `stale_count` for memories never updated past the threshold.
- **Time-based memory expiry (TTL)** — `expires_at` field on `Memory`; expired memories silently skipped by `recall`/`scan`; `purge_expired` MCP tool deletes expired rows.
- **`remember_file` for non-markdown formats** — `file_import.py` added with `.org` (headline), `.txt` (paragraph), and source-code (regex function/class) parsers; `remember_file` dispatches by extension.
- **Bulk re-embed tool** — `reembed_all(batch_size)` in store.py; MCP tool exposed.
- **`recall` result explanation** — `include_score` param; results include `score` key (cosine similarity for dense, RRF score for hybrid).
- **Project-scope memory listing without git** — `detect_project_id_from_path` in project.py; `scope_path` param on all relevant MCP tools.

---

## Bugs

- **`_recall_hybrid` mutates cached row dicts (critical)** — when `include_score=True`, the code does `row["score"] = …` directly on the dict in `id_to_row`, mutating the shared reference. If the same memory appears in a later query in the same session, it will carry a stale score. Fix: copy the dict before mutating — `row = dict(id_to_row[mid])`.

- **`expires_at` accepts arbitrary strings without validation (high)** — the expiry filter is lexical string comparison, so a typo like `"not-a-date"` silently never expires while `"0000-00-00"` immediately expires. Add ISO 8601 validation in `remember` / `remember_batch` in server.py and return an error string for invalid values.

- **`scan` / `export_memories` have no stable sort order (high)** — LanceDB result order is non-deterministic without an explicit order-by, so exports can differ between runs for the same filter. Add a deterministic sort (e.g. `created_at` DESC, then `id`) before limiting in `scan()`.

- **`reembed_all` silently misses rows added during execution (medium)** — `total = table.count_rows()` is captured once; rows added while re-embedding are skipped. Document this limitation and recommend pausing writes during migration, or re-check count each iteration.

---

## Testing

- **No tests for `file_import.py` (critical)** — `.org`, `.txt`, and source-code parsers have zero test coverage. Add a `tests/test_file_import.py` covering: headline splitting, paragraph fallback, function/class detection per language family, empty files, files with only whitespace, and very large single sections.

- **No tests for TTL / `expires_at` lifecycle (high)** — no coverage for: recall silently skipping an expired memory, `purge_expired()` deleting the right rows, a memory with empty `expires_at` surviving, or `remember_batch` writing `expires_at` per item. Add to `test_integration.py`.

- **No tests for `project.py` failure modes (high)** — `detect_project_id` is tested for the happy path but not for: running outside a git repo, git command timeout/failure, and `detect_project_id_from_path` on nonexistent or symlinked paths.

- **No server-level tool tests for newer tools (high)** — `remember_file`, `purge_expired`, `reembed_all`, and the `scope_path` parameter wiring have no tests that validate tool return shapes and parameter mapping. Add thin server integration tests (can use `tmp_path` + monkeypatch like the existing suite).

- **No tests for recall `include_score` (medium)** — the score key and its correct range (0–1 for dense, positive float for RRF) are not tested.

---

## Documentation

- **README under-documents the current tool set (high)** — `scope_path`, `expires_at`, `purge_expired`, `reembed_all`, and non-markdown `remember_file` are missing from README examples and tool reference. Update to reflect the current MCP API surface.

- **No first-run / troubleshooting section (medium)** — README doesn't explain `CONTEXTWELL_STORE_DIR`, what happens on embedding dimension mismatch at startup, or how to run `reembed_all` after switching `CONTEXTWELL_EMBED_MODEL`.

- **`source="cwd:<path>|..."` encoding trick is undocumented (low)** — the `cwd:` prefix convention in `source` is referenced in tool docstrings but not in the README. Either document it clearly or replace with an explicit `cwd` parameter.

---

## Features

- **Bulk delete/purge by filter (medium)** — `forget` is ID-only. Add a `purge_memories(scope, type, tags, older_than)` tool for bulk cleanup without exporting everything.

- **Import from JSON export (medium)** — `export_memories(format="json")` claims re-importability via `remember_batch`, but there is no `import_memories` tool and the JSON schema isn't documented. Add an `import_memories(path)` tool or document the manual round-trip.

- **Pagination for `recall` / `list_memories` (medium)** — results are capped by `k`/`limit` with no cursor or offset. Add `offset: int = 0` to `list_memories` (and optionally `recall`) to allow page-through for large stores.

---

## Performance

- **`memory_stats` full-table scan and no server-side aggregation (medium)** — currently paginates through all rows in Python. For large stores this is slow. Investigate LanceDB server-side count/aggregation APIs or cache the stat on a schedule.

- **Serial duplicate checks in `remember` / `remember_batch` (low)** — each item triggers a separate nearest-neighbour search. For `remember_batch` with many items, consider a single batched ANN pass to detect all duplicates at once.

---

## Operations

- **Startup dimension mismatch has no recovery path (high)** — the guard in `store.py` raises a clear error, but the error message doesn't guide the user to `reembed_all`. Update the error to mention running `reembed_all` after changing `CONTEXTWELL_EMBED_MODEL`/`CONTEXTWELL_EMBED_DIM`.

- **OpenAI embedder provider underspecified (medium)** — `CONTEXTWELL_EMBED_PROVIDER=openai` is partially supported but the required env vars (`OPENAI_API_KEY`, model name) and failure modes are not documented.

