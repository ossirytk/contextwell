# Contextwell — Improvement Roadmap

Tracked improvements for making contextwell a better long-term memory source for Copilot.
Items are roughly ordered by priority. Cross off when done.

---

## ~~1. Markdown File Ingestion~~ ✅

**Tool**: `remember_file(path, scope, tags, type_hint)`

Parse a markdown file and store its sections as individual memories. Designed to ingest Copilot's own artifacts: `plan.md`, compact summaries, `copilot-instructions.md`, `.github/copilot-instructions.md`.

- Split on `##`/`###` headers — each section becomes one memory
- YAML front-matter sets default `type`, `tags`, `scope` for the whole file
- `source` is set to the file path automatically
- Type heuristics: section titled `Decision` → `decision`, `TODO` → `todo`, etc.
- Flat fallback: if no headers, store the whole file as one memory (with truncation warning if large)
- Returns a summary of how many memories were stored and their IDs

---

## ~~2. Tags Filtering in `recall` and `list_memories`~~ ✅

Tags are stored today but have zero query power. Add a `tags` filter parameter to both `recall` and `list_memories` so the model can ask for memories with a specific label.

- `list_memories(tags=["auth", "security"])` returns memories with any of the given tags
- `recall(query=..., tags=["architecture"])` restricts vector search to tagged subset
- No schema change required — `tags` column already exists; needs a WHERE clause

---

## ~~3. `update` Tool — In-Place Memory Editing~~ ✅

Allow editing an existing memory without delete + recreate. Preserves original ID and `created_at`.

- Accept partial updates: `content`, `tags`, `type`, `source`
- Re-embed automatically if `content` changes
- Add `updated_at` timestamp to schema (nullable, set on first update)
- Partial ID match (first 8 chars) consistent with `forget`

---

## ~~4. Deduplication Check in `remember`~~ ✅

Before storing a new memory, check for near-identical existing memories.

- Run `recall(k=3)` on the new content before storing
- If cosine similarity > 0.95, return a warning and the existing memory ID instead of storing
- `allow_duplicate=True` parameter on `remember` to bypass the check
- Prevents silent quality decay as the store grows

---

## ~~5. Hybrid Search (BM25 + Vector + RRF)~~ ✅

Wire up the existing `search_candidates` RRF function in `_core` with a BM25 sparse index.

- `rank-bm25` (already listed as optional dep) for in-memory BM25 index
- Index rebuilt at startup from the LanceDB table
- `recall` runs both searches, fuses via `_core.search_candidates(dense_ids, sparse_ids)`
- Opt-in via `CONTEXTWELL_HYBRID=1` env var initially
- Catches exact keyword matches (function names, error codes) that pure vector search misses

---

## ~~6. Configurable Embedding Model~~ ✅

Make the model name and dimension configurable without patching source.

- `CONTEXTWELL_EMBED_MODEL` env var overrides the default (`all-MiniLM-L6-v2`)
- `CONTEXTWELL_EMBED_DIM` overrides dimension (baked into LanceDB schema at table creation)
- Guard: if an existing table has a different dimension, refuse to start with a clear error
- API adapter: thin `embedder_openai.py` for `CONTEXTWELL_EMBED_PROVIDER=openai`
- Candidate upgrade: `bge-small-en-v1.5` (same size, better MTEB retrieval score)

---

## ✅ 7. Memory Compression / Summarization

Mirrors Copilot's `/compact` for the memory store.

- `compress_memories(type, scope, summary)` — caller (the model) provides the summary text
- Clusters semantically similar memories (cosine > 0.85) within the same scope/type
- Replaces the cluster with one condensed memory; stores source IDs in a `parent_ids` field
- Requires schema change: new `parent_ids` nullable list column
- Depends on item 3 (`update`) for clean implementation

---

## ✅ 8. Date Range Filtering

Add `since` / `until` parameters to `list_memories`.

- `list_memories(since="2025-01-01", until="2025-03-31")` filters by `created_at`
- Useful for "what did we decide this sprint?" queries
- No schema change — `created_at` already stored as ISO string; needs WHERE clause

---

## ✅ 9. Batch `remember`

`remember_batch(memories: list[dict])` — store multiple memories in one MCP call.

- Uses existing `embed_batch` for efficient batched embedding
- Each item has the same fields as `remember`
- Returns a list of IDs and a summary
- Needed to make markdown import of large files efficient

---

## ✅ 10. Export (JSON / Markdown)

`export_memories(format, scope, path)` — portable snapshot of the memory store.

- **JSON**: full fidelity, re-importable
- **Markdown**: human-readable, grouped by type/scope, metadata as front-matter per section
- **Org-mode**: grouped by type/scope with `CREATED:` timestamps and tag drawers
- Enables backup, cross-machine migration, and team sharing of project memories

---

## ✅ 11. Cross-Encoder Reranking

After initial vector recall, re-score with a cross-encoder for higher-precision ordering.

- `cross-encoder/ms-marco-MiniLM-L-6-v2` (sentence-transformers, already installed)
- `recall(query, rerank=True)` opt-in parameter
- Most impactful when `k > 5`; adds ~100–300 ms latency per call

---

## ✅ 12. Chunking for Long Content

Automatically split content > N tokens in `remember` into overlapping chunks.

- Store chunks with shared `parent_id` (new nullable schema column)
- On recall, deduplicate chunks from the same parent before returning
- Captures local context better than single-vector embedding of long blocks
- High complexity; defer until long-form content (> 512 tokens) is common in practice

---

## ✅ 13. Memory Stats Tool

`memory_stats()` — dashboard summary for situational awareness.

- Total count, breakdown by type and scope
- Oldest / newest entry timestamps
- Estimated duplicate clusters (count of pairs with similarity > 0.9)
- Store size on disk
- Very low effort; mostly `scan` + aggregation

---

## Notes

- Items 5, 9 depend on no schema changes — safe to implement in any order
- Item 7 depends on item 3 (`update`)
- Item 12 requires a schema migration; defer
- Optional heavy dependencies (`rank-bm25`, cross-encoder) should remain opt-in and not listed in core `[project.dependencies]`
