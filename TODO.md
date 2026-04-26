# Contextwell — Improvement Ideas

## Fixes

- **`recall` missing date filters** — `list_memories` supports `since`/`until` but semantic search (`recall`) doesn't. Add optional date-range params so you can scope recall to recent memories.
- **Embedding model migration path** — upgrading from the old default model (`all-MiniLM-L6-v2`) wipes the store with no recovery. Document a re-embed migration script or add a `contextwell migrate` CLI command that re-embeds all records with the new model.
- **`memory_stats` doesn't flag stale memories** — add a `stale_days` threshold that reports memories untouched (not recalled, not updated) for longer than N days, so users can prune without exporting everything.

## Enhancements

- **Time-based memory expiry (TTL)** — add an optional `expires_at` field to `remember` / `remember_batch`. `list_memories` and `recall` should silently skip expired memories. Add a `purge_expired` tool or auto-purge on startup.
- **`remember_file` for non-markdown formats** — currently only `.md` is supported. Add chunking strategies for `.org` (headline-based), plain `.txt` (paragraph-based), and source code files (function/class-based via AST or regex).
- **Bulk re-embed tool** — expose a `reembed` tool (or CLI subcommand) that iterates all stored memories and re-computes embeddings with the currently configured model. Essential when switching `CONTEXTWELL_EMBED_MODEL`.
- **`recall` result explanation** — optionally return the similarity score alongside each result so the user/agent can judge relevance quality, not just ordering.
- **Project-scope memory listing without git** — currently project scope silently falls back or errors if not in a git repo. Add a `scope_path` override param so non-git project directories can still be used.
