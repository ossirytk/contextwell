---
name: contextwell
description: Persistent semantic memory across sessions. Use this skill when the user wants to remember a fact, decision, code snippet, or note for later; recall something from past sessions; manage stored memories; or export/compress the memory store. Invoke for prompts like "remember this", "recall what we decided about X", "what do you know about Y", "store this decision", "forget memory ID", or "show my memories".
---

## Overview

contextwell is a vector-backed semantic memory store. Memories are embedded and stored in LanceDB at `~/.contextwell/memories/`. Recall works by meaning (semantic similarity), not keyword matching. Memories have a `scope`: `global` (across all projects) or `project` (tied to the current git repository, auto-detected from CWD).

## Memory Types

`fact` · `decision` · `code` · `chat` · `todo`

## Available Tools

| Tool | When to use |
|------|-------------|
| `contextwell-remember` | Store a new memory. Required: `content`. Optional: `type`, `scope` (`global`/`project`), `tags`, `source`, `allow_duplicate`. |
| `contextwell-recall` | Search memories by meaning. Required: `query`. Optional: `scope`, `type`, `tags`, `k` (max results), `rerank`. |
| `contextwell-forget` | Delete a memory by ID (full or first 8 chars). |
| `contextwell-list_memories` | Browse stored memories. Filters: `scope`, `type`, `tags`, `since`, `until`, `limit`. |
| `contextwell-update` | Edit an existing memory in-place. Required: `memory_id`. Optional: `content`, `type`, `tags`, `source`. Re-embeds automatically if content changes. |
| `contextwell-remember_file` | Ingest a Markdown file, splitting on headers into individual memories. Required: `path`. Optional: `scope`, `tags`, `type_hint`, `source`. |
| `contextwell-remember_batch` | Store many memories in one call with batched embedding. Required: `memories` list. |
| `contextwell-compress_memories` | Replace a cluster of similar memories with a single summary. Required: `summary`. Optional: `type`, `scope`, `threshold`, `tags`. |
| `contextwell-export_memories` | Export memories to JSON, Markdown, or Org-mode. Optional: `format`, `scope`, `type`, `tags`, `since`, `until`, `path`, `limit`. |
| `contextwell-memory_stats` | Show total count, breakdown by type/scope, oldest/newest timestamps, and store size. |

## Guidance

- **Remembering**: infer `type` from context — use `decision` for architectural choices, `code` for snippets, `fact` for general knowledge, `todo` for action items. Default to `global` scope unless the user is clearly working on a specific project.
- **Recalling**: prefer `contextwell-recall` for semantic search. Use `contextwell-list_memories` when the user wants to browse or filter by date/tag.
- **Project scope**: when `scope='project'`, contextwell auto-detects the git root from CWD. To set the working directory explicitly, prefix `source` with `cwd:<path>`.
- **Before storing duplicates**: the tool checks for near-duplicates by default. Set `allow_duplicate=true` only when the user explicitly wants to store something similar.
- **Compression**: use `contextwell-compress_memories` after long sessions to condense overlapping memories. Write a high-quality `summary` yourself that captures the essential meaning.
