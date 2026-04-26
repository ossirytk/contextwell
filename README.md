# contextwell

A persistent semantic memory MCP server for GitHub Copilot CLI and VS Code Copilot, built with Python and Rust (PyO3).

Store facts, decisions, code snippets, and notes across sessions. Recall them by meaning using vector similarity search. Gives Copilot a long-term memory layer that persists across projects and conversations.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Option A — uv tool install (recommended)](#option-a--uv-tool-install-recommended)
  - [Option B — Clone the repository](#option-b--clone-the-repository)
- [Client Configuration](#client-configuration)
  - [GitHub Copilot CLI](#github-copilot-cli)
  - [VS Code](#vs-code)
- [Available Tools](#available-tools)
- [Environment Variables](#environment-variables)
- [Notes & Caveats](#notes--caveats)
- [Development](#development)
- [Architecture](#architecture)

---

## Prerequisites

- **Python 3.12+** — managed by `uv`
- **Rust toolchain** — required to compile the `_core` extension ([install rustup](https://rustup.rs))
- **uv** — Python package manager ([install uv](https://docs.astral.sh/uv/getting-started/installation/))

Verify your setup:

```bash
python --version   # 3.12+
rustc --version    # any recent stable
uv --version
```

---

## Installation

### Option A — uv tool install (recommended)

Installs `contextwell` as an isolated tool and puts the `contextwell` command on your `PATH`. The Rust extension is compiled automatically during install.

```bash
uv tool install git+https://github.com/ossirytk/contextwell
```

Verify the install:

```bash
contextwell --help
```

To upgrade later:

```bash
uv tool upgrade contextwell
```

To uninstall:

```bash
uv tool uninstall contextwell
```

### Option B — Clone the repository

Use this if you want to inspect or modify the source.

```bash
git clone https://github.com/ossirytk/contextwell
cd contextwell
uv sync
uv run maturin develop   # compiles the Rust extension into the venv
```

Run the server manually to test it:

```bash
uv run contextwell
```

---

## Client Configuration

### GitHub Copilot CLI

Add to `~/.copilot/mcp-config.json`.

**If you used `uv tool install` (Option A):**

```json
{
  "mcpServers": {
    "contextwell": {
      "type": "stdio",
      "command": "contextwell"
    }
  }
}
```

**If you cloned the repository (Option B):**

```json
{
  "mcpServers": {
    "contextwell": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "--directory", "/absolute/path/to/contextwell", "contextwell"]
    }
  }
}
```

Replace `/absolute/path/to/contextwell` with the actual path on your machine.

Restart Copilot CLI after saving. Use `/mcp` or `/env` to confirm the server is loaded.

---

### VS Code

Add to your **User Settings** (`settings.json`) or a **workspace-level** `.vscode/mcp.json`.

**If you used `uv tool install` (Option A):**

```json
{
  "mcp": {
    "servers": {
      "contextwell": {
        "type": "stdio",
        "command": "contextwell"
      }
    }
  }
}
```

**If you cloned the repository (Option B):**

```json
{
  "mcp": {
    "servers": {
      "contextwell": {
        "type": "stdio",
        "command": "uv",
        "args": ["run", "--directory", "/absolute/path/to/contextwell", "contextwell"]
      }
    }
  }
}
```

Reload the VS Code window after saving. Open the Copilot Chat panel and confirm contextwell tools appear.

---

## Available Tools

| Tool | Description |
|------|-------------|
| `remember` | Store a new memory — fact, decision, code snippet, todo, or chat extract |
| `recall` | Search memories by meaning using semantic similarity |
| `forget` | Delete a memory by ID |
| `list_memories` | Browse memories with scope/type/tag/date filters |
| `update` | Edit content, type, tags, or source of an existing memory in-place; re-embeds automatically if content changes |
| `remember_file` | Ingest a markdown file into memory chunks with inferred metadata |
| `remember_batch` | Store many memories in one call with batched embedding |
| `compress_memories` | Replace similar memories with a single summary memory |
| `export_memories` | Export memories to JSON, Markdown, or Org-mode |
| `memory_stats` | Show aggregate counts, timestamps, and store size |

Memories can be scoped as `global` (across all projects) or `project` (tied to the current git repository, auto-detected from the working directory).

### Example prompts

> *"Remember that we chose LanceDB over ChromaDB for its hybrid search support"*  
> *"What decisions have we made about the database layer?"*  
> *"Recall anything about authentication from last week"*  
> *"Remember this as a project decision — scope='project'"*  
> *"What have we decided in this project so far?"*

Memory is stored at `~/.contextwell/memories/`.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CONTEXTWELL_EMBED_PROVIDER` | `sentence-transformers` | Embedding backend: `sentence-transformers` or `openai` |
| `CONTEXTWELL_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Model name; defaults to `text-embedding-3-small` when provider is `openai` |
| `CONTEXTWELL_EMBED_DIM` | `384` | Embedding vector dimension — must match the model; change if you switch models |
| `CONTEXTWELL_STORE_DIR` | `~/.contextwell/memories` | Path to the LanceDB vector store |
| `CONTEXTWELL_CHUNKING` | _(unset)_ | Set to `1` to enable automatic content chunking on `remember` |
| `CONTEXTWELL_CHUNK_SIZE` | `400` | Word count threshold per chunk (requires `CONTEXTWELL_CHUNKING=1`) |
| `CONTEXTWELL_HYBRID` | _(unset)_ | Set to `1` to enable BM25 + vector hybrid search (requires the `hybrid` extra) |

---

## Notes & Caveats

**lancedb is pinned to `0.30.0`** — newer versions may lack a Windows wheel.

**Embedding model changed in v0.1:** The default model changed from `all-MiniLM-L6-v2` to
`BAAI/bge-small-en-v1.5` (same 384 dimensions, different embedding space). Existing vectors are
incompatible. If you have an existing store, delete `~/.contextwell/memories` and re-add your
memories, or pin the old model:

```bash
# PowerShell
$env:CONTEXTWELL_EMBED_MODEL="all-MiniLM-L6-v2"
# bash/fish
export CONTEXTWELL_EMBED_MODEL=all-MiniLM-L6-v2
```

**Optional hybrid search** (BM25 + vector RRF) — not included in the default install:

```bash
uv tool install "contextwell[hybrid] @ git+https://github.com/ossirytk/contextwell"
```

Then enable hybrid retrieval at runtime by setting the environment variable:

```bash
# PowerShell
$env:CONTEXTWELL_HYBRID="1"
# bash/fish
export CONTEXTWELL_HYBRID=1
```

Or pass it directly in your MCP client configuration via the `env` block (see [Client Configuration](#client-configuration)).

---

## Development

```powershell
git clone https://github.com/ossirytk/contextwell
cd contextwell
uv sync

# Build Rust extension (required before running)
uv run maturin develop

# Lint Python
uv run ruff check .

# Format Python
uv run ruff format .

# Lint Rust
cargo clippy -- -D warnings

# Format Rust
cargo fmt

# Run tests
uv run pytest
```

---

## Architecture

```
contextwell/
├── python/contextwell/
│   ├── server.py       # FastMCP server and tool definitions
│   ├── embedder.py     # Sentence-transformers embedding wrapper (lazy load)
│   ├── store.py        # LanceDB vector store I/O (search, scan, store, forget)
│   ├── schema.py       # Memory dataclass and type literals
│   ├── project.py      # Git root detection for project-scoped memories
│   └── _core           # ← compiled from src/lib.rs via PyO3
├── src/lib.rs          # Rust: MemoryRecord, search_candidates (RRF)
└── Cargo.toml
```

The Python layer handles the MCP protocol (FastMCP), embedding models, and the LanceDB vector store.
The Rust extension (`_core`) handles performance-critical operations: the `MemoryRecord` data container
and Reciprocal Rank Fusion (RRF) scoring for hybrid dense + sparse retrieval.

The `_core` import is guarded in `__init__.py` — the Python package is importable without a compiled extension (e.g. in CI before the Rust build step).
