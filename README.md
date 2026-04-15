# contextwell

A persistent semantic memory MCP server for GitHub Copilot CLI, built with Python and Rust (PyO3).

Store facts, decisions, code snippets, and notes across sessions. Recall them by meaning using vector similarity search. Designed to give Copilot a long-term memory layer that persists across projects and conversations.

The Python layer handles the MCP protocol (FastMCP), embedding models, and the LanceDB vector store. The Rust extension (`_core`) handles performance-critical operations: the `MemoryRecord` data container and Reciprocal Rank Fusion (RRF) scoring for hybrid dense + sparse retrieval.

---

## Tools

| Tool | Description |
|------|-------------|
| `remember` | Store a new memory — fact, decision, code snippet, todo, or chat extract |
| `recall` | Search memories by meaning using semantic similarity |
| `forget` | Delete a memory by ID |
| `update` | Edit content, type, tags, or source of an existing memory in-place; re-embeds automatically if content changes |

Memories can be scoped as `global` (across all projects) or `project` (tied to the current git repository, auto-detected from the working directory).

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

The `_core` import is guarded in `__init__.py` — the Python package is importable without a compiled extension (e.g. in CI before the Rust build step).

---

## Usage

Build the Rust extension and run the MCP server:

```powershell
uv run maturin develop
uv run contextwell
```

### GitHub Copilot CLI

Add to `~/.copilot/mcp-config.json`:

```json
{
  "mcpServers": {
    "contextwell": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "--directory", "/path/to/contextwell", "contextwell"],
      "tools": ["*"]
    }
  }
}
```

Replace `/path/to/contextwell` with the absolute path to this repository. Restart Copilot CLI after saving — use `/mcp` or `/env` to verify the server is loaded.

### VS Code

Add to your `settings.json` or a workspace MCP config file:

```json
{
  "mcp": {
    "servers": {
      "contextwell": {
        "type": "stdio",
        "command": "uv",
        "args": ["run", "--directory", "/path/to/contextwell", "contextwell"]
      }
    }
  }
}
```

### Use naturally:

> *"Remember that we chose LanceDB over ChromaDB for its hybrid search support"*  
> *"What decisions have we made about the database layer?"*  
> *"Recall anything about authentication from last week"*  
> *"Remember this as a project decision — scope='project'"*  
> *"What have we decided in this project so far?"*

Memory is stored at `~/.contextwell/memories/`.

---

## Dependencies

Core dependencies (installed with `uv sync`):

| Package | Purpose |
|---------|---------|
| `fastmcp` | MCP server framework |
| `sentence-transformers` | Embedding model (`all-MiniLM-L6-v2`, 384-dim) |
| `lancedb==0.30.0` | Vector store with scalar index support |

> **Note:** `lancedb` is pinned to `0.30.0` — newer versions may lack a Windows wheel.

Optional, for future hybrid search:

```powershell
uv add rank-bm25   # sparse retrieval (BM25) for RRF hybrid search
```

---

## Development

```powershell
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

