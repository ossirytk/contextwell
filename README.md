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
| `list_memories` | Browse stored memories with optional scope and type filters |

---

## Architecture

```
contextwell/
├── python/contextwell/
│   ├── server.py       # FastMCP server and tool definitions
│   ├── embedder.py     # Sentence-transformers embedding wrapper (lazy load)
│   ├── store.py        # LanceDB vector store I/O
│   ├── schema.py       # Memory dataclass and type literals
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

Register in your MCP client config and use naturally:

> *"Remember that we chose LanceDB over ChromaDB for its hybrid search support"*  
> *"What decisions have we made about the database layer?"*  
> *"Recall anything about authentication from last week"*

Memory is stored at `~/.contextwell/memories/`.

---

## Optional dependencies

The heavy dependencies are not installed by default. Add them when ready:

```powershell
uv add sentence-transformers   # embedding model (all-MiniLM-L6-v2)
uv add lancedb                 # vector store
uv add rank-bm25               # sparse retrieval for hybrid search
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

Local MCP memory storage for copilot. Make copilot even better.
