# AGENTS.md — Project Rules for AI Assistants (Python)

This repository is currently developed primarily from a **Windows dev drive** using **PowerShell** and **VS Code**.  
**WSL:Ubuntu**, **fish**, and **NvChad Neovim** remain supported alternative workflows.  
All tooling runs via terminal commands. Dependencies are managed with **uv**, and code quality is ensured by **ruff** (formatter + linter) and **pyrefly** (editor integration).

Contributors maintain CLI-first workflows with minimal, deterministic diffs that comply with repository standards.

GitHub Copilot agents and other LLM-based assistants use this file to align with project-specific practices.  
VS Code's agentic AI features can apply multi-file coordinated changes, so the rules below constrain that behavior.

---

## 0. Development Environment

- **Primary OS/workspace:** Windows dev drive
- **Supported alternative OS/workspace:** WSL 2 with Ubuntu
- **Editors:** VS Code (primary), NvChad Neovim (supported alternative)
- **Shells:** PowerShell (current default), `fish` in WSL (supported alternative)
- **Package Manager:** `uv` (fast, dependency-locked)
- **Formatter + Linter:** `ruff` (installed as tool via uv)
- **Editor Integration:** `pyrefly` (installed as tool via uv, provides code insights and can suggest modifications)

All terminal commands should be reproducible from the supported shell/editor combinations.

---

## 0.1 Available CLI Tools

The following tools are installed locally and available for use in terminal workflows and agent tasks:

| Tool | Purpose |
|------|---------|
| `diffutils` | File comparison (`diff`, `cmp`, `diff3`, `sdiff`) |
| `fd` | Fast, user-friendly alternative to `find` for file search |
| `fzf` | General-purpose fuzzy finder for interactive filtering |
| `ripgrep` (`rg`) | Fast regex search across files; prefer over `grep`/`Select-String` |
| `zip` | Archive creation and extraction |
| `tokei` | Count lines of code by language |
| `ast-grep` (`sg`) | Structural code search and rewriting using AST patterns |
| `jq` | JSON query and transformation CLI |
| `yq` | YAML/JSON/TOML query and transformation CLI |
| `hyperfine` | Command-line benchmarking with statistical output |
| `pre-commit` | Run and manage repository pre-commit hooks |
| `http` / `https` (HTTPie) | Human-friendly HTTP API client |
| `just` | Project task runner via `justfile` recipes |
| `difft` (difftastic) | Syntax-aware structural diffing |

Prefer these tools over PowerShell built-ins where applicable (e.g., use `rg` instead of `Select-String`, use `fd` instead of `Get-ChildItem` for file discovery).

### Preferred command order

- Content search: `rg` first, then `ast-grep` for structural/language-aware matching
- File discovery: `fd` first, then `rg --files` as a fallback
- JSON config inspection: `jq`
- YAML/TOML inspection: `yq`
- HTTP/API smoke checks: `http` / `https` (HTTPie)
- Task orchestration: `just` recipes when a `justfile` exists
- Diff/review: `difft` for syntax-aware diffs, `diff` for plain text diffs
- Performance comparisons: `hyperfine` for repeatable timing

### Avoid in autonomous runs

- Avoid interactive-only flows (for example `fzf` prompts) unless the user explicitly asks for interactive selection
- Avoid destructive git/file operations unless the user explicitly approves them
- Avoid long-running watch commands by default; use one-shot checks first, then switch to watch mode only when requested
- Avoid invoking `pre-commit run --all-files` on very large repos when a targeted path or hook is enough for the task

---

## 1. Authoritative Tools & Source of Truth

### Python
- Ruff is the ONLY formatter + linter.
- Ruff’s configuration in `pyproject.toml` is authoritative.
- Do NOT reformat using Black, isort, yapf, or any editor formatter.
- Pyrefly is allowed to suggest code modifications; incorporate them if they make sense.

### Cross-Editor Compatibility
- Contributors primarily use VS Code and still support Neovim workflows.  
- All changes must be reproducible via terminal commands.

---

## 2. Ruff Configuration Source

- Do not duplicate Ruff rules in this file.
- Treat `pyproject.toml` as the single source of truth for all Ruff settings (`[tool.ruff]`).
- If lint behavior changes, update `pyproject.toml` and rely on `uv run ruff ...` output for validation.

---

## 2.1 Tooling Reliability Notes

- VS Code agent/tool output can truncate long file previews and append helper text such as `Continue. The file is too long...`.
- Never paste tool helper/output text into source files.
- After large scripted edits, immediately validate file tails and rerun Ruff/tests before proceeding.

---

## 3. Terminal Workflows with uv

### Running Python Code

```powershell
# Using uv run (handles the environment automatically)
uv run python script.py
uv run python -m module_name

# PowerShell virtualenv activation
.venv\Scripts\Activate.ps1
python script.py
deactivate

# WSL/fish virtualenv activation (supported alternative)
. .venv/bin/activate.fish
python script.py
deactivate
```

### Linting and Formatting

```powershell
# Format code (in-place)
uv run ruff format .

# Check for lint issues
uv run ruff check .

# Fix auto-fixable lint issues
uv run ruff check --fix .

# Check specific file
uv run ruff check path/to/file.py
```

### Installing Dependencies

```powershell
# Add new dependency
uv add package_name

# Add dev dependency
uv add --dev package_name

# Sync environment from lock
uv sync

# Install in editable mode
uv pip install -e .
```

### Managing Python Versions

```powershell
# Check Python version used by uv
uv python list

# Pin to specific version
uv python pin 3.13
```

---

## 4. Editor Configuration

### VS Code Settings
- Use Ruff extension for real-time linting
- Pyrefly provides code insights and can identify improvements; use as needed
- Terminal integration: Use the integrated PowerShell terminal by default; WSL + `fish` remains a supported alternative

### NvChad/Neovim
- Configure LSP to use linting results from Ruff
- Consider nvim-lint with Ruff as linter
- Do not rely on editor auto-formatting; use `uv run ruff format` before committing

---

## 5. Git Workflow Discipline

- Run `uv run ruff check --fix .` before committing
- Run `uv run ruff format .` before committing
- Verify with `uv run ruff check .` (should be clean)
- Keep diffs minimal and focused on the change
- Do not include unrelated reformatting in commits

---

## 6. Rust / PyO3 Workflow

This project is a **mixed Rust + Python** package built with [maturin](https://www.maturin.rs/).
The Rust extension (`contextwell._core`) is compiled by maturin and imported by the Python layer.

### Layout

```
contextwell/
├── python/contextwell/   # Python package (FastMCP server, embedder, store, schema)
├── src/lib.rs            # Rust extension — contextwell._core (PyO3)
├── Cargo.toml
└── pyproject.toml        # maturin build backend
```

### Build commands

```powershell
# Compile the Rust extension and install into the active venv (development mode)
uv run maturin develop

# Build a release wheel
uv run maturin build --release

# Lint Rust code
cargo clippy -- -D warnings

# Format Rust code
cargo fmt

# Run Rust tests
cargo test
```

### Rust conventions

- Edition: 2021
- Lint standard: `cargo clippy -- -D warnings`
- Formatter: `cargo fmt` (always run before committing)
- The Rust module is exposed to Python as `contextwell._core`
- `MemoryRecord` and `search_candidates` (RRF fusion) live in Rust; the rest is Python
- PyO3 version is pinned in `Cargo.toml`; keep in sync with the maturin version in dev deps

### Python / Rust boundary rules

- The Rust layer owns: data containers, scoring/ranking algorithms (RRF), future BM25 index
- The Python layer owns: MCP protocol (FastMCP), embedding model calls, LanceDB I/O, config
- Never call the embedding model from Rust — keep that in `embedder.py`
- The `_core` module import in `__init__.py` is guarded with a try/except so the Python
  package remains importable even without a compiled extension (e.g. in CI before build)

---

## 7. Improvements

Potential features and enhancements, grouped by what they require.
Mark ideas here rather than opening issues for every passing thought.

### No additional dependencies (pure Python / Rust)

- **Hybrid search (RRF)** — `search_candidates` in `_core` already implements RRF;
  wire it up once a BM25/FTS index is added alongside LanceDB vector search.
- **Project scope auto-detection** — detect git root from CWD and hash it as `project_id`
  so `scope='project'` works without manual configuration.
- **Cross-encoder reranking** — add a `reranker.py` using MS-Marco MiniLM (already used
  in light-chat); call after `recall` when `k > 5` for better precision.
- **Chunking** — split long content in `remember` before embedding; store chunks with
  a shared `parent_id` for reassembly on recall.
- **Org-mode export** — `export_to_org` tool writes memories to `~/.contextwell/memories.org`
  grouped by type and scope, with `CREATED:` timestamps and tag drawers.

### Requires optional CLI tools

These enhancements depend on tools that may not be present on all systems.
Check with `Get-Command <tool>` before implementing; degrade gracefully if absent.

| Tool | Enhancement |
|------|-------------|
| `rg` | Full-text search across exported org/markdown files as a fallback index |
| `jq` | Structured output transformation for piping `list_memories` output into other tools |
| `fzf` | Interactive memory selection — only when user explicitly requests interactive mode |

### Optional Python dependencies (not in default install)

These are heavy and not installed by default. Add with `uv add <package>` when needed.

| Package | Enhancement |
|---------|-------------|
| `sentence-transformers` | Embedding model (`all-MiniLM-L6-v2`); required for `remember`/`recall` |
| `lancedb` | Vector store; required for persistence beyond the skeleton |
| `rank-bm25` | BM25 sparse retrieval; pairs with RRF in `_core` for hybrid search |

