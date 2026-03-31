"""contextwell — Persistent semantic memory MCP server.

The _core extension module (Rust/PyO3) is imported here when available.
Falls back gracefully so the Python layer remains testable without a compiled extension.
"""

from contextwell.__version__ import __version__

try:
    from contextwell._core import MemoryRecord, search_candidates

    _CORE_AVAILABLE = True
except ImportError:
    _CORE_AVAILABLE = False

__all__ = ["_CORE_AVAILABLE", "MemoryRecord", "__version__", "search_candidates"]
