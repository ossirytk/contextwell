"""Project scope utilities — auto-detect git root for project-scoped memories."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def detect_project_id(cwd: str | None = None) -> str | None:
    """Return a stable project ID derived from the git root of cwd.

    Runs `git rev-parse --show-toplevel` from the given directory (or CWD).
    Returns a short SHA-256 hash of the absolute git root path, or None if
    not inside a git repository.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=3,
            check=False,
        )
        if result.returncode != 0:
            return None
        root = Path(result.stdout.strip()).resolve()
        return hashlib.sha256(str(root).encode()).hexdigest()[:16]
    except Exception:
        return None
