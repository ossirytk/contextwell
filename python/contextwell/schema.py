"""Memory data model and type definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

MemoryType = Literal["code", "chat", "decision", "todo", "fact"]
MemoryScope = Literal["project", "global"]


@dataclass
class Memory:
    content: str
    type: MemoryType = "fact"
    scope: MemoryScope = "global"
    project_id: str | None = None
    tags: list[str] = field(default_factory=list)
    source: str | None = None
    id: str = field(default_factory=lambda: str(uuid4()))
    embedding: list[float] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    parent_ids: list[str] = field(default_factory=list)
    chunk_of: str = ""
