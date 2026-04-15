"""Memory export formatters: JSON, Markdown, and Org-mode."""

from __future__ import annotations

import json
from datetime import UTC, datetime

_EXPORT_FIELDS = (
    "id",
    "content",
    "type",
    "scope",
    "project_id",
    "tags",
    "source",
    "created_at",
    "updated_at",
    "parent_ids",
    "chunk_of",
)


def _filter_fields(row: dict) -> dict:
    """Return only the exportable fields from a raw store row."""
    return {k: row.get(k, "") for k in _EXPORT_FIELDS}


def _group_by_type(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        key = str(row.get("type", "fact"))
        groups.setdefault(key, []).append(row)
    return groups


def to_json(rows: list[dict]) -> str:
    """Serialise memories as a JSON array (no embeddings)."""
    return json.dumps([_filter_fields(r) for r in rows], indent=2, ensure_ascii=False)


def to_markdown(rows: list[dict], scope_label: str = "") -> str:
    """Render memories as human-readable Markdown grouped by type."""
    now = datetime.now(UTC).strftime("%Y-%m-%d")
    header_parts = [f"*Exported: {now}*"]
    if scope_label:
        header_parts.append(f"*Scope: {scope_label}*")
    header_parts.append(f"*Total: {len(rows)} memor{'y' if len(rows) == 1 else 'ies'}*")

    lines: list[str] = [
        "# Contextwell Memory Export",
        "",
        "  ".join(header_parts),
        "",
        "---",
    ]

    for memory_type, group in _group_by_type(rows).items():
        lines += ["", f"## {memory_type}", ""]
        for row in group:
            mid = str(row.get("id", ""))
            snippet = str(row.get("content", ""))[:60].replace("\n", " ")
            lines.append(f"### #{mid[:8]} — {snippet}")
            meta: list[str] = []
            if row.get("created_at"):
                meta.append(f"- **Created:** {row['created_at']}")
            if row.get("updated_at"):
                meta.append(f"- **Updated:** {row['updated_at']}")
            if row.get("tags"):
                meta.append(f"- **Tags:** {', '.join(row['tags'])}")
            if row.get("source"):
                meta.append(f"- **Source:** {row['source']}")
            if row.get("project_id"):
                meta.append(f"- **Project:** {str(row['project_id'])[:16]}")
            if row.get("parent_ids"):
                parents = ", ".join(f"#{p[:8]}" for p in row["parent_ids"])
                meta.append(f"- **Compresses:** {parents}")
            lines += meta
            lines += ["", str(row.get("content", "")), "", "---"]

    return "\n".join(lines) + "\n"


def to_org(rows: list[dict], scope_label: str = "") -> str:
    """Render memories as Org-mode with PROPERTIES drawers grouped by type."""
    now_date = datetime.now(UTC).strftime("%Y-%m-%d %a")
    lines: list[str] = [
        "#+TITLE: Contextwell Memory Export",
        f"#+DATE: [{now_date}]",
        "#+STARTUP: overview",
    ]
    if scope_label:
        lines.append(f"#+DESCRIPTION: scope={scope_label}")
    lines += ["", f"# {len(rows)} memor{'y' if len(rows) == 1 else 'ies'} exported", ""]

    for memory_type, group in _group_by_type(rows).items():
        lines += [f"* {memory_type}", ""]
        for row in group:
            mid = str(row.get("id", ""))
            snippet = str(row.get("content", ""))[:60].replace("\n", " ")
            heading = f"** {snippet} (#{mid[:8]})"
            # Build optional Org tag string from memory tags
            tags = list(row.get("tags") or [])
            if tags:
                safe_tags = ":".join(t.replace(" ", "_") for t in tags)
                heading += f"  :{safe_tags}:"
            lines.append(heading)
            lines.append(":PROPERTIES:")
            lines.append(f":ID:       {mid}")
            created = str(row.get("created_at", ""))
            if created:
                try:
                    dt = datetime.fromisoformat(created)
                    org_ts = dt.strftime("[%Y-%m-%d %a %H:%M]")
                except ValueError:
                    org_ts = created
                lines.append(f":CREATED:  {org_ts}")
            updated = str(row.get("updated_at", ""))
            if updated:
                try:
                    dt = datetime.fromisoformat(updated)
                    org_ts = dt.strftime("[%Y-%m-%d %a %H:%M]")
                except ValueError:
                    org_ts = updated
                lines.append(f":UPDATED:  {org_ts}")
            if row.get("source"):
                lines.append(f":SOURCE:   {row['source']}")
            if row.get("project_id"):
                lines.append(f":PROJECT:  {str(row['project_id'])[:16]}")
            if row.get("parent_ids"):
                parents = " ".join(row["parent_ids"])
                lines.append(f":PARENTS:  {parents}")
            lines.append(":END:")
            lines += ["", str(row.get("content", "")), ""]

    return "\n".join(lines) + "\n"
