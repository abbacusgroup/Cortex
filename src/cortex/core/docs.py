"""Shared document helpers used across retrieval and pipeline layers."""

from __future__ import annotations

from typing import Any


def summarize_doc(doc: dict[str, Any]) -> dict[str, Any]:
    """Compact summary of a knowledge object — id, title, type, project, created_at."""
    return {
        "id": doc.get("id", ""),
        "title": doc.get("title", ""),
        "type": doc.get("type", ""),
        "project": doc.get("project", ""),
        "created_at": doc.get("created_at", ""),
    }
