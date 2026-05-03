"""Shared CLI helpers.

Single chokepoint utilities used by multiple CLI entry points (the main
typer app and the setup wizard). Keeping them here prevents the helpers
from drifting independently in each module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from cortex.core.config import CortexConfig
from cortex.core.errors import StoreLockedError
from cortex.db.store import Store


def open_store_or_exit(config: CortexConfig) -> Store:
    """Open a Store, exiting cleanly with a user-friendly message if the graph DB is locked."""
    try:
        return Store(config)
    except StoreLockedError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from e


def register_with_claude_code(spec: dict[str, Any]) -> Path:
    """Set ``mcpServers["cortex"] = spec`` in ``~/.claude/settings.json``.

    Preserves other entries in the file. Creates the file (and parent
    directory) if missing. Returns the path written.
    """
    settings_path = Path.home() / ".claude" / "settings.json"
    settings: dict[str, Any] = {}
    if settings_path.exists():
        settings = json.loads(settings_path.read_text())
    mcp_servers = settings.setdefault("mcpServers", {})
    mcp_servers["cortex"] = spec
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    return settings_path
