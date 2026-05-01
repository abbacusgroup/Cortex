"""Shared CLI helpers.

Single chokepoint utilities used by multiple CLI entry points (the main
typer app and the setup wizard). Keeping them here prevents the helpers
from drifting independently in each module.
"""

from __future__ import annotations

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
