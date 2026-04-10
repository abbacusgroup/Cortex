"""Configuration loader for Cortex.

Precedence (highest → lowest):
  1. Environment variables (CORTEX_*)
  2. .env file in working directory
  3. ~/.cortex/.env
  4. Built-in defaults
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values

from cortex.core.constants import (
    DEFAULT_DATA_DIR,
    DEFAULT_HOST,
    DEFAULT_PORT,
    ENV_PREFIX,
    GRAPH_DB_FILE,
    SQLITE_DB_FILE,
)
from cortex.core.errors import ConfigError, ConfigPermissionError


def _env_key(name: str) -> str:
    return f"{ENV_PREFIX}{name.upper()}"


@dataclass(frozen=True)
class CortexConfig:
    """Immutable configuration for a Cortex instance."""

    # Paths
    data_dir: Path = DEFAULT_DATA_DIR

    # Server
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT

    # LLM
    llm_provider: str = ""
    llm_model: str = ""
    llm_api_key: str = ""

    # Embeddings
    embedding_provider: str = "sentence-transformers"  # "sentence-transformers" | "litellm"
    embedding_model: str = "all-mpnet-base-v2"
    embedding_api_key: str = ""

    # Logging
    log_level: str = "INFO"
    log_json: bool = True

    # Dashboard
    dashboard_password: str = ""
    # URL of the MCP HTTP server the dashboard connects to.
    # The dashboard never opens graph.db directly — it forwards every request
    # to this URL via an MCP client. The server is started separately with
    # ``cortex serve --transport mcp-http``.
    mcp_server_url: str = "http://127.0.0.1:1314/mcp"

    # Vault export path
    vault_path: str = ""

    # Derived paths (computed from data_dir)
    _extra: dict[str, str] = field(default_factory=dict, repr=False)

    @property
    def graph_db_path(self) -> Path:
        return self.data_dir / GRAPH_DB_FILE

    @property
    def sqlite_db_path(self) -> Path:
        return self.data_dir / SQLITE_DB_FILE

    @property
    def ontology_dir(self) -> Path:
        return self.data_dir / "ontology"


def load_config(
    *,
    env_file: Path | None = None,
    data_dir: Path | None = None,
) -> CortexConfig:
    """Load configuration with full precedence chain.

    Args:
        env_file: Explicit .env file path. If None, searches default locations.
        data_dir: Override data directory. If None, uses env or default.

    Returns:
        Frozen CortexConfig instance.

    Raises:
        ConfigError: If data_dir cannot be created.
        ConfigPermissionError: If data_dir exists but isn't writable.
    """
    # Collect .env values (lowest priority first, later overwrites)
    env_vals: dict[str, str] = {}

    # ~/.cortex/.env
    home_env = DEFAULT_DATA_DIR / ".env"
    if home_env.is_file():
        env_vals.update({k: v for k, v in dotenv_values(home_env).items() if v is not None})

    # Working directory .env
    cwd_env = Path.cwd() / ".env"
    if cwd_env.is_file():
        env_vals.update({k: v for k, v in dotenv_values(cwd_env).items() if v is not None})

    # Explicit .env file
    if env_file is not None and env_file.is_file():
        env_vals.update({k: v for k, v in dotenv_values(env_file).items() if v is not None})

    def _get(name: str, default: str = "") -> str:
        """Get config value with precedence: env var > .env > default."""
        env_key = _env_key(name)
        # Environment variables win
        if env_key in os.environ:
            return os.environ[env_key]
        # Then .env files (already merged by precedence)
        if env_key in env_vals:
            return env_vals[env_key]
        return default

    # Resolve data directory
    resolved_dir = data_dir or Path(_get("data_dir", str(DEFAULT_DATA_DIR)))
    resolved_dir = resolved_dir.expanduser().resolve()

    # Ensure data dir exists and is writable
    if resolved_dir.exists():
        if not os.access(resolved_dir, os.W_OK):
            raise ConfigPermissionError(
                f"Data directory exists but is not writable: {resolved_dir}",
                context={"path": str(resolved_dir)},
            )
    else:
        try:
            resolved_dir.mkdir(parents=True, mode=0o700)
        except OSError as e:
            raise ConfigError(
                f"Cannot create data directory: {resolved_dir}",
                context={"path": str(resolved_dir)},
                cause=e,
            ) from e

    port_str = _get("port", str(DEFAULT_PORT))
    try:
        port = int(port_str)
    except ValueError:
        port = DEFAULT_PORT

    mcp_server_url = _get("mcp_server_url", "http://127.0.0.1:1314/mcp")
    if mcp_server_url and not (
        mcp_server_url.startswith("http://") or mcp_server_url.startswith("https://")
    ):
        raise ConfigError(
            f"CORTEX_MCP_SERVER_URL must be http:// or https://, got: {mcp_server_url!r}",
            context={"value": mcp_server_url},
        )

    return CortexConfig(
        data_dir=resolved_dir,
        host=_get("host", DEFAULT_HOST),
        port=port,
        llm_provider=_get("llm_provider"),
        llm_model=_get("llm_model"),
        llm_api_key=_get("llm_api_key"),
        embedding_provider=_get("embedding_provider", "sentence-transformers"),
        embedding_model=_get("embedding_model", "all-mpnet-base-v2"),
        embedding_api_key=_get("embedding_api_key"),
        log_level=_get("log_level", "INFO"),
        log_json=_get("log_json", "true").lower() in ("true", "1", "yes"),
        dashboard_password=_get("dashboard_password"),
        mcp_server_url=mcp_server_url,
        vault_path=_get("vault_path"),
    )
