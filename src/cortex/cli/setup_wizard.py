"""Unified setup wizard for Cortex.

One command to go from ``pip install abbacus-cortex`` to fully operational:
data directory, stores, LLM, embeddings, dashboard password, background
services, Claude Code registration, and verification.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import typer

from cortex.cli._helpers import open_store_or_exit
from cortex.core.config import CortexConfig, load_config
from cortex.core.logging import setup_logging
from cortex.db.store import Store
from cortex.ontology.resolver import find_ontology

# ---------------------------------------------------------------------------
# LLM provider catalogue
# ---------------------------------------------------------------------------

_PROVIDERS: dict[int, dict[str, Any]] = {
    1: {
        "name": "Anthropic",
        "label": "Anthropic  (Claude) — recommended",
        "provider": "anthropic",
        "default_model": "claude-sonnet-4-20250514",
        "key_url": "https://console.anthropic.com/settings/keys",
        "needs_key": True,
    },
    2: {
        "name": "OpenAI",
        "label": "OpenAI     (GPT-4o, etc.)",
        "provider": "openai",
        "default_model": "gpt-4o",
        "key_url": "https://platform.openai.com/api-keys",
        "needs_key": True,
    },
    3: {
        "name": "Ollama",
        "label": "Ollama     (local, no API key)",
        "provider": "ollama",
        "default_model": "ollama/llama3.1",
        "key_url": None,
        "needs_key": False,
    },
    4: {
        "name": "Other",
        "label": "Other      (any litellm provider)",
        "provider": None,
        "default_model": None,
        "key_url": None,
        "needs_key": True,
    },
    5: {
        "name": "Skip",
        "label": "Skip       (configure later)",
        "provider": "",
        "default_model": "",
        "key_url": None,
        "needs_key": False,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _echo(msg: str = "") -> None:
    typer.echo(msg)


def _test_llm(config: CortexConfig, model: str, api_key: str, provider: str) -> tuple[bool, str]:
    """Attempt an LLM test call. Returns (success, message)."""
    from cortex.services.llm import LLMClient

    test_config = replace(config, llm_model=model, llm_api_key=api_key, llm_provider=provider)
    try:
        client = LLMClient(test_config)
        if not client.available:
            return False, "LLM client not available (check litellm installation)"
        client.complete("Say 'connected' in one word.")
        return True, "connected"
    except Exception as e:
        return False, str(e)


def _probe_http(url: str, retries: int = 5, delay: float = 1.0) -> bool:
    """Probe an HTTP endpoint, retrying on failure."""
    import httpx

    for attempt in range(retries):
        try:
            r = httpx.get(url, timeout=3.0)
            if r.status_code < 500:
                return True
        except (httpx.ConnectError, httpx.TimeoutException, OSError):
            pass
        if attempt < retries - 1:
            time.sleep(delay)
    return False


# ---------------------------------------------------------------------------
# Wizard steps
# ---------------------------------------------------------------------------

def _step_data_dir(config: CortexConfig) -> None:
    """Step 1: Ensure data directory exists."""
    _echo("[1/7] Data directory")
    if config.data_dir.exists():
        _echo(f"      {config.data_dir}/ (exists)")
    else:
        config.data_dir.mkdir(parents=True, mode=0o700)
        _echo(f"      {config.data_dir}/ (created)")


def _step_stores(config: CortexConfig) -> Store:
    """Step 2: Initialize stores and load ontology."""
    _echo("\n[2/7] Stores & ontology")
    store = open_store_or_exit(config)
    try:
        ontology_path = find_ontology()
        store.initialize(ontology_path)
        _echo("      Ontology loaded")
    except FileNotFoundError:
        _echo("      Ontology not found — skipping")
    _echo(f"      Graph:  {config.graph_db_path}")
    _echo(f"      SQLite: {config.sqlite_db_path}")
    return store


def _step_llm(config: CortexConfig, auto: bool) -> dict[str, str]:
    """Step 3: Configure LLM provider. Returns env vars to persist."""
    _echo("\n[3/7] LLM provider")

    env_updates: dict[str, str] = {}

    has_llm = bool(config.llm_model and config.llm_api_key)

    if has_llm and not auto:
        _echo(f"      Currently: {config.llm_model}")
        if typer.confirm("      Keep current LLM settings?", default=True):
            _echo("      Kept.")
            return env_updates

    if auto:
        # In auto mode, use whatever is already configured
        if config.llm_model:
            env_updates["CORTEX_LLM_MODEL"] = config.llm_model
        if config.llm_api_key:
            env_updates["CORTEX_LLM_API_KEY"] = config.llm_api_key
        if config.llm_provider:
            env_updates["CORTEX_LLM_PROVIDER"] = config.llm_provider
        if has_llm:
            _echo(f"      LLM: {config.llm_model}")
            ok, msg = _test_llm(config, config.llm_model, config.llm_api_key, config.llm_provider)
            _echo(f"      Testing... {msg}")
        else:
            _echo("      LLM: not configured (skipped in auto mode)")
        return env_updates

    # Interactive: show provider menu
    _echo("      Cortex uses an LLM to classify and discover relationships.")
    _echo("      Without one, content is stored but not classified.")
    _echo()
    for num, info in _PROVIDERS.items():
        _echo(f"        {num}. {info['label']}")
    _echo()

    while True:
        choice_str = typer.prompt("      Choose", default="1")
        try:
            choice = int(choice_str)
            if choice not in _PROVIDERS:
                raise ValueError
            break
        except ValueError:
            _echo("      Please enter a number 1-5.")

    provider_info = _PROVIDERS[choice]

    if choice == 5:
        _echo("      Skipped. Run `cortex setup` later to configure.")
        return env_updates

    # Determine provider string
    if provider_info["provider"] is None:
        provider_str = typer.prompt("      Provider (litellm prefix, e.g. 'together')")
    else:
        provider_str = provider_info["provider"]

    # Determine model
    if provider_info["default_model"] is None:
        model = typer.prompt("      Model (e.g. 'together/meta-llama/Llama-3-70b')")
    else:
        model = typer.prompt("      Model", default=provider_info["default_model"])

    # API key
    api_key = ""
    if provider_info["needs_key"]:
        while True:
            api_key = typer.prompt("      API key", hide_input=True, default="")
            if api_key:
                break
            if provider_info["key_url"]:
                _echo(f"      Get a key at {provider_info['key_url']}")
            skip = typer.prompt(
                "      Press Enter to try again, or type 'skip' to continue without LLM",
                default="",
            )
            if skip.lower() == "skip":
                _echo("      Skipped LLM configuration.")
                return env_updates

    # Test connection
    _echo("      Testing...")
    ok, msg = _test_llm(config, model, api_key, provider_str)
    if ok:
        _echo(f"      Connected ({model})")
        env_updates["CORTEX_LLM_PROVIDER"] = provider_str
        env_updates["CORTEX_LLM_MODEL"] = model
        if api_key:
            env_updates["CORTEX_LLM_API_KEY"] = api_key
    else:
        _echo(f"      Failed: {msg}")
        while True:
            action = typer.prompt("      [r]etry / [d]ifferent provider / [s]kip", default="r")
            if action.lower() == "s":
                _echo("      Skipped.")
                return env_updates
            elif action.lower() == "d":
                return _step_llm(config, auto)  # restart step
            elif action.lower() == "r":
                api_key = typer.prompt("      API key", hide_input=True, default="")
                if not api_key:
                    continue
                _echo("      Testing...")
                ok, msg = _test_llm(config, model, api_key, provider_str)
                if ok:
                    _echo(f"      Connected ({model})")
                    env_updates["CORTEX_LLM_PROVIDER"] = provider_str
                    env_updates["CORTEX_LLM_MODEL"] = model
                    env_updates["CORTEX_LLM_API_KEY"] = api_key
                    break
                _echo(f"      Failed: {msg}")

    return env_updates


def _step_embeddings(config: CortexConfig, auto: bool) -> None:
    """Step 4: Check/warm embeddings, offer to install if missing."""
    import subprocess

    from cortex.services.embeddings import create_embedding_provider

    _echo("\n[4/7] Embeddings")
    provider = create_embedding_provider(config)

    if provider is None:
        _echo("      sentence-transformers is not installed.")
        _echo("      Semantic search requires it. Keyword search works without it.")

        if auto:
            _echo("      Skipped (auto mode). Install later: pip install sentence-transformers")
            return

        if typer.confirm("\n      Install now? (~400MB download)", default=True):
            _echo("      Installing sentence-transformers...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "sentence-transformers>=3.4"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                _echo("      Installed.")
                provider = create_embedding_provider(config)
            else:
                stderr = result.stderr.strip()
                last_line = stderr.splitlines()[-1] if stderr else "unknown error"
                _echo(f"      Install failed: {last_line}")
                _echo("      Install manually later: pip install sentence-transformers")
                return
        else:
            _echo("      Skipped. Install later: pip install sentence-transformers")
            return

    if provider is not None:
        _echo(f"      Loading {provider.model_name}...")
        if provider.warmup():
            _echo(f"      {provider.model_name} ready")
        else:
            _echo("      Warm-up failed — will retry on first use")


def _step_dashboard_password(store: Store, auto: bool) -> None:
    """Step 5: Set dashboard password."""
    import bcrypt

    _echo("\n[5/7] Dashboard password")

    existing = store.content.get_config("dashboard_password_hash")

    if auto:
        if existing:
            _echo("      Password already set (kept)")
        else:
            _echo("      Open access (no password)")
        return

    if existing and not typer.confirm("      Password already set. Change it?", default=False):
        _echo("      Kept.")
        return

    if not typer.confirm("      Set a dashboard password?", default=False):
        _echo("      Open access (no password)")
        return

    pw = typer.prompt("      Password", hide_input=True)
    pw_hash = bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()
    store.content.set_config("dashboard_password_hash", pw_hash)
    _echo("      Password set")


def _step_services(config: CortexConfig, env_updates: dict[str, str], auto: bool) -> None:
    """Step 6: Write .env, install services, register with Claude Code, fix PATH."""
    from cortex.cli.env_writer import write_env
    from cortex.cli.install import do_install

    _echo("\n[6/7] Background services")

    if not auto:
        install_svc = typer.confirm(
            "      Install MCP + dashboard as background services?", default=True
        )
    else:
        install_svc = True

    # 6a. Write config to .env (always, even if skipping service install —
    # so that manual `cortex serve` also picks up the config)
    if env_updates:
        env_path = config.data_dir / ".env"
        write_env(env_path, env_updates)
        _echo(f"      Config written to {env_path}")

    # 6b. Install services
    if install_svc:
        try:
            do_install(config=config, service="all")
        except Exception as e:
            _echo(f"      Service install failed: {e}")
            _echo("      You can start manually: cortex serve --transport mcp-http")
    else:
        _echo("      Skipped services. Start manually: cortex serve --transport mcp-http")

    # 6c. Register with Claude Code
    register_cc = (
        typer.confirm("      Register with Claude Code?", default=True)
        if not auto
        else True
    )
    if register_cc:
        try:
            settings_path = Path.home() / ".claude" / "settings.json"
            settings: dict[str, Any] = {}
            if settings_path.exists():
                settings = json.loads(settings_path.read_text())
            mcp_servers = settings.setdefault("mcpServers", {})
            mcp_servers["cortex"] = {
                "type": "http",
                "url": config.mcp_server_url,
            }
            settings_path.parent.mkdir(parents=True, exist_ok=True)
            settings_path.write_text(json.dumps(settings, indent=2) + "\n")
            _echo(f"      Registered with Claude Code ({config.mcp_server_url})")
        except Exception as e:
            _echo(f"      Registration failed: {e}")
            _echo("      You can register later: cortex register")

    # 6d. PATH check
    if not shutil.which("cortex"):
        cortex_bin = Path(sys.executable).parent / "cortex"
        if cortex_bin.exists():
            link_path = Path("/usr/local/bin/cortex")
            if not auto:
                add_path = typer.confirm(
                    "      Add `cortex` to your PATH? (symlink in /usr/local/bin)",
                    default=True,
                )
            else:
                add_path = True

            if add_path:
                try:
                    link_path.symlink_to(cortex_bin)
                    _echo(f"      Linked: {link_path} -> {cortex_bin}")
                except PermissionError:
                    _echo(f"      Needs sudo. Run: sudo ln -sf {cortex_bin} {link_path}")
                except FileExistsError:
                    _echo(f"      {link_path} already exists — skipping")
            else:
                _echo(f"      Skipped. Run: sudo ln -sf {cortex_bin} /usr/local/bin/cortex")


def _step_verify(config: CortexConfig) -> None:
    """Step 7: Verify services are running."""
    _echo("\n[7/7] Verification")
    _echo("      Waiting for services to start...")
    time.sleep(2)

    mcp_url = f"http://{config.host}:{config.port}/mcp"
    dash_url = f"http://{config.host}:{config.port + 1}/"

    # MCP server
    mcp_ok = _probe_http(mcp_url)
    if mcp_ok:
        _echo(f"      MCP server ({config.host}:{config.port})     ok")
    else:
        _echo(f"      MCP server ({config.host}:{config.port})     not responding")
        err_file = config.data_dir / "mcp-http.err"
        if err_file.exists():
            lines = err_file.read_text().splitlines()[-5:]
            if lines:
                _echo("      Recent errors:")
                for line in lines:
                    _echo(f"        {line}")

    # Dashboard
    dash_ok = _probe_http(dash_url, retries=3, delay=1.0)
    if dash_ok:
        _echo(f"      Dashboard  ({config.host}:{config.port + 1})     ok")
    else:
        _echo(f"      Dashboard  ({config.host}:{config.port + 1})     not responding")
        err_file = config.data_dir / "dashboard.err"
        if err_file.exists():
            lines = err_file.read_text().splitlines()[-5:]
            if lines:
                _echo("      Recent errors:")
                for line in lines:
                    _echo(f"        {line}")

    # Claude Code registration
    settings_path = Path.home() / ".claude" / "settings.json"
    cc_ok = False
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text())
            cc_ok = "cortex" in settings.get("mcpServers", {})
        except (json.JSONDecodeError, KeyError):
            pass
    if cc_ok:
        _echo("      Claude Code registration       ok")
    else:
        _echo("      Claude Code registration       not found")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_setup_wizard(auto: bool = False) -> None:
    """Run the full Cortex setup wizard."""
    import cortex.cli.main as cli_mod

    config = load_config()
    setup_logging(level=config.log_level, json_output=False)

    _echo("Cortex Setup\n")

    # Step 1: Data directory
    _step_data_dir(config)

    # Step 2: Stores & ontology
    store = _step_stores(config)

    # Step 3: LLM provider
    env_updates = _step_llm(config, auto)

    # Step 4: Embeddings
    _step_embeddings(config, auto)

    # Step 5: Dashboard password
    _step_dashboard_password(store, auto)

    # Step 6: Services, registration, PATH
    _step_services(config, env_updates, auto)

    # Step 7: Verification
    _step_verify(config)

    # Expose the store to the CLI module so other commands (and test
    # fixtures) can reuse or clean up the same instance.
    cli_mod._store = store

    # Done
    _echo("\nCortex is ready.")
    _echo(f"  MCP:       http://{config.host}:{config.port}/mcp")
    _echo(f"  Dashboard: http://{config.host}:{config.port + 1}/")
    _echo(f"  Data:      {config.data_dir}/")
    _echo('\n  Try: cortex capture "My first note" --type idea --content "Hello Cortex!"')
    _echo("\n  Restart Claude Code to activate the MCP connection.")
