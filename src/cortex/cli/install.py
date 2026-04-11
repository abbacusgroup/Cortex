"""Platform-aware service installation for Cortex.

Supports macOS (LaunchAgent) and Linux (systemd user unit).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import typer

from cortex.core.config import CortexConfig

# ---------------------------------------------------------------------------
# Label constants
# ---------------------------------------------------------------------------

_MCP_LABEL = "ai.abbacus.cortex.mcp"
_DASHBOARD_LABEL = "ai.abbacus.cortex.dashboard"

# ---------------------------------------------------------------------------
# macOS LaunchAgent templates
# ---------------------------------------------------------------------------

_MACOS_MCP_PLIST = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" \
"http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>

    <key>ProgramArguments</key>
    <array>
        <string>{cortex_bin}</string>
        <string>serve</string>
        <string>--transport</string>
        <string>mcp-http</string>
        <string>--host</string>
        <string>{host}</string>
        <string>--port</string>
        <string>{port}</string>
    </array>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>{bin_dir}:/usr/local/bin:/usr/bin:/bin</string>
        <key>HOME</key>
        <string>{home}</string>
    </dict>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <true/>

    <key>StandardOutPath</key>
    <string>{data_dir}/mcp-http.log</string>

    <key>StandardErrorPath</key>
    <string>{data_dir}/mcp-http.err</string>

    <key>ProcessType</key>
    <string>Interactive</string>
</dict>
</plist>
"""

_MACOS_DASHBOARD_PLIST = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" \
"http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>

    <key>ProgramArguments</key>
    <array>
        <string>{cortex_bin}</string>
        <string>dashboard</string>
        <string>--host</string>
        <string>{host}</string>
        <string>--port</string>
        <string>{port}</string>
    </array>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>{bin_dir}:/usr/local/bin:/usr/bin:/bin</string>
        <key>HOME</key>
        <string>{home}</string>
    </dict>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <true/>

    <key>StandardOutPath</key>
    <string>{data_dir}/dashboard.log</string>

    <key>StandardErrorPath</key>
    <string>{data_dir}/dashboard.err</string>

    <key>ProcessType</key>
    <string>Interactive</string>
</dict>
</plist>
"""

# ---------------------------------------------------------------------------
# Linux systemd user unit templates
# ---------------------------------------------------------------------------

_LINUX_MCP_UNIT = """\
[Unit]
Description=Cortex MCP HTTP Server
After=network.target

[Service]
Type=simple
ExecStart={cortex_bin} serve --transport mcp-http --host {host} --port {port}
Environment=CORTEX_DATA_DIR={data_dir}
Restart=always
RestartSec=5
StandardOutput=append:{data_dir}/mcp-http.log
StandardError=append:{data_dir}/mcp-http.err

[Install]
WantedBy=default.target
"""

_LINUX_DASHBOARD_UNIT = """\
[Unit]
Description=Cortex Web Dashboard
After=network.target

[Service]
Type=simple
ExecStart={cortex_bin} dashboard --host {host} --port {port}
Environment=CORTEX_DATA_DIR={data_dir}
Restart=always
RestartSec=5
StandardOutput=append:{data_dir}/dashboard.log
StandardError=append:{data_dir}/dashboard.err

[Install]
WantedBy=default.target
"""


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def detect_cortex_binary() -> str:
    """Find the cortex binary path.

    Prefers the binary next to the running Python interpreter (most reliable
    when the user runs ``cortex install``), falls back to ``shutil.which``.
    """
    candidate = Path(sys.executable).parent / "cortex"
    if candidate.exists():
        return str(candidate.resolve())
    found = shutil.which("cortex")
    if found:
        return str(Path(found).resolve())
    msg = (
        "Cannot find the cortex binary. Ensure it is installed "
        "(pip install cortex) and on your PATH."
    )
    raise FileNotFoundError(msg)


def detect_platform() -> str:
    """Return 'macos' or 'linux'."""
    if sys.platform == "darwin":
        return "macos"
    if sys.platform.startswith("linux"):
        return "linux"
    msg = f"Unsupported platform: {sys.platform}. Only macOS and Linux are supported."
    raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------


def _render_vars(config: CortexConfig, binary: str) -> dict[str, str]:
    """Build the template variable dict."""
    bin_dir = str(Path(binary).parent)
    return {
        "cortex_bin": binary,
        "bin_dir": bin_dir,
        "home": str(Path.home()),
        "host": config.host,
        "port": str(config.port),
        "data_dir": str(config.data_dir),
    }


def render_mcp_plist(config: CortexConfig, binary: str) -> str:
    """Render the MCP LaunchAgent plist."""
    vs = _render_vars(config, binary)
    vs["label"] = _MCP_LABEL
    return _MACOS_MCP_PLIST.format(**vs)


def render_dashboard_plist(config: CortexConfig, binary: str) -> str:
    """Render the dashboard LaunchAgent plist."""
    vs = _render_vars(config, binary)
    vs["label"] = _DASHBOARD_LABEL
    vs["port"] = str(config.port + 1)
    return _MACOS_DASHBOARD_PLIST.format(**vs)


def render_mcp_unit(config: CortexConfig, binary: str) -> str:
    """Render the MCP systemd user unit."""
    return _LINUX_MCP_UNIT.format(**_render_vars(config, binary))


def render_dashboard_unit(config: CortexConfig, binary: str) -> str:
    """Render the dashboard systemd user unit."""
    vs = _render_vars(config, binary)
    vs["port"] = str(config.port + 1)
    return _LINUX_DASHBOARD_UNIT.format(**vs)


# ---------------------------------------------------------------------------
# macOS install / uninstall
# ---------------------------------------------------------------------------

_LAUNCH_AGENTS_DIR = Path.home() / "Library" / "LaunchAgents"


def _install_launchagent(label: str, content: str) -> Path:
    """Write a plist and load it via launchctl."""
    _LAUNCH_AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    plist_path = _LAUNCH_AGENTS_DIR / f"{label}.plist"

    if plist_path.exists():
        typer.echo(f"  Updating {plist_path.name} (already exists)")
        subprocess.run(
            ["launchctl", "unload", str(plist_path)],
            check=False,
            capture_output=True,
        )
    else:
        typer.echo(f"  Creating {plist_path.name}")

    plist_path.write_text(content)
    subprocess.run(["launchctl", "load", str(plist_path)], check=True)
    typer.echo(f"  Loaded {label}")
    return plist_path


def _uninstall_launchagent(label: str) -> None:
    """Unload and remove a LaunchAgent plist."""
    plist_path = _LAUNCH_AGENTS_DIR / f"{label}.plist"
    if not plist_path.exists():
        typer.echo(f"  {label}: not installed")
        return
    subprocess.run(
        ["launchctl", "unload", str(plist_path)],
        check=False,
        capture_output=True,
    )
    plist_path.unlink()
    typer.echo(f"  Removed {label}")


# ---------------------------------------------------------------------------
# Linux install / uninstall
# ---------------------------------------------------------------------------

_SYSTEMD_USER_DIR = Path.home() / ".config" / "systemd" / "user"


def _unit_name(label: str) -> str:
    """Convert label to systemd unit file name."""
    return label.replace(".", "-") + ".service"


def _install_systemd_unit(label: str, content: str) -> Path:
    """Write a unit file and enable it."""
    if not shutil.which("systemctl"):
        typer.echo(
            "  Error: systemctl not found — systemd is required for service install on Linux"
        )
        typer.echo("  You can still run Cortex manually: cortex serve --transport mcp-http")
        raise typer.Exit(1)

    _SYSTEMD_USER_DIR.mkdir(parents=True, exist_ok=True)
    unit_path = _SYSTEMD_USER_DIR / _unit_name(label)
    unit_name = unit_path.name

    if unit_path.exists():
        typer.echo(f"  Updating {unit_name} (already exists)")
        subprocess.run(
            ["systemctl", "--user", "stop", unit_name],
            check=False,
            capture_output=True,
        )
    else:
        typer.echo(f"  Creating {unit_name}")

    unit_path.write_text(content)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", unit_name], check=True)
    typer.echo(f"  Enabled {unit_name}")
    return unit_path


def _uninstall_systemd_unit(label: str) -> None:
    """Disable and remove a systemd user unit."""
    unit_path = _SYSTEMD_USER_DIR / _unit_name(label)
    unit_name = unit_path.name
    if not unit_path.exists():
        typer.echo(f"  {unit_name}: not installed")
        return
    if shutil.which("systemctl"):
        subprocess.run(
            ["systemctl", "--user", "disable", "--now", unit_name],
            check=False,
            capture_output=True,
        )
    unit_path.unlink()
    subprocess.run(
        ["systemctl", "--user", "daemon-reload"],
        check=False,
        capture_output=True,
    )
    typer.echo(f"  Removed {unit_name}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def do_install(config: CortexConfig, service: str) -> None:
    """Install Cortex as a system service."""
    binary = detect_cortex_binary()
    platform = detect_platform()
    services = ["mcp", "dashboard"] if service == "all" else [service]

    typer.echo(f"Installing Cortex services ({platform})")
    typer.echo(f"  Binary: {binary}")
    typer.echo(f"  Data:   {config.data_dir}")
    typer.echo()

    for svc in services:
        if platform == "macos":
            if svc == "mcp":
                _install_launchagent(_MCP_LABEL, render_mcp_plist(config, binary))
            else:
                _install_launchagent(_DASHBOARD_LABEL, render_dashboard_plist(config, binary))
        else:
            if svc == "mcp":
                _install_systemd_unit(_MCP_LABEL, render_mcp_unit(config, binary))
            else:
                _install_systemd_unit(_DASHBOARD_LABEL, render_dashboard_unit(config, binary))

    typer.echo()
    if "mcp" in services:
        typer.echo(f"  MCP server: http://{config.host}:{config.port}/mcp")
    if "dashboard" in services:
        typer.echo(f"  Dashboard:  http://{config.host}:{config.port + 1}/")
    typer.echo("\nRun `cortex status` to verify the server is running.")


def do_uninstall(config: CortexConfig, service: str) -> None:
    """Remove Cortex system services."""
    platform = detect_platform()
    services = ["mcp", "dashboard"] if service == "all" else [service]

    typer.echo(f"Uninstalling Cortex services ({platform})")
    for svc in services:
        label = _MCP_LABEL if svc == "mcp" else _DASHBOARD_LABEL
        if platform == "macos":
            _uninstall_launchagent(label)
        else:
            _uninstall_systemd_unit(label)

    typer.echo("\nCortex services removed.")
