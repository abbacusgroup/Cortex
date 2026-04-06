"""Entry point for `python -m cortex.transport.mcp`."""

from cortex.transport.mcp.server import run_stdio

if __name__ == "__main__":
    run_stdio()
