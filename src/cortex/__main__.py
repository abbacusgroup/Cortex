"""Allow running Cortex as ``python -m cortex``."""

from cortex.cli.main import app

if __name__ == "__main__":
    app()
