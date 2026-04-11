# Contributing to Cortex

Thanks for your interest in contributing to Cortex! This guide covers everything you need to get started.

## Reporting Bugs

Open a [bug report](https://github.com/abbacusgroup/Cortex/issues/new?template=bug_report.yml) with:

- Steps to reproduce
- Expected vs actual behavior
- Cortex version (`cortex --help` shows the version)
- Python version and OS

## Suggesting Features

Open a [feature request](https://github.com/abbacusgroup/Cortex/issues/new?template=feature_request.yml) describing:

- The problem you're trying to solve
- Your proposed solution
- Alternatives you've considered

## Development Setup

```bash
# Clone the repo
git clone https://github.com/abbacusgroup/Cortex.git
cd Cortex

# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies (including dev tools)
uv sync

# Verify everything works
uv run pytest tests/ -q
```

## Running Tests

```bash
# Full suite
uv run pytest tests/ -q

# Specific test file
uv run pytest tests/db/test_graph_store.py -q

# With verbose output
uv run pytest tests/ -v --tb=short
```

## Code Style

Cortex uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.

- **Line length:** 100 characters
- **Rules:** B, C4, E, F, I, N, RUF, SIM, W, UP

```bash
# Check lint
uv run ruff check src/ tests/

# Auto-format
uv run ruff format src/ tests/
```

## Pull Request Process

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Run `uv run pytest tests/ -q` — all tests must pass
4. Run `uv run ruff check src/ tests/` — no lint errors
5. Run `uv run ruff format src/ tests/` — all files formatted
6. Write a clear PR description explaining what and why
7. Submit the PR against `main`

## Commit Messages

Use concise, descriptive commit messages:

- `fix: resolve UnboundLocalError in reason.py when max_iterations=0`
- `feat: add cortex_list_entities admin tool`
- `docs: update llms.txt with missing tools`

## Architecture Overview

See [CODEMAP.md](CODEMAP.md) for the project structure and data flow.

## Questions?

Open a [discussion](https://github.com/abbacusgroup/Cortex/discussions) or file an issue.
