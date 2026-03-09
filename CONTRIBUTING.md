# Contributing to Rolling Gym

Thank you for your interest in contributing to Rolling Gym! This document provides guidelines for contributing.

## Reporting Issues

- Use [GitHub Issues](https://github.com/nima-siboni/rolling-gym/issues) to report bugs or request features
- Include steps to reproduce the problem, expected behavior, and your environment (OS, Python version)

## Development Setup

```bash
# Clone the repo
git clone https://github.com/nima-siboni/rolling-gym.git
cd rolling-gym

# Install with dev dependencies
uv sync --python 3.10 --dev

# Install pre-commit hooks
pre-commit install
```

## Making Changes

1. Fork the repository and create a branch from `main`
2. Make your changes
3. Add or update tests as needed
4. Run the test suite: `pytest`
5. Ensure pre-commit hooks pass
6. Open a pull request with a clear description of the changes

## Code Style

- This project uses `pylint` for linting
- Pre-commit hooks enforce formatting and style automatically
- Keep changes focused — one feature or fix per PR

## Questions

Open an issue for questions about the codebase or how to contribute.
