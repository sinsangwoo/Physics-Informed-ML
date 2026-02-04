# Contributing to Physics-Informed ML

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/Physics-Informed-ML.git`
3. Create a virtual environment: `python -m venv venv`
4. Activate it: `source venv/bin/activate` (Unix) or `venv\Scripts\activate` (Windows)
5. Install development dependencies: `pip install -e ".[dev]"`
6. Install pre-commit hooks: `pre-commit install`

## Development Workflow

1. Create a new branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Run tests: `pytest`
4. Run linting: `ruff check src tests`
5. Run type checking: `mypy src`
6. Commit your changes: `git commit -m "feat: add amazing feature"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Open a Pull Request

## Code Style

We use:
- **Ruff** for linting and formatting
- **mypy** for type checking
- **pytest** for testing

All code must pass these checks before merging.

## Commit Message Convention

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation changes
- `style:` code style changes (formatting, etc.)
- `refactor:` code refactoring
- `test:` adding or updating tests
- `chore:` maintenance tasks

## Testing

- Write tests for all new features
- Maintain test coverage above 80%
- Run `pytest --cov` to check coverage

## Documentation

- Add docstrings to all public functions and classes
- Follow Google style docstrings
- Update README.md if needed

## Pull Request Process

1. Ensure all tests pass
2. Update documentation
3. Add a clear description of changes
4. Link related issues
5. Request review from maintainers

## Code Review

All submissions require review. We use GitHub pull requests for this purpose.

## Questions?

Feel free to open an issue for discussion!