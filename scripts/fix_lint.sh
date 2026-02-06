#!/bin/bash
# Auto-fix linting issues

set -e

echo "Installing ruff..."
pip install ruff

echo "Running ruff fix..."
ruff check --fix --unsafe-fixes src/ tests/ examples/

echo "Running ruff format..."
ruff format src/ tests/ examples/

echo "Done! Please review changes and commit."
