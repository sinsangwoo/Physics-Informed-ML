# Contributing to Physics-Informed ML Framework

Thank you for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/sinsangwoo/Physics-Informed-ML.git
cd Physics-Informed-ML
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Development Workflow

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make changes and add tests
3. Run quality checks:
   ```bash
   ruff check src tests
   ruff format src tests
   pytest tests/ -v
   ```
4. Commit and push
5. Open a Pull Request

## Code Style

- Follow PEP 8
- Maximum line length: 100 characters
- Use type hints
- Write docstrings for public APIs

## Testing

All new code should include tests:

```python
def test_your_feature():
    result = your_function()
    assert result == expected_value
```

Thank you for contributing!