# Contributing to Kizuna Privacy Engine

Thank you for your interest in contributing to Kizuna! This document provides guidelines and instructions for contributors.

## Code of Conduct

Be respectful, inclusive, and constructive in all interactions.

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/kizuna-multimodal-privacy-engine.git
cd kizuna-multimodal-privacy-engine
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[all]"

# Install pre-commit hooks
pre-commit install
```

### 3. Verify Setup

```bash
# Run tests
make test

# Run linters
make lint

# Run type checker
make typecheck
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Changes

- Write clear, documented code
- Follow the existing code style
- Add tests for new functionality
- Update documentation as needed

### 3. Run Tests and Linters

```bash
# Format code
make format

# Run linters
make lint

# Run type checker
make typecheck

# Run tests
make test

# Run specific test category
make test-unit
make test-integration
make test-privacy
```

### 4. Commit Changes

```bash
git add .
git commit -m "feat: add new multimodal encoder"
```

**Commit Message Format**:
- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation changes
- `test:` test additions/changes
- `refactor:` code refactoring
- `perf:` performance improvements
- `chore:` maintenance tasks

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

## Code Style

### Python

- **Formatting**: Black (line length 100)
- **Import sorting**: isort
- **Linting**: Ruff
- **Type checking**: mypy (strict mode)

### Docstrings

Use Google-style docstrings:

```python
def function(arg1: str, arg2: int) -> bool:
    """Short description.

    Longer description if needed.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2

    Returns:
        Description of return value

    Raises:
        ValueError: Description of when this is raised
    """
    pass
```

## Testing

### Test Structure

```
tests/
├── unit/              # Unit tests (fast, isolated)
├── integration/       # Integration tests (slower, multi-component)
├── performance/       # Performance benchmarks
└── privacy/           # Privacy guarantee verification
```

### Writing Tests

```python
import pytest

class TestYourFeature:
    """Tests for your feature."""

    def test_basic_functionality(self) -> None:
        """Test basic functionality."""
        result = your_function()
        assert result == expected_value

    @pytest.mark.slow
    def test_slow_operation(self) -> None:
        """Test that takes longer to run."""
        pass

    @pytest.mark.requires_models
    def test_with_onnx_model(self) -> None:
        """Test that requires ONNX models."""
        pass
```

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/unit/test_config.py

# Specific test
pytest tests/unit/test_config.py::TestConfigManager::test_load_default_config

# With markers
pytest -m "not slow"  # Skip slow tests
pytest -m privacy     # Only privacy tests

# With coverage
pytest --cov=src --cov-report=html
```

## Privacy Considerations

When contributing code that handles sensitive data:

1. **Never log raw data** - Use structured logging with PII redaction
2. **Test memory wiping** - Ensure raw data is destroyed after processing
3. **Verify DP guarantees** - Add property-based tests for differential privacy
4. **Document privacy implications** - Update APPI compliance documentation

## Documentation

### Code Documentation

- All public functions, classes, and modules must have docstrings
- Complex algorithms should have inline comments
- Update type hints for all function signatures

### User Documentation

- Update `README.md` if adding user-facing features
- Add examples to `docs/` if introducing new workflows
- Update `KIZUNA_ARCHITECTURE.md` for architectural changes

## Pull Request Guidelines

### Before Submitting

- [ ] All tests pass
- [ ] Code is formatted (black, isort)
- [ ] Linters pass (ruff)
- [ ] Type checker passes (mypy)
- [ ] Documentation is updated
- [ ] Commit messages follow convention
- [ ] Branch is up to date with main

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
How were these changes tested?

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

## Review Process

1. Automated CI checks must pass
2. At least one maintainer review required
3. Address review feedback
4. Maintainer merges PR

## Areas for Contribution

We especially welcome contributions in:

- 🎨 **Frontend**: Improve Streamlit dashboard UX
- 🧠 **ML Models**: Optimize ONNX models for edge devices
- 🔒 **Privacy**: Enhance differential privacy mechanisms
- 📊 **Benchmarks**: Test on real edge hardware (Jetson, RPi)
- 🌐 **Localization**: Add Japanese language support
- 📖 **Documentation**: Improve tutorials and examples

## Getting Help

- **Documentation**: See `docs/` directory
- **Issues**: Search existing issues or create a new one
- **Discussions**: Use GitHub Discussions for questions

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.

---

Thank you for contributing to Kizuna! 🙏
