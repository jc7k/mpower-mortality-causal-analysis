# Code Style & Conventions

## Language Standards (from CLAUDE.md)
- **Primary Language**: Python
- **Code Style**: Follow PEP8
- **Formatting**: Use `black` for code formatting
- **Linting**: Use `ruff` for fast linting
- **Type Hints**: Required - use type hints throughout

## Documentation Standards
### Docstrings
- **Required for every function** using Google style:
```python
def example():
    """
    Brief summary.

    Args:
        param1 (type): Description.

    Returns:
        type: Description.
    """
```

### Comments
- **Comment non-obvious code** - ensure understandable to mid-level developer
- **Inline reasoning comments** - add `# Reason:` explaining why, not just what
- **No assumption comments** - never assume missing context

## Data Validation Standards
- **Use `pydantic` for data validation**
- **Environment variables** - use `python_dotenv` and `load_env()`

## Framework Preferences
- **APIs**: FastAPI
- **ORM**: SQLAlchemy or SQLModel
- **Testing**: Pytest

## Import Conventions
- **Prefer relative imports** within packages
- **Clear, consistent imports**
- **Never hallucinate libraries** - only use known, verified Python packages

## File and Module Naming
- **Confirm paths exist** before referencing in code or tests
- **Module organization** by feature/responsibility
- **Follow package structure** with proper `__init__.py` files

## Error Handling
- **Never delete/overwrite existing code** unless explicitly instructed
- **Always ask questions** if uncertain about context
- **Validate all assumptions** before implementation

## AI Assistant Specific Rules
- **Context awareness** - always check with Serena MCP and README.md at conversation start
- **Consistent patterns** - use architecture patterns from PLANNING.md (if exists)
- **Virtual environment** - use `.venv` for Python commands and tests
- **Task tracking** - mark completed tasks in TASK.md immediately after finishing

## Quality Standards
- **Testing required** - create Pytest unit tests for new features
- **Test organization** - tests in `/tests` folder mirroring main app structure
- **Minimum test coverage**:
  - 1 test for expected use
  - 1 edge case
  - 1 failure case

## Development Workflow
- **Update README.md** when features added, dependencies change, or setup modified
- **Check existing tests** - update when logic changes
- **Add discovered tasks** to TASK.md under "Discovered During Work" section
