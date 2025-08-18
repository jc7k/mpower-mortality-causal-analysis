### 🔄 Project Awareness & Context
- **Always check with `Serena` MCP and @README.md** at the start of a new conversation to understand the project's architecture, goals, style, and constraints.
- **Use consistent naming conventions, file structure, and architecture patterns** as described in `PLANNING.md`.
- **Use .venv** (the virtual environment) whenever executing Python commands, including for unit tests.

### 🧱 Code Structure & Modularity
- **Never create a file longer than 500 lines of code.** If a file approaches this limit, refactor by splitting it into modules or helper files.
- **Organize code into clearly separated modules**, grouped by feature or responsibility.
  For agents this looks like:
    - `agent.py` - Main agent definition and execution logic
    - `tools.py` - Tool functions used by the agent
    - `prompts.py` - System prompts
- **Use clear, consistent imports** (prefer relative imports within packages).
- **Use clear, consistent imports** (prefer relative imports within packages).
- **Use python_dotenv and load_env()** for environment variables.

### 🧪 Testing & Reliability
- **Always create Pytest unit tests for new features** (functions, classes, routes, etc).
- **After updating any logic**, check whether existing unit tests need to be updated. If so, do it.
- **Tests should live in a `/tests` folder** mirroring the main app structure.
  - Include at least:
    - 1 test for expected use
    - 1 edge case
    - 1 failure case

### ✅ Task Completion
- **Mark completed tasks in `TASK.md`** immediately after finishing them.
- Add new sub-tasks or TODOs discovered during development to `TASK.md` under a “Discovered During Work” section.

### 📎 Style & Conventions
- **Use Python** as the primary language.
- **Follow PEP8**, use type hints, and format with `black`.
- **Use `pydantic` for data validation**.
- Use `FastAPI` for APIs and `SQLAlchemy` or `SQLModel` for ORM if applicable.
- Write **docstrings for every function** using the Google style:
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

### 🎯 Code Quality & Linting Rules
- **Keep lines under 88 characters** to pass ruff line length checks.
- **Avoid magic numbers** - use named constants for numeric thresholds (e.g., `SIGNIFICANCE_LEVEL = 0.05`).
- **Function arguments**: Keep to 5 or fewer parameters. Use dataclasses or config objects for complex parameters.
- **Import organization**:
  - Place all imports at the top of files (avoid mid-function imports except for optional dependencies)
  - Remove unused imports immediately
  - Use `importlib.util.find_spec()` to test for optional package availability instead of importing unused packages
- **Exception handling**:
  - Use specific exception types instead of generic `Exception`
  - Keep exception messages concise (under 80 chars) or use exception classes with predefined messages
  - Avoid immediately re-raising exceptions (`except Exception: raise`) - either handle or let propagate
- **Type hints**:
  - Always provide type hints for function parameters and return values
  - Use `Union` types appropriately (e.g., `str | None` instead of just `str`)
  - Use proper generic types for collections (e.g., `list[str]` not `list`)
- **Variable naming**:
  - Use descriptive names for loop variables (avoid single letters except for short, obvious loops)
  - Rename unused variables with underscore prefix (e.g., `_unused_var`)
- **Code reachability**:
  - Ensure all code paths are reachable (no unreachable statements after returns)
  - Use early returns to avoid deep nesting instead of unreachable else clauses

### 🔬 Scientific Computing Standards
- **Statistical significance**: Use named constants for significance levels (e.g., `ALPHA = 0.05` instead of magic number `0.05`)
- **Numeric thresholds**: Define meaningful constants for year cutoffs, sample size limits, etc.
- **Method interfaces**: Ensure consistent parameter names and types across related classes (e.g., `unit_col` vs `country_col`)
- **Fallback implementations**: Always provide graceful degradation when external packages (R, specialized libraries) are unavailable
- **Reproducibility**: Include random seeds and version information where applicable
- **Error handling for statistical methods**: Catch specific exceptions (e.g., `np.linalg.LinAlgError`) rather than generic `Exception`
- **Data validation**: Validate input data structure, required columns, and statistical assumptions before analysis

### 📚 Documentation & Explainability
- **Update `README.md`** when new features are added, dependencies change, or setup steps are modified.
- **Comment non-obvious code** and ensure everything is understandable to a mid-level developer.
- When writing complex logic, **add an inline `# Reason:` comment** explaining the why, not just the what.

### 🧠 AI Behavior Rules
- **Never assume missing context. Ask questions if uncertain.**
- **Never hallucinate libraries or functions** – only use known, verified Python packages.
- **Always confirm file paths and module names** exist before referencing them in code or tests.
- **Never delete or overwrite existing code** unless explicitly instructed to or if part of a task from `TASK.md`.
