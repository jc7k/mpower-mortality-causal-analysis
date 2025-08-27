# Repository Guidelines

## Project Structure & Module Organization
- Code: `src/mpower_mortality_causal_analysis/` (core pipeline in `analysis.py`; methods in `causal_inference/methods/`; utilities in `causal_inference/utils/`; data prep in `causal_inference/data/`).
- Tests: `tests/` mirrors package layout (e.g., `tests/causal_inference/methods/`).
- Data: `data/` (keep large/raw data out of Git if possible).
- Results: `results/` and `htmlcov/` are generated artifacts.
- Extras: `scripts/`, `examples/`, `docs/`.

## Architecture Overview
- `MPOWERAnalysisPipeline` (in `analysis.py`): Orchestrates full workflow (descriptive → DiD → event study → mechanism → synthetic control → export).
- `causal_inference/methods/callaway_did.py`: `CallawayDiD` with multiple backends and a pure-Python fallback.
- `causal_inference/methods/synthetic_control.py`: `MPOWERSyntheticControl` for multi-unit, staggered adoption.
- `causal_inference/utils/`: Descriptives, event study, robustness, mechanism analysis helpers.
- `causal_inference/data/preparation.py`: `MPOWERDataPrep` for cohort creation, panel balancing, and feature engineering.

## Build, Test, and Development Commands
- Environment (choose one):
  - `uv sync -g dev`  # uses `pyproject.toml` dependency groups and `uv.lock`
  - `pip install -e .[dev]`  # editable install with dev extras
- Pre-commit: `pre-commit install` then run `pre-commit run --all-files`.
- Tests: `pytest -v` or `pytest --maxfail=1 -q`.
- Coverage: `pytest --cov=src --cov-report=term-missing --cov-report=html` (outputs to `htmlcov/`).

## Coding Style & Naming Conventions
- Python 3.11, 4-space indentation, target line length 88.
- Quotes: prefer double quotes (configured via `ruff` formatter).
- Imports and linting: `ruff format` and `ruff check` (configured in `pyproject.toml`).
- Naming: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_CASE`.
- Layout: place new modules under `src/mpower_mortality_causal_analysis/<area>/...` aligned with existing subpackages.
- Type hints encouraged; docstrings optional but helpful for public APIs.

## Testing Guidelines
- Framework: `pytest` with config in `pyproject.toml` (`tests` as `testpaths`).
- File names: `tests/test_*.py` or `tests/**/*_test.py`.
- Structure: mirror package paths (e.g., tests for `.../methods/` go in `tests/causal_inference/methods/`).
- Add focused unit tests for new code; prefer deterministic fixtures.
- Aim to maintain or improve coverage of changed lines.

## Commit & Pull Request Guidelines
- Commit style: Conventional Commits (e.g., `feat:`, `fix:`, `docs:`, `style:`, `refactor:`, `test:`, `chore:`). Keep subjects imperative and concise (≈50 chars), add scope when helpful (e.g., `feat(methods): add Sun-Abraham`).
- Before opening a PR: ensure `pre-commit` passes, tests are green, and coverage isn’t reduced.
- PR description: problem/solution summary, linked issues, verification steps (commands), and relevant outputs (e.g., paths under `results/` or screenshots of plots).

## Security & Configuration
- Do not commit secrets or credentials; prefer environment variables (`python-dotenv` supported).
- Avoid committing large/raw datasets; document reproducible data steps in `docs/` or `scripts/`.
