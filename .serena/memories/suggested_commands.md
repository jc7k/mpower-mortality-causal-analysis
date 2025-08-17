# Suggested Commands & Development Workflow

## Environment Setup Commands
```bash
# Install uv package manager (if not present)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies (when requirements.txt exists)
uv pip install -r requirements.txt

# Create project structure
mkdir -p src/{data,analysis,visualization,utils}
mkdir -p tests output/{tables,figures,reports}
```

## Development Commands
```bash
# Code formatting
black src/ tests/

# Code linting
ruff check src/ tests/

# Type checking (if mypy is configured)
mypy src/

# Run tests
pytest tests/

# Run tests with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/test_specific_module.py
```

## Data Processing Commands
```bash
# Python execution (use virtual environment)
python src/data/cleaning.py
python src/analysis/did_analysis.py

# Jupyter notebook
jupyter notebook

# Interactive Python
python -i
```

## Git Commands
```bash
# Check status
git status

# Check current branch
git branch --show-current

# View recent commits
git log --oneline -10

# Stage changes
git add .

# Commit changes
git commit -m "Descriptive commit message"

# Push changes
git push origin main
```

## Claude Code Specific Commands
```bash
# Generate PRP from initial specification
/generate-prp INITIAL.md

# Execute PRP implementation
/execute-prp PRPs/your-feature-name.md
```

## Research Analysis Commands (when implemented)
```bash
# Data cleaning pipeline
python src/data/cleaning.py --input data/raw --output data/processed

# Run main analysis
python src/analysis/did_analysis.py

# Generate reports
python src/visualization/generate_report.py

# Run robustness checks
python src/analysis/robustness_checks.py
```

## System Utilities (Linux)
```bash
# List files
ls -la

# Find files
find . -name "*.py" -type f

# Search content
grep -r "search_term" src/

# Directory navigation
cd project_directory

# File operations
cp source destination
mv old_name new_name
mkdir directory_name
```

## Project Validation Commands
```bash
# Validate data quality
python src/data/validation.py

# Check project structure
tree src/ -I "__pycache__"

# Lint all Python files
find . -name "*.py" -exec ruff check {} \;

# Run all tests and generate report
pytest --html=output/test_report.html
```

## Package Installation Commands
```bash
# Add new dependency with uv
uv add package_name

# Add development dependency
uv add --dev pytest black ruff

# Update all dependencies
uv pip install --upgrade -r requirements.txt

# Generate requirements file
uv pip freeze > requirements.txt
```

## Note on Virtual Environment
**Important**: Always use `.venv` virtual environment for Python commands, especially for unit tests, as specified in CLAUDE.md.
