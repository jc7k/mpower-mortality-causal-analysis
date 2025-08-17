# Task Completion Workflow & Quality Assurance

## Immediate Task Completion Steps (from CLAUDE.md)

### 1. Mark Tasks as Complete
- **Update TASK.md** immediately after finishing tasks
- **Add discovered subtasks** to TASK.md under "Discovered During Work" section
- **Be specific** about what was accomplished

### 2. Code Quality Checks (REQUIRED)
```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type checking (if configured)
mypy src/

# CRITICAL: Must pass before considering task complete
```

### 3. Testing Requirements
- **Run all tests** before task completion:
```bash
pytest tests/
```
- **Update existing tests** if logic changed
- **Create new tests** for new features (minimum requirements):
  - 1 test for expected use
  - 1 edge case test
  - 1 failure case test

### 4. Documentation Updates
- **Update README.md** when:
  - New features added
  - Dependencies change
  - Setup steps modified
- **Add/update docstrings** for new functions (Google style required)
- **Comment non-obvious code** with reasoning

### 5. Environment Validation
- **Use .venv virtual environment** for all Python commands
- **Verify dependencies** are properly installed
- **Test in clean environment** if possible

## Research-Specific Completion Workflow

### Data Processing Tasks
1. **Validate data quality**:
```bash
python src/data/validation.py
```
2. **Check data coverage**: Ensure 150+ countries represented
3. **Verify data ranges**: MPOWER scores 0-37, mortality rates reasonable
4. **Test data integration**: No duplicate country-years

### Analysis Tasks
1. **Model convergence**: All econometric models must converge
2. **Statistical significance**: Effects detectable at p<0.05 level
3. **Robustness checks**: Results consistent across specifications
4. **Event study plots**: Generate and validate timing of effects

### Output Generation
1. **Tables generated**: Publication-ready regression tables
2. **Figures created**: Event studies, trend plots, effect sizes
3. **Reports compiled**: Using WeasyPrint or Quarto
4. **Code reproducible**: Independent replication possible

## Quality Gates Before Task Completion

### Code Quality Gates
- [ ] All linting passes (ruff)
- [ ] All formatting applied (black)
- [ ] All tests pass (pytest)
- [ ] Type hints present and valid
- [ ] Docstrings complete

### Research Quality Gates
- [ ] Data validation passes
- [ ] Models converge successfully
- [ ] Statistical inference robust
- [ ] Results replicate independently
- [ ] Output files generated correctly

### Documentation Quality Gates
- [ ] README.md updated if needed
- [ ] Code comments explain reasoning
- [ ] Functions have proper docstrings
- [ ] TASK.md reflects completion

## Never Consider Complete If:
- **Tests are failing**
- **Linting errors present**
- **Code doesn't follow style guide**
- **Documentation missing or outdated**
- **Virtual environment not used**
- **Dependencies not properly managed**

## Post-Completion Verification
```bash
# Full validation pipeline
black src/ tests/ && \
ruff check src/ tests/ && \
pytest tests/ && \
echo "Task completion validated!"
```

## Context Engineering Integration
- **Validate against PRP success criteria** before marking complete
- **Update PRP documentation** if implementation differs from plan
- **Consider broader project impact** of changes made
