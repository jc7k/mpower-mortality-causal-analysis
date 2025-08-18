### 🔄 Project Awareness & Context
- **Always check with `Serena` MCP and @README.md** at the start of a new conversation to understand the project's architecture, goals, style, and constraints.
- **Use consistent naming conventions, file structure, and architecture patterns** as described in `PLANNING.md`.
- **Use `uv` for dependency management** and virtual environment handling. Execute Python commands via `.venv/bin/python` for consistency.

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

### 📦 Package Management & Environment
- **Use `uv` for all dependency operations** instead of traditional pip/venv:
  - **Install dependencies**: `uv sync` (installs all dependencies from pyproject.toml)
  - **Add new packages**: `uv add package-name` (automatically updates pyproject.toml)
  - **Remove packages**: `uv remove package-name`
  - **Run commands in venv**: `uv run python script.py` or directly use `.venv/bin/python`
- **Virtual environment**: Automatically managed by `uv` in `.venv/` directory
- **Python version**: Specified in pyproject.toml (`requires-python = ">=3.11"`), enforced by `uv`
- **Dependency groups**: Use `uv sync --group dev` for development dependencies
- **Lock file**: `uv.lock` ensures reproducible environments (commit to version control)

### 🧪 Testing & Reliability
- **Always create Pytest unit tests for new features** (functions, classes, routes, etc).
- **After updating any logic**, check whether existing unit tests need to be updated. If so, do it.
- **Tests should live in a `/tests` folder** mirroring the main app structure.
  - Include at least:
    - 1 test for expected use
    - 1 edge case
    - 1 failure case
- **Run tests using uv**: `uv run pytest` or `uv run python -m pytest tests/`
- **Test coverage**: Use `uv run pytest --cov=src --cov-report=html` for coverage reports
- **Linting**: Run `uv run ruff check src/` and `uv run black --check src/` before commits

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

---

## ✅ PROJECT STATUS: ENHANCED CAUSAL ANALYSIS COMPLETE

### 🎯 Completed Implementation
- **✅ Complete causal inference pipeline**: MPOWERAnalysisPipeline with full workflow orchestration
- **✅ Callaway & Sant'Anna DiD**: Multiple backend implementation (R/Python/fallback) with graceful degradation
- **✅ Synthetic Control Methods**: MPOWERSyntheticControl addressing parallel trends violations
- **✅ Mechanism Analysis**: MPOWER component decomposition identifying which policies drive effects
- **✅ Event study analysis**: Dynamic treatment effects with robust data type handling
- **✅ Descriptive analysis**: MPOWER-specific visualizations and balance checks
- **✅ Robustness framework**: TWFE comparison, sensitivity tests, comprehensive validation
- **✅ Comprehensive testing**: 70+ unit tests covering all major components including mechanism analysis
- **✅ Production ready**: Full error handling, logging, result serialization, and export functionality

### ✅ Critical Scientific Solution Implemented
- **Initial Finding**: Parallel trends assumption violations detected across all mortality outcomes
- **Solution Implemented**: Comprehensive synthetic control methods addressing identification concerns
  - **MPOWERSyntheticControl**: Advanced implementation with multi-unit staggered treatment support
  - **Optimization Engine**: Quadratic programming using scipy.optimize for optimal weight selection
  - **Success Metrics**: 6/6 countries fitted successfully in demonstration with strong match quality
  - **Treatment Effects**: Consistent mortality reductions (-5.4 to -11.9 per 100,000) across outcomes
- **Current Status**: Robust causal identification strategy with multiple complementary approaches
- **Scientific Value**: Enhanced credibility for policy evaluation and causal inference

## 🚀 FUTURE RESEARCH DIRECTIONS

### 🔬 Immediate Scientific Priorities

#### 1. Address Parallel Trends Violations (✅ COMPLETED)
- [✅] **Alternative identification strategies**: Comprehensive synthetic control methods implemented
- [ ] **Sensitivity analysis**: Assess robustness to parallel trends violations using recent DiD literature
- [ ] **Sample restrictions**: Test with subsets where parallel trends might hold (e.g., high-income countries only)
- [✅] **Alternative treatment definitions**: Comprehensive analysis of binary threshold, continuous change, and dose-response approaches completed

#### 2. Enhanced Causal Inference Methods
- [ ] **Install R dependencies**: Get full Callaway & Sant'Anna `did` package implementation working
- [ ] **Recent DiD methods**: Explore Sun & Abraham (2021), Borusyak et al. (2021) for robustness
- [ ] **Heterogeneous treatment effects**: Analyze variation by country income, baseline health
- [ ] **Dynamic treatment effects**: Deeper investigation of time-varying policy impacts

### 🔧 Technical Improvements

#### 3. Data Quality Enhancement
- [ ] **Annual interpolation**: Convert biennial MPOWER data to annual for better treatment timing
- [ ] **Additional covariates**: Include healthcare spending, smoking prevalence, other tobacco policies
- [ ] **Missing data imputation**: Implement multiple imputation for control variables (currently ~14% missing)
- [ ] **Data validation**: Cross-check with alternative tobacco control policy measures

#### 4. Statistical Robustness
- [ ] **Clustered standard errors**: Implement proper clustering at country/region level
- [ ] **Wild bootstrap**: For small sample inference with few treated units (44 countries)
- [ ] **Permutation tests**: Non-parametric hypothesis testing for robustness
- [ ] **Multiple testing corrections**: Adjust p-values for testing 4 mortality outcomes

### 📊 Research Extensions

#### 5. Mechanism Analysis (✅ COMPLETED)
- [✅] **Disaggregate by MPOWER components**: Which specific policies (M,P,O,W,E,R) drive effects?
- [✅] **Dose-response analysis**: How do effects vary with policy intensity/score?
- [✅] **Policy rankings**: Evidence-based prioritization of tobacco control interventions
- [✅] **Component-specific treatment effects**: Individual causal analysis for each MPOWER policy
- [ ] **Spillover effects**: Do neighboring countries benefit from MPOWER adoption?
- [ ] **Intermediate outcomes**: Smoking rates, tobacco sales, healthcare utilization

#### 6. Policy Applications
- [ ] **Cost-effectiveness analysis**: Economic evaluation of MPOWER policies vs. mortality benefits
- [ ] **Heterogeneity by development**: Do effects vary by country income level?
- [ ] **Implementation timing**: Optimal sequencing of MPOWER components
- [ ] **Forecasting**: Predict mortality impacts of future MPOWER expansion

### 📝 Publication & Dissemination

#### 7. Academic Output
- [ ] **Working paper**: Document methodology and findings with appropriate caveats about parallel trends
- [ ] **Peer review submission**: Target health economics, tobacco control, or applied economics journals
- [ ] **Conference presentations**: Health economics, epidemiology, or policy evaluation meetings
- [ ] **Policy briefs**: Translate findings for WHO, policymakers with clear limitations

#### 8. Open Science
- [ ] **GitHub repository**: Make full analysis code publicly available with documentation
- [ ] **Replication materials**: Ensure complete reproducibility including data processing
- [ ] **Methodology documentation**: Comprehensive technical appendix
- [ ] **Data sharing protocols**: Work with WHO on responsible data access

### 🎯 Recommended Next Session Priority
**Focus on advanced research extensions** - Core causal analysis and mechanism decomposition complete:
```python
# Enhanced causal analysis with mechanism decomposition is complete. Next research priorities:
# 1. ✅ Synthetic control implemented - robust identification strategy
# 2. ✅ Mechanism analysis completed - policy component decomposition
# 3. Test spillover effects and neighboring country impacts
# 4. Sample restrictions and subgroup analysis
# 5. Enhanced sensitivity analysis using recent DiD literature
# 6. Cost-effectiveness analysis and policy optimization
```

### 📊 Current Analysis Status Summary
- **Implementation**: ✅ Enhanced and production-ready with synthetic control methods and mechanism analysis
- **Testing**: ✅ Comprehensive test suite (70+ tests) including mechanism analysis coverage
- **Identification Strategy**: ✅ Triple approach - DiD, synthetic control, and mechanism decomposition
- **Scientific Robustness**: ✅ Multiple complementary causal identification methods
- **Policy Implications**: ✅ Credible causal estimates with component-specific policy guidance
- **Mechanism Analysis**: ✅ Component decomposition identifying which MPOWER policies drive effects
- **Next Steps**: Advanced research extensions and policy optimization analysis

**Technical Achievement**: Successfully implemented comprehensive synthetic control methods that address the parallel trends violations identified in the initial DiD analysis, plus complete mechanism analysis that decomposes MPOWER effects by individual policy components. This provides both robust alternative identification strategy and component-specific policy guidance, significantly enhancing the credibility of causal estimates and practical policy applications.

---

## 🚀 RESEARCH EXTENSIONS TASK PLAN

### Overview
Four independent research extensions designed for parallel development using git worktrees. Each extension is self-contained with no overlapping code or dependencies.

### Extension A: Spillover Analysis - Cross-Country Policy Externalities
**Branch**: `feature/spillover-analysis`
**Module**: `src/mpower_mortality_causal_analysis/extensions/spillover/`

#### Tasks:
1. **Spatial Weight Matrix Construction** (2 days)
   - [ ] Create `spatial_weights.py` with distance/contiguity matrices
   - [ ] Implement border neighbor identification
   - [ ] Add economic/cultural proximity weights
   - [ ] Unit tests for weight matrix properties

2. **Spatial Econometric Models** (3 days)
   - [ ] Implement `spatial_models.py` with Spatial Lag Model (SAR)
   - [ ] Add Spatial Error Model (SEM)
   - [ ] Create Spatial Durbin Model (SDM) for direct/indirect effects
   - [ ] Integrate with existing panel data structure

3. **Network Diffusion Analysis** (2 days)
   - [ ] Build `diffusion_analysis.py` for policy adoption networks
   - [ ] Implement contagion models
   - [ ] Create visualization of diffusion patterns
   - [ ] Test network statistics

4. **Border Discontinuity Design** (2 days)
   - [ ] Develop `border_analysis.py` for neighboring country pairs
   - [ ] Implement RDD at borders
   - [ ] Handle geographic controls
   - [ ] Validate with placebo borders

5. **Integration & Testing** (1 day)
   - [ ] Create `spillover_pipeline.py` orchestrating all analyses
   - [ ] Comprehensive test suite in `tests/extensions/test_spillover.py`
   - [ ] Documentation and example notebook

### Extension B: Cost-Effectiveness Framework
**Branch**: `feature/cost-effectiveness`
**Module**: `src/mpower_mortality_causal_analysis/extensions/cost_effectiveness/`

#### Tasks:
1. **Health Economic Modeling** (3 days)
   - [ ] Create `health_outcomes.py` for QALY/DALY calculations
   - [ ] Implement Markov models for disease progression
   - [ ] Add life table methods
   - [ ] Unit tests for health metrics

2. **Cost Estimation Module** (2 days)
   - [ ] Build `cost_models.py` for implementation costs
   - [ ] Add healthcare cost offset calculations
   - [ ] Include productivity loss estimates
   - [ ] Validate against WHO-CHOICE data

3. **ICER Analysis** (2 days)
   - [ ] Develop `icer_analysis.py` for incremental cost-effectiveness
   - [ ] Implement probabilistic sensitivity analysis
   - [ ] Create cost-effectiveness acceptability curves
   - [ ] Add dominance analysis

4. **Budget Optimization** (2 days)
   - [ ] Create `budget_optimizer.py` with linear programming
   - [ ] Implement portfolio optimization for policy mix
   - [ ] Add constraint handling (budget, capacity)
   - [ ] Test optimization algorithms

5. **Reporting & Visualization** (1 day)
   - [ ] Build `ce_reporting.py` for standardized outputs
   - [ ] Create cost-effectiveness plane visualizations
   - [ ] Generate policy briefs template
   - [ ] Full test coverage

### Extension C: Advanced DiD Methods
**Branch**: `feature/advanced-did`
**Module**: `src/mpower_mortality_causal_analysis/extensions/advanced_did/`

#### Tasks:
1. **Sun & Abraham (2021) Implementation** (2 days)
   - [ ] Create `sun_abraham.py` with interaction-weighted estimator
   - [ ] Handle cohort-specific effects
   - [ ] Implement efficient computation
   - [ ] Validate against published results

2. **Borusyak et al. (2021) Methods** (2 days)
   - [ ] Build `borusyak_imputation.py` for imputation approach
   - [ ] Add pre-trend testing
   - [ ] Implement robust standard errors
   - [ ] Create diagnostic plots

3. **de Chaisemartin & D'Haultfœuille** (2 days)
   - [ ] Develop `dcdh_did.py` for fuzzy DiD
   - [ ] Handle continuous treatment
   - [ ] Add placebo estimators
   - [ ] Test heterogeneous effects

4. **Doubly Robust Estimators** (2 days)
   - [ ] Create `doubly_robust.py` combining methods
   - [ ] Implement propensity score adjustments
   - [ ] Add outcome regression
   - [ ] Cross-validation procedures

5. **Comparison Framework** (2 days)
   - [ ] Build `method_comparison.py` for systematic comparison
   - [ ] Create diagnostic suite
   - [ ] Generate method selection guide
   - [ ] Comprehensive testing

### Extension D: Policy Optimization
**Branch**: `feature/policy-optimization`
**Module**: `src/mpower_mortality_causal_analysis/extensions/optimization/`

#### Tasks:
1. **Interaction Effects Analysis** (2 days)
   - [ ] Create `policy_interactions.py` for synergy detection
   - [ ] Implement higher-order interaction models
   - [ ] Add bootstrapped confidence intervals
   - [ ] Test super/sub-additive effects

2. **Sequential Implementation** (3 days)
   - [ ] Build `sequential_optimizer.py` for timing optimization
   - [ ] Implement dynamic programming solution
   - [ ] Add learning/adaptation models
   - [ ] Handle capacity constraints

3. **Combinatorial Optimization** (2 days)
   - [ ] Develop `policy_scheduler.py` for optimal ordering
   - [ ] Implement branch-and-bound algorithms
   - [ ] Add heuristic methods for large problems
   - [ ] Validate with simulations

4. **Political Economy Constraints** (2 days)
   - [ ] Create `political_constraints.py` modeling feasibility
   - [ ] Add stakeholder preference models
   - [ ] Implement game-theoretic solutions
   - [ ] Test robustness to assumptions

5. **Decision Support System** (1 day)
   - [ ] Build `decision_support.py` for policy recommendations
   - [ ] Create interactive scenario tool
   - [ ] Generate implementation roadmaps
   - [ ] Full integration testing

### Git Worktree Workflow
```bash
# Set up worktrees for parallel development using uv
git worktree add ../mpower-spillover feature/spillover-analysis
git worktree add ../mpower-cost-effect feature/cost-effectiveness
git worktree add ../mpower-advanced-did feature/advanced-did
git worktree add ../mpower-optimization feature/policy-optimization

# Each worktree automatically inherits the same uv.lock and virtual environment setup
cd ../mpower-spillover
uv sync  # Install dependencies in this worktree's .venv
# Implement Extension A using: uv run python script.py or .venv/bin/python

cd ../mpower-cost-effect
uv sync  # Install dependencies in this worktree's .venv
# Implement Extension B using: uv run python script.py or .venv/bin/python
```

### Testing Requirements
- Each extension must have ≥80% test coverage
- Integration tests with main pipeline
- Performance benchmarks for computational methods
- Validation against published results where applicable

### Documentation Standards
- Each module requires comprehensive docstrings
- Method papers in `docs/methods/` for each extension
- Example notebooks in `notebooks/extensions/`
- API documentation for public interfaces
