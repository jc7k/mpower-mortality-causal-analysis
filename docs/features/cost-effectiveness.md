# Extension B: Cost-Effectiveness Framework

## Overview

This extension builds a comprehensive economic evaluation framework quantifying return on investment for MPOWER policies across different country contexts.

**Branch**: `feature/cost-effectiveness`
**Module**: `src/mpower_mortality_causal_analysis/extensions/cost_effectiveness/`
**Timeline**: 10 days (2 weeks)

## Scientific Value

- Enables evidence-based resource allocation
- Provides country-specific implementation guidance
- Quantifies health system savings
- Supports donor funding decisions
- **Reader Benefit**: Gain hands-on experience building health economic models that are the gold standard for policy evaluation, skills directly transferable to pharmaceutical pricing, healthcare technology assessment, or any field requiring rigorous cost-benefit analysis

## Technical Architecture

```python
# Module structure
src/mpower_mortality_causal_analysis/extensions/cost_effectiveness/
├── __init__.py
├── health_outcomes.py      # QALY/DALY calculations
├── cost_models.py          # Cost estimation
├── icer_analysis.py        # Cost-effectiveness ratios
├── budget_optimizer.py     # Resource allocation
├── ce_pipeline.py          # Main orchestration
└── ce_reporting.py         # Standardized outputs
```

## Implementation Plan

### Phase 1: Health Economic Modeling (Days 1-3)

```python
# health_outcomes.py
class HealthOutcomeModel:
    """Models health outcomes in economic terms."""

    def __init__(self, mortality_data: pd.DataFrame, life_tables: pd.DataFrame):
        self.mortality = mortality_data
        self.life_tables = life_tables

    def calculate_qalys(self, mortality_reduction: float) -> float:
        """Quality-Adjusted Life Years gained."""
        pass

    def calculate_dalys(self, mortality_reduction: float) -> float:
        """Disability-Adjusted Life Years averted."""
        pass

    def markov_model(self, transition_probs: dict) -> pd.DataFrame:
        """Disease progression modeling."""
        pass
```

**Deliverables:**
- [ ] QALY calculation with age-weighting
- [ ] DALY computation (YLL + YLD)
- [ ] Markov chain for smoking-related diseases
- [ ] Uncertainty distributions
- [ ] Validation against GBD estimates

### Phase 2: Cost Estimation (Days 4-5)

```python
# cost_models.py
class CostEstimator:
    """Estimates implementation and offset costs."""

    def __init__(self, country_data: pd.DataFrame):
        self.country_data = country_data

    def implementation_costs(self, policy: str, country: str) -> dict:
        """Policy implementation costs by country."""
        pass

    def healthcare_savings(self, cases_prevented: int) -> float:
        """Cost offsets from prevention."""
        pass

    def productivity_gains(self, mortality_reduction: float) -> float:
        """Economic value of prevented deaths."""
        pass
```

**Deliverables:**
- [ ] WHO-CHOICE cost database integration
- [ ] Country-specific cost adjustments
- [ ] Healthcare utilization models
- [ ] Productivity loss calculations
- [ ] Time discounting (3% annually)

### Phase 3: ICER Analysis (Days 6-7)

```python
# icer_analysis.py
class ICERAnalysis:
    """Incremental Cost-Effectiveness Ratio analysis."""

    def __init__(self, costs: dict, effects: dict):
        self.costs = costs
        self.effects = effects

    def calculate_icer(self, intervention: str, comparator: str) -> float:
        """ICER = ΔCost / ΔEffect"""
        pass

    def probabilistic_sensitivity(self, n_simulations: int = 1000) -> pd.DataFrame:
        """PSA with parameter uncertainty."""
        pass

    def ceac_curve(self, thresholds: list[float]) -> pd.DataFrame:
        """Cost-Effectiveness Acceptability Curve."""
        pass
```

**Deliverables:**
- [ ] Pairwise ICER calculations
- [ ] Dominance identification
- [ ] Monte Carlo PSA
- [ ] CEAC generation
- [ ] Tornado diagrams for sensitivity

### Phase 4: Budget Optimization (Days 8-9)

```python
# budget_optimizer.py
class BudgetOptimizer:
    """Optimizes resource allocation across policies."""

    def __init__(self, budget: float, policies: list[str]):
        self.budget = budget
        self.policies = policies

    def optimize_allocation(self, constraints: dict) -> dict:
        """Linear programming for budget allocation."""
        pass

    def portfolio_optimization(self) -> dict:
        """Markowitz-style policy portfolio."""
        pass
```

**Deliverables:**
- [ ] Linear programming formulation
- [ ] Constraint handling (capacity, political)
- [ ] Efficient frontier calculation
- [ ] Sensitivity to budget changes
- [ ] Multi-objective optimization

### Phase 5: Reporting & Integration (Day 10)

**Deliverables:**
- [ ] Standardized CE report templates
- [ ] Cost-effectiveness planes
- [ ] League tables by country
- [ ] Policy brief generator
- [ ] Full test coverage

## Dependencies

```yaml
dependencies:
  - lifelines     # Survival analysis
  - scipy.optimize # Optimization
  - pyomo         # Mathematical programming
  - SALib         # Sensitivity analysis
  - plotly        # Interactive visualizations
```

## Testing Strategy

```python
# tests/extensions/test_cost_effectiveness.py
class TestHealthOutcomes:
    def test_qaly_calculation(self):
        """Validate QALY computation."""
        pass

    def test_markov_convergence(self):
        """Test Markov model steady state."""
        pass

class TestICER:
    def test_dominance_detection(self):
        """Identify dominated strategies."""
        pass

    def test_psa_convergence(self):
        """PSA stability with iterations."""
        pass
```

## Success Metrics

- ICERs below country WTP thresholds
- Healthcare savings > implementation costs within 5 years
- Clear policy rankings by cost-effectiveness
- Robust to parameter uncertainty (PSA stable)

## Git Worktree Setup

```bash
# Create cost-effectiveness branch and worktree
git checkout -b feature/cost-effectiveness
git worktree add ../mpower-cost-effect feature/cost-effectiveness

# Work in cost-effectiveness directory
cd ../mpower-cost-effect

# Install dependencies
pip install -e ".[cost-effectiveness]"

# Run cost-effectiveness specific tests
pytest tests/extensions/test_cost_effectiveness.py -v --cov=src/mpower_mortality_causal_analysis/extensions/cost_effectiveness
```

## Performance Benchmarks

- Target: <5 minutes per country analysis
- Memory usage: <2GB for full economic modeling
- Support for batch processing multiple countries

## Documentation Requirements

1. Technical paper: `docs/methods/cost_effectiveness.md`
2. Example notebook: `notebooks/extensions/cost_effectiveness.ipynb`
3. API documentation in module docstrings
4. README in `src/mpower_mortality_causal_analysis/extensions/cost_effectiveness/README.md`

## Integration Guidelines

- **No Core Modifications**: Do not modify `src/mpower_mortality_causal_analysis/analysis.py` or `causal_inference/`
- **Import Pattern**: `from mpower_mortality_causal_analysis.analysis import MPOWERAnalysisPipeline`
- **Self-Contained**: All cost-effectiveness code in `extensions/cost_effectiveness/`
- **Testing**: Minimum 80% coverage for all cost-effectiveness modules

## Health Economics Standards

- Follow CHEERS reporting guidelines for economic evaluations
- Use appropriate discount rates (3% annually for costs and outcomes)
- Include both societal and healthcare system perspectives
- Validate health outcome calculations against published literature
- Provide transparent uncertainty analysis

## Data Sources Integration

- WHO-CHOICE cost database for implementation costs
- Global Burden of Disease (GBD) for health outcome valuations
- World Bank data for country-specific cost adjustments
- Published literature for transition probabilities
- National health accounts for healthcare cost offsets

---

This document provides a focused roadmap for implementing the cost-effectiveness framework extension independently of other research directions.
