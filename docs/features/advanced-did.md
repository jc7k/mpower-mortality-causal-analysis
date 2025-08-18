# Extension C: Advanced DiD Methods

## Overview

This extension implements state-of-the-art difference-in-differences methods addressing limitations of traditional approaches, ensuring robust causal inference.

**Branch**: `feature/advanced-did`
**Module**: `src/mpower_mortality_causal_analysis/extensions/advanced_did/`
**Timeline**: 10 days (2 weeks)

## Scientific Value

- Addresses negative weighting problems
- Handles heterogeneous treatment effects
- Improves inference with few treated units
- Provides methodological robustness
- **Reader Benefit**: Become proficient in state-of-the-art causal inference methods that are revolutionizing empirical economics, positioning yourself at the forefront of rigorous policy evaluation and potentially improving your publication prospects in top journals

## Technical Architecture

```python
# Module structure
src/mpower_mortality_causal_analysis/extensions/advanced_did/
├── __init__.py
├── sun_abraham.py          # Sun & Abraham (2021)
├── borusyak_imputation.py  # Borusyak et al. (2021)
├── dcdh_did.py             # de Chaisemartin & D'Haultfœuille
├── doubly_robust.py        # DR estimators
├── method_comparison.py    # Systematic comparison
└── diagnostics.py          # Common diagnostics
```

## Implementation Plan

### Phase 1: Sun & Abraham (2021) (Days 1-2)

```python
# sun_abraham.py
class SunAbrahamEstimator:
    """Interaction-weighted estimator for staggered DiD."""

    def __init__(self, data: pd.DataFrame, cohort_col: str, time_col: str):
        self.data = data
        self.cohort_col = cohort_col
        self.time_col = time_col

    def estimate(self, outcome: str, covariates: list[str] = None) -> dict:
        """Estimates cohort-specific and aggregated effects."""
        pass

    def event_study(self, horizon: int = 5) -> pd.DataFrame:
        """Dynamic effects with proper weighting."""
        pass
```

**Deliverables:**
- [ ] Cohort-specific ATT estimation
- [ ] Interaction-weighted aggregation
- [ ] Never-treated and last-treated controls
- [ ] Efficient computation for large panels
- [ ] Validation against Stata implementation

### Phase 2: Borusyak et al. (2021) (Days 3-4)

```python
# borusyak_imputation.py
class BorusyakImputation:
    """Imputation-based estimator for staggered adoption."""

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def impute_counterfactuals(self, outcome: str) -> pd.DataFrame:
        """Imputes Y(0) for treated units."""
        pass

    def estimate_effects(self) -> dict:
        """Treatment effects from imputed counterfactuals."""
        pass
```

**Deliverables:**
- [ ] Counterfactual imputation algorithm
- [ ] Pre-trend testing framework
- [ ] Robust variance estimation
- [ ] Heterogeneity analysis
- [ ] Diagnostic plots

### Phase 3: de Chaisemartin & D'Haultfœuille (Days 5-6)

```python
# dcdh_did.py
class DCDHEstimator:
    """Fuzzy DiD for continuous/heterogeneous treatment."""

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def fuzzy_did(self, outcome: str, treatment: str) -> dict:
        """Estimates with continuous treatment."""
        pass

    def placebo_tests(self, n_placebos: int = 100) -> pd.DataFrame:
        """Placebo estimators for inference."""
        pass
```

**Deliverables:**
- [ ] Fuzzy DiD implementation
- [ ] Continuous treatment handling
- [ ] Placebo-based inference
- [ ] Treatment effect heterogeneity
- [ ] Comparison with binary treatment

### Phase 4: Doubly Robust Methods (Days 7-8)

```python
# doubly_robust.py
class DoublyRobustDiD:
    """Combines propensity score and outcome regression."""

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def estimate_propensity(self, treatment: str, covariates: list[str]) -> pd.Series:
        """Propensity score estimation."""
        pass

    def outcome_regression(self, outcome: str, covariates: list[str]) -> dict:
        """Outcome model estimation."""
        pass

    def doubly_robust_att(self) -> dict:
        """DR-ATT estimator."""
        pass
```

**Deliverables:**
- [ ] Propensity score models
- [ ] Outcome regression specifications
- [ ] DR combination
- [ ] Cross-fitting procedures
- [ ] Robustness to misspecification

### Phase 5: Method Comparison (Days 9-10)

```python
# method_comparison.py
class MethodComparison:
    """Systematic comparison of DiD methods."""

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.methods = {}

    def run_all_methods(self, outcome: str) -> pd.DataFrame:
        """Estimates from all methods."""
        pass

    def diagnostic_suite(self) -> dict:
        """Comprehensive diagnostics."""
        pass
```

**Deliverables:**
- [ ] Side-by-side comparisons
- [ ] Diagnostic test battery
- [ ] Method selection guide
- [ ] Performance benchmarks
- [ ] Comprehensive testing

## Dependencies

```yaml
dependencies:
  - linearmodels  # Panel data models
  - doubleml      # Double ML methods
  - econml        # Causal ML
  - statsmodels   # Regression models
```

## Testing Strategy

```python
# tests/extensions/test_advanced_did.py
class TestSunAbraham:
    def test_weights_sum_to_one(self):
        """Verify weight properties."""
        pass

    def test_replication(self):
        """Replicate published results."""
        pass

class TestMethodComparison:
    def test_consistency(self):
        """Methods agree under homogeneity."""
        pass
```

## Success Metrics

- Methods converge under homogeneous effects
- Pre-trend tests pass for all methods
- Robust to specification choices
- Computational efficiency (<1 min for full analysis)

## Git Worktree Setup

```bash
# Create advanced-did branch and worktree
git checkout -b feature/advanced-did
git worktree add ../mpower-advanced-did feature/advanced-did

# Work in advanced-did directory
cd ../mpower-advanced-did

# Install dependencies
pip install -e ".[advanced-did]"

# Run advanced-did specific tests
pytest tests/extensions/test_advanced_did.py -v --cov=src/mpower_mortality_causal_analysis/extensions/advanced_did
```

## Performance Benchmarks

- Target: <1 minute for standard analysis
- Memory usage: <1GB for all methods comparison
- Support for large panels (>10,000 observations)

## Documentation Requirements

1. Technical paper: `docs/methods/advanced_did.md`
2. Example notebook: `notebooks/extensions/advanced_did.ipynb`
3. API documentation in module docstrings
4. README in `src/mpower_mortality_causal_analysis/extensions/advanced_did/README.md`

## Integration Guidelines

- **No Core Modifications**: Do not modify `src/mpower_mortality_causal_analysis/analysis.py` or `causal_inference/`
- **Import Pattern**: `from mpower_mortality_causal_analysis.analysis import MPOWERAnalysisPipeline`
- **Self-Contained**: All advanced DiD code in `extensions/advanced_did/`
- **Testing**: Minimum 80% coverage for all advanced DiD modules

## Methodological Standards

- Validate against published replication packages where available
- Implement standard diagnostic tests for each method
- Provide clear guidance on method selection
- Document assumptions and limitations clearly
- Include computational complexity analysis

## Validation Requirements

- Cross-validate results with existing R packages (did, DIDmultiplegt)
- Replicate published examples from method papers
- Test on simulated data with known treatment effects
- Compare performance across different data structures
- Document discrepancies and investigate sources

---

This document provides a focused roadmap for implementing the advanced DiD methods extension independently of other research directions.
