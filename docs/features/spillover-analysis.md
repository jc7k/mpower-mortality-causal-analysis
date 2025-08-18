# Extension A: Spillover Analysis - Cross-Country Policy Externalities

## Overview

This extension investigates whether MPOWER policies create positive externalities across national borders, quantifying spatial dependencies in policy adoption and health outcomes.

**Branch**: `feature/spillover-analysis`
**Module**: `src/mpower_mortality_causal_analysis/extensions/spillover/`
**Timeline**: 10 days (2 weeks)

## Scientific Value

- Reveals regional coordination benefits
- Identifies optimal policy clusters
- Quantifies cross-border health impacts
- Informs international cooperation frameworks
- **Reader Benefit**: Master spatial econometric techniques widely applicable to any policy with geographic spillovers (environmental regulations, tax policy, public health measures)

## Technical Architecture

```python
# Module structure
src/mpower_mortality_causal_analysis/extensions/spillover/
├── __init__.py
├── spatial_weights.py      # Weight matrix construction
├── spatial_models.py       # SAR, SEM, SDM models
├── diffusion_analysis.py   # Network diffusion models
├── border_analysis.py      # Border discontinuity designs
├── spillover_pipeline.py   # Main orchestration
└── visualization.py        # Spatial visualizations
```

## Implementation Plan

### Phase 1: Spatial Weight Matrices (Days 1-2)

```python
# spatial_weights.py
class SpatialWeightMatrix:
    """Constructs various spatial weight matrices."""

    def __init__(self, countries: list[str], geography_data: pd.DataFrame):
        self.countries = countries
        self.geography = geography_data

    def contiguity_matrix(self) -> np.ndarray:
        """Binary matrix for shared borders."""
        pass

    def distance_matrix(self, cutoff: float = None) -> np.ndarray:
        """Inverse distance weights."""
        pass

    def economic_proximity(self, trade_data: pd.DataFrame) -> np.ndarray:
        """Trade-weighted proximity."""
        pass
```

**Deliverables:**
- [ ] Border adjacency data collection
- [ ] Distance calculations between capitals
- [ ] Trade flow integration
- [ ] Row-standardization methods
- [ ] Unit tests for matrix properties (symmetry, row sums)

### Phase 2: Spatial Econometric Models (Days 3-5)

```python
# spatial_models.py
class SpatialPanelModel:
    """Spatial panel data models for spillover estimation."""

    def __init__(self, data: pd.DataFrame, W: np.ndarray):
        self.data = data
        self.W = W  # Spatial weight matrix

    def spatial_lag_model(self, outcome: str, covariates: list[str]) -> dict:
        """Y = ρWY + Xβ + ε"""
        pass

    def spatial_error_model(self, outcome: str, covariates: list[str]) -> dict:
        """Y = Xβ + u, u = λWu + ε"""
        pass

    def spatial_durbin_model(self, outcome: str, covariates: list[str]) -> dict:
        """Y = ρWY + Xβ + WXθ + ε"""
        pass
```

**Deliverables:**
- [ ] Maximum likelihood estimation
- [ ] Direct, indirect, and total effects decomposition
- [ ] Spatial autocorrelation tests (Moran's I, Geary's C)
- [ ] Model selection criteria (AIC, BIC)
- [ ] Robust standard errors

### Phase 3: Network Diffusion Analysis (Days 6-7)

```python
# diffusion_analysis.py
class PolicyDiffusionNetwork:
    """Analyzes MPOWER policy diffusion through networks."""

    def __init__(self, adoption_data: pd.DataFrame):
        self.adoption_data = adoption_data

    def estimate_contagion(self) -> dict:
        """Estimates peer influence on adoption."""
        pass

    def identify_influencers(self) -> list[str]:
        """Identifies key countries in diffusion."""
        pass
```

**Deliverables:**
- [ ] Threshold models of adoption
- [ ] Cascade detection algorithms
- [ ] Influence metrics (centrality, betweenness)
- [ ] Temporal network evolution
- [ ] Visualization of diffusion waves

### Phase 4: Border Discontinuity Design (Days 8-9)

```python
# border_analysis.py
class BorderDiscontinuity:
    """RDD analysis at international borders."""

    def __init__(self, border_data: pd.DataFrame):
        self.border_data = border_data

    def estimate_border_effect(self, outcome: str) -> dict:
        """Estimates discontinuity at borders."""
        pass
```

**Deliverables:**
- [ ] Geographic RDD implementation
- [ ] Bandwidth selection (Imbens-Kalyanaraman)
- [ ] Local polynomial regression
- [ ] Placebo borders validation
- [ ] Heterogeneity by border characteristics

### Phase 5: Integration & Testing (Day 10)

**Deliverables:**
- [ ] Complete pipeline orchestration
- [ ] Comprehensive test suite (>80% coverage)
- [ ] Performance benchmarks
- [ ] Documentation and examples
- [ ] Validation against published spatial studies

## Dependencies

```yaml
dependencies:
  - pysal>=2.0  # Spatial analysis library
  - libpysal    # Spatial weights
  - spreg       # Spatial regression
  - networkx    # Network analysis
  - geopandas   # Geographic data
  - folium      # Interactive maps
```

## Testing Strategy

```python
# tests/extensions/test_spillover.py
class TestSpatialWeights:
    def test_contiguity_matrix_properties(self):
        """Test symmetry and row-standardization."""
        pass

    def test_distance_decay(self):
        """Test inverse distance calculations."""
        pass

class TestSpatialModels:
    def test_spatial_lag_estimation(self):
        """Validate against known results."""
        pass

    def test_effect_decomposition(self):
        """Test direct/indirect effect calculation."""
        pass
```

## Success Metrics

- Spatial autocorrelation detected in policy adoption (Moran's I > 0.3)
- Significant spillover effects (indirect effects p < 0.05)
- Border discontinuities in health outcomes
- Network influencers identified (top 5 countries)

## Git Worktree Setup

```bash
# Create spillover analysis branch and worktree
git checkout -b feature/spillover-analysis
git worktree add ../mpower-spillover feature/spillover-analysis

# Work in spillover directory
cd ../mpower-spillover

# Install dependencies
pip install -e ".[spillover]"

# Run spillover-specific tests
pytest tests/extensions/test_spillover.py -v --cov=src/mpower_mortality_causal_analysis/extensions/spillover
```

## Performance Benchmarks

- Target: <10 minutes for 195 countries
- Memory usage: <4GB for full spatial analysis
- Parallel processing support for large weight matrices

## Documentation Requirements

1. Technical paper: `docs/methods/spillover_analysis.md`
2. Example notebook: `notebooks/extensions/spillover_analysis.ipynb`
3. API documentation in module docstrings
4. README in `src/mpower_mortality_causal_analysis/extensions/spillover/README.md`

## Integration Guidelines

- **No Core Modifications**: Do not modify `src/mpower_mortality_causal_analysis/analysis.py` or `causal_inference/`
- **Import Pattern**: `from mpower_mortality_causal_analysis.analysis import MPOWERAnalysisPipeline`
- **Self-Contained**: All spillover-specific code in `extensions/spillover/`
- **Testing**: Minimum 80% coverage for all spillover modules

---

This document provides a focused roadmap for implementing the spillover analysis extension independently of other research directions.
