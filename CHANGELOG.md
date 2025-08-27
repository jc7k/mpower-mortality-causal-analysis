# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.1] - 2025-08-27

### Fixed
- Correct p-value calculations in Callaway & Sant'Anna aggregation (use normal CDF across simple/group/calendar/event).
- Pass properly indexed data to `linearmodels` predict in panel methods to avoid misalignment.

### Improved
- Standardized logging across core modules (analysis, synthetic_control, mechanism_analysis); removed prints.
- Ensure plot directories exist before saving figures (descriptives and analysis paths).
- Vectorized group-wise forward/backward fill in data preparation for better performance.
- Harden JSON export for results and avoid unsafe `isinstance` unions.

### Cleanups
- Replaced bare `except:` with `except Exception` in core utilities and spillover pipeline.
- Removed unused imports/variables; clarified ambiguous names in spillover utilities.

### CI
- Scoped ruff lint/format to `src/**` to avoid noise from examples/scripts.

### Docs
- Added AGENTS.md contributor guidelines.
- Added CODE_REVIEW.md with identified issues and suggested patches.

## [2.0.0] - 2025-01-18

### 🚀 Major Research Extensions Added

This release introduces four comprehensive research extensions to the MPOWER mortality causal analysis framework, significantly expanding the analytical capabilities for tobacco control policy evaluation.

### Added

#### 📊 Advanced Difference-in-Differences Methods (`extensions/advanced_did/`)
- **Sun & Abraham (2021)** interaction-weighted estimator for staggered treatment
- **Borusyak et al. (2021)** imputation-based approach with pre-trend testing
- **de Chaisemartin & D'Haultfœuille** fuzzy DiD for continuous treatments
- **Doubly robust estimators** combining propensity scores and outcome regression
- **Method comparison framework** for systematic evaluation of DiD approaches
- Comprehensive test suite with 80%+ coverage

#### 💰 Cost-Effectiveness Analysis Framework (`extensions/cost_effectiveness/`)
- **Health economic modeling** with QALY/DALY calculations
- **Markov models** for disease progression simulation
- **ICER analysis** with probabilistic sensitivity analysis
- **Budget optimization** using linear programming for policy portfolios
- **Cost-effectiveness acceptability curves** and dominance analysis
- **WHO-CHOICE validation** for cost estimates

#### 🌍 Spillover Analysis (`extensions/spillover/`)
- **Spatial econometric models** (SAR, SEM, SDM) for cross-country effects
- **Network diffusion analysis** for policy adoption patterns
- **Border discontinuity design** for neighboring country effects
- **Spatial weight matrices** with distance, contiguity, and cultural proximity
- **Contagion models** for policy spread analysis
- **Visualization tools** for diffusion patterns

#### 🎯 Policy Optimization (`extensions/optimization/`)
- **Interaction effects analysis** for policy synergies
- **Sequential implementation optimizer** with dynamic programming
- **Combinatorial optimization** for policy scheduling
- **Political economy constraints** modeling feasibility
- **Decision support system** for policy recommendations
- **Game-theoretic solutions** for stakeholder preferences

### Enhanced

#### Core Analysis Pipeline
- **Mechanism analysis** now integrated with all new extension methods
- **Synthetic control** methods enhanced with extension compatibility
- **Event study** analysis now supports advanced DiD methods
- **Robustness checks** expanded with new validation approaches

### Technical Improvements
- **Modular architecture**: Each extension is self-contained with clear interfaces
- **Consistent API design**: Uniform method signatures across extensions
- **Comprehensive testing**: 70+ new tests covering all extensions
- **Documentation**: Method papers and example notebooks for each extension
- **Performance optimization**: Efficient implementations for large-scale analysis

### Dependencies Added
- `linearmodels` - For advanced panel data methods
- `cvxpy` - For optimization problems in policy scheduling
- `networkx` - For network analysis in spillover detection
- `geopandas` - For spatial analysis and border discontinuities
- `lifelines` - For survival analysis in health economic modeling

### Statistics
- **Files added**: 40 (28 source, 7 test, 5 documentation)
- **Lines of code**: 16,666+ new lines
- **Test coverage**: Maintained at 80%+ across all modules
- **Methods implemented**: 15+ new causal inference and optimization methods

## [1.5.0] - 2025-01-17

### Added
- Complete mechanism analysis framework for MPOWER component decomposition
- Synthetic control methods addressing parallel trends violations
- Component-specific policy effectiveness rankings

## [1.0.0] - 2025-01-16

### Added
- Initial implementation of Callaway & Sant'Anna DiD
- MPOWER mortality causal analysis pipeline
- Event study analysis with parallel trends testing
- Descriptive statistics and visualization tools
- Basic robustness checks framework

---

## Migration Guide

### Upgrading from v1.x to v2.0

The new extensions are fully backward compatible. To use the new features:

```python
# Import new extension modules
from mpower_mortality_causal_analysis.extensions.advanced_did import SunAbraham
from mpower_mortality_causal_analysis.extensions.cost_effectiveness import HealthEconomicModel
from mpower_mortality_causal_analysis.extensions.spillover import SpatialAnalysis
from mpower_mortality_causal_analysis.extensions.optimization import PolicyOptimizer

# Or use through the main pipeline
pipeline = MPOWERAnalysisPipeline(data_path)
results = pipeline.run_full_analysis(
    include_extensions=['advanced_did', 'cost_effectiveness', 'spillover', 'optimization']
)
```

### Breaking Changes
None - all v1.x code remains fully functional.

### Deprecations
None - no features have been deprecated.

---

## Contributors
- Advanced DiD Methods: Implemented via feature/advanced-did branch
- Cost-Effectiveness: Implemented via feature/cost-effectiveness branch
- Spillover Analysis: Implemented via feature/spillover-analysis branch
- Policy Optimization: Implemented via feature/policy-optimization branch

## License
This project maintains its original license terms.
