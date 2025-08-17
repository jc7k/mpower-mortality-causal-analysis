name: "MPOWER Tobacco Control Policy Causal Analysis"
description: |

## Purpose
Comprehensive Product Requirements Plan (PRP) for implementing a rigorous causal inference analysis of WHO MPOWER tobacco control policies' impact on smoking prevalence, lung cancer mortality, and cardiovascular disease mortality using modern econometric methods.

## Core Principles
1. **Context is King**: Include ALL necessary documentation, examples, and caveats
2. **Validation Loops**: Provide executable tests/lints the AI can run and fix
3. **Information Dense**: Use keywords and patterns from the codebase
4. **Progressive Success**: Start simple, validate, then enhance
5. **Global rules**: Be sure to follow all rules in CLAUDE.md

---

## Goal
Build a complete causal inference analysis pipeline that evaluates the impact of WHO MPOWER tobacco control policies (2008-2019) on health outcomes using state-of-the-art econometric methods, specifically:
- Callaway & Sant'Anna (2021) staggered difference-in-differences for policy evaluation
- Two-way fixed effects models with robust standard errors
- Synthetic control methods for robustness checks
- Comprehensive event study analysis with lag structures

## Why
- **Scientific Impact**: First empirical study linking MPOWER policies to actual mortality outcomes (vs. simulation)
- **Methodological Advance**: First application of Callaway & Sant'Anna methods to tobacco policy evaluation
- **Policy Relevance**: Provides evidence-based guidance for governments on which tobacco control measures save lives most effectively
- **Technical Innovation**: Demonstrates modern causal inference methods in Python ecosystem

## What
A complete research analysis system that:
- Processes manually downloaded WHO MPOWER, IHME GBD, and World Bank data into analysis-ready panel dataset
- Implements multiple causal inference estimators with proper statistical inference
- Generates publication-ready tables, figures, and robustness checks
- Provides reproducible code following academic standards

### Success Criteria
- [ ] Clean panel dataset with 150+ countries, 2008-2019 (12 years)
- [ ] Significant treatment effects detectable at p<0.05 level
- [ ] Event study plots showing policy impact timing
- [ ] Robustness across multiple estimators and specifications
- [ ] All code passes linting, type checking, and unit tests
- [ ] Publication-ready tables and figures generated

## All Needed Context

### Documentation & References (list all context needed to implement the feature)
```yaml
# MUST READ - Include these in your context window
- docfile: PRPs/ai_docs/causal_inference_methods.md
  why: Complete implementation details for Callaway & Sant'Anna DiD, synthetic control methods, and panel fixed effects

- docfile: PRPs/ai_docs/data_sources_apis.md
  why: WHO MPOWER data structure, IHME GBD mortality variables, World Bank control variables, and integration requirements

- docfile: PRPs/ai_docs/python_packages.md
  why: Usage patterns for differences, pyfixest, pysyncon, linearmodels with critical implementation details

- url: https://bcallaway11.github.io/did/articles/did-basics.html
  why: Original R implementation patterns for Callaway & Sant'Anna estimator

- url: https://sdfordham.github.io/pysyncon/synth.html
  why: Synthetic control method implementation in Python

- file: PRPs/initial_mpower.md
  why: Research design, hypotheses, methodology specifications, and expected results

- file: mpower-research-plan.md
  why: Complete research context, data definitions, technical requirements
```

### Current Codebase Tree
```bash
project/
├── CLAUDE.md              # Project conventions and style guide
├── PRPs/
│   ├── ai_docs/           # Research documentation
│   ├── initial_mpower.md  # Feature specification
│   └── templates/         # PRP templates
├── data/                  # Data directory (manually downloaded files exist)
│   ├── raw/               # WHO MPOWER, IHME GBD, World Bank data
│   └── processed/         # Target for analysis-ready datasets
├── README.md              # Project documentation
└── mpower-research-plan.md # Detailed research methodology
```

### Desired Codebase Tree with Files to be Added
```bash
project/
├── src/                   # Main source code (following CLAUDE.md 500-line limit)
│   ├── __init__.py
│   ├── data/              # Data processing modules
│   │   ├── __init__.py
│   │   ├── cleaning.py    # Raw data cleaning and validation
│   │   ├── integration.py # Multi-source data merging
│   │   └── validation.py  # Data quality checks
│   ├── analysis/          # Econometric analysis modules
│   │   ├── __init__.py
│   │   ├── did_analysis.py        # Callaway & Sant'Anna implementation
│   │   ├── fixed_effects.py      # Two-way FE models with pyfixest
│   │   ├── synthetic_control.py  # Synthetic control robustness
│   │   └── event_study.py        # Event study specifications
│   ├── visualization/     # Plotting and tables
│   │   ├── __init__.py
│   │   ├── plots.py       # Event study plots, trend analysis
│   │   └── tables.py      # Regression tables, summary stats
│   └── utils/             # Utility functions
│       ├── __init__.py
│       ├── config.py      # Configuration and constants
│       └── helpers.py     # Common utility functions
├── tests/                 # Pytest test suite
│   ├── __init__.py
│   ├── test_data_cleaning.py
│   ├── test_analysis.py
│   ├── test_integration.py
│   └── fixtures/          # Test data fixtures
├── notebooks/             # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_main_analysis.ipynb
│   └── 03_robustness_checks.ipynb
├── output/                # Generated results
│   ├── tables/
│   ├── figures/
│   └── reports/
├── pyproject.toml         # uv project configuration and dependencies
└── .venv/           # Virtual environment (already exists)
```

### Known Gotchas of our Codebase & Library Quirks
```python
# CRITICAL: differences package requires specific data structure
# Cohort variable must be year of first treatment (not 0/1 indicator)
df['treatment_year'] = df['first_high_mpower_year']  # Not binary treatment
df.loc[df['never_treated'], 'treatment_year'] = 0   # Never-treated coded as 0

# CRITICAL: pyfixest uses R-style formula syntax
# Must use | for fixed effects, not +
model = pf.feols('outcome ~ treatment + controls | country + year', data=df)

# CRITICAL: Panel data must be sorted by unit-time for lag creation
df = df.sort_values(['iso3', 'year'])  # Essential for groupby operations

# CRITICAL: Use .venv for all Python commands per CLAUDE.md
# Example: source .venv/bin/activate && python src/analysis/main.py

# CRITICAL: Follow 500-line file limit - split complex modules
# Example: separate did_analysis.py into estimation + aggregation files if needed

# CRITICAL: Use type hints and docstrings for all functions per CLAUDE.md
def estimate_did_effects(df: pd.DataFrame, outcome: str) -> Dict[str, Any]:
    """
    Estimate Callaway & Sant'Anna DiD effects.

    Args:
        df: Panel data with required columns
        outcome: Name of outcome variable

    Returns:
        Dict containing ATT estimates and aggregations
    """
```

## Implementation Blueprint

### Data Models and Structure

Create the core data models to ensure type safety and consistency.
```python
# src/utils/config.py - Data structure definitions
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import pandas as pd

@dataclass
class MPOWERData:
    """MPOWER policy data structure"""
    iso3: str
    country_name: str
    year: int
    mpower_total: float  # 0-37
    m_score: float      # 0-4 (Monitor)
    p_score: float      # 0-4 (Protect)
    o_score: float      # 0-4 (Offer)
    w_score: float      # 0-4 (Warn)
    e_score: float      # 0-4 (Enforce)
    r_score: float      # 0-5 (Raise taxes)

@dataclass
class HealthOutcomes:
    """Health outcome variables from IHME GBD"""
    iso3: str
    year: int
    prev_smoking_adult: float           # Smoking prevalence %
    mort_lung_cancer_asr: float         # Age-standardized lung cancer mortality
    mort_cvd_asr: float                 # CVD mortality rate

@dataclass
class ControlVariables:
    """Economic and demographic controls from World Bank"""
    iso3: str
    year: int
    gdp_pc_constant_2010: float         # GDP per capita (log transformed)
    urban_pop_pct: float                # Urban population %
    edu_index: Optional[float]          # Education index
    haq_index: Optional[float]          # Healthcare access quality
    alcohol_consumption: Optional[float] # Liters per capita (log)
    air_pollution_pm25: Optional[float] # PM2.5 exposure

@dataclass
class AnalysisResults:
    """Container for econometric results"""
    method: str                         # 'callaway_santanna', 'fixed_effects', etc.
    outcome: str                        # Outcome variable name
    treatment_effect: float             # Main treatment effect
    standard_error: float               # Clustered standard error
    p_value: float                      # Statistical significance
    confidence_interval: tuple         # (lower, upper) bounds
    n_observations: int                 # Sample size
    n_countries: int                    # Number of countries
    additional_stats: Dict[str, Any]    # Method-specific statistics
```

### List of Tasks to be Completed (in order)

```yaml
Task 1: Data Cleaning and Integration
MODIFY src/data/cleaning.py:
  - CREATE functions to clean WHO MPOWER data from raw CSV files
  - VALIDATE MPOWER scores within bounds (0-37 total, component limits)
  - HANDLE missing years with linear interpolation
  - IMPLEMENT country code standardization (ISO3)

CREATE src/data/integration.py:
  - MERGE MPOWER, IHME GBD, and World Bank data sources
  - RESOLVE country name mismatches between datasets
  - CREATE balanced panel structure (country-year observations)
  - GENERATE treatment timing variables for staggered DiD

CREATE src/data/validation.py:
  - IMPLEMENT comprehensive data quality checks
  - VALIDATE panel structure (no duplicate country-years)
  - CHECK outcome variable distributions and outliers
  - ENSURE sufficient coverage (150+ countries, 5+ years per country)

Task 2: Core Econometric Analysis
CREATE src/analysis/did_analysis.py:
  - IMPLEMENT Callaway & Sant'Anna estimator using `differences` package
  - CREATE treatment cohort variables (year of high MPOWER adoption)
  - ESTIMATE group-time average treatment effects
  - AGGREGATE results for event study, overall ATT, by-group effects

CREATE src/analysis/fixed_effects.py:
  - IMPLEMENT two-way fixed effects using `pyfixest`
  - ESTIMATE main specifications with country and year FE
  - CREATE lag structure models (0-5 year lags)
  - COMPUTE clustered standard errors at country level

CREATE src/analysis/event_study.py:
  - GENERATE event time indicators relative to treatment
  - ESTIMATE dynamic treatment effects (leads and lags)
  - IMPLEMENT pre-trend testing
  - CREATE event study plots with confidence intervals

Task 3: Robustness and Sensitivity Analysis
CREATE src/analysis/synthetic_control.py:
  - IMPLEMENT synthetic control using `pysyncon` for case studies
  - SELECT early/strong MPOWER adopters for individual analysis
  - CONSTRUCT synthetic counterfactuals using donor pool
  - GENERATE placebo tests and permutation inference

MODIFY src/analysis/did_analysis.py:
  - ADD alternative treatment definitions (continuous MPOWER, components)
  - IMPLEMENT heterogeneity analysis by income, region, baseline smoking
  - CREATE placebo outcome tests (unrelated mortality causes)
  - ADD sample robustness (exclude tobacco producers, high-income)

Task 4: Visualization and Reporting
CREATE src/visualization/plots.py:
  - GENERATE event study plots with confidence bands
  - CREATE parallel trends visualization for pre-treatment periods
  - IMPLEMENT treatment effect heterogeneity plots
  - ADD synthetic control gap plots

CREATE src/visualization/tables.py:
  - FORMAT regression results for publication
  - CREATE summary statistics tables
  - GENERATE robustness check comparison tables
  - IMPLEMENT LaTeX output for academic papers

Task 5: Testing and Quality Assurance
CREATE tests/test_data_cleaning.py:
  - TEST MPOWER score bounds validation
  - VERIFY panel structure integrity
  - CHECK data integration accuracy
  - VALIDATE treatment timing construction

CREATE tests/test_analysis.py:
  - TEST econometric estimator implementations
  - VERIFY standard error calculations
  - CHECK aggregation methods
  - VALIDATE coefficient interpretation

CREATE tests/test_integration.py:
  - TEST end-to-end analysis pipeline
  - VERIFY reproducibility of results
  - CHECK output file generation
  - VALIDATE notebook execution

Task 6: Documentation and Reproducibility
CREATE notebooks/01_data_exploration.ipynb:
  - EXPLORE raw data characteristics
  - VISUALIZE missing data patterns
  - ANALYZE treatment adoption timing
  - CREATE descriptive statistics

CREATE notebooks/02_main_analysis.ipynb:
  - IMPLEMENT complete analysis pipeline
  - GENERATE main results tables and figures
  - DOCUMENT key findings and interpretation
  - CREATE reproducible workflow

UPDATE README.md:
  - DOCUMENT installation and setup procedures
  - EXPLAIN data sources and acquisition
  - PROVIDE analysis replication instructions
  - INCLUDE citation information and licensing
```

### Per Task Pseudocode

```python
# Task 1: Data Integration Pseudocode
def integrate_panel_data(mpower_path: str, gbd_path: str, wb_path: str) -> pd.DataFrame:
    """
    Integrate MPOWER, GBD, and World Bank data into analysis-ready panel.

    CRITICAL: Manual data already downloaded - use existing files
    PATTERN: Follow existing data cleaning patterns in codebase
    """
    # Load raw data (manual downloads)
    mpower_raw = pd.read_csv(mpower_path)  # WHO MPOWER scores
    gbd_raw = pd.read_csv(gbd_path)        # IHME mortality data
    wb_raw = pd.read_csv(wb_path)          # World Bank controls

    # Clean MPOWER data
    mpower_clean = clean_mpower_scores(mpower_raw)
    mpower_panel = interpolate_missing_years(mpower_clean)

    # Process health outcomes
    gbd_clean = filter_gbd_outcomes(gbd_raw,
                                   outcomes=['lung_cancer', 'cvd', 'smoking_prev'])

    # Merge on ISO3-year
    panel = mpower_panel.merge(gbd_clean, on=['iso3', 'year'], how='inner')
    panel = panel.merge(wb_raw, on=['iso3', 'year'], how='left')

    # Create treatment variables for staggered DiD
    panel['treatment_year'] = calculate_treatment_timing(panel, threshold=25)

    return validate_panel_structure(panel)

# Task 2: Callaway & Sant'Anna Implementation
def estimate_staggered_did(df: pd.DataFrame, outcome: str) -> AnalysisResults:
    """
    CRITICAL: differences package requires specific data format
    GOTCHA: treatment_year must be year of adoption (not 0/1)
    """
    from differences import ATTgt

    # Prepare data for differences package
    df_cs = df[['iso3', 'year', 'treatment_year', outcome, 'gdp_log']].copy()
    df_cs = df_cs.dropna()  # ATTgt requires complete cases

    # Initialize estimator
    att_gt = ATTgt(data=df_cs,
                   cohort_name='treatment_year',
                   time_name='year',
                   id_name='iso3')

    # Fit with covariates (PATTERN: condition on pre-treatment characteristics)
    att_gt.fit(formula=f'{outcome} ~ gdp_log')

    # Aggregate results
    overall_att = att_gt.aggregate('simple')      # Overall effect
    event_study = att_gt.aggregate('event')       # Dynamic effects
    by_cohort = att_gt.aggregate('group')         # Heterogeneity

    return format_cs_results(overall_att, event_study, by_cohort)

# Task 3: Two-Way Fixed Effects with pyfixest
def estimate_twfe_model(df: pd.DataFrame, outcome: str) -> Dict[str, Any]:
    """
    CRITICAL: Use | for fixed effects in pyfixest formula
    PATTERN: Cluster standard errors at country level
    """
    import pyfixest as pf

    # Main specification with country and year FE
    main_spec = pf.feols(
        f'{outcome} ~ mpower_total + gdp_log + urban_pct | iso3 + year',
        data=df,
        vcov='CL1(iso3)'  # Cluster by country
    )

    # Event study specification
    event_spec = pf.feols(
        f'{outcome} ~ i(event_time, ref=-1) + gdp_log | iso3 + year',
        data=df,
        vcov='CL1(iso3)'
    )

    # Lag structure
    lag_spec = pf.feols(
        f'{outcome} ~ mpower_lag_0 + mpower_lag_1 + mpower_lag_2 + gdp_log | iso3 + year',
        data=df,
        vcov='CL1(iso3)'
    )

    return {
        'main': main_spec,
        'event_study': event_spec,
        'distributed_lags': lag_spec
    }
```

### Integration Points
```yaml
DATABASE:
  - No database required: Use CSV/Parquet files in data/processed/
  - DuckDB optional: For fast analytical queries on large datasets

CONFIG:
  - add to: src/utils/config.py
  - pattern: "DATA_PATH = Path('data/processed/panel_data.csv')"

NOTEBOOKS:
  - integration: Import analysis modules from src/
  - pattern: "from src.analysis.did_analysis import estimate_staggered_did"

TESTING:
  - integration: pytest discovers tests/ automatically
  - pattern: "pytest tests/ -v" runs all tests
```

## Validation Loop

### Level 1: Syntax & Style
```bash
# Run these FIRST - fix any errors before proceeding
source .venv/bin/activate  # Use existing virtual environment
ruff check src/ --fix           # Auto-fix style issues
ruff format src/                 # Format code
mypy src/                        # Type checking

# Expected: No errors. If errors, READ the error and fix.
```

### Level 2: Unit Tests
```python
# CREATE comprehensive test suite following patterns
def test_mpower_data_validation():
    """Test MPOWER score bounds and structure"""
    df = pd.read_csv('data/processed/panel_data.csv')

    # Score bounds
    assert df['mpower_total'].between(0, 37).all()
    assert df[['m_score', 'p_score', 'o_score', 'w_score', 'e_score']].between(0, 4).all()
    assert df['r_score'].between(0, 5).all()

    # Panel structure
    assert not df.groupby(['iso3', 'year']).size().gt(1).any()

def test_treatment_assignment():
    """Test staggered DiD treatment construction"""
    df = pd.read_csv('data/processed/panel_data.csv')

    # Treatment year should be consistent within country
    treatment_consistency = df.groupby('iso3')['treatment_year'].nunique()
    assert (treatment_consistency == 1).all()

def test_callaway_santanna_estimation():
    """Test Callaway & Sant'Anna implementation"""
    from src.analysis.did_analysis import estimate_staggered_did

    # Use small test dataset
    df_test = create_test_panel_data()
    results = estimate_staggered_did(df_test, 'mortality_rate')

    # Check result structure
    assert 'overall_att' in results
    assert 'standard_error' in results
    assert results['n_observations'] > 0
```

```bash
# Run and iterate until passing:
source .venv/bin/activate
uv run pytest tests/ -v
# If failing: Read error, understand root cause, fix code, re-run
```

### Level 3: Integration Test
```bash
# Test complete analysis pipeline
source .venv/bin/activate

# Run data processing
python src/data/cleaning.py
python src/data/integration.py

# Run main analysis
python src/analysis/did_analysis.py
python src/analysis/fixed_effects.py

# Expected: Clean panel data file created, results tables generated
# Check: ls data/processed/ should show panel_data.csv
# Check: ls output/tables/ should show regression results
```

### Level 4: Notebook Execution
```bash
# Test reproducible analysis workflow
source .venv/bin/activate
jupyter nbconvert --execute notebooks/02_main_analysis.ipynb

# Expected: Notebook executes without errors, generates all outputs
# If error: Check notebook cell outputs for stack traces
```

## Final Validation Checklist
- [ ] All tests pass: `uv run pytest tests/ -v`
- [ ] No linting errors: `ruff check src/`
- [ ] No type errors: `mypy src/`
- [ ] Panel data validates: 150+ countries, 2008-2019 coverage
- [ ] Treatment effects estimated: Callaway & Sant'Anna and TWFE results
- [ ] Event study plots generated: Pre-trends and dynamic effects visible
- [ ] Robustness checks complete: Synthetic control and alternative specs
- [ ] All notebooks execute: End-to-end reproducible workflow
- [ ] Output files created: Tables in output/tables/, figures in output/figures/

---

## Anti-Patterns to Avoid
- ❌ Don't use wbgapi or API calls - data already manually downloaded
- ❌ Don't create files longer than 500 lines (CLAUDE.md requirement)
- ❌ Don't skip data validation - econometric results depend on clean data
- ❌ Don't ignore clustered standard errors - panel data requires proper inference
- ❌ Don't hardcode file paths - use configurable paths in config.py
- ❌ Don't mix estimation methods in single file - separate analysis modules
- ❌ Don't skip pre-trends testing - essential for DiD validity

## Expected Implementation Confidence Score: 9/10

### Reasoning for High Confidence
- **Complete Context**: All necessary documentation and implementation patterns provided
- **Proven Methods**: Well-established econometric techniques with Python implementations
- **Clear Data Structure**: Manual data downloads eliminate API complexity
- **Modular Design**: CLAUDE.md conventions ensure maintainable code structure
- **Comprehensive Testing**: Multi-level validation ensures robustness
- **Academic Standards**: Follows established econometric best practices

### Potential Challenges (1 point deduction)
- **Package Quirks**: `differences` package has specific data format requirements
- **Computational Demands**: Large panel data may require memory optimization
- **Result Interpretation**: Complex econometric output requires careful handling

This PRP provides comprehensive context for successful one-pass implementation of a rigorous causal inference analysis meeting academic publication standards.
