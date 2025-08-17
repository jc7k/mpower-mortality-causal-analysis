# Python Packages and Implementation Patterns for MPOWER Analysis

## Core Econometric Packages

### 1. differences - Callaway & Sant'Anna Difference-in-Differences

#### Package Details
- **Version**: 0.2.0 (latest)
- **Install**: `pip install differences`
- **Maintainer**: Bernardo Dionisi
- **Documentation**: Limited but functional

#### Key Features
- Implements Callaway & Sant'Anna (2021) staggered DiD
- Handles multiple time periods and staggered treatment
- Built-in aggregation methods (event study, simple, group, calendar)
- Supports both balanced and unbalanced panels

#### Critical Usage Pattern
```python
from differences import ATTgt

# Data must have specific column structure
att_gt = ATTgt(
    data=df,
    cohort_name='treatment_year',  # Year of first treatment (0 for never-treated)
    time_name='year',              # Time variable
    id_name='country',             # Unit identifier
    outcome_name='mortality_rate'   # Outcome variable
)

# Fit model (can include covariates)
att_gt.fit(formula='mortality_rate ~ gdp_log + urban_pct')

# Aggregate results
event_study = att_gt.aggregate('event')      # Event study plot
overall = att_gt.aggregate('simple')         # Overall ATT
by_group = att_gt.aggregate('group')         # By treatment cohort
```

#### Data Requirements
- **Cohort Variable**: Must be year of first treatment (not 0/1 indicator)
- **Panel Structure**: Long format with unit-time observations
- **Never-Treated**: Essential for identification (code as 0 or large number)

### 2. pyfixest - High-Dimensional Fixed Effects

#### Package Details
- **Repository**: https://github.com/py-econometrics/pyfixest
- **Install**: `pip install pyfixest`
- **Philosophy**: Mirror R fixest syntax exactly

#### Key Advantages
- Fast estimation of high-dimensional fixed effects
- Familiar fixest-style formula syntax
- Multiple standard error corrections
- Built-in multi-outcome models

#### Critical Usage Pattern
```python
import pyfixest as pf

# Two-way fixed effects with clustered SEs
mod = pf.feols(
    'mortality_rate ~ mpower_total + gdp_log + urban_pct | country + year',
    data=df,
    vcov='CL1(country)'  # Cluster by country
)

# Multiple outcomes simultaneously
mod_multi = pf.feols(
    ['lung_cancer', 'cvd_mortality', 'smoking_prev'] ~
    'mpower_total + gdp_log | country + year',
    data=df,
    vcov='CL1(country)'
)

# Event study specification
mod_event = pf.feols(
    'mortality_rate ~ i(event_time, ref=-1) + gdp_log | country + year',
    data=df_event,
    vcov='CL1(country)'
)
```

#### Fixed Effects Syntax
- `| country + year`: Entity and time fixed effects
- `| country^year`: Interacted fixed effects
- `| country[gdp_log]`: Varying slopes

### 3. pysyncon - Synthetic Control Methods

#### Package Details
- **Repository**: https://github.com/sdfordham/pysyncon
- **Install**: `pip install pysyncon`
- **Methods**: Standard, Robust, Augmented, Penalized synthetic control

#### Implementation Pattern
```python
from pysyncon import Synth

# For individual country case studies
synth = Synth(
    data=df,
    unit='country',
    time='year',
    treatment='treatment_indicator',
    outcome='mortality_rate'
)

# Fit with predictors (pre-treatment characteristics)
synth.fit(
    predictors=['gdp_log', 'urban_pct', 'education'],
    predictors_op='mean'  # How to aggregate predictors over time
)

# Results and visualization
synth.summary()
synth.plot()
synth.gaps_plot()  # Treatment effect over time
```

#### When to Use
- Case studies of early MPOWER adopters
- Countries with unique treatment timing
- Robustness checks for DiD results

### 4. linearmodels - Advanced Panel Methods

#### Package Details
- **Install**: `pip install linearmodels`
- **Strength**: Comprehensive panel data methods
- **Use Case**: Alternative specifications and robustness

#### Key Methods
```python
from linearmodels import PanelOLS, RandomEffects, FirstDifferenceOLS

# Panel OLS with entity/time effects
mod_panel = PanelOLS(
    dependent=df['mortality_rate'],
    exog=df[['mpower_total', 'gdp_log']],
    entity_effects=True,
    time_effects=True
)
results = mod_panel.fit(cov_type='clustered', cluster_entity=True)

# First differences (for non-stationary data)
mod_fd = FirstDifferenceOLS(
    dependent=df['mortality_rate'],
    exog=df[['mpower_total', 'gdp_log']]
)

# Random effects (alternative to fixed effects)
mod_re = RandomEffects(
    dependent=df['mortality_rate'],
    exog=df[['mpower_total', 'gdp_log']]
)
```

## Supporting Data Analysis Packages

### 5. pandas - Data Management

#### Critical Patterns for Panel Data
```python
import pandas as pd

# Panel data setup
df = df.set_index(['country', 'year']).sort_index()

# Lag creation for distributed lag models
for lag in range(1, 6):
    df[f'mpower_lag_{lag}'] = df.groupby('country')['mpower_total'].shift(lag)

# Lead creation for pre-trends testing
for lead in range(1, 4):
    df[f'mpower_lead_{lead}'] = df.groupby('country')['mpower_total'].shift(-lead)

# Event time creation for event studies
df['years_since_treatment'] = df['year'] - df['first_treatment_year']
```

### 6. statsmodels - Traditional Econometrics

#### Usage for Diagnostic Tests
```python
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white, acorr_ljungbox

# Heteroskedasticity tests
_, pval_het, _, _ = het_white(residuals, exog)

# Serial correlation tests
ljung_box = acorr_ljungbox(residuals, lags=5, return_df=True)

# Unit root tests for stationarity
from statsmodels.tsa.stattools import adfuller
adf_result = adfuller(df['mortality_rate'])
```

### 7. matplotlib/seaborn/plotly - Visualization

#### Essential Plot Types for Causal Inference
```python
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Event study plot
def plot_event_study(results, outcome_name):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot coefficients with confidence intervals
    ax.errorbar(event_times, coefficients, yerr=std_errors,
               fmt='o-', capsize=5)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)

    ax.set_xlabel('Years Relative to Policy Adoption')
    ax.set_ylabel(f'Effect on {outcome_name}')
    ax.set_title('Event Study: MPOWER Policy Effects')

# Parallel trends visualization
def plot_parallel_trends(df, treatment_var, outcome_var):
    fig = px.line(df, x='year', y=outcome_var,
                  color=treatment_var,
                  title='Pre-Treatment Trends')
    return fig
```

## Data Processing Packages

### 8. DuckDB - Analytical Database

#### Usage for Large Panel Operations
```python
import duckdb

# Fast analytical queries on large datasets
conn = duckdb.connect(':memory:')
conn.register('panel_df', df)

# Complex aggregations
result = conn.execute("""
    SELECT country,
           AVG(mortality_rate) as avg_mortality,
           FIRST(mpower_total ORDER BY year) as initial_mpower
    FROM panel_df
    WHERE year BETWEEN 2008 AND 2012
    GROUP BY country
""").df()
```

### 9. pytest - Testing Framework

#### Essential Test Patterns
```python
import pytest
import pandas as pd

def test_mpower_score_bounds():
    """Test MPOWER scores are within valid ranges"""
    df = pd.read_csv('data/processed/panel_data.csv')

    assert df['mpower_total'].between(0, 37).all()
    assert df[['m_score', 'p_score', 'o_score', 'w_score', 'e_score']].between(0, 4).all()
    assert df['r_score'].between(0, 5).all()

def test_panel_structure():
    """Test panel data structure integrity"""
    df = pd.read_csv('data/processed/panel_data.csv')

    # No duplicate country-years
    duplicates = df.groupby(['iso3', 'year']).size()
    assert not duplicates.gt(1).any()

    # Required coverage
    countries_per_year = df.groupby('year')['iso3'].nunique()
    assert countries_per_year.min() >= 100

def test_treatment_assignment():
    """Test treatment variable construction"""
    df = pd.read_csv('data/processed/panel_data.csv')

    # Treatment year should be monotonic within country
    treatment_years = df.groupby('iso3')['treatment_year'].apply(lambda x: x.nunique() == 1)
    assert treatment_years.all()
```

## Package Integration Patterns

### Combined Workflow Example
```python
# Complete analysis pipeline combining packages

import pandas as pd
import pyfixest as pf
from differences import ATTgt
from pysyncon import Synth
import matplotlib.pyplot as plt

# 1. Data preparation with pandas
df = pd.read_csv('data/processed/panel_data.csv')
df = df.sort_values(['iso3', 'year'])

# 2. Main specification with pyfixest
main_spec = pf.feols(
    'mortality_rate ~ mpower_total + gdp_log + urban_pct | iso3 + year',
    data=df,
    vcov='CL1(iso3)'
)

# 3. Staggered DiD with differences
att_gt = ATTgt(data=df, cohort_name='treatment_year')
att_gt.fit(formula='mortality_rate ~ gdp_log + urban_pct')
cs_results = att_gt.aggregate('event')

# 4. Synthetic control for case study
synth = Synth(data=df_case_study, unit='iso3', time='year',
              treatment='treated', outcome='mortality_rate')
synth.fit(predictors=['gdp_log', 'urban_pct'])

# 5. Results compilation
results_summary = {
    'main_effect': main_spec.summary(),
    'event_study': cs_results,
    'synthetic_control': synth.summary()
}
```

## Installation and Environment

### Complete Requirements File
```txt
# Core analysis
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0

# Econometrics
differences>=0.2.0
pyfixest>=0.18.0
pysyncon>=1.5.0
linearmodels>=5.0
statsmodels>=0.14.0

# Data processing
duckdb>=0.9.0
openpyxl>=3.1.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0

# Testing and quality
pytest>=7.3.0
black>=23.0.0
ruff>=0.1.0

# Optional utilities
jupyter>=1.0.0
python-dotenv>=1.0.0
```

### Environment Setup with uv
```bash
# Create environment
uv venv
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows

# Install dependencies
uv pip install -r requirements.txt

# Development dependencies
uv pip install black ruff pytest pre-commit
```
