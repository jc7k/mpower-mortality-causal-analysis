# Causal Inference Methods for MPOWER Analysis

## Callaway & Sant'Anna (2021) Staggered Difference-in-Differences

### Overview
The Callaway & Sant'Anna method extends traditional DiD to handle:
- Multiple treatment periods (staggered adoption)
- Treatment effect heterogeneity across units and time
- Negative weighting issues in two-way fixed effects models
- Proper parallel trends conditioning

### Key Methodological Features

#### Group-Time Average Treatment Effects (ATT_gt)
- Primary parameter of interest: ATT(g,t)
- g = group (year of treatment adoption)
- t = time period
- Allows for heterogeneous treatment effects

#### Identification Strategy
```
Y_it(0) = μ_i + λ_t + ε_it  (potential outcome without treatment)
```
- Parallel trends: E[Y_it(0) - Y_i,t-1(0) | G_i = g] = E[Y_it(0) - Y_i,t-1(0) | G_i = ∞]
- Can condition on covariates to strengthen assumption

### Python Implementation: `differences` Package

#### Installation
```bash
pip install differences
```

#### Basic Usage Pattern
```python
from differences import ATTgt, simulate_data

# Load panel data with required columns:
# - cohort: treatment adoption year (or 0 for never-treated)
# - time: time period
# - outcome variable
# - unit identifier

att_gt = ATTgt(data=df, cohort_name='cohort')
att_gt.fit(formula='outcome_var ~ 1')  # Can include covariates

# Aggregate results
event_study = att_gt.aggregate('event')
overall_att = att_gt.aggregate('simple')
```

#### Critical Implementation Details
- **Cohort Variable**: Year of first treatment (0 for never-treated units)
- **Balance Requirement**: Need sufficient pre-treatment periods for each cohort
- **Never-Treated Units**: Essential for identification (comparison group)
- **Standard Errors**: Clustered at unit level by default

#### Aggregation Options
1. **Event Study**: Effects by periods relative to treatment
2. **Simple**: Overall average treatment effect
3. **Group**: Average effect by treatment cohort
4. **Calendar**: Average effect by calendar time

### Application to MPOWER

#### Treatment Definition Options
1. **Binary**: Countries achieving "high" MPOWER score (≥25/37)
2. **Continuous**: Changes in MPOWER component scores
3. **Multiple Treatments**: Separate analysis for each MPOWER component

#### Data Structure Requirements
```python
# Required panel structure:
mpower_panel = pd.DataFrame({
    'country': [...],           # Country identifier
    'year': [...],             # 2008-2019
    'cohort': [...],           # Year country achieved high MPOWER (0 if never)
    'mpower_total': [...],     # Policy score
    'mortality_rate': [...],   # Outcome variable
    'gdp_log': [...]          # Control variables
})
```

## Synthetic Control Methods

### Overview
Constructs synthetic counterfactual using weighted combination of control units that best matches pre-treatment characteristics of treated unit.

### Python Implementation: `pysyncon`

#### Installation
```bash
pip install pysyncon
```

#### Key Methods Available
1. **Standard Synthetic Control** (`Synth`)
2. **Robust Synthetic Control** (`RobustSynth`)
3. **Augmented Synthetic Control** (`AugSynth`)
4. **Penalized Synthetic Control** (`PenSynth`)

#### Usage Pattern for MPOWER
```python
from pysyncon import Synth

# For country-specific analysis (e.g., Uruguay MPOWER pioneer)
synth = Synth(data=df, unit='country', time='year',
              treatment='treatment_year', outcome='mortality_rate')

# Fit synthetic control
synth.fit(predictors=['gdp_log', 'urban_pct'],
          predictors_op='mean')

# Generate results
results = synth.summary()
synth.plot()  # Visualize treatment effect
```

### When to Use Each Method

#### Callaway & Sant'Anna DiD
- **Best for**: Multiple countries adopting MPOWER at different times
- **Strength**: Handles staggered adoption naturally
- **Requirement**: Sufficient never-treated or late-treated units

#### Synthetic Control
- **Best for**: Case studies of early/strong MPOWER adopters
- **Strength**: Transparent weighting, robust to model misspecification
- **Requirement**: Rich set of potential control units

## Panel Data Fixed Effects

### PyFixest for High-Dimensional Fixed Effects

#### Installation
```bash
pip install pyfixest
```

#### Syntax (mirrors R fixest)
```python
import pyfixest as pf

# Two-way fixed effects with clustered SE
mod = pf.feols('mortality_rate ~ mpower_total + gdp_log | country + year',
               data=df, vcov='CL1(country)')

# Multiple outcomes
mod_multi = pf.feols(['lung_cancer', 'cvd_mortality'] ~
                     'mpower_total + gdp_log | country + year',
                     data=df, vcov='CL1(country)')
```

### Alternative: LinearModels

#### For Advanced Panel Methods
```python
from linearmodels import PanelOLS
from linearmodels import FamaMacBeth

# Entity and time effects
mod = PanelOLS(dependent=df['mortality_rate'],
               exog=df[['mpower_total', 'gdp_log']],
               entity_effects=True,
               time_effects=True)
results = mod.fit(cov_type='clustered', cluster_entity=True)
```

## Robustness and Sensitivity

### Event Study Specification
```python
# Create event time indicators
def create_event_time_dummies(df, treatment_var, time_var, max_lag=5, max_lead=5):
    """Create leads and lags relative to treatment timing"""
    # Implementation for event study plots

# Event study with multiple lags/leads
event_formula = ('mortality_rate ~ ' +
                ' + '.join([f'event_time_{i}' for i in range(-max_lead, max_lag+1)]) +
                '| country + year')
```

### Alternative Estimators
1. **First Differences**: For non-stationary series
2. **GMM**: For dynamic panel bias
3. **Random Effects**: If fixed effects too restrictive

## Common Pitfalls and Solutions

### Data Requirements
- **Balanced vs Unbalanced**: CS DiD handles unbalanced panels
- **Treatment Timing**: Must be clearly defined
- **Never-Treated Units**: Essential for identification

### Diagnostic Tests
1. **Pre-trends Testing**: Built into Callaway & Sant'Anna
2. **Placebo Tests**: Use outcomes unaffected by treatment
3. **Sensitivity to Sample**: Exclude specific countries/years

### Implementation Gotchas
- **Memory Management**: Large panels may require chunking
- **Convergence**: Multiple optimization algorithms available
- **Standard Errors**: Always cluster at treatment unit level
