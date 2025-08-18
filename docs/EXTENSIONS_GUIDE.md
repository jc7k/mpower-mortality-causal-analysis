# MPOWER Research Extensions Guide

Comprehensive documentation for the four major research extensions added to the MPOWER mortality causal analysis framework.

## Table of Contents
1. [Advanced Difference-in-Differences](#advanced-difference-in-differences)
2. [Cost-Effectiveness Analysis](#cost-effectiveness-analysis)
3. [Spillover Analysis](#spillover-analysis)
4. [Policy Optimization](#policy-optimization)
5. [Integration with Core Pipeline](#integration-with-core-pipeline)

---

## Advanced Difference-in-Differences

Modern DiD methods that address limitations of traditional two-way fixed effects.

### Available Methods

#### Sun & Abraham (2021)
Handles heterogeneous treatment effects in staggered adoption settings:

```python
from mpower_mortality_causal_analysis.extensions.advanced_did import SunAbraham

# Initialize estimator
sa = SunAbraham(
    data=panel_data,
    unit_col='country',
    time_col='year',
    treatment_col='mpower_high'
)

# Fit model with covariates
results = sa.fit(
    outcome='lung_cancer_mortality_rate',
    covariates=['gdp_per_capita_log', 'urban_population_pct']
)

# Get event study coefficients
event_study = sa.event_study(periods=range(-5, 6))

# Aggregate treatment effects
att = sa.aggregate('simple')  # Overall ATT
group_att = sa.aggregate('group')  # By treatment cohort
```

#### Borusyak et al. (2021)
Imputation-based approach with robust pre-trend testing:

```python
from mpower_mortality_causal_analysis.extensions.advanced_did import BorusyakImputation

bi = BorusyakImputation(data, unit_col='country', time_col='year')
results = bi.fit(outcome='cardiovascular_mortality_rate')

# Test for pre-trends
pre_trends = bi.test_pre_trends(leads=5)

# Get imputed counterfactuals
counterfactuals = bi.get_imputed_outcomes()
```

#### de Chaisemartin & D'Haultfœuille
Handles continuous treatments and fuzzy designs:

```python
from mpower_mortality_causal_analysis.extensions.advanced_did import DCDH

dcdh = DCDH(data, unit_col='country', time_col='year')

# Continuous treatment (MPOWER score)
continuous_effects = dcdh.fit_continuous(
    outcome='copd_mortality_rate',
    treatment='mpower_total_score'
)

# Fuzzy design with compliance
fuzzy_effects = dcdh.fit_fuzzy(
    outcome='ihd_mortality_rate',
    treatment='mpower_high',
    instrument='policy_announcement'
)
```

### Method Comparison Framework

```python
from mpower_mortality_causal_analysis.extensions.advanced_did import MethodComparison

# Compare all available methods
comparison = MethodComparison(data)
results = comparison.compare_methods(
    outcome='lung_cancer_mortality_rate',
    methods=['sun_abraham', 'borusyak', 'dcdh', 'twfe']
)

# Generate comparison report
comparison.create_report('method_comparison.xlsx')

# Diagnostic plots
comparison.plot_estimates()  # Forest plot of estimates
comparison.plot_pre_trends()  # Pre-trend tests across methods
```

---

## Cost-Effectiveness Analysis

Health economic evaluation framework for policy assessment.

### Health Outcomes Modeling

```python
from mpower_mortality_causal_analysis.extensions.cost_effectiveness import HealthOutcomes

# Initialize health outcomes calculator
health = HealthOutcomes(
    mortality_data=data,
    life_tables='who_standard'  # or custom life tables
)

# Calculate QALYs gained from intervention
qalys = health.calculate_qalys(
    baseline_mortality=data['mortality_baseline'],
    intervention_mortality=data['mortality_intervention'],
    discount_rate=0.03
)

# Calculate DALYs averted
dalys = health.calculate_dalys(
    disease='lung_cancer',
    reduction_in_incidence=0.15,
    population=1000000
)

# Markov model for disease progression
markov = health.create_markov_model(
    states=['healthy', 'disease', 'death'],
    transition_matrix=transition_probs,
    time_horizon=30
)
outcomes = markov.simulate(initial_distribution=[0.95, 0.05, 0])
```

### Cost Estimation

```python
from mpower_mortality_causal_analysis.extensions.cost_effectiveness import CostModels

costs = CostModels(currency='USD', year=2023)

# Implementation costs
impl_costs = costs.estimate_implementation(
    intervention='mpower_full',
    country='Brazil',
    population=212000000,
    duration_years=5
)

# Healthcare cost offsets
savings = costs.calculate_healthcare_savings(
    cases_prevented={'lung_cancer': 5000, 'cvd': 12000},
    unit_costs={'lung_cancer': 45000, 'cvd': 25000}
)

# Net costs
net_costs = impl_costs - savings
```

### ICER Analysis

```python
from mpower_mortality_causal_analysis.extensions.cost_effectiveness import ICERAnalysis

icer = ICERAnalysis()

# Calculate ICER
result = icer.calculate(
    costs=net_costs,
    effects=qalys,
    comparator_costs=0,
    comparator_effects=0
)

# Probabilistic sensitivity analysis
psa_results = icer.probabilistic_sensitivity(
    n_simulations=10000,
    cost_distribution='gamma',
    effect_distribution='normal'
)

# Cost-effectiveness acceptability curve
ceac = icer.acceptability_curve(
    willingness_to_pay=range(0, 100000, 1000)
)

# Generate CE plane
icer.plot_ce_plane(psa_results)
```

### Budget Optimization

```python
from mpower_mortality_causal_analysis.extensions.cost_effectiveness import BudgetOptimizer

optimizer = BudgetOptimizer(budget=10000000)

# Add intervention options
optimizer.add_intervention('monitor', cost=1000000, effect=500)
optimizer.add_intervention('protect', cost=2000000, effect=800)
optimizer.add_intervention('offer_help', cost=1500000, effect=600)

# Optimize allocation
optimal = optimizer.optimize(
    objective='maximize_health',
    constraints=['budget', 'capacity']
)

# Portfolio analysis
portfolio = optimizer.efficient_frontier()
```

---

## Spillover Analysis

Cross-country policy externalities and spatial effects.

### Spatial Weight Matrices

```python
from mpower_mortality_causal_analysis.extensions.spillover import SpatialWeights

# Create different types of spatial weights
weights = SpatialWeights(countries=data['country'].unique())

# Geographic contiguity
W_contiguity = weights.create_contiguity_matrix()

# Inverse distance
W_distance = weights.create_distance_matrix(
    coordinates=country_coords,
    decay_function='exponential'
)

# Economic proximity
W_economic = weights.create_economic_matrix(
    gdp_data=gdp_by_country,
    trade_data=bilateral_trade
)

# Combined weights
W_combined = weights.combine_matrices(
    [W_contiguity, W_economic],
    weights=[0.5, 0.5]
)
```

### Spatial Econometric Models

```python
from mpower_mortality_causal_analysis.extensions.spillover import SpatialModels

spatial = SpatialModels(data, weights_matrix=W_combined)

# Spatial Lag Model (SAR)
sar_results = spatial.fit_sar(
    outcome='lung_cancer_mortality_rate',
    treatment='mpower_high',
    covariates=['gdp_per_capita_log']
)

# Spatial Error Model (SEM)
sem_results = spatial.fit_sem(
    outcome='cardiovascular_mortality_rate',
    treatment='mpower_high'
)

# Spatial Durbin Model (SDM) - direct and indirect effects
sdm_results = spatial.fit_sdm(
    outcome='copd_mortality_rate',
    treatment='mpower_high'
)

# Extract effects
direct_effects = sdm_results['direct_effects']
indirect_effects = sdm_results['indirect_effects']  # Spillovers
total_effects = sdm_results['total_effects']
```

### Network Diffusion Analysis

```python
from mpower_mortality_causal_analysis.extensions.spillover import DiffusionAnalysis

diffusion = DiffusionAnalysis(adoption_data=mpower_adoption)

# Estimate contagion model
contagion = diffusion.fit_contagion_model(
    network=trade_network,
    characteristics=['gdp', 'health_system_strength']
)

# Simulate policy spread
simulation = diffusion.simulate_spread(
    initial_adopters=['Uruguay', 'Brazil'],
    time_periods=20,
    contagion_params=contagion
)

# Identify influential countries
influencers = diffusion.identify_key_players(
    centrality_measure='eigenvector'
)
```

### Border Discontinuity Design

```python
from mpower_mortality_causal_analysis.extensions.spillover import BorderAnalysis

borders = BorderAnalysis(
    data=data,
    border_pairs=neighboring_countries
)

# RDD at borders
rdd_results = borders.estimate_discontinuity(
    outcome='smoking_prevalence',
    bandwidth=50  # km from border
)

# Geographic spillovers
spillovers = borders.measure_spillovers(
    treatment_country='Argentina',
    outcome='lung_cancer_mortality_rate',
    distance_bands=[0, 50, 100, 200]  # km
)
```

---

## Policy Optimization

Sequential implementation and interaction effects.

### Interaction Effects Analysis

```python
from mpower_mortality_causal_analysis.extensions.optimization import InteractionAnalysis

interactions = InteractionAnalysis(data)

# Test for synergies between policies
synergies = interactions.test_synergies(
    policies=['monitor', 'protect', 'warn'],
    outcome='lung_cancer_mortality_rate'
)

# Higher-order interactions
three_way = interactions.estimate_three_way(
    policy1='protect',
    policy2='warn',
    policy3='raise_taxes',
    outcome='smoking_prevalence'
)

# Complementarity analysis
complements = interactions.find_complements(
    threshold=0.05  # Significance level
)
```

### Sequential Implementation

```python
from mpower_mortality_causal_analysis.extensions.optimization import SequentialOptimizer

seq_opt = SequentialOptimizer(
    policies=['M', 'P', 'O', 'W', 'E', 'R'],
    effects_matrix=component_effects,
    costs=implementation_costs
)

# Find optimal sequence
optimal_sequence = seq_opt.optimize_sequence(
    budget_per_period=1000000,
    time_horizon=10,
    discount_rate=0.03
)

# With learning effects
sequence_with_learning = seq_opt.optimize_with_learning(
    learning_rate=0.1,
    spillover_matrix=policy_spillovers
)

# Robust sequencing (minimax)
robust_sequence = seq_opt.robust_sequence(
    uncertainty_sets=parameter_ranges
)
```

### Combinatorial Optimization

```python
from mpower_mortality_causal_analysis.extensions.optimization import PolicyScheduler

scheduler = PolicyScheduler(
    policies=mpower_components,
    constraints=implementation_constraints
)

# Optimal scheduling
schedule = scheduler.optimize_schedule(
    objective='minimize_time_to_target',
    target_reduction=0.20  # 20% mortality reduction
)

# With capacity constraints
constrained_schedule = scheduler.schedule_with_capacity(
    max_simultaneous=3,
    implementation_times=policy_durations
)

# Branch and bound for large problems
bb_solution = scheduler.branch_and_bound(
    max_iterations=10000
)
```

### Political Economy Constraints

```python
from mpower_mortality_causal_analysis.extensions.optimization import PoliticalConstraints

political = PoliticalConstraints(
    stakeholder_preferences=preference_data,
    veto_players=['tobacco_industry', 'finance_ministry']
)

# Feasibility analysis
feasibility = political.assess_feasibility(
    policy='raise_taxes',
    country='Brazil'
)

# Coalition building
coalition = political.build_coalition(
    target_policy='comprehensive_mpower',
    available_transfers=1000000
)

# Game-theoretic equilibrium
equilibrium = political.find_equilibrium(
    players=['government', 'industry', 'health_advocates'],
    payoff_matrix=payoffs
)
```

---

## Integration with Core Pipeline

### Using Extensions in Main Pipeline

```python
from mpower_mortality_causal_analysis.analysis import MPOWERAnalysisPipeline

# Initialize pipeline with extensions
pipeline = MPOWERAnalysisPipeline(
    data_path='data/processed/analysis_ready_data.csv',
    enable_extensions=True
)

# Run analysis with specific extensions
results = pipeline.run_full_analysis(
    extensions=['advanced_did', 'spillover'],
    extension_config={
        'advanced_did': {
            'methods': ['sun_abraham', 'borusyak'],
            'run_comparison': True
        },
        'spillover': {
            'weights_type': 'contiguity',
            'models': ['sar', 'sdm']
        }
    }
)

# Access extension results
adv_did_results = results['extensions']['advanced_did']
spillover_results = results['extensions']['spillover']
```

### Custom Workflows

```python
# Combine multiple extensions
from mpower_mortality_causal_analysis.extensions import (
    advanced_did, cost_effectiveness, spillover, optimization
)

# Step 1: Estimate effects with advanced DiD
sa = advanced_did.SunAbraham(data)
effects = sa.fit(outcome='mortality')

# Step 2: Calculate spillovers
spatial = spillover.SpatialModels(data, W)
total_effects = spatial.fit_sdm(outcome='mortality')

# Step 3: Cost-effectiveness with spillovers
health = cost_effectiveness.HealthOutcomes(data)
qalys_direct = health.calculate_qalys(effects['att'])
qalys_indirect = health.calculate_qalys(total_effects['indirect'])

# Step 4: Optimize implementation
opt = optimization.SequentialOptimizer(
    policies=['M', 'P', 'O', 'W', 'E', 'R'],
    effects_matrix=total_effects
)
optimal_path = opt.optimize_sequence()
```

### Performance Considerations

```python
# Parallel processing for large datasets
from mpower_mortality_causal_analysis.extensions.utils import parallel_config

parallel_config.set_n_jobs(8)  # Use 8 cores

# Chunked processing for memory efficiency
pipeline.run_full_analysis(
    chunk_size=1000,
    low_memory=True
)

# GPU acceleration (if available)
spatial.enable_gpu()  # For spatial models
icer.enable_gpu()  # For Monte Carlo simulations
```

---

## API Reference

For detailed API documentation, see:
- `/docs/api/extensions/advanced_did.md`
- `/docs/api/extensions/cost_effectiveness.md`
- `/docs/api/extensions/spillover.md`
- `/docs/api/extensions/optimization.md`

## Example Notebooks

Complete working examples available in:
- `/notebooks/extensions/advanced_did_tutorial.ipynb`
- `/notebooks/extensions/cost_effectiveness_demo.ipynb`
- `/notebooks/extensions/spillover_analysis.ipynb`
- `/notebooks/extensions/policy_optimization.ipynb`

## Citation

If you use these extensions in your research, please cite:

```bibtex
@software{mpower_extensions_2025,
  title={MPOWER Mortality Causal Analysis: Research Extensions},
  author={Contributors},
  year={2025},
  version={2.0.0},
  url={https://github.com/your-repo/mpower-mortality-causal-analysis}
}
```
