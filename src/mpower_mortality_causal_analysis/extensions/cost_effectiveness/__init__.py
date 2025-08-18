"""Cost-Effectiveness Framework for MPOWER Policy Analysis.

This module provides comprehensive economic evaluation tools for quantifying
the return on investment of MPOWER tobacco control policies across different
country contexts.

Key Components:
    - HealthOutcomeModel: QALY/DALY calculations and disease modeling
    - CostEstimator: Implementation and healthcare cost estimation
    - ICERAnalysis: Incremental cost-effectiveness ratio analysis
    - BudgetOptimizer: Resource allocation optimization
    - CEPipeline: Main orchestration for cost-effectiveness analysis
    - CEReporting: Standardized reporting and visualization

Example:
    >>> from mpower_mortality_causal_analysis.extensions.cost_effectiveness import CEPipeline
    >>> pipeline = CEPipeline(mortality_data, cost_data)
    >>> results = pipeline.run_analysis(country='Brazil')
    >>> pipeline.generate_report('output/brazil_ce_report.pdf')
"""

from .budget_optimizer import BudgetOptimizer
from .ce_pipeline import CEPipeline
from .ce_reporting import CEReporting
from .cost_models import CostEstimator
from .health_outcomes import HealthOutcomeModel
from .icer_analysis import ICERAnalysis

__all__ = [
    "HealthOutcomeModel",
    "CostEstimator",
    "ICERAnalysis",
    "BudgetOptimizer",
    "CEPipeline",
    "CEReporting",
]

__version__ = "0.1.0"
