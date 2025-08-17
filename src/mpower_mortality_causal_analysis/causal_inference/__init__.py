"""Causal Inference Framework for MPOWER Mortality Analysis.

This module provides implementations of modern causal inference methods
for analyzing the impact of WHO MPOWER tobacco control policies on mortality outcomes.

Key Methods:
- Callaway & Sant'Anna (2021) Staggered Difference-in-Differences
- Synthetic Control Methods
- Panel Data Fixed Effects
- Event Study Analysis
- Robustness and Sensitivity Testing

Example:
    >>> from mpower_mortality_causal_analysis.causal_inference import CallawayDiD
    >>> did = CallawayDiD(data=panel_data, cohort_col='treatment_year')
    >>> results = did.fit(outcome='mortality_rate', covariates=['gdp_log'])
    >>> event_study = did.aggregate('event')
"""

from .data.preparation import MPOWERDataPrep
from .methods.callaway_did import CallawayDiD
from .methods.panel_methods import PanelFixedEffects
from .methods.synthetic_control import SyntheticControl
from .utils.event_study import EventStudyAnalysis
from .utils.robustness import RobustnessTests

__version__ = "0.1.0"

__all__ = [
    "CallawayDiD",
    "SyntheticControl",
    "PanelFixedEffects",
    "EventStudyAnalysis",
    "RobustnessTests",
    "MPOWERDataPrep",
]
