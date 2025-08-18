"""Advanced Difference-in-Differences Methods Extension.

This module implements state-of-the-art DiD methods that address
limitations of traditional approaches, ensuring robust causal inference.
"""

from .borusyak_imputation import BorusyakImputation
from .dcdh_did import DCDHEstimator
from .doubly_robust import DoublyRobustDiD
from .method_comparison import MethodComparison
from .sun_abraham import SunAbrahamEstimator

__all__ = [
    "SunAbrahamEstimator",
    "BorusyakImputation",
    "DCDHEstimator",
    "DoublyRobustDiD",
    "MethodComparison",
]

__version__ = "0.1.0"
