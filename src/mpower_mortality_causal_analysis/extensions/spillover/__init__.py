"""Spillover Analysis Extension for MPOWER Mortality Causal Analysis.

This module implements spatial econometric methods to analyze cross-country
policy externalities and spillover effects from MPOWER tobacco control policies.
"""

from .spatial_models import SpatialPanelModel
from .spatial_weights import SpatialWeightMatrix
from .spillover_pipeline import SpilloverPipeline

__all__ = ["SpilloverPipeline", "SpatialPanelModel", "SpatialWeightMatrix"]

__version__ = "0.1.0"
