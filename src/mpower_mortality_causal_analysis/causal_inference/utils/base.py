"""Base classes and utilities for causal inference methods.

This module provides common interfaces and utilities shared across
different causal inference implementations.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from pandas import DataFrame


class CausalInferenceBase(ABC):
    """Abstract base class for causal inference methods.

    Provides common interface and utilities for different causal inference
    estimators in the MPOWER analysis framework.
    """

    def __init__(self, data: DataFrame):
        """Initialize base causal inference estimator.

        Args:
            data (DataFrame): Panel data for analysis
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")

        if data.empty:
            raise ValueError("Data cannot be empty")

        self.data = data.copy()
        self._fitted = False

    def validate_panel_structure(
        self, unit_col: str, time_col: str, require_balanced: bool = False
    ) -> dict[str, Any]:
        """Validate panel data structure.

        Args:
            unit_col (str): Column name for unit identifier
            time_col (str): Column name for time identifier
            require_balanced (bool): Whether to require balanced panel

        Returns:
            Dict with validation results and panel statistics
        """
        if unit_col not in self.data.columns:
            raise ValueError(f"Unit column '{unit_col}' not found in data")
        if time_col not in self.data.columns:
            raise ValueError(f"Time column '{time_col}' not found in data")

        # Panel statistics
        n_units = self.data[unit_col].nunique()
        n_periods = self.data[time_col].nunique()
        n_obs = len(self.data)

        # Check for balanced panel
        obs_per_unit = self.data.groupby(unit_col)[time_col].count()
        is_balanced = obs_per_unit.nunique() == 1

        if require_balanced and not is_balanced:
            raise ValueError("Balanced panel required but data is unbalanced")

        # Check for missing periods
        expected_obs = n_units * n_periods
        missing_obs = expected_obs - n_obs

        return {
            "n_units": n_units,
            "n_periods": n_periods,
            "n_observations": n_obs,
            "is_balanced": is_balanced,
            "missing_observations": missing_obs,
            "balance_ratio": n_obs / expected_obs if expected_obs > 0 else 0,
        }

    def check_treatment_variation(
        self, treatment_col: str, unit_col: str, time_col: str
    ) -> dict[str, Any]:
        """Check treatment variation for identification.

        Args:
            treatment_col (str): Treatment indicator column
            unit_col (str): Unit identifier column
            time_col (str): Time identifier column

        Returns:
            Dict with treatment variation statistics
        """
        if treatment_col not in self.data.columns:
            raise ValueError(f"Treatment column '{treatment_col}' not found in data")

        # Overall treatment statistics
        treated_obs = self.data[treatment_col].sum()
        treatment_rate = treated_obs / len(self.data)

        # Cross-sectional variation
        units_ever_treated = self.data.groupby(unit_col)[treatment_col].max().sum()
        cross_sectional_variation = units_ever_treated / self.data[unit_col].nunique()

        # Time series variation
        periods_with_treatment = self.data.groupby(time_col)[treatment_col].max().sum()
        time_series_variation = periods_with_treatment / self.data[time_col].nunique()

        # Treatment timing variation (for staggered designs)
        treatment_starts = (
            self.data[self.data[treatment_col] == 1].groupby(unit_col)[time_col].min()
        )
        timing_variation = (
            treatment_starts.nunique() > 1 if len(treatment_starts) > 0 else False
        )

        return {
            "treatment_rate": treatment_rate,
            "cross_sectional_variation": cross_sectional_variation,
            "time_series_variation": time_series_variation,
            "timing_variation": timing_variation,
            "n_treated_units": units_ever_treated,
            "n_treatment_periods": periods_with_treatment,
            "treatment_start_periods": treatment_starts.unique().tolist()
            if len(treatment_starts) > 0
            else [],
        }

    @abstractmethod
    def fit(self, *args, **kwargs) -> "CausalInferenceBase":
        """Fit the causal inference model. Must be implemented by subclasses."""

    def create_lag_lead_variables(
        self,
        data: DataFrame,
        var_col: str,
        unit_col: str,
        time_col: str,
        lags: int = 0,
        leads: int = 0,
    ) -> DataFrame:
        """Create lagged and lead variables for a panel dataset.

        Args:
            data (DataFrame): Panel data
            var_col (str): Variable to lag/lead
            unit_col (str): Unit identifier
            time_col (str): Time identifier
            lags (int): Number of lags to create
            leads (int): Number of leads to create

        Returns:
            DataFrame with additional lag/lead columns
        """
        data_with_lags = data.copy()

        # Sort data for proper lagging
        data_with_lags = data_with_lags.sort_values([unit_col, time_col])

        # Create lags
        for lag in range(1, lags + 1):
            lag_col = f"{var_col}_lag{lag}"
            data_with_lags[lag_col] = data_with_lags.groupby(unit_col)[var_col].shift(
                lag
            )

        # Create leads
        for lead in range(1, leads + 1):
            lead_col = f"{var_col}_lead{lead}"
            data_with_lags[lead_col] = data_with_lags.groupby(unit_col)[var_col].shift(
                -lead
            )

        return data_with_lags

    def generate_summary_stats(
        self, variables: list[str], by_treatment: str | None = None
    ) -> DataFrame:
        """Generate summary statistics for key variables.

        Args:
            variables (List[str]): Variables to summarize
            by_treatment (str, optional): Treatment column for group comparison

        Returns:
            DataFrame with summary statistics
        """
        missing_vars = [var for var in variables if var not in self.data.columns]
        if missing_vars:
            raise ValueError(f"Variables not found in data: {missing_vars}")

        if by_treatment:
            summary = (
                self.data.groupby(by_treatment)[variables]
                .agg(["count", "mean", "std", "min", "max"])
                .round(3)
            )
        else:
            summary = (
                self.data[variables]
                .agg(["count", "mean", "std", "min", "max"])
                .round(3)
            )

        return summary

    def export_results(
        self, results: dict[str, Any], filepath: str, format: str = "csv"
    ) -> None:
        """Export results to file.

        Args:
            results (Dict): Results dictionary to export
            filepath (str): Output file path
            format (str): Export format ('csv', 'json', 'excel')
        """
        if format == "json":
            import json

            with open(filepath, "w") as f:
                json.dump(results, f, indent=2, default=str)

        elif format == "csv":
            # Convert results to DataFrame if possible
            if isinstance(results, dict) and any(
                isinstance(v, list | np.ndarray) for v in results.values()
            ):
                df = pd.DataFrame(results)
                df.to_csv(filepath, index=False)
            else:
                # Flatten dictionary
                flattened = pd.DataFrame([results])
                flattened.to_csv(filepath, index=False)

        elif format == "excel":
            if isinstance(results, dict) and any(
                isinstance(v, list | np.ndarray) for v in results.values()
            ):
                df = pd.DataFrame(results)
                df.to_excel(filepath, index=False)
            else:
                flattened = pd.DataFrame([results])
                flattened.to_excel(filepath, index=False)

        else:
            raise ValueError(f"Unsupported export format: {format}")


class ModelValidation:
    """Utilities for model validation and diagnostics."""

    @staticmethod
    def parallel_trends_test(
        data: DataFrame,
        outcome: str,
        treatment_col: str,
        unit_col: str,
        time_col: str,
        pre_periods: int = 3,
    ) -> dict[str, Any]:
        """Test parallel trends assumption using pre-treatment periods.

        Args:
            data (DataFrame): Panel data
            outcome (str): Outcome variable
            treatment_col (str): Treatment indicator
            unit_col (str): Unit identifier
            time_col (str): Time identifier
            pre_periods (int): Number of pre-periods to test

        Returns:
            Dict with test results
        """
        # This would implement a formal parallel trends test
        # For now, returning a placeholder
        return {
            "test_statistic": np.nan,
            "p_value": np.nan,
            "conclusion": "Parallel trends test not implemented",
            "note": "Use visual inspection of pre-treatment trends",
        }

    @staticmethod
    def placebo_test(
        estimator: CausalInferenceBase, outcome: str, placebo_outcomes: list[str]
    ) -> dict[str, Any]:
        """Run placebo tests using outcomes that should not be affected by treatment.

        Args:
            estimator: Fitted causal inference estimator
            outcome (str): Main outcome variable
            placebo_outcomes (List[str]): Outcomes that should not be affected

        Returns:
            Dict with placebo test results
        """
        # Placeholder for placebo test implementation
        return {
            "placebo_outcomes": placebo_outcomes,
            "results": "Placebo test not implemented",
            "note": "Manual implementation required",
        }
