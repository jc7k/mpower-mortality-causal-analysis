"""Callaway & Sant'Anna (2021) Staggered Difference-in-Differences Implementation.

This module provides a wrapper for the Callaway & Sant'Anna DiD estimator,
with fallback implementations when the `differences` package is not available
due to dependency conflicts.
"""

import warnings

from typing import Any, Literal

import numpy as np
import pandas as pd

from pandas import DataFrame

try:
    from differences import ATTgt

    DIFFERENCES_AVAILABLE = True
except ImportError:
    DIFFERENCES_AVAILABLE = False
    warnings.warn(
        "The 'differences' package is not available. "
        "This may be due to dependency conflicts with linearmodels. "
        "Using fallback implementation with limited functionality."
    )

from ..utils.base import CausalInferenceBase


class CallawayDiD(CausalInferenceBase):
    """Callaway & Sant'Anna (2021) Staggered Difference-in-Differences Estimator.

    Implements the staggered DiD method that handles:
    - Multiple treatment periods (staggered adoption)
    - Treatment effect heterogeneity across units and time
    - Negative weighting issues in two-way fixed effects models
    - Proper parallel trends conditioning

    Parameters:
        data (DataFrame): Panel data with required columns
        cohort_col (str): Column name for treatment cohort (year of first treatment, 0 for never-treated)
        unit_col (str): Column name for unit identifier
        time_col (str): Column name for time period
        never_treated_value (Union[int, float]): Value indicating never-treated units (default: 0)

    Example:
        >>> did = CallawayDiD(data=panel_data, cohort_col='treatment_year')
        >>> results = did.fit(outcome='mortality_rate', covariates=['gdp_log'])
        >>> event_study = did.aggregate('event')
    """

    def __init__(
        self,
        data: DataFrame,
        cohort_col: str,
        unit_col: str = "unit_id",
        time_col: str = "year",
        never_treated_value: int | float = 0,
    ):
        """Initialize Callaway DiD estimator."""
        super().__init__(data)

        self.cohort_col = cohort_col
        self.unit_col = unit_col
        self.time_col = time_col
        self.never_treated_value = never_treated_value

        # Validate required columns
        required_cols = [cohort_col, unit_col, time_col]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Store fitted model
        self._fitted_model = None
        self._results = None

        # Data validation
        self._validate_data()

    def _validate_data(self) -> None:
        """Validate data structure for Callaway DiD requirements."""
        # Check for balanced panel (optional but recommended)
        panel_structure = self.data.groupby(self.unit_col)[self.time_col].count()
        if panel_structure.nunique() > 1:
            warnings.warn(
                "Unbalanced panel detected. Callaway & Sant'Anna can handle this, "
                "but ensure missing periods are truly random."
            )

        # Check for sufficient never-treated units
        never_treated_count = (
            self.data[self.cohort_col] == self.never_treated_value
        ).sum()
        if never_treated_count == 0:
            raise ValueError(
                "No never-treated units found. Callaway & Sant'Anna requires "
                "either never-treated or late-treated units for identification."
            )

        # Check treatment timing
        treatment_years = self.data[
            self.data[self.cohort_col] != self.never_treated_value
        ][self.cohort_col].unique()
        time_range = self.data[self.time_col].unique()

        if len(treatment_years) == 0:
            warnings.warn("No treated units found in the data.")

        # Ensure treatment years are within time range
        invalid_treatment_years = [
            year for year in treatment_years if year not in time_range
        ]
        if invalid_treatment_years:
            warnings.warn(
                f"Treatment years {invalid_treatment_years} not found in time range. "
                "This may cause identification issues."
            )

    def fit(
        self, outcome: str, covariates: list[str] | None = None, **kwargs
    ) -> "CallawayDiD":
        """Fit the Callaway & Sant'Anna DiD model.

        Args:
            outcome (str): Outcome variable name
            covariates (List[str], optional): List of covariate column names
            **kwargs: Additional arguments passed to the underlying estimator

        Returns:
            CallawayDiD: Fitted estimator instance

        Raises:
            ValueError: If outcome variable is not in data or if estimation fails
        """
        if outcome not in self.data.columns:
            raise ValueError(f"Outcome variable '{outcome}' not found in data")

        # Prepare formula
        if covariates is None:
            formula = f"{outcome} ~ 1"
        else:
            missing_covs = [cov for cov in covariates if cov not in self.data.columns]
            if missing_covs:
                raise ValueError(f"Covariates not found in data: {missing_covs}")
            formula = f"{outcome} ~ {' + '.join(covariates)}"

        if DIFFERENCES_AVAILABLE:
            try:
                # Use the differences package implementation
                self._fitted_model = ATTgt(
                    data=self.data, cohort_name=self.cohort_col, **kwargs
                )
                self._fitted_model.fit(formula=formula)
                self._results = self._extract_differences_results()

            except Exception as e:
                warnings.warn(f"Failed to fit with differences package: {e}")
                self._fitted_model = self._fit_fallback(outcome, covariates)
        else:
            # Use fallback implementation
            self._fitted_model = self._fit_fallback(outcome, covariates)

        return self

    def _fit_fallback(
        self, outcome: str, covariates: list[str] | None = None
    ) -> dict[str, Any]:
        """Fallback implementation using traditional methods.

        This provides basic DiD functionality when the differences package
        is not available, though with reduced functionality.
        """
        warnings.warn(
            "Using fallback DiD implementation. "
            "For full Callaway & Sant'Anna functionality, resolve the differences package dependency."
        )

        # Create treatment indicator
        data_copy = self.data.copy()
        data_copy["treated"] = (
            data_copy[self.cohort_col] != self.never_treated_value
        ).astype(int)
        data_copy["post"] = 0

        # Create post-treatment indicator for each cohort
        for cohort in data_copy[data_copy["treated"] == 1][self.cohort_col].unique():
            cohort_mask = data_copy[self.cohort_col] == cohort
            time_mask = data_copy[self.time_col] >= cohort
            data_copy.loc[cohort_mask & time_mask, "post"] = 1

        # Create interaction term
        data_copy["treated_post"] = data_copy["treated"] * data_copy["post"]

        # Simple DiD regression (not the full Callaway & Sant'Anna method)
        try:
            import statsmodels.api as sm

            # Prepare features
            features = ["treated", "post", "treated_post"]
            if covariates:
                features.extend(covariates)

            # Add unit and time fixed effects using dummies
            unit_dummies = pd.get_dummies(
                data_copy[self.unit_col], prefix="unit", drop_first=True
            )
            time_dummies = pd.get_dummies(
                data_copy[self.time_col], prefix="time", drop_first=True
            )

            X = pd.concat([data_copy[features], unit_dummies, time_dummies], axis=1)
            X = sm.add_constant(X)
            y = data_copy[outcome]

            # Fit model
            model = sm.OLS(y, X).fit(
                cov_type="cluster", cov_kwds={"groups": data_copy[self.unit_col]}
            )

            return {
                "model": model,
                "type": "fallback_did",
                "att_estimate": model.params.get("treated_post", np.nan),
                "se": model.bse.get("treated_post", np.nan),
                "pvalue": model.pvalues.get("treated_post", np.nan),
            }

        except ImportError:
            raise ImportError(
                "Fallback implementation requires statsmodels. "
                "Please install: pip install statsmodels"
            )

    def _extract_differences_results(self) -> dict[str, Any]:
        """Extract results from fitted differences model."""
        if not self._fitted_model:
            raise ValueError("Model must be fitted first")

        # Extract ATT(g,t) estimates
        try:
            att_gt = self._fitted_model.att_gt
            return {
                "att_gt": att_gt,
                "type": "callaway_santanna",
                "model": self._fitted_model,
            }
        except Exception as e:
            warnings.warn(f"Could not extract results: {e}")
            return {"type": "callaway_santanna", "model": self._fitted_model}

    def aggregate(
        self, method: Literal["event", "simple", "group", "calendar"] = "simple"
    ) -> dict[str, Any]:
        """Aggregate ATT(g,t) estimates using different methods.

        Args:
            method (str): Aggregation method
                - 'event': Event study (effects by periods relative to treatment)
                - 'simple': Overall average treatment effect
                - 'group': Average effect by treatment cohort
                - 'calendar': Average effect by calendar time

        Returns:
            Dict containing aggregated results
        """
        if not self._fitted_model:
            raise ValueError("Model must be fitted first")

        if DIFFERENCES_AVAILABLE and hasattr(self._fitted_model, "aggregate"):
            try:
                return self._fitted_model.aggregate(method)
            except Exception as e:
                warnings.warn(f"Aggregation failed: {e}")
                return self._fallback_aggregate(method)
        else:
            return self._fallback_aggregate(method)

    def _fallback_aggregate(self, method: str) -> dict[str, Any]:
        """Fallback aggregation for when differences package is not available."""
        if not isinstance(self._fitted_model, dict):
            return {"error": "No aggregation available for fallback implementation"}

        if method == "simple":
            return {
                "att": self._fitted_model.get("att_estimate", np.nan),
                "se": self._fitted_model.get("se", np.nan),
                "pvalue": self._fitted_model.get("pvalue", np.nan),
                "method": "simple_did_fallback",
            }
        return {
            "error": f"Aggregation method {method} not available in fallback implementation"
        }

    def summary(self) -> str:
        """Return a summary of the fitted model."""
        if not self._fitted_model:
            return "Model not fitted yet. Call fit() first."

        if DIFFERENCES_AVAILABLE and hasattr(self._fitted_model, "summary"):
            try:
                return str(self._fitted_model.summary())
            except:
                pass

        # Fallback summary
        simple_results = self.aggregate("simple")
        return f"""
Callaway & Sant'Anna DiD Results (Fallback Implementation)
=========================================================
ATT Estimate: {simple_results.get("att", "N/A")}
Standard Error: {simple_results.get("se", "N/A")}
P-value: {simple_results.get("pvalue", "N/A")}

Note: This is a simplified implementation. For full Callaway & Sant'Anna
functionality, please resolve the 'differences' package dependency conflict.
"""

    def plot_event_study(self, **kwargs) -> Any:
        """Plot event study results."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Plotting requires matplotlib: pip install matplotlib")

        event_results = self.aggregate("event")

        if "error" in event_results:
            raise ValueError(f"Cannot plot event study: {event_results['error']}")

        # Implementation depends on the structure of event_results
        # This would need to be adapted based on the actual differences package output
        warnings.warn("Event study plotting not fully implemented in fallback mode")

        return None
