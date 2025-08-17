"""Event Study Analysis Utilities.

This module provides utilities for conducting event study analysis
to test parallel trends and visualize treatment effects over time.
"""

import warnings

from typing import Any

import numpy as np
import pandas as pd

from pandas import DataFrame

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    warnings.warn(
        "Plotting packages not available. Install matplotlib and seaborn for plotting."
    )


class EventStudyAnalysis:
    """Event Study Analysis for testing parallel trends and visualizing treatment effects.

    Creates event time indicators relative to treatment timing and estimates
    leads and lags to examine pre-treatment trends and post-treatment effects.

    Parameters:
        data (DataFrame): Panel data
        unit_col (str): Column name for unit identifier
        time_col (str): Column name for time identifier
        treatment_col (str): Column name for treatment timing (year of treatment or 0 for never-treated)
        never_treated_value (Union[int, float]): Value indicating never-treated units

    Example:
        >>> event_study = EventStudyAnalysis(data=panel_data, unit_col='country',
        ...                                   time_col='year', treatment_col='treatment_year')
        >>> results = event_study.estimate(outcome='mortality_rate', max_lag=5, max_lead=3)
        >>> event_study.plot_event_study(results)
    """

    def __init__(
        self,
        data: DataFrame,
        unit_col: str,
        time_col: str,
        treatment_col: str,
        never_treated_value: int | float = 0,
    ):
        """Initialize Event Study Analysis."""
        self.data = data.copy()
        self.unit_col = unit_col
        self.time_col = time_col
        self.treatment_col = treatment_col
        self.never_treated_value = never_treated_value

        # Validate required columns
        required_cols = [unit_col, time_col, treatment_col]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Create event time variables
        self._create_event_time_variables()

    def _create_event_time_variables(self) -> None:
        """Create event time variables relative to treatment timing."""
        # Create event time for each observation
        treated_mask = self.data[self.treatment_col] != self.never_treated_value

        # For treated units, calculate periods relative to treatment
        self.data["event_time"] = np.nan
        self.data.loc[treated_mask, "event_time"] = (
            self.data.loc[treated_mask, self.time_col]
            - self.data.loc[treated_mask, self.treatment_col]
        )

        # For never-treated units, set a special value (we'll exclude them from event time analysis)
        never_treated_mask = self.data[self.treatment_col] == self.never_treated_value
        self.data.loc[never_treated_mask, "event_time"] = np.inf

    def create_event_time_dummies(
        self,
        max_lag: int = 5,
        max_lead: int = 5,
        reference_period: int = -1,
        exclude_never_treated: bool = False,
    ) -> DataFrame:
        """Create event time dummy variables for regression analysis.

        Args:
            max_lag (int): Maximum number of post-treatment periods
            max_lead (int): Maximum number of pre-treatment periods
            reference_period (int): Reference period to omit (default: -1, one period before treatment)
            exclude_never_treated (bool): Whether to exclude never-treated units

        Returns:
            DataFrame with event time dummy variables added
        """
        data_with_dummies = self.data.copy()

        # Filter data if requested
        if exclude_never_treated:
            data_with_dummies = data_with_dummies[
                data_with_dummies[self.treatment_col] != self.never_treated_value
            ].copy()

        # Create dummy variables for each event time
        event_times = range(-max_lead, max_lag + 1)

        for event_time in event_times:
            if event_time == reference_period:
                continue  # Skip reference period

            dummy_name = f"event_time_{event_time}"

            if event_time < 0:
                # Pre-treatment periods (leads)
                dummy_name = f"event_time_lead_{abs(event_time)}"
            elif event_time > 0:
                # Post-treatment periods (lags)
                dummy_name = f"event_time_lag_{event_time}"
            else:
                # Treatment period
                dummy_name = "event_time_0"

            # Create dummy
            data_with_dummies[dummy_name] = (
                data_with_dummies["event_time"] == event_time
            ).astype(int)

        # Create binned endpoints if we have observations beyond the range
        if not exclude_never_treated:
            # Pre-treatment endpoint
            pre_endpoint = f"event_time_lead_{max_lead}_plus"
            data_with_dummies[pre_endpoint] = (
                (data_with_dummies["event_time"] <= -max_lead)
                & (data_with_dummies["event_time"] != np.inf)
            ).astype(int)

            # Post-treatment endpoint
            post_endpoint = f"event_time_lag_{max_lag}_plus"
            data_with_dummies[post_endpoint] = (
                data_with_dummies["event_time"] >= max_lag
            ).astype(int)

        return data_with_dummies

    def estimate(
        self,
        outcome: str,
        covariates: list[str] | None = None,
        max_lag: int = 5,
        max_lead: int = 5,
        reference_period: int = -1,
        method: str = "fixed_effects",
        cluster_var: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Estimate event study regression.

        Args:
            outcome (str): Outcome variable name
            covariates (List[str], optional): Control variables
            max_lag (int): Maximum post-treatment periods
            max_lead (int): Maximum pre-treatment periods
            reference_period (int): Reference period to omit
            method (str): Estimation method ('fixed_effects', 'ols')
            cluster_var (str, optional): Variable to cluster standard errors on
            **kwargs: Additional arguments for the estimator

        Returns:
            Dict with estimation results
        """
        if outcome not in self.data.columns:
            raise ValueError(f"Outcome variable '{outcome}' not found in data")

        # Create event time dummies
        data_with_dummies = self.create_event_time_dummies(
            max_lag=max_lag, max_lead=max_lead, reference_period=reference_period
        )

        # Identify event time dummy variables
        event_time_vars = [
            col for col in data_with_dummies.columns if col.startswith("event_time_")
        ]

        # Build regression specification
        if covariates:
            missing_covs = [
                cov for cov in covariates if cov not in data_with_dummies.columns
            ]
            if missing_covs:
                raise ValueError(f"Covariates not found in data: {missing_covs}")
            rhs_vars = event_time_vars + covariates
        else:
            rhs_vars = event_time_vars

        if method == "fixed_effects":
            return self._estimate_fixed_effects(
                data_with_dummies, outcome, rhs_vars, cluster_var, **kwargs
            )
        if method == "ols":
            return self._estimate_ols(
                data_with_dummies, outcome, rhs_vars, cluster_var, **kwargs
            )
        raise ValueError(f"Unknown estimation method: {method}")

    def _estimate_fixed_effects(
        self,
        data: DataFrame,
        outcome: str,
        rhs_vars: list[str],
        cluster_var: str | None,
        **kwargs,
    ) -> dict[str, Any]:
        """Estimate event study using fixed effects."""
        try:
            import statsmodels.api as sm

            from statsmodels.stats.sandwich_covariance import cov_cluster

            # Create fixed effects dummies
            unit_dummies = pd.get_dummies(
                data[self.unit_col], prefix="unit", drop_first=True
            )
            time_dummies = pd.get_dummies(
                data[self.time_col], prefix="time", drop_first=True
            )

            # Combine all variables
            X = pd.concat([data[rhs_vars], unit_dummies, time_dummies], axis=1)
            X = sm.add_constant(X)
            y = data[outcome]

            # Estimate model
            if cluster_var and cluster_var in data.columns:
                model = sm.OLS(y, X).fit(
                    cov_type="cluster", cov_kwds={"groups": data[cluster_var]}
                )
            else:
                model = sm.OLS(y, X).fit(cov_type="HC1")  # Robust standard errors

            # Extract event time coefficients
            event_time_coeffs = {}
            event_time_ses = {}
            event_time_pvals = {}

            for var in rhs_vars:
                if var.startswith("event_time_"):
                    if var in model.params.index:
                        event_time_coeffs[var] = model.params[var]
                        event_time_ses[var] = model.bse[var]
                        event_time_pvals[var] = model.pvalues[var]

            return {
                "model": model,
                "event_time_coefficients": event_time_coeffs,
                "event_time_std_errors": event_time_ses,
                "event_time_pvalues": event_time_pvals,
                "method": "fixed_effects",
                "outcome": outcome,
                "r_squared": model.rsquared,
                "n_obs": model.nobs,
            }

        except ImportError:
            raise ImportError(
                "Fixed effects estimation requires statsmodels: pip install statsmodels"
            )

    def _estimate_ols(
        self,
        data: DataFrame,
        outcome: str,
        rhs_vars: list[str],
        cluster_var: str | None,
        **kwargs,
    ) -> dict[str, Any]:
        """Estimate event study using OLS."""
        try:
            import statsmodels.api as sm

            X = data[rhs_vars]
            X = sm.add_constant(X)
            y = data[outcome]

            # Estimate model
            if cluster_var and cluster_var in data.columns:
                model = sm.OLS(y, X).fit(
                    cov_type="cluster", cov_kwds={"groups": data[cluster_var]}
                )
            else:
                model = sm.OLS(y, X).fit(cov_type="HC1")

            # Extract event time coefficients
            event_time_coeffs = {}
            event_time_ses = {}
            event_time_pvals = {}

            for var in rhs_vars:
                if var.startswith("event_time_"):
                    if var in model.params.index:
                        event_time_coeffs[var] = model.params[var]
                        event_time_ses[var] = model.bse[var]
                        event_time_pvals[var] = model.pvalues[var]

            return {
                "model": model,
                "event_time_coefficients": event_time_coeffs,
                "event_time_std_errors": event_time_ses,
                "event_time_pvalues": event_time_pvals,
                "method": "ols",
                "outcome": outcome,
                "r_squared": model.rsquared,
                "n_obs": model.nobs,
            }

        except ImportError:
            raise ImportError(
                "OLS estimation requires statsmodels: pip install statsmodels"
            )

    def plot_event_study(
        self,
        results: dict[str, Any],
        max_lag: int = 5,
        max_lead: int = 5,
        reference_period: int = -1,
        confidence_level: float = 0.95,
        save_path: str | None = None,
        **plot_kwargs,
    ) -> Any:
        """Plot event study results.

        Args:
            results (Dict): Results from estimate() method
            max_lag (int): Maximum post-treatment periods to plot
            max_lead (int): Maximum pre-treatment periods to plot
            reference_period (int): Reference period (shown as 0)
            confidence_level (float): Confidence level for confidence intervals
            save_path (str, optional): Path to save the plot
            **plot_kwargs: Additional plotting arguments

        Returns:
            Matplotlib figure object
        """
        if not PLOTTING_AVAILABLE:
            raise ImportError(
                "Plotting requires matplotlib and seaborn: pip install matplotlib seaborn"
            )

        coeffs = results["event_time_coefficients"]
        ses = results["event_time_std_errors"]

        # Parse event times and coefficients
        event_times = []
        estimates = []
        confidence_intervals = []

        # Calculate critical value for confidence intervals
        alpha = 1 - confidence_level
        z_score = 1.96  # Approximate for large samples

        # Process each event time
        for var_name, coeff in coeffs.items():
            # Parse event time from variable name
            if "lead_" in var_name:
                event_time = -int(var_name.split("lead_")[-1].split("_")[0])
            elif "lag_" in var_name:
                event_time = int(var_name.split("lag_")[-1].split("_")[0])
            elif var_name == "event_time_0":
                event_time = 0
            else:
                continue

            event_times.append(event_time)
            estimates.append(coeff)

            # Calculate confidence interval
            se = ses.get(var_name, 0)
            ci_lower = coeff - z_score * se
            ci_upper = coeff + z_score * se
            confidence_intervals.append((ci_lower, ci_upper))

        # Add reference period (coefficient = 0)
        event_times.append(reference_period)
        estimates.append(0)
        confidence_intervals.append((0, 0))

        # Sort by event time
        sorted_data = sorted(
            zip(event_times, estimates, confidence_intervals, strict=False)
        )
        event_times, estimates, confidence_intervals = zip(*sorted_data, strict=False)

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot estimates
        ax.plot(
            event_times,
            estimates,
            "o-",
            linewidth=2,
            markersize=8,
            color="blue",
            label="Point Estimates",
        )

        # Plot confidence intervals
        ci_lower, ci_upper = zip(*confidence_intervals, strict=False)
        ax.fill_between(
            event_times,
            ci_lower,
            ci_upper,
            alpha=0.3,
            color="blue",
            label=f"{confidence_level * 100:.0f}% Confidence Interval",
        )

        # Add reference line at y=0
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.7)

        # Add treatment time line
        ax.axvline(x=0, color="red", linestyle=":", alpha=0.7, label="Treatment Time")

        # Customize plot
        ax.set_xlabel("Event Time (Periods Relative to Treatment)", fontsize=12)
        ax.set_ylabel(f"Effect on {results['outcome']}", fontsize=12)
        ax.set_title(
            "Event Study: Treatment Effects Over Time", fontsize=14, fontweight="bold"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Set x-axis ticks
        x_ticks = list(range(-max_lead, max_lag + 1))
        ax.set_xticks(x_ticks)

        # Add pre/post treatment shading
        ax.axvspan(-max_lead, 0, alpha=0.1, color="gray", label="Pre-Treatment")
        ax.axvspan(0, max_lag, alpha=0.1, color="orange", label="Post-Treatment")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def test_parallel_trends(
        self, results: dict[str, Any], max_lead: int = 3, alpha: float = 0.05
    ) -> dict[str, Any]:
        """Test parallel trends assumption using pre-treatment coefficients.

        Args:
            results (Dict): Results from estimate() method
            max_lead (int): Number of pre-treatment periods to test
            alpha (float): Significance level for the test

        Returns:
            Dict with test results
        """
        coeffs = results["event_time_coefficients"]
        pvals = results["event_time_pvalues"]

        # Extract pre-treatment coefficients and p-values
        pre_treatment_pvals = []
        pre_treatment_coeffs = []

        for var_name, pval in pvals.items():
            if "lead_" in var_name:
                lead_num = int(var_name.split("lead_")[-1].split("_")[0])
                if lead_num <= max_lead:
                    pre_treatment_pvals.append(pval)
                    pre_treatment_coeffs.append(coeffs[var_name])

        if not pre_treatment_pvals:
            return {
                "test_result": "No pre-treatment periods found",
                "conclusion": "Cannot test parallel trends",
            }

        # Individual significance test
        significant_pre_treatment = sum(p < alpha for p in pre_treatment_pvals)

        # Joint F-test (simplified version)
        # In practice, you'd want to use the actual F-statistic from the regression
        joint_test_pval = min(pre_treatment_pvals) * len(
            pre_treatment_pvals
        )  # Bonferroni correction
        joint_test_significant = joint_test_pval < alpha

        return {
            "n_pre_treatment_periods": len(pre_treatment_pvals),
            "significant_individual_tests": significant_pre_treatment,
            "individual_test_rate": significant_pre_treatment
            / len(pre_treatment_pvals),
            "joint_test_pvalue": joint_test_pval,
            "joint_test_significant": joint_test_significant,
            "conclusion": "Parallel trends violated"
            if joint_test_significant
            else "Parallel trends supported",
            "pre_treatment_coefficients": pre_treatment_coeffs,
            "pre_treatment_pvalues": pre_treatment_pvals,
        }

    def summary_statistics(self) -> DataFrame:
        """Generate summary statistics for the event study setup.

        Returns:
            DataFrame with summary statistics
        """
        # Treatment timing distribution
        treatment_timing = (
            self.data[self.data[self.treatment_col] != self.never_treated_value][
                self.treatment_col
            ]
            .value_counts()
            .sort_index()
        )

        # Event time distribution
        event_time_dist = self.data["event_time"].value_counts().sort_index()

        # Panel structure
        n_units = self.data[self.unit_col].nunique()
        n_periods = self.data[self.time_col].nunique()
        n_treated_units = (
            self.data[self.treatment_col] != self.never_treated_value
        ).sum()
        n_never_treated = (
            self.data[self.treatment_col] == self.never_treated_value
        ).sum()

        summary = {
            "total_units": n_units,
            "total_periods": n_periods,
            "treated_observations": n_treated_units,
            "never_treated_observations": n_never_treated,
            "treatment_years": treatment_timing.index.tolist(),
            "units_per_treatment_year": treatment_timing.values.tolist(),
            "event_time_range": [
                event_time_dist.index.min(),
                event_time_dist.index.max(),
            ],
        }

        return pd.DataFrame([summary])
