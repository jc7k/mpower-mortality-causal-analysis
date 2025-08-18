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
        "Plotting packages not available. Install matplotlib and seaborn for plotting.",
        stacklevel=2,
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
        1 - confidence_level
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
        ses = results["event_time_std_errors"]

        # Extract pre-treatment coefficients and p-values
        pre_treatment_pvals = []
        pre_treatment_coeffs = []
        pre_treatment_ses = []
        pre_treatment_vars = []

        for var_name, pval in pvals.items():
            if "lead_" in var_name:
                lead_num = int(var_name.split("lead_")[-1].split("_")[0])
                if lead_num <= max_lead:
                    pre_treatment_pvals.append(pval)
                    pre_treatment_coeffs.append(coeffs[var_name])
                    pre_treatment_ses.append(ses[var_name])
                    pre_treatment_vars.append(var_name)

        if not pre_treatment_pvals:
            return {
                "test_result": "No pre-treatment periods found",
                "conclusion": "Cannot test parallel trends",
            }

        # Individual significance test
        significant_pre_treatment = sum(p < alpha for p in pre_treatment_pvals)

        # Joint F-test using the actual model
        joint_f_pval = self._conduct_joint_f_test(results, pre_treatment_vars)

        # Linear trend test
        trend_test_results = self._test_linear_pre_trend(
            pre_treatment_coeffs, pre_treatment_ses
        )

        return {
            "n_pre_treatment_periods": len(pre_treatment_pvals),
            "significant_individual_tests": significant_pre_treatment,
            "individual_test_rate": significant_pre_treatment
            / len(pre_treatment_pvals),
            "joint_f_test_pvalue": joint_f_pval,
            "joint_f_test_significant": joint_f_pval < alpha
            if joint_f_pval is not None
            else None,
            "linear_trend_test": trend_test_results,
            "conclusion": self._interpret_parallel_trends_tests(
                joint_f_pval, trend_test_results, alpha
            ),
            "pre_treatment_coefficients": dict(
                zip(pre_treatment_vars, pre_treatment_coeffs, strict=False)
            ),
            "pre_treatment_pvalues": dict(
                zip(pre_treatment_vars, pre_treatment_pvals, strict=False)
            ),
            "pre_treatment_std_errors": dict(
                zip(pre_treatment_vars, pre_treatment_ses, strict=False)
            ),
        }

    def _conduct_joint_f_test(
        self, results: dict[str, Any], pre_treatment_vars: list[str]
    ) -> float | None:
        """Conduct joint F-test for pre-treatment coefficients."""
        try:
            model = results.get("model")
            if model is None:
                return None

            # Create restriction matrix for joint test
            param_names = model.params.index.tolist()

            # Find indices of pre-treatment variables in the model
            restriction_indices = []
            for var in pre_treatment_vars:
                if var in param_names:
                    restriction_indices.append(param_names.index(var))

            if not restriction_indices:
                return None

            # Create restriction matrix (R) where R @ beta = 0
            n_params = len(param_names)
            n_restrictions = len(restriction_indices)
            R = np.zeros((n_restrictions, n_params))

            for i, idx in enumerate(restriction_indices):
                R[i, idx] = 1

            # Conduct F-test
            from scipy import stats

            # Calculate F-statistic
            beta = model.params.values
            cov_matrix = model.cov_params().values

            # F = (R @ beta)' @ (R @ cov @ R')^(-1) @ (R @ beta) / n_restrictions
            R_beta = R @ beta
            R_cov_R = R @ cov_matrix @ R.T

            try:
                R_cov_R_inv = np.linalg.inv(R_cov_R)
                f_stat = (R_beta.T @ R_cov_R_inv @ R_beta) / n_restrictions

                # Calculate p-value
                df_num = n_restrictions
                df_denom = model.df_resid
                return 1 - stats.f.cdf(f_stat, df_num, df_denom)

            except np.linalg.LinAlgError:
                # Singular matrix, fall back to individual tests
                return None

        except Exception:
            return None

    def _test_linear_pre_trend(
        self, coeffs: list[float], ses: list[float]
    ) -> dict[str, Any]:
        """Test for linear pre-treatment trend."""
        if len(coeffs) < 3:
            return {
                "test": "insufficient_data",
                "pvalue": None,
                "significant": None,
                "trend_coefficient": None,
            }

        try:
            # Create event time indices (negative for pre-treatment)
            event_times = np.array(range(-len(coeffs), 0))
            coeffs_array = np.array(coeffs)
            weights = 1 / (np.array(ses) ** 2)  # Inverse variance weighting

            # Weighted linear regression: coeff = alpha + beta * event_time
            X = np.column_stack([np.ones(len(event_times)), event_times])
            W = np.diag(weights)

            # Weighted least squares: (X'WX)^(-1) X'Wy
            XtWX = X.T @ W @ X
            XtWy = X.T @ W @ coeffs_array

            try:
                beta_hat = np.linalg.solve(XtWX, XtWy)
                trend_coeff = beta_hat[1]  # Slope coefficient

                # Calculate standard error of trend coefficient
                cov_matrix = np.linalg.inv(XtWX)
                trend_se = np.sqrt(cov_matrix[1, 1])

                # T-test for trend coefficient
                t_stat = trend_coeff / trend_se

                # Two-tailed p-value
                from scipy import stats

                df = len(coeffs) - 2
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

                return {
                    "test": "linear_trend",
                    "trend_coefficient": trend_coeff,
                    "trend_std_error": trend_se,
                    "t_statistic": t_stat,
                    "pvalue": p_value,
                    "significant": p_value < 0.05,
                    "interpretation": "Significant linear pre-trend"
                    if p_value < 0.05
                    else "No significant linear pre-trend",
                }

            except np.linalg.LinAlgError:
                return {
                    "test": "linear_trend",
                    "pvalue": None,
                    "significant": None,
                    "error": "Singular matrix in trend test",
                }

        except Exception as e:
            return {
                "test": "linear_trend",
                "pvalue": None,
                "significant": None,
                "error": str(e),
            }

    def _interpret_parallel_trends_tests(
        self, joint_f_pval: float | None, trend_test: dict[str, Any], alpha: float
    ) -> str:
        """Interpret the results of parallel trends tests."""
        conclusions = []

        # Joint F-test interpretation
        if joint_f_pval is not None:
            if joint_f_pval < alpha:
                conclusions.append("Joint F-test rejects parallel trends")
            else:
                conclusions.append("Joint F-test supports parallel trends")

        # Linear trend test interpretation
        if trend_test.get("pvalue") is not None:
            if trend_test["significant"]:
                conclusions.append("Significant linear pre-trend detected")
            else:
                conclusions.append("No significant linear pre-trend")

        # Overall conclusion
        if not conclusions:
            return "Insufficient data for parallel trends testing"

        # If any test suggests violation, conclude violation
        violations = any("rejects" in c or "Significant" in c for c in conclusions)

        if violations:
            overall = "Parallel trends assumption likely violated"
        else:
            overall = "Parallel trends assumption supported"

        return f"{overall}. Details: {'; '.join(conclusions)}"

    def comprehensive_parallel_trends_analysis(
        self,
        outcome: str,
        max_lead: int = 4,
        covariates: list[str] | None = None,
        alpha: float = 0.05,
    ) -> dict[str, Any]:
        """Conduct comprehensive parallel trends analysis.

        Args:
            outcome (str): Outcome variable
            max_lead (int): Maximum pre-treatment periods to test
            covariates (List[str], optional): Control variables
            alpha (float): Significance level

        Returns:
            Dict with comprehensive parallel trends analysis
        """
        # Estimate event study
        results = self.estimate(
            outcome=outcome,
            covariates=covariates,
            max_lead=max_lead,
            max_lag=5,
            method="fixed_effects",
            cluster_var=self.unit_col,
        )

        # Conduct parallel trends tests
        pt_tests = self.test_parallel_trends(results, max_lead=max_lead, alpha=alpha)

        # Visual inspection data
        visual_data = self._prepare_visual_inspection_data(outcome, max_lead)

        # Placebo test suggestions
        placebo_suggestions = self._suggest_placebo_tests()

        return {
            "event_study_results": results,
            "parallel_trends_tests": pt_tests,
            "visual_inspection_data": visual_data,
            "placebo_test_suggestions": placebo_suggestions,
            "overall_assessment": self._overall_parallel_trends_assessment(
                pt_tests, alpha
            ),
        }

    def _prepare_visual_inspection_data(
        self, outcome: str, max_lead: int
    ) -> dict[str, Any]:
        """Prepare data for visual inspection of parallel trends."""
        # Calculate outcome means by treatment cohort and time
        cohort_time_means = []

        cohorts = sorted(self.data[self.treatment_col].unique())
        years = sorted(self.data[self.time_col].unique())

        for cohort in cohorts:
            cohort_data = self.data[self.data[self.treatment_col] == cohort]
            yearly_means = cohort_data.groupby(self.time_col)[outcome].mean()

            cohort_info = {
                "cohort": cohort,
                "is_treated": cohort != self.never_treated_value,
                "treatment_year": cohort
                if cohort != self.never_treated_value
                else None,
                "yearly_means": yearly_means.to_dict(),
                "years": yearly_means.index.tolist(),
            }
            cohort_time_means.append(cohort_info)

        return {
            "cohort_time_means": cohort_time_means,
            "outcome_variable": outcome,
            "time_range": [min(years), max(years)],
            "n_cohorts": len(cohorts),
            "suggestion": "Plot these trends to visually inspect parallel pre-treatment trends",
        }

    def _suggest_placebo_tests(self) -> dict[str, Any]:
        """Suggest placebo tests for robustness."""
        # Find treatment years
        treatment_years = sorted(
            [
                year
                for year in self.data[self.treatment_col].unique()
                if year != self.never_treated_value
            ]
        )

        if not treatment_years:
            return {"placebo_tests": [], "message": "No treatment years found"}

        # Suggest artificial treatment dates
        time_range = self.data[self.time_col].unique()
        min_year, _max_year = min(time_range), max(time_range)

        suggestions = []

        # Pre-treatment placebo tests
        for t_year in treatment_years:
            # Suggest 2-3 years before actual treatment
            for offset in [2, 3]:
                placebo_year = t_year - offset
                if placebo_year >= min_year:
                    suggestions.append(
                        {
                            "type": "pre_treatment_placebo",
                            "original_treatment_year": t_year,
                            "placebo_treatment_year": placebo_year,
                            "description": f"Test artificial treatment in {placebo_year} for cohort actually treated in {t_year}",
                        }
                    )

        # Never-treated placebo test
        mid_year = int(np.median(time_range))
        suggestions.append(
            {
                "type": "never_treated_placebo",
                "placebo_treatment_year": mid_year,
                "description": f"Randomly assign never-treated units artificial treatment in {mid_year}",
            }
        )

        return {
            "placebo_tests": suggestions,
            "recommendation": "Conduct these placebo tests to validate the identification strategy",
        }

    def _overall_parallel_trends_assessment(
        self, pt_tests: dict[str, Any], alpha: float
    ) -> dict[str, Any]:
        """Provide overall assessment of parallel trends assumption."""
        # Count evidence for/against parallel trends
        evidence_against = 0
        evidence_for = 0
        total_tests = 0

        # Individual test evidence
        individual_rate = pt_tests.get("individual_test_rate", 0)
        if individual_rate > 0.2:  # More than 20% significant
            evidence_against += 1
        else:
            evidence_for += 1
        total_tests += 1

        # Joint F-test evidence
        joint_pval = pt_tests.get("joint_f_test_pvalue")
        if joint_pval is not None:
            if joint_pval < alpha:
                evidence_against += 1
            else:
                evidence_for += 1
            total_tests += 1

        # Linear trend test evidence
        trend_test = pt_tests.get("linear_trend_test", {})
        trend_sig = trend_test.get("significant")
        if trend_sig is not None:
            if trend_sig:
                evidence_against += 1
            else:
                evidence_for += 1
            total_tests += 1

        # Overall assessment
        if total_tests == 0:
            assessment = "insufficient_data"
            confidence = "none"
        elif evidence_against == 0:
            assessment = "strong_support"
            confidence = "high"
        elif evidence_against < evidence_for:
            assessment = "weak_support"
            confidence = "medium"
        elif evidence_against == evidence_for:
            assessment = "mixed_evidence"
            confidence = "low"
        else:
            assessment = "violation_likely"
            confidence = "medium" if evidence_against > evidence_for else "high"

        return {
            "assessment": assessment,
            "confidence": confidence,
            "evidence_for": evidence_for,
            "evidence_against": evidence_against,
            "total_tests": total_tests,
            "recommendation": self._get_assessment_recommendation(assessment),
        }

    def _get_assessment_recommendation(self, assessment: str) -> str:
        """Get recommendation based on parallel trends assessment."""
        recommendations = {
            "strong_support": (
                "Parallel trends assumption is well-supported. "
                "Proceed with DiD analysis but consider robustness checks."
            ),
            "weak_support": (
                "Parallel trends assumption has some support but consider: "
                "1) Adding more control variables, 2) Restricting sample, "
                "3) Alternative identification strategies."
            ),
            "mixed_evidence": (
                "Mixed evidence on parallel trends. Consider: "
                "1) Investigating which cohorts/periods drive violations, "
                "2) Matching methods, 3) Synthetic controls, 4) Alternative designs."
            ),
            "violation_likely": (
                "Parallel trends assumption likely violated. Consider: "
                "1) Alternative research designs, 2) Matching on observables, "
                "3) Synthetic control methods, 4) Instrumental variables."
            ),
            "insufficient_data": (
                "Insufficient data for reliable parallel trends testing. "
                "Consider expanding the time series or alternative methods."
            ),
        }

        return recommendations.get(assessment, "Unknown assessment category.")

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
