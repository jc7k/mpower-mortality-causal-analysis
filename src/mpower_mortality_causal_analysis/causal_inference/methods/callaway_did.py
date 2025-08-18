"""Callaway & Sant'Anna (2021) Staggered Difference-in-Differences Implementation.

This module provides a comprehensive implementation of the Callaway & Sant'Anna
staggered DiD estimator using multiple approaches:
1. R's 'did' package via rpy2 (preferred, most robust)
2. Python 'differences' package (if available)
3. Python-only fallback implementation
"""

import warnings

from typing import Any, Literal

import numpy as np
import pandas as pd

from pandas import DataFrame

# Try to import R interface
try:
    from rpy2 import robjects
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr

    pandas2ri.activate()
    R_AVAILABLE = True

    # Try to load R's did package
    try:
        r_did = importr("did")
        r_base = importr("base")
        r_stats = importr("stats")
        R_DID_AVAILABLE = True
    except Exception as e:
        R_DID_AVAILABLE = False
        warnings.warn(f"R 'did' package not available: {e}", stacklevel=2)

except ImportError:
    R_AVAILABLE = False
    R_DID_AVAILABLE = False
    warnings.warn(
        "rpy2 not available. Install with: pip install rpy2. "
        "You'll also need R and the 'did' package: install.packages('did')",
        stacklevel=2,
    )

# Try to import Python differences package
try:
    from differences import ATTgt

    DIFFERENCES_AVAILABLE = True
except ImportError:
    DIFFERENCES_AVAILABLE = False
    warnings.warn(
        "The 'differences' package is not available. "
        "This may be due to dependency conflicts with linearmodels.",
        stacklevel=2,
    )

from ..utils.base import CausalInferenceBase


class CallawayDiD(CausalInferenceBase):
    """Callaway & Sant'Anna (2021) Staggered Difference-in-Differences Estimator.

    Implements the staggered DiD method that handles:
    - Multiple treatment periods (staggered adoption)
    - Treatment effect heterogeneity across units and time
    - Negative weighting issues in two-way fixed effects models
    - Proper parallel trends conditioning
    - Multiple comparison groups and aggregation schemes

    This implementation provides three backends:
    1. R's 'did' package (preferred, most comprehensive)
    2. Python 'differences' package (if available)
    3. Python-only fallback (basic functionality)

    Parameters:
        data (DataFrame): Panel data with required columns
        cohort_col (str): Column name for treatment cohort
            (year of first treatment, 0 for never-treated)
        unit_col (str): Column name for unit identifier
        time_col (str): Column name for time period
        never_treated_value (Union[int, float]): Value indicating
            never-treated units (default: 0)
        prefer_r (bool): Whether to prefer R implementation when available
            (default: True)

    Example:
        >>> # Basic usage
        >>> did = CallawayDiD(data=panel_data, cohort_col='treatment_cohort')
        >>> results = did.fit(outcome='mort_lung_cancer_asr', covariates=['gdp_log'])
        >>> overall_att = did.aggregate('simple')
        >>> event_study = did.aggregate('event')

        >>> # With custom aggregation
        >>> group_effects = did.aggregate('group')
        >>> calendar_effects = did.aggregate('calendar')
    """

    def __init__(
        self,
        data: DataFrame,
        cohort_col: str,
        unit_col: str = "country",
        time_col: str = "year",
        never_treated_value: int | float = 0,
        prefer_r: bool = True,
    ):
        """Initialize Callaway DiD estimator."""
        super().__init__(data)

        self.cohort_col = cohort_col
        self.unit_col = unit_col
        self.time_col = time_col
        self.never_treated_value = never_treated_value
        self.prefer_r = prefer_r

        # Validate required columns
        required_cols = [cohort_col, unit_col, time_col]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Store fitted model and results
        self._fitted_model = None
        self._results = None
        self._backend_used = None
        self._att_gt = None  # Store group-time specific ATTs

        # Data validation and preprocessing
        self._validate_data()
        self._preprocess_data()

    def _validate_data(self) -> None:
        """Validate data structure for Callaway DiD requirements."""
        # Check for balanced panel (optional but recommended)
        panel_structure = self.data.groupby(self.unit_col)[self.time_col].count()
        if panel_structure.nunique() > 1:
            warnings.warn(
                "Unbalanced panel detected. Callaway & Sant'Anna can handle this, "
                "but ensure missing periods are truly random.",
                stacklevel=2,
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
            warnings.warn("No treated units found in the data.", stacklevel=2)

        # Ensure treatment years are within time range
        invalid_treatment_years = [
            year for year in treatment_years if year not in time_range
        ]
        if invalid_treatment_years:
            warnings.warn(
                f"Treatment years {invalid_treatment_years} not found in time range. "
                "This may cause identification issues.",
                stacklevel=2,
            )

    def _preprocess_data(self) -> None:
        """Preprocess data for DiD analysis."""
        # Sort by unit and time
        self.data = self.data.sort_values([self.unit_col, self.time_col]).reset_index(
            drop=True
        )

        # Create treatment indicator
        self.data["treated"] = (
            self.data[self.cohort_col] != self.never_treated_value
        ).astype(int)

        # Create post-treatment indicator for each unit
        self.data["post"] = 0
        for cohort in self.data[self.data["treated"] == 1][self.cohort_col].unique():
            cohort_mask = self.data[self.cohort_col] == cohort
            time_mask = self.data[self.time_col] >= cohort
            self.data.loc[cohort_mask & time_mask, "post"] = 1

    def fit(
        self,
        outcome: str,
        covariates: list[str] | None = None,
        control_group: Literal["nevertreated", "notyettreated"] = "nevertreated",
        anticipation: int = 0,
        base_period: str = "varying",
        **kwargs,
    ) -> "CallawayDiD":
        """Fit the Callaway & Sant'Anna DiD model.

        Args:
            outcome (str): Outcome variable name
            covariates (List[str], optional): List of covariate column names
            control_group (str): Control group type
                - 'nevertreated': Use never-treated units as controls
                - 'notyettreated': Use not-yet-treated units as controls
            anticipation (int): Number of periods before treatment that
                treatment effects might occur
            base_period (str): Base period for normalizing treatment effects
            **kwargs: Additional arguments passed to the underlying estimator

        Returns:
            CallawayDiD: Fitted estimator instance

        Raises:
            ValueError: If outcome variable is not in data or if estimation fails
        """
        if outcome not in self.data.columns:
            raise ValueError(f"Outcome variable '{outcome}' not found in data")

        self.outcome = outcome
        self.covariates = covariates or []

        # Validate covariates
        if self.covariates:
            missing_covs = [
                cov for cov in self.covariates if cov not in self.data.columns
            ]
            if missing_covs:
                raise ValueError(f"Covariates not found in data: {missing_covs}")

        # Try different backends in order of preference
        success = False

        if self.prefer_r and R_DID_AVAILABLE:
            try:
                self._fit_with_r(
                    outcome,
                    covariates,
                    control_group,
                    anticipation,
                    base_period,
                    **kwargs,
                )
                self._backend_used = "R_did"
                success = True
            except Exception as e:
                warnings.warn(
                    f"R implementation failed: {e}. Trying fallback methods.",
                    stacklevel=2,
                )

        if not success and DIFFERENCES_AVAILABLE:
            try:
                self._fit_with_differences(outcome, covariates, **kwargs)
                self._backend_used = "differences"
                success = True
            except Exception as e:
                warnings.warn(
                    f"Differences package failed: {e}. Using Python fallback.",
                    stacklevel=2,
                )

        if not success:
            self._fit_with_python_fallback(outcome, covariates, control_group)
            self._backend_used = "python_fallback"

        return self

    def _fit_with_r(
        self,
        outcome: str,
        covariates: list[str] | None,
        control_group: str,
        anticipation: int,
        base_period: str,
        **kwargs,
    ) -> None:
        """Fit model using R's did package."""
        # Prepare data for R
        r_data = self.data.copy()

        # Create formula
        if covariates:
            formula = f"{outcome} ~ {' + '.join(covariates)}"
        else:
            formula = f"{outcome} ~ 1"

        # Convert data to R dataframe
        r_df = pandas2ri.py2rpy(r_data)

        # Call R's att_gt function
        r_result = r_did.att_gt(
            yname=outcome,
            tname=self.time_col,
            idname=self.unit_col,
            gname=self.cohort_col,
            xformla=robjects.Formula(formula) if covariates else robjects.NULL,
            data=r_df,
            control_group=control_group,
            anticipation=anticipation,
            base_period=base_period,
            **kwargs,
        )

        self._fitted_model = r_result
        self._results = self._extract_r_results(r_result)

    def _fit_with_differences(
        self, outcome: str, covariates: list[str] | None, **kwargs
    ) -> None:
        """Fit model using Python differences package."""
        formula = f"{outcome} ~ 1"
        if covariates:
            formula = f"{outcome} ~ {' + '.join(covariates)}"

        self._fitted_model = ATTgt(
            data=self.data, cohort_name=self.cohort_col, **kwargs
        )
        self._fitted_model.fit(formula=formula)
        self._results = self._extract_differences_results()

    def _fit_with_python_fallback(
        self, outcome: str, covariates: list[str] | None, control_group: str
    ) -> None:
        """Fit model using Python-only fallback implementation."""
        warnings.warn(
            "Using simplified Python fallback. Results may not match full Callaway & Sant'Anna method.",
            stacklevel=2,
        )

        # Create comprehensive ATT(g,t) estimates manually
        att_gt_results = []

        # Get treatment cohorts and time periods
        treatment_cohorts = sorted(
            [
                g
                for g in self.data[self.cohort_col].unique()
                if g != self.never_treated_value
            ]
        )
        time_periods = sorted(self.data[self.time_col].unique())

        # Estimate ATT(g,t) for each cohort-time combination
        for g in treatment_cohorts:
            for t in time_periods:
                if t >= g:  # Only post-treatment periods
                    att_gt = self._estimate_att_gt_python(
                        outcome, g, t, covariates, control_group
                    )
                    att_gt_results.append(
                        {
                            "group": g,
                            "time": t,
                            "att": att_gt["att"],
                            "se": att_gt["se"],
                            "pvalue": att_gt["pvalue"],
                        }
                    )

        self._att_gt = pd.DataFrame(att_gt_results)
        self._fitted_model = {
            "type": "python_fallback",
            "att_gt": self._att_gt,
            "outcome": outcome,
            "covariates": covariates,
        }
        self._results = {"type": "python_fallback", "att_gt": self._att_gt}

    def _estimate_att_gt_python(
        self,
        outcome: str,
        group: int,
        time_period: int,
        covariates: list[str] | None,
        control_group: str,
    ) -> dict[str, float]:
        """Estimate ATT(g,t) for specific group and time period using Python."""
        try:
            import statsmodels.api as sm
        except ImportError:
            raise ImportError(
                "Python fallback requires statsmodels: pip install statsmodels"
            )

        # Filter data for this specific comparison
        data_subset = self.data.copy()

        # Treated group at time t
        treated_mask = (data_subset[self.cohort_col] == group) & (
            data_subset[self.time_col] == time_period
        )

        # Control group definition
        if control_group == "nevertreated":
            control_mask = (
                data_subset[self.cohort_col] == self.never_treated_value
            ) & (data_subset[self.time_col] == time_period)
        else:  # notyettreated
            control_mask = (
                (data_subset[self.cohort_col] > time_period)
                & (data_subset[self.time_col] == time_period)
            ) | (
                (data_subset[self.cohort_col] == self.never_treated_value)
                & (data_subset[self.time_col] == time_period)
            )

        # Get comparison data (treated at time t vs treated at pre-treatment period)
        pre_period = group - 1  # One period before treatment
        if pre_period not in data_subset[self.time_col].values:
            # Use earliest available pre-treatment period
            available_pre = data_subset[
                (data_subset[self.time_col] < group)
                & (data_subset[self.cohort_col] == group)
            ][self.time_col]
            if len(available_pre) > 0:
                pre_period = available_pre.max()
            else:
                return {"att": np.nan, "se": np.nan, "pvalue": np.nan}

        # Get pre-treatment data for treated group
        treated_pre_mask = (data_subset[self.cohort_col] == group) & (
            data_subset[self.time_col] == pre_period
        )

        # Get pre-treatment data for control group
        control_pre_mask = control_mask & (data_subset[self.time_col] == pre_period)

        # Calculate difference-in-differences
        try:
            # Treated group difference (post - pre)
            treated_post = data_subset.loc[treated_mask, outcome].mean()
            treated_pre = data_subset.loc[treated_pre_mask, outcome].mean()
            treated_diff = treated_post - treated_pre

            # Control group difference (post - pre)
            control_post = data_subset.loc[control_mask, outcome].mean()
            control_pre = data_subset.loc[control_pre_mask, outcome].mean()
            control_diff = control_post - control_pre

            # ATT(g,t) estimate
            att = treated_diff - control_diff

            # Simple standard error calculation (not as robust as full CS method)
            treated_post_var = data_subset.loc[treated_mask, outcome].var()
            treated_pre_var = data_subset.loc[treated_pre_mask, outcome].var()
            control_post_var = data_subset.loc[control_mask, outcome].var()
            control_pre_var = data_subset.loc[control_pre_mask, outcome].var()

            n_treated_post = treated_mask.sum()
            n_treated_pre = treated_pre_mask.sum()
            n_control_post = control_mask.sum()
            n_control_pre = control_pre_mask.sum()

            se = np.sqrt(
                treated_post_var / max(n_treated_post, 1)
                + treated_pre_var / max(n_treated_pre, 1)
                + control_post_var / max(n_control_post, 1)
                + control_pre_var / max(n_control_pre, 1)
            )

            # T-statistic and p-value
            t_stat = att / se if se > 0 else 0
            pvalue = 2 * (1 - np.abs(t_stat))  # Simplified p-value

            return {"att": att, "se": se, "pvalue": pvalue}

        except Exception as e:
            warnings.warn(
                f"Failed to estimate ATT({group}, {time_period}): {e}", stacklevel=2
            )
            return {"att": np.nan, "se": np.nan, "pvalue": np.nan}

    def _extract_r_results(self, r_result) -> dict[str, Any]:
        """Extract results from R did package."""
        # Convert R results to Python
        try:
            # Extract ATT(g,t) matrix
            att_gt_matrix = pandas2ri.rpy2py(r_result.rx2("att"))
            se_matrix = pandas2ri.rpy2py(r_result.rx2("se"))
            groups = pandas2ri.rpy2py(r_result.rx2("group"))
            times = pandas2ri.rpy2py(r_result.rx2("t"))

            # Create ATT(g,t) dataframe
            att_gt_df = pd.DataFrame(
                {
                    "group": groups,
                    "time": times,
                    "att": att_gt_matrix,
                    "se": se_matrix,
                    "pvalue": 2
                    * (1 - np.abs(att_gt_matrix / se_matrix)),  # Two-tailed test
                }
            )

            self._att_gt = att_gt_df
            return {
                "type": "callaway_santanna_r",
                "att_gt": att_gt_df,
                "r_result": r_result,
            }
        except Exception as e:
            warnings.warn(f"Could not extract R results: {e}", stacklevel=2)
            return {"type": "callaway_santanna_r", "r_result": r_result}

    def _extract_differences_results(self) -> dict[str, Any]:
        """Extract results from fitted differences model."""
        try:
            att_gt = self._fitted_model.att_gt
            self._att_gt = att_gt
            return {
                "att_gt": att_gt,
                "type": "callaway_santanna_python",
                "model": self._fitted_model,
            }
        except Exception as e:
            warnings.warn(f"Could not extract differences results: {e}", stacklevel=2)
            return {"type": "callaway_santanna_python", "model": self._fitted_model}

    def aggregate(
        self,
        method: Literal["event", "simple", "group", "calendar"] = "simple",
        min_e: int = -np.inf,
        max_e: int = np.inf,
    ) -> dict[str, Any]:
        """Aggregate ATT(g,t) estimates using different methods.

        Args:
            method (str): Aggregation method
                - 'event': Event study (effects by periods relative to treatment)
                - 'simple': Overall average treatment effect
                - 'group': Average effect by treatment cohort
                - 'calendar': Average effect by calendar time
            min_e (int): Minimum event time for event study
            max_e (int): Maximum event time for event study

        Returns:
            Dict containing aggregated results
        """
        if not self._fitted_model:
            raise ValueError("Model must be fitted first")

        if self._backend_used == "R_did":
            return self._aggregate_r_results(method, min_e, max_e)
        if self._backend_used == "differences" and hasattr(
            self._fitted_model, "aggregate"
        ):
            try:
                return self._fitted_model.aggregate(method)
            except Exception as e:
                warnings.warn(f"Differences aggregation failed: {e}", stacklevel=2)
                return self._aggregate_python_fallback(method, min_e, max_e)
        else:
            return self._aggregate_python_fallback(method, min_e, max_e)

    def _aggregate_r_results(
        self, method: str, min_e: int, max_e: int
    ) -> dict[str, Any]:
        """Aggregate results from R did package."""
        try:
            if method == "simple":
                agg_result = r_did.aggte(self._fitted_model, type="simple")
            elif method == "group":
                agg_result = r_did.aggte(self._fitted_model, type="group")
            elif method == "calendar":
                agg_result = r_did.aggte(self._fitted_model, type="calendar")
            elif method == "event":
                agg_result = r_did.aggte(
                    self._fitted_model, type="dynamic", min_e=min_e, max_e=max_e
                )
            else:
                raise ValueError(f"Unknown aggregation method: {method}")

            # Extract aggregated results
            overall_att = pandas2ri.rpy2py(agg_result.rx2("overall.att"))[0]
            overall_se = pandas2ri.rpy2py(agg_result.rx2("overall.se"))[0]

            return {
                "method": method,
                "att": overall_att,
                "se": overall_se,
                "pvalue": 2 * (1 - np.abs(overall_att / overall_se)),
                "backend": "R_did",
                "r_result": agg_result,
            }

        except Exception as e:
            warnings.warn(f"R aggregation failed: {e}", stacklevel=2)
            return self._aggregate_python_fallback(method, min_e, max_e)

    def _aggregate_python_fallback(
        self, method: str, min_e: int, max_e: int
    ) -> dict[str, Any]:
        """Aggregate ATT(g,t) estimates using Python fallback."""
        if self._att_gt is None or len(self._att_gt) == 0:
            return {"error": "No ATT(g,t) estimates available for aggregation"}

        att_gt_df = self._att_gt.copy()

        # Remove missing values
        att_gt_df = att_gt_df.dropna(subset=["att", "se"])

        if len(att_gt_df) == 0:
            return {"error": "No valid ATT(g,t) estimates found"}

        if method == "simple":
            # Simple average of all ATT(g,t)
            weights = 1 / (att_gt_df["se"] ** 2)  # Inverse variance weighting
            weighted_att = np.average(att_gt_df["att"], weights=weights)
            pooled_se = np.sqrt(1 / weights.sum())

            return {
                "method": "simple",
                "att": weighted_att,
                "se": pooled_se,
                "pvalue": 2 * (1 - np.abs(weighted_att / pooled_se)),
                "n_estimates": len(att_gt_df),
                "backend": self._backend_used,
            }

        if method == "group":
            # Average effect by treatment group
            group_results = []
            for group in att_gt_df["group"].unique():
                group_data = att_gt_df[att_gt_df["group"] == group]
                weights = 1 / (group_data["se"] ** 2)
                weighted_att = np.average(group_data["att"], weights=weights)
                pooled_se = np.sqrt(1 / weights.sum())

                group_results.append(
                    {
                        "group": group,
                        "att": weighted_att,
                        "se": pooled_se,
                        "pvalue": 2 * (1 - np.abs(weighted_att / pooled_se)),
                        "n_periods": len(group_data),
                    }
                )

            return {
                "method": "group",
                "results": group_results,
                "backend": self._backend_used,
            }

        if method == "calendar":
            # Average effect by calendar time
            time_results = []
            for time_period in att_gt_df["time"].unique():
                time_data = att_gt_df[att_gt_df["time"] == time_period]
                weights = 1 / (time_data["se"] ** 2)
                weighted_att = np.average(time_data["att"], weights=weights)
                pooled_se = np.sqrt(1 / weights.sum())

                time_results.append(
                    {
                        "time": time_period,
                        "att": weighted_att,
                        "se": pooled_se,
                        "pvalue": 2 * (1 - np.abs(weighted_att / pooled_se)),
                        "n_groups": len(time_data),
                    }
                )

            return {
                "method": "calendar",
                "results": time_results,
                "backend": self._backend_used,
            }

        if method == "event":
            # Event study: effects by event time
            att_gt_df["event_time"] = att_gt_df["time"] - att_gt_df["group"]

            # Filter by event time range
            event_data = att_gt_df[
                (att_gt_df["event_time"] >= min_e) & (att_gt_df["event_time"] <= max_e)
            ]

            event_results = []
            for event_time in sorted(event_data["event_time"].unique()):
                et_data = event_data[event_data["event_time"] == event_time]
                weights = 1 / (et_data["se"] ** 2)
                weighted_att = np.average(et_data["att"], weights=weights)
                pooled_se = np.sqrt(1 / weights.sum())

                event_results.append(
                    {
                        "event_time": event_time,
                        "att": weighted_att,
                        "se": pooled_se,
                        "pvalue": 2 * (1 - np.abs(weighted_att / pooled_se)),
                        "n_estimates": len(et_data),
                    }
                )

            return {
                "method": "event",
                "results": event_results,
                "backend": self._backend_used,
            }

        return {"error": f"Aggregation method '{method}' not supported in fallback"}

    def summary(self) -> str:
        """Return a summary of the fitted model."""
        if not self._fitted_model:
            return "Model not fitted yet. Call fit() first."

        # Try R summary first
        if self._backend_used == "R_did":
            try:
                r_summary = r_base.summary(self._fitted_model)
                return str(pandas2ri.rpy2py(r_summary))
            except:
                pass

        # Try differences package summary
        if self._backend_used == "differences" and hasattr(
            self._fitted_model, "summary"
        ):
            try:
                return str(self._fitted_model.summary())
            except:
                pass

        # Fallback summary
        simple_results = self.aggregate("simple")

        summary_text = f"""
Callaway & Sant'Anna DiD Results ({self._backend_used} backend)
============================================================
Outcome Variable: {getattr(self, "outcome", "Unknown")}
Covariates: {", ".join(getattr(self, "covariates", [])) or "None"}

Overall Average Treatment Effect:
- ATT: {simple_results.get("att", "N/A"):.4f}
- Standard Error: {simple_results.get("se", "N/A"):.4f}
- P-value: {simple_results.get("pvalue", "N/A"):.4f}

Number of ATT(g,t) estimates: {simple_results.get("n_estimates", "N/A")}

Backend: {self._backend_used}
"""

        if self._backend_used == "python_fallback":
            summary_text += """
Note: This uses a simplified Python implementation. For full Callaway & Sant'Anna
functionality, install R with the 'did' package or resolve Python package conflicts.
"""

        return summary_text

    def plot_event_study(
        self, min_e: int = -4, max_e: int = 4, figsize: tuple = (10, 6), **kwargs
    ) -> Any:
        """Plot event study results.

        Args:
            min_e (int): Minimum event time to plot
            max_e (int): Maximum event time to plot
            figsize (tuple): Figure size
            **kwargs: Additional matplotlib arguments
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Plotting requires matplotlib: pip install matplotlib")

        event_results = self.aggregate("event", min_e=min_e, max_e=max_e)

        if "error" in event_results:
            raise ValueError(f"Cannot plot event study: {event_results['error']}")

        results_df = pd.DataFrame(event_results["results"])

        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)

        # Plot point estimates with confidence intervals
        ax.errorbar(
            results_df["event_time"],
            results_df["att"],
            yerr=1.96 * results_df["se"],  # 95% confidence intervals
            fmt="o-",
            capsize=5,
            **kwargs,
        )

        # Add reference line at y=0
        ax.axhline(y=0, color="red", linestyle="--", alpha=0.7)

        # Add reference line at x=0 (treatment timing)
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

        # Labels and title
        ax.set_xlabel("Event Time (Periods Relative to Treatment)")
        ax.set_ylabel("Average Treatment Effect")
        ax.set_title("Event Study: Dynamic Treatment Effects")
        ax.grid(True, alpha=0.3)

        # Add backend info
        ax.text(
            0.02,
            0.98,
            f"Backend: {self._backend_used}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )

        plt.tight_layout()
        return fig

    def get_att_gt(self) -> pd.DataFrame | None:
        """Get the ATT(g,t) estimates dataframe.

        Returns:
            DataFrame with group-time specific treatment effects
        """
        return self._att_gt

    def get_backend_info(self) -> dict[str, Any]:
        """Get information about which backend was used.

        Returns:
            Dict with backend information
        """
        return {
            "backend_used": self._backend_used,
            "r_available": R_AVAILABLE,
            "r_did_available": R_DID_AVAILABLE,
            "differences_available": DIFFERENCES_AVAILABLE,
            "prefer_r": self.prefer_r,
        }
