"""de Chaisemartin & D'Haultfœuille DiD Estimator.

This module implements the fuzzy difference-in-differences estimator
for continuous and heterogeneous treatment effects.
"""

import warnings

import numpy as np
import pandas as pd

from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

# Constants
SIGNIFICANCE_LEVEL = 0.05
MIN_SWITCHERS = 10
DEFAULT_PLACEBO_PERIODS = 100
MIN_GROUPS_FOR_HETEROGENEITY = 2
DEFAULT_THRESHOLD = 0.5


class DCDHEstimator:
    """Fuzzy DiD estimator for continuous/heterogeneous treatment.

    Implements de Chaisemartin & D'Haultfœuille approach
    for robust estimation with treatment heterogeneity.
    """

    def __init__(
        self, data: pd.DataFrame, unit_col: str = "unit", time_col: str = "time"
    ):
        """Initialize DCDH estimator.

        Args:
            data: Panel data DataFrame
            unit_col: Column name for unit identifier
            time_col: Column name for time period
        """
        self.data = data.copy()
        self.unit_col = unit_col
        self.time_col = time_col

        # Validate data
        self._validate_data()

        # Store units and periods
        self.units = sorted(data[unit_col].unique())
        self.time_periods = sorted(data[time_col].unique())

        # Results storage
        self.fuzzy_results = None
        self.placebo_results = None
        self.heterogeneity_results = None

    def _validate_data(self) -> None:
        """Validate input data structure."""
        required_cols = [self.unit_col, self.time_col]
        missing_cols = [col for col in required_cols if col not in self.data.columns]

        if missing_cols:
            msg = f"Missing required columns: {missing_cols}"
            raise ValueError(msg)

        # Check for balanced panel
        panel_check = self.data.groupby(self.unit_col)[self.time_col].nunique()
        if panel_check.nunique() > 1:
            warnings.warn(
                "Unbalanced panel detected. Results may be affected.",
                UserWarning,
                stacklevel=2,
            )

    def _identify_switchers(
        self, treatment: str, threshold: float | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Identify units that switch treatment status.

        Args:
            treatment: Treatment variable column name
            threshold: Optional threshold for binary treatment

        Returns:
            Tuple of (switchers_in, switchers_out) DataFrames
        """
        # Create lagged treatment
        self.data["treatment_lag"] = self.data.groupby(self.unit_col)[treatment].shift(
            1
        )

        if threshold is not None:
            # Binary treatment from continuous
            self.data["treatment_binary"] = (self.data[treatment] >= threshold).astype(
                int
            )
            self.data["treatment_binary_lag"] = (
                self.data["treatment_lag"] >= threshold
            ).astype(int)

            # Switchers in (0 to 1)
            switchers_in = self.data[
                (self.data["treatment_binary_lag"] == 0)
                & (self.data["treatment_binary"] == 1)
            ].copy()

            # Switchers out (1 to 0)
            switchers_out = self.data[
                (self.data["treatment_binary_lag"] == 1)
                & (self.data["treatment_binary"] == 0)
            ].copy()
        else:
            # Continuous treatment changes
            self.data["treatment_change"] = (
                self.data[treatment] - self.data["treatment_lag"]
            )

            # Positive switchers (increase in treatment)
            switchers_in = self.data[self.data["treatment_change"] > 0].copy()

            # Negative switchers (decrease in treatment)
            switchers_out = self.data[self.data["treatment_change"] < 0].copy()

        return switchers_in, switchers_out

    def fuzzy_did(
        self,
        outcome: str,
        treatment: str,
        covariates: list[str] | None = None,
        continuous: bool = True,
        weights: str | None = None,
    ) -> dict:
        """Estimate fuzzy DiD with continuous or heterogeneous treatment.

        Args:
            outcome: Outcome variable column name
            treatment: Treatment variable column name
            covariates: Optional list of control variables
            continuous: Whether treatment is continuous
            weights: Optional column name for observation weights

        Returns:
            Dictionary with fuzzy DiD estimates
        """
        if outcome not in self.data.columns:
            msg = f"Outcome column '{outcome}' not found"
            raise ValueError(msg)
        if treatment not in self.data.columns:
            msg = f"Treatment column '{treatment}' not found"
            raise ValueError(msg)

        # Identify switchers
        if continuous:
            switchers_in, switchers_out = self._identify_switchers(treatment)
        else:
            # For binary treatment, use traditional approach
            switchers_in, switchers_out = self._identify_switchers(
                treatment, threshold=0.5
            )

        # Check sufficient switchers
        if len(switchers_in) < MIN_SWITCHERS:
            warnings.warn(
                f"Only {len(switchers_in)} switchers in. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
        if len(switchers_out) < MIN_SWITCHERS:
            warnings.warn(
                f"Only {len(switchers_out)} switchers out. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )

        # Calculate outcome changes
        self.data["outcome_lag"] = self.data.groupby(self.unit_col)[outcome].shift(1)
        self.data["outcome_change"] = self.data[outcome] - self.data["outcome_lag"]

        # Estimate effects for switchers in
        if len(switchers_in) > 0:
            if continuous:
                # Continuous treatment: use treatment change as instrument
                y_in = switchers_in["outcome_change"]
                d_in = switchers_in["treatment_change"]

                if covariates:
                    x_in = switchers_in[covariates]
                    x_in = add_constant(x_in)
                else:
                    x_in = add_constant(pd.DataFrame(index=switchers_in.index))

                # 2SLS estimation
                # First stage: regress treatment change on covariates
                first_stage = OLS(d_in, x_in, missing="drop").fit()
                d_hat = first_stage.fittedvalues

                # Second stage: regress outcome change on fitted treatment
                x_in["d_hat"] = d_hat
                second_stage = OLS(y_in, x_in[["const", "d_hat"]], missing="drop").fit()

                att_in = second_stage.params["d_hat"]
                se_in = second_stage.bse["d_hat"]
            else:
                # Binary treatment: simple difference
                att_in = switchers_in["outcome_change"].mean()
                se_in = switchers_in["outcome_change"].std() / np.sqrt(
                    len(switchers_in)
                )
        else:
            att_in = np.nan
            se_in = np.nan

        # Estimate effects for switchers out
        if len(switchers_out) > 0:
            if continuous:
                y_out = switchers_out["outcome_change"]
                d_out = switchers_out["treatment_change"]

                if covariates:
                    x_out = switchers_out[covariates]
                    x_out = add_constant(x_out)
                else:
                    x_out = add_constant(pd.DataFrame(index=switchers_out.index))

                # 2SLS estimation
                first_stage = OLS(d_out, x_out, missing="drop").fit()
                d_hat = first_stage.fittedvalues

                x_out["d_hat"] = d_hat
                second_stage = OLS(
                    y_out, x_out[["const", "d_hat"]], missing="drop"
                ).fit()

                att_out = second_stage.params["d_hat"]
                se_out = second_stage.bse["d_hat"]
            else:
                att_out = switchers_out["outcome_change"].mean()
                se_out = switchers_out["outcome_change"].std() / np.sqrt(
                    len(switchers_out)
                )
        else:
            att_out = np.nan
            se_out = np.nan

        # Combine estimates (weighted average if both available)
        if not np.isnan(att_in) and not np.isnan(att_out):
            n_in = len(switchers_in)
            n_out = len(switchers_out)
            w_in = n_in / (n_in + n_out)
            w_out = n_out / (n_in + n_out)

            att_combined = w_in * att_in - w_out * att_out
            se_combined = np.sqrt(w_in**2 * se_in**2 + w_out**2 * se_out**2)
        elif not np.isnan(att_in):
            att_combined = att_in
            se_combined = se_in
        elif not np.isnan(att_out):
            att_combined = -att_out
            se_combined = se_out
        else:
            att_combined = np.nan
            se_combined = np.nan

        self.fuzzy_results = {
            "att": att_combined,
            "se": se_combined,
            "t_stat": att_combined / se_combined if se_combined > 0 else np.nan,
            "p_value": 2 * (1 - stats.norm.cdf(np.abs(att_combined / se_combined)))
            if se_combined > 0
            else np.nan,
            "att_switchers_in": att_in,
            "se_switchers_in": se_in,
            "n_switchers_in": len(switchers_in),
            "att_switchers_out": att_out,
            "se_switchers_out": se_out,
            "n_switchers_out": len(switchers_out),
            "continuous": continuous,
        }

        return self.fuzzy_results

    def placebo_tests(
        self,
        outcome: str,
        treatment: str,
        n_placebos: int = DEFAULT_PLACEBO_PERIODS,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """Run placebo tests for inference.

        Args:
            outcome: Outcome variable column name
            treatment: Treatment variable column name
            n_placebos: Number of placebo estimations
            seed: Random seed for reproducibility

        Returns:
            DataFrame with placebo test results
        """
        if seed is not None:
            np.random.seed(seed)

        placebo_results = []

        for i in range(n_placebos):
            # Create placebo treatment by randomly shuffling treatment timing
            placebo_data = self.data.copy()

            # Shuffle treatment within units (preserve panel structure)
            for unit in self.units:
                unit_mask = placebo_data[self.unit_col] == unit
                unit_treatment = placebo_data.loc[unit_mask, treatment].values
                np.random.shuffle(unit_treatment)
                placebo_data.loc[unit_mask, treatment] = unit_treatment

            # Estimate placebo effect
            try:
                placebo_est = DCDHEstimator(placebo_data, self.unit_col, self.time_col)
                placebo_result = placebo_est.fuzzy_did(
                    outcome, treatment, continuous=True
                )

                placebo_results.append(
                    {
                        "iteration": i,
                        "placebo_att": placebo_result["att"],
                        "placebo_se": placebo_result["se"],
                    }
                )
            except Exception:
                placebo_results.append(
                    {"iteration": i, "placebo_att": np.nan, "placebo_se": np.nan}
                )

        self.placebo_results = pd.DataFrame(placebo_results)

        # Calculate p-value based on placebo distribution
        if self.fuzzy_results is not None:
            actual_att = self.fuzzy_results["att"]
            placebo_atts = self.placebo_results["placebo_att"].dropna()

            if len(placebo_atts) > 0:
                # Two-sided p-value
                p_value = np.mean(np.abs(placebo_atts) >= np.abs(actual_att))
                self.fuzzy_results["placebo_p_value"] = p_value

        return self.placebo_results

    def heterogeneity_analysis(
        self, outcome: str, treatment: str, heterogeneity_var: str, n_groups: int = 3
    ) -> dict:
        """Analyze treatment effect heterogeneity.

        Args:
            outcome: Outcome variable column name
            treatment: Treatment variable column name
            heterogeneity_var: Variable to analyze heterogeneity
            n_groups: Number of groups for heterogeneity analysis

        Returns:
            Dictionary with heterogeneous treatment effects
        """
        if heterogeneity_var not in self.data.columns:
            msg = f"Heterogeneity variable '{heterogeneity_var}' not found"
            raise ValueError(msg)

        # Create groups based on heterogeneity variable
        self.data["het_group"] = pd.qcut(
            self.data[heterogeneity_var], q=n_groups, labels=range(n_groups)
        )

        heterogeneity_results = {}

        for group in range(n_groups):
            group_data = self.data[self.data["het_group"] == group].copy()

            if len(group_data) < MIN_SWITCHERS * 2:
                heterogeneity_results[f"group_{group}"] = {
                    "att": np.nan,
                    "se": np.nan,
                    "n_obs": len(group_data),
                    "message": "Insufficient observations",
                }
                continue

            # Estimate group-specific effect
            group_estimator = DCDHEstimator(group_data, self.unit_col, self.time_col)

            try:
                group_result = group_estimator.fuzzy_did(
                    outcome, treatment, continuous=True
                )

                heterogeneity_results[f"group_{group}"] = {
                    "att": group_result["att"],
                    "se": group_result["se"],
                    "p_value": group_result["p_value"],
                    "n_obs": len(group_data),
                    "n_switchers_in": group_result["n_switchers_in"],
                    "n_switchers_out": group_result["n_switchers_out"],
                }
            except Exception as e:
                heterogeneity_results[f"group_{group}"] = {
                    "att": np.nan,
                    "se": np.nan,
                    "n_obs": len(group_data),
                    "message": str(e),
                }

        # Test for heterogeneity
        valid_groups = [
            g
            for g in heterogeneity_results.values()
            if not np.isnan(g.get("att", np.nan))
        ]

        if len(valid_groups) >= MIN_GROUPS_FOR_HETEROGENEITY:
            atts = [g["att"] for g in valid_groups]
            ses = [g["se"] for g in valid_groups]

            # Chi-square test for heterogeneity
            weights = 1 / np.array(ses) ** 2
            weighted_mean = np.average(atts, weights=weights)
            chi_sq = np.sum(((np.array(atts) - weighted_mean) / np.array(ses)) ** 2)
            p_value = 1 - stats.chi2.cdf(chi_sq, df=len(atts) - 1)

            heterogeneity_results["heterogeneity_test"] = {
                "chi_square": chi_sq,
                "p_value": p_value,
                "reject_homogeneity": p_value < SIGNIFICANCE_LEVEL,
            }

        self.heterogeneity_results = heterogeneity_results
        return heterogeneity_results

    def compare_binary_continuous(
        self, outcome: str, treatment: str, threshold: float = DEFAULT_THRESHOLD
    ) -> dict:
        """Compare binary vs continuous treatment specifications.

        Args:
            outcome: Outcome variable column name
            treatment: Treatment variable column name
            threshold: Threshold for binary treatment

        Returns:
            Dictionary comparing binary and continuous estimates
        """
        # Continuous treatment estimate
        continuous_result = self.fuzzy_did(outcome, treatment, continuous=True)

        # Binary treatment estimate
        binary_result = self.fuzzy_did(outcome, treatment, continuous=False)

        # Compare results
        comparison = {
            "continuous": {
                "att": continuous_result["att"],
                "se": continuous_result["se"],
                "p_value": continuous_result["p_value"],
            },
            "binary": {
                "att": binary_result["att"],
                "se": binary_result["se"],
                "p_value": binary_result["p_value"],
            },
        }

        # Test if estimates are significantly different
        if not np.isnan(continuous_result["att"]) and not np.isnan(
            binary_result["att"]
        ):
            diff = continuous_result["att"] - binary_result["att"]
            se_diff = np.sqrt(continuous_result["se"] ** 2 + binary_result["se"] ** 2)
            t_stat = diff / se_diff if se_diff > 0 else np.nan
            p_value = (
                2 * (1 - stats.norm.cdf(np.abs(t_stat))) if se_diff > 0 else np.nan
            )

            comparison["difference"] = {
                "estimate": diff,
                "se": se_diff,
                "t_stat": t_stat,
                "p_value": p_value,
                "significant": p_value < SIGNIFICANCE_LEVEL
                if not np.isnan(p_value)
                else False,
            }

        return comparison
