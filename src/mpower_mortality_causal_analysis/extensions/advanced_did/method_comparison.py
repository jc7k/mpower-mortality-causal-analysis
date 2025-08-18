"""Method Comparison Framework for Advanced DiD Estimators.

This module provides a systematic comparison framework for evaluating
different DiD methods on the same dataset.
"""

import time
import warnings

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .borusyak_imputation import BorusyakImputation
from .dcdh_did import DCDHEstimator
from .doubly_robust import DoublyRobustDiD
from .sun_abraham import SunAbrahamEstimator

# Constants
SIGNIFICANCE_LEVEL = 0.05
PERFORMANCE_TIMEOUT = 300  # seconds
CONSISTENCY_THRESHOLD = 0.5  # For method agreement


@dataclass
class MethodResult:
    """Container for method comparison results."""

    method_name: str
    att: float
    se: float
    p_value: float
    computation_time: float
    n_treated: int
    n_control: int
    additional_info: dict[str, Any]


class MethodComparison:
    """Systematic comparison of DiD methods.

    Provides comprehensive comparison of different DiD estimators,
    including performance benchmarks and diagnostic tests.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        unit_col: str = "unit",
        time_col: str = "time",
        cohort_col: str | None = None,
    ):
        """Initialize method comparison framework.

        Args:
            data: Panel data DataFrame
            unit_col: Column name for unit identifier
            time_col: Column name for time period
            cohort_col: Optional column for treatment cohort
        """
        self.data = data.copy()
        self.unit_col = unit_col
        self.time_col = time_col
        self.cohort_col = cohort_col

        # Validate data
        self._validate_data()

        # Store available methods
        self.methods = {
            "sun_abraham": SunAbrahamEstimator,
            "borusyak": BorusyakImputation,
            "dcdh": DCDHEstimator,
            "doubly_robust": DoublyRobustDiD,
        }

        # Results storage
        self.results = {}
        self.comparison_df = None
        self.diagnostics = {}

    def _validate_data(self) -> None:
        """Validate input data structure."""
        required_cols = [self.unit_col, self.time_col]
        if self.cohort_col:
            required_cols.append(self.cohort_col)

        missing_cols = [col for col in required_cols if col not in self.data.columns]

        if missing_cols:
            msg = f"Missing required columns: {missing_cols}"
            raise ValueError(msg)

    def _prepare_treatment_data(
        self, treatment: str, never_treated_value: int | None = None
    ) -> pd.DataFrame:
        """Prepare data with treatment indicators for methods that need it.

        Args:
            treatment: Treatment column name
            never_treated_value: Value indicating never-treated units

        Returns:
            DataFrame with prepared treatment indicators
        """
        prepared_data = self.data.copy()

        # If cohort column exists, create treatment indicator
        if self.cohort_col and self.cohort_col in prepared_data.columns:
            if never_treated_value is None:
                never_treated_value = prepared_data[self.time_col].max() + 1

            # Create binary treatment indicator
            prepared_data["treatment_binary"] = (
                (prepared_data[self.cohort_col] != never_treated_value)
                & (prepared_data[self.time_col] >= prepared_data[self.cohort_col])
            ).astype(int)

            # Create continuous treatment (time since treatment)
            prepared_data["treatment_continuous"] = np.where(
                prepared_data["treatment_binary"] == 1,
                prepared_data[self.time_col] - prepared_data[self.cohort_col],
                0,
            )

        return prepared_data

    def run_sun_abraham(
        self, outcome: str, covariates: list[str] | None = None, **kwargs
    ) -> MethodResult:
        """Run Sun & Abraham (2021) estimator.

        Args:
            outcome: Outcome variable column name
            covariates: Optional list of control variables
            **kwargs: Additional arguments for the estimator

        Returns:
            MethodResult with estimation results
        """
        if not self.cohort_col:
            msg = "Sun & Abraham requires cohort_col to be specified"
            raise ValueError(msg)

        start_time = time.time()

        try:
            # Initialize estimator
            estimator = SunAbrahamEstimator(
                self.data,
                self.cohort_col,
                self.time_col,
                self.unit_col,
                **kwargs
            )

            # Estimate effects
            results = estimator.estimate(outcome, covariates)

            # Extract aggregate results
            if results["aggregate_effects"]:
                att = results["aggregate_effects"]["att"]
                se = results["aggregate_effects"]["se"]
                p_value = results["aggregate_effects"]["p_value"]
                n_cohorts = results["aggregate_effects"]["n_cohorts"]
            else:
                att = np.nan
                se = np.nan
                p_value = np.nan
                n_cohorts = 0

            computation_time = time.time() - start_time

            return MethodResult(
                method_name="Sun & Abraham (2021)",
                att=att,
                se=se,
                p_value=p_value,
                computation_time=computation_time,
                n_treated=len(
                    self.data[
                        self.data[self.cohort_col]
                        != self.data[self.time_col].max() + 1
                    ]
                ),
                n_control=len(
                    self.data[
                        self.data[self.cohort_col]
                        == self.data[self.time_col].max() + 1
                    ]
                ),
                additional_info={
                    "n_cohorts": n_cohorts,
                    "cohort_effects": results.get("cohort_effects", {}),
                },
            )

        except Exception as e:
            warnings.warn(
                f"Sun & Abraham estimation failed: {e}",
                UserWarning,
                stacklevel=2
            )
            return MethodResult(
                method_name="Sun & Abraham (2021)",
                att=np.nan,
                se=np.nan,
                p_value=np.nan,
                computation_time=time.time() - start_time,
                n_treated=0,
                n_control=0,
                additional_info={"error": str(e)},
            )

    def run_borusyak(
        self,
        outcome: str,
        treatment: str,
        covariates: list[str] | None = None,
        **kwargs,
    ) -> MethodResult:
        """Run Borusyak et al. (2021) imputation estimator.

        Args:
            outcome: Outcome variable column name
            treatment: Treatment column name
            covariates: Optional list of control variables
            **kwargs: Additional arguments for the estimator

        Returns:
            MethodResult with estimation results
        """
        start_time = time.time()

        try:
            # Prepare data with treatment indicator
            prepared_data = self._prepare_treatment_data(treatment)

            # Initialize estimator
            estimator = BorusyakImputation(
                prepared_data, self.unit_col, self.time_col, "treatment_binary"
            )

            # Impute counterfactuals
            estimator.impute_counterfactuals(outcome, covariates)

            # Estimate effects
            results = estimator.estimate_effects()

            computation_time = time.time() - start_time

            return MethodResult(
                method_name="Borusyak et al. (2021)",
                att=results["att"],
                se=results["se"],
                p_value=results["p_value"],
                computation_time=computation_time,
                n_treated=results["n_treated"],
                n_control=len(prepared_data) - results["n_treated"],
                additional_info={"t_stat": results.get("t_stat", np.nan)},
            )

        except Exception as e:
            warnings.warn(
                f"Borusyak estimation failed: {e}",
                UserWarning,
                stacklevel=2
            )
            return MethodResult(
                method_name="Borusyak et al. (2021)",
                att=np.nan,
                se=np.nan,
                p_value=np.nan,
                computation_time=time.time() - start_time,
                n_treated=0,
                n_control=0,
                additional_info={"error": str(e)},
            )

    def run_dcdh(
        self,
        outcome: str,
        treatment: str,
        covariates: list[str] | None = None,
        continuous: bool = True,
        **kwargs,
    ) -> MethodResult:
        """Run de Chaisemartin & D'Haultfœuille estimator.

        Args:
            outcome: Outcome variable column name
            treatment: Treatment column name
            covariates: Optional list of control variables
            continuous: Whether to treat as continuous
            **kwargs: Additional arguments for the estimator

        Returns:
            MethodResult with estimation results
        """
        start_time = time.time()

        try:
            # Prepare data
            prepared_data = self._prepare_treatment_data(treatment)

            # Use appropriate treatment variable
            treatment_var = "treatment_continuous" if continuous else "treatment_binary"

            # Initialize estimator
            estimator = DCDHEstimator(prepared_data, self.unit_col, self.time_col)

            # Estimate fuzzy DiD
            results = estimator.fuzzy_did(
                outcome, treatment_var, covariates, continuous=continuous
            )

            computation_time = time.time() - start_time

            return MethodResult(
                method_name="de Chaisemartin & D'Haultfœuille",
                att=results["att"],
                se=results["se"],
                p_value=results["p_value"],
                computation_time=computation_time,
                n_treated=results["n_switchers_in"],
                n_control=results["n_switchers_out"],
                additional_info={
                    "continuous": results["continuous"],
                    "att_switchers_in": results["att_switchers_in"],
                    "att_switchers_out": results["att_switchers_out"],
                },
            )

        except Exception as e:
            warnings.warn(
                f"DCDH estimation failed: {e}", UserWarning, stacklevel=2
            )
            return MethodResult(
                method_name="de Chaisemartin & D'Haultfœuille",
                att=np.nan,
                se=np.nan,
                p_value=np.nan,
                computation_time=time.time() - start_time,
                n_treated=0,
                n_control=0,
                additional_info={"error": str(e)},
            )

    def run_doubly_robust(
        self, outcome: str, treatment: str, covariates: list[str], **kwargs
    ) -> MethodResult:
        """Run doubly robust DiD estimator.

        Args:
            outcome: Outcome variable column name
            treatment: Treatment column name
            covariates: List of control variables
            **kwargs: Additional arguments for the estimator

        Returns:
            MethodResult with estimation results
        """
        start_time = time.time()

        try:
            # Prepare data
            prepared_data = self._prepare_treatment_data(treatment)

            # Initialize estimator
            estimator = DoublyRobustDiD(prepared_data, self.unit_col, self.time_col)

            # Estimate doubly robust ATT
            results = estimator.doubly_robust_att(
                outcome, "treatment_binary", covariates
            )

            computation_time = time.time() - start_time

            return MethodResult(
                method_name="Doubly Robust DiD",
                att=results["dr_att"],
                se=results["se"],
                p_value=results["p_value"],
                computation_time=computation_time,
                n_treated=results["n_treated"],
                n_control=results["n_control"],
                additional_info={
                    "ipw_att": results["ipw_att"],
                    "regression_att": results["regression_att"],
                    "mean_ps_treated": results["mean_propensity_treated"],
                    "mean_ps_control": results["mean_propensity_control"],
                },
            )

        except Exception as e:
            warnings.warn(
                f"Doubly robust estimation failed: {e}", UserWarning, stacklevel=2
            )
            return MethodResult(
                method_name="Doubly Robust DiD",
                att=np.nan,
                se=np.nan,
                p_value=np.nan,
                computation_time=time.time() - start_time,
                n_treated=0,
                n_control=0,
                additional_info={"error": str(e)},
            )

    def run_all_methods(
        self,
        outcome: str,
        treatment: str = None,
        covariates: list[str] | None = None,
        methods_to_run: list[str] | None = None,
    ) -> pd.DataFrame:
        """Run all specified DiD methods.

        Args:
            outcome: Outcome variable column name
            treatment: Treatment column name (or use cohort_col)
            covariates: Optional list of control variables
            methods_to_run: List of methods to run (default: all)

        Returns:
            DataFrame with comparison results
        """
        if methods_to_run is None:
            methods_to_run = list(self.methods.keys())

        # Use cohort column as treatment if not specified
        if treatment is None and self.cohort_col:
            treatment = self.cohort_col

        results = []

        # Run each method
        for method_name in methods_to_run:
            if method_name == "sun_abraham":
                result = self.run_sun_abraham(outcome, covariates)
            elif method_name == "borusyak":
                result = self.run_borusyak(outcome, treatment, covariates)
            elif method_name == "dcdh":
                result = self.run_dcdh(outcome, treatment, covariates)
            elif method_name == "doubly_robust":
                if covariates:
                    result = self.run_doubly_robust(outcome, treatment, covariates)
                else:
                    warnings.warn(
                        "Doubly robust requires covariates, skipping",
                        UserWarning,
                        stacklevel=2
                    )
                    continue
            else:
                warnings.warn(
                    f"Unknown method: {method_name}",
                    UserWarning,
                    stacklevel=2
                )
                continue

            self.results[method_name] = result

            results.append(
                {
                    "method": result.method_name,
                    "att": result.att,
                    "se": result.se,
                    "ci_lower": result.att - 1.96 * result.se
                    if not np.isnan(result.se)
                    else np.nan,
                    "ci_upper": result.att + 1.96 * result.se
                    if not np.isnan(result.se)
                    else np.nan,
                    "p_value": result.p_value,
                    "significant": result.p_value < SIGNIFICANCE_LEVEL
                    if not np.isnan(result.p_value)
                    else False,
                    "computation_time": result.computation_time,
                    "n_treated": result.n_treated,
                    "n_control": result.n_control,
                }
            )

        self.comparison_df = pd.DataFrame(results)
        return self.comparison_df

    def diagnostic_suite(
        self, outcome: str, treatment: str = None, covariates: list[str] | None = None
    ) -> dict:
        """Run comprehensive diagnostic tests.

        Args:
            outcome: Outcome variable column name
            treatment: Treatment column name
            covariates: Optional list of control variables

        Returns:
            Dictionary with diagnostic test results
        """
        diagnostics = {}

        # 1. Check for sufficient variation
        if treatment:
            treatment_var = (
                self.data[treatment].var() if treatment in self.data.columns else 0
            )
            diagnostics["treatment_variation"] = {
                "variance": treatment_var,
                "sufficient": treatment_var > 0,
            }

        # 2. Check for common support
        if self.cohort_col:
            treated_periods = self.data[
                self.data[self.cohort_col] != self.data[self.time_col].max() + 1
            ][self.time_col].unique()
            control_periods = self.data[
                self.data[self.cohort_col] == self.data[self.time_col].max() + 1
            ][self.time_col].unique()

            common_periods = set(treated_periods) & set(control_periods)
            diagnostics["common_support"] = {
                "n_common_periods": len(common_periods),
                "pct_overlap": len(common_periods)
                / len(self.data[self.time_col].unique()),
            }

        # 3. Test for heterogeneous treatment timing
        if self.cohort_col:
            cohorts = self.data[self.cohort_col].unique()
            treated_cohorts = [
                c for c in cohorts if c != self.data[self.time_col].max() + 1
            ]
            diagnostics["treatment_timing"] = {
                "n_cohorts": len(treated_cohorts),
                "staggered": len(treated_cohorts) > 1,
                "cohort_years": sorted(treated_cohorts),
            }

        # 4. Sample size checks
        diagnostics["sample_size"] = {
            "total_obs": len(self.data),
            "n_units": self.data[self.unit_col].nunique(),
            "n_periods": self.data[self.time_col].nunique(),
            "balanced_panel": len(self.data)
            == (
                self.data[self.unit_col].nunique() * self.data[self.time_col].nunique()
            ),
        }

        # 5. Method agreement (if results available)
        if self.comparison_df is not None and len(self.comparison_df) > 1:
            atts = self.comparison_df["att"].dropna()
            if len(atts) > 1:
                diagnostics["method_agreement"] = {
                    "mean_att": atts.mean(),
                    "std_att": atts.std(),
                    "coef_variation": atts.std() / abs(atts.mean())
                    if atts.mean() != 0
                    else np.inf,
                    "range": atts.max() - atts.min(),
                    "consistent": atts.std() / abs(atts.mean()) < CONSISTENCY_THRESHOLD
                    if atts.mean() != 0
                    else False,
                }

        self.diagnostics = diagnostics
        return diagnostics

    def plot_comparison(self, save_path: str | None = None) -> None:
        """Create visualization of method comparison results.

        Args:
            save_path: Optional path to save figure
        """
        if self.comparison_df is None or len(self.comparison_df) == 0:
            warnings.warn(
                "No results to plot. Run run_all_methods first.",
                UserWarning,
                stacklevel=2
            )
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: ATT estimates with confidence intervals
        df_sorted = self.comparison_df.sort_values("att")
        x_pos = np.arange(len(df_sorted))

        axes[0, 0].errorbar(
            x_pos,
            df_sorted["att"],
            yerr=1.96 * df_sorted["se"],
            fmt="o",
            capsize=5,
            capthick=2,
        )
        axes[0, 0].axhline(0, color="red", linestyle="--", alpha=0.5)
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(df_sorted["method"], rotation=45, ha="right")
        axes[0, 0].set_ylabel("ATT Estimate")
        axes[0, 0].set_title("Treatment Effect Estimates by Method")
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: P-values
        axes[0, 1].bar(x_pos, df_sorted["p_value"])
        axes[0, 1].axhline(
            SIGNIFICANCE_LEVEL,
            color="red",
            linestyle="--",
            label=f"α = {SIGNIFICANCE_LEVEL}",
        )
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(df_sorted["method"], rotation=45, ha="right")
        axes[0, 1].set_ylabel("P-value")
        axes[0, 1].set_title("Statistical Significance")
        axes[0, 1].legend()

        # Plot 3: Computation time
        axes[1, 0].bar(x_pos, df_sorted["computation_time"])
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(df_sorted["method"], rotation=45, ha="right")
        axes[1, 0].set_ylabel("Time (seconds)")
        axes[1, 0].set_title("Computational Performance")

        # Plot 4: Sample sizes
        width = 0.35
        axes[1, 1].bar(
            x_pos - width / 2, df_sorted["n_treated"], width, label="Treated", alpha=0.7
        )
        axes[1, 1].bar(
            x_pos + width / 2, df_sorted["n_control"], width, label="Control", alpha=0.7
        )
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(df_sorted["method"], rotation=45, ha="right")
        axes[1, 1].set_ylabel("Number of Units")
        axes[1, 1].set_title("Sample Sizes")
        axes[1, 1].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def generate_report(self) -> str:
        """Generate text report of comparison results.

        Returns:
            String with formatted report
        """
        report = ["=" * 60]
        report.append("ADVANCED DID METHODS COMPARISON REPORT")
        report.append("=" * 60)
        report.append("")

        if self.comparison_df is not None and len(self.comparison_df) > 0:
            report.append("ESTIMATION RESULTS:")
            report.append("-" * 40)

            for _, row in self.comparison_df.iterrows():
                report.append(f"\n{row['method']}:")
                report.append(f"  ATT: {row['att']:.4f} (SE: {row['se']:.4f})")
                report.append(
                    f"  95% CI: [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]"
                )
                report.append(f"  P-value: {row['p_value']:.4f}")
                report.append(f"  Significant: {'Yes' if row['significant'] else 'No'}")
                report.append(f"  Computation time: {row['computation_time']:.2f}s")

        if self.diagnostics:
            report.append("\n" + "=" * 40)
            report.append("DIAGNOSTIC TESTS:")
            report.append("-" * 40)

            for test_name, test_results in self.diagnostics.items():
                report.append(f"\n{test_name.replace('_', ' ').title()}:")
                for key, value in test_results.items():
                    report.append(f"  {key}: {value}")

        report.append("\n" + "=" * 60)

        return "\n".join(report)
