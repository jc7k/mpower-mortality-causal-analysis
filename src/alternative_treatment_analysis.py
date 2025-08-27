"""Alternative Treatment Definitions Analysis for MPOWER Study.

This module implements and compares different treatment definitions:
1. Binary threshold (≥25 MPOWER score)
2. Continuous change (20% improvement from baseline)
3. Dose-response (continuous MPOWER scores)

The analysis provides robustness checks for the main causal inference results
by testing sensitivity to treatment definition choices.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pandas import DataFrame

# Import causal inference modules
from mpower_mortality_causal_analysis.causal_inference.data.preparation import (
    MPOWERDataPrep,
)
from mpower_mortality_causal_analysis.causal_inference.methods.callaway_did import (
    CallawayDiD,
)

# Constants
MPOWER_THRESHOLD = 25.0
IMPROVEMENT_THRESHOLD = 0.20  # 20% improvement from baseline
SIGNIFICANCE_LEVEL = 0.05


class AlternativeTreatmentAnalysis:
    """Comprehensive analysis comparing alternative treatment definitions.

    This class implements multiple ways to define MPOWER treatment:
    1. Binary threshold approach (original)
    2. Continuous change approach
    3. Dose-response using continuous scores

    Args:
        data_path: Path to analysis-ready dataset
        outcomes: Mortality outcome variables
        unit_col: Country identifier column
        time_col: Year identifier column
        control_vars: Control variables for analysis
    """

    def __init__(
        self,
        data_path: str | Path,
        outcomes: list[str] | None = None,
        unit_col: str = "country_name",
        time_col: str = "year",
        control_vars: list[str] | None = None,
    ):
        """Initialize alternative treatment analysis."""
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        self.unit_col = unit_col
        self.time_col = time_col

        # Default outcome variables
        self.outcomes = outcomes or [
            "mort_lung_cancer_asr",
            "mort_cvd_asr",
            "mort_ihd_asr",
            "mort_copd_asr",
        ]

        # Default control variables
        self.control_vars = control_vars or [
            "gdp_pc_constant_log",
            "urban_pop_pct",
            "population_total",
            "edu_exp_pct_gdp",
        ]

        # Load data
        self.data = self._load_and_validate_data()

        # Initialize results storage
        self.results = {
            "binary_threshold": {},
            "continuous_change": {},
            "dose_response": {},
            "comparison_summary": {},
            "metadata": {
                "outcomes": self.outcomes,
                "control_vars": self.control_vars,
                "n_countries": self.data[self.unit_col].nunique(),
                "time_range": [
                    self.data[self.time_col].min(),
                    self.data[self.time_col].max(),
                ],
                "n_observations": len(self.data),
            },
        }

    def _load_and_validate_data(self) -> DataFrame:
        """Load and validate the analysis dataset."""
        data = pd.read_csv(self.data_path)

        # Validate required columns
        required_cols = (
            [self.unit_col, self.time_col, "mpower_total"]
            + self.outcomes
            + self.control_vars
        )

        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Handle missing values
        missing_summary = data[required_cols].isnull().sum()
        if missing_summary.sum() > 0:
            print(f"Missing values found:\n{missing_summary[missing_summary > 0]}")

        return data

    def create_binary_treatment(
        self, threshold: float = MPOWER_THRESHOLD, min_years_high: int = 2
    ) -> DataFrame:
        """Create binary treatment definition based on MPOWER threshold.

        Args:
            threshold: MPOWER score threshold for treatment
            min_years_high: Minimum years above threshold

        Returns:
            Data with binary treatment variables
        """
        prep = MPOWERDataPrep(
            data=self.data, country_col=self.unit_col, year_col=self.time_col
        )

        data_with_treatment = prep.create_treatment_cohorts(
            mpower_col="mpower_total",
            treatment_definition="binary_threshold",
            threshold=threshold,
            min_years_high=min_years_high,
        )

        # Create additional binary variables
        data_with_treatment["mpower_high_binary"] = (
            data_with_treatment["mpower_total"] >= threshold
        ).astype(int)

        data_with_treatment["first_high_year"] = data_with_treatment.groupby(
            self.unit_col
        )["treatment_cohort"].transform("first")

        # Create post-treatment indicator
        data_with_treatment["post_treatment"] = (
            (data_with_treatment["first_high_year"] > 0)
            & (
                data_with_treatment[self.time_col]
                >= data_with_treatment["first_high_year"]
            )
        ).astype(int)

        return data_with_treatment

    def create_continuous_treatment(
        self,
        baseline_years: list[int] | None = None,
        improvement_threshold: float = IMPROVEMENT_THRESHOLD,
    ) -> DataFrame:
        """Create continuous change treatment definition.

        Args:
            baseline_years: Years to use for baseline calculation
            improvement_threshold: Threshold for substantial improvement

        Returns:
            Data with continuous treatment variables
        """
        if baseline_years is None:
            baseline_years = [2008, 2010, 2012]  # First 3 periods

        prep = MPOWERDataPrep(
            data=self.data, country_col=self.unit_col, year_col=self.time_col
        )

        data_with_treatment = prep.create_treatment_cohorts(
            mpower_col="mpower_total",
            treatment_definition="continuous_change",
            baseline_years=baseline_years,
        )

        # Add continuous treatment intensity
        data_with_treatment = self._add_continuous_intensity(
            data_with_treatment, baseline_years
        )

        return data_with_treatment

    def _add_continuous_intensity(
        self, data: DataFrame, baseline_years: list[int]
    ) -> DataFrame:
        """Add continuous treatment intensity measures."""
        data = data.copy()

        # Calculate baseline scores by country
        baseline_scores = (
            data[data[self.time_col].isin(baseline_years)]
            .groupby(self.unit_col)["mpower_total"]
            .mean()
        )

        # Merge baseline scores
        data["baseline_mpower"] = data[self.unit_col].map(baseline_scores)

        # Calculate improvement measures
        data["mpower_improvement"] = data["mpower_total"] - data["baseline_mpower"]

        data["mpower_pct_improvement"] = (
            data["mpower_improvement"] / data["baseline_mpower"]
        )

        # Create treatment intensity (continuous)
        data["treatment_intensity"] = np.maximum(0, data["mpower_pct_improvement"])

        return data

    def create_dose_response_data(self) -> DataFrame:
        """Create data for dose-response analysis using continuous MPOWER scores.

        Returns:
            Data prepared for dose-response analysis
        """
        data = self.data.copy()

        # Normalize MPOWER scores (0-1 scale)
        data["mpower_normalized"] = (
            data["mpower_total"] - data["mpower_total"].min()
        ) / (data["mpower_total"].max() - data["mpower_total"].min())

        # Create lagged MPOWER scores for identification
        data["mpower_lag1"] = data.groupby(self.unit_col)["mpower_total"].shift(1)

        data["mpower_lag2"] = data.groupby(self.unit_col)["mpower_total"].shift(2)

        # MPOWER change variables
        data["mpower_change"] = data["mpower_total"] - data["mpower_lag1"]

        data["mpower_cumulative_change"] = (
            data.groupby(self.unit_col)["mpower_change"].cumsum().fillna(0)
        )

        return data

    def run_binary_threshold_analysis(
        self, threshold: float = MPOWER_THRESHOLD
    ) -> dict[str, Any]:
        """Run analysis with binary threshold treatment definition.

        Args:
            threshold: MPOWER score threshold

        Returns:
            Analysis results for binary treatment
        """
        print(f"Running binary threshold analysis (threshold={threshold})...")

        # Create binary treatment data
        binary_data = self.create_binary_treatment(threshold=threshold)

        # Count treated units
        treated_countries = binary_data[binary_data["treatment_cohort"] > 0][
            self.unit_col
        ].nunique()

        results = {
            "treatment_definition": f"binary_threshold_{threshold}",
            "n_treated_countries": treated_countries,
            "treatment_years": sorted(
                binary_data[binary_data["treatment_cohort"] > 0][
                    "treatment_cohort"
                ].unique()
            ),
            "outcomes": {},
        }

        # Run DiD analysis for each outcome
        for outcome in self.outcomes:
            print(f"  Analyzing {outcome}...")

            try:
                # Callaway & Sant'Anna DiD
                did = CallawayDiD(
                    data=binary_data,
                    cohort_col="first_high_year",
                    unit_col=self.unit_col,
                    time_col=self.time_col,
                )

                did.fit(outcome=outcome, covariates=self.control_vars)

                # Get aggregated results
                simple_att = did.aggregate("simple")

                results["outcomes"][outcome] = {
                    "att": float(simple_att.get("att", np.nan)),
                    "std_error": float(simple_att.get("std_error", np.nan)),
                    "p_value": float(simple_att.get("p_value", np.nan)),
                    "ci_lower": float(simple_att.get("ci_lower", np.nan)),
                    "ci_upper": float(simple_att.get("ci_upper", np.nan)),
                    "method": "callaway_did",
                }

            except Exception as e:
                print(f"    Error in DiD analysis: {e}")
                results["outcomes"][outcome] = {
                    "error": str(e),
                    "method": "callaway_did",
                }

        return results

    def run_continuous_change_analysis(self) -> dict[str, Any]:
        """Run analysis with continuous change treatment definition.

        Returns:
            Analysis results for continuous treatment
        """
        print("Running continuous change analysis...")

        # Create continuous treatment data
        continuous_data = self.create_continuous_treatment()

        # Count treated units
        treated_countries = continuous_data[continuous_data["treatment_cohort"] > 0][
            self.unit_col
        ].nunique()

        results = {
            "treatment_definition": "continuous_change",
            "n_treated_countries": treated_countries,
            "treatment_years": sorted(
                continuous_data[continuous_data["treatment_cohort"] > 0][
                    "treatment_cohort"
                ].unique()
            ),
            "outcomes": {},
        }

        # Run DiD analysis for each outcome
        for outcome in self.outcomes:
            print(f"  Analyzing {outcome}...")

            try:
                # Use treatment_cohort as cohort variable
                did = CallawayDiD(
                    data=continuous_data,
                    cohort_col="treatment_cohort",
                    unit_col=self.unit_col,
                    time_col=self.time_col,
                )

                did.fit(outcome=outcome, covariates=self.control_vars)

                # Get aggregated results
                simple_att = did.aggregate("simple")

                results["outcomes"][outcome] = {
                    "att": float(simple_att.get("att", np.nan)),
                    "std_error": float(simple_att.get("std_error", np.nan)),
                    "p_value": float(simple_att.get("p_value", np.nan)),
                    "ci_lower": float(simple_att.get("ci_lower", np.nan)),
                    "ci_upper": float(simple_att.get("ci_upper", np.nan)),
                    "method": "callaway_did",
                }

            except Exception as e:
                print(f"    Error in DiD analysis: {e}")
                results["outcomes"][outcome] = {
                    "error": str(e),
                    "method": "callaway_did",
                }

        return results

    def run_dose_response_analysis(self) -> dict[str, Any]:
        """Run dose-response analysis using continuous MPOWER scores.

        Returns:
            Dose-response analysis results
        """
        print("Running dose-response analysis...")

        # Create dose-response data
        dose_data = self.create_dose_response_data()

        results = {
            "treatment_definition": "dose_response",
            "mpower_range": [
                float(dose_data["mpower_total"].min()),
                float(dose_data["mpower_total"].max()),
            ],
            "outcomes": {},
        }

        # Simple panel regression with continuous MPOWER
        for outcome in self.outcomes:
            print(f"  Analyzing {outcome}...")

            try:
                # Fixed effects regression using statsmodels
                from statsmodels.formula.api import ols

                # Create regression formula
                controls_str = " + ".join(self.control_vars)
                formula = (
                    f"{outcome} ~ mpower_normalized + {controls_str} + "
                    f"C({self.unit_col}) + C({self.time_col})"
                )

                # Run regression
                model = ols(formula, data=dose_data).fit()

                # Extract MPOWER coefficient
                mpower_coef = model.params.get("mpower_normalized", np.nan)
                mpower_se = model.bse.get("mpower_normalized", np.nan)
                mpower_pval = model.pvalues.get("mpower_normalized", np.nan)

                # Calculate confidence interval
                ci_lower = mpower_coef - 1.96 * mpower_se
                ci_upper = mpower_coef + 1.96 * mpower_se

                results["outcomes"][outcome] = {
                    "coefficient": float(mpower_coef),
                    "std_error": float(mpower_se),
                    "p_value": float(mpower_pval),
                    "ci_lower": float(ci_lower),
                    "ci_upper": float(ci_upper),
                    "method": "fixed_effects_ols",
                    "r_squared": float(model.rsquared),
                    "n_obs": int(model.nobs),
                }

            except Exception as e:
                print(f"    Error in dose-response analysis: {e}")
                results["outcomes"][outcome] = {
                    "error": str(e),
                    "method": "fixed_effects_ols",
                }

        return results

    def run_comprehensive_comparison(self) -> dict[str, Any]:
        """Run comprehensive comparison of all treatment definitions.

        Returns:
            Complete analysis results comparing all approaches
        """
        print("Starting comprehensive alternative treatment analysis...\n")

        # Run all analyses
        self.results["binary_threshold"] = self.run_binary_threshold_analysis()
        self.results["continuous_change"] = self.run_continuous_change_analysis()
        self.results["dose_response"] = self.run_dose_response_analysis()

        # Create comparison summary
        self.results["comparison_summary"] = self._create_comparison_summary()

        print("\nAlternative treatment analysis complete!")
        return self.results

    def _create_comparison_summary(self) -> dict[str, Any]:
        """Create summary comparing all treatment definitions."""
        summary = {
            "treatment_definitions": {
                "binary_threshold": {
                    "description": f"Countries with MPOWER ≥ {MPOWER_THRESHOLD}",
                    "n_treated": self.results["binary_threshold"].get(
                        "n_treated_countries", 0
                    ),
                },
                "continuous_change": {
                    "description": f"Countries with ≥{IMPROVEMENT_THRESHOLD * 100}% improvement",
                    "n_treated": self.results["continuous_change"].get(
                        "n_treated_countries", 0
                    ),
                },
                "dose_response": {
                    "description": "Continuous MPOWER score (normalized 0-1)",
                    "estimation_method": "Fixed effects OLS",
                },
            },
            "outcome_comparison": {},
        }

        # Compare results across outcomes
        for outcome in self.outcomes:
            outcome_summary = {}

            for method in ["binary_threshold", "continuous_change", "dose_response"]:
                outcome_data = self.results[method].get("outcomes", {}).get(outcome, {})

                if "error" not in outcome_data:
                    coef_key = "att" if method != "dose_response" else "coefficient"
                    outcome_summary[method] = {
                        "effect": outcome_data.get(coef_key, np.nan),
                        "std_error": outcome_data.get("std_error", np.nan),
                        "p_value": outcome_data.get("p_value", np.nan),
                        "significant": outcome_data.get("p_value", 1)
                        < SIGNIFICANCE_LEVEL,
                    }
                else:
                    outcome_summary[method] = {"error": outcome_data["error"]}

            summary["outcome_comparison"][outcome] = outcome_summary

        return summary

    def export_results(self, output_dir: str | Path) -> None:
        """Export comprehensive results to files.

        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export to JSON
        import json

        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        # Deep convert all numpy types
        def deep_convert(data):
            if isinstance(data, dict):
                return {k: deep_convert(v) for k, v in data.items()}
            if isinstance(data, list):
                return [deep_convert(v) for v in data]
            return convert_numpy(data)

        results_serializable = deep_convert(self.results)

        with open(output_path / "alternative_treatment_results.json", "w") as f:
            json.dump(results_serializable, f, indent=2)

        # Export summary to Excel
        try:
            # Create summary DataFrame
            summary_data = []
            for outcome in self.outcomes:
                for method in [
                    "binary_threshold",
                    "continuous_change",
                    "dose_response",
                ]:
                    outcome_data = (
                        self.results[method].get("outcomes", {}).get(outcome, {})
                    )

                    if "error" not in outcome_data:
                        coef_key = "att" if method != "dose_response" else "coefficient"
                        summary_data.append(
                            {
                                "outcome": outcome,
                                "treatment_definition": method,
                                "effect": outcome_data.get(coef_key, np.nan),
                                "std_error": outcome_data.get("std_error", np.nan),
                                "p_value": outcome_data.get("p_value", np.nan),
                                "ci_lower": outcome_data.get("ci_lower", np.nan),
                                "ci_upper": outcome_data.get("ci_upper", np.nan),
                                "significant": outcome_data.get("p_value", 1)
                                < SIGNIFICANCE_LEVEL,
                            }
                        )

            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(
                output_path / "alternative_treatment_summary.xlsx", index=False
            )

            print(f"Results exported to {output_path}")

        except ImportError:
            print("openpyxl not available. Skipping Excel export.")


def main():
    """Main execution function for alternative treatment analysis."""
    # Configuration
    data_path = "data/processed/analysis_ready_data.csv"
    output_dir = "results/alternative_treatment/"

    try:
        # Initialize analysis
        analysis = AlternativeTreatmentAnalysis(data_path=data_path)

        # Run comprehensive comparison
        results = analysis.run_comprehensive_comparison()

        # Export results
        analysis.export_results(output_dir)

        # Print summary
        print("\n" + "=" * 60)
        print("ALTERNATIVE TREATMENT DEFINITIONS - SUMMARY")
        print("=" * 60)

        comparison = results["comparison_summary"]

        print("\nTreatment Definitions:")
        for method, info in comparison["treatment_definitions"].items():
            print(f"  {method}:")
            print(f"    Description: {info['description']}")
            if "n_treated" in info:
                print(f"    Treated countries: {info['n_treated']}")

        print("\nEffect Estimates by Outcome:")
        for outcome, methods in comparison["outcome_comparison"].items():
            print(f"\n  {outcome}:")
            for method, results in methods.items():
                if "error" not in results:
                    effect = results["effect"]
                    p_val = results["p_value"]
                    significant = (
                        "***"
                        if p_val < 0.01
                        else "**"
                        if p_val < 0.05
                        else "*"
                        if p_val < 0.1
                        else ""
                    )
                    print(f"    {method}: {effect:.3f} (p={p_val:.3f}){significant}")
                else:
                    print(f"    {method}: Error - {results['error']}")

        print(f"\nDetailed results saved to: {output_dir}")

    except Exception as e:
        print(f"Error in alternative treatment analysis: {e}")
        raise


if __name__ == "__main__":
    main()
