"""Main Analysis Script for MPOWER Mortality Causal Analysis.

This script provides a comprehensive analysis pipeline for evaluating the causal
impact of WHO MPOWER tobacco control policies on mortality outcomes using
modern causal inference methods.

The pipeline includes:
1. Data loading and preparation
2. Descriptive analysis and visualization
3. Parallel trends testing
4. Callaway & Sant'Anna staggered difference-in-differences
5. Event study analysis
6. Robustness checks (TWFE, synthetic control, sensitivity tests)
7. Results compilation and export

Usage:
    from mpower_mortality_causal_analysis.analysis import MPOWERAnalysisPipeline

    pipeline = MPOWERAnalysisPipeline(
        data_path='data/processed/analysis_ready_data.csv'
    )
    results = pipeline.run_full_analysis()
    pipeline.export_results('results/', format='comprehensive')
"""

import warnings

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pandas import DataFrame

try:
    import matplotlib.pyplot as plt

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    warnings.warn(
        "Plotting packages not available. Install matplotlib for visualization.",
        stacklevel=2,
    )

# Import causal inference modules
from mpower_mortality_causal_analysis.causal_inference.methods.callaway_did import (
    CallawayDiD,
)
from mpower_mortality_causal_analysis.causal_inference.methods.synthetic_control import (
    MPOWERSyntheticControl,
)
from mpower_mortality_causal_analysis.causal_inference.utils.descriptive import (
    MPOWERDescriptives,
)
from mpower_mortality_causal_analysis.causal_inference.utils.event_study import (
    EventStudyAnalysis,
)
from mpower_mortality_causal_analysis.causal_inference.utils.robustness_comprehensive import (
    RobustnessChecks,
)


class MPOWERAnalysisPipeline:
    """Comprehensive Analysis Pipeline for MPOWER Mortality Study.

    This class orchestrates the complete causal inference analysis,
    from data loading through final results compilation.

    Parameters:
        data_path (str): Path to the analysis-ready dataset
        outcomes (List[str]): Mortality outcome variables to analyze
        treatment_col (str): Treatment indicator variable
        unit_col (str): Unit identifier (country)
        time_col (str): Time identifier (year)

    Example:
        >>> pipeline = MPOWERAnalysisPipeline('data/processed/analysis_ready_data.csv')
        >>> results = pipeline.run_full_analysis()
        >>> pipeline.export_results('results/')
    """

    def __init__(
        self,
        data_path: str | Path,
        outcomes: list[str] | None = None,
        treatment_col: str = "mpower_high_binary",
        treatment_year_col: str = "first_high_year",
        unit_col: str = "country",
        time_col: str = "year",
        control_vars: list[str] | None = None,
    ):
        """Initialize the analysis pipeline."""
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        self.treatment_col = treatment_col
        self.treatment_year_col = treatment_year_col
        self.unit_col = unit_col
        self.time_col = time_col

        # Default outcome variables (mortality rates)
        self.outcomes = outcomes or [
            "lung_cancer_mortality_rate",
            "cardiovascular_mortality_rate",
            "ihd_mortality_rate",
            "copd_mortality_rate",
        ]

        # Default control variables
        self.control_vars = control_vars or [
            "gdp_per_capita_log",
            "urban_population_pct",
            "population_log",
            "education_expenditure_pct_gdp",
        ]

        # Load and validate data
        self.data = self._load_and_validate_data()

        # Initialize results storage
        self.results = {
            "descriptive": {},
            "parallel_trends": {},
            "callaway_did": {},
            "event_study": {},
            "synthetic_control": {},
            "robustness": {},
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
        # Load data
        data = pd.read_csv(self.data_path)

        # Validate required columns
        required_cols = (
            [self.unit_col, self.time_col, self.treatment_col, self.treatment_year_col]
            + self.outcomes
            + self.control_vars
        )

        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Basic data validation

        # Check for missing values in key variables
        missing_summary = data[required_cols].isnull().sum()
        if missing_summary.sum() > 0:
            pass

        return data

    def run_descriptive_analysis(self) -> dict[str, Any]:
        """Run comprehensive descriptive analysis."""
        descriptives = MPOWERDescriptives(
            data=self.data,
            country_col=self.unit_col,
            year_col=self.time_col,
            cohort_col=self.treatment_year_col,
        )

        # Treatment adoption timeline
        adoption_timeline = descriptives.plot_treatment_adoption_timeline()

        # Outcome trends
        outcome_trends = {}
        for outcome in self.outcomes:
            trends = descriptives.plot_outcome_trends_by_cohort(
                outcomes=[outcome],
                save_path=f"results/descriptive/trends_{outcome}.png"
                if PLOTTING_AVAILABLE
                else None,
            )
            outcome_trends[outcome] = trends

        # Treatment balance
        balance_results = descriptives.plot_treatment_balance_check(
            save_path="results/descriptive/treatment_balance.png"
            if PLOTTING_AVAILABLE
            else None,
        )

        # Correlation analysis
        correlation_results = descriptives.plot_correlation_heatmap(
            variables=self.outcomes + self.control_vars + [self.treatment_col],
            save_path="results/descriptive/correlation_heatmap.png"
            if PLOTTING_AVAILABLE
            else None,
        )

        self.results["descriptive"] = {
            "adoption_timeline": adoption_timeline,
            "outcome_trends": outcome_trends,
            "treatment_balance": balance_results,
            "correlations": correlation_results,
        }

        return self.results["descriptive"]

    def run_parallel_trends_analysis(self) -> dict[str, Any]:
        """Run comprehensive parallel trends testing."""
        # Initialize event study for parallel trends testing
        event_study = EventStudyAnalysis(
            data=self.data,
            unit_col=self.unit_col,
            time_col=self.time_col,
            treatment_col=self.treatment_year_col,
            never_treated_value=0,
        )

        parallel_trends_results = {}

        for outcome in self.outcomes:
            # Comprehensive parallel trends analysis
            pt_analysis = event_study.comprehensive_parallel_trends_analysis(
                outcome=outcome, max_lead=4, covariates=self.control_vars, alpha=0.05
            )

            parallel_trends_results[outcome] = pt_analysis

            # Print summary
            pt_analysis["overall_assessment"]

        self.results["parallel_trends"] = parallel_trends_results

        return parallel_trends_results

    def run_callaway_did_analysis(self) -> dict[str, Any]:
        """Run Callaway & Sant'Anna staggered DiD analysis."""
        callaway_results = {}

        for outcome in self.outcomes:
            try:
                # Initialize Callaway DiD estimator
                did_estimator = CallawayDiD(
                    data=self.data,
                    cohort_col=self.treatment_year_col,
                    unit_col=self.unit_col,
                    time_col=self.time_col,
                )

                # Fit the model
                did_estimator.fit(outcome=outcome, covariates=self.control_vars)

                # Get aggregated treatment effects
                simple_att = did_estimator.aggregate("simple")
                group_att = did_estimator.aggregate("group")
                dynamic_att = did_estimator.aggregate("event")

                # Event study plot
                if PLOTTING_AVAILABLE:
                    did_estimator.plot_event_study(
                        save_path=f"results/callaway_did/event_study_{outcome}.png"
                    )

                callaway_results[outcome] = {
                    "simple_att": simple_att,
                    "group_att": group_att,
                    "dynamic_att": dynamic_att,
                    "model_summary": did_estimator.summary(),
                }

                # Print key results
                if isinstance(simple_att, dict) and "att" in simple_att:
                    simple_att["att"]
                    simple_att.get("se", "N/A")
                    simple_att.get("pvalue", "N/A")

            except Exception as e:
                callaway_results[outcome] = {"error": str(e)}

        self.results["callaway_did"] = callaway_results

        return callaway_results

    def run_event_study_analysis(self) -> dict[str, Any]:
        """Run detailed event study analysis."""
        event_study = EventStudyAnalysis(
            data=self.data,
            unit_col=self.unit_col,
            time_col=self.time_col,
            treatment_col=self.treatment_year_col,
            never_treated_value=0,
        )

        event_study_results = {}

        for outcome in self.outcomes:
            try:
                # Estimate event study
                results = event_study.estimate(
                    outcome=outcome,
                    covariates=self.control_vars,
                    max_lag=5,
                    max_lead=4,
                    method="fixed_effects",
                    cluster_var=self.unit_col,
                )

                # Plot results
                if PLOTTING_AVAILABLE:
                    event_study.plot_event_study(
                        results,
                        save_path=f"results/event_study/event_study_{outcome}.png",
                    )

                event_study_results[outcome] = results

                # Print summary
                sum(
                    1
                    for var, pval in results["event_time_pvalues"].items()
                    if "lead_" in var and pval < 0.05
                )
                sum(
                    1
                    for var, pval in results["event_time_pvalues"].items()
                    if "lag_" in var and pval < 0.05
                )

            except Exception as e:
                event_study_results[outcome] = {"error": str(e)}

        self.results["event_study"] = event_study_results

        return event_study_results

    def run_synthetic_control_analysis(self) -> dict[str, Any]:
        """Run comprehensive synthetic control analysis for all treated countries.

        This method addresses parallel trends violations by creating optimal
        synthetic controls for each MPOWER-adopting country.

        Returns:
            Dict with synthetic control results for all outcomes
        """
        print("\n=== Running MPOWER Synthetic Control Analysis ===")
        print("Addressing parallel trends violations with synthetic controls...")

        synthetic_control_results = {}

        # Extract treatment information from the data
        treated_countries = self.data[self.data[self.treatment_col] == 1][
            self.unit_col
        ].unique()
        treatment_info = {}

        for country in treated_countries:
            country_data = self.data[self.data[self.unit_col] == country]
            first_treatment_year = country_data[country_data[self.treatment_col] == 1][
                self.time_col
            ].min()
            if pd.notna(first_treatment_year):
                treatment_info[country] = int(first_treatment_year)

        print(
            f"Found {len(treatment_info)} treated countries with valid treatment timing"
        )

        for outcome in self.outcomes:
            print(f"\nAnalyzing {outcome}...")

            try:
                # Initialize synthetic control estimator
                sc_estimator = MPOWERSyntheticControl(
                    data=self.data,
                    unit_col=self.unit_col,
                    time_col=self.time_col,
                )

                # Fit synthetic control for all treated units
                sc_results = sc_estimator.fit_all_units(
                    treatment_info=treatment_info,
                    outcome=outcome,
                    predictors=self.control_vars,
                    pre_periods=2,  # Minimum 2 pre-treatment periods
                )

                # Generate plots if plotting is available
                if PLOTTING_AVAILABLE:
                    try:
                        plot_results = sc_estimator.plot_all_units(
                            outcome=outcome,
                            save_dir=f"results/synthetic_control/{outcome}",
                        )
                        sc_results["plot_info"] = plot_results
                    except Exception as e:
                        print(f"  Warning: Could not generate plots - {e}")

                # Add summary information
                sc_results["outcome"] = outcome
                sc_results["treatment_info"] = treatment_info
                sc_results["summary"] = sc_estimator.summary()

                synthetic_control_results[outcome] = sc_results

                # Print summary
                agg = sc_results["aggregated"]
                if (
                    "avg_treatment_effect" in agg
                    and agg["avg_treatment_effect"] is not None
                ):
                    print(
                        f"  Average Treatment Effect: {agg['avg_treatment_effect']:.4f}"
                    )
                    print(
                        f"  Successful Fits: {len(sc_results['successful_units'])}/{len(treatment_info)}"
                    )
                    print(
                        f"  Average RMSE: {agg.get('match_quality', {}).get('avg_rmse', 'N/A'):.4f}"
                    )
                else:
                    print("  No successful synthetic control fits")

            except Exception as e:
                print(f"  Error in synthetic control analysis: {e}")
                synthetic_control_results[outcome] = {
                    "error": str(e),
                    "treatment_info": treatment_info,
                }

        self.results["synthetic_control"] = synthetic_control_results

        print("\n=== Synthetic Control Analysis Complete ===")
        return synthetic_control_results

    def run_robustness_checks(self) -> dict[str, Any]:
        """Run comprehensive robustness checks."""
        # Initialize comprehensive robustness checker
        robustness_checker = RobustnessChecks(
            data=self.data,
            country_col=self.unit_col,
            year_col=self.time_col,
            cohort_col=self.treatment_year_col,
        )

        robustness_results = {}

        for outcome in self.outcomes:
            outcome_robustness = {}

            # 1. TWFE comparison
            try:
                twfe_results = robustness_checker._run_twfe_comparison(
                    outcome, self.control_vars
                )
                outcome_robustness["twfe_comparison"] = twfe_results
            except Exception as e:
                outcome_robustness["twfe_comparison"] = {"error": str(e)}

            # 2. Synthetic control analysis (for select countries)
            try:
                # Use the run_all_checks method which includes synthetic control
                sc_results = robustness_checker._run_synthetic_control_analysis(
                    outcome, self.control_vars
                )
                outcome_robustness["synthetic_control"] = sc_results

            except Exception as e:
                outcome_robustness["synthetic_control"] = {"error": str(e)}

            # 3. Sample robustness
            try:
                sample_robustness = robustness_checker._run_sample_robustness(
                    outcome, self.control_vars
                )
                outcome_robustness["sample_robustness"] = sample_robustness
            except Exception as e:
                outcome_robustness["sample_robustness"] = {"error": str(e)}

            # 4. Placebo tests
            try:
                placebo_results = robustness_checker._run_placebo_tests(
                    outcome, self.control_vars
                )
                outcome_robustness["placebo_tests"] = placebo_results
            except Exception as e:
                outcome_robustness["placebo_tests"] = {"error": str(e)}

            robustness_results[outcome] = outcome_robustness

        self.results["robustness"] = robustness_results

        return robustness_results

    def run_full_analysis(
        self, skip_robustness: bool = False, run_synthetic_control: bool = True
    ) -> dict[str, Any]:
        """Run the complete analysis pipeline.

        Args:
            skip_robustness (bool): Whether to skip time-intensive robustness checks
            run_synthetic_control (bool): Whether to run synthetic control analysis

        Returns:
            Dict with all analysis results
        """
        try:
            # 1. Descriptive analysis
            self.run_descriptive_analysis()

            # 2. Parallel trends testing
            self.run_parallel_trends_analysis()

            # 3. Main causal analysis (Callaway & Sant'Anna)
            self.run_callaway_did_analysis()

            # 4. Event study analysis
            self.run_event_study_analysis()

            # 5. Synthetic control analysis (addresses parallel trends violations)
            if run_synthetic_control:
                self.run_synthetic_control_analysis()
            else:
                print(
                    "\nSkipping synthetic control analysis (run_synthetic_control=False)"
                )

            # 6. Robustness checks (optional, can be time-intensive)
            if not skip_robustness:
                self.run_robustness_checks()
            else:
                print("\nSkipping robustness checks (skip_robustness=True)")

            # 7. Generate summary
            self._generate_analysis_summary()

        except Exception:
            raise

        return self.results

    def _generate_analysis_summary(self) -> None:
        """Generate a summary of key findings."""
        # Summary of parallel trends tests
        for outcome in self.outcomes:
            if outcome in self.results.get("parallel_trends", {}):
                pt_result = self.results["parallel_trends"][outcome]
                assessment = pt_result.get("overall_assessment", {})
                assessment.get("assessment", "unknown")
                assessment.get("confidence", "unknown")

        # Summary of main treatment effects
        for outcome in self.outcomes:
            if outcome in self.results.get("callaway_did", {}):
                did_result = self.results["callaway_did"][outcome]
                if "simple_att" in did_result:
                    att = did_result["simple_att"]
                    if isinstance(att, dict) and "att" in att:
                        att["att"]
                        att.get("pvalue", "N/A")

        # Data summary
        self.results.get("metadata", {})

    def export_results(
        self, output_dir: str | Path, format: str = "comprehensive"
    ) -> None:
        """Export analysis results to files.

        Args:
            output_dir (str or Path): Directory to save results
            format (str): Export format ('comprehensive', 'summary', 'excel')
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if format in ["comprehensive", "summary"]:
            # Export main results as JSON
            import json

            # Prepare serializable results
            export_results = self._prepare_results_for_export()

            with open(output_path / "analysis_results.json", "w") as f:
                json.dump(export_results, f, indent=2, default=str)

        if format in ["comprehensive", "excel"]:
            # Export key results to Excel
            self._export_excel_summary(output_path)

        # Export coefficient tables
        self._export_coefficient_tables(output_path)

    def _prepare_results_for_export(self) -> dict[str, Any]:
        """Prepare results dictionary for JSON export."""
        # Create a clean copy of results for export
        export_results = {}

        for key, value in self.results.items():
            if key == "metadata":
                export_results[key] = value
            else:
                # Recursively clean results
                export_results[key] = self._clean_results_for_json(value)

        return export_results

    def _clean_results_for_json(self, obj: Any) -> Any:
        """Clean results object for JSON serialization."""
        if isinstance(obj, dict):
            cleaned = {}
            for k, v in obj.items():
                # Skip non-serializable objects like fitted models
                if isinstance(v, pd.DataFrame | plt.Figure) or k in [
                    "model",
                    "fitted_model",
                    "synth_model",
                ]:
                    continue
                cleaned[k] = self._clean_results_for_json(v)
            return cleaned
        if isinstance(obj, list | tuple):
            return [self._clean_results_for_json(item) for item in obj]
        if isinstance(obj, np.integer | np.floating | np.ndarray):
            return obj.tolist() if hasattr(obj, "tolist") else float(obj)
        return obj

    def _export_excel_summary(self, output_path: Path) -> None:
        """Export summary results to Excel."""
        try:
            with pd.ExcelWriter(output_path / "analysis_summary.xlsx") as writer:
                # Treatment effects summary
                self._create_treatment_effects_table().to_excel(
                    writer, sheet_name="Treatment_Effects", index=False
                )

                # Parallel trends summary
                self._create_parallel_trends_table().to_excel(
                    writer, sheet_name="Parallel_Trends", index=False
                )

                # Data summary
                pd.DataFrame([self.results["metadata"]]).to_excel(
                    writer, sheet_name="Data_Summary", index=False
                )

        except ImportError:
            pass

    def _create_treatment_effects_table(self) -> DataFrame:
        """Create summary table of treatment effects."""
        effects_data = []

        for outcome in self.outcomes:
            # Callaway & Sant'Anna results
            if outcome in self.results.get("callaway_did", {}):
                did_result = self.results["callaway_did"][outcome]
                if "simple_att" in did_result and isinstance(
                    did_result["simple_att"], dict
                ):
                    att = did_result["simple_att"]
                    effects_data.append(
                        {
                            "outcome": outcome,
                            "method": "Callaway_SantAnna",
                            "att": att.get("att", np.nan),
                            "std_error": att.get("se", np.nan),
                            "p_value": att.get("pvalue", np.nan),
                            "ci_lower": att.get("ci_lower", np.nan),
                            "ci_upper": att.get("ci_upper", np.nan),
                        }
                    )

            # Synthetic control results
            if outcome in self.results.get("synthetic_control", {}):
                sc_result = self.results["synthetic_control"][outcome]
                if "aggregated" in sc_result and isinstance(
                    sc_result["aggregated"], dict
                ):
                    agg = sc_result["aggregated"]
                    effects_data.append(
                        {
                            "outcome": outcome,
                            "method": "Synthetic_Control",
                            "att": agg.get("avg_treatment_effect", np.nan),
                            "std_error": agg.get("std_treatment_effect", np.nan),
                            "p_value": np.nan,  # Synthetic control doesn't provide p-values by default
                            "ci_lower": np.nan,
                            "ci_upper": np.nan,
                        }
                    )

        return pd.DataFrame(effects_data)

    def _create_parallel_trends_table(self) -> DataFrame:
        """Create summary table of parallel trends tests."""
        pt_data = []

        for outcome in self.outcomes:
            if outcome in self.results.get("parallel_trends", {}):
                pt_result = self.results["parallel_trends"][outcome]
                overall = pt_result.get("overall_assessment", {})
                tests = pt_result.get("parallel_trends_tests", {})

                pt_data.append(
                    {
                        "outcome": outcome,
                        "assessment": overall.get("assessment", "unknown"),
                        "confidence": overall.get("confidence", "unknown"),
                        "joint_f_pvalue": tests.get("joint_f_test_pvalue", np.nan),
                        "individual_violation_rate": tests.get(
                            "individual_test_rate", np.nan
                        ),
                        "linear_trend_pvalue": tests.get("linear_trend_test", {}).get(
                            "pvalue", np.nan
                        ),
                    }
                )

        return pd.DataFrame(pt_data)

    def _export_coefficient_tables(self, output_path: Path) -> None:
        """Export detailed coefficient tables."""
        coef_dir = output_path / "coefficients"
        coef_dir.mkdir(exist_ok=True)

        # Export event study coefficients
        for outcome in self.outcomes:
            if outcome in self.results.get("event_study", {}):
                es_result = self.results["event_study"][outcome]
                if "event_time_coefficients" in es_result:
                    # Create coefficient DataFrame
                    coeffs = es_result["event_time_coefficients"]
                    ses = es_result.get("event_time_std_errors", {})
                    pvals = es_result.get("event_time_pvalues", {})

                    coef_data = []
                    for var in coeffs:
                        coef_data.append(
                            {
                                "variable": var,
                                "coefficient": coeffs[var],
                                "std_error": ses.get(var, np.nan),
                                "p_value": pvals.get(var, np.nan),
                            }
                        )

                    coef_df = pd.DataFrame(coef_data)
                    coef_df.to_csv(coef_dir / f"event_study_{outcome}.csv", index=False)


# Convenience function for quick analysis
def run_mpower_analysis(
    data_path: str,
    output_dir: str = "results",
    skip_robustness: bool = False,
    run_synthetic_control: bool = True,
) -> dict[str, Any]:
    """Convenience function to run the complete MPOWER analysis.

    Args:
        data_path (str): Path to analysis-ready data
        output_dir (str): Directory to save results
        skip_robustness (bool): Whether to skip robustness checks
        run_synthetic_control (bool): Whether to run synthetic control analysis

    Returns:
        Dict with complete analysis results
    """
    pipeline = MPOWERAnalysisPipeline(data_path)
    results = pipeline.run_full_analysis(
        skip_robustness=skip_robustness, run_synthetic_control=run_synthetic_control
    )
    pipeline.export_results(output_dir)

    return results


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "results"
        skip_robust = len(sys.argv) > 3 and sys.argv[3].lower() == "true"

        run_mpower_analysis(data_path, output_dir, skip_robust)
    else:
        pass
