"""Advanced Comparison Example for MPOWER Causal Inference Analysis.

This example demonstrates how to compare multiple causal inference methods
and conduct comprehensive robustness analysis for MPOWER policy evaluation.
"""

from typing import Any

import numpy as np
import pandas as pd

# Import the causal inference framework
from mpower_mortality_causal_analysis.causal_inference import (
    CallawayDiD,
    EventStudyAnalysis,
    MPOWERDataPrep,
    PanelFixedEffects,
    RobustnessTests,
    SyntheticControl,
)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: Matplotlib/Seaborn not available. Plotting disabled.")


class MethodComparison:
    """Class for comparing multiple causal inference methods on MPOWER data."""

    def __init__(self, data: pd.DataFrame):
        """Initialize with prepared data."""
        self.data = data
        self.results = {}

    def run_all_methods(
        self,
        outcome: str = "mortality_rate",
        treatment_col: str = "treatment_cohort",
        controls: list[str] = None,
    ) -> dict[str, Any]:
        """Run all available causal inference methods.

        Args:
            outcome (str): Outcome variable
            treatment_col (str): Treatment cohort column
            controls (List[str]): Control variables

        Returns:
            Dict with results from all methods
        """
        if controls is None:
            controls = ["gdp_log", "urban_percentage"]

        methods_results = {}

        # 1. Callaway & Sant'Anna DiD
        print("Running Callaway & Sant'Anna DiD...")
        try:
            did = CallawayDiD(
                data=self.data,
                cohort_col=treatment_col,
                unit_col="country",
                time_col="year",
            )
            did.fit(outcome=outcome, covariates=controls)

            simple_att = did.aggregate("simple")
            event_study = did.aggregate("event")

            methods_results["callaway_did"] = {
                "estimator": did,
                "simple_att": simple_att,
                "event_study": event_study,
                "method": "Callaway & Sant'Anna DiD",
            }

        except Exception as e:
            methods_results["callaway_did"] = {"error": str(e)}
            print(f"  Error: {e}")

        # 2. Panel Fixed Effects (multiple specifications)
        print("Running Panel Fixed Effects...")
        try:
            # Two-way fixed effects
            panel_twfe = PanelFixedEffects(
                data=self.data, unit_col="country", time_col="year"
            )

            # Create treatment intensity variable
            data_with_treatment = self.data.copy()
            data_with_treatment["mpower_high"] = (
                data_with_treatment["mpower_total_score"] >= 25
            ).astype(int)
            panel_twfe.data = data_with_treatment

            panel_twfe.fit(
                outcome=outcome, covariates=["mpower_high"] + controls, method="twfe"
            )

            coeffs_twfe = panel_twfe.get_coefficients()

            methods_results["panel_twfe"] = {
                "estimator": panel_twfe,
                "coefficients": coeffs_twfe,
                "treatment_effect": coeffs_twfe.loc["mpower_high", "coefficient"]
                if "mpower_high" in coeffs_twfe.index
                else np.nan,
                "method": "Two-Way Fixed Effects",
            }

        except Exception as e:
            methods_results["panel_twfe"] = {"error": str(e)}
            print(f"  Error: {e}")

        # 3. Event Study
        print("Running Event Study...")
        try:
            event_study = EventStudyAnalysis(
                data=self.data,
                unit_col="country",
                time_col="year",
                treatment_col=treatment_col,
            )

            event_results = event_study.estimate(
                outcome=outcome, covariates=controls, max_lag=4, max_lead=4
            )

            parallel_trends = event_study.test_parallel_trends(event_results)

            methods_results["event_study"] = {
                "estimator": event_study,
                "results": event_results,
                "parallel_trends": parallel_trends,
                "method": "Event Study",
            }

        except Exception as e:
            methods_results["event_study"] = {"error": str(e)}
            print(f"  Error: {e}")

        # 4. Synthetic Control (for key treatment countries)
        print("Running Synthetic Control...")
        treatment_countries = self.data[self.data[treatment_col] > 0][
            "country"
        ].unique()

        synthetic_results = {}
        for country in treatment_countries[:3]:  # Limit to first 3 for demonstration
            try:
                treatment_year = self.data[self.data["country"] == country][
                    treatment_col
                ].iloc[0]

                sc = SyntheticControl(
                    data=self.data,
                    unit_col="country",
                    time_col="year",
                    treatment_time=treatment_year,
                    treated_unit=country,
                )

                sc.fit(
                    outcome=outcome,
                    predictors=controls + ["healthcare_spending_log"]
                    if "healthcare_spending_log" in self.data.columns
                    else controls,
                )

                treatment_effect = sc.get_treatment_effect()

                synthetic_results[country] = {
                    "estimator": sc,
                    "treatment_effect": treatment_effect,
                    "treatment_year": treatment_year,
                }

            except Exception as e:
                synthetic_results[country] = {"error": str(e)}

        methods_results["synthetic_control"] = {
            "results": synthetic_results,
            "method": "Synthetic Control",
        }

        self.results = methods_results
        return methods_results

    def compare_estimates(self) -> pd.DataFrame:
        """Create a comparison table of treatment effect estimates.

        Returns:
            DataFrame with treatment effects from different methods
        """
        comparison_data = []

        # Callaway & Sant'Anna
        if (
            "callaway_did" in self.results
            and "error" not in self.results["callaway_did"]
        ):
            simple_att = self.results["callaway_did"]["simple_att"]
            if isinstance(simple_att, dict) and "att" in simple_att:
                comparison_data.append(
                    {
                        "Method": "Callaway & Sant'Anna DiD",
                        "Treatment_Effect": simple_att["att"],
                        "Standard_Error": simple_att.get("se", np.nan),
                        "P_Value": simple_att.get("pvalue", np.nan),
                    }
                )

        # Panel Fixed Effects
        if "panel_twfe" in self.results and "error" not in self.results["panel_twfe"]:
            effect = self.results["panel_twfe"]["treatment_effect"]
            coeffs = self.results["panel_twfe"]["coefficients"]
            if "mpower_high" in coeffs.index:
                comparison_data.append(
                    {
                        "Method": "Two-Way Fixed Effects",
                        "Treatment_Effect": coeffs.loc["mpower_high", "coefficient"],
                        "Standard_Error": coeffs.loc["mpower_high", "std_error"],
                        "P_Value": coeffs.loc["mpower_high", "p_value"],
                    }
                )

        # Synthetic Control (average across countries)
        if "synthetic_control" in self.results:
            sc_results = self.results["synthetic_control"]["results"]
            valid_effects = []
            for country, result in sc_results.items():
                if "error" not in result:
                    effect = result["treatment_effect"]
                    if isinstance(effect, dict) and "avg_treatment_effect" in effect:
                        valid_effects.append(effect["avg_treatment_effect"])

            if valid_effects:
                comparison_data.append(
                    {
                        "Method": "Synthetic Control (Average)",
                        "Treatment_Effect": np.mean(valid_effects),
                        "Standard_Error": np.std(valid_effects),
                        "P_Value": np.nan,
                    }
                )

        if not comparison_data:
            return pd.DataFrame(
                columns=["Method", "Treatment_Effect", "Standard_Error", "P_Value"]
            )

        return pd.DataFrame(comparison_data)

    def run_robustness_analysis(self) -> dict[str, Any]:
        """Run comprehensive robustness analysis.

        Returns:
            Dict with robustness test results
        """
        print("Running Comprehensive Robustness Analysis...")

        robustness = RobustnessTests(
            data=self.data, unit_col="country", time_col="year"
        )

        robustness_results = {}

        # Define simple DiD estimator for robustness testing
        def simple_did_estimator(data):
            """Simple DiD estimator for robustness testing."""
            data = data.copy()
            data["treated"] = (data["treatment_cohort"] > 0).astype(int)
            data["post"] = 0

            # Create post-treatment indicator
            for country in data["country"].unique():
                country_data = data[data["country"] == country]
                if country_data["treatment_cohort"].iloc[0] > 0:
                    treatment_year = country_data["treatment_cohort"].iloc[0]
                    post_mask = (data["country"] == country) & (
                        data["year"] >= treatment_year
                    )
                    data.loc[post_mask, "post"] = 1

            data["treated_post"] = data["treated"] * data["post"]

            try:
                import statsmodels.api as sm

                y = data["mortality_rate"]
                X = data[["treated", "post", "treated_post", "gdp_log"]]
                X = sm.add_constant(X)
                model = sm.OLS(y, X).fit(
                    cov_type="cluster", cov_kwds={"groups": data["country"]}
                )
                return {
                    "att": model.params.get("treated_post", np.nan),
                    "se": model.bse.get("treated_post", np.nan),
                    "pvalue": model.pvalues.get("treated_post", np.nan),
                }
            except:
                return {"att": np.nan, "se": np.nan, "pvalue": np.nan}

        # 1. Sample sensitivity
        print("  Sample sensitivity analysis...")
        try:
            treatment_countries = self.data[self.data["treatment_cohort"] > 0][
                "country"
            ].unique()

            sensitivity = robustness.sample_sensitivity(
                estimator_func=simple_did_estimator,
                exclude_units=treatment_countries[
                    :2
                ].tolist(),  # Exclude top 2 treatment countries
                random_exclusions=True,
                bootstrap_samples=100,
            )

            robustness_results["sample_sensitivity"] = sensitivity

        except Exception as e:
            robustness_results["sample_sensitivity"] = {"error": str(e)}

        # 2. Specification sensitivity
        print("  Specification sensitivity analysis...")
        try:
            spec_variants = [
                {"outcome": "mortality_rate", "covariates": ["gdp_log"]},
                {
                    "outcome": "mortality_rate",
                    "covariates": ["gdp_log", "urban_percentage"],
                },
                {
                    "outcome": "mortality_rate",
                    "covariates": [
                        "gdp_log",
                        "urban_percentage",
                        "healthcare_spending_log",
                    ],
                }
                if "healthcare_spending_log" in self.data.columns
                else {
                    "outcome": "mortality_rate",
                    "covariates": ["gdp_log", "urban_percentage"],
                },
            ]

            def spec_estimator(data, **kwargs):
                return simple_did_estimator(data)

            spec_results = robustness.specification_sensitivity(
                estimator_func=spec_estimator, specification_variants=spec_variants
            )

            robustness_results["specification_sensitivity"] = spec_results

        except Exception as e:
            robustness_results["specification_sensitivity"] = {"error": str(e)}

        # 3. Leave-one-out analysis
        print("  Leave-one-out analysis...")
        try:
            loo_results = robustness.leave_one_out_analysis(
                estimator_func=simple_did_estimator,
                by_unit=True,
                by_period=False,  # Skip period analysis for speed
            )

            robustness_results["leave_one_out"] = loo_results

        except Exception as e:
            robustness_results["leave_one_out"] = {"error": str(e)}

        return robustness_results

    def plot_method_comparison(self, save_path: str = None) -> None:
        """Plot comparison of treatment effects across methods.

        Args:
            save_path (str, optional): Path to save the plot
        """
        if not PLOTTING_AVAILABLE:
            print("Plotting not available. Install matplotlib and seaborn.")
            return

        comparison_df = self.compare_estimates()

        if comparison_df.empty:
            print("No valid estimates to plot.")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        # Create error bars
        methods = comparison_df["Method"]
        effects = comparison_df["Treatment_Effect"]
        errors = comparison_df["Standard_Error"].fillna(0)

        # Plot points with error bars
        ax.errorbar(
            range(len(methods)),
            effects,
            yerr=errors,
            fmt="o",
            capsize=5,
            capthick=2,
            markersize=8,
        )

        # Customize plot
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha="right")
        ax.set_ylabel("Treatment Effect on Mortality Rate")
        ax.set_title("Comparison of Treatment Effect Estimates Across Methods")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.7)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()


def run_advanced_comparison():
    """Run the advanced comparison workflow."""
    print("=" * 80)
    print("MPOWER CAUSAL INFERENCE ANALYSIS - ADVANCED COMPARISON")
    print("=" * 80)

    # Load data (using the sample data function from basic workflow)
    from basic_workflow import create_sample_data

    print("\n1. Data Preparation")
    print("-" * 50)

    raw_data = create_sample_data()

    # Prepare data
    prep = MPOWERDataPrep(raw_data, country_col="country", year_col="year")

    data_with_cohorts = prep.create_treatment_cohorts(
        mpower_col="mpower_total_score",
        treatment_definition="binary_threshold",
        threshold=25.0,
        min_years_high=2,
    )

    balanced_data = prep.balance_panel(method="drop_unbalanced", min_years=10)
    analysis_data = prep.prepare_for_analysis(
        outcome_cols=["mortality_rate"],
        control_cols=["gdp_log", "urban_percentage"],
        log_transform=["healthcare_spending"],
    )

    print(
        f"Analysis dataset: {len(analysis_data)} observations, {analysis_data['country'].nunique()} countries"
    )

    # Run method comparison
    print("\n2. Method Comparison")
    print("-" * 50)

    comparison = MethodComparison(analysis_data)

    # Run all methods
    all_results = comparison.run_all_methods()

    # Create comparison table
    comparison_table = comparison.compare_estimates()
    print("\nTreatment Effect Estimates:")
    print(comparison_table.round(4))

    # Plot comparison if plotting is available
    if PLOTTING_AVAILABLE:
        try:
            comparison.plot_method_comparison()
        except Exception as e:
            print(f"Plotting failed: {e}")

    # Run robustness analysis
    print("\n3. Robustness Analysis")
    print("-" * 50)

    robustness_results = comparison.run_robustness_analysis()

    # Summary of robustness results
    for test_name, result in robustness_results.items():
        print(f"\n{test_name.replace('_', ' ').title()}:")
        if "error" in result:
            print(f"  Error: {result['error']}")
        elif test_name == "sample_sensitivity":
            summary = result.get("summary", {})
            print(f"  Baseline effect: {summary.get('baseline_effect', 'N/A')}")
            print(f"  Effect std across samples: {summary.get('effect_std', 'N/A')}")
            print(
                f"  Robust to exclusions: {summary.get('robust_to_exclusions', 'N/A')}"
            )
        elif test_name == "leave_one_out":
            summary = result.get("summary", {})
            unit_analysis = summary.get("unit_analysis", {})
            if unit_analysis:
                print(
                    f"  Max effect change (unit exclusion): {unit_analysis.get('max_change', 'N/A')}"
                )
                print(
                    f"  Effect robust to unit exclusion: {unit_analysis.get('robust', 'N/A')}"
                )

    # Advanced diagnostics
    print("\n4. Advanced Diagnostics")
    print("-" * 50)

    # Event study parallel trends test
    if "event_study" in all_results and "error" not in all_results["event_study"]:
        parallel_trends = all_results["event_study"]["parallel_trends"]
        print(f"Parallel trends test: {parallel_trends['conclusion']}")
        print(
            f"Pre-treatment periods tested: {parallel_trends['n_pre_treatment_periods']}"
        )

        # Individual pre-treatment p-values
        pre_pvals = parallel_trends.get("pre_treatment_pvalues", [])
        significant_pre = sum(1 for p in pre_pvals if p < 0.05)
        print(f"Significant pre-treatment effects: {significant_pre}/{len(pre_pvals)}")

    # Treatment effect heterogeneity analysis
    print("\n5. Treatment Effect Heterogeneity")
    print("-" * 50)

    # Analyze treatment effects by treatment timing
    treatment_cohorts = analysis_data[analysis_data["treatment_cohort"] > 0][
        "treatment_cohort"
    ].unique()

    print("Treatment effects by cohort:")
    for cohort in sorted(treatment_cohorts):
        cohort_countries = analysis_data[analysis_data["treatment_cohort"] == cohort][
            "country"
        ].unique()
        print(f"  {cohort}: {list(cohort_countries)}")

    # Summary statistics by treatment status
    print("\n6. Descriptive Statistics")
    print("-" * 50)

    treated_data = analysis_data[analysis_data["treatment_cohort"] > 0]
    control_data = analysis_data[analysis_data["treatment_cohort"] == 0]

    print("Pre-treatment characteristics (means):")
    pre_period = analysis_data["year"] <= 2010  # Early period

    for var in ["mortality_rate", "gdp_log", "urban_percentage"]:
        if var in analysis_data.columns:
            treated_mean = treated_data[pre_period][var].mean()
            control_mean = control_data[pre_period][var].mean()
            print(f"  {var}: Treated={treated_mean:.3f}, Control={control_mean:.3f}")

    print("\n" + "=" * 80)
    print("ADVANCED COMPARISON COMPLETED")
    print("=" * 80)

    return {
        "analysis_data": analysis_data,
        "method_results": all_results,
        "comparison_table": comparison_table,
        "robustness_results": robustness_results,
    }


if __name__ == "__main__":
    # Run the advanced comparison workflow
    results = run_advanced_comparison()

    print("\nWorkflow completed successfully!")
    print(f"Methods tested: {len(results['method_results'])}")
    print(f"Robustness tests: {len(results['robustness_results'])}")

    # Optionally save results
    # results['comparison_table'].to_csv('examples/output/method_comparison.csv', index=False)
