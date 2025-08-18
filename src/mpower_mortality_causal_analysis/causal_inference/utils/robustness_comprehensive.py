"""Comprehensive Robustness Checks and Sensitivity Analysis for MPOWER Causal Analysis.

This module provides comprehensive robustness checking and sensitivity analysis
functions to validate the main Callaway & Sant'Anna DiD results using:
- Two-way fixed effects (TWFE) models
- Synthetic control methods
- Alternative sample restrictions
- Placebo tests
- Alternative treatment definitions
"""

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pandas import DataFrame

from ..methods.callaway_did import CallawayDiD
from ..methods.panel_methods import PanelFixedEffects
from ..methods.synthetic_control import SyntheticControl
from ..utils.event_study import EventStudyAnalysis


class RobustnessChecks:
    """Comprehensive robustness checking framework for MPOWER causal analysis.

    Implements multiple robustness checks to validate main DiD results:
    1. Two-way fixed effects comparisons
    2. Synthetic control methods
    3. Alternative sample restrictions
    4. Placebo tests
    5. Alternative treatment definitions
    6. Sensitivity to specification choices

    Parameters:
        data (DataFrame): MPOWER panel data
        country_col (str): Column name for country identifier
        year_col (str): Column name for year
        cohort_col (str): Column name for treatment cohort
        outcome_cols (List[str]): List of outcome variables

    Example:
        >>> robustness = RobustnessChecks(
        ...     data=analysis_data,
        ...     outcome_cols=['mort_lung_cancer_asr', 'mort_cvd_asr']
        ... )
        >>> results = robustness.run_all_checks(
        ...     main_outcome='mort_lung_cancer_asr',
        ...     covariates=['gdp_log', 'urban_pop_pct']
        ... )
        >>> robustness.create_robustness_report(results)
    """

    def __init__(
        self,
        data: DataFrame,
        country_col: str = "country",
        year_col: str = "year",
        cohort_col: str = "treatment_cohort",
        outcome_cols: list[str] | None = None,
        mpower_cols: list[str] | None = None,
        control_cols: list[str] | None = None,
        never_treated_value: int | float = 0,
    ):
        """Initialize robustness checking framework."""
        self.data = data.copy()
        self.country_col = country_col
        self.year_col = year_col
        self.cohort_col = cohort_col
        self.never_treated_value = never_treated_value

        # Set default column lists
        self.outcome_cols = outcome_cols or [
            "mort_lung_cancer_asr",
            "mort_cvd_asr",
            "mort_ihd_asr",
            "mort_copd_asr",
        ]

        self.mpower_cols = mpower_cols or [
            "mpower_total_score",
            "mpower_m",
            "mpower_p",
            "mpower_o",
            "mpower_w",
            "mpower_e",
            "mpower_r",
        ]

        self.control_cols = control_cols or [
            "gdp_pc_constant",
            "urban_pop_pct",
            "population_total",
            "edu_exp_pct_gdp",
        ]

        # Filter to existing columns
        self.outcome_cols = [col for col in self.outcome_cols if col in data.columns]
        self.mpower_cols = [col for col in self.mpower_cols if col in data.columns]
        self.control_cols = [col for col in self.control_cols if col in data.columns]

        # Validate data
        self._validate_data()

        # Set up plotting
        plt.style.use("default")
        sns.set_palette("husl")

    def _validate_data(self) -> None:
        """Validate data structure for robustness checks."""
        required_cols = [self.country_col, self.year_col, self.cohort_col]
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        if not self.outcome_cols:
            raise ValueError("No valid outcome columns found in data")

    def run_all_checks(
        self,
        main_outcome: str,
        covariates: list[str] | None = None,
        save_plots: bool = False,
        plot_dir: str = "robustness_plots",
        **kwargs,
    ) -> dict[str, Any]:
        """Run comprehensive robustness checks.

        Args:
            main_outcome (str): Primary outcome variable
            covariates (List[str], optional): Control variables
            save_plots (bool): Whether to save plots
            plot_dir (str): Directory for saving plots
            **kwargs: Additional arguments for specific checks

        Returns:
            Dict with all robustness check results
        """
        if save_plots:
            from pathlib import Path

            Path(plot_dir).mkdir(exist_ok=True)

        results = {}

        # 1. Main Callaway & Sant'Anna results (for comparison)
        results["main_callaway_santanna"] = self._run_main_callaway_santanna(
            main_outcome, covariates
        )

        # 2. Two-way fixed effects comparison
        results["twfe_comparison"] = self._run_twfe_comparison(main_outcome, covariates)

        # 3. Alternative DiD specifications
        results["alternative_did_specs"] = self._run_alternative_did_specs(
            main_outcome, covariates
        )

        # 4. Synthetic control analysis
        results["synthetic_control"] = self._run_synthetic_control_analysis(
            main_outcome, covariates
        )

        # 5. Sample robustness checks
        results["sample_robustness"] = self._run_sample_robustness(
            main_outcome, covariates
        )

        # 6. Placebo tests
        results["placebo_tests"] = self._run_placebo_tests(main_outcome, covariates)

        # 7. Alternative treatment definitions
        results["alternative_treatments"] = self._run_alternative_treatments(
            main_outcome, covariates
        )

        # 8. Specification sensitivity
        results["specification_sensitivity"] = self._run_specification_sensitivity(
            main_outcome, covariates
        )

        # 9. Cross-outcome validation
        if len(self.outcome_cols) > 1:
            results["cross_outcome_validation"] = self._run_cross_outcome_validation(
                covariates
            )

        return results

    def _run_main_callaway_santanna(
        self, outcome: str, covariates: list[str] | None
    ) -> dict[str, Any]:
        """Run main Callaway & Sant'Anna analysis for comparison."""
        try:
            cs_did = CallawayDiD(
                data=self.data,
                cohort_col=self.cohort_col,
                unit_col=self.country_col,
                time_col=self.year_col,
                never_treated_value=self.never_treated_value,
            )

            cs_did.fit(outcome=outcome, covariates=covariates)

            # Get aggregated results
            simple_att = cs_did.aggregate("simple")
            event_study = cs_did.aggregate("event")

            return {
                "method": "callaway_santanna",
                "backend": cs_did.get_backend_info()["backend_used"],
                "simple_att": simple_att,
                "event_study": event_study,
                "summary": cs_did.summary(),
                "status": "success",
            }

        except Exception as e:
            return {"method": "callaway_santanna", "status": "failed", "error": str(e)}

    def _run_twfe_comparison(
        self, outcome: str, covariates: list[str] | None
    ) -> dict[str, Any]:
        """Compare with traditional two-way fixed effects."""
        results = {}

        # Standard TWFE
        try:
            twfe = PanelFixedEffects(
                data=self.data, unit_col=self.country_col, time_col=self.year_col
            )

            # Create treatment variables
            data_with_treatment = self.data.copy()
            data_with_treatment["ever_treated"] = (
                data_with_treatment[self.cohort_col] != self.never_treated_value
            ).astype(int)

            data_with_treatment["post_treatment"] = 0
            for cohort in data_with_treatment[self.cohort_col].unique():
                if cohort != self.never_treated_value:
                    mask = (data_with_treatment[self.cohort_col] == cohort) & (
                        data_with_treatment[self.year_col] >= cohort
                    )
                    data_with_treatment.loc[mask, "post_treatment"] = 1

            data_with_treatment["treated_post"] = (
                data_with_treatment["ever_treated"]
                * data_with_treatment["post_treatment"]
            )

            # Update data in TWFE object
            twfe.data = data_with_treatment

            # Fit standard TWFE model
            twfe_covariates = ["treated_post"] + (covariates or [])

            twfe.fit(outcome=outcome, covariates=twfe_covariates, method="twfe")

            coef_table = twfe.get_coefficients()
            treated_post_coef = (
                coef_table.loc["treated_post"]
                if "treated_post" in coef_table.index
                else None
            )

            results["standard_twfe"] = {
                "coefficient": treated_post_coef["coefficient"]
                if treated_post_coef is not None
                else None,
                "std_error": treated_post_coef["std_error"]
                if treated_post_coef is not None
                else None,
                "p_value": treated_post_coef["p_value"]
                if treated_post_coef is not None
                else None,
                "summary": twfe.summary(),
                "full_results": coef_table.to_dict(),
            }

        except Exception as e:
            results["standard_twfe"] = {"status": "failed", "error": str(e)}

        # TWFE with cohort-specific trends
        try:
            # Add linear trends for each cohort
            data_with_trends = data_with_treatment.copy()

            for cohort in data_with_trends[self.cohort_col].unique():
                if cohort != self.never_treated_value:
                    cohort_mask = data_with_trends[self.cohort_col] == cohort
                    data_with_trends.loc[cohort_mask, f"trend_cohort_{cohort}"] = (
                        data_with_trends.loc[cohort_mask, self.year_col] - cohort
                    )
                    data_with_trends[f"trend_cohort_{cohort}"] = data_with_trends[
                        f"trend_cohort_{cohort}"
                    ].fillna(0)

            # Fit TWFE with trends
            trend_vars = [
                col
                for col in data_with_trends.columns
                if col.startswith("trend_cohort_")
            ]
            twfe_trend_covariates = ["treated_post"] + trend_vars + (covariates or [])

            twfe_trends = PanelFixedEffects(
                data=data_with_trends, unit_col=self.country_col, time_col=self.year_col
            )

            twfe_trends.fit(
                outcome=outcome, covariates=twfe_trend_covariates, method="twfe"
            )

            coef_table_trends = twfe_trends.get_coefficients()
            treated_post_trends = (
                coef_table_trends.loc["treated_post"]
                if "treated_post" in coef_table_trends.index
                else None
            )

            results["twfe_with_trends"] = {
                "coefficient": treated_post_trends["coefficient"]
                if treated_post_trends is not None
                else None,
                "std_error": treated_post_trends["std_error"]
                if treated_post_trends is not None
                else None,
                "p_value": treated_post_trends["p_value"]
                if treated_post_trends is not None
                else None,
                "summary": twfe_trends.summary(),
            }

        except Exception as e:
            results["twfe_with_trends"] = {"status": "failed", "error": str(e)}

        return results

    def _run_alternative_did_specs(
        self, outcome: str, covariates: list[str] | None
    ) -> dict[str, Any]:
        """Test alternative DiD specifications."""
        results = {}

        # Different control groups
        for control_group in ["nevertreated", "notyettreated"]:
            try:
                cs_did = CallawayDiD(
                    data=self.data,
                    cohort_col=self.cohort_col,
                    unit_col=self.country_col,
                    time_col=self.year_col,
                    never_treated_value=self.never_treated_value,
                )

                cs_did.fit(
                    outcome=outcome, covariates=covariates, control_group=control_group
                )

                simple_att = cs_did.aggregate("simple")

                results[f"control_group_{control_group}"] = {
                    "simple_att": simple_att,
                    "status": "success",
                }

            except Exception as e:
                results[f"control_group_{control_group}"] = {
                    "status": "failed",
                    "error": str(e),
                }

        # Different anticipation periods
        for anticipation in [0, 1, 2]:
            try:
                cs_did = CallawayDiD(
                    data=self.data,
                    cohort_col=self.cohort_col,
                    unit_col=self.country_col,
                    time_col=self.year_col,
                    never_treated_value=self.never_treated_value,
                )

                cs_did.fit(
                    outcome=outcome, covariates=covariates, anticipation=anticipation
                )

                simple_att = cs_did.aggregate("simple")

                results[f"anticipation_{anticipation}"] = {
                    "simple_att": simple_att,
                    "status": "success",
                }

            except Exception as e:
                results[f"anticipation_{anticipation}"] = {
                    "status": "failed",
                    "error": str(e),
                }

        return results

    def _run_synthetic_control_analysis(
        self, outcome: str, covariates: list[str] | None
    ) -> dict[str, Any]:
        """Run synthetic control analysis for key treated countries."""
        results = {}

        # Get treated countries and their treatment years
        treated_data = self.data[self.data[self.cohort_col] != self.never_treated_value]

        # Select a few countries with good data coverage for SC analysis
        country_data_counts = treated_data.groupby(self.country_col).size()
        top_countries = country_data_counts.nlargest(5).index.tolist()

        for country in top_countries[
            :3
        ]:  # Limit to 3 countries for computational efficiency
            try:
                country_treatment_year = treated_data[
                    treated_data[self.country_col] == country
                ][self.cohort_col].iloc[0]

                # Check if we have sufficient pre-treatment data
                pre_treatment_data = self.data[
                    (self.data[self.country_col] == country)
                    & (self.data[self.year_col] < country_treatment_year)
                ]

                if len(pre_treatment_data) < 3:
                    results[f"sc_{country}"] = {
                        "status": "skipped",
                        "reason": "insufficient_pre_treatment_data",
                    }
                    continue

                # Run synthetic control
                sc = SyntheticControl(
                    data=self.data,
                    unit_col=self.country_col,
                    time_col=self.year_col,
                    treatment_time=country_treatment_year,
                    treated_unit=country,
                )

                sc.fit(outcome=outcome, predictors=covariates)

                treatment_effect = sc.get_treatment_effect()
                weights = sc.get_weights()

                results[f"sc_{country}"] = {
                    "treatment_year": country_treatment_year,
                    "treatment_effect": treatment_effect,
                    "weights": weights,
                    "summary": sc.summary(),
                    "status": "success",
                }

            except Exception as e:
                results[f"sc_{country}"] = {"status": "failed", "error": str(e)}

        return results

    def _run_sample_robustness(
        self, outcome: str, covariates: list[str] | None
    ) -> dict[str, Any]:
        """Test robustness to sample restrictions."""
        results = {}

        # Balanced panel only
        try:
            # Get countries with observations in all years
            country_year_counts = self.data.groupby(self.country_col)[
                self.year_col
            ].nunique()
            max_years = country_year_counts.max()
            balanced_countries = country_year_counts[
                country_year_counts == max_years
            ].index

            balanced_data = self.data[
                self.data[self.country_col].isin(balanced_countries)
            ]

            cs_did_balanced = CallawayDiD(
                data=balanced_data,
                cohort_col=self.cohort_col,
                unit_col=self.country_col,
                time_col=self.year_col,
                never_treated_value=self.never_treated_value,
            )

            cs_did_balanced.fit(outcome=outcome, covariates=covariates)
            simple_att_balanced = cs_did_balanced.aggregate("simple")

            results["balanced_panel"] = {
                "simple_att": simple_att_balanced,
                "n_countries": len(balanced_countries),
                "status": "success",
            }

        except Exception as e:
            results["balanced_panel"] = {"status": "failed", "error": str(e)}

        # Exclude early adopters
        try:
            treatment_years = sorted(
                [
                    year
                    for year in self.data[self.cohort_col].unique()
                    if year != self.never_treated_value
                ]
            )

            if len(treatment_years) > 2:
                early_year = treatment_years[0]
                later_data = self.data[
                    (self.data[self.cohort_col] != early_year)
                    | (self.data[self.cohort_col] == self.never_treated_value)
                ]

                cs_did_later = CallawayDiD(
                    data=later_data,
                    cohort_col=self.cohort_col,
                    unit_col=self.country_col,
                    time_col=self.year_col,
                    never_treated_value=self.never_treated_value,
                )

                cs_did_later.fit(outcome=outcome, covariates=covariates)
                simple_att_later = cs_did_later.aggregate("simple")

                results["exclude_early_adopters"] = {
                    "simple_att": simple_att_later,
                    "excluded_year": early_year,
                    "status": "success",
                }

        except Exception as e:
            results["exclude_early_adopters"] = {"status": "failed", "error": str(e)}

        # High-income countries only (if GDP data available)
        if "gdp_pc_constant" in self.data.columns:
            try:
                median_gdp = self.data["gdp_pc_constant"].median()
                high_income_data = self.data[self.data["gdp_pc_constant"] >= median_gdp]

                cs_did_high_income = CallawayDiD(
                    data=high_income_data,
                    cohort_col=self.cohort_col,
                    unit_col=self.country_col,
                    time_col=self.year_col,
                    never_treated_value=self.never_treated_value,
                )

                cs_did_high_income.fit(outcome=outcome, covariates=covariates)
                simple_att_high_income = cs_did_high_income.aggregate("simple")

                results["high_income_only"] = {
                    "simple_att": simple_att_high_income,
                    "gdp_threshold": median_gdp,
                    "status": "success",
                }

            except Exception as e:
                results["high_income_only"] = {"status": "failed", "error": str(e)}

        return results

    def _run_placebo_tests(
        self, outcome: str, covariates: list[str] | None
    ) -> dict[str, Any]:
        """Run placebo tests for identification validation."""
        results = {}

        # Event study placebo (test for pre-trends)
        try:
            event_study = EventStudyAnalysis(
                data=self.data,
                unit_col=self.country_col,
                time_col=self.year_col,
                treatment_col=self.cohort_col,
                never_treated_value=self.never_treated_value,
            )

            parallel_trends = event_study.comprehensive_parallel_trends_analysis(
                outcome=outcome, covariates=covariates
            )

            results["parallel_trends_test"] = parallel_trends

        except Exception as e:
            results["parallel_trends_test"] = {"status": "failed", "error": str(e)}

        # Artificial treatment timing
        try:
            # Create artificial treatment 2 years before actual treatment
            placebo_data = self.data.copy()
            placebo_data["placebo_cohort"] = placebo_data[self.cohort_col].copy()

            # Shift treatment years back by 2
            treatment_mask = placebo_data["placebo_cohort"] != self.never_treated_value
            placebo_data.loc[treatment_mask, "placebo_cohort"] -= 2

            # Only keep observations before actual treatment
            actual_min_treatment = self.data[
                self.data[self.cohort_col] != self.never_treated_value
            ][self.cohort_col].min()

            placebo_data = placebo_data[
                placebo_data[self.year_col] < actual_min_treatment
            ]

            if len(placebo_data) > 0:
                cs_did_placebo = CallawayDiD(
                    data=placebo_data,
                    cohort_col="placebo_cohort",
                    unit_col=self.country_col,
                    time_col=self.year_col,
                    never_treated_value=self.never_treated_value,
                )

                cs_did_placebo.fit(outcome=outcome, covariates=covariates)
                placebo_att = cs_did_placebo.aggregate("simple")

                results["artificial_treatment_timing"] = {
                    "placebo_att": placebo_att,
                    "status": "success",
                }
            else:
                results["artificial_treatment_timing"] = {
                    "status": "skipped",
                    "reason": "insufficient_pre_treatment_data",
                }

        except Exception as e:
            results["artificial_treatment_timing"] = {
                "status": "failed",
                "error": str(e),
            }

        return results

    def _run_alternative_treatments(
        self, outcome: str, covariates: list[str] | None
    ) -> dict[str, Any]:
        """Test alternative treatment definitions."""
        results = {}

        # Higher threshold for MPOWER score
        if "mpower_total_score" in self.data.columns:
            try:
                # Create alternative treatment with higher threshold
                alt_data = self.data.copy()
                alt_data["alt_treatment_cohort"] = 0

                # Use threshold of 27 instead of 25
                high_threshold = 27

                for country in alt_data[self.country_col].unique():
                    country_data = alt_data[alt_data[self.country_col] == country]
                    high_score_years = country_data[
                        country_data["mpower_total_score"] >= high_threshold
                    ][self.year_col]

                    if len(high_score_years) >= 2:  # Sustained high score
                        first_high_year = high_score_years.min()
                        country_mask = alt_data[self.country_col] == country
                        alt_data.loc[country_mask, "alt_treatment_cohort"] = (
                            first_high_year
                        )

                cs_did_alt = CallawayDiD(
                    data=alt_data,
                    cohort_col="alt_treatment_cohort",
                    unit_col=self.country_col,
                    time_col=self.year_col,
                    never_treated_value=0,
                )

                cs_did_alt.fit(outcome=outcome, covariates=covariates)
                alt_att = cs_did_alt.aggregate("simple")

                results["higher_mpower_threshold"] = {
                    "threshold": high_threshold,
                    "alternative_att": alt_att,
                    "status": "success",
                }

            except Exception as e:
                results["higher_mpower_threshold"] = {
                    "status": "failed",
                    "error": str(e),
                }

        # Component-based treatment (focus on specific MPOWER components)
        component_cols = [
            col
            for col in self.mpower_cols
            if col.startswith("mpower_") and col != "mpower_total_score"
        ]

        if component_cols:
            try:
                # Treatment based on strong implementation of multiple components
                comp_data = self.data.copy()
                comp_data["component_treatment_cohort"] = 0

                for country in comp_data[self.country_col].unique():
                    country_data = comp_data[comp_data[self.country_col] == country]

                    # Count years with multiple components >= 3
                    component_scores = country_data[component_cols]
                    strong_years = country_data[
                        (component_scores >= 3).sum(axis=1)
                        >= 4  # At least 4 components strong
                    ][self.year_col]

                    if len(strong_years) >= 2:
                        first_strong_year = strong_years.min()
                        country_mask = comp_data[self.country_col] == country
                        comp_data.loc[country_mask, "component_treatment_cohort"] = (
                            first_strong_year
                        )

                cs_did_comp = CallawayDiD(
                    data=comp_data,
                    cohort_col="component_treatment_cohort",
                    unit_col=self.country_col,
                    time_col=self.year_col,
                    never_treated_value=0,
                )

                cs_did_comp.fit(outcome=outcome, covariates=covariates)
                comp_att = cs_did_comp.aggregate("simple")

                results["component_based_treatment"] = {
                    "component_att": comp_att,
                    "status": "success",
                }

            except Exception as e:
                results["component_based_treatment"] = {
                    "status": "failed",
                    "error": str(e),
                }

        return results

    def _run_specification_sensitivity(
        self, outcome: str, covariates: list[str] | None
    ) -> dict[str, Any]:
        """Test sensitivity to specification choices."""
        results = {}

        # Different covariate sets
        covariate_sets = {
            "minimal": ["gdp_pc_constant"]
            if "gdp_pc_constant" in self.control_cols
            else [],
            "baseline": covariates or [],
            "extended": self.control_cols,
        }

        for set_name, cov_set in covariate_sets.items():
            try:
                # Filter to existing columns
                valid_covs = [cov for cov in cov_set if cov in self.data.columns]

                cs_did = CallawayDiD(
                    data=self.data,
                    cohort_col=self.cohort_col,
                    unit_col=self.country_col,
                    time_col=self.year_col,
                    never_treated_value=self.never_treated_value,
                )

                cs_did.fit(
                    outcome=outcome, covariates=valid_covs if valid_covs else None
                )
                simple_att = cs_did.aggregate("simple")

                results[f"covariates_{set_name}"] = {
                    "covariates": valid_covs,
                    "simple_att": simple_att,
                    "status": "success",
                }

            except Exception as e:
                results[f"covariates_{set_name}"] = {
                    "status": "failed",
                    "error": str(e),
                }

        return results

    def _run_cross_outcome_validation(
        self, covariates: list[str] | None
    ) -> dict[str, Any]:
        """Test consistency across multiple outcomes."""
        results = {}

        for outcome in self.outcome_cols:
            try:
                cs_did = CallawayDiD(
                    data=self.data,
                    cohort_col=self.cohort_col,
                    unit_col=self.country_col,
                    time_col=self.year_col,
                    never_treated_value=self.never_treated_value,
                )

                cs_did.fit(outcome=outcome, covariates=covariates)
                simple_att = cs_did.aggregate("simple")

                results[outcome] = {"simple_att": simple_att, "status": "success"}

            except Exception as e:
                results[outcome] = {"status": "failed", "error": str(e)}

        return results

    def create_robustness_report(
        self, results: dict[str, Any], save_path: str | None = None
    ) -> str:
        """Create comprehensive robustness check report.

        Args:
            results (Dict): Results from run_all_checks()
            save_path (str, optional): Path to save the report

        Returns:
            str: Formatted report text
        """
        report = []
        report.append("# MPOWER Causal Analysis - Robustness Checks Report")
        report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Main results summary
        main_results = results.get("main_callaway_santanna", {})
        if main_results.get("status") == "success":
            main_att = main_results.get("simple_att", {})
            report.append("## Main Results (Callaway & Sant'Anna)")
            report.append(f"- **ATT Estimate**: {main_att.get('att', 'N/A'):.4f}")
            report.append(f"- **Standard Error**: {main_att.get('se', 'N/A'):.4f}")
            report.append(f"- **P-value**: {main_att.get('pvalue', 'N/A'):.4f}")
            report.append(f"- **Backend**: {main_results.get('backend', 'N/A')}")
            report.append("")

        # TWFE comparison
        twfe_results = results.get("twfe_comparison", {})
        report.append("## Two-Way Fixed Effects Comparison")

        if "standard_twfe" in twfe_results:
            std_twfe = twfe_results["standard_twfe"]
            if "coefficient" in std_twfe:
                report.append(
                    f"- **Standard TWFE**: {std_twfe['coefficient']:.4f} (SE: {std_twfe['std_error']:.4f})"
                )

        if "twfe_with_trends" in twfe_results:
            trend_twfe = twfe_results["twfe_with_trends"]
            if "coefficient" in trend_twfe:
                report.append(
                    f"- **TWFE with Trends**: {trend_twfe['coefficient']:.4f} (SE: {trend_twfe['std_error']:.4f})"
                )

        report.append("")

        # Alternative specifications
        alt_specs = results.get("alternative_did_specs", {})
        if alt_specs:
            report.append("## Alternative DiD Specifications")
            for spec_name, spec_results in alt_specs.items():
                if spec_results.get("status") == "success":
                    att = spec_results.get("simple_att", {})
                    report.append(
                        f"- **{spec_name.replace('_', ' ').title()}**: {att.get('att', 'N/A'):.4f}"
                    )
            report.append("")

        # Sample robustness
        sample_rob = results.get("sample_robustness", {})
        if sample_rob:
            report.append("## Sample Robustness")
            for sample_name, sample_results in sample_rob.items():
                if sample_results.get("status") == "success":
                    att = sample_results.get("simple_att", {})
                    report.append(
                        f"- **{sample_name.replace('_', ' ').title()}**: {att.get('att', 'N/A'):.4f}"
                    )
            report.append("")

        # Placebo tests
        placebo_results = results.get("placebo_tests", {})
        if placebo_results:
            report.append("## Placebo Tests")

            pt_test = placebo_results.get("parallel_trends_test", {})
            if "overall_assessment" in pt_test:
                assessment = pt_test["overall_assessment"]
                report.append(
                    f"- **Parallel Trends Assessment**: {assessment.get('assessment', 'N/A')}"
                )
                report.append(
                    f"- **Confidence**: {assessment.get('confidence', 'N/A')}"
                )

            artificial_test = placebo_results.get("artificial_treatment_timing", {})
            if artificial_test.get("status") == "success":
                placebo_att = artificial_test.get("placebo_att", {})
                report.append(
                    f"- **Artificial Treatment Timing**: {placebo_att.get('att', 'N/A'):.4f}"
                )

            report.append("")

        # Cross-outcome validation
        cross_outcome = results.get("cross_outcome_validation", {})
        if cross_outcome:
            report.append("## Cross-Outcome Validation")
            for outcome, outcome_results in cross_outcome.items():
                if outcome_results.get("status") == "success":
                    att = outcome_results.get("simple_att", {})
                    report.append(f"- **{outcome}**: {att.get('att', 'N/A'):.4f}")
            report.append("")

        # Overall assessment
        report.append("## Overall Robustness Assessment")
        assessment = self._assess_overall_robustness(results)
        report.append(f"**Robustness Score**: {assessment['score']}/10")
        report.append(f"**Assessment**: {assessment['assessment']}")
        report.append("")
        report.append("**Key Findings**:")
        for finding in assessment["findings"]:
            report.append(f"- {finding}")
        report.append("")

        report_text = "\n".join(report)

        if save_path:
            with open(save_path, "w") as f:
                f.write(report_text)

        return report_text

    def _assess_overall_robustness(self, results: dict[str, Any]) -> dict[str, Any]:
        """Assess overall robustness of the main results."""
        score = 0
        max_score = 10
        findings = []

        # Check main results
        main_results = results.get("main_callaway_santanna", {})
        if main_results.get("status") == "success":
            score += 2
            findings.append("Main Callaway & Sant'Anna analysis successful")

        # Check TWFE consistency
        twfe_results = results.get("twfe_comparison", {})
        if twfe_results.get("standard_twfe", {}).get("coefficient") is not None:
            main_att = main_results.get("simple_att", {}).get("att", 0)
            twfe_att = twfe_results["standard_twfe"]["coefficient"]

            if abs(main_att - twfe_att) / abs(main_att) < 0.5:  # Within 50%
                score += 1
                findings.append("TWFE results consistent with main results")
            else:
                findings.append("TWFE results differ substantially from main results")

        # Check parallel trends
        placebo_results = results.get("placebo_tests", {})
        pt_test = placebo_results.get("parallel_trends_test", {})
        if "overall_assessment" in pt_test:
            assessment = pt_test["overall_assessment"]["assessment"]
            if assessment in ["strong_support", "weak_support"]:
                score += 2
                findings.append("Parallel trends assumption supported")
            else:
                findings.append("Parallel trends assumption questionable")

        # Check sample robustness
        sample_rob = results.get("sample_robustness", {})
        consistent_samples = 0
        total_samples = 0

        for _sample_name, sample_results in sample_rob.items():
            if sample_results.get("status") == "success":
                total_samples += 1
                att = sample_results.get("simple_att", {}).get("att", 0)
                main_att = main_results.get("simple_att", {}).get("att", 0)

                if abs(att - main_att) / abs(main_att) < 0.3:  # Within 30%
                    consistent_samples += 1

        if total_samples > 0:
            consistency_rate = consistent_samples / total_samples
            if consistency_rate >= 0.7:
                score += 2
                findings.append(
                    f"Results consistent across {consistency_rate:.0%} of sample restrictions"
                )
            else:
                findings.append(
                    f"Results vary across sample restrictions ({consistency_rate:.0%} consistent)"
                )

        # Check cross-outcome consistency
        cross_outcome = results.get("cross_outcome_validation", {})
        significant_outcomes = 0
        total_outcomes = 0

        for _outcome, outcome_results in cross_outcome.items():
            if outcome_results.get("status") == "success":
                total_outcomes += 1
                att_pval = outcome_results.get("simple_att", {}).get("pvalue", 1)
                if att_pval < 0.05:
                    significant_outcomes += 1

        if total_outcomes > 1:
            if significant_outcomes / total_outcomes >= 0.5:
                score += 1
                findings.append(
                    f"Effects significant for {significant_outcomes}/{total_outcomes} outcomes"
                )
            else:
                findings.append(
                    f"Effects only significant for {significant_outcomes}/{total_outcomes} outcomes"
                )

        # Check alternative specifications
        alt_specs = results.get("alternative_did_specs", {})
        consistent_specs = 0
        total_specs = 0

        for _spec_name, spec_results in alt_specs.items():
            if spec_results.get("status") == "success":
                total_specs += 1
                att = spec_results.get("simple_att", {}).get("att", 0)
                main_att = main_results.get("simple_att", {}).get("att", 0)

                if abs(att - main_att) / abs(main_att) < 0.4:  # Within 40%
                    consistent_specs += 1

        if total_specs > 0:
            spec_consistency = consistent_specs / total_specs
            if spec_consistency >= 0.7:
                score += 1
                findings.append(
                    f"Results consistent across {spec_consistency:.0%} of alternative specifications"
                )

        # Overall assessment
        if score >= 8:
            assessment = "Highly robust - results are very reliable"
        elif score >= 6:
            assessment = "Moderately robust - results are generally reliable"
        elif score >= 4:
            assessment = "Somewhat robust - results have some support but need caution"
        else:
            assessment = "Low robustness - results should be interpreted with caution"

        return {
            "score": score,
            "max_score": max_score,
            "assessment": assessment,
            "findings": findings,
        }

    def plot_robustness_comparison(
        self, results: dict[str, Any], save_path: str | None = None
    ) -> plt.Figure:
        """Create visualization comparing robustness check results.

        Args:
            results (Dict): Results from run_all_checks()
            save_path (str, optional): Path to save the plot

        Returns:
            matplotlib Figure object
        """
        # Extract point estimates and confidence intervals
        estimates = []

        # Main result
        main_results = results.get("main_callaway_santanna", {})
        if main_results.get("status") == "success":
            main_att = main_results.get("simple_att", {})
            estimates.append(
                {
                    "estimate": main_att.get("att", 0),
                    "se": main_att.get("se", 0),
                    "label": "Main CS DiD",
                    "color": "red",
                }
            )

        # TWFE comparison
        twfe_results = results.get("twfe_comparison", {})
        if (
            "standard_twfe" in twfe_results
            and "coefficient" in twfe_results["standard_twfe"]
        ):
            std_twfe = twfe_results["standard_twfe"]
            estimates.append(
                {
                    "estimate": std_twfe["coefficient"],
                    "se": std_twfe["std_error"],
                    "label": "Standard TWFE",
                    "color": "blue",
                }
            )

        # Alternative specifications
        alt_specs = results.get("alternative_did_specs", {})
        for spec_name, spec_results in alt_specs.items():
            if spec_results.get("status") == "success":
                att = spec_results.get("simple_att", {})
                estimates.append(
                    {
                        "estimate": att.get("att", 0),
                        "se": att.get("se", 0),
                        "label": spec_name.replace("_", " ").title(),
                        "color": "green",
                    }
                )

        # Sample robustness
        sample_rob = results.get("sample_robustness", {})
        for sample_name, sample_results in sample_rob.items():
            if sample_results.get("status") == "success":
                att = sample_results.get("simple_att", {})
                estimates.append(
                    {
                        "estimate": att.get("att", 0),
                        "se": att.get("se", 0),
                        "label": sample_name.replace("_", " ").title(),
                        "color": "orange",
                    }
                )

        if not estimates:
            raise ValueError("No successful robustness check results to plot")

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))

        y_positions = range(len(estimates))

        for i, est in enumerate(estimates):
            estimate = est["estimate"]
            se = est["se"]
            color = est["color"]

            # Plot point estimate
            ax.scatter(estimate, i, color=color, s=100, zorder=3)

            # Plot confidence interval
            ci_lower = estimate - 1.96 * se
            ci_upper = estimate + 1.96 * se
            ax.plot([ci_lower, ci_upper], [i, i], color=color, linewidth=2, alpha=0.7)

        # Customize plot
        ax.set_yticks(y_positions)
        ax.set_yticklabels([est["label"] for est in estimates])
        ax.set_xlabel("Treatment Effect Estimate")
        ax.set_title("Robustness Check Comparison\n95% Confidence Intervals")

        # Add vertical line at zero
        ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)

        # Add grid
        ax.grid(True, alpha=0.3)

        # Invert y-axis to have main result at top
        ax.invert_yaxis()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig
