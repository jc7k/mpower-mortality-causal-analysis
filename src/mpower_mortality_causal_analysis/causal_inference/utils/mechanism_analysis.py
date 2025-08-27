"""Mechanism Analysis for MPOWER Component Decomposition.

This module provides comprehensive analysis of individual MPOWER components
to understand which specific tobacco control policies drive mortality effects.

The MPOWER framework includes six components:
- M: Monitor tobacco use and prevention policies
- P: Protect from tobacco smoke
- O: Offer help to quit tobacco use
- W: Warn about the dangers of tobacco
- E: Enforce bans on tobacco advertising, promotion and sponsorship
- R: Raise taxes on tobacco

This analysis enables:
1. Component-specific treatment effect estimation
2. Relative importance ranking of MPOWER components
3. Policy prioritization for maximum health impact
4. Dose-response relationships within components
"""

import logging
import warnings

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    warnings.warn(
        "Plotting packages not available. Mechanism analysis visualization disabled.",
        stacklevel=2,
    )

# Import causal inference methods for component analysis
from mpower_mortality_causal_analysis.causal_inference.methods.callaway_did import (
    CallawayDiD,
)
from mpower_mortality_causal_analysis.causal_inference.methods.synthetic_control import (
    MPOWERSyntheticControl,
)

# MPOWER component definitions and thresholds
MPOWER_COMPONENTS = {
    "M": {
        "name": "Monitor",
        "description": "Monitor tobacco use and prevention policies",
        "max_score": 5,
        "high_threshold": 4,  # High implementation threshold
    },
    "P": {
        "name": "Protect",
        "description": "Protect from tobacco smoke",
        "max_score": 5,
        "high_threshold": 4,
    },
    "O": {
        "name": "Offer",
        "description": "Offer help to quit tobacco use",
        "max_score": 5,
        "high_threshold": 4,
    },
    "W": {
        "name": "Warn",
        "description": "Warn about the dangers of tobacco",
        "max_score": 4,
        "high_threshold": 3,
    },
    "E": {
        "name": "Enforce",
        "description": "Enforce bans on tobacco advertising, promotion and sponsorship",
        "max_score": 5,
        "high_threshold": 4,
    },
    "R": {
        "name": "Raise",
        "description": "Raise taxes on tobacco",
        "max_score": 5,
        "high_threshold": 4,
    },
}

# Define significance level constant
SIGNIFICANCE_LEVEL = 0.05


class MPOWERMechanismAnalysis:
    """Comprehensive mechanism analysis for MPOWER component decomposition.

    This class implements component-specific causal inference to understand
    which MPOWER policies drive mortality effects and their relative importance.

    Parameters:
        data (pd.DataFrame): Panel data with MPOWER components and outcomes
        unit_col (str): Country identifier column
        time_col (str): Year identifier column
        component_cols (Dict[str, str]): Mapping of component names to data columns

    Example:
        >>> mechanism = MPOWERMechanismAnalysis(
        ...     data=df,
        ...     component_cols={
        ...         'M': 'mpower_m_score',
        ...         'P': 'mpower_p_score',
        ...         # ... other components
        ...     }
        ... )
        >>> results = mechanism.run_component_analysis(
        ...     outcome='lung_cancer_mortality_rate'
        ... )
    """

    def __init__(
        self,
        data: pd.DataFrame,
        unit_col: str = "country",
        time_col: str = "year",
        component_cols: dict[str, str] | None = None,
        control_vars: list[str] | None = None,
    ):
        """Initialize mechanism analysis."""
        self.data = data.copy()
        self.unit_col = unit_col
        self.time_col = time_col
        self.control_vars = control_vars or []

        # Set up component columns (with fallback for missing data)
        if component_cols is None:
            # Try to auto-detect component columns
            self.component_cols = self._detect_component_columns()
        else:
            self.component_cols = component_cols

        # Validate component columns exist
        missing_cols = [
            col for col in self.component_cols.values() if col not in self.data.columns
        ]
        if missing_cols:
            warnings.warn(
                f"Missing component columns: {missing_cols}. "
                "Component analysis will use simulated data for demonstration.",
                stacklevel=2,
            )
            self._create_simulated_components()

        # Create component treatment indicators
        self._create_component_treatments()

    def _detect_component_columns(self) -> dict[str, str]:
        """Auto-detect MPOWER component columns in data."""
        component_cols = {}

        # Common patterns for MPOWER component columns
        patterns = {
            "M": ["mpower_m", "m_score", "monitor"],
            "P": ["mpower_p", "p_score", "protect"],
            "O": ["mpower_o", "o_score", "offer"],
            "W": ["mpower_w", "w_score", "warn"],
            "E": ["mpower_e", "e_score", "enforce"],
            "R": ["mpower_r", "r_score", "raise"],
        }

        for component, possible_names in patterns.items():
            for pattern in possible_names:
                matching_cols = [
                    col for col in self.data.columns if pattern in col.lower()
                ]
                if matching_cols:
                    component_cols[component] = matching_cols[0]
                    break

        return component_cols

    def _create_simulated_components(self) -> None:
        """Create simulated MPOWER component data for demonstration."""
        np.random.seed(42)  # For reproducibility

        # Get unique countries and years
        countries = self.data[self.unit_col].unique()
        years = sorted(self.data[self.time_col].unique())

        # Create component columns if missing
        for component, info in MPOWER_COMPONENTS.items():
            col_name = f"mpower_{component.lower()}_score"
            if col_name not in self.data.columns:
                self.component_cols[component] = col_name
                # Simulate realistic component scores per-row to preserve index alignment
                simulated = pd.Series(index=self.data.index, dtype=float)
                for country in countries:
                    country_data = self.data[self.data[self.unit_col] == country].copy()

                    if len(country_data) > 0:
                        # Simulate gradual policy implementation
                        base_score = np.random.randint(0, info["max_score"] + 1)
                        country_scores = []

                        for year in years:
                            # Some countries improve over time
                            improvement_prob = 0.1  # 10% chance of improvement per year
                            if (
                                len(country_scores) > 0
                                and np.random.random() < improvement_prob
                                and country_scores[-1] < info["max_score"]
                            ):
                                base_score = min(base_score + 1, info["max_score"])

                            country_scores.append(base_score)

                        # Match to actual data using row index to preserve alignment
                        for idx, row in country_data.iterrows():
                            year_idx = years.index(row[self.time_col])
                            simulated.loc[idx] = country_scores[year_idx]

                # Add to data
                self.data[col_name] = simulated

        _logger = logging.getLogger(__name__)
        _msg = f"Created simulated MPOWER component data for: {list(self.component_cols.keys())}"
        _logger.info(_msg)

    def _create_component_treatments(self) -> None:
        """Create binary treatment indicators for each component."""
        self.component_treatment_cols = {}
        self.component_first_high_cols = {}

        for component, col_name in self.component_cols.items():
            if col_name in self.data.columns:
                threshold = MPOWER_COMPONENTS[component]["high_threshold"]

                # Binary treatment indicator
                treatment_col = f"{component.lower()}_high_binary"
                self.data[treatment_col] = (self.data[col_name] >= threshold).astype(
                    int
                )
                self.component_treatment_cols[component] = treatment_col

                # First treatment year
                first_high_col = f"first_{component.lower()}_high_year"

                # Calculate first treatment year for each unit
                first_high_data = []
                for unit in self.data[self.unit_col].unique():
                    unit_data = self.data[self.data[self.unit_col] == unit]
                    first_year = self._get_first_treatment_year(
                        unit_data, treatment_col
                    )
                    first_high_data.append(
                        {self.unit_col: unit, first_high_col: first_year}
                    )

                first_high = pd.DataFrame(first_high_data)

                self.data = self.data.merge(first_high, on=self.unit_col, how="left")
                self.component_first_high_cols[component] = first_high_col

    def _get_first_treatment_year(
        self, country_data: pd.DataFrame, treatment_col: str
    ) -> int | None:
        """Get first year of sustained high implementation for a country."""
        # Require at least 2 consecutive years of high implementation
        MIN_CONSECUTIVE_YEARS = 2

        treated_years = (
            country_data[country_data[treatment_col] == 1][self.time_col]
            .sort_values()
            .tolist()
        )

        if len(treated_years) < MIN_CONSECUTIVE_YEARS:
            return None

        # Find first sustained treatment
        for i in range(len(treated_years) - MIN_CONSECUTIVE_YEARS + 1):
            consecutive_years = treated_years[i : i + MIN_CONSECUTIVE_YEARS]
            if all(
                consecutive_years[j] == consecutive_years[0] + j
                for j in range(MIN_CONSECUTIVE_YEARS)
            ):
                return consecutive_years[0]

        return None

    def run_component_analysis(
        self,
        outcome: str,
        methods: list[str] = ["callaway_did", "synthetic_control"],
        covariates: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run component-specific causal analysis for all MPOWER components.

        Args:
            outcome: Mortality outcome variable
            methods: Causal inference methods to use
            covariates: Control variables for analysis

        Returns:
            Dict containing component-specific results
        """
        if covariates is None:
            covariates = self.control_vars

        results = {
            "outcome": outcome,
            "methods": methods,
            "components": {},
            "summary": {},
        }

        # Analyze each component separately
        for component in self.component_cols.keys():
            _logger = logging.getLogger(__name__)
            _msg = f"Analyzing component {component} ({MPOWER_COMPONENTS[component]['name']})..."
            _logger.info(_msg)

            component_results = self._analyze_single_component(
                component=component,
                outcome=outcome,
                methods=methods,
                covariates=covariates,
            )

            results["components"][component] = component_results

        # Create summary comparisons
        results["summary"] = self._create_component_summary(results["components"])

        return results

    def _analyze_single_component(
        self,
        component: str,
        outcome: str,
        methods: list[str],
        covariates: list[str],
    ) -> dict[str, Any]:
        """Analyze a single MPOWER component."""
        treatment_col = self.component_treatment_cols.get(component)
        first_high_col = self.component_first_high_cols.get(component)

        if not treatment_col or not first_high_col:
            return {"error": f"Component {component} not available"}

        component_results = {
            "component_info": MPOWER_COMPONENTS[component],
            "treatment_summary": self._get_component_treatment_summary(component),
            "methods": {},
        }

        # Run each requested method
        for method in methods:
            try:
                if method == "callaway_did":
                    method_results = self._run_component_callaway_did(
                        component=component,
                        outcome=outcome,
                        covariates=covariates,
                    )
                elif method == "synthetic_control":
                    method_results = self._run_component_synthetic_control(
                        component=component,
                        outcome=outcome,
                        predictors=covariates,
                    )
                else:
                    method_results = {"error": f"Unknown method: {method}"}

                component_results["methods"][method] = method_results

            except Exception as e:
                component_results["methods"][method] = {
                    "error": f"Method {method} failed: {str(e)}"
                }

        return component_results

    def _get_component_treatment_summary(self, component: str) -> dict[str, Any]:
        """Get treatment adoption summary for a component."""
        treatment_col = self.component_treatment_cols[component]
        first_high_col = self.component_first_high_cols[component]

        # Treatment counts
        treated_units = self.data[self.data[treatment_col] == 1][
            self.unit_col
        ].nunique()
        total_units = self.data[self.unit_col].nunique()

        # Treatment timing
        first_treatment_years = self.data[self.data[first_high_col].notna()][
            first_high_col
        ].unique()

        return {
            "treated_countries": int(treated_units),
            "total_countries": int(total_units),
            "treatment_rate": treated_units / total_units,
            "treatment_years": sorted([int(y) for y in first_treatment_years]),
            "first_treatment_year": int(min(first_treatment_years))
            if len(first_treatment_years) > 0
            else None,
            "last_treatment_year": int(max(first_treatment_years))
            if len(first_treatment_years) > 0
            else None,
        }

    def _run_component_callaway_did(
        self,
        component: str,
        outcome: str,
        covariates: list[str],
    ) -> dict[str, Any]:
        """Run Callaway & Sant'Anna DiD for a specific component."""
        first_high_col = self.component_first_high_cols[component]

        try:
            did = CallawayDiD(
                data=self.data,
                cohort_col=first_high_col,
                unit_col=self.unit_col,
                time_col=self.time_col,
            )

            did.fit(outcome=outcome, covariates=covariates)

            # Get aggregated results
            simple_att = did.aggregate("simple")
            group_att = did.aggregate("group")
            dynamic_att = did.aggregate("dynamic")

            return {
                "simple_att": {
                    "att": float(simple_att.get("att", np.nan)),
                    "std_error": float(simple_att.get("std_error", np.nan)),
                    "p_value": float(simple_att.get("p_value", np.nan)),
                },
                "group_att": group_att,
                "dynamic_att": dynamic_att,
                "method": "callaway_did",
            }

        except Exception as e:
            return {"error": f"Callaway DiD failed: {str(e)}"}

    def _run_component_synthetic_control(
        self,
        component: str,
        outcome: str,
        predictors: list[str],
    ) -> dict[str, Any]:
        """Run synthetic control for a specific component."""
        first_high_col = self.component_first_high_cols[component]

        try:
            # Get treatment info for synthetic control
            treatment_info = {}
            treated_units = self.data[self.data[first_high_col].notna()][
                [self.unit_col, first_high_col]
            ].drop_duplicates()

            for _, row in treated_units.iterrows():
                treatment_info[row[self.unit_col]] = int(row[first_high_col])

            if len(treatment_info) == 0:
                return {"error": "No treated units found for synthetic control"}

            # Limit to reasonable number for computational efficiency
            MAX_TREATED_UNITS = 10
            if len(treatment_info) > MAX_TREATED_UNITS:
                # Select representative subset
                import random

                random.seed(42)
                selected_units = random.sample(
                    list(treatment_info.keys()), MAX_TREATED_UNITS
                )
                treatment_info = {
                    unit: year
                    for unit, year in treatment_info.items()
                    if unit in selected_units
                }

            # Run synthetic control
            sc = MPOWERSyntheticControl(
                data=self.data,
                unit_col=self.unit_col,
                time_col=self.time_col,
            )

            results = sc.fit_all_units(
                treatment_info=treatment_info,
                outcome=outcome,
                predictors=predictors,
            )

            # Aggregate results
            treatment_effects = []
            match_qualities = []

            for unit_results in results.values():
                if "treatment_effect" in unit_results:
                    treatment_effects.append(unit_results["treatment_effect"])
                if "match_quality" in unit_results:
                    match_qualities.append(unit_results["match_quality"]["rmse"])

            return {
                "individual_results": results,
                "aggregated_effect": {
                    "mean_effect": float(np.mean(treatment_effects))
                    if treatment_effects
                    else np.nan,
                    "median_effect": float(np.median(treatment_effects))
                    if treatment_effects
                    else np.nan,
                    "std_effect": float(np.std(treatment_effects))
                    if treatment_effects
                    else np.nan,
                },
                "match_quality": {
                    "mean_rmse": float(np.mean(match_qualities))
                    if match_qualities
                    else np.nan,
                    "median_rmse": float(np.median(match_qualities))
                    if match_qualities
                    else np.nan,
                },
                "n_fitted_units": len(results),
                "method": "synthetic_control",
            }

        except Exception as e:
            return {"error": f"Synthetic control failed: {str(e)}"}

    def _create_component_summary(
        self, component_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Create summary comparison across components."""
        summary = {
            "effect_comparison": {},
            "treatment_coverage": {},
            "statistical_significance": {},
            "policy_rankings": {},
        }

        # Extract effects from each component
        for component, results in component_results.items():
            if "error" in results:
                continue

            # Get treatment coverage
            if "treatment_summary" in results:
                ts = results["treatment_summary"]
                summary["treatment_coverage"][component] = {
                    "countries": ts["treated_countries"],
                    "rate": ts["treatment_rate"],
                }

            # Get effect estimates from each method
            for method, method_results in results.get("methods", {}).items():
                if "error" in method_results:
                    continue

                if method not in summary["effect_comparison"]:
                    summary["effect_comparison"][method] = {}
                    summary["statistical_significance"][method] = {}

                if method == "callaway_did":
                    simple_att = method_results.get("simple_att", {})
                    effect = simple_att.get("att", np.nan)
                    p_value = simple_att.get("p_value", np.nan)

                elif method == "synthetic_control":
                    agg_effect = method_results.get("aggregated_effect", {})
                    effect = agg_effect.get("mean_effect", np.nan)
                    p_value = (
                        np.nan
                    )  # Synthetic control doesn't provide p-values by default

                summary["effect_comparison"][method][component] = effect
                summary["statistical_significance"][method][component] = p_value

        # Create policy rankings based on effect sizes
        for method in summary["effect_comparison"]:
            effects = summary["effect_comparison"][method]
            valid_effects = {
                comp: eff for comp, eff in effects.items() if not np.isnan(eff)
            }

            if valid_effects:
                # Rank by absolute effect size (larger reductions = better)
                ranked = sorted(
                    valid_effects.items(), key=lambda x: abs(x[1]), reverse=True
                )
                summary["policy_rankings"][method] = [
                    {
                        "component": comp,
                        "effect": eff,
                        "rank": i + 1,
                        "component_name": MPOWER_COMPONENTS[comp]["name"],
                    }
                    for i, (comp, eff) in enumerate(ranked)
                ]

        return summary

    def create_mechanism_visualization(
        self,
        results: dict[str, Any],
        save_path: str | None = None,
    ) -> None:
        """Create comprehensive visualization of mechanism analysis results."""
        if not PLOTTING_AVAILABLE:
            _logger = logging.getLogger(__name__)
            _msg = "Plotting not available. Skipping visualization."
            _logger.info(_msg)
            return

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            f"MPOWER Mechanism Analysis: {results['outcome']}",
            fontsize=16,
            fontweight="bold",
        )

        # 1. Treatment coverage by component
        self._plot_treatment_coverage(results, axes[0, 0])

        # 2. Effect size comparison
        self._plot_effect_comparison(results, axes[0, 1])

        # 3. Statistical significance
        self._plot_significance(results, axes[1, 0])

        # 4. Policy ranking
        self._plot_policy_ranking(results, axes[1, 1])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            _logger = logging.getLogger(__name__)
            _msg = f"Mechanism analysis visualization saved to: {save_path}"
            _logger.info(_msg)
        else:
            plt.show()

    def _plot_treatment_coverage(self, results: dict[str, Any], ax) -> None:
        """Plot treatment coverage by component."""
        summary = results.get("summary", {})
        coverage = summary.get("treatment_coverage", {})

        if not coverage:
            ax.text(
                0.5,
                0.5,
                "No coverage data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Treatment Coverage by Component")
            return

        components = list(coverage.keys())
        countries = [coverage[comp]["countries"] for comp in components]
        rates = [coverage[comp]["rate"] * 100 for comp in components]

        bars = ax.bar(components, countries, alpha=0.7, color="skyblue")
        ax.set_ylabel("Number of Countries")
        ax.set_title("Treatment Coverage by Component")

        # Add percentage labels
        for i, (bar, rate) in enumerate(zip(bars, rates, strict=False)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.5,
                f"{rate:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    def _plot_effect_comparison(self, results: dict[str, Any], ax) -> None:
        """Plot effect size comparison across components."""
        summary = results.get("summary", {})
        effects = summary.get("effect_comparison", {})

        if not effects:
            ax.text(
                0.5,
                0.5,
                "No effect data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Effect Size Comparison")
            return

        # Use first available method
        method = list(effects.keys())[0]
        method_effects = effects[method]

        components = list(method_effects.keys())
        effect_sizes = [method_effects[comp] for comp in components]

        # Color bars by effect direction
        colors = [
            "red" if x < 0 else "green" if x > 0 else "gray" for x in effect_sizes
        ]

        bars = ax.bar(components, effect_sizes, color=colors, alpha=0.7)
        ax.set_ylabel("Effect Size (Mortality Rate Change)")
        ax.set_title(f"Effect Size Comparison ({method.replace('_', ' ').title()})")
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)

        # Add value labels
        for bar, value in zip(bars, effect_sizes, strict=False):
            if not np.isnan(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    value + (0.1 if value >= 0 else -0.1),
                    f"{value:.2f}",
                    ha="center",
                    va="bottom" if value >= 0 else "top",
                    fontsize=9,
                )

    def _plot_significance(self, results: dict[str, Any], ax) -> None:
        """Plot statistical significance by component."""
        summary = results.get("summary", {})
        significance = summary.get("statistical_significance", {})

        if not significance:
            ax.text(
                0.5,
                0.5,
                "No significance data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Statistical Significance")
            return

        # Use first available method
        method = list(significance.keys())[0]
        method_pvalues = significance[method]

        components = list(method_pvalues.keys())
        p_values = [method_pvalues[comp] for comp in components]

        # Color by significance level
        colors = [
            "darkgreen"
            if p < 0.01
            else "green"
            if p < 0.05
            else "orange"
            if p < 0.1
            else "red"
            for p in p_values
            if not np.isnan(p)
        ]

        valid_components = [
            comp
            for comp, p in zip(components, p_values, strict=False)
            if not np.isnan(p)
        ]
        valid_pvalues = [p for p in p_values if not np.isnan(p)]

        if valid_components:
            _ = ax.bar(valid_components, valid_pvalues, color=colors, alpha=0.7)
            ax.set_ylabel("P-value")
            ax.set_title("Statistical Significance (Lower = More Significant)")
            ax.axhline(
                y=SIGNIFICANCE_LEVEL,
                color="red",
                linestyle="--",
                alpha=0.5,
                label="α = 0.05",
            )
            ax.legend()
        else:
            ax.text(
                0.5,
                0.5,
                "No valid p-values available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    def _plot_policy_ranking(self, results: dict[str, Any], ax) -> None:
        """Plot policy ranking by effectiveness."""
        summary = results.get("summary", {})
        rankings = summary.get("policy_rankings", {})

        if not rankings:
            ax.text(
                0.5,
                0.5,
                "No ranking data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Policy Ranking by Effectiveness")
            return

        # Use first available method
        method = list(rankings.keys())[0]
        method_rankings = rankings[method]

        components = [item["component"] for item in method_rankings]
        effects = [abs(item["effect"]) for item in method_rankings]
        names = [item["component_name"] for item in method_rankings]

        _ = ax.barh(components, effects, alpha=0.7, color="steelblue")
        ax.set_xlabel("Effect Magnitude (Absolute)")
        ax.set_title("Policy Ranking by Effectiveness")

        # Add component names as labels
        for i, (comp, name) in enumerate(zip(components, names, strict=False)):
            ax.text(0.01 * max(effects), i, f"{comp}: {name}", va="center", fontsize=9)

    def export_mechanism_results(
        self,
        results: dict[str, Any],
        output_dir: str = "results/mechanism_analysis",
    ) -> None:
        """Export mechanism analysis results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export comprehensive results as JSON
        import json

        # Convert numpy types for JSON serialization
        json_results = self._convert_for_json(results)

        with open(output_path / "mechanism_analysis_results.json", "w") as f:
            json.dump(json_results, f, indent=2)

        # Export summary tables
        self._export_summary_tables(results, output_path)

        # Export visualizations
        if PLOTTING_AVAILABLE:
            self.create_mechanism_visualization(
                results, save_path=output_path / "mechanism_analysis_plots.png"
            )
        _logger = logging.getLogger(__name__)
        _msg = f"Mechanism analysis results exported to: {output_path}"
        _logger.info(_msg)

    def _convert_for_json(self, obj: Any) -> Any:
        """Convert numpy types to JSON-serializable types."""
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj) if not np.isnan(obj) else None
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def _export_summary_tables(
        self, results: dict[str, Any], output_path: Path
    ) -> None:
        """Export summary tables as CSV files."""
        summary = results.get("summary", {})

        # Component comparison table
        comparison_data = []
        for component in results.get("components", {}):
            row = {"Component": component}

            # Add treatment info
            ts = results["components"][component].get("treatment_summary", {})
            row["Treated_Countries"] = ts.get("treated_countries", 0)
            row["Treatment_Rate"] = ts.get("treatment_rate", 0)

            # Add effects from each method
            for method in ["callaway_did", "synthetic_control"]:
                method_results = (
                    results["components"][component].get("methods", {}).get(method, {})
                )

                if method == "callaway_did":
                    simple_att = method_results.get("simple_att", {})
                    row[f"{method}_effect"] = simple_att.get("att", np.nan)
                    row[f"{method}_pvalue"] = simple_att.get("p_value", np.nan)
                elif method == "synthetic_control":
                    agg_effect = method_results.get("aggregated_effect", {})
                    row[f"{method}_effect"] = agg_effect.get("mean_effect", np.nan)

            comparison_data.append(row)

        if comparison_data:
            pd.DataFrame(comparison_data).to_csv(
                output_path / "component_comparison.csv", index=False
            )

        # Policy rankings
        rankings = summary.get("policy_rankings", {})
        for method, ranking_list in rankings.items():
            if ranking_list:
                ranking_df = pd.DataFrame(ranking_list)
                ranking_df.to_csv(
                    output_path / f"policy_ranking_{method}.csv", index=False
                )
