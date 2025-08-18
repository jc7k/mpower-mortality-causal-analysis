"""Policy Interaction Analysis for MPOWER Component Synergies.

This module analyzes synergistic effects between MPOWER policy components
to identify super-additive policy combinations.
"""

import warnings

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from scipy import stats

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    warnings.warn(
        "Plotting packages not available. Visualization disabled.",
        stacklevel=2,
    )

# Define significance level constant
SIGNIFICANCE_LEVEL = 0.05

# MPOWER component definitions
MPOWER_COMPONENTS = ["M", "P", "O", "W", "E", "R"]


class PolicyInteractionAnalysis:
    """Analyzes synergies between MPOWER policy components.

    This class implements methods to detect super-additive effects when
    multiple MPOWER components are implemented together.

    Parameters:
        data (pd.DataFrame): Panel data with MPOWER components and outcomes
        unit_col (str): Column name for unit identifier (e.g., 'country')
        time_col (str): Column name for time identifier (e.g., 'year')
        bootstrap_samples (int): Number of bootstrap samples for CI estimation
    """

    def __init__(
        self,
        data: pd.DataFrame,
        unit_col: str = "country",
        time_col: str = "year",
        bootstrap_samples: int = 1000,
    ) -> None:
        self.data = data.copy()
        self.unit_col = unit_col
        self.time_col = time_col
        self.bootstrap_samples = bootstrap_samples
        self.interaction_results = {}

    def _create_interaction_terms(self, policies: list[str]) -> pd.DataFrame:
        """Create interaction terms for given policies.

        Args:
            policies: List of MPOWER component names to interact

        Returns:
            DataFrame with original data plus interaction terms
        """
        data_with_interactions = self.data.copy()

        # Create pairwise interactions
        for policy1, policy2 in combinations(policies, 2):
            col1 = f"mpower_{policy1.lower()}_score"
            col2 = f"mpower_{policy2.lower()}_score"

            if (
                col1 in data_with_interactions.columns
                and col2 in data_with_interactions.columns
            ):
                interaction_name = f"{policy1}_{policy2}_interaction"
                data_with_interactions[interaction_name] = (
                    data_with_interactions[col1] * data_with_interactions[col2]
                )

        # Create three-way interactions for important combinations
        if len(policies) >= 3:
            for policy1, policy2, policy3 in combinations(policies, 3):
                col1 = f"mpower_{policy1.lower()}_score"
                col2 = f"mpower_{policy2.lower()}_score"
                col3 = f"mpower_{policy3.lower()}_score"

                if all(
                    col in data_with_interactions.columns for col in [col1, col2, col3]
                ):
                    interaction_name = f"{policy1}_{policy2}_{policy3}_interaction"
                    data_with_interactions[interaction_name] = (
                        data_with_interactions[col1]
                        * data_with_interactions[col2]
                        * data_with_interactions[col3]
                    )

        return data_with_interactions

    def estimate_interactions(
        self,
        outcome: str,
        policies: list[str] | None = None,
        covariates: list[str] | None = None,
    ) -> dict[str, Any]:
        """Estimate pairwise and higher-order interactions.

        Uses factorial design analysis to estimate interaction effects
        with bootstrap confidence intervals.

        Args:
            outcome: Name of outcome variable
            policies: List of MPOWER components to analyze (default: all)
            covariates: List of control variables

        Returns:
            Dictionary with interaction estimates and confidence intervals
        """
        if policies is None:
            policies = MPOWER_COMPONENTS

        if covariates is None:
            covariates = ["gdp_per_capita_log", "urban_population_pct"]

        # Create interaction terms
        data_with_interactions = self._create_interaction_terms(policies)

        # Prepare regression specification
        main_effects = [f"mpower_{p.lower()}_score" for p in policies]
        interaction_terms = [
            col for col in data_with_interactions.columns if "_interaction" in col
        ]

        # Filter to existing columns
        main_effects = [
            col for col in main_effects if col in data_with_interactions.columns
        ]
        available_covariates = [
            col for col in covariates if col in data_with_interactions.columns
        ]

        all_regressors = main_effects + interaction_terms + available_covariates

        # Remove rows with missing values
        analysis_data = data_with_interactions[
            [outcome] + all_regressors + [self.unit_col, self.time_col]
        ].dropna()

        if len(analysis_data) == 0:
            return {"error": "No valid observations after removing missing values"}

        # Estimate main model
        try:
            from statsmodels.formula.api import ols

            # Create formula
            formula = f"{outcome} ~ " + " + ".join(all_regressors)

            # Add fixed effects if sufficient variation
            unique_units = analysis_data[self.unit_col].nunique()
            unique_times = analysis_data[self.time_col].nunique()

            if unique_units > 10:
                formula += f" + C({self.unit_col})"
            if unique_times > 3:
                formula += f" + C({self.time_col})"

            model = ols(formula, data=analysis_data).fit(
                cov_type="cluster", cov_kwds={"groups": analysis_data[self.unit_col]}
            )

            # Extract interaction effects
            interaction_effects = {}
            for term in interaction_terms:
                if term in model.params.index:
                    interaction_effects[term] = {
                        "coefficient": model.params[term],
                        "std_error": model.bse[term],
                        "p_value": model.pvalues[term],
                        "conf_int": model.conf_int().loc[term].tolist(),
                    }

            # Bootstrap confidence intervals for robustness
            bootstrap_results = self._bootstrap_interactions(
                analysis_data, outcome, all_regressors, interaction_terms
            )

            return {
                "main_model": {
                    "rsquared": model.rsquared,
                    "nobs": int(model.nobs),
                    "interaction_effects": interaction_effects,
                },
                "bootstrap_results": bootstrap_results,
                "data_summary": {
                    "n_observations": len(analysis_data),
                    "n_units": analysis_data[self.unit_col].nunique(),
                    "n_periods": analysis_data[self.time_col].nunique(),
                },
            }

        except Exception as e:
            return {"error": f"Model estimation failed: {str(e)}"}

    def _bootstrap_interactions(
        self,
        data: pd.DataFrame,
        outcome: str,
        regressors: list[str],
        interaction_terms: list[str],
    ) -> dict[str, Any]:
        """Bootstrap confidence intervals for interaction terms.

        Args:
            data: Analysis dataset
            outcome: Outcome variable name
            regressors: All regressor names
            interaction_terms: Interaction term names

        Returns:
            Bootstrap results with percentile confidence intervals
        """

        def estimate_model(sample_data: pd.DataFrame) -> dict[str, float]:
            try:
                from statsmodels.formula.api import ols

                formula = f"{outcome} ~ " + " + ".join(regressors)
                model = ols(formula, data=sample_data).fit()

                results = {}
                for term in interaction_terms:
                    if term in model.params.index:
                        results[term] = model.params[term]
                return results
            except Exception:
                return dict.fromkeys(interaction_terms, np.nan)

        # Perform bootstrap
        bootstrap_estimates = []
        n_samples = len(data)

        for _ in range(self.bootstrap_samples):
            # Sample with replacement
            sample_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            sample_data = data.iloc[sample_indices].copy()

            estimates = estimate_model(sample_data)
            bootstrap_estimates.append(estimates)

        # Calculate percentile confidence intervals
        bootstrap_results = {}
        for term in interaction_terms:
            estimates = [est.get(term, np.nan) for est in bootstrap_estimates]
            valid_estimates = [est for est in estimates if not np.isnan(est)]

            if len(valid_estimates) > 10:  # Minimum valid estimates
                bootstrap_results[term] = {
                    "mean": np.mean(valid_estimates),
                    "std": np.std(valid_estimates),
                    "ci_lower": np.percentile(valid_estimates, 2.5),
                    "ci_upper": np.percentile(valid_estimates, 97.5),
                    "n_valid": len(valid_estimates),
                }

        return bootstrap_results

    def identify_synergies(
        self,
        interaction_results: dict[str, Any] | None = None,
        significance_threshold: float = SIGNIFICANCE_LEVEL,
    ) -> list[tuple[str, float, bool]]:
        """Identify super-additive policy combinations.

        Args:
            interaction_results: Results from estimate_interactions (use last if None)
            significance_threshold: P-value threshold for significance

        Returns:
            List of tuples (interaction_name, effect_size, is_significant)
        """
        if interaction_results is None:
            if not self.interaction_results:
                return []
            interaction_results = self.interaction_results

        synergies = []

        if (
            "main_model" in interaction_results
            and "interaction_effects" in interaction_results["main_model"]
        ):
            effects = interaction_results["main_model"]["interaction_effects"]

            for interaction_name, effect_data in effects.items():
                coefficient = effect_data["coefficient"]
                p_value = effect_data["p_value"]

                # Super-additive if positive and significant
                is_synergistic = coefficient > 0 and p_value < significance_threshold

                synergies.append((interaction_name, coefficient, is_synergistic))

        # Sort by effect size (descending)
        synergies.sort(key=lambda x: abs(x[1]), reverse=True)

        return synergies

    def plot_interaction_network(
        self,
        synergies: list[tuple[str, float, bool]] | None = None,
        save_path: str | None = None,
    ) -> None:
        """Visualize interaction network of policy synergies.

        Args:
            synergies: Results from identify_synergies (calculate if None)
            save_path: Path to save plot (optional)
        """
        if not PLOTTING_AVAILABLE:
            warnings.warn("Plotting not available. Install matplotlib and seaborn.")
            return

        if synergies is None:
            synergies = self.identify_synergies()

        if not synergies:
            warnings.warn("No synergies to plot.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Effect sizes with significance
        interaction_names = [s[0] for s in synergies]
        effect_sizes = [s[1] for s in synergies]
        significances = [s[2] for s in synergies]

        colors = ["red" if sig else "gray" for sig in significances]

        ax1.barh(range(len(interaction_names)), effect_sizes, color=colors, alpha=0.7)
        ax1.set_yticks(range(len(interaction_names)))
        ax1.set_yticklabels(
            [name.replace("_interaction", "") for name in interaction_names]
        )
        ax1.set_xlabel("Interaction Effect Size")
        ax1.set_title("Policy Interaction Effects")
        ax1.axvline(x=0, color="black", linestyle="--", alpha=0.5)

        # Add significance legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="red", alpha=0.7, label="Significant"),
            Patch(facecolor="gray", alpha=0.7, label="Not Significant"),
        ]
        ax1.legend(handles=legend_elements)

        # Plot 2: Synergy strength distribution
        significant_effects = [s[1] for s in synergies if s[2]]
        if significant_effects:
            ax2.hist(
                significant_effects,
                bins=min(10, len(significant_effects)),
                alpha=0.7,
                color="red",
            )
            ax2.set_xlabel("Effect Size")
            ax2.set_ylabel("Frequency")
            ax2.set_title("Distribution of Significant Synergies")
            ax2.axvline(
                x=np.mean(significant_effects),
                color="black",
                linestyle="--",
                label="Mean",
            )
            ax2.legend()
        else:
            ax2.text(
                0.5,
                0.5,
                "No significant synergies found",
                ha="center",
                va="center",
                transform=ax2.transAxes,
            )
            ax2.set_title("Distribution of Significant Synergies")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def analyze_dose_response(
        self,
        outcome: str,
        component: str,
        levels: list[int] | None = None,
    ) -> dict[str, Any]:
        """Analyze dose-response relationship for a component.

        Args:
            outcome: Outcome variable name
            component: MPOWER component to analyze
            levels: Score levels to analyze (default: 0-5 or 0-4 for W)

        Returns:
            Dose-response analysis results
        """
        component_col = f"mpower_{component.lower()}_score"

        if component_col not in self.data.columns:
            return {"error": f"Component column {component_col} not found"}

        if levels is None:
            max_score = 4 if component == "W" else 5
            levels = list(range(max_score + 1))

        # Calculate mean outcomes by component level
        dose_response = []

        for level in levels:
            level_data = self.data[self.data[component_col] == level]
            if len(level_data) > 0 and outcome in level_data.columns:
                mean_outcome = level_data[outcome].mean()
                std_outcome = level_data[outcome].std()
                n_obs = len(level_data)

                # Calculate confidence interval
                if n_obs > 1:
                    se = std_outcome / np.sqrt(n_obs)
                    ci_lower = mean_outcome - 1.96 * se
                    ci_upper = mean_outcome + 1.96 * se
                else:
                    ci_lower = ci_upper = mean_outcome

                dose_response.append(
                    {
                        "level": level,
                        "mean_outcome": mean_outcome,
                        "std_outcome": std_outcome,
                        "n_observations": n_obs,
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                    }
                )

        # Test for linear trend
        if len(dose_response) > 2:
            levels_list = [dr["level"] for dr in dose_response]
            means_list = [dr["mean_outcome"] for dr in dose_response]

            # Weighted by sample size
            weights = [dr["n_observations"] for dr in dose_response]

            correlation, p_value = stats.pearsonr(levels_list, means_list)

            # Linear regression for trend
            try:
                slope, intercept, r_value, p_trend, std_err = stats.linregress(
                    levels_list, means_list
                )
                trend_test = {
                    "correlation": correlation,
                    "p_value_correlation": p_value,
                    "slope": slope,
                    "intercept": intercept,
                    "r_squared": r_value**2,
                    "p_value_trend": p_trend,
                    "std_error": std_err,
                }
            except Exception:
                trend_test = {"error": "Could not compute trend test"}
        else:
            trend_test = {"error": "Insufficient data points for trend analysis"}

        return {
            "component": component,
            "outcome": outcome,
            "dose_response": dose_response,
            "trend_analysis": trend_test,
        }
