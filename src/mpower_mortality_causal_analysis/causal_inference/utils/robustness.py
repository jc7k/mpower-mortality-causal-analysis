"""Robustness and Sensitivity Testing Utilities.

This module provides utilities for conducting robustness checks and
sensitivity analysis for causal inference results.
"""

import warnings

from collections.abc import Callable
from typing import Any

import numpy as np

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


class RobustnessTests:
    """Robustness and Sensitivity Testing for Causal Inference.

    Provides methods for:
    - Placebo tests using unaffected outcomes
    - Sample sensitivity analysis
    - Alternative specification testing
    - Randomization inference
    - Leave-one-out analysis

    Parameters:
        data (DataFrame): Panel data
        unit_col (str): Column name for unit identifier
        time_col (str): Column name for time identifier

    Example:
        >>> robustness = RobustnessTests(data=panel_data, unit_col='country', time_col='year')
        >>> placebo_results = robustness.placebo_test(estimator=did_model,
        ...                                          placebo_outcomes=['unrelated_outcome'])
        >>> sensitivity_results = robustness.sample_sensitivity(estimator_func=fit_model,
        ...                                                    exclude_units=['outlier_country'])
    """

    def __init__(self, data: DataFrame, unit_col: str, time_col: str):
        """Initialize Robustness Tests."""
        self.data = data.copy()
        self.unit_col = unit_col
        self.time_col = time_col

        # Validate required columns
        required_cols = [unit_col, time_col]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def placebo_test(
        self,
        estimator: Any,
        placebo_outcomes: list[str],
        main_outcome: str,
        significance_threshold: float = 0.05,
    ) -> dict[str, Any]:
        """Conduct placebo tests using outcomes that should not be affected by treatment.

        Args:
            estimator: Fitted causal inference estimator with a refit method
            placebo_outcomes (List[str]): Outcomes that should not be affected by treatment
            main_outcome (str): Main outcome for comparison
            significance_threshold (float): Threshold for significance

        Returns:
            Dict with placebo test results
        """
        missing_outcomes = [
            outcome for outcome in placebo_outcomes if outcome not in self.data.columns
        ]
        if missing_outcomes:
            raise ValueError(f"Placebo outcomes not found in data: {missing_outcomes}")

        if main_outcome not in self.data.columns:
            raise ValueError(f"Main outcome '{main_outcome}' not found in data")

        placebo_results = {}

        # Test each placebo outcome
        for outcome in placebo_outcomes:
            try:
                # Re-fit the estimator with the placebo outcome
                # This assumes the estimator has a method to refit with different outcomes
                if hasattr(estimator, "fit"):
                    # Create a copy of the estimator and refit
                    placebo_estimator = type(estimator)(
                        data=self.data,
                        **{
                            k: v
                            for k, v in estimator.__dict__.items()
                            if not k.startswith("_") and k != "data"
                        },
                    )
                    placebo_estimator.fit(outcome=outcome)

                    # Extract treatment effect estimate
                    if hasattr(placebo_estimator, "aggregate"):
                        placebo_effect = placebo_estimator.aggregate("simple")
                    elif hasattr(placebo_estimator, "get_coefficients"):
                        coeffs = placebo_estimator.get_coefficients()
                        # Look for treatment-related coefficients
                        treatment_coeffs = coeffs[
                            coeffs.index.str.contains("treat|post|mpower", case=False)
                        ]
                        if not treatment_coeffs.empty:
                            placebo_effect = {
                                "att": treatment_coeffs["coefficient"].iloc[0],
                                "se": treatment_coeffs["std_error"].iloc[0],
                                "pvalue": treatment_coeffs["p_value"].iloc[0],
                            }
                        else:
                            placebo_effect = {"error": "No treatment coefficient found"}
                    else:
                        placebo_effect = {"error": "Cannot extract treatment effect"}

                    placebo_results[outcome] = {
                        "effect": placebo_effect,
                        "significant": placebo_effect.get("pvalue", 1)
                        < significance_threshold,
                        "estimator_type": type(estimator).__name__,
                    }
                else:
                    # Estimator doesn't have fit method
                    placebo_results[outcome] = {
                        "error": "Estimator does not support refitting",
                        "significant": None,
                    }

            except Exception as e:
                placebo_results[outcome] = {"error": str(e), "significant": None}

        # Summary statistics
        successful_tests = [k for k, v in placebo_results.items() if "error" not in v]
        significant_placebo_tests = sum(
            1 for k in successful_tests if placebo_results[k]["significant"]
        )

        return {
            "placebo_results": placebo_results,
            "main_outcome": main_outcome,
            "n_placebo_tests": len(placebo_outcomes),
            "n_successful_tests": len(successful_tests),
            "n_significant_placebo": significant_placebo_tests,
            "false_positive_rate": significant_placebo_tests / len(successful_tests)
            if successful_tests
            else 0,
            "conclusion": self._interpret_placebo_results(
                significant_placebo_tests, len(successful_tests)
            ),
        }

    def _interpret_placebo_results(self, n_significant: int, n_total: int) -> str:
        """Interpret placebo test results."""
        if n_total == 0:
            return "No successful placebo tests conducted"

        false_positive_rate = n_significant / n_total

        if false_positive_rate == 0:
            return "No significant placebo effects - supports main results"
        if false_positive_rate <= 0.1:
            return "Low false positive rate - main results likely robust"
        if false_positive_rate <= 0.25:
            return "Moderate false positive rate - some concern about main results"
        return "High false positive rate - serious concern about main results"

    def sample_sensitivity(
        self,
        estimator_func: Callable,
        exclude_units: list[str] | None = None,
        exclude_periods: list[int | str] | None = None,
        bootstrap_samples: int = 100,
        random_exclusions: bool = True,
        exclusion_rate: float = 0.1,
    ) -> dict[str, Any]:
        """Test sensitivity to sample composition by excluding units or time periods.

        Args:
            estimator_func (Callable): Function that fits the estimator and returns results
            exclude_units (List[str], optional): Specific units to exclude
            exclude_periods (List, optional): Specific time periods to exclude
            bootstrap_samples (int): Number of bootstrap samples for random exclusions
            random_exclusions (bool): Whether to test random exclusions
            exclusion_rate (float): Proportion of units/periods to exclude in random tests

        Returns:
            Dict with sensitivity analysis results
        """
        sensitivity_results = {
            "baseline": None,
            "exclude_units": {},
            "exclude_periods": {},
            "random_exclusions": [],
        }

        # Baseline results
        try:
            baseline_results = estimator_func(self.data)
            sensitivity_results["baseline"] = baseline_results
        except Exception as e:
            raise RuntimeError(f"Failed to fit baseline model: {e}")

        # Test excluding specific units
        if exclude_units:
            for unit in exclude_units:
                if unit in self.data[self.unit_col].values:
                    try:
                        filtered_data = self.data[self.data[self.unit_col] != unit]
                        results = estimator_func(filtered_data)
                        sensitivity_results["exclude_units"][unit] = results
                    except Exception as e:
                        sensitivity_results["exclude_units"][unit] = {"error": str(e)}

        # Test excluding specific time periods
        if exclude_periods:
            for period in exclude_periods:
                if period in self.data[self.time_col].values:
                    try:
                        filtered_data = self.data[self.data[self.time_col] != period]
                        results = estimator_func(filtered_data)
                        sensitivity_results["exclude_periods"][period] = results
                    except Exception as e:
                        sensitivity_results["exclude_periods"][period] = {
                            "error": str(e)
                        }

        # Random exclusions
        if random_exclusions:
            np.random.seed(42)  # For reproducibility

            for i in range(bootstrap_samples):
                try:
                    # Randomly exclude units
                    units = self.data[self.unit_col].unique()
                    n_exclude = max(1, int(len(units) * exclusion_rate))
                    excluded_units = np.random.choice(units, n_exclude, replace=False)

                    filtered_data = self.data[
                        ~self.data[self.unit_col].isin(excluded_units)
                    ]
                    results = estimator_func(filtered_data)

                    sensitivity_results["random_exclusions"].append(
                        {
                            "sample": i,
                            "excluded_units": excluded_units.tolist(),
                            "results": results,
                        }
                    )

                except Exception as e:
                    sensitivity_results["random_exclusions"].append(
                        {"sample": i, "error": str(e)}
                    )

        # Calculate summary statistics
        sensitivity_results["summary"] = self._summarize_sensitivity_results(
            sensitivity_results
        )

        return sensitivity_results

    def _summarize_sensitivity_results(self, results: dict[str, Any]) -> dict[str, Any]:
        """Summarize sensitivity analysis results."""
        baseline = results["baseline"]
        if not baseline or "error" in baseline:
            return {"error": "No valid baseline results"}

        # Extract baseline effect size (this depends on the estimator structure)
        baseline_effect = self._extract_effect_size(baseline)

        # Analyze random exclusions
        random_effects = []
        for exclusion in results["random_exclusions"]:
            if "error" not in exclusion:
                effect = self._extract_effect_size(exclusion["results"])
                if effect is not None:
                    random_effects.append(effect)

        if random_effects:
            effect_std = np.std(random_effects)
            effect_range = [np.min(random_effects), np.max(random_effects)]
            effect_percentiles = np.percentile(random_effects, [5, 25, 75, 95])
        else:
            effect_std = None
            effect_range = None
            effect_percentiles = None

        return {
            "baseline_effect": baseline_effect,
            "n_random_samples": len(random_effects),
            "effect_std": effect_std,
            "effect_range": effect_range,
            "effect_percentiles": effect_percentiles,
            "robust_to_exclusions": effect_std < 0.1 * abs(baseline_effect)
            if effect_std and baseline_effect
            else None,
        }

    def _extract_effect_size(self, results: Any) -> float | None:
        """Extract effect size from results (implementation depends on estimator)."""
        try:
            if isinstance(results, dict):
                if "att" in results:
                    return results["att"]
                if "treatment_effect" in results:
                    return results["treatment_effect"]
                if "coefficients" in results:
                    # Look for treatment-related coefficient
                    coeffs = results["coefficients"]
                    if isinstance(coeffs, dict):
                        for key, value in coeffs.items():
                            if "treat" in key.lower() or "post" in key.lower():
                                return value
            return None
        except:
            return None

    def specification_sensitivity(
        self, estimator_func: Callable, specification_variants: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Test sensitivity to different model specifications.

        Args:
            estimator_func (Callable): Function that fits the estimator
            specification_variants (List[Dict]): List of specification parameters to test

        Returns:
            Dict with results from different specifications
        """
        spec_results = {}

        for i, spec in enumerate(specification_variants):
            try:
                results = estimator_func(self.data, **spec)
                spec_results[f"specification_{i}"] = {
                    "parameters": spec,
                    "results": results,
                }
            except Exception as e:
                spec_results[f"specification_{i}"] = {
                    "parameters": spec,
                    "error": str(e),
                }

        return spec_results

    def randomization_inference(
        self,
        estimator_func: Callable,
        treatment_col: str,
        n_permutations: int = 1000,
        random_seed: int = 42,
    ) -> dict[str, Any]:
        """Conduct randomization inference by permuting treatment assignment.

        Args:
            estimator_func (Callable): Function that fits the estimator
            treatment_col (str): Treatment variable column name
            n_permutations (int): Number of random permutations
            random_seed (int): Random seed for reproducibility

        Returns:
            Dict with randomization inference results
        """
        if treatment_col not in self.data.columns:
            raise ValueError(f"Treatment column '{treatment_col}' not found in data")

        np.random.seed(random_seed)

        # Original treatment effect
        try:
            original_results = estimator_func(self.data)
            original_effect = self._extract_effect_size(original_results)
        except Exception as e:
            raise RuntimeError(f"Failed to estimate original treatment effect: {e}")

        if original_effect is None:
            raise ValueError("Could not extract treatment effect from original results")

        # Permutation tests
        permutation_effects = []

        for _i in range(n_permutations):
            try:
                # Create permuted data
                permuted_data = self.data.copy()

                # Randomly shuffle treatment within units (maintaining time structure)
                for unit in permuted_data[self.unit_col].unique():
                    unit_mask = permuted_data[self.unit_col] == unit
                    unit_treatment = permuted_data.loc[unit_mask, treatment_col].values
                    np.random.shuffle(unit_treatment)
                    permuted_data.loc[unit_mask, treatment_col] = unit_treatment

                # Estimate effect with permuted treatment
                permuted_results = estimator_func(permuted_data)
                permuted_effect = self._extract_effect_size(permuted_results)

                if permuted_effect is not None:
                    permutation_effects.append(permuted_effect)

            except Exception:
                continue  # Skip failed permutations

        if not permutation_effects:
            return {"error": "No successful permutations"}

        # Calculate p-value
        permutation_effects = np.array(permutation_effects)

        if original_effect >= 0:
            p_value = np.mean(permutation_effects >= original_effect)
        else:
            p_value = np.mean(permutation_effects <= original_effect)

        # Two-tailed p-value
        p_value_two_tailed = 2 * min(p_value, 1 - p_value)

        return {
            "original_effect": original_effect,
            "permutation_effects": permutation_effects.tolist(),
            "n_permutations": len(permutation_effects),
            "p_value_one_tailed": p_value,
            "p_value_two_tailed": p_value_two_tailed,
            "permutation_mean": np.mean(permutation_effects),
            "permutation_std": np.std(permutation_effects),
            "significant_at_05": p_value_two_tailed < 0.05,
        }

    def leave_one_out_analysis(
        self, estimator_func: Callable, by_unit: bool = True, by_period: bool = False
    ) -> dict[str, Any]:
        """Conduct leave-one-out analysis by systematically excluding each unit or period.

        Args:
            estimator_func (Callable): Function that fits the estimator
            by_unit (bool): Whether to do leave-one-unit-out analysis
            by_period (bool): Whether to do leave-one-period-out analysis

        Returns:
            Dict with leave-one-out results
        """
        loo_results = {"baseline": None, "by_unit": {}, "by_period": {}}

        # Baseline results
        try:
            baseline_results = estimator_func(self.data)
            loo_results["baseline"] = baseline_results
            baseline_effect = self._extract_effect_size(baseline_results)
        except Exception as e:
            raise RuntimeError(f"Failed to fit baseline model: {e}")

        # Leave-one-unit-out
        if by_unit:
            units = self.data[self.unit_col].unique()

            for unit in units:
                try:
                    filtered_data = self.data[self.data[self.unit_col] != unit]
                    results = estimator_func(filtered_data)
                    effect = self._extract_effect_size(results)

                    loo_results["by_unit"][unit] = {
                        "results": results,
                        "effect": effect,
                        "effect_change": effect - baseline_effect
                        if effect and baseline_effect
                        else None,
                    }

                except Exception as e:
                    loo_results["by_unit"][unit] = {"error": str(e)}

        # Leave-one-period-out
        if by_period:
            periods = self.data[self.time_col].unique()

            for period in periods:
                try:
                    filtered_data = self.data[self.data[self.time_col] != period]
                    results = estimator_func(filtered_data)
                    effect = self._extract_effect_size(results)

                    loo_results["by_period"][period] = {
                        "results": results,
                        "effect": effect,
                        "effect_change": effect - baseline_effect
                        if effect and baseline_effect
                        else None,
                    }

                except Exception as e:
                    loo_results["by_period"][period] = {"error": str(e)}

        # Summary statistics
        loo_results["summary"] = self._summarize_loo_results(
            loo_results, baseline_effect
        )

        return loo_results

    def _summarize_loo_results(
        self, results: dict[str, Any], baseline_effect: float
    ) -> dict[str, Any]:
        """Summarize leave-one-out results."""
        summary = {}

        # Analyze unit leave-one-out
        if results["by_unit"]:
            unit_effects = []
            unit_changes = []

            for _unit, unit_result in results["by_unit"].items():
                if "error" not in unit_result and unit_result["effect"] is not None:
                    unit_effects.append(unit_result["effect"])
                    if unit_result["effect_change"] is not None:
                        unit_changes.append(unit_result["effect_change"])

            if unit_effects:
                summary["unit_analysis"] = {
                    "n_successful": len(unit_effects),
                    "effect_range": [min(unit_effects), max(unit_effects)],
                    "effect_std": np.std(unit_effects),
                    "max_change": max(np.abs(unit_changes)) if unit_changes else None,
                    "robust": max(np.abs(unit_changes)) < 0.1 * abs(baseline_effect)
                    if unit_changes and baseline_effect
                    else None,
                }

        # Analyze period leave-one-out
        if results["by_period"]:
            period_effects = []
            period_changes = []

            for _period, period_result in results["by_period"].items():
                if "error" not in period_result and period_result["effect"] is not None:
                    period_effects.append(period_result["effect"])
                    if period_result["effect_change"] is not None:
                        period_changes.append(period_result["effect_change"])

            if period_effects:
                summary["period_analysis"] = {
                    "n_successful": len(period_effects),
                    "effect_range": [min(period_effects), max(period_effects)],
                    "effect_std": np.std(period_effects),
                    "max_change": max(np.abs(period_changes))
                    if period_changes
                    else None,
                    "robust": max(np.abs(period_changes)) < 0.1 * abs(baseline_effect)
                    if period_changes and baseline_effect
                    else None,
                }

        return summary

    def plot_sensitivity_results(
        self, sensitivity_results: dict[str, Any], save_path: str | None = None
    ) -> Any:
        """Plot sensitivity analysis results.

        Args:
            sensitivity_results (Dict): Results from sensitivity analysis
            save_path (str, optional): Path to save the plot

        Returns:
            Matplotlib figure object
        """
        if not PLOTTING_AVAILABLE:
            raise ImportError(
                "Plotting requires matplotlib and seaborn: pip install matplotlib seaborn"
            )

        # Extract effects from random exclusions
        if "random_exclusions" in sensitivity_results:
            effects = []
            for exclusion in sensitivity_results["random_exclusions"]:
                if "error" not in exclusion:
                    effect = self._extract_effect_size(exclusion["results"])
                    if effect is not None:
                        effects.append(effect)
        else:
            effects = []

        if not effects:
            warnings.warn("No effects to plot", stacklevel=2)
            return None

        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Histogram of effects
        ax1.hist(effects, bins=20, alpha=0.7, edgecolor="black")

        # Add baseline effect line if available
        baseline_effect = sensitivity_results.get("summary", {}).get("baseline_effect")
        if baseline_effect is not None:
            ax1.axvline(
                baseline_effect,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Baseline Effect: {baseline_effect:.3f}",
            )
            ax1.legend()

        ax1.set_xlabel("Treatment Effect")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Distribution of Treatment Effects\n(Random Sample Exclusions)")
        ax1.grid(True, alpha=0.3)

        # Box plot
        ax2.boxplot(effects)
        ax2.set_ylabel("Treatment Effect")
        ax2.set_title("Treatment Effect Distribution")
        ax2.grid(True, alpha=0.3)

        if baseline_effect is not None:
            ax2.axhline(baseline_effect, color="red", linestyle="--", linewidth=2)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig
