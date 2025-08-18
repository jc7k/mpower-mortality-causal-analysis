"""Incremental Cost-Effectiveness Ratio (ICER) Analysis for MPOWER Policies.

This module provides methods for calculating ICERs, performing probabilistic
sensitivity analysis, and generating cost-effectiveness acceptability curves.
"""

from typing import Any

import numpy as np
import pandas as pd


class ICERAnalysis:
    """Incremental Cost-Effectiveness Ratio analysis for policy evaluation.

    This class calculates ICERs, performs dominance analysis, and conducts
    probabilistic sensitivity analysis for MPOWER policy combinations.

    Args:
        costs: Dictionary of intervention costs
        effects: Dictionary of health effects (QALYs/DALYs)
        wtp_threshold: Willingness-to-pay threshold per QALY/DALY
    """

    def __init__(
        self,
        costs: dict[str, float] | None = None,
        effects: dict[str, float] | None = None,
        wtp_threshold: float = 50000,
    ):
        """Initialize ICER analysis."""
        self.costs = costs if costs is not None else {}
        self.effects = effects if effects is not None else {}
        self.wtp_threshold = wtp_threshold
        self.icer_results = {}

    def calculate_icer(
        self,
        intervention: str,
        comparator: str = "status_quo",
        costs_dict: dict[str, float] | None = None,
        effects_dict: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Calculate ICER between intervention and comparator.

        ICER = (Cost_int - Cost_comp) / (Effect_int - Effect_comp)

        Args:
            intervention: Name of intervention strategy
            comparator: Name of comparator strategy (default: status_quo)
            costs_dict: Optional costs dictionary to use
            effects_dict: Optional effects dictionary to use

        Returns:
            Dictionary with ICER and component values
        """
        # Use provided dictionaries or instance variables
        costs = costs_dict if costs_dict is not None else self.costs
        effects = effects_dict if effects_dict is not None else self.effects

        # Handle missing comparator (assume zero cost/effect)
        if comparator not in costs:
            costs[comparator] = 0
        if comparator not in effects:
            effects[comparator] = 0

        # Get values
        cost_intervention = costs.get(intervention, 0)
        cost_comparator = costs.get(comparator, 0)
        effect_intervention = effects.get(intervention, 0)
        effect_comparator = effects.get(comparator, 0)

        # Calculate incremental values
        incremental_cost = cost_intervention - cost_comparator
        incremental_effect = effect_intervention - effect_comparator

        # Calculate ICER
        if incremental_effect == 0:
            if incremental_cost > 0:
                icer = float("inf")  # Dominated (more cost, no benefit)
            elif incremental_cost < 0:
                icer = float("-inf")  # Dominant (less cost, same effect)
            else:
                icer = 0  # Same cost and effect
        else:
            icer = incremental_cost / incremental_effect

        # Determine dominance status
        if incremental_cost < 0 and incremental_effect > 0:
            dominance = "dominant"  # Less cost, more effect
        elif incremental_cost > 0 and incremental_effect < 0:
            dominance = "dominated"  # More cost, less effect
        elif incremental_cost < 0 and incremental_effect < 0:
            dominance = "trade-off_negative"  # Less cost, less effect
        else:
            dominance = "trade-off"  # More cost, more effect

        # Cost-effectiveness determination
        cost_effective = (
            icer < self.wtp_threshold if not np.isinf(icer) else dominance == "dominant"
        )

        result = {
            "intervention": intervention,
            "comparator": comparator,
            "incremental_cost": incremental_cost,
            "incremental_effect": incremental_effect,
            "icer": icer,
            "dominance_status": dominance,
            "cost_effective": cost_effective,
            "wtp_threshold": self.wtp_threshold,
        }

        # Store result
        self.icer_results[f"{intervention}_vs_{comparator}"] = result

        return result

    def calculate_all_icers(
        self, strategies: list[str] | None = None, reference: str = "status_quo"
    ) -> pd.DataFrame:
        """Calculate ICERs for all strategy pairs.

        Args:
            strategies: List of strategies to compare (uses all if None)
            reference: Reference strategy for comparison

        Returns:
            DataFrame with all ICER comparisons
        """
        if strategies is None:
            strategies = list(set(list(self.costs.keys()) + list(self.effects.keys())))

        results = []

        # Compare each strategy to reference
        for strategy in strategies:
            if strategy != reference:
                icer_result = self.calculate_icer(strategy, reference)
                results.append(icer_result)

        # Pairwise comparisons
        for i, strategy1 in enumerate(strategies):
            for strategy2 in strategies[i + 1 :]:
                if reference not in (strategy1, strategy2):
                    icer_result = self.calculate_icer(strategy1, strategy2)
                    results.append(icer_result)

        return pd.DataFrame(results)

    def identify_efficient_frontier(
        self, strategies: list[str] | None = None
    ) -> list[str]:
        """Identify strategies on the efficient frontier.

        The efficient frontier consists of non-dominated strategies that
        maximize health benefits for a given cost level.

        Args:
            strategies: List of strategies to evaluate

        Returns:
            List of strategies on the efficient frontier
        """
        if strategies is None:
            strategies = list(set(list(self.costs.keys()) + list(self.effects.keys())))

        # Create strategy data
        strategy_data = []
        for strategy in strategies:
            strategy_data.append(
                {
                    "strategy": strategy,
                    "cost": self.costs.get(strategy, 0),
                    "effect": self.effects.get(strategy, 0),
                }
            )

        df = pd.DataFrame(strategy_data)
        df = df.sort_values("cost")

        # Identify efficient frontier
        frontier = []
        max_effect = -float("inf")

        for _, row in df.iterrows():
            # A strategy is on the frontier if it has higher effect than
            # all less expensive strategies
            if row["effect"] > max_effect:
                frontier.append(row["strategy"])
                max_effect = row["effect"]

        return frontier

    def probabilistic_sensitivity(
        self,
        n_simulations: int = 1000,
        cost_distributions: dict[str, tuple[str, float, float]] | None = None,
        effect_distributions: dict[str, tuple[str, float, float]] | None = None,
    ) -> pd.DataFrame:
        """Perform probabilistic sensitivity analysis with parameter uncertainty.

        Args:
            n_simulations: Number of Monte Carlo simulations
            cost_distributions: Dict of (distribution, mean, std) for costs
            effect_distributions: Dict of (distribution, mean, std) for effects

        Returns:
            DataFrame with simulation results
        """
        # Default distributions if not provided
        if cost_distributions is None:
            cost_distributions = {
                strategy: ("normal", cost, cost * 0.2)
                for strategy, cost in self.costs.items()
            }

        if effect_distributions is None:
            effect_distributions = {
                strategy: ("normal", effect, effect * 0.1)
                for strategy, effect in self.effects.items()
            }

        results = []
        strategies = list(set(list(self.costs.keys()) + list(self.effects.keys())))

        for sim in range(n_simulations):
            # Sample costs and effects
            sampled_costs = {}
            sampled_effects = {}

            for strategy in strategies:
                # Sample cost
                if strategy in cost_distributions:
                    dist_type, mean, std = cost_distributions[strategy]
                    if dist_type == "normal":
                        sampled_costs[strategy] = max(
                            0, np.random.normal(mean, abs(std))
                        )
                    elif dist_type == "gamma":
                        shape = (mean / std) ** 2
                        scale = std**2 / mean
                        sampled_costs[strategy] = np.random.gamma(shape, scale)
                    else:
                        sampled_costs[strategy] = mean
                else:
                    sampled_costs[strategy] = self.costs.get(strategy, 0)

                # Sample effect
                if strategy in effect_distributions:
                    dist_type, mean, std = effect_distributions[strategy]
                    if dist_type == "normal":
                        sampled_effects[strategy] = max(
                            0, np.random.normal(mean, abs(std))
                        )
                    elif dist_type == "beta":
                        # Convert mean and std to alpha and beta parameters
                        var = std**2
                        alpha = mean * (mean * (1 - mean) / var - 1)
                        beta = (1 - mean) * (mean * (1 - mean) / var - 1)
                        sampled_effects[strategy] = (
                            np.random.beta(alpha, beta) * mean * 2
                        )
                    else:
                        sampled_effects[strategy] = mean
                else:
                    sampled_effects[strategy] = self.effects.get(strategy, 0)

            # Calculate ICERs with sampled values
            for _i, strategy in enumerate(strategies):
                if strategy != "status_quo":
                    icer_result = self.calculate_icer(
                        strategy, "status_quo", sampled_costs, sampled_effects
                    )

                    results.append(
                        {
                            "simulation": sim,
                            "strategy": strategy,
                            "cost": sampled_costs[strategy],
                            "effect": sampled_effects[strategy],
                            "incremental_cost": icer_result["incremental_cost"],
                            "incremental_effect": icer_result["incremental_effect"],
                            "icer": icer_result["icer"]
                            if not np.isinf(icer_result["icer"])
                            else np.nan,
                            "cost_effective": icer_result["cost_effective"],
                        }
                    )

        return pd.DataFrame(results)

    def ceac_curve(
        self,
        thresholds: list[float] | None = None,
        psa_results: pd.DataFrame | None = None,
        n_simulations: int = 1000,
    ) -> pd.DataFrame:
        """Generate Cost-Effectiveness Acceptability Curve data.

        Args:
            thresholds: List of WTP thresholds to evaluate
            psa_results: Pre-computed PSA results (runs new if None)
            n_simulations: Number of simulations if running new PSA

        Returns:
            DataFrame with probability of cost-effectiveness at each threshold
        """
        if thresholds is None:
            thresholds = np.linspace(0, 100000, 21).tolist()

        # Run PSA if results not provided
        if psa_results is None:
            psa_results = self.probabilistic_sensitivity(n_simulations)

        ceac_data = []
        strategies = psa_results["strategy"].unique()

        for threshold in thresholds:
            for strategy in strategies:
                strategy_results = psa_results[psa_results["strategy"] == strategy]

                # Calculate probability of cost-effectiveness at this threshold
                cost_effective = (
                    strategy_results["incremental_cost"]
                    / strategy_results["incremental_effect"]
                    < threshold
                ).fillna(False)  # Handle inf/nan

                # Handle dominant cases
                dominant = (strategy_results["incremental_cost"] < 0) & (
                    strategy_results["incremental_effect"] > 0
                )
                cost_effective = cost_effective | dominant

                prob_ce = cost_effective.mean()

                ceac_data.append(
                    {
                        "threshold": threshold,
                        "strategy": strategy,
                        "probability_cost_effective": prob_ce,
                    }
                )

        return pd.DataFrame(ceac_data)

    def calculate_evpi(
        self, psa_results: pd.DataFrame | None = None, n_simulations: int = 1000
    ) -> float:
        """Calculate Expected Value of Perfect Information.

        EVPI represents the maximum amount a decision-maker should be willing
        to pay for perfect information about uncertain parameters.

        Args:
            psa_results: PSA results DataFrame
            n_simulations: Number of simulations if running new PSA

        Returns:
            EVPI value
        """
        if psa_results is None:
            psa_results = self.probabilistic_sensitivity(n_simulations)

        # Calculate net monetary benefit for each strategy in each simulation
        strategies = psa_results["strategy"].unique()
        simulations = psa_results["simulation"].unique()

        nmb_data = []
        for sim in simulations:
            sim_data = psa_results[psa_results["simulation"] == sim]
            for strategy in strategies:
                strategy_data = sim_data[sim_data["strategy"] == strategy]
                if not strategy_data.empty:
                    row = strategy_data.iloc[0]
                    nmb = row["effect"] * self.wtp_threshold - row["cost"]
                    nmb_data.append(
                        {
                            "simulation": sim,
                            "strategy": strategy,
                            "nmb": nmb,
                        }
                    )

        nmb_df = pd.DataFrame(nmb_data)

        # Expected value with perfect information
        evpi_value = 0
        for sim in simulations:
            sim_nmb = nmb_df[nmb_df["simulation"] == sim]
            max_nmb = sim_nmb["nmb"].max()
            evpi_value += max_nmb
        evpi_value /= len(simulations)

        # Expected value with current information
        avg_nmb_by_strategy = nmb_df.groupby("strategy")["nmb"].mean()
        evci = avg_nmb_by_strategy.max()

        # EVPI is the difference
        return evpi_value - evci

    def generate_tornado_diagram_data(
        self,
        base_case_strategy: str,
        parameter_ranges: dict[str, tuple[float, float]] | None = None,
    ) -> pd.DataFrame:
        """Generate data for tornado diagram (one-way sensitivity).

        Args:
            base_case_strategy: Strategy to analyze
            parameter_ranges: Dict of parameter names and (low, high) values

        Returns:
            DataFrame with parameter sensitivity results
        """
        if parameter_ranges is None:
            # Default: vary each parameter by ±20%
            base_cost = self.costs.get(base_case_strategy, 0)
            base_effect = self.effects.get(base_case_strategy, 0)

            parameter_ranges = {
                f"{base_case_strategy}_cost": (base_cost * 0.8, base_cost * 1.2),
                f"{base_case_strategy}_effect": (base_effect * 0.8, base_effect * 1.2),
            }

        tornado_data = []
        base_icer = self.calculate_icer(base_case_strategy)["icer"]

        for param, (low_val, high_val) in parameter_ranges.items():
            # Test low value
            if "cost" in param:
                orig_cost = self.costs.get(base_case_strategy, 0)
                self.costs[base_case_strategy] = low_val
                low_icer = self.calculate_icer(base_case_strategy)["icer"]
                self.costs[base_case_strategy] = high_val
                high_icer = self.calculate_icer(base_case_strategy)["icer"]
                self.costs[base_case_strategy] = orig_cost
            else:  # effect parameter
                orig_effect = self.effects.get(base_case_strategy, 0)
                self.effects[base_case_strategy] = low_val
                low_icer = self.calculate_icer(base_case_strategy)["icer"]
                self.effects[base_case_strategy] = high_val
                high_icer = self.calculate_icer(base_case_strategy)["icer"]
                self.effects[base_case_strategy] = orig_effect

            # Calculate range of ICER change
            icer_range = (
                abs(high_icer - low_icer)
                if not (np.isinf(high_icer) or np.isinf(low_icer))
                else 0
            )

            tornado_data.append(
                {
                    "parameter": param,
                    "low_value": low_val,
                    "high_value": high_val,
                    "low_icer": low_icer,
                    "high_icer": high_icer,
                    "icer_range": icer_range,
                    "base_icer": base_icer,
                }
            )

        # Sort by impact (largest range first)
        tornado_df = pd.DataFrame(tornado_data)
        return tornado_df.sort_values("icer_range", ascending=False)
