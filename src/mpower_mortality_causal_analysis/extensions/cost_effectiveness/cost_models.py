"""Cost Estimation Models for MPOWER Policy Implementation.

This module provides methods for estimating implementation costs and
healthcare savings from MPOWER tobacco control policies.
"""

from typing import Any

import numpy as np
import pandas as pd


class CostEstimator:
    """Estimates implementation and offset costs for MPOWER policies.

    This class provides methods for calculating policy implementation costs,
    healthcare cost savings, and productivity gains from reduced mortality.

    Args:
        country_data: DataFrame with country economic indicators
        discount_rate: Annual discount rate for costs (default 3%)
    """

    # WHO-CHOICE based implementation costs per capita (USD)
    MPOWER_COSTS_PER_CAPITA = {
        "M": {"low_income": 0.05, "middle_income": 0.08, "high_income": 0.15},
        "P": {"low_income": 0.10, "middle_income": 0.20, "high_income": 0.40},
        "O": {"low_income": 0.50, "middle_income": 1.00, "high_income": 2.50},
        "W": {"low_income": 0.08, "middle_income": 0.15, "high_income": 0.30},
        "E": {"low_income": 0.12, "middle_income": 0.25, "high_income": 0.50},
        "R": {"low_income": 0.03, "middle_income": 0.05, "high_income": 0.10},
    }

    # Healthcare costs per case by disease (USD, PPP-adjusted)
    HEALTHCARE_COSTS = {
        "lung_cancer": {"treatment": 50000, "palliative": 15000, "screening": 500},
        "cardiovascular": {"treatment": 30000, "management": 5000, "emergency": 10000},
        "ihd": {"treatment": 40000, "surgery": 60000, "rehabilitation": 8000},
        "copd": {"treatment": 25000, "management": 3000, "oxygen": 2000},
        "stroke": {"acute": 35000, "rehabilitation": 20000, "long_term": 50000},
    }

    # Productivity loss multipliers by age group
    PRODUCTIVITY_MULTIPLIERS = {
        "15-24": 0.5,
        "25-34": 1.0,
        "35-44": 1.2,
        "45-54": 1.1,
        "55-64": 0.8,
        "65-74": 0.3,
        "75+": 0.1,
    }

    def __init__(
        self, country_data: pd.DataFrame | None = None, discount_rate: float = 0.03
    ):
        """Initialize the cost estimator."""
        self.country_data = (
            country_data
            if country_data is not None
            else self._generate_default_country_data()
        )
        self.discount_rate = discount_rate

    def _generate_default_country_data(self) -> pd.DataFrame:
        """Generate default country economic data if not provided."""
        return pd.DataFrame(
            {
                "country": ["Example"],
                "gdp_per_capita": [10000],
                "income_group": ["middle_income"],
                "population": [10000000],
                "health_expenditure_per_capita": [500],
                "avg_wage": [15000],
            }
        )

    def implementation_costs(
        self,
        policy: str | list[str],
        country: str,
        years: int = 5,
        scale_factor: float = 1.0,
    ) -> dict[str, Any]:
        """Calculate policy implementation costs by country.

        Args:
            policy: MPOWER component(s) to implement ('M', 'P', 'O', 'W', 'E', 'R')
            country: Country name for cost calculation
            years: Implementation period in years
            scale_factor: Scaling factor for costs (e.g., partial implementation)

        Returns:
            Dictionary with total, annual, and per capita costs
        """
        # Get country data
        country_info = self._get_country_info(country)
        income_group = country_info["income_group"]
        population = country_info["population"]

        # Ensure policy is a list
        if isinstance(policy, str):
            policy = [policy]

        # Calculate costs for each policy component
        total_per_capita = 0
        component_costs = {}

        for component in policy:
            if component in self.MPOWER_COSTS_PER_CAPITA:
                cost_per_capita = self.MPOWER_COSTS_PER_CAPITA[component][income_group]
                cost_per_capita *= scale_factor
                component_costs[component] = cost_per_capita * population
                total_per_capita += cost_per_capita

        # Total implementation cost
        total_cost = total_per_capita * population

        # Annual costs (front-loaded: 40% year 1, 25% year 2, 15% year 3, 10% years 4-5)
        if years >= 5:
            annual_distribution = [0.4, 0.25, 0.15, 0.1, 0.1]
        else:
            annual_distribution = [1.0 / years] * years

        annual_costs = [total_cost * weight for weight in annual_distribution[:years]]

        # Discounted costs
        discounted_total = sum(
            cost / (1 + self.discount_rate) ** year
            for year, cost in enumerate(annual_costs)
        )

        return {
            "total_cost": total_cost,
            "discounted_cost": discounted_total,
            "annual_costs": annual_costs,
            "per_capita_cost": total_per_capita,
            "component_costs": component_costs,
            "years": years,
            "population": population,
        }

    def healthcare_savings(
        self,
        cases_prevented: dict[str, float],
        country: str | None = None,
        time_horizon: int = 10,
    ) -> dict[str, Any]:
        """Calculate healthcare cost offsets from prevention.

        Args:
            cases_prevented: Dictionary of disease types and cases prevented
            country: Country for cost adjustment (optional)
            time_horizon: Years over which savings accrue

        Returns:
            Dictionary with total and disease-specific savings
        """
        country_info = self._get_country_info(country) if country else None
        cost_adjustment = 1.0

        if country_info:
            # Adjust costs based on country's health expenditure
            base_health_exp = 500  # Reference health expenditure per capita
            actual_health_exp = country_info.get(
                "health_expenditure_per_capita", base_health_exp
            )
            cost_adjustment = actual_health_exp / base_health_exp

        total_savings = 0
        savings_by_disease = {}
        annual_savings = []

        for disease, cases in cases_prevented.items():
            if disease in self.HEALTHCARE_COSTS:
                # Average treatment cost for the disease
                disease_costs = self.HEALTHCARE_COSTS[disease]
                avg_cost = np.mean(list(disease_costs.values())) * cost_adjustment

                # Total savings for this disease
                disease_savings = cases * avg_cost
                savings_by_disease[disease] = disease_savings
                total_savings += disease_savings

        # Distribute savings over time (more savings in later years as cases accumulate)
        for year in range(time_horizon):
            # Sigmoid curve for accumulation of savings
            proportion = 1 / (1 + np.exp(-0.5 * (year - time_horizon / 2)))
            annual_saving = (total_savings / time_horizon) * proportion * 2
            annual_savings.append(annual_saving)

        # Normalize to ensure total matches
        scaling = total_savings / sum(annual_savings) if sum(annual_savings) > 0 else 1
        annual_savings = [s * scaling for s in annual_savings]

        # Discounted savings
        discounted_savings = sum(
            saving / (1 + self.discount_rate) ** year
            for year, saving in enumerate(annual_savings)
        )

        return {
            "total_savings": total_savings,
            "discounted_savings": discounted_savings,
            "savings_by_disease": savings_by_disease,
            "annual_savings": annual_savings,
            "cost_adjustment_factor": cost_adjustment,
            "time_horizon": time_horizon,
        }

    def productivity_gains(
        self,
        mortality_reduction: float,
        age_distribution: pd.DataFrame | None = None,
        country: str | None = None,
        working_years: int = 20,
    ) -> dict[str, Any]:
        """Calculate economic value of prevented premature deaths.

        Args:
            mortality_reduction: Deaths prevented per 100,000
            age_distribution: Distribution of prevented deaths by age
            country: Country for wage adjustment
            working_years: Average remaining working years

        Returns:
            Dictionary with productivity gains
        """
        country_info = self._get_country_info(country) if country else None
        avg_wage = country_info.get("avg_wage", 15000) if country_info else 15000
        population = (
            country_info.get("population", 1000000) if country_info else 1000000
        )

        # Total deaths prevented
        deaths_prevented = (mortality_reduction / 100000) * population

        # Default age distribution if not provided
        if age_distribution is None:
            age_distribution = pd.DataFrame(
                {
                    "age_group": list(self.PRODUCTIVITY_MULTIPLIERS.keys()),
                    "proportion": [1 / len(self.PRODUCTIVITY_MULTIPLIERS)]
                    * len(self.PRODUCTIVITY_MULTIPLIERS),
                }
            )

        total_productivity_gain = 0
        gains_by_age = {}

        for _, row in age_distribution.iterrows():
            age_group = row["age_group"]
            proportion = row["proportion"]

            if age_group in self.PRODUCTIVITY_MULTIPLIERS:
                multiplier = self.PRODUCTIVITY_MULTIPLIERS[age_group]
                deaths_in_group = deaths_prevented * proportion

                # Calculate productivity loss prevented
                productivity = deaths_in_group * avg_wage * multiplier * working_years
                gains_by_age[age_group] = productivity
                total_productivity_gain += productivity

        # Discounted productivity gains
        annual_gain = total_productivity_gain / working_years
        discounted_gains = sum(
            annual_gain / (1 + self.discount_rate) ** year
            for year in range(working_years)
        )

        return {
            "total_productivity_gains": total_productivity_gain,
            "discounted_gains": discounted_gains,
            "gains_by_age": gains_by_age,
            "deaths_prevented": deaths_prevented,
            "avg_wage_used": avg_wage,
            "working_years": working_years,
        }

    def calculate_net_costs(
        self,
        policy: str | list[str],
        country: str,
        mortality_reduction: float,
        cases_prevented: dict[str, float],
        time_horizon: int = 10,
    ) -> dict[str, Any]:
        """Calculate net costs considering all offsets.

        Args:
            policy: MPOWER component(s) to implement
            country: Country name
            mortality_reduction: Deaths prevented per 100,000
            cases_prevented: Disease cases prevented
            time_horizon: Analysis time horizon

        Returns:
            Dictionary with net cost analysis
        """
        # Implementation costs
        impl_costs = self.implementation_costs(
            policy, country, years=min(5, time_horizon)
        )

        # Healthcare savings
        health_savings = self.healthcare_savings(cases_prevented, country, time_horizon)

        # Productivity gains
        prod_gains = self.productivity_gains(mortality_reduction, country=country)

        # Net costs (negative means net savings)
        net_cost = (
            impl_costs["discounted_cost"]
            - health_savings["discounted_savings"]
            - prod_gains["discounted_gains"]
        )

        # Break-even analysis
        cumulative_costs = []
        cumulative_savings = []
        for year in range(time_horizon):
            # Costs (front-loaded)
            if year < len(impl_costs["annual_costs"]):
                cum_cost = sum(impl_costs["annual_costs"][: year + 1])
            else:
                cum_cost = impl_costs["total_cost"]

            # Savings (accumulating)
            if year < len(health_savings["annual_savings"]):
                cum_saving = sum(health_savings["annual_savings"][: year + 1])
                cum_saving += (
                    prod_gains["total_productivity_gains"] / time_horizon
                ) * (year + 1)
            else:
                cum_saving = (
                    health_savings["total_savings"]
                    + prod_gains["total_productivity_gains"]
                )

            cumulative_costs.append(cum_cost)
            cumulative_savings.append(cum_saving)

        # Find break-even point
        break_even_year = None
        for year, (cost, saving) in enumerate(
            zip(cumulative_costs, cumulative_savings, strict=False)
        ):
            if saving >= cost:
                break_even_year = year + 1
                break

        return {
            "implementation_costs": impl_costs["discounted_cost"],
            "healthcare_savings": health_savings["discounted_savings"],
            "productivity_gains": prod_gains["discounted_gains"],
            "net_cost": net_cost,
            "roi": (
                health_savings["discounted_savings"] + prod_gains["discounted_gains"]
            )
            / impl_costs["discounted_cost"]
            if impl_costs["discounted_cost"] > 0
            else float("inf"),
            "break_even_year": break_even_year,
            "cumulative_costs": cumulative_costs,
            "cumulative_savings": cumulative_savings,
            "cost_effective": net_cost < 0,
        }

    def _get_country_info(self, country: str | None) -> dict[str, Any]:
        """Get economic information for a country."""
        if country is None or self.country_data is None:
            return self._generate_default_country_data().iloc[0].to_dict()

        country_row = self.country_data[self.country_data["country"] == country]
        if country_row.empty:
            # Return default if country not found
            return self._generate_default_country_data().iloc[0].to_dict()

        return country_row.iloc[0].to_dict()

    def sensitivity_analysis(
        self,
        policy: str | list[str],
        country: str,
        base_mortality_reduction: float,
        parameter_ranges: dict[str, tuple[float, float]] | None = None,
        n_simulations: int = 1000,
    ) -> pd.DataFrame:
        """Perform sensitivity analysis on cost estimates.

        Args:
            policy: MPOWER component(s)
            country: Country name
            base_mortality_reduction: Base case mortality reduction
            parameter_ranges: Parameter ranges for sensitivity
            n_simulations: Number of Monte Carlo simulations

        Returns:
            DataFrame with simulation results
        """
        if parameter_ranges is None:
            parameter_ranges = {
                "cost_multiplier": (0.5, 2.0),
                "savings_multiplier": (0.7, 1.3),
                "mortality_effect": (0.5, 1.5),
            }

        results = []

        for _ in range(n_simulations):
            # Sample parameters
            cost_mult = np.random.uniform(
                *parameter_ranges.get("cost_multiplier", (1, 1))
            )
            savings_mult = np.random.uniform(
                *parameter_ranges.get("savings_multiplier", (1, 1))
            )
            mort_mult = np.random.uniform(
                *parameter_ranges.get("mortality_effect", (1, 1))
            )

            # Adjusted mortality reduction
            adj_mortality = base_mortality_reduction * mort_mult

            # Simple cost calculation for sensitivity
            impl_cost = self.implementation_costs(
                policy, country, scale_factor=cost_mult
            )

            # Simplified savings calculation
            base_cases = {
                "cardiovascular": adj_mortality * 0.5,
                "lung_cancer": adj_mortality * 0.3,
            }
            health_savings = self.healthcare_savings(base_cases, country)
            health_savings["discounted_savings"] *= savings_mult

            prod_gains = self.productivity_gains(adj_mortality, country=country)

            net_cost = (
                impl_cost["discounted_cost"]
                - health_savings["discounted_savings"]
                - prod_gains["discounted_gains"]
            )

            results.append(
                {
                    "cost_multiplier": cost_mult,
                    "savings_multiplier": savings_mult,
                    "mortality_multiplier": mort_mult,
                    "implementation_cost": impl_cost["discounted_cost"],
                    "healthcare_savings": health_savings["discounted_savings"],
                    "productivity_gains": prod_gains["discounted_gains"],
                    "net_cost": net_cost,
                    "cost_effective": net_cost < 0,
                }
            )

        return pd.DataFrame(results)
