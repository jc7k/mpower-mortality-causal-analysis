"""Health Economic Outcome Modeling for MPOWER Policy Analysis.

This module calculates health outcomes in economic terms including QALYs
(Quality-Adjusted Life Years) and DALYs (Disability-Adjusted Life Years)
for tobacco control policy evaluation.
"""

from typing import Any

import numpy as np
import pandas as pd


class HealthOutcomeModel:
    """Models health outcomes in economic terms for policy evaluation.

    This class provides methods for calculating QALYs gained and DALYs averted
    from mortality reductions, as well as Markov modeling for disease progression.

    Args:
        mortality_data: DataFrame with mortality rates by age, country, year
        life_tables: Life expectancy data by age and country
        discount_rate: Annual discount rate for future health benefits (default 3%)
    """

    # Standard disability weights from GBD for tobacco-related diseases
    DISABILITY_WEIGHTS = {
        "lung_cancer": 0.451,
        "cardiovascular": 0.235,
        "ihd": 0.268,
        "copd": 0.328,
        "stroke": 0.362,
    }

    # Age-specific utility weights for QALY calculations
    AGE_UTILITY_WEIGHTS = {
        "0-4": 0.99,
        "5-14": 0.98,
        "15-24": 0.97,
        "25-34": 0.96,
        "35-44": 0.94,
        "45-54": 0.91,
        "55-64": 0.87,
        "65-74": 0.82,
        "75+": 0.75,
    }

    def __init__(
        self,
        mortality_data: pd.DataFrame,
        life_tables: pd.DataFrame | None = None,
        discount_rate: float = 0.03,
    ):
        """Initialize the health outcome model."""
        self.mortality = mortality_data
        self.life_tables = (
            life_tables
            if life_tables is not None
            else self._generate_default_life_tables()
        )
        self.discount_rate = discount_rate

    def _generate_default_life_tables(self) -> pd.DataFrame:
        """Generate default life tables if not provided."""
        # Simplified life expectancy by age group
        default_life_exp = {
            "0-4": 75,
            "5-14": 71,
            "15-24": 61,
            "25-34": 52,
            "35-44": 42,
            "45-54": 33,
            "55-64": 24,
            "65-74": 16,
            "75+": 9,
        }
        return pd.DataFrame(
            list(default_life_exp.items()), columns=["age_group", "life_expectancy"]
        )

    def calculate_qalys(
        self,
        mortality_reduction: float,
        age_distribution: pd.DataFrame | None = None,
        time_horizon: int = 30,
    ) -> dict[str, float]:
        """Calculate Quality-Adjusted Life Years gained from mortality reduction.

        Args:
            mortality_reduction: Reduction in mortality rate per 100,000
            age_distribution: Population distribution by age (optional)
            time_horizon: Years to project benefits (default 30)

        Returns:
            Dictionary with total QALYs, discounted QALYs, and by age group
        """
        qaly_results = {
            "total_qalys": 0,
            "discounted_qalys": 0,
            "qalys_by_age": {},
        }

        # Use uniform distribution if not provided
        if age_distribution is None:
            age_groups = list(self.AGE_UTILITY_WEIGHTS.keys())
            age_distribution = pd.DataFrame(
                {
                    "age_group": age_groups,
                    "proportion": [1 / len(age_groups)] * len(age_groups),
                }
            )

        for _, row in age_distribution.iterrows():
            age_group = row["age_group"]
            proportion = row["proportion"]

            # Get life expectancy for age group
            life_exp = (
                self.life_tables[self.life_tables["age_group"] == age_group][
                    "life_expectancy"
                ].values[0]
                if age_group in self.life_tables["age_group"].values
                else 20
            )

            # Calculate years of life saved
            years_saved = min(life_exp, time_horizon)

            # Apply utility weight
            utility = self.AGE_UTILITY_WEIGHTS.get(age_group, 0.8)
            qalys = mortality_reduction * proportion * years_saved * utility / 100000

            # Calculate discounted QALYs
            discounted_qalys = self._apply_discounting(qalys, years_saved)

            qaly_results["qalys_by_age"][age_group] = qalys
            qaly_results["total_qalys"] += qalys
            qaly_results["discounted_qalys"] += discounted_qalys

        return qaly_results

    def calculate_dalys(
        self,
        mortality_reduction: float,
        disease: str,
        prevalence_rate: float = 0.05,
        duration: float = 5.0,
    ) -> dict[str, float]:
        """Calculate Disability-Adjusted Life Years averted.

        DALYs = YLL (Years of Life Lost) + YLD (Years Lived with Disability)

        Args:
            mortality_reduction: Reduction in mortality rate per 100,000
            disease: Disease type for disability weight
            prevalence_rate: Disease prevalence in population
            duration: Average duration of disability

        Returns:
            Dictionary with YLL, YLD, total DALYs averted
        """
        # Years of Life Lost (YLL) from premature mortality
        avg_life_expectancy = self.life_tables["life_expectancy"].mean()
        yll = mortality_reduction * avg_life_expectancy / 100000

        # Years Lived with Disability (YLD)
        disability_weight = self.DISABILITY_WEIGHTS.get(disease, 0.3)
        cases_prevented = mortality_reduction * (1 / 0.3)  # Rough case fatality ratio
        yld = cases_prevented * prevalence_rate * duration * disability_weight / 100000

        # Total DALYs averted
        total_dalys = yll + yld

        # Apply discounting
        discounted_dalys = self._apply_discounting(total_dalys, avg_life_expectancy)

        return {
            "yll_averted": yll,
            "yld_averted": yld,
            "total_dalys_averted": total_dalys,
            "discounted_dalys": discounted_dalys,
        }

    def markov_model(
        self,
        transition_probs: dict[str, dict[str, float]],
        initial_state: dict[str, float],
        time_steps: int = 50,
        intervention_effect: float = 0.2,
    ) -> pd.DataFrame:
        """Run Markov model for disease progression with/without intervention.

        Args:
            transition_probs: State transition probability matrix
            initial_state: Initial distribution across health states
            time_steps: Number of yearly cycles to simulate
            intervention_effect: Reduction in disease progression rate

        Returns:
            DataFrame with state distributions over time
        """
        # Default transition matrix for smoking-related disease progression
        if not transition_probs:
            transition_probs = {
                "healthy": {
                    "healthy": 0.95,
                    "at_risk": 0.04,
                    "diseased": 0.01,
                    "dead": 0.0,
                },
                "at_risk": {
                    "healthy": 0.02,
                    "at_risk": 0.90,
                    "diseased": 0.07,
                    "dead": 0.01,
                },
                "diseased": {
                    "healthy": 0.0,
                    "at_risk": 0.01,
                    "diseased": 0.94,
                    "dead": 0.05,
                },
                "dead": {"healthy": 0.0, "at_risk": 0.0, "diseased": 0.0, "dead": 1.0},
            }

        if not initial_state:
            initial_state = {
                "healthy": 0.7,
                "at_risk": 0.2,
                "diseased": 0.1,
                "dead": 0.0,
            }

        states = list(initial_state.keys())
        n_states = len(states)

        # Convert to transition matrix
        trans_matrix = np.zeros((n_states, n_states))
        for i, from_state in enumerate(states):
            for j, to_state in enumerate(states):
                trans_matrix[i, j] = transition_probs[from_state][to_state]

        # Apply intervention effect (reduce disease progression)
        trans_matrix_intervention = trans_matrix.copy()
        if intervention_effect > 0:
            # Reduce transition to worse states
            trans_matrix_intervention[0, 1] *= (
                1 - intervention_effect
            )  # healthy to at_risk
            trans_matrix_intervention[1, 2] *= (
                1 - intervention_effect
            )  # at_risk to diseased
            trans_matrix_intervention[2, 3] *= (
                1 - intervention_effect
            )  # diseased to dead

            # Normalize rows
            for i in range(n_states - 1):  # Don't normalize dead state
                row_sum = trans_matrix_intervention[i, :].sum()
                if row_sum > 0:
                    trans_matrix_intervention[i, :] /= row_sum

        # Run simulation
        results = []
        current_state = np.array([initial_state[s] for s in states])
        current_state_intervention = current_state.copy()

        for t in range(time_steps):
            results.append(
                {
                    "time": t,
                    **{f"{s}_baseline": current_state[i] for i, s in enumerate(states)},
                    **{
                        f"{s}_intervention": current_state_intervention[i]
                        for i, s in enumerate(states)
                    },
                }
            )

            # Update states
            current_state = current_state @ trans_matrix
            current_state_intervention = (
                current_state_intervention @ trans_matrix_intervention
            )

        return pd.DataFrame(results)

    def _apply_discounting(self, value: float, years: float) -> float:
        """Apply exponential discounting to future health benefits.

        Args:
            value: Undiscounted health outcome value
            years: Time horizon in years

        Returns:
            Discounted value
        """
        if self.discount_rate == 0:
            return value

        # Calculate present value using continuous discounting
        discount_factor = (1 - np.exp(-self.discount_rate * years)) / self.discount_rate
        return value * discount_factor / years

    def calculate_life_years_saved(
        self, mortality_reduction: float, population: int = 1000000
    ) -> dict[str, Any]:
        """Calculate total life years saved from mortality reduction.

        Args:
            mortality_reduction: Reduction in mortality rate per 100,000
            population: Total population affected

        Returns:
            Dictionary with life years saved metrics
        """
        # Deaths prevented
        deaths_prevented = (mortality_reduction / 100000) * population

        # Average remaining life expectancy
        avg_life_exp = self.life_tables["life_expectancy"].mean()

        # Total life years saved
        total_lys = deaths_prevented * avg_life_exp

        # Discounted life years
        discounted_lys = self._apply_discounting(total_lys, avg_life_exp)

        return {
            "deaths_prevented": deaths_prevented,
            "life_years_saved": total_lys,
            "discounted_life_years": discounted_lys,
            "avg_years_per_death_prevented": avg_life_exp,
        }

    def sensitivity_analysis(
        self,
        mortality_reduction: float,
        parameter_ranges: dict[str, tuple[float, float]] | None = None,
        n_simulations: int = 1000,
    ) -> pd.DataFrame:
        """Perform sensitivity analysis on health outcome calculations.

        Args:
            mortality_reduction: Base case mortality reduction
            parameter_ranges: Dict of parameter names and (min, max) ranges
            n_simulations: Number of Monte Carlo simulations

        Returns:
            DataFrame with simulation results
        """
        if parameter_ranges is None:
            parameter_ranges = {
                "discount_rate": (0.0, 0.05),
                "utility_weight": (0.7, 1.0),
                "life_expectancy": (15, 30),
            }

        results = []

        for _ in range(n_simulations):
            # Sample parameters
            params = {}
            for param, (min_val, max_val) in parameter_ranges.items():
                params[param] = np.random.uniform(min_val, max_val)

            # Calculate outcomes with sampled parameters
            orig_discount = self.discount_rate
            self.discount_rate = params.get("discount_rate", orig_discount)

            qaly_results = self.calculate_qalys(mortality_reduction)
            daly_results = self.calculate_dalys(mortality_reduction, "cardiovascular")

            self.discount_rate = orig_discount

            results.append(
                {
                    **params,
                    "qalys": qaly_results["discounted_qalys"],
                    "dalys": daly_results["discounted_dalys"],
                }
            )

        return pd.DataFrame(results)
