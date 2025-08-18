"""Main Cost-Effectiveness Analysis Pipeline for MPOWER Policies.

This module orchestrates the complete cost-effectiveness analysis workflow,
integrating health outcomes, costs, ICERs, and budget optimization.
"""

import json

from pathlib import Path
from typing import Any

import pandas as pd

from .budget_optimizer import BudgetOptimizer
from .ce_reporting import CEReporting
from .cost_models import CostEstimator
from .health_outcomes import HealthOutcomeModel
from .icer_analysis import ICERAnalysis


class CEPipeline:
    """Main orchestration class for cost-effectiveness analysis.

    This class integrates all components of the cost-effectiveness framework
    to provide a complete analysis workflow for MPOWER policy evaluation.

    Args:
        mortality_data: DataFrame with mortality outcomes from causal analysis
        cost_data: DataFrame with country-specific economic data
        life_tables: Life expectancy data by age and country
        wtp_threshold: Willingness-to-pay threshold for ICERs
    """

    def __init__(
        self,
        mortality_data: pd.DataFrame | None = None,
        cost_data: pd.DataFrame | None = None,
        life_tables: pd.DataFrame | None = None,
        wtp_threshold: float = 50000,
    ):
        """Initialize the cost-effectiveness pipeline."""
        self.mortality_data = mortality_data
        self.cost_data = cost_data
        self.life_tables = life_tables
        self.wtp_threshold = wtp_threshold

        # Initialize components
        self.health_model = None
        self.cost_estimator = None
        self.icer_analyzer = None
        self.budget_optimizer = None
        self.reporter = None

        # Results storage
        self.results = {}

    def load_causal_results(self, results_path: str | Path) -> None:
        """Load mortality reduction estimates from causal analysis.

        Args:
            results_path: Path to causal analysis results
        """
        results_path = Path(results_path)

        if results_path.is_file():
            # Load from file
            if results_path.suffix == ".csv":
                self.mortality_data = pd.read_csv(results_path)
            elif results_path.suffix == ".json":
                with open(results_path) as f:
                    data = json.load(f)
                self.mortality_data = pd.DataFrame(data)
        else:
            # Try to load from MPOWERAnalysisPipeline results
            summary_file = results_path / "analysis_summary.xlsx"
            if summary_file.exists():
                self.mortality_data = pd.read_excel(
                    summary_file, sheet_name="Main Results"
                )

    def initialize_components(self) -> None:
        """Initialize all analysis components."""
        # Health outcomes model
        self.health_model = HealthOutcomeModel(
            mortality_data=self.mortality_data, life_tables=self.life_tables
        )

        # Cost estimator
        self.cost_estimator = CostEstimator(country_data=self.cost_data)

        # ICER analyzer
        self.icer_analyzer = ICERAnalysis(wtp_threshold=self.wtp_threshold)

        # Reporting
        self.reporter = CEReporting()

    def run_analysis(
        self,
        country: str | None = None,
        policies: list[str] | None = None,
        budget: float | None = None,
        sensitivity: bool = True,
    ) -> dict[str, Any]:
        """Run complete cost-effectiveness analysis.

        Args:
            country: Country to analyze (uses all if None)
            policies: MPOWER policies to evaluate (default: all)
            budget: Budget constraint for optimization
            sensitivity: Whether to run sensitivity analysis

        Returns:
            Dictionary with complete analysis results
        """
        if policies is None:
            policies = ["M", "P", "O", "W", "E", "R"]

        # Initialize components if not done
        if self.health_model is None:
            self.initialize_components()

        results = {
            "country": country,
            "policies": policies,
        }

        # Get mortality reduction estimates
        if self.mortality_data is not None and country:
            country_mortality = self._get_country_mortality(country)
        else:
            # Use default values for demonstration
            country_mortality = {
                "lung_cancer": 10.5,
                "cardiovascular": 8.2,
                "copd": 6.3,
                "ihd": 7.8,
            }

        # 1. Calculate health outcomes
        health_results = self._calculate_health_outcomes(country_mortality, policies)
        results["health_outcomes"] = health_results

        # 2. Estimate costs
        cost_results = self._estimate_costs(policies, country, country_mortality)
        results["costs"] = cost_results

        # 3. Calculate ICERs
        icer_results = self._calculate_icers(cost_results, health_results)
        results["icers"] = icer_results

        # 4. Budget optimization (if budget provided)
        if budget:
            optimization_results = self._optimize_budget(
                policies, budget, cost_results, health_results
            )
            results["optimization"] = optimization_results

        # 5. Sensitivity analysis
        if sensitivity:
            sensitivity_results = self._run_sensitivity_analysis(
                policies, country, country_mortality
            )
            results["sensitivity"] = sensitivity_results

        # Store results
        self.results = results
        return results

    def _get_country_mortality(self, country: str) -> dict[str, float]:
        """Extract mortality reductions for a country."""
        if self.mortality_data is None:
            return {}

        country_data = self.mortality_data[
            self.mortality_data.get("country", "") == country
        ]

        if country_data.empty:
            # Return average if country not found
            return {
                "lung_cancer": self.mortality_data.get(
                    "lung_cancer_reduction", pd.Series([10])
                ).mean(),
                "cardiovascular": self.mortality_data.get(
                    "cardiovascular_reduction", pd.Series([8])
                ).mean(),
                "copd": self.mortality_data.get(
                    "copd_reduction", pd.Series([6])
                ).mean(),
                "ihd": self.mortality_data.get("ihd_reduction", pd.Series([7])).mean(),
            }

        # Extract only mortality reduction values
        row = country_data.iloc[0]
        result = {}
        for col in [
            "lung_cancer_reduction",
            "cardiovascular_reduction",
            "copd_reduction",
            "ihd_reduction",
        ]:
            if col in row:
                # Strip '_reduction' suffix for cleaner keys
                key = col.replace("_reduction", "")
                result[key] = float(row[col])

        # Fallback to simplified names if columns don't have '_reduction' suffix
        if not result:
            for col in ["lung_cancer", "cardiovascular", "copd", "ihd"]:
                if col in row:
                    result[col] = float(row[col])

        return result

    def _calculate_health_outcomes(
        self, mortality_reduction: dict[str, float], policies: list[str]
    ) -> dict[str, Any]:
        """Calculate QALYs and DALYs for policy combinations."""
        results = {}

        # Calculate for individual policies
        for policy in policies:
            # Attribute fraction of mortality reduction to each policy
            # This is simplified - in reality would use mechanism analysis
            policy_effect = {
                "M": 0.1,
                "P": 0.25,
                "O": 0.20,
                "W": 0.15,
                "E": 0.15,
                "R": 0.15,
            }.get(policy, 0.15)

            policy_mortality = {
                disease: reduction * policy_effect
                for disease, reduction in mortality_reduction.items()
            }

            # Calculate QALYs
            avg_mortality = sum(policy_mortality.values()) / len(policy_mortality)
            qaly_results = self.health_model.calculate_qalys(avg_mortality)

            # Calculate DALYs
            daly_results = self.health_model.calculate_dalys(
                avg_mortality, "cardiovascular"
            )

            results[policy] = {
                "qalys": qaly_results["discounted_qalys"],
                "dalys": daly_results["discounted_dalys"],
                "mortality_reduction": avg_mortality,
            }

        # Calculate for combined policies
        combined_mortality = sum(mortality_reduction.values()) / len(
            mortality_reduction
        )
        combined_qalys = self.health_model.calculate_qalys(combined_mortality)
        combined_dalys = self.health_model.calculate_dalys(
            combined_mortality, "cardiovascular"
        )

        results["combined"] = {
            "qalys": combined_qalys["discounted_qalys"],
            "dalys": combined_dalys["discounted_dalys"],
            "mortality_reduction": combined_mortality,
        }

        return results

    def _estimate_costs(
        self,
        policies: list[str],
        country: str | None,
        mortality_reduction: dict[str, float],
    ) -> dict[str, Any]:
        """Estimate implementation costs and savings."""
        results = {}

        # Individual policy costs
        for policy in policies:
            impl_costs = self.cost_estimator.implementation_costs(
                policy, country or "Example"
            )

            # Estimate cases prevented (simplified)
            avg_mortality = sum(mortality_reduction.values()) / len(mortality_reduction)
            policy_effect = {
                "M": 0.1,
                "P": 0.25,
                "O": 0.20,
                "W": 0.15,
                "E": 0.15,
                "R": 0.15,
            }.get(policy, 0.15)

            cases_prevented = {
                "cardiovascular": avg_mortality * policy_effect * 0.5,
                "lung_cancer": avg_mortality * policy_effect * 0.3,
                "copd": avg_mortality * policy_effect * 0.2,
            }

            health_savings = self.cost_estimator.healthcare_savings(
                cases_prevented, country
            )

            prod_gains = self.cost_estimator.productivity_gains(
                avg_mortality * policy_effect, country=country
            )

            results[policy] = {
                "implementation_cost": impl_costs["discounted_cost"],
                "healthcare_savings": health_savings["discounted_savings"],
                "productivity_gains": prod_gains["discounted_gains"],
                "net_cost": (
                    impl_costs["discounted_cost"]
                    - health_savings["discounted_savings"]
                    - prod_gains["discounted_gains"]
                ),
            }

        # Combined policy costs
        combined_impl = self.cost_estimator.implementation_costs(
            policies, country or "Example"
        )

        combined_cases = {
            "cardiovascular": sum(mortality_reduction.values()) * 0.5,
            "lung_cancer": sum(mortality_reduction.values()) * 0.3,
            "copd": sum(mortality_reduction.values()) * 0.2,
        }

        combined_savings = self.cost_estimator.healthcare_savings(
            combined_cases, country
        )

        combined_prod = self.cost_estimator.productivity_gains(
            sum(mortality_reduction.values()), country=country
        )

        results["combined"] = {
            "implementation_cost": combined_impl["discounted_cost"],
            "healthcare_savings": combined_savings["discounted_savings"],
            "productivity_gains": combined_prod["discounted_gains"],
            "net_cost": (
                combined_impl["discounted_cost"]
                - combined_savings["discounted_savings"]
                - combined_prod["discounted_gains"]
            ),
        }

        return results

    def _calculate_icers(
        self, cost_results: dict[str, Any], health_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Calculate ICERs for all policy combinations."""
        # Prepare costs and effects dictionaries
        costs = {
            strategy: cost_results[strategy]["net_cost"] for strategy in cost_results
        }

        effects = {
            strategy: health_results[strategy]["qalys"] for strategy in health_results
        }

        # Update ICER analyzer
        self.icer_analyzer.costs = costs
        self.icer_analyzer.effects = effects

        # Calculate all ICERs
        icer_df = self.icer_analyzer.calculate_all_icers()

        # Identify efficient frontier
        frontier = self.icer_analyzer.identify_efficient_frontier()

        # Run PSA
        psa_results = self.icer_analyzer.probabilistic_sensitivity(n_simulations=500)

        # Generate CEAC
        ceac_data = self.icer_analyzer.ceac_curve(psa_results=psa_results)

        return {
            "icer_table": icer_df.to_dict("records"),
            "efficient_frontier": frontier,
            "psa_summary": {
                "mean_icer": psa_results["icer"].mean(),
                "std_icer": psa_results["icer"].std(),
                "prob_cost_effective": (
                    psa_results["cost_effective"].mean()
                    if "cost_effective" in psa_results
                    else 0
                ),
            },
            "ceac_data": ceac_data.to_dict("records"),
        }

    def _optimize_budget(
        self,
        policies: list[str],
        budget: float,
        cost_results: dict[str, Any],
        health_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Optimize budget allocation across policies."""
        # Prepare costs and effects for optimization
        policy_costs = {
            p: cost_results[p]["implementation_cost"]
            for p in policies
            if p in cost_results
        }

        policy_effects = {
            p: health_results[p]["qalys"] for p in policies if p in health_results
        }

        # Initialize optimizer
        self.budget_optimizer = BudgetOptimizer(
            budget=budget, policies=policies, costs=policy_costs, effects=policy_effects
        )

        # Run optimization
        linear_result = self.budget_optimizer.optimize_allocation(method="linear")
        nonlinear_result = self.budget_optimizer.optimize_allocation(method="nonlinear")

        # Incremental allocation
        incremental_steps = self.budget_optimizer.incremental_allocation()

        # Sensitivity to budget
        budget_sensitivity = self.budget_optimizer.sensitivity_to_budget()

        return {
            "linear_optimization": linear_result,
            "nonlinear_optimization": nonlinear_result,
            "incremental_allocation": incremental_steps,
            "budget_sensitivity": budget_sensitivity.to_dict("records"),
        }

    def _run_sensitivity_analysis(
        self,
        policies: list[str],
        country: str | None,
        mortality_reduction: dict[str, float],
    ) -> dict[str, Any]:
        """Run comprehensive sensitivity analysis."""
        results = {}

        # Health outcome sensitivity
        avg_mortality = sum(mortality_reduction.values()) / len(mortality_reduction)
        health_sensitivity = self.health_model.sensitivity_analysis(
            avg_mortality, n_simulations=500
        )
        results["health_outcomes"] = health_sensitivity.to_dict("records")

        # Cost sensitivity
        if country:
            cost_sensitivity = self.cost_estimator.sensitivity_analysis(
                policies[0] if policies else "M",
                country,
                avg_mortality,
                n_simulations=500,
            )
            results["costs"] = cost_sensitivity.to_dict("records")

        # ICER tornado diagram
        if self.icer_analyzer.costs and self.icer_analyzer.effects:
            base_strategy = (
                "combined" if "combined" in self.icer_analyzer.costs else "M"
            )
            tornado_data = self.icer_analyzer.generate_tornado_diagram_data(
                base_strategy
            )
            results["tornado"] = tornado_data.to_dict("records")

        return results

    def generate_report(
        self, output_path: str | Path, report_format: str = "comprehensive"
    ) -> None:
        """Generate cost-effectiveness analysis report.

        Args:
            output_path: Path for report output
            report_format: Report format ('summary', 'detailed', 'comprehensive')
        """
        if self.reporter is None:
            self.reporter = CEReporting()

        self.reporter.generate_report(self.results, output_path, format=report_format)

    def export_results(
        self, output_dir: str | Path, formats: list[str] | None = None
    ) -> None:
        """Export analysis results in multiple formats.

        Args:
            output_dir: Directory for output files
            formats: List of formats ('json', 'excel', 'csv')
        """
        if formats is None:
            formats = ["json", "excel"]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if "json" in formats:
            with open(output_dir / "ce_results.json", "w") as f:
                json.dump(self.results, f, indent=2, default=str)

        if "excel" in formats:
            with pd.ExcelWriter(output_dir / "ce_results.xlsx") as writer:
                # Health outcomes sheet
                if "health_outcomes" in self.results:
                    health_df = pd.DataFrame(self.results["health_outcomes"])
                    health_df.to_excel(writer, sheet_name="Health Outcomes")

                # Costs sheet
                if "costs" in self.results:
                    costs_df = pd.DataFrame(self.results["costs"])
                    costs_df.to_excel(writer, sheet_name="Costs")

                # ICERs sheet
                if "icers" in self.results:
                    icer_df = pd.DataFrame(self.results["icers"].get("icer_table", []))
                    icer_df.to_excel(writer, sheet_name="ICERs")

                # Optimization sheet
                if "optimization" in self.results:
                    opt_data = self.results["optimization"]
                    if "linear_optimization" in opt_data:
                        opt_df = pd.DataFrame(
                            [opt_data["linear_optimization"]["allocation"]]
                        )
                        opt_df.to_excel(writer, sheet_name="Optimization")

        if "csv" in formats and "icers" in self.results:
            # Export key tables as CSV
            icer_df = pd.DataFrame(self.results["icers"].get("icer_table", []))
            icer_df.to_csv(output_dir / "icer_results.csv", index=False)
