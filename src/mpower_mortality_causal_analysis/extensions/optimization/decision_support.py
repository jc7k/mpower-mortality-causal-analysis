"""Decision Support System for MPOWER Policy Implementation.

This module generates actionable recommendations and implementation roadmaps
based on optimization results and feasibility analysis.
"""

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
        "Plotting packages not available. Visualization disabled.",
        stacklevel=2,
    )

# Constants
MPOWER_COMPONENTS = ["M", "P", "O", "W", "E", "R"]
COMPONENT_NAMES = {
    "M": "Monitor tobacco use and prevention policies",
    "P": "Protect from tobacco smoke",
    "O": "Offer help to quit tobacco use",
    "W": "Warn about the dangers of tobacco",
    "E": "Enforce bans on tobacco advertising",
    "R": "Raise taxes on tobacco",
}


class PolicyDecisionSupport:
    """Generates actionable policy recommendations and implementation guidance.

    This class integrates optimization results, feasibility analysis, and
    stakeholder considerations to provide practical implementation roadmaps.

    Parameters:
        optimization_results (dict): Results from optimization algorithms
        feasibility_results (dict): Political feasibility analysis results
        interaction_results (dict): Policy interaction analysis results
        country_data (pd.DataFrame): Country-specific contextual data
    """

    def __init__(
        self,
        optimization_results: dict[str, Any],
        feasibility_results: dict[str, Any] | None = None,
        interaction_results: dict[str, Any] | None = None,
        country_data: pd.DataFrame | None = None,
    ) -> None:
        self.optimization_results = optimization_results
        self.feasibility_results = feasibility_results or {}
        self.interaction_results = interaction_results or {}
        self.country_data = country_data
        self.recommendation_cache = {}

    def generate_roadmap(
        self,
        country: str,
        budget: float,
        time_horizon: int = 5,
        priority_weights: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Generate country-specific implementation roadmap.

        Creates a detailed implementation plan considering budget constraints,
        political feasibility, and policy synergies.

        Args:
            country: Target country for roadmap
            budget: Total available budget
            time_horizon: Implementation timeline in years
            priority_weights: Weights for different objectives

        Returns:
            Comprehensive implementation roadmap
        """
        if priority_weights is None:
            priority_weights = {
                "health_impact": 0.4,
                "feasibility": 0.3,
                "cost_effectiveness": 0.2,
                "synergies": 0.1,
            }

        # Extract optimal sequence from optimization results
        optimal_sequence = self._get_optimal_sequence(country)

        # Adjust for country-specific feasibility
        feasible_sequence = self._adjust_for_feasibility(country, optimal_sequence)

        # Consider budget constraints
        budget_adjusted_sequence = self._adjust_for_budget(
            feasible_sequence, budget, time_horizon
        )

        # Add implementation details
        detailed_roadmap = self._create_detailed_roadmap(
            country, budget_adjusted_sequence, budget, time_horizon
        )

        # Generate risk assessment
        risk_assessment = self._assess_implementation_risks(country, detailed_roadmap)

        # Create success metrics
        success_metrics = self._define_success_metrics(detailed_roadmap)

        return {
            "country": country,
            "budget": budget,
            "time_horizon": time_horizon,
            "implementation_sequence": detailed_roadmap,
            "risk_assessment": risk_assessment,
            "success_metrics": success_metrics,
            "estimated_impact": self._estimate_cumulative_impact(detailed_roadmap),
            "priority_weights": priority_weights,
        }

    def scenario_analysis(
        self,
        scenarios: list[dict[str, Any]],
        base_country: str = "Global",
    ) -> pd.DataFrame:
        """Perform what-if analysis across different scenarios.

        Compares outcomes under different budget, timeline, and priority
        configurations to support decision-making.

        Args:
            scenarios: List of scenario configurations
            base_country: Default country for analysis

        Returns:
            DataFrame comparing scenario outcomes
        """
        scenario_results = []

        for i, scenario in enumerate(scenarios):
            scenario_name = scenario.get("name", f"Scenario_{i + 1}")
            country = scenario.get("country", base_country)
            budget = scenario.get("budget", 1000000)
            time_horizon = scenario.get("time_horizon", 5)
            priority_weights = scenario.get("priority_weights", None)

            try:
                roadmap = self.generate_roadmap(
                    country, budget, time_horizon, priority_weights
                )

                scenario_results.append(
                    {
                        "scenario_name": scenario_name,
                        "country": country,
                        "budget": budget,
                        "time_horizon": time_horizon,
                        "total_policies": len(
                            [
                                p
                                for period in roadmap["implementation_sequence"]
                                for p in period.get("policies", [])
                            ]
                        ),
                        "estimated_impact": roadmap["estimated_impact"]["total_impact"],
                        "cost_effectiveness": roadmap["estimated_impact"][
                            "total_impact"
                        ]
                        / budget
                        if budget > 0
                        else 0,
                        "implementation_probability": np.mean(
                            [
                                period.get("feasibility_score", 0.5)
                                for period in roadmap["implementation_sequence"]
                            ]
                        ),
                        "high_risk_periods": sum(
                            1
                            for period in roadmap["implementation_sequence"]
                            if period.get("risk_level", "Medium") == "High"
                        ),
                    }
                )

            except Exception as e:
                scenario_results.append(
                    {
                        "scenario_name": scenario_name,
                        "country": country,
                        "budget": budget,
                        "time_horizon": time_horizon,
                        "error": str(e),
                    }
                )

        return pd.DataFrame(scenario_results)

    def _get_optimal_sequence(self, country: str) -> list[str]:
        """Extract optimal policy sequence for country.

        Args:
            country: Target country

        Returns:
            List of policies in optimal implementation order
        """
        # Try to get country-specific sequence from optimization results
        if "country_specific" in self.optimization_results:
            country_results = self.optimization_results["country_specific"].get(
                country, {}
            )
            if "optimal_sequence" in country_results:
                return country_results["optimal_sequence"]

        # Fall back to global optimal sequence
        if "optimal_sequence" in self.optimization_results:
            sequence = self.optimization_results["optimal_sequence"]
            # Handle case where sequence might be nested lists (from periods)
            if sequence and isinstance(sequence[0], list):
                # Flatten if it's a list of periods
                return [policy for period in sequence for policy in period]
            return sequence

        # Default sequence based on typical implementation priorities
        return ["W", "M", "P", "O", "E", "R"]

    def _adjust_for_feasibility(
        self,
        country: str,
        sequence: list[str],
    ) -> list[str]:
        """Adjust sequence based on political feasibility.

        Args:
            country: Target country
            sequence: Original policy sequence

        Returns:
            Feasibility-adjusted sequence
        """
        if not self.feasibility_results:
            return sequence

        # Get feasibility scores for country
        country_feasibility = {}
        for policy, feasibility_data in self.feasibility_results.items():
            if isinstance(feasibility_data, pd.DataFrame):
                country_scores = feasibility_data[
                    feasibility_data.get("country", feasibility_data.columns[0])
                    == country
                ]
                if not country_scores.empty:
                    score_col = f"{policy}_feasibility_score"
                    if score_col in country_scores.columns:
                        country_feasibility[policy] = country_scores[score_col].mean()

        if not country_feasibility:
            return sequence

        # Reorder sequence by feasibility (keeping high-impact policies early)
        _sequence_with_feasibility = [
            (policy, country_feasibility.get(policy, 0.5)) for policy in sequence
        ]

        # Sort by feasibility, but keep some high-impact policies early
        high_impact_policies = ["W", "P", "R"]  # Often high-impact policies

        adjusted_sequence = []
        remaining_policies = sequence.copy()

        # Add highly feasible high-impact policies first
        for policy in high_impact_policies:
            if policy in remaining_policies:
                feasibility = country_feasibility.get(policy, 0.5)
                if feasibility > 0.6:  # High feasibility threshold
                    adjusted_sequence.append(policy)
                    remaining_policies.remove(policy)

        # Add remaining policies by feasibility
        remaining_with_feasibility = [
            (policy, country_feasibility.get(policy, 0.5))
            for policy in remaining_policies
        ]
        remaining_with_feasibility.sort(key=lambda x: x[1], reverse=True)

        adjusted_sequence.extend([policy for policy, _ in remaining_with_feasibility])

        return adjusted_sequence

    def _adjust_for_budget(
        self,
        sequence: list[str],
        total_budget: float,
        time_horizon: int,
    ) -> list[list[str]]:
        """Adjust sequence for budget constraints over time periods.

        Args:
            sequence: Policy sequence
            total_budget: Total available budget
            time_horizon: Number of implementation periods

        Returns:
            List of policy lists for each time period
        """
        # Default policy costs (can be customized)
        default_costs = {
            "M": 50000,  # Monitoring systems
            "P": 100000,  # Smoke-free enforcement
            "O": 75000,  # Cessation services
            "W": 25000,  # Warning labels
            "E": 150000,  # Advertising bans enforcement
            "R": 30000,  # Tax policy implementation
        }

        budget_per_period = total_budget / time_horizon
        adjusted_sequence = []
        current_period = []
        current_budget = budget_per_period
        period = 0

        for policy in sequence:
            if period >= time_horizon:
                break

            cost = default_costs.get(policy, 50000)

            if cost <= current_budget:
                # Add to current period
                current_period.append(policy)
                current_budget -= cost
            else:
                # Start new period
                if current_period:
                    adjusted_sequence.append(current_period)
                    period += 1

                if period >= time_horizon:
                    break

                current_period = [policy]
                current_budget = budget_per_period - cost

        # Add final period if not empty
        if current_period and period < time_horizon:
            adjusted_sequence.append(current_period)

        # Fill remaining periods if needed
        while len(adjusted_sequence) < time_horizon:
            adjusted_sequence.append([])

        return adjusted_sequence

    def _create_detailed_roadmap(
        self,
        country: str,
        sequence: list[list[str]],
        budget: float,
        time_horizon: int,
    ) -> list[dict[str, Any]]:
        """Create detailed implementation roadmap.

        Args:
            country: Target country
            sequence: Policies by time period
            budget: Total budget
            time_horizon: Implementation timeline

        Returns:
            Detailed roadmap with implementation guidance
        """
        default_costs = {
            "M": 50000,
            "P": 100000,
            "O": 75000,
            "W": 25000,
            "E": 150000,
            "R": 30000,
        }

        detailed_roadmap = []

        for period, policies in enumerate(sequence):
            period_cost = sum(default_costs.get(policy, 50000) for policy in policies)

            # Calculate feasibility for this period
            period_feasibility = 0.7  # Default
            if self.feasibility_results and policies:
                feasibility_scores = []
                for policy in policies:
                    if policy in self.feasibility_results:
                        # Simplified feasibility lookup
                        feasibility_scores.append(0.7)  # Default value
                if feasibility_scores:
                    period_feasibility = np.mean(feasibility_scores)

            # Determine risk level
            risk_level = "Low"
            if len(policies) > 2:
                risk_level = "Medium"
            if len(policies) > 3 or period_feasibility < 0.4:
                risk_level = "High"

            # Create implementation guidance
            implementation_guidance = []
            for policy in policies:
                guidance = self._get_policy_guidance(policy, country)
                implementation_guidance.append(guidance)

            period_info = {
                "period": period + 1,
                "year": f"Year {period + 1}",
                "policies": policies,
                "policy_descriptions": [
                    f"{policy}: {COMPONENT_NAMES.get(policy, policy)}"
                    for policy in policies
                ],
                "estimated_cost": period_cost,
                "feasibility_score": period_feasibility,
                "risk_level": risk_level,
                "implementation_guidance": implementation_guidance,
                "success_indicators": self._get_period_success_indicators(policies),
                "potential_challenges": self._get_period_challenges(policies, country),
            }

            detailed_roadmap.append(period_info)

        return detailed_roadmap

    def _get_policy_guidance(self, policy: str, country: str) -> dict[str, Any]:
        """Get implementation guidance for specific policy.

        Args:
            policy: MPOWER component
            country: Target country

        Returns:
            Implementation guidance
        """
        guidance_templates = {
            "M": {
                "key_actions": [
                    "Establish tobacco surveillance system",
                    "Conduct regular population surveys",
                    "Monitor tobacco industry activities",
                ],
                "timeline": "6-12 months",
                "key_stakeholders": [
                    "Health Ministry",
                    "Statistics Office",
                    "Research Institutions",
                ],
                "critical_success_factors": [
                    "Adequate funding for surveys",
                    "Technical capacity building",
                    "Data quality assurance",
                ],
            },
            "P": {
                "key_actions": [
                    "Enact comprehensive smoke-free laws",
                    "Train enforcement officers",
                    "Establish penalty system",
                ],
                "timeline": "12-18 months",
                "key_stakeholders": ["Health Ministry", "Local Authorities", "Police"],
                "critical_success_factors": [
                    "Strong legal framework",
                    "Enforcement capacity",
                    "Public support",
                ],
            },
            "O": {
                "key_actions": [
                    "Establish quitlines",
                    "Train healthcare providers",
                    "Integrate into health services",
                ],
                "timeline": "8-15 months",
                "key_stakeholders": ["Health Ministry", "Healthcare Providers", "NGOs"],
                "critical_success_factors": [
                    "Healthcare system integration",
                    "Provider training",
                    "Service accessibility",
                ],
            },
            "W": {
                "key_actions": [
                    "Design warning labels",
                    "Update regulations",
                    "Monitor compliance",
                ],
                "timeline": "6-10 months",
                "key_stakeholders": [
                    "Health Ministry",
                    "Regulatory Authority",
                    "Industry",
                ],
                "critical_success_factors": [
                    "Clear regulations",
                    "Industry compliance",
                    "Public awareness",
                ],
            },
            "E": {
                "key_actions": [
                    "Ban tobacco advertising",
                    "Regulate promotion activities",
                    "Enforce compliance",
                ],
                "timeline": "12-24 months",
                "key_stakeholders": [
                    "Health Ministry",
                    "Media Regulators",
                    "Legal System",
                ],
                "critical_success_factors": [
                    "Comprehensive legislation",
                    "Enforcement mechanisms",
                    "Industry cooperation",
                ],
            },
            "R": {
                "key_actions": [
                    "Increase tobacco taxes",
                    "Simplify tax structure",
                    "Strengthen enforcement",
                ],
                "timeline": "6-12 months",
                "key_stakeholders": [
                    "Finance Ministry",
                    "Tax Authority",
                    "Health Ministry",
                ],
                "critical_success_factors": [
                    "Political commitment",
                    "Tax administration capacity",
                    "Anti-smuggling measures",
                ],
            },
        }

        return guidance_templates.get(
            policy,
            {
                "key_actions": [f"Implement {COMPONENT_NAMES.get(policy, policy)}"],
                "timeline": "12 months",
                "key_stakeholders": ["Health Ministry"],
                "critical_success_factors": [
                    "Political commitment",
                    "Adequate resources",
                ],
            },
        )

    def _get_period_success_indicators(self, policies: list[str]) -> list[str]:
        """Get success indicators for a period.

        Args:
            policies: Policies in the period

        Returns:
            List of success indicators
        """
        indicators = []
        for policy in policies:
            policy_indicators = {
                "M": ["Survey completion rate > 80%", "Data quality score > 0.8"],
                "P": ["100% coverage of public places", "Compliance rate > 90%"],
                "O": [
                    "Quitline utilization > 1000 calls/month",
                    "Provider training completion > 95%",
                ],
                "W": ["100% package compliance", "Public awareness > 70%"],
                "E": ["Advertisement violations < 5%", "Legal framework established"],
                "R": ["Tax increase implemented", "Revenue collection target met"],
            }
            indicators.extend(
                policy_indicators.get(policy, [f"{policy} implementation completed"])
            )

        return indicators

    def _get_period_challenges(self, policies: list[str], country: str) -> list[str]:
        """Get potential challenges for a period.

        Args:
            policies: Policies in the period
            country: Target country

        Returns:
            List of potential challenges
        """
        challenges = []

        # General challenges by policy
        policy_challenges = {
            "M": [
                "Limited technical capacity",
                "Funding constraints",
                "Data quality issues",
            ],
            "P": ["Enforcement challenges", "Industry resistance", "Public compliance"],
            "O": [
                "Healthcare system capacity",
                "Provider training",
                "Service integration",
            ],
            "W": [
                "Industry compliance",
                "Design effectiveness",
                "Regulatory complexity",
            ],
            "E": [
                "Industry legal challenges",
                "Enforcement capacity",
                "Media cooperation",
            ],
            "R": [
                "Political resistance",
                "Smuggling concerns",
                "Economic impact fears",
            ],
        }

        for policy in policies:
            challenges.extend(
                policy_challenges.get(policy, [f"{policy} implementation challenges"])
            )

        # Add period-specific challenges
        if len(policies) > 2:
            challenges.append("Coordination across multiple policies")
        if len(policies) > 3:
            challenges.append("Resource strain from simultaneous implementation")

        return list(set(challenges))  # Remove duplicates

    def _assess_implementation_risks(
        self,
        country: str,
        roadmap: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Assess implementation risks for roadmap.

        Args:
            country: Target country
            roadmap: Implementation roadmap

        Returns:
            Risk assessment results
        """
        risk_factors = []
        mitigation_strategies = []

        # Analyze each period
        for period in roadmap:
            if period["risk_level"] == "High":
                risk_factors.append(
                    f"Period {period['period']}: High complexity with {len(period['policies'])} policies"
                )
                mitigation_strategies.append(
                    f"Consider splitting Period {period['period']} policies across multiple periods"
                )

            if period["feasibility_score"] < 0.5:
                risk_factors.append(
                    f"Period {period['period']}: Low political feasibility ({period['feasibility_score']:.2f})"
                )
                mitigation_strategies.append(
                    f"Build stakeholder support before Period {period['period']}"
                )

        # Overall assessment
        total_cost = sum(period.get("estimated_cost", 0) for period in roadmap)
        avg_feasibility = np.mean(
            [period.get("feasibility_score", 0.5) for period in roadmap]
        )

        overall_risk = "Low"
        if avg_feasibility < 0.6 or total_cost > 500000:
            overall_risk = "Medium"
        if avg_feasibility < 0.4 or total_cost > 1000000:
            overall_risk = "High"

        return {
            "overall_risk_level": overall_risk,
            "key_risk_factors": risk_factors,
            "mitigation_strategies": mitigation_strategies,
            "total_estimated_cost": total_cost,
            "average_feasibility": avg_feasibility,
            "high_risk_periods": [
                period["period"] for period in roadmap if period["risk_level"] == "High"
            ],
        }

    def _define_success_metrics(self, roadmap: list[dict[str, Any]]) -> dict[str, Any]:
        """Define success metrics for implementation.

        Args:
            roadmap: Implementation roadmap

        Returns:
            Success metrics and targets
        """
        total_policies = sum(len(period.get("policies", [])) for period in roadmap)

        return {
            "implementation_completion": {
                "target": f"{total_policies}/{len(MPOWER_COMPONENTS)} MPOWER components",
                "measurement": "Number of policies successfully implemented",
            },
            "timeline_adherence": {
                "target": "90% of milestones met on schedule",
                "measurement": "Percentage of period targets achieved on time",
            },
            "budget_efficiency": {
                "target": "Implementation within 110% of budget",
                "measurement": "Actual cost vs. estimated cost ratio",
            },
            "stakeholder_engagement": {
                "target": "Support from key stakeholders maintained",
                "measurement": "Stakeholder satisfaction surveys > 70%",
            },
            "policy_effectiveness": {
                "target": "Measurable improvement in tobacco control indicators",
                "measurement": "Pre/post implementation outcome comparison",
            },
        }

    def _estimate_cumulative_impact(
        self, roadmap: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Estimate cumulative impact of implementation.

        Args:
            roadmap: Implementation roadmap

        Returns:
            Impact estimates
        """
        # Default impact values per policy (mortality reduction per 100,000)
        policy_impacts = {
            "M": 1.2,  # Monitoring enables better policies
            "P": 5.5,  # Smoke-free policies
            "O": 3.2,  # Cessation services
            "W": 2.8,  # Warning labels
            "E": 4.1,  # Advertising bans
            "R": 8.3,  # Tax increases
        }

        cumulative_impact = 0.0
        annual_impacts = []

        for period in roadmap:
            period_impact = sum(
                policy_impacts.get(policy, 2.0) for policy in period.get("policies", [])
            )

            # Discount factor for implementation delays
            discount_factor = 0.95 ** (period["period"] - 1)
            discounted_impact = period_impact * discount_factor

            cumulative_impact += discounted_impact
            annual_impacts.append(discounted_impact)

        return {
            "total_impact": cumulative_impact,
            "annual_impacts": annual_impacts,
            "average_annual_impact": np.mean(annual_impacts) if annual_impacts else 0.0,
            "impact_unit": "Mortality reduction per 100,000 population",
        }

    def export_roadmap(
        self,
        roadmap_data: dict[str, Any],
        output_path: str | Path,
        format_type: str = "excel",
    ) -> None:
        """Export roadmap to file.

        Args:
            roadmap_data: Roadmap data from generate_roadmap()
            output_path: Output file path
            format_type: Export format ('excel', 'json', 'pdf')
        """
        output_path = Path(output_path)

        if format_type == "excel":
            self._export_to_excel(roadmap_data, output_path)
        elif format_type == "json":
            import json

            with open(output_path, "w") as f:
                json.dump(roadmap_data, f, indent=2, default=str)
        else:
            warnings.warn(f"Export format '{format_type}' not supported. Using JSON.")
            import json

            json_path = output_path.with_suffix(".json")
            with open(json_path, "w") as f:
                json.dump(roadmap_data, f, indent=2, default=str)

    def _export_to_excel(self, roadmap_data: dict[str, Any], output_path: Path) -> None:
        """Export roadmap to Excel format.

        Args:
            roadmap_data: Roadmap data
            output_path: Excel file path
        """
        try:
            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                # Summary sheet
                summary_data = {
                    "Country": [roadmap_data["country"]],
                    "Budget": [roadmap_data["budget"]],
                    "Time Horizon": [roadmap_data["time_horizon"]],
                    "Total Impact": [roadmap_data["estimated_impact"]["total_impact"]],
                    "Risk Level": [
                        roadmap_data["risk_assessment"]["overall_risk_level"]
                    ],
                }
                pd.DataFrame(summary_data).to_excel(
                    writer, sheet_name="Summary", index=False
                )

                # Implementation schedule
                schedule_data = []
                for period in roadmap_data["implementation_sequence"]:
                    for policy in period.get("policies", []):
                        schedule_data.append(
                            {
                                "Period": period["period"],
                                "Year": period["year"],
                                "Policy": policy,
                                "Description": COMPONENT_NAMES.get(policy, policy),
                                "Cost": period["estimated_cost"]
                                / len(period["policies"])
                                if period["policies"]
                                else 0,
                                "Feasibility": period["feasibility_score"],
                                "Risk Level": period["risk_level"],
                            }
                        )

                if schedule_data:
                    pd.DataFrame(schedule_data).to_excel(
                        writer, sheet_name="Schedule", index=False
                    )

                # Risk assessment
                risk_data = {
                    "Risk Factor": roadmap_data["risk_assessment"]["key_risk_factors"],
                    "Mitigation Strategy": roadmap_data["risk_assessment"][
                        "mitigation_strategies"
                    ][: len(roadmap_data["risk_assessment"]["key_risk_factors"])],
                }
                pd.DataFrame(
                    dict([(k, pd.Series(v)) for k, v in risk_data.items()])
                ).to_excel(writer, sheet_name="Risk Assessment", index=False)

        except Exception as e:
            warnings.warn(f"Excel export failed: {e}. Falling back to JSON.")
            import json

            json_path = output_path.with_suffix(".json")
            with open(json_path, "w") as f:
                json.dump(roadmap_data, f, indent=2, default=str)

    def plot_implementation_timeline(
        self,
        roadmap_data: dict[str, Any],
        save_path: str | None = None,
    ) -> None:
        """Plot implementation timeline visualization.

        Args:
            roadmap_data: Roadmap data from generate_roadmap()
            save_path: Path to save plot (optional)
        """
        if not PLOTTING_AVAILABLE:
            warnings.warn("Plotting not available. Install matplotlib and seaborn.")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Timeline chart
        periods = []
        policies_per_period = []
        costs_per_period = []
        feasibility_per_period = []

        for period in roadmap_data["implementation_sequence"]:
            periods.append(f"Period {period['period']}")
            policies_per_period.append(len(period.get("policies", [])))
            costs_per_period.append(period.get("estimated_cost", 0))
            feasibility_per_period.append(period.get("feasibility_score", 0.5))

        # Plot 1: Policies and costs per period
        ax1_twin = ax1.twinx()

        _bars = ax1.bar(
            periods,
            policies_per_period,
            alpha=0.7,
            color="skyblue",
            label="Number of Policies",
        )
        _line = ax1_twin.plot(
            periods,
            costs_per_period,
            color="red",
            marker="o",
            linewidth=2,
            label="Cost",
        )

        ax1.set_ylabel("Number of Policies")
        ax1_twin.set_ylabel("Cost ($)")
        ax1.set_title(f"Implementation Timeline - {roadmap_data['country']}")

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        # Plot 2: Feasibility scores
        colors = [
            "green" if f >= 0.7 else "orange" if f >= 0.4 else "red"
            for f in feasibility_per_period
        ]
        _bars2 = ax2.bar(periods, feasibility_per_period, color=colors, alpha=0.7)

        ax2.set_ylabel("Feasibility Score")
        ax2.set_ylim(0, 1)
        ax2.set_title("Political Feasibility by Period")
        ax2.axhline(y=0.5, color="black", linestyle="--", alpha=0.5, label="Neutral")
        ax2.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()
