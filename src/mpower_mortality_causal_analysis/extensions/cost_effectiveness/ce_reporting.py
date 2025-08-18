"""Reporting and Visualization for Cost-Effectiveness Analysis.

This module provides standardized reporting formats and visualizations
for cost-effectiveness analysis results.
"""

import json

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


class CEReporting:
    """Generate standardized reports for cost-effectiveness analysis.

    This class provides methods for creating reports, visualizations,
    and standardized outputs for policy decision-making.
    """

    def __init__(self):
        """Initialize the reporting module."""
        self.report_data = {}
        self.figures = []

    def generate_report(
        self,
        results: dict[str, Any],
        output_path: str | Path,
        format: str = "comprehensive",
    ) -> None:
        """Generate cost-effectiveness analysis report.

        Args:
            results: Complete analysis results dictionary
            output_path: Path for report output
            format: Report format ('summary', 'detailed', 'comprehensive')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "summary":
            self._generate_summary_report(results, output_path)
        elif format == "detailed":
            self._generate_detailed_report(results, output_path)
        else:  # comprehensive
            self._generate_comprehensive_report(results, output_path)

    def _generate_summary_report(
        self, results: dict[str, Any], output_path: Path
    ) -> None:
        """Generate executive summary report."""
        summary = {
            "country": results.get("country", "Not specified"),
            "policies_evaluated": results.get("policies", []),
            "key_findings": {},
        }

        # Extract key findings
        if "health_outcomes" in results:
            health = results["health_outcomes"]
            if "combined" in health:
                summary["key_findings"]["total_qalys_gained"] = health["combined"][
                    "qalys"
                ]
                summary["key_findings"]["total_dalys_averted"] = health["combined"][
                    "dalys"
                ]

        if "costs" in results:
            costs = results["costs"]
            if "combined" in costs:
                summary["key_findings"]["net_cost"] = costs["combined"]["net_cost"]
                summary["key_findings"]["cost_effective"] = (
                    costs["combined"]["net_cost"] < 0
                )

        if "icers" in results:
            icers = results["icers"]
            summary["key_findings"]["efficient_strategies"] = icers.get(
                "efficient_frontier", []
            )
            if "psa_summary" in icers:
                summary["key_findings"]["probability_cost_effective"] = icers[
                    "psa_summary"
                ].get("prob_cost_effective", 0)

        # Write summary
        if output_path.suffix == ".json":
            with open(output_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
        else:
            # Write as markdown
            with open(output_path.with_suffix(".md"), "w") as f:
                f.write("# Cost-Effectiveness Analysis Summary\n\n")
                f.write(f"**Country**: {summary['country']}\n\n")
                f.write(
                    f"**Policies Evaluated**: {', '.join(summary['policies_evaluated'])}\n\n"
                )
                f.write("## Key Findings\n\n")
                for key, value in summary["key_findings"].items():
                    f.write(f"- **{key.replace('_', ' ').title()}**: {value}\n")

    def _generate_detailed_report(
        self, results: dict[str, Any], output_path: Path
    ) -> None:
        """Generate detailed analysis report."""
        # Create Excel workbook
        with pd.ExcelWriter(
            output_path.with_suffix(".xlsx"), engine="openpyxl"
        ) as writer:
            # Summary sheet
            summary_data = self._create_summary_table(results)
            summary_data.to_excel(writer, sheet_name="Summary", index=False)

            # Health outcomes
            if "health_outcomes" in results:
                health_df = self._create_health_outcomes_table(
                    results["health_outcomes"]
                )
                health_df.to_excel(writer, sheet_name="Health Outcomes", index=False)

            # Costs
            if "costs" in results:
                costs_df = self._create_costs_table(results["costs"])
                costs_df.to_excel(writer, sheet_name="Costs", index=False)

            # ICERs
            if "icers" in results and "icer_table" in results["icers"]:
                icer_df = pd.DataFrame(results["icers"]["icer_table"])
                icer_df.to_excel(writer, sheet_name="ICERs", index=False)

            # Optimization
            if "optimization" in results:
                opt_df = self._create_optimization_table(results["optimization"])
                opt_df.to_excel(writer, sheet_name="Budget Optimization", index=False)

            # Sensitivity
            if "sensitivity" in results:
                sens_df = self._create_sensitivity_table(results["sensitivity"])
                sens_df.to_excel(writer, sheet_name="Sensitivity Analysis", index=False)

    def _generate_comprehensive_report(
        self, results: dict[str, Any], output_path: Path
    ) -> None:
        """Generate comprehensive report with visualizations."""
        # Generate detailed report
        self._generate_detailed_report(results, output_path)

        # Generate visualizations if available
        if PLOTTING_AVAILABLE:
            self._create_visualizations(results, output_path.parent)

        # Generate narrative report
        self._generate_narrative_report(results, output_path)

    def _create_summary_table(self, results: dict[str, Any]) -> pd.DataFrame:
        """Create summary table of key metrics."""
        summary_rows = []

        # Add health outcomes
        if "health_outcomes" in results:
            for strategy, outcomes in results["health_outcomes"].items():
                row = {
                    "Strategy": strategy,
                    "QALYs Gained": outcomes.get("qalys", 0),
                    "DALYs Averted": outcomes.get("dalys", 0),
                }

                # Add costs if available
                if "costs" in results and strategy in results["costs"]:
                    costs = results["costs"][strategy]
                    row.update(
                        {
                            "Implementation Cost": costs.get("implementation_cost", 0),
                            "Healthcare Savings": costs.get("healthcare_savings", 0),
                            "Net Cost": costs.get("net_cost", 0),
                        }
                    )

                summary_rows.append(row)

        return pd.DataFrame(summary_rows)

    def _create_health_outcomes_table(self, health_outcomes: dict) -> pd.DataFrame:
        """Create detailed health outcomes table."""
        rows = []
        for strategy, outcomes in health_outcomes.items():
            rows.append(
                {
                    "Strategy": strategy,
                    "QALYs": outcomes.get("qalys", 0),
                    "DALYs": outcomes.get("dalys", 0),
                    "Mortality Reduction": outcomes.get("mortality_reduction", 0),
                }
            )
        return pd.DataFrame(rows)

    def _create_costs_table(self, costs: dict) -> pd.DataFrame:
        """Create detailed costs table."""
        rows = []
        for strategy, cost_data in costs.items():
            rows.append(
                {
                    "Strategy": strategy,
                    "Implementation Cost": cost_data.get("implementation_cost", 0),
                    "Healthcare Savings": cost_data.get("healthcare_savings", 0),
                    "Productivity Gains": cost_data.get("productivity_gains", 0),
                    "Net Cost": cost_data.get("net_cost", 0),
                }
            )
        return pd.DataFrame(rows)

    def _create_optimization_table(self, optimization: dict) -> pd.DataFrame:
        """Create budget optimization results table."""
        rows = []

        if "linear_optimization" in optimization:
            linear = optimization["linear_optimization"]
            if linear.get("success"):
                for policy, allocation in linear["allocation"].items():
                    rows.append(
                        {
                            "Method": "Linear",
                            "Policy": policy,
                            "Allocation": allocation,
                            "Total Effect": linear.get("total_effect", 0),
                        }
                    )

        if "nonlinear_optimization" in optimization:
            nonlinear = optimization["nonlinear_optimization"]
            if nonlinear.get("success"):
                for policy, allocation in nonlinear["allocation"].items():
                    rows.append(
                        {
                            "Method": "Nonlinear",
                            "Policy": policy,
                            "Allocation": allocation,
                            "Total Effect": nonlinear.get("total_effect", 0),
                        }
                    )

        return pd.DataFrame(rows)

    def _create_sensitivity_table(self, sensitivity: dict) -> pd.DataFrame:
        """Create sensitivity analysis results table."""
        rows = []

        if "health_outcomes" in sensitivity:
            for item in sensitivity["health_outcomes"]:
                row = {"Analysis": "Health Outcomes"}
                row.update(item)
                rows.append(row)

        if "costs" in sensitivity:
            for item in sensitivity["costs"]:
                row = {"Analysis": "Costs"}
                row.update(item)
                rows.append(row)

        return pd.DataFrame(rows)

    def _create_visualizations(self, results: dict[str, Any], output_dir: Path) -> None:
        """Create visualization plots."""
        if not PLOTTING_AVAILABLE:
            return

        output_dir = output_dir / "figures"
        output_dir.mkdir(exist_ok=True)

        # Set style
        sns.set_style("whitegrid")

        # 1. Cost-effectiveness plane
        self._plot_ce_plane(results, output_dir / "ce_plane.png")

        # 2. CEAC curve
        self._plot_ceac(results, output_dir / "ceac.png")

        # 3. Tornado diagram
        self._plot_tornado(results, output_dir / "tornado.png")

        # 4. Budget optimization
        self._plot_budget_optimization(results, output_dir / "budget_optimization.png")

    def _plot_ce_plane(self, results: dict[str, Any], output_path: Path) -> None:
        """Plot cost-effectiveness plane."""
        if "icers" not in results or "icer_table" not in results["icers"]:
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        # Extract data
        icer_df = pd.DataFrame(results["icers"]["icer_table"])

        if (
            not icer_df.empty
            and "incremental_cost" in icer_df
            and "incremental_effect" in icer_df
        ):
            # Plot points
            ax.scatter(
                icer_df["incremental_effect"],
                icer_df["incremental_cost"],
                s=100,
                alpha=0.6,
            )

            # Add labels
            for idx, row in icer_df.iterrows():
                ax.annotate(
                    row.get("intervention", ""),
                    (row["incremental_effect"], row["incremental_cost"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )

            # Add WTP threshold line
            wtp = results.get("wtp_threshold", 50000)
            max_effect = icer_df["incremental_effect"].max()
            ax.plot(
                [0, max_effect],
                [0, max_effect * wtp],
                "r--",
                alpha=0.5,
                label=f"WTP = ${wtp:,}",
            )

            ax.set_xlabel("Incremental Effects (QALYs)")
            ax.set_ylabel("Incremental Costs ($)")
            ax.set_title("Cost-Effectiveness Plane")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
            ax.axvline(x=0, color="k", linestyle="-", linewidth=0.5)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_ceac(self, results: dict[str, Any], output_path: Path) -> None:
        """Plot cost-effectiveness acceptability curve."""
        if "icers" not in results or "ceac_data" not in results["icers"]:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        ceac_df = pd.DataFrame(results["icers"]["ceac_data"])

        if not ceac_df.empty:
            # Plot for each strategy
            for strategy in ceac_df["strategy"].unique():
                strategy_data = ceac_df[ceac_df["strategy"] == strategy]
                ax.plot(
                    strategy_data["threshold"],
                    strategy_data["probability_cost_effective"],
                    label=strategy,
                    marker="o",
                    markersize=4,
                )

            ax.set_xlabel("Willingness to Pay Threshold ($/QALY)")
            ax.set_ylabel("Probability Cost-Effective")
            ax.set_title("Cost-Effectiveness Acceptability Curves")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_tornado(self, results: dict[str, Any], output_path: Path) -> None:
        """Plot tornado diagram for sensitivity analysis."""
        if "sensitivity" not in results or "tornado" not in results["sensitivity"]:
            return

        tornado_df = pd.DataFrame(results["sensitivity"]["tornado"])

        if tornado_df.empty:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        # Sort by impact
        tornado_df = tornado_df.sort_values("icer_range", ascending=True)

        # Plot bars
        y_pos = np.arange(len(tornado_df))

        for i, row in tornado_df.iterrows():
            # Low value bar
            low_width = (
                abs(row["low_icer"] - row["base_icer"])
                if not np.isinf(row["low_icer"])
                else 0
            )
            ax.barh(i, -low_width, left=0, color="blue", alpha=0.6)

            # High value bar
            high_width = (
                abs(row["high_icer"] - row["base_icer"])
                if not np.isinf(row["high_icer"])
                else 0
            )
            ax.barh(i, high_width, left=0, color="red", alpha=0.6)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(tornado_df["parameter"])
        ax.set_xlabel("Change in ICER from Base Case")
        ax.set_title("Tornado Diagram - Sensitivity Analysis")
        ax.axvline(x=0, color="k", linestyle="-", linewidth=1)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_budget_optimization(
        self, results: dict[str, Any], output_path: Path
    ) -> None:
        """Plot budget optimization results."""
        if "optimization" not in results:
            return

        opt = results["optimization"]

        if "budget_sensitivity" not in opt:
            return

        budget_df = pd.DataFrame(opt["budget_sensitivity"])

        if budget_df.empty:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Total effect vs budget
        ax1.plot(budget_df["budget"], budget_df["total_effect"], "b-", linewidth=2)
        ax1.set_xlabel("Budget ($)")
        ax1.set_ylabel("Total Health Effect (QALYs)")
        ax1.set_title("Health Effects vs Budget")
        ax1.grid(True, alpha=0.3)

        # Plot 2: Policy allocations vs budget
        policy_cols = [col for col in budget_df.columns if col.endswith("_allocation")]
        for col in policy_cols:
            policy = col.replace("_allocation", "")
            ax2.plot(
                budget_df["budget"],
                budget_df[col],
                label=policy,
                marker="o",
                markersize=3,
            )

        ax2.set_xlabel("Budget ($)")
        ax2.set_ylabel("Allocation Fraction")
        ax2.set_title("Optimal Policy Mix vs Budget")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _generate_narrative_report(
        self, results: dict[str, Any], output_path: Path
    ) -> None:
        """Generate narrative text report."""
        report_text = []

        report_text.append("# MPOWER Cost-Effectiveness Analysis Report\n")
        report_text.append(
            f"**Analysis Date**: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n"
        )
        report_text.append(
            f"**Country**: {results.get('country', 'Not specified')}\n\n"
        )

        # Executive Summary
        report_text.append("## Executive Summary\n")
        report_text.append(self._generate_executive_summary(results))
        report_text.append("\n")

        # Health Outcomes
        report_text.append("## Health Outcomes\n")
        report_text.append(self._generate_health_outcomes_narrative(results))
        report_text.append("\n")

        # Economic Analysis
        report_text.append("## Economic Analysis\n")
        report_text.append(self._generate_economic_narrative(results))
        report_text.append("\n")

        # Recommendations
        report_text.append("## Policy Recommendations\n")
        report_text.append(self._generate_recommendations(results))
        report_text.append("\n")

        # Write report
        with open(output_path.with_suffix(".md"), "w") as f:
            f.writelines(report_text)

    def _generate_executive_summary(self, results: dict[str, Any]) -> str:
        """Generate executive summary text."""
        summary = []

        if "health_outcomes" in results and "combined" in results["health_outcomes"]:
            qalys = results["health_outcomes"]["combined"].get("qalys", 0)
            summary.append(
                f"The implementation of MPOWER policies is projected to generate "
                f"{qalys:.0f} quality-adjusted life years (QALYs). "
            )

        if "costs" in results and "combined" in results["costs"]:
            net_cost = results["costs"]["combined"].get("net_cost", 0)
            if net_cost < 0:
                summary.append(
                    f"The intervention is cost-saving, with net savings of "
                    f"${abs(net_cost):,.0f}. "
                )
            else:
                summary.append(f"The net cost of implementation is ${net_cost:,.0f}. ")

        if "icers" in results and "efficient_frontier" in results["icers"]:
            frontier = results["icers"]["efficient_frontier"]
            if frontier:
                summary.append(
                    f"The most cost-effective strategies are: {', '.join(frontier)}. "
                )

        return "".join(summary)

    def _generate_health_outcomes_narrative(self, results: dict[str, Any]) -> str:
        """Generate health outcomes narrative."""
        narrative = []

        if "health_outcomes" not in results:
            return "Health outcomes data not available."

        health = results["health_outcomes"]

        # Find best performing policy
        best_policy = None
        best_qalys = 0
        for policy, outcomes in health.items():
            if policy != "combined" and outcomes.get("qalys", 0) > best_qalys:
                best_policy = policy
                best_qalys = outcomes["qalys"]

        if best_policy:
            narrative.append(
                f"Among individual policies, {best_policy} generates the highest "
                f"health benefit with {best_qalys:.0f} QALYs gained. "
            )

        if "combined" in health:
            combined_qalys = health["combined"].get("qalys", 0)
            combined_dalys = health["combined"].get("dalys", 0)
            narrative.append(
                f"The combined implementation of all policies yields {combined_qalys:.0f} "
                f"QALYs gained and {combined_dalys:.0f} DALYs averted."
            )

        return "".join(narrative)

    def _generate_economic_narrative(self, results: dict[str, Any]) -> str:
        """Generate economic analysis narrative."""
        narrative = []

        if "costs" not in results:
            return "Economic analysis data not available."

        costs = results["costs"]

        # Calculate ROI
        if "combined" in costs:
            impl_cost = costs["combined"].get("implementation_cost", 1)
            savings = costs["combined"].get("healthcare_savings", 0) + costs[
                "combined"
            ].get("productivity_gains", 0)
            roi = (savings / impl_cost - 1) * 100 if impl_cost > 0 else 0

            narrative.append(f"The total implementation cost is ${impl_cost:,.0f}, ")
            narrative.append(
                f"with healthcare savings of ${costs['combined'].get('healthcare_savings', 0):,.0f} "
            )
            narrative.append(
                f"and productivity gains of ${costs['combined'].get('productivity_gains', 0):,.0f}. "
            )

            if roi > 0:
                narrative.append(
                    f"This represents a return on investment of {roi:.0f}%. "
                )

        return "".join(narrative)

    def _generate_recommendations(self, results: dict[str, Any]) -> str:
        """Generate policy recommendations."""
        recommendations = []

        # Check if cost-effective
        if "costs" in results and "combined" in results["costs"]:
            if results["costs"]["combined"].get("net_cost", 0) < 0:
                recommendations.append(
                    "1. **Immediate Implementation Recommended**: "
                    "The analysis shows that MPOWER policies are cost-saving.\n"
                )

        # Check optimization results
        if "optimization" in results:
            if "incremental_allocation" in results["optimization"]:
                steps = results["optimization"]["incremental_allocation"]
                if steps and len(steps) > 0:
                    first_policy = steps[0].get("last_added")
                    if first_policy:
                        recommendations.append(
                            f"2. **Priority Implementation**: "
                            f"Begin with policy {first_policy} for maximum cost-effectiveness.\n"
                        )

        # Check efficient frontier
        if "icers" in results and "efficient_frontier" in results["icers"]:
            frontier = results["icers"]["efficient_frontier"]
            if frontier:
                recommendations.append(
                    f"3. **Efficient Strategies**: Focus on {', '.join(frontier[:3])} "
                    f"for optimal resource allocation.\n"
                )

        if not recommendations:
            recommendations.append(
                "Further analysis recommended with country-specific data."
            )

        return "".join(recommendations)
