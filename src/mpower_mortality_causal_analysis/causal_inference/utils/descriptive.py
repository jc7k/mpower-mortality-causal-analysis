"""Descriptive Statistics and Visualization for MPOWER Causal Analysis.

This module provides comprehensive descriptive statistics and visualization functions
for examining the MPOWER data before conducting causal inference analysis.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pandas import DataFrame


class MPOWERDescriptives:
    """Descriptive statistics and visualizations for MPOWER data.

    Provides comprehensive exploratory data analysis functions specifically
    designed for MPOWER tobacco control policy and mortality data.

    Parameters:
        data (DataFrame): MPOWER panel data
        country_col (str): Column name for country identifier
        year_col (str): Column name for year
        cohort_col (str): Column name for treatment cohort
        outcome_cols (List[str]): List of outcome variable columns

    Example:
        >>> descriptives = MPOWERDescriptives(
        ...     data=analysis_data,
        ...     outcome_cols=['mort_lung_cancer_asr', 'mort_cvd_asr']
        ... )
        >>> summary = descriptives.generate_summary_statistics()
        >>> descriptives.plot_treatment_adoption_timeline()
        >>> descriptives.plot_outcome_trends_by_cohort()
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
        """Initialize MPOWER descriptives analyzer."""
        self.data = data.copy()
        self.country_col = country_col
        self.year_col = year_col
        self.cohort_col = cohort_col
        self.never_treated_value = never_treated_value

        # Set default column lists if not provided
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

        # Filter columns to only those that exist in data
        self.outcome_cols = [col for col in self.outcome_cols if col in data.columns]
        self.mpower_cols = [col for col in self.mpower_cols if col in data.columns]
        self.control_cols = [col for col in self.control_cols if col in data.columns]

        # Validate required columns
        required_cols = [country_col, year_col, cohort_col]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Set up plotting style
        plt.style.use("default")
        sns.set_palette("husl")

    def generate_summary_statistics(self) -> dict[str, Any]:
        """Generate comprehensive summary statistics for the dataset.

        Returns:
            Dict containing detailed summary statistics
        """
        summary = {}

        # Basic dataset structure
        summary["dataset_structure"] = {
            "n_countries": self.data[self.country_col].nunique(),
            "n_years": self.data[self.year_col].nunique(),
            "n_observations": len(self.data),
            "year_range": [
                self.data[self.year_col].min(),
                self.data[self.year_col].max(),
            ],
            "countries_list": sorted(self.data[self.country_col].unique().tolist()),
        }

        # Treatment cohort analysis
        cohort_summary = self._analyze_treatment_cohorts()
        summary["treatment_cohorts"] = cohort_summary

        # Outcome variable statistics
        outcome_summary = self._analyze_outcomes()
        summary["outcomes"] = outcome_summary

        # MPOWER policy statistics
        if self.mpower_cols:
            mpower_summary = self._analyze_mpower_policies()
            summary["mpower_policies"] = mpower_summary

        # Control variable statistics
        if self.control_cols:
            control_summary = self._analyze_controls()
            summary["control_variables"] = control_summary

        # Missing data analysis
        missing_summary = self._analyze_missing_data()
        summary["missing_data"] = missing_summary

        # Panel balance analysis
        balance_summary = self._analyze_panel_balance()
        summary["panel_balance"] = balance_summary

        return summary

    def _analyze_treatment_cohorts(self) -> dict[str, Any]:
        """Analyze treatment cohort structure."""
        # Basic cohort distribution
        cohort_counts = self.data.groupby(self.cohort_col)[self.country_col].nunique()

        # Never-treated and treated countries
        never_treated = (self.data[self.cohort_col] == self.never_treated_value).sum()
        ever_treated = (self.data[self.cohort_col] != self.never_treated_value).sum()

        # Treatment years
        treatment_years = sorted(
            [
                year
                for year in self.data[self.cohort_col].unique()
                if year != self.never_treated_value
            ]
        )

        return {
            "cohort_distribution": cohort_counts.to_dict(),
            "n_never_treated_obs": never_treated,
            "n_treated_obs": ever_treated,
            "n_never_treated_countries": cohort_counts.get(self.never_treated_value, 0),
            "n_treated_countries": len(
                [c for c in cohort_counts.index if c != self.never_treated_value]
            ),
            "treatment_years": treatment_years,
            "first_treatment_year": min(treatment_years) if treatment_years else None,
            "last_treatment_year": max(treatment_years) if treatment_years else None,
        }

    def _analyze_outcomes(self) -> dict[str, Any]:
        """Analyze outcome variables."""
        outcome_analysis = {}

        for outcome in self.outcome_cols:
            if outcome in self.data.columns:
                outcome_data = self.data[outcome].dropna()

                outcome_analysis[outcome] = {
                    "mean": outcome_data.mean(),
                    "std": outcome_data.std(),
                    "min": outcome_data.min(),
                    "max": outcome_data.max(),
                    "median": outcome_data.median(),
                    "q25": outcome_data.quantile(0.25),
                    "q75": outcome_data.quantile(0.75),
                    "n_obs": len(outcome_data),
                    "missing_pct": (len(self.data) - len(outcome_data))
                    / len(self.data)
                    * 100,
                }

                # Treatment vs control comparison
                treated_mask = self.data[self.cohort_col] != self.never_treated_value
                control_mask = self.data[self.cohort_col] == self.never_treated_value

                treated_outcome = self.data.loc[treated_mask, outcome].dropna()
                control_outcome = self.data.loc[control_mask, outcome].dropna()

                outcome_analysis[outcome]["by_treatment_status"] = {
                    "treated_mean": treated_outcome.mean()
                    if len(treated_outcome) > 0
                    else None,
                    "control_mean": control_outcome.mean()
                    if len(control_outcome) > 0
                    else None,
                    "difference": (treated_outcome.mean() - control_outcome.mean())
                    if len(treated_outcome) > 0 and len(control_outcome) > 0
                    else None,
                }

        return outcome_analysis

    def _analyze_mpower_policies(self) -> dict[str, Any]:
        """Analyze MPOWER policy variables."""
        mpower_analysis = {}

        for mpower_var in self.mpower_cols:
            if mpower_var in self.data.columns:
                mpower_data = self.data[mpower_var].dropna()

                mpower_analysis[mpower_var] = {
                    "mean": mpower_data.mean(),
                    "std": mpower_data.std(),
                    "min": mpower_data.min(),
                    "max": mpower_data.max(),
                    "median": mpower_data.median(),
                    "n_obs": len(mpower_data),
                    "missing_pct": (len(self.data) - len(mpower_data))
                    / len(self.data)
                    * 100,
                }

                # Time trend analysis
                yearly_means = self.data.groupby(self.year_col)[mpower_var].mean()
                mpower_analysis[mpower_var]["time_trend"] = {
                    "correlation_with_year": yearly_means.corr(yearly_means.index),
                    "yearly_means": yearly_means.to_dict(),
                }

        return mpower_analysis

    def _analyze_controls(self) -> dict[str, Any]:
        """Analyze control variables."""
        control_analysis = {}

        for control_var in self.control_cols:
            if control_var in self.data.columns:
                control_data = self.data[control_var].dropna()

                control_analysis[control_var] = {
                    "mean": control_data.mean(),
                    "std": control_data.std(),
                    "min": control_data.min(),
                    "max": control_data.max(),
                    "median": control_data.median(),
                    "n_obs": len(control_data),
                    "missing_pct": (len(self.data) - len(control_data))
                    / len(self.data)
                    * 100,
                }

        return control_analysis

    def _analyze_missing_data(self) -> dict[str, Any]:
        """Analyze missing data patterns."""
        missing_analysis = {}

        # Overall missing data
        all_vars = self.outcome_cols + self.mpower_cols + self.control_cols
        existing_vars = [var for var in all_vars if var in self.data.columns]

        missing_counts = self.data[existing_vars].isnull().sum()
        missing_pcts = (missing_counts / len(self.data)) * 100

        missing_analysis["by_variable"] = {
            var: {"count": missing_counts[var], "percentage": missing_pcts[var]}
            for var in existing_vars
        }

        # Missing data by country
        country_missing = self.data.groupby(self.country_col)[existing_vars].apply(
            lambda x: x.isnull().sum().sum()
        )

        missing_analysis["by_country"] = {
            "worst_countries": country_missing.nlargest(10).to_dict(),
            "best_countries": country_missing.nsmallest(10).to_dict(),
        }

        return missing_analysis

    def _analyze_panel_balance(self) -> dict[str, Any]:
        """Analyze panel data balance."""
        balance_analysis = {}

        # Observations per country
        obs_per_country = self.data.groupby(self.country_col).size()

        balance_analysis = {
            "is_balanced": obs_per_country.nunique() == 1,
            "min_obs_per_country": obs_per_country.min(),
            "max_obs_per_country": obs_per_country.max(),
            "mean_obs_per_country": obs_per_country.mean(),
            "countries_with_full_data": (
                obs_per_country == obs_per_country.max()
            ).sum(),
        }

        # Year coverage by country
        year_coverage = self.data.groupby(self.country_col)[self.year_col].apply(
            lambda x: x.max() - x.min() + 1
        )

        balance_analysis["year_coverage"] = {
            "min_years": year_coverage.min(),
            "max_years": year_coverage.max(),
            "mean_years": year_coverage.mean(),
        }

        return balance_analysis

    def plot_treatment_adoption_timeline(
        self, figsize: tuple[int, int] = (12, 8), save_path: str | None = None
    ) -> plt.Figure:
        """Plot timeline of MPOWER treatment adoption across countries.

        Args:
            figsize: Figure size tuple
            save_path: Optional path to save the figure

        Returns:
            matplotlib Figure object
        """
        # Get treatment adoption data
        treatment_data = (
            self.data[self.data[self.cohort_col] != self.never_treated_value]
            .groupby([self.cohort_col, self.country_col])
            .size()
            .reset_index()
        )
        treatment_data = treatment_data.drop(columns=[0])  # Remove count column

        # Count countries by treatment year
        adoption_counts = treatment_data.groupby(self.cohort_col).size()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

        # Plot 1: Number of countries adopting treatment each year
        ax1.bar(
            adoption_counts.index, adoption_counts.values, alpha=0.7, color="steelblue"
        )
        ax1.set_xlabel("Treatment Adoption Year")
        ax1.set_ylabel("Number of Countries")
        ax1.set_title("MPOWER Treatment Adoption Timeline")
        ax1.grid(True, alpha=0.3)

        # Plot 2: Cumulative adoption
        cumulative_adoption = adoption_counts.cumsum()
        ax2.plot(
            cumulative_adoption.index,
            cumulative_adoption.values,
            "o-",
            color="darkgreen",
            linewidth=2,
            markersize=6,
        )
        ax2.set_xlabel("Year")
        ax2.set_ylabel("Cumulative Number of Treated Countries")
        ax2.set_title("Cumulative MPOWER Treatment Adoption")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_outcome_trends_by_cohort(
        self,
        outcomes: list[str] | None = None,
        figsize: tuple[int, int] = (15, 10),
        save_path: str | None = None,
    ) -> plt.Figure:
        """Plot outcome variable trends by treatment cohort.

        Args:
            outcomes: List of outcome variables to plot (defaults to all)
            figsize: Figure size tuple
            save_path: Optional path to save the figure

        Returns:
            matplotlib Figure object
        """
        if outcomes is None:
            outcomes = self.outcome_cols[:4]  # Limit to first 4 outcomes

        # Filter to existing outcomes
        outcomes = [outcome for outcome in outcomes if outcome in self.data.columns]

        n_outcomes = len(outcomes)
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten() if n_outcomes > 1 else [axes]

        # Create cohort labels
        cohort_labels = {}
        for cohort in sorted(self.data[self.cohort_col].unique()):
            if cohort == self.never_treated_value:
                cohort_labels[cohort] = "Never Treated"
            else:
                cohort_labels[cohort] = f"Treated {int(cohort)}"

        for i, outcome in enumerate(outcomes):
            ax = axes[i] if i < len(axes) else axes[-1]

            # Plot trends by cohort
            for cohort in sorted(self.data[self.cohort_col].unique()):
                cohort_data = self.data[self.data[self.cohort_col] == cohort]
                yearly_means = cohort_data.groupby(self.year_col)[outcome].mean()

                # Different styling for treated vs control
                if cohort == self.never_treated_value:
                    ax.plot(
                        yearly_means.index,
                        yearly_means.values,
                        "k--",
                        linewidth=2,
                        alpha=0.8,
                        label=cohort_labels[cohort],
                    )
                else:
                    ax.plot(
                        yearly_means.index,
                        yearly_means.values,
                        "-",
                        linewidth=1.5,
                        alpha=0.7,
                        label=cohort_labels[cohort],
                    )

                    # Add vertical line at treatment year
                    ax.axvline(x=cohort, color="red", linestyle=":", alpha=0.5)

            ax.set_xlabel("Year")
            ax.set_ylabel(outcome.replace("_", " ").title())
            ax.set_title(f"{outcome.replace('_', ' ').title()} by Treatment Cohort")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(n_outcomes, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_mpower_score_distributions(
        self, figsize: tuple[int, int] = (15, 10), save_path: str | None = None
    ) -> plt.Figure:
        """Plot distributions of MPOWER policy scores.

        Args:
            figsize: Figure size tuple
            save_path: Optional path to save the figure

        Returns:
            matplotlib Figure object
        """
        # Filter to existing MPOWER variables
        available_mpower = [col for col in self.mpower_cols if col in self.data.columns]

        if not available_mpower:
            raise ValueError("No MPOWER columns found in data")

        n_vars = len(available_mpower)
        n_cols = 3
        n_rows = (n_vars + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()

        for i, mpower_var in enumerate(available_mpower):
            ax = axes[i]

            # Create histogram
            data_clean = self.data[mpower_var].dropna()
            ax.hist(data_clean, bins=20, alpha=0.7, color="skyblue", edgecolor="black")

            # Add statistics
            mean_val = data_clean.mean()
            median_val = data_clean.median()
            ax.axvline(
                mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.2f}"
            )
            ax.axvline(
                median_val,
                color="orange",
                linestyle="--",
                label=f"Median: {median_val:.2f}",
            )

            ax.set_xlabel(mpower_var.replace("_", " ").title())
            ax.set_ylabel("Frequency")
            ax.set_title(f"Distribution of {mpower_var.replace('_', ' ').title()}")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(n_vars, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_correlation_heatmap(
        self,
        variables: list[str] | None = None,
        figsize: tuple[int, int] = (12, 10),
        save_path: str | None = None,
    ) -> plt.Figure:
        """Plot correlation heatmap of key variables.

        Args:
            variables: List of variables to include (defaults to outcomes + MPOWER + key controls)
            figsize: Figure size tuple
            save_path: Optional path to save the figure

        Returns:
            matplotlib Figure object
        """
        if variables is None:
            # Use key variables
            variables = (
                self.outcome_cols
                + self.mpower_cols
                + [
                    col
                    for col in self.control_cols
                    if "gdp" in col.lower() or "urban" in col.lower()
                ]
            )

        # Filter to existing variables
        variables = [var for var in variables if var in self.data.columns]

        # Calculate correlation matrix
        corr_data = self.data[variables].corr()

        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)

        mask = np.triu(np.ones_like(corr_data, dtype=bool))  # Mask upper triangle

        sns.heatmap(
            corr_data,
            mask=mask,
            annot=True,
            cmap="RdBu_r",
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.5},
            ax=ax,
        )

        ax.set_title("Correlation Matrix of Key Variables")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_treatment_balance_check(
        self,
        pre_treatment_years: list[int] | None = None,
        figsize: tuple[int, int] = (15, 8),
        save_path: str | None = None,
    ) -> plt.Figure:
        """Plot balance check: pre-treatment characteristics by treatment status.

        Args:
            pre_treatment_years: Years to use for pre-treatment period
            figsize: Figure size tuple
            save_path: Optional path to save the figure

        Returns:
            matplotlib Figure object
        """
        # Use earliest years if not specified
        if pre_treatment_years is None:
            all_years = sorted(self.data[self.year_col].unique())
            pre_treatment_years = all_years[:3]  # First 3 years

        # Filter to pre-treatment data
        pre_data = self.data[self.data[self.year_col].isin(pre_treatment_years)]

        # Get baseline characteristics by treatment status
        variables_to_check = self.outcome_cols + self.control_cols
        variables_to_check = [
            var for var in variables_to_check if var in pre_data.columns
        ]

        # Calculate means by treatment status
        treated_means = pre_data[pre_data[self.cohort_col] != self.never_treated_value][
            variables_to_check
        ].mean()

        control_means = pre_data[pre_data[self.cohort_col] == self.never_treated_value][
            variables_to_check
        ].mean()

        differences = treated_means - control_means

        # Create bar plot
        fig, ax = plt.subplots(figsize=figsize)

        x_pos = np.arange(len(variables_to_check))

        ax.bar(x_pos - 0.2, treated_means, 0.4, label="Ever Treated", alpha=0.7)
        ax.bar(x_pos + 0.2, control_means, 0.4, label="Never Treated", alpha=0.7)

        ax.set_xlabel("Variables")
        ax.set_ylabel("Mean Values")
        ax.set_title(
            f"Pre-Treatment Balance Check (Years {min(pre_treatment_years)}-{max(pre_treatment_years)})"
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [var.replace("_", " ").title() for var in variables_to_check],
            rotation=45,
            ha="right",
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add difference annotations
        for i, diff in enumerate(differences):
            ax.annotate(
                f"Δ={diff:.2f}",
                xy=(
                    i,
                    max(treated_means[i], control_means[i]) + 0.01 * ax.get_ylim()[1],
                ),
                ha="center",
                fontsize=8,
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def export_summary_report(
        self, filepath: str, include_plots: bool = True, plot_dir: str | None = None
    ) -> None:
        """Export comprehensive summary report to file.

        Args:
            filepath: Path for the summary report file
            include_plots: Whether to generate and reference plots
            plot_dir: Directory to save plots (if None, uses same dir as report)
        """
        import json

        from pathlib import Path

        # Generate summary statistics
        summary = self.generate_summary_statistics()

        # Set up paths
        report_path = Path(filepath)
        plot_dir = report_path.parent / "plots" if plot_dir is None else Path(plot_dir)

        if include_plots:
            plot_dir.mkdir(exist_ok=True)

        # Create report content
        report_content = {
            "MPOWER Causal Analysis - Descriptive Statistics Report": {
                "Generated": pd.Timestamp.now().isoformat(),
                "Summary_Statistics": summary,
            }
        }

        # Generate plots if requested
        if include_plots:
            plot_files = {}

            try:
                fig1 = self.plot_treatment_adoption_timeline()
                plot1_path = plot_dir / "treatment_adoption_timeline.png"
                fig1.savefig(plot1_path, dpi=300, bbox_inches="tight")
                plt.close(fig1)
                plot_files["treatment_adoption_timeline"] = str(plot1_path)
            except Exception as e:
                plot_files["treatment_adoption_timeline"] = f"Error: {e}"

            try:
                fig2 = self.plot_outcome_trends_by_cohort()
                plot2_path = plot_dir / "outcome_trends_by_cohort.png"
                fig2.savefig(plot2_path, dpi=300, bbox_inches="tight")
                plt.close(fig2)
                plot_files["outcome_trends_by_cohort"] = str(plot2_path)
            except Exception as e:
                plot_files["outcome_trends_by_cohort"] = f"Error: {e}"

            try:
                fig3 = self.plot_mpower_score_distributions()
                plot3_path = plot_dir / "mpower_score_distributions.png"
                fig3.savefig(plot3_path, dpi=300, bbox_inches="tight")
                plt.close(fig3)
                plot_files["mpower_score_distributions"] = str(plot3_path)
            except Exception as e:
                plot_files["mpower_score_distributions"] = f"Error: {e}"

            try:
                fig4 = self.plot_correlation_heatmap()
                plot4_path = plot_dir / "correlation_heatmap.png"
                fig4.savefig(plot4_path, dpi=300, bbox_inches="tight")
                plt.close(fig4)
                plot_files["correlation_heatmap"] = str(plot4_path)
            except Exception as e:
                plot_files["correlation_heatmap"] = f"Error: {e}"

            try:
                fig5 = self.plot_treatment_balance_check()
                plot5_path = plot_dir / "treatment_balance_check.png"
                fig5.savefig(plot5_path, dpi=300, bbox_inches="tight")
                plt.close(fig5)
                plot_files["treatment_balance_check"] = str(plot5_path)
            except Exception as e:
                plot_files["treatment_balance_check"] = f"Error: {e}"

            report_content["Plots"] = plot_files

        # Save report
        if filepath.endswith(".json"):
            with open(filepath, "w") as f:
                json.dump(report_content, f, indent=2, default=str)
        else:
            # Create markdown report
            self._create_markdown_report(report_content, filepath)

    def _create_markdown_report(self, content: dict[str, Any], filepath: str) -> None:
        """Create a markdown version of the summary report."""
        with open(filepath, "w") as f:
            f.write("# MPOWER Causal Analysis - Descriptive Statistics Report\n\n")
            f.write(
                f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )

            summary = content["MPOWER Causal Analysis - Descriptive Statistics Report"][
                "Summary_Statistics"
            ]

            # Dataset structure
            f.write("## Dataset Structure\n\n")
            structure = summary["dataset_structure"]
            f.write(f"- **Countries:** {structure['n_countries']}\n")
            f.write(
                f"- **Years:** {structure['n_years']} ({structure['year_range'][0]}-{structure['year_range'][1]})\n"
            )
            f.write(f"- **Total Observations:** {structure['n_observations']}\n\n")

            # Treatment cohorts
            f.write("## Treatment Cohorts\n\n")
            cohorts = summary["treatment_cohorts"]
            f.write(
                f"- **Never-treated Countries:** {cohorts['n_never_treated_countries']}\n"
            )
            f.write(f"- **Ever-treated Countries:** {cohorts['n_treated_countries']}\n")
            f.write(f"- **Treatment Years:** {cohorts['treatment_years']}\n\n")

            # Outcomes
            f.write("## Outcome Variables\n\n")
            for outcome, stats in summary["outcomes"].items():
                f.write(f"### {outcome.replace('_', ' ').title()}\n")
                f.write(f"- Mean: {stats['mean']:.2f}\n")
                f.write(f"- Standard Deviation: {stats['std']:.2f}\n")
                f.write(f"- Range: {stats['min']:.2f} - {stats['max']:.2f}\n")
                f.write(f"- Missing: {stats['missing_pct']:.1f}%\n\n")

            # Add plot references if available
            if "Plots" in content:
                f.write("## Visualizations\n\n")
                for plot_name, plot_path in content["Plots"].items():
                    if not plot_path.startswith("Error"):
                        f.write(
                            f"- **{plot_name.replace('_', ' ').title()}:** `{plot_path}`\n"
                        )
                f.write("\n")
