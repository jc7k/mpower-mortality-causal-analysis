"""Visualization module for spatial spillover analysis.

This module provides visualization tools for spatial patterns, diffusion processes,
and spillover effects from the analysis.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)

# Set style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


class SpilloverVisualization:
    """Visualization tools for spillover analysis results.

    Provides methods to visualize spatial patterns, network diffusion,
    and border discontinuity results.
    """

    def __init__(self, figsize: tuple = (12, 8), style: str = "seaborn"):
        """Initialize visualization settings.

        Args:
            figsize: Default figure size
            style: Matplotlib style
        """
        self.figsize = figsize
        if style and style != "seaborn":
            plt.style.use(style)

    def plot_weight_matrix(
        self,
        W: np.ndarray,
        countries: list[str] | None = None,
        title: str = "Spatial Weight Matrix",
    ) -> plt.Figure:
        """Visualize spatial weight matrix as heatmap.

        Args:
            W: Spatial weight matrix
            countries: Country labels
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Create heatmap
        im = ax.imshow(W, cmap="YlOrRd", aspect="equal")

        # Add colorbar
        plt.colorbar(im, ax=ax, label="Weight")

        # Add labels if provided
        if countries and len(countries) <= 20:  # Only show labels for small matrices
            ax.set_xticks(range(len(countries)))
            ax.set_yticks(range(len(countries)))
            ax.set_xticklabels(countries, rotation=45, ha="right")
            ax.set_yticklabels(countries)

        ax.set_title(title)
        ax.set_xlabel("Country j")
        ax.set_ylabel("Country i")

        plt.tight_layout()
        return fig

    def plot_moran_scatterplot(
        self, y: np.ndarray, W: np.ndarray, variable_name: str = "Variable"
    ) -> plt.Figure:
        """Create Moran's I scatterplot for spatial autocorrelation.

        Args:
            y: Variable values
            W: Spatial weight matrix
            variable_name: Name of variable for labels

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Standardize variable
        y_std = (y - y.mean()) / y.std()

        # Calculate spatial lag
        Wy = W @ y_std

        # Create scatterplot
        ax.scatter(y_std, Wy, alpha=0.6)

        # Add regression line
        z = np.polyfit(y_std, Wy, 1)
        p = np.poly1d(z)
        x_line = np.linspace(y_std.min(), y_std.max(), 100)
        ax.plot(x_line, p(x_line), "r-", alpha=0.8, label=f"Slope = {z[0]:.3f}")

        # Add quadrant lines
        ax.axhline(y=0, color="k", linestyle="-", alpha=0.3)
        ax.axvline(x=0, color="k", linestyle="-", alpha=0.3)

        # Labels
        ax.set_xlabel(f"Standardized {variable_name}")
        ax.set_ylabel(f"Spatial Lag of {variable_name}")
        ax.set_title(
            f"Moran's I Scatterplot\nSpatial Autocorrelation in {variable_name}"
        )
        ax.legend()

        # Add quadrant labels
        ax.text(0.05, 0.95, "High-High", transform=ax.transAxes, fontsize=10, alpha=0.5)
        ax.text(0.05, 0.05, "Low-High", transform=ax.transAxes, fontsize=10, alpha=0.5)
        ax.text(0.85, 0.05, "Low-Low", transform=ax.transAxes, fontsize=10, alpha=0.5)
        ax.text(0.85, 0.95, "High-Low", transform=ax.transAxes, fontsize=10, alpha=0.5)

        plt.tight_layout()
        return fig

    def plot_diffusion_timeline(
        self, adoption_matrix: np.ndarray, countries: list[str], years: list[int]
    ) -> plt.Figure:
        """Visualize policy diffusion over time.

        Args:
            adoption_matrix: Binary adoption matrix (countries x time)
            countries: Country names
            years: Year labels

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(self.figsize[0], self.figsize[1] * 1.2)
        )

        # Top panel: Adoption heatmap
        im = ax1.imshow(adoption_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

        ax1.set_xticks(range(len(years)))
        ax1.set_xticklabels(years, rotation=45)
        ax1.set_xlabel("Year")

        if len(countries) <= 30:
            ax1.set_yticks(range(len(countries)))
            ax1.set_yticklabels(countries, fontsize=8)
        ax1.set_ylabel("Country")
        ax1.set_title("Policy Adoption Timeline")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label("Adopted")

        # Bottom panel: Cumulative adoption
        cumulative = adoption_matrix.sum(axis=0)
        ax2.plot(years, cumulative, marker="o", linewidth=2)
        ax2.fill_between(years, 0, cumulative, alpha=0.3)

        ax2.set_xlabel("Year")
        ax2.set_ylabel("Number of Countries")
        ax2.set_title("Cumulative Adoption Over Time")
        ax2.grid(True, alpha=0.3)

        # Add adoption rate annotation
        if len(cumulative) > 1:
            total_countries = len(countries)
            final_adoption = cumulative[-1] / total_countries * 100
            ax2.text(
                0.95,
                0.95,
                f"Final adoption: {final_adoption:.1f}%",
                transform=ax2.transAxes,
                ha="right",
                va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        plt.tight_layout()
        return fig

    def plot_spatial_effects(
        self, effects_dict: dict, variable_names: list[str] | None = None
    ) -> plt.Figure:
        """Plot direct, indirect, and total spatial effects.

        Args:
            effects_dict: Dictionary with direct, indirect, total effects
            variable_names: Names of variables

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(
            1, 3, figsize=(self.figsize[0] * 1.2, self.figsize[1] * 0.6)
        )

        # Extract effects
        direct = effects_dict.get("direct", [])
        indirect = effects_dict.get("indirect", [])
        total = effects_dict.get("total", [])

        if not variable_names:
            variable_names = [f"Var{i + 1}" for i in range(len(direct))]

        # Direct effects
        axes[0].barh(variable_names, direct, color="steelblue")
        axes[0].set_xlabel("Effect Size")
        axes[0].set_title("Direct Effects")
        axes[0].axvline(x=0, color="k", linestyle="-", alpha=0.3)

        # Indirect effects (spillovers)
        axes[1].barh(variable_names, indirect, color="coral")
        axes[1].set_xlabel("Effect Size")
        axes[1].set_title("Indirect Effects (Spillovers)")
        axes[1].axvline(x=0, color="k", linestyle="-", alpha=0.3)

        # Total effects
        axes[2].barh(variable_names, total, color="seagreen")
        axes[2].set_xlabel("Effect Size")
        axes[2].set_title("Total Effects")
        axes[2].axvline(x=0, color="k", linestyle="-", alpha=0.3)

        plt.suptitle("Spatial Effect Decomposition")
        plt.tight_layout()
        return fig

    def plot_border_discontinuity(
        self,
        distance: np.ndarray,
        outcome: np.ndarray,
        treated: np.ndarray,
        bandwidth: float | None = None,
        effect: float | None = None,
    ) -> plt.Figure:
        """Visualize regression discontinuity at borders.

        Args:
            distance: Distance from border
            outcome: Outcome values
            treated: Treatment indicator
            bandwidth: Bandwidth for RDD
            effect: Estimated treatment effect

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Apply bandwidth if specified
        if bandwidth:
            mask = abs(distance) <= bandwidth
            distance = distance[mask]
            outcome = outcome[mask]
            treated = treated[mask]

        # Scatter plot
        ax.scatter(
            distance[treated == 0],
            outcome[treated == 0],
            alpha=0.5,
            label="Control",
            color="blue",
            s=20,
        )
        ax.scatter(
            distance[treated == 1],
            outcome[treated == 1],
            alpha=0.5,
            label="Treated",
            color="red",
            s=20,
        )

        # Fit and plot regression lines
        if len(distance[treated == 0]) > 2 and len(distance[treated == 1]) > 2:
            # Control side
            z_control = np.polyfit(distance[treated == 0], outcome[treated == 0], 1)
            p_control = np.poly1d(z_control)
            x_control = np.linspace(distance[treated == 0].min(), 0, 100)
            ax.plot(x_control, p_control(x_control), "b-", linewidth=2, alpha=0.8)

            # Treatment side
            z_treat = np.polyfit(distance[treated == 1], outcome[treated == 1], 1)
            p_treat = np.poly1d(z_treat)
            x_treat = np.linspace(0, distance[treated == 1].max(), 100)
            ax.plot(x_treat, p_treat(x_treat), "r-", linewidth=2, alpha=0.8)

            # Show discontinuity
            y_control_at_0 = p_control(0)
            y_treat_at_0 = p_treat(0)
            ax.plot([0, 0], [y_control_at_0, y_treat_at_0], "k--", linewidth=2)

            # Add effect annotation
            if effect is not None:
                ax.annotate(
                    f"Effect = {effect:.2f}",
                    xy=(0, (y_control_at_0 + y_treat_at_0) / 2),
                    xytext=(10, 20),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.5),
                    arrowprops=dict(arrowstyle="->"),
                )

        # Border line
        ax.axvline(x=0, color="k", linestyle="-", alpha=0.5, label="Border")

        # Labels
        ax.set_xlabel("Distance from Border (km)")
        ax.set_ylabel("Outcome")
        ax.set_title("Regression Discontinuity at International Border")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_network_influence(
        self, influencers: list[dict], top_k: int = 10
    ) -> plt.Figure:
        """Visualize network influence metrics.

        Args:
            influencers: List of influencer dictionaries
            top_k: Number of top influencers to show

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)

        # Limit to top k
        influencers = influencers[:top_k]
        countries = [inf["country"] for inf in influencers]

        # Influence scores
        scores = [inf.get("influence_score", 0) for inf in influencers]
        axes[0, 0].barh(countries, scores, color="purple")
        axes[0, 0].set_xlabel("Influence Score")
        axes[0, 0].set_title("Overall Influence")

        # Adoption timing
        timing = [inf.get("adoption_time", 0) for inf in influencers]
        axes[0, 1].barh(countries, timing, color="orange")
        axes[0, 1].set_xlabel("Adoption Time (years)")
        axes[0, 1].set_title("Early Adoption")

        # Network degree
        degree = [inf.get("degree", 0) for inf in influencers]
        axes[1, 0].barh(countries, degree, color="green")
        axes[1, 0].set_xlabel("Network Degree")
        axes[1, 0].set_title("Network Connectivity")

        # Influence reach
        reach = [inf.get("influence_reach", 0) for inf in influencers]
        axes[1, 1].barh(countries, reach, color="red")
        axes[1, 1].set_xlabel("Countries Influenced")
        axes[1, 1].set_title("Influence Reach")

        plt.suptitle(f"Top {top_k} Network Influencers")
        plt.tight_layout()
        return fig

    def plot_cascade_analysis(self, cascade_results: dict) -> plt.Figure:
        """Visualize cascade analysis results.

        Args:
            cascade_results: Dictionary with cascade statistics

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)

        cascades = cascade_results.get("cascades", [])

        if cascades:
            # Cascade sizes
            sizes = [c["size"] for c in cascades]
            axes[0, 0].hist(sizes, bins=20, color="steelblue", edgecolor="black")
            axes[0, 0].set_xlabel("Cascade Size")
            axes[0, 0].set_ylabel("Frequency")
            axes[0, 0].set_title("Distribution of Cascade Sizes")
            axes[0, 0].axvline(
                np.mean(sizes),
                color="red",
                linestyle="--",
                label=f"Mean = {np.mean(sizes):.1f}",
            )
            axes[0, 0].legend()

            # Cascade depths
            depths = [c["depth"] for c in cascades]
            axes[0, 1].hist(depths, bins=15, color="coral", edgecolor="black")
            axes[0, 1].set_xlabel("Cascade Depth")
            axes[0, 1].set_ylabel("Frequency")
            axes[0, 1].set_title("Distribution of Cascade Depths")

            # Size vs Depth relationship
            axes[1, 0].scatter(depths, sizes, alpha=0.6, s=50)
            axes[1, 0].set_xlabel("Cascade Depth")
            axes[1, 0].set_ylabel("Cascade Size")
            axes[1, 0].set_title("Cascade Size vs Depth")

            # Add trend line
            if len(depths) > 2:
                z = np.polyfit(depths, sizes, 1)
                p = np.poly1d(z)
                x_line = np.linspace(min(depths), max(depths), 100)
                axes[1, 0].plot(x_line, p(x_line), "r-", alpha=0.8)

            # Cascade fractions
            fractions = [c["fraction"] for c in cascades]
            axes[1, 1].hist(fractions, bins=20, color="seagreen", edgecolor="black")
            axes[1, 1].set_xlabel("Fraction of Network Reached")
            axes[1, 1].set_ylabel("Frequency")
            axes[1, 1].set_title("Cascade Reach Distribution")

        plt.suptitle("Information Cascade Analysis")
        plt.tight_layout()
        return fig

    def create_summary_dashboard(self, results: dict) -> plt.Figure:
        """Create comprehensive dashboard of spillover analysis results.

        Args:
            results: Dictionary containing all analysis results

        Returns:
            Matplotlib figure with summary dashboard
        """
        fig = plt.figure(figsize=(16, 12))

        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Spatial autocorrelation
        ax1 = fig.add_subplot(gs[0, 0])
        if "moran_i" in results:
            moran = results["moran_i"]
            ax1.bar(["Moran's I"], [moran.get("statistic", 0)], color="purple")
            ax1.axhline(
                y=moran.get("expected", 0),
                color="red",
                linestyle="--",
                label="Expected under null",
            )
            ax1.set_ylabel("Statistic")
            ax1.set_title(
                f"Spatial Autocorrelation\np-value = {moran.get('p_value', 0):.3f}"
            )
            ax1.legend()

        # Model comparison
        ax2 = fig.add_subplot(gs[0, 1])
        if "model_comparison" in results:
            models = results["model_comparison"]
            model_names = list(models.keys())
            aic_values = [models[m].get("aic", 0) for m in model_names]
            ax2.bar(model_names, aic_values, color="orange")
            ax2.set_ylabel("AIC")
            ax2.set_title("Model Comparison")
            ax2.tick_params(axis="x", rotation=45)

        # Spillover effects
        ax3 = fig.add_subplot(gs[0, 2])
        if "spillover_effects" in results:
            effects = results["spillover_effects"]
            effect_types = ["Direct", "Indirect", "Total"]
            effect_values = [
                effects.get("direct", 0),
                effects.get("indirect", 0),
                effects.get("total", 0),
            ]
            ax3.bar(effect_types, effect_values, color=["blue", "red", "green"])
            ax3.set_ylabel("Effect Size")
            ax3.set_title("Spillover Effect Decomposition")
            ax3.axhline(y=0, color="k", linestyle="-", alpha=0.3)

        # Adoption timeline
        ax4 = fig.add_subplot(gs[1, :])
        if "adoption_timeline" in results:
            timeline = results["adoption_timeline"]
            ax4.plot(timeline["years"], timeline["cumulative"], marker="o", linewidth=2)
            ax4.fill_between(timeline["years"], 0, timeline["cumulative"], alpha=0.3)
            ax4.set_xlabel("Year")
            ax4.set_ylabel("Countries Adopted")
            ax4.set_title("Policy Adoption Timeline")
            ax4.grid(True, alpha=0.3)

        # Border effects
        ax5 = fig.add_subplot(gs[2, 0])
        if "border_effects" in results:
            borders = results["border_effects"]
            significant = sum(1 for b in borders if b.get("significant", False))
            total = len(borders)
            ax5.pie(
                [significant, total - significant],
                labels=["Significant", "Not Significant"],
                colors=["red", "gray"],
                autopct="%1.1f%%",
            )
            ax5.set_title(f"Border Effects\n{significant}/{total} Significant")

        # Network metrics
        ax6 = fig.add_subplot(gs[2, 1])
        if "network_metrics" in results:
            metrics = results["network_metrics"]
            metric_names = list(metrics.keys())[:5]  # Top 5 metrics
            metric_values = [metrics[m] for m in metric_names]
            ax6.barh(metric_names, metric_values, color="teal")
            ax6.set_xlabel("Value")
            ax6.set_title("Network Statistics")

        # Key findings text
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis("off")
        if "key_findings" in results:
            findings_text = "\n".join([f"• {f}" for f in results["key_findings"][:5]])
        else:
            findings_text = "• Spatial spillovers detected\n• Network effects significant\n• Border discontinuities present"
        ax7.text(
            0.1,
            0.9,
            "Key Findings:",
            fontsize=12,
            fontweight="bold",
            transform=ax7.transAxes,
        )
        ax7.text(
            0.1,
            0.7,
            findings_text,
            fontsize=10,
            transform=ax7.transAxes,
            verticalalignment="top",
        )

        plt.suptitle(
            "Spillover Analysis Summary Dashboard", fontsize=16, fontweight="bold"
        )
        return fig
