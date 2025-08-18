"""Common Diagnostic Tools for Advanced DiD Methods.

This module provides shared diagnostic functions for evaluating
DiD assumptions and method performance.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

# Constants
SIGNIFICANCE_LEVEL = 0.05
MIN_PRE_PERIODS = 2


def test_parallel_trends(
    data: pd.DataFrame,
    outcome: str,
    unit_col: str,
    time_col: str,
    treatment_col: str,
    pre_periods: int = MIN_PRE_PERIODS,
) -> dict:
    """Test parallel trends assumption using pre-treatment data.

    Args:
        data: Panel data DataFrame
        outcome: Outcome variable column name
        unit_col: Unit identifier column
        time_col: Time period column
        treatment_col: Treatment indicator column
        pre_periods: Number of pre-treatment periods to test

    Returns:
        Dictionary with test results
    """
    # Identify treatment timing for each unit
    treatment_timing = data[data[treatment_col] == 1].groupby(unit_col)[time_col].min()

    # Collect pre-treatment trends
    pre_trends_treated = []
    pre_trends_control = []

    for unit in data[unit_col].unique():
        unit_data = data[data[unit_col] == unit].sort_values(time_col)

        if unit in treatment_timing.index:
            # Treated unit
            treat_time = treatment_timing[unit]
            pre_data = unit_data[unit_data[time_col] < treat_time].tail(pre_periods)

            if len(pre_data) >= MIN_PRE_PERIODS:
                # Calculate trend
                y = pre_data[outcome].values
                x = np.arange(len(y))
                if len(y) > 1:
                    slope = np.polyfit(x, y, 1)[0]
                    pre_trends_treated.append(slope)
        else:
            # Control unit
            pre_data = unit_data.head(pre_periods)

            if len(pre_data) >= MIN_PRE_PERIODS:
                y = pre_data[outcome].values
                x = np.arange(len(y))
                if len(y) > 1:
                    slope = np.polyfit(x, y, 1)[0]
                    pre_trends_control.append(slope)

    if not pre_trends_treated or not pre_trends_control:
        return {
            "test_statistic": np.nan,
            "p_value": np.nan,
            "reject_parallel": False,
            "message": "Insufficient pre-treatment data",
        }

    # Test for difference in trends
    t_stat, p_value = stats.ttest_ind(pre_trends_treated, pre_trends_control)

    return {
        "test_statistic": t_stat,
        "p_value": p_value,
        "reject_parallel": p_value < SIGNIFICANCE_LEVEL,
        "mean_trend_treated": np.mean(pre_trends_treated),
        "mean_trend_control": np.mean(pre_trends_control),
        "n_treated": len(pre_trends_treated),
        "n_control": len(pre_trends_control),
    }


def check_common_support(
    data: pd.DataFrame,
    treatment_col: str,
    covariates: list[str],
    method: str = "propensity",
) -> dict:
    """Check common support between treatment and control groups.

    Args:
        data: Panel data DataFrame
        treatment_col: Treatment indicator column
        covariates: List of covariate columns
        method: Method for checking support ('propensity' or 'covariate')

    Returns:
        Dictionary with common support diagnostics
    """
    treated = data[data[treatment_col] == 1]
    control = data[data[treatment_col] == 0]

    if method == "propensity":
        # Estimate propensity scores

        x = data[covariates].fillna(0)
        y = data[treatment_col]

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)

        model = LogisticRegression(max_iter=1000)
        model.fit(x_scaled, y)

        propensity_scores = model.predict_proba(x_scaled)[:, 1]

        ps_treated = propensity_scores[data[treatment_col] == 1]
        ps_control = propensity_scores[data[treatment_col] == 0]

        # Check overlap
        min_treated = ps_treated.min()
        max_treated = ps_treated.max()
        min_control = ps_control.min()
        max_control = ps_control.max()

        overlap_min = max(min_treated, min_control)
        overlap_max = min(max_treated, max_control)

        return {
            "method": "propensity_score",
            "overlap_range": (overlap_min, overlap_max),
            "overlap_proportion": (overlap_max - overlap_min)
            / (max(max_treated, max_control) - min(min_treated, min_control)),
            "mean_ps_treated": ps_treated.mean(),
            "mean_ps_control": ps_control.mean(),
            "good_support": overlap_min < overlap_max,
        }

    if method == "covariate":
        # Check covariate ranges
        support_results = {}

        for cov in covariates:
            if cov not in data.columns:
                continue

            cov_treated = treated[cov].dropna()
            cov_control = control[cov].dropna()

            if len(cov_treated) == 0 or len(cov_control) == 0:
                continue

            # Check overlap
            min_treated = cov_treated.min()
            max_treated = cov_treated.max()
            min_control = cov_control.min()
            max_control = cov_control.max()

            overlap_min = max(min_treated, min_control)
            overlap_max = min(max_treated, max_control)

            support_results[cov] = {
                "overlap_range": (overlap_min, overlap_max),
                "has_overlap": overlap_min <= overlap_max,
                "treated_range": (min_treated, max_treated),
                "control_range": (min_control, max_control),
            }

        # Overall assessment
        n_with_overlap = sum(1 for r in support_results.values() if r["has_overlap"])

        return {
            "method": "covariate_ranges",
            "covariate_support": support_results,
            "n_covariates": len(support_results),
            "n_with_overlap": n_with_overlap,
            "good_support": n_with_overlap == len(support_results),
        }

    msg = f"Unknown method: {method}"
    raise ValueError(msg)


def test_no_anticipation(
    data: pd.DataFrame,
    outcome: str,
    unit_col: str,
    time_col: str,
    treatment_col: str,
    leads: int = 2,
) -> dict:
    """Test no anticipation assumption using lead coefficients.

    Args:
        data: Panel data DataFrame
        outcome: Outcome variable column name
        unit_col: Unit identifier column
        time_col: Time period column
        treatment_col: Treatment indicator column
        leads: Number of lead periods to test

    Returns:
        Dictionary with no anticipation test results
    """
    # Create lead treatment indicators
    lead_results = []

    for lead in range(1, leads + 1):
        data[f"treatment_lead_{lead}"] = data.groupby(unit_col)[treatment_col].shift(
            -lead
        )

        # Run regression with lead

        # Simple regression (should include fixed effects in practice)
        valid_data = data.dropna(subset=[outcome, f"treatment_lead_{lead}"])

        if len(valid_data) > 0:
            x = add_constant(valid_data[f"treatment_lead_{lead}"])
            y = valid_data[outcome]

            model = OLS(y, x)
            result = model.fit()

            lead_results.append(
                {
                    "lead": lead,
                    "coefficient": result.params[f"treatment_lead_{lead}"],
                    "se": result.bse[f"treatment_lead_{lead}"],
                    "p_value": result.pvalues[f"treatment_lead_{lead}"],
                }
            )

    if not lead_results:
        return {
            "test_statistic": np.nan,
            "p_value": np.nan,
            "reject_no_anticipation": False,
            "message": "Could not test anticipation",
        }

    # Joint test on all leads
    lead_coefs = [r["coefficient"] for r in lead_results]
    lead_ses = [r["se"] for r in lead_results]

    # Wald statistic
    wald_stat = sum(
        (c / s) ** 2 for c, s in zip(lead_coefs, lead_ses, strict=False) if s > 0
    )
    p_value = 1 - stats.chi2.cdf(wald_stat, df=len(lead_coefs))

    return {
        "test_statistic": wald_stat,
        "p_value": p_value,
        "reject_no_anticipation": p_value < SIGNIFICANCE_LEVEL,
        "lead_results": lead_results,
        "n_leads_tested": len(lead_results),
    }


def compute_effective_sample_size(
    data: pd.DataFrame, treatment_col: str, weights: pd.Series | None = None
) -> dict:
    """Compute effective sample size for treatment and control groups.

    Args:
        data: Panel data DataFrame
        treatment_col: Treatment indicator column
        weights: Optional weights for observations

    Returns:
        Dictionary with effective sample sizes
    """
    treated = data[treatment_col] == 1
    control = data[treatment_col] == 0

    if weights is not None:
        # Weighted effective sample size
        weights_treated = weights[treated]
        weights_control = weights[control]

        ess_treated = (weights_treated.sum() ** 2) / (weights_treated**2).sum()
        ess_control = (weights_control.sum() ** 2) / (weights_control**2).sum()
    else:
        # Unweighted sample size
        ess_treated = treated.sum()
        ess_control = control.sum()

    return {
        "n_treated": treated.sum(),
        "n_control": control.sum(),
        "ess_treated": ess_treated,
        "ess_control": ess_control,
        "ess_ratio": ess_treated / treated.sum() if treated.sum() > 0 else 0,
        "total_ess": ess_treated + ess_control,
    }


def plot_diagnostic_summary(diagnostics: dict, save_path: str | None = None) -> None:
    """Create summary visualization of diagnostic tests.

    Args:
        diagnostics: Dictionary with diagnostic test results
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Extract test results
    tests = []
    p_values = []

    if "parallel_trends" in diagnostics:
        tests.append("Parallel Trends")
        p_values.append(diagnostics["parallel_trends"].get("p_value", np.nan))

    if "no_anticipation" in diagnostics:
        tests.append("No Anticipation")
        p_values.append(diagnostics["no_anticipation"].get("p_value", np.nan))

    if "common_support" in diagnostics:
        tests.append("Common Support")
        p_values.append(
            1.0 if diagnostics["common_support"].get("good_support", False) else 0.0
        )

    # Plot 1: Test p-values
    if tests:
        x_pos = np.arange(len(tests))
        colors = ["red" if p < SIGNIFICANCE_LEVEL else "green" for p in p_values]

        axes[0, 0].bar(x_pos, p_values, color=colors, alpha=0.7)
        axes[0, 0].axhline(
            SIGNIFICANCE_LEVEL,
            color="black",
            linestyle="--",
            label=f"α = {SIGNIFICANCE_LEVEL}",
        )
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(tests, rotation=45)
        axes[0, 0].set_ylabel("P-value")
        axes[0, 0].set_title("Diagnostic Test Results")
        axes[0, 0].legend()

    # Plot 2: Sample sizes
    if "effective_sample_size" in diagnostics:
        ess = diagnostics["effective_sample_size"]

        categories = ["Nominal", "Effective"]
        treated_sizes = [ess["n_treated"], ess["ess_treated"]]
        control_sizes = [ess["n_control"], ess["ess_control"]]

        x = np.arange(len(categories))
        width = 0.35

        axes[0, 1].bar(x - width / 2, treated_sizes, width, label="Treated", alpha=0.7)
        axes[0, 1].bar(x + width / 2, control_sizes, width, label="Control", alpha=0.7)
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(categories)
        axes[0, 1].set_ylabel("Sample Size")
        axes[0, 1].set_title("Effective Sample Sizes")
        axes[0, 1].legend()

    # Plot 3: Trend comparison
    if "parallel_trends" in diagnostics:
        pt = diagnostics["parallel_trends"]
        if "mean_trend_treated" in pt and "mean_trend_control" in pt:
            trends = [pt["mean_trend_treated"], pt["mean_trend_control"]]
            groups = ["Treated", "Control"]

            axes[1, 0].bar(groups, trends, alpha=0.7)
            axes[1, 0].set_ylabel("Mean Pre-Treatment Trend")
            axes[1, 0].set_title("Pre-Treatment Trends Comparison")
            axes[1, 0].axhline(0, color="black", linestyle="-", alpha=0.3)

    # Plot 4: Summary table
    axes[1, 1].axis("tight")
    axes[1, 1].axis("off")

    summary_data = []
    for test_name, test_result in diagnostics.items():
        if isinstance(test_result, dict) and "p_value" in test_result:
            summary_data.append(
                [
                    test_name.replace("_", " ").title(),
                    f"{test_result.get('p_value', np.nan):.4f}",
                    "Pass"
                    if test_result.get("p_value", 0) >= SIGNIFICANCE_LEVEL
                    else "Fail",
                ]
            )

    if summary_data:
        table = axes[1, 1].table(
            cellText=summary_data,
            colLabels=["Test", "P-value", "Result"],
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

    plt.suptitle("DiD Diagnostic Summary", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
