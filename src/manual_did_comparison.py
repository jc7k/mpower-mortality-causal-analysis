"""Manual DiD Analysis for Alternative Treatment Definitions.

This script implements simple two-way fixed effects DiD analysis to compare
different treatment definitions when the advanced Callaway & Sant'Anna
implementation has compatibility issues.
"""

import numpy as np
import pandas as pd

from statsmodels.formula.api import ols

# Constants
MPOWER_THRESHOLD = 25.0
SIGNIFICANCE_LEVEL = 0.05


def create_binary_treatment_data(
    data: pd.DataFrame, threshold: float = MPOWER_THRESHOLD
) -> pd.DataFrame:
    """Create binary treatment definition based on MPOWER threshold."""
    data = data.copy()

    # Create binary treatment variable
    data["mpower_high_binary"] = (data["mpower_total"] >= threshold).astype(int)

    # Find first treatment year for each country
    treated_data = (
        data[data["mpower_high_binary"] == 1].groupby("country_name")["year"].min()
    )
    data["first_high_year"] = data["country_name"].map(treated_data).fillna(0)

    # Create post-treatment indicator
    data["post_treatment"] = (
        (data["first_high_year"] > 0) & (data["year"] >= data["first_high_year"])
    ).astype(int)

    # Only count as treated if they maintain high score for at least 2 periods
    for country in data["country_name"].unique():
        country_data = data[data["country_name"] == country].sort_values("year")
        high_periods = (
            country_data["mpower_high_binary"].rolling(window=2, min_periods=2).sum()
        )

        if (high_periods >= 2).any():
            # Keep the treatment
            continue
        # Remove treatment assignment
        mask = data["country_name"] == country
        data.loc[mask, "first_high_year"] = 0
        data.loc[mask, "post_treatment"] = 0

    return data


def create_continuous_treatment_data(
    data: pd.DataFrame, baseline_years: list[int] = None
) -> pd.DataFrame:
    """Create continuous change treatment definition."""
    if baseline_years is None:
        baseline_years = [2008, 2010, 2012]

    data = data.copy()

    # Calculate baseline MPOWER score by country
    baseline_scores = (
        data[data["year"].isin(baseline_years)]
        .groupby("country_name")["mpower_total"]
        .mean()
    )

    data["baseline_mpower"] = data["country_name"].map(baseline_scores)
    data["mpower_improvement"] = data["mpower_total"] - data["baseline_mpower"]
    data["mpower_pct_improvement"] = (
        data["mpower_improvement"] / data["baseline_mpower"]
    )

    # Find first year with substantial improvement (20% increase)
    improvement_threshold = 0.20
    substantial_improvement = data["mpower_pct_improvement"] >= improvement_threshold

    first_improvement_year = (
        data[substantial_improvement].groupby("country_name")["year"].min()
    )

    data["first_improvement_year"] = (
        data["country_name"].map(first_improvement_year).fillna(0)
    )

    # Create post-improvement indicator
    data["post_improvement"] = (
        (data["first_improvement_year"] > 0)
        & (data["year"] >= data["first_improvement_year"])
    ).astype(int)

    return data


def run_manual_did_analysis(
    data: pd.DataFrame, treatment_col: str, outcome: str, control_vars: list[str]
) -> dict:
    """Run manual two-way fixed effects DiD analysis."""
    # Prepare data
    analysis_data = data.dropna(subset=[outcome, treatment_col] + control_vars)

    if len(analysis_data) == 0:
        return {"error": "No valid observations after dropping missing values"}

    # Create formula
    controls_str = " + ".join(control_vars) if control_vars else ""
    if controls_str:
        formula = (
            f"{outcome} ~ {treatment_col} + {controls_str} + C(country_name) + C(year)"
        )
    else:
        formula = f"{outcome} ~ {treatment_col} + C(country_name) + C(year)"

    try:
        # Run regression
        model = ols(formula, data=analysis_data).fit(
            cov_type="cluster", cov_kwds={"groups": analysis_data["country_name"]}
        )

        # Extract treatment coefficient
        treatment_coef = model.params.get(treatment_col, np.nan)
        treatment_se = model.bse.get(treatment_col, np.nan)
        treatment_pval = model.pvalues.get(treatment_col, np.nan)

        # Calculate confidence interval
        ci_lower = treatment_coef - 1.96 * treatment_se
        ci_upper = treatment_coef + 1.96 * treatment_se

        return {
            "att": float(treatment_coef),
            "std_error": float(treatment_se),
            "p_value": float(treatment_pval),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "r_squared": float(model.rsquared),
            "n_obs": int(model.nobs),
            "method": "manual_twfe_did",
        }

    except Exception as e:
        return {"error": str(e)}


def main():
    """Run comprehensive manual DiD comparison."""
    print("Loading data...")
    data = pd.read_csv("data/processed/analysis_ready_data.csv")

    outcomes = ["mort_lung_cancer_asr", "mort_cvd_asr", "mort_ihd_asr", "mort_copd_asr"]
    control_vars = [
        "gdp_pc_constant_log",
        "urban_pop_pct",
        "population_total",
        "edu_exp_pct_gdp",
    ]

    results = {
        "binary_threshold_manual": {},
        "continuous_change_manual": {},
        "dose_response_manual": {},
    }

    print("\\nRunning Manual DiD Analysis...")
    print("=" * 50)

    # 1. Binary threshold analysis
    print("\\n1. Binary Threshold Analysis (MPOWER ≥ 25)")
    binary_data = create_binary_treatment_data(data, threshold=MPOWER_THRESHOLD)

    treated_countries_binary = (binary_data["first_high_year"] > 0).sum() / len(
        binary_data["country_name"].unique()
    )
    print(
        f"   Treated countries: {(binary_data['first_high_year'] > 0).any():} ({treated_countries_binary:.1%} of countries)"
    )

    results["binary_threshold_manual"]["outcomes"] = {}
    for outcome in outcomes:
        print(f"   Analyzing {outcome}...")
        result = run_manual_did_analysis(
            binary_data, "post_treatment", outcome, control_vars
        )
        results["binary_threshold_manual"]["outcomes"][outcome] = result

        if "error" not in result:
            att = result["att"]
            pval = result["p_value"]
            sig = (
                "***"
                if pval < 0.01
                else "**"
                if pval < 0.05
                else "*"
                if pval < 0.1
                else ""
            )
            print(f"     Effect: {att:.3f} (p={pval:.3f}){sig}")
        else:
            print(f"     Error: {result['error']}")

    # 2. Continuous change analysis
    print("\\n2. Continuous Change Analysis (≥20% improvement)")
    continuous_data = create_continuous_treatment_data(data)

    treated_countries_continuous = (continuous_data["first_improvement_year"] > 0).any()
    print(f"   Treated countries: {treated_countries_continuous}")

    results["continuous_change_manual"]["outcomes"] = {}
    for outcome in outcomes:
        print(f"   Analyzing {outcome}...")
        result = run_manual_did_analysis(
            continuous_data, "post_improvement", outcome, control_vars
        )
        results["continuous_change_manual"]["outcomes"][outcome] = result

        if "error" not in result:
            att = result["att"]
            pval = result["p_value"]
            sig = (
                "***"
                if pval < 0.01
                else "**"
                if pval < 0.05
                else "*"
                if pval < 0.1
                else ""
            )
            print(f"     Effect: {att:.3f} (p={pval:.3f}){sig}")
        else:
            print(f"     Error: {result['error']}")

    # 3. Dose-response analysis
    print("\\n3. Dose-Response Analysis (Continuous MPOWER)")

    # Normalize MPOWER scores
    dose_data = data.copy()
    dose_data["mpower_normalized"] = (
        dose_data["mpower_total"] - dose_data["mpower_total"].min()
    ) / (dose_data["mpower_total"].max() - dose_data["mpower_total"].min())

    results["dose_response_manual"]["outcomes"] = {}
    for outcome in outcomes:
        print(f"   Analyzing {outcome}...")
        result = run_manual_did_analysis(
            dose_data, "mpower_normalized", outcome, control_vars
        )
        results["dose_response_manual"]["outcomes"][outcome] = result

        if "error" not in result:
            coef = result["att"]  # In dose-response, this is the coefficient not ATT
            pval = result["p_value"]
            sig = (
                "***"
                if pval < 0.01
                else "**"
                if pval < 0.05
                else "*"
                if pval < 0.1
                else ""
            )
            print(f"     Coefficient: {coef:.3f} (p={pval:.3f}){sig}")
        else:
            print(f"     Error: {result['error']}")

    # Summary comparison
    print("\\n" + "=" * 60)
    print("SUMMARY COMPARISON - ALTERNATIVE TREATMENT DEFINITIONS")
    print("=" * 60)

    comparison_table = []
    for outcome in outcomes:
        row = {"outcome": outcome}

        for method in [
            "binary_threshold_manual",
            "continuous_change_manual",
            "dose_response_manual",
        ]:
            result = results[method]["outcomes"].get(outcome, {})
            if "error" not in result:
                effect = result["att"]
                pval = result["p_value"]
                significant = pval < SIGNIFICANCE_LEVEL
                row[f"{method}_effect"] = effect
                row[f"{method}_pval"] = pval
                row[f"{method}_significant"] = significant
            else:
                row[f"{method}_effect"] = np.nan
                row[f"{method}_pval"] = np.nan
                row[f"{method}_significant"] = False

        comparison_table.append(row)

    # Create summary DataFrame
    summary_df = pd.DataFrame(comparison_table)

    print("\\nEffect Estimates (with significance):")
    print("-" * 40)
    for _, row in summary_df.iterrows():
        print(f"\\n{row['outcome']}:")

        binary_effect = row["binary_threshold_manual_effect"]
        binary_sig = (
            "***"
            if row["binary_threshold_manual_pval"] < 0.01
            else "**"
            if row["binary_threshold_manual_pval"] < 0.05
            else "*"
            if row["binary_threshold_manual_pval"] < 0.1
            else ""
        )
        print(f"  Binary threshold (≥25):    {binary_effect:.3f}{binary_sig}")

        continuous_effect = row["continuous_change_manual_effect"]
        continuous_sig = (
            "***"
            if row["continuous_change_manual_pval"] < 0.01
            else "**"
            if row["continuous_change_manual_pval"] < 0.05
            else "*"
            if row["continuous_change_manual_pval"] < 0.1
            else ""
        )
        print(f"  Continuous change (≥20%):  {continuous_effect:.3f}{continuous_sig}")

        dose_effect = row["dose_response_manual_effect"]
        dose_sig = (
            "***"
            if row["dose_response_manual_pval"] < 0.01
            else "**"
            if row["dose_response_manual_pval"] < 0.05
            else "*"
            if row["dose_response_manual_pval"] < 0.1
            else ""
        )
        print(f"  Dose-response (continuous): {dose_effect:.3f}{dose_sig}")

    # Export results
    import json

    from pathlib import Path

    output_dir = Path("results/alternative_treatment/")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    with open(output_dir / "manual_did_results.json", "w") as f:
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        def deep_convert(data):
            if isinstance(data, dict):
                return {k: deep_convert(v) for k, v in data.items()}
            if isinstance(data, list):
                return [deep_convert(v) for v in data]
            return convert_numpy(data)

        json.dump(deep_convert(results), f, indent=2)

    # Save summary table
    summary_df.to_excel(output_dir / "treatment_comparison_summary.xlsx", index=False)
    summary_df.to_csv(output_dir / "treatment_comparison_summary.csv", index=False)

    print(f"\\nResults saved to: {output_dir}")
    print("\\nKey Finding:")
    print("- Binary threshold and continuous change approaches provide DiD estimates")
    print("- Dose-response provides marginal effects of MPOWER score increases")
    print("- All approaches complement each other for robustness checking")


if __name__ == "__main__":
    main()
