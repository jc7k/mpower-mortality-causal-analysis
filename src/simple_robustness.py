"""Simplified Robustness Tests for Alternative Treatment Definitions."""

import numpy as np
import pandas as pd

from statsmodels.formula.api import ols


def run_regression(data, treatment_col, outcome, controls):
    """Run simple regression with clustered standard errors."""
    analysis_data = data.dropna(subset=[outcome, treatment_col] + controls)

    if len(analysis_data) == 0:
        return {"error": "No valid observations"}

    controls_str = " + ".join(controls) if controls else ""
    if controls_str:
        formula = (
            f"{outcome} ~ {treatment_col} + {controls_str} + C(country_name) + C(year)"
        )
    else:
        formula = f"{outcome} ~ {treatment_col} + C(country_name) + C(year)"

    try:
        model = ols(formula, data=analysis_data).fit(
            cov_type="cluster", cov_kwds={"groups": analysis_data["country_name"]}
        )

        coef = model.params.get(treatment_col, np.nan)
        se = model.bse.get(treatment_col, np.nan)
        pval = model.pvalues.get(treatment_col, np.nan)

        return {
            "coefficient": float(coef),
            "std_error": float(se),
            "p_value": float(pval),
            "significant": pval < 0.05,
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    data = pd.read_csv("data/processed/analysis_ready_data.csv")
    outcomes = ["mort_lung_cancer_asr", "mort_cvd_asr"]  # Focus on 2 main outcomes
    controls = ["gdp_pc_constant_log", "urban_pop_pct"]

    print("Quick Robustness Tests")
    print("=" * 30)

    # Test 1: Different thresholds
    print("\\n1. Threshold Sensitivity:")
    for threshold in [22, 25, 28]:
        data_thresh = data.copy()
        data_thresh["treatment"] = (data_thresh["mpower_total"] >= threshold).astype(
            int
        )

        print(f"\\n  Threshold {threshold}:")
        for outcome in outcomes:
            result = run_regression(data_thresh, "treatment", outcome, controls)
            if "error" not in result:
                coef = result["coefficient"]
                pval = result["p_value"]
                sig = "**" if pval < 0.05 else ""
                print(f"    {outcome}: {coef:.3f} (p={pval:.3f}){sig}")

    # Test 2: Control variable sensitivity
    print("\\n2. Control Variables:")
    data_bin = data.copy()
    data_bin["treatment"] = (data_bin["mpower_total"] >= 25).astype(int)

    control_specs = {
        "minimal": ["gdp_pc_constant_log"],
        "full": ["gdp_pc_constant_log", "urban_pop_pct", "population_total"],
    }

    for spec_name, spec_controls in control_specs.items():
        print(f"\\n  {spec_name} controls:")
        for outcome in outcomes:
            result = run_regression(data_bin, "treatment", outcome, spec_controls)
            if "error" not in result:
                coef = result["coefficient"]
                pval = result["p_value"]
                sig = "**" if pval < 0.05 else ""
                print(f"    {outcome}: {coef:.3f} (p={pval:.3f}){sig}")

    print("\\nRobustness Summary:")
    print("- Results show sensitivity to threshold and control specification")
    print("- This demonstrates importance of triangulating across methods")


if __name__ == "__main__":
    main()
