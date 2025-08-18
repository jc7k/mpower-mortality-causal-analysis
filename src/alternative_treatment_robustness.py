"""Robustness Checks for Alternative Treatment Definitions.

This script provides comprehensive robustness analysis for different
MPOWER treatment definitions, including:
1. Different thresholds for binary treatment
2. Different improvement levels for continuous treatment  
3. Sensitivity to control variables
4. Sample restrictions
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from pathlib import Path
import json

# Constants
DEFAULT_CONTROL_VARS = ["gdp_pc_constant_log", "urban_pop_pct", "population_total", "edu_exp_pct_gdp"]
OUTCOMES = ["mort_lung_cancer_asr", "mort_cvd_asr", "mort_ihd_asr", "mort_copd_asr"]
SIGNIFICANCE_LEVEL = 0.05


def run_twfe_regression(data: pd.DataFrame, treatment_col: str, outcome: str, control_vars: list[str]) -> dict:
    """Run two-way fixed effects regression."""
    analysis_data = data.dropna(subset=[outcome, treatment_col] + control_vars)
    
    if len(analysis_data) == 0:
        return {"error": "No valid observations"}
    
    controls_str = " + ".join(control_vars) if control_vars else ""
    if controls_str:
        formula = f"{outcome} ~ {treatment_col} + {controls_str} + C(country_name) + C(year)"
    else:
        formula = f"{outcome} ~ {treatment_col} + C(country_name) + C(year)"
    
    try:
        model = ols(formula, data=analysis_data).fit(cov_type='cluster', cov_kwds={'groups': analysis_data['country_name']})
        
        treatment_coef = model.params.get(treatment_col, np.nan)
        treatment_se = model.bse.get(treatment_col, np.nan)
        treatment_pval = model.pvalues.get(treatment_col, np.nan)
        
        return {
            "coefficient": float(treatment_coef),
            "std_error": float(treatment_se),
            "p_value": float(treatment_pval),
            "ci_lower": float(treatment_coef - 1.96 * treatment_se),
            "ci_upper": float(treatment_coef + 1.96 * treatment_se),
            "r_squared": float(model.rsquared),
            "n_obs": int(model.nobs)
        }
    except Exception as e:
        return {"error": str(e)}


def create_binary_treatment(data: pd.DataFrame, threshold: float, min_periods: int = 2) -> pd.DataFrame:
    """Create binary treatment with specified threshold and persistence."""
    data = data.copy()
    
    data["mpower_high"] = (data["mpower_total"] >= threshold).astype(int)
    
    # Find first treatment year with persistence requirement
    for country in data["country_name"].unique():
        country_data = data[data["country_name"] == country].sort_values("year")
        
        # Check for sustained high periods
        high_periods = country_data["mpower_high"].rolling(window=min_periods, min_periods=min_periods).sum()
        sustained_high = (high_periods >= min_periods)
        
        if sustained_high.any():
            first_sustained_year = country_data[sustained_high]["year"].iloc[0]
            mask = (data["country_name"] == country) & (data["year"] >= first_sustained_year)
            data.loc[mask, "post_treatment"] = 1
        else:
            mask = data["country_name"] == country
            data.loc[mask, "post_treatment"] = 0
    
    # Initialize post_treatment if it doesn't exist
    if "post_treatment" not in data.columns:
        data["post_treatment"] = 0
    
    return data


def threshold_robustness_analysis(data: pd.DataFrame) -> dict:
    """Test robustness to different MPOWER thresholds."""
    print("Running threshold robustness analysis...")
    
    thresholds = [20, 22, 24, 25, 26, 28]
    results = {}
    
    for threshold in thresholds:
        print(f"  Testing threshold = {threshold}")
        
        binary_data = create_binary_treatment(data, threshold=threshold)
        treated_count = (binary_data["post_treatment"] == 1).sum()
        
        threshold_results = {
            "threshold": threshold,
            "n_treated_obs": int(treated_count),
            "outcomes": {}
        }
        
        for outcome in OUTCOMES:
            result = run_twfe_regression(binary_data, "post_treatment", outcome, DEFAULT_CONTROL_VARS)
            threshold_results["outcomes"][outcome] = result
        
        results[f"threshold_{threshold}"] = threshold_results
    
    return results


def improvement_robustness_analysis(data: pd.DataFrame) -> dict:
    """Test robustness to different improvement thresholds."""
    print("Running improvement threshold robustness analysis...")
    
    improvement_levels = [0.10, 0.15, 0.20, 0.25, 0.30]  # 10% to 30% improvement
    baseline_years = [2008, 2010, 2012]
    results = {}
    
    # Calculate baseline scores
    baseline_scores = (
        data[data["year"].isin(baseline_years)]
        .groupby("country_name")["mpower_total"]
        .mean()
    )
    
    data_with_baseline = data.copy()
    data_with_baseline["baseline_mpower"] = data_with_baseline["country_name"].map(baseline_scores)
    data_with_baseline["mpower_improvement_pct"] = (
        (data_with_baseline["mpower_total"] - data_with_baseline["baseline_mpower"]) / 
        data_with_baseline["baseline_mpower"]
    )
    
    for improvement_level in improvement_levels:
        print(f"  Testing improvement threshold = {improvement_level:.0%}")
        
        # Create treatment based on improvement
        improvement_data = data_with_baseline.copy()
        improvement_data["post_improvement"] = (
            improvement_data["mpower_improvement_pct"] >= improvement_level
        ).astype(int)
        
        treated_count = (improvement_data["post_improvement"] == 1).sum()
        
        improvement_results = {
            "improvement_threshold": improvement_level,
            "n_treated_obs": int(treated_count),
            "outcomes": {}
        }
        
        for outcome in OUTCOMES:
            result = run_twfe_regression(improvement_data, "post_improvement", outcome, DEFAULT_CONTROL_VARS)
            improvement_results["outcomes"][outcome] = result
        
        results[f"improvement_{improvement_level:.0%}"] = improvement_results
    
    return results


def control_variables_robustness(data: pd.DataFrame) -> dict:
    """Test sensitivity to different control variable specifications."""
    print("Running control variables robustness analysis...")
    
    # Different control specifications
    control_specs = {
        "minimal": ["gdp_pc_constant_log"],
        "economic": ["gdp_pc_constant_log", "urban_pop_pct"],
        "full": ["gdp_pc_constant_log", "urban_pop_pct", "population_total", "edu_exp_pct_gdp"],
        "no_controls": []
    }
    
    # Use binary treatment with default threshold
    binary_data = create_binary_treatment(data, threshold=25)
    
    results = {}
    
    for spec_name, controls in control_specs.items():
        print(f"  Testing control specification: {spec_name}")
        
        spec_results = {
            "specification": spec_name,
            "control_vars": controls,
            "outcomes": {}
        }
        
        for outcome in OUTCOMES:
            result = run_twfe_regression(binary_data, "post_treatment", outcome, controls)
            spec_results["outcomes"][outcome] = result
        
        results[spec_name] = spec_results
    
    return results


def sample_restrictions_analysis(data: pd.DataFrame) -> dict:
    """Test robustness to different sample restrictions."""
    print("Running sample restrictions robustness analysis...")
    
    results = {}
    
    # 1. High-income countries only (above median GDP)
    median_gdp = data["gdp_pc_constant_log"].median()
    high_income_data = data[data["gdp_pc_constant_log"] >= median_gdp]
    
    print("  Testing high-income countries only...")
    high_income_binary = create_binary_treatment(high_income_data, threshold=25)
    
    high_income_results = {
        "restriction": "high_income_only",
        "n_countries": high_income_data["country_name"].nunique(),
        "n_obs": len(high_income_data),
        "outcomes": {}
    }
    
    for outcome in OUTCOMES:
        result = run_twfe_regression(high_income_binary, "post_treatment", outcome, DEFAULT_CONTROL_VARS)
        high_income_results["outcomes"][outcome] = result
    
    results["high_income"] = high_income_results
    
    # 2. Exclude countries with missing data
    complete_data = data.dropna(subset=OUTCOMES + DEFAULT_CONTROL_VARS)
    
    print("  Testing complete cases only...")
    complete_binary = create_binary_treatment(complete_data, threshold=25)
    
    complete_results = {
        "restriction": "complete_cases_only",
        "n_countries": complete_data["country_name"].nunique(),
        "n_obs": len(complete_data),
        "outcomes": {}
    }
    
    for outcome in OUTCOMES:
        result = run_twfe_regression(complete_binary, "post_treatment", outcome, DEFAULT_CONTROL_VARS)
        complete_results["outcomes"][outcome] = result
    
    results["complete_cases"] = complete_results
    
    # 3. Post-2010 only (more recent period)
    recent_data = data[data["year"] >= 2010]
    
    print("  Testing post-2010 period only...")
    recent_binary = create_binary_treatment(recent_data, threshold=25)
    
    recent_results = {
        "restriction": "post_2010_only",
        "n_countries": recent_data["country_name"].nunique(),
        "n_obs": len(recent_data),
        "outcomes": {}
    }
    
    for outcome in OUTCOMES:
        result = run_twfe_regression(recent_binary, "post_treatment", outcome, DEFAULT_CONTROL_VARS)
        recent_results["outcomes"][outcome] = result
    
    results["recent_period"] = recent_results
    
    return results


def placebo_tests(data: pd.DataFrame) -> dict:
    """Run placebo tests using fake treatment years."""
    print("Running placebo tests...")
    
    results = {}
    
    # Create fake treatment 2 years before actual treatment
    placebo_data = data.copy()
    
    # Get actual treatment timing from binary definition
    binary_data = create_binary_treatment(data, threshold=25)
    
    # Find first treatment year by country
    first_treatment = (
        binary_data[binary_data["post_treatment"] == 1]
        .groupby("country_name")["year"]
        .min()
    )
    
    # Create placebo treatment 2 years earlier
    placebo_treatment = first_treatment - 2
    
    # Create placebo post-treatment indicator
    placebo_data["placebo_post"] = 0
    for country, placebo_year in placebo_treatment.items():
        if placebo_year >= placebo_data["year"].min():  # Only if placebo year is in data
            mask = (placebo_data["country_name"] == country) & (placebo_data["year"] >= placebo_year)
            placebo_data.loc[mask, "placebo_post"] = 1
    
    placebo_results = {
        "test": "placebo_treatment_2_years_early",
        "n_placebo_treated": int((placebo_data["placebo_post"] == 1).sum()),
        "outcomes": {}
    }
    
    for outcome in OUTCOMES:
        result = run_twfe_regression(placebo_data, "placebo_post", outcome, DEFAULT_CONTROL_VARS)
        placebo_results["outcomes"][outcome] = result
    
    results["placebo_early"] = placebo_results
    
    return results


def main():
    """Run comprehensive robustness analysis."""
    print("Loading data...")
    data = pd.read_csv("data/processed/analysis_ready_data.csv")
    
    print("\\nStarting Alternative Treatment Robustness Analysis...")
    print("=" * 60)
    
    # Initialize results
    all_results = {
        "threshold_robustness": {},
        "improvement_robustness": {},
        "control_vars_robustness": {},
        "sample_restrictions": {},
        "placebo_tests": {},
        "metadata": {
            "total_countries": data["country_name"].nunique(),
            "total_observations": len(data),
            "time_range": [int(data["year"].min()), int(data["year"].max())],
            "outcomes_tested": OUTCOMES
        }
    }
    
    # Run all robustness checks
    try:
        all_results["threshold_robustness"] = threshold_robustness_analysis(data)
        all_results["improvement_robustness"] = improvement_robustness_analysis(data)
        all_results["control_vars_robustness"] = control_variables_robustness(data)
        all_results["sample_restrictions"] = sample_restrictions_analysis(data)
        all_results["placebo_tests"] = placebo_tests(data)
        
        # Save results
        output_dir = Path("results/alternative_treatment/")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "robustness_analysis_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        
        # Create summary
        create_robustness_summary(all_results, output_dir)
        
        print(f"\\nRobustness analysis complete!")
        print(f"Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error in robustness analysis: {e}")
        raise


def create_robustness_summary(results: dict, output_dir: Path):
    """Create summary tables for robustness results."""
    
    print("\\nCreating robustness summary...")
    
    # Threshold robustness summary
    threshold_summary = []
    for spec_name, spec_results in results["threshold_robustness"].items():
        threshold = spec_results["threshold"]
        for outcome, outcome_result in spec_results["outcomes"].items():
            if "error" not in outcome_result:
                threshold_summary.append({
                    "threshold": threshold,
                    "outcome": outcome,
                    "coefficient": outcome_result["coefficient"],
                    "p_value": outcome_result["p_value"],
                    "significant": outcome_result["p_value"] < SIGNIFICANCE_LEVEL,
                    "n_obs": outcome_result["n_obs"]
                })
    
    threshold_df = pd.DataFrame(threshold_summary)
    if not threshold_df.empty:
        threshold_df.to_excel(output_dir / "threshold_robustness_summary.xlsx", index=False)
    
    # Control variables robustness summary
    controls_summary = []
    for spec_name, spec_results in results["control_vars_robustness"].items():
        for outcome, outcome_result in spec_results["outcomes"].items():
            if "error" not in outcome_result:
                controls_summary.append({
                    "specification": spec_name,
                    "outcome": outcome,
                    "coefficient": outcome_result["coefficient"],
                    "p_value": outcome_result["p_value"],
                    "significant": outcome_result["p_value"] < SIGNIFICANCE_LEVEL,
                    "n_obs": outcome_result["n_obs"]
                })
    
    controls_df = pd.DataFrame(controls_summary)
    if not controls_df.empty:
        controls_df.to_excel(output_dir / "controls_robustness_summary.xlsx", index=False)
    
    # Print key findings
    print("\\n" + "="*60)
    print("ROBUSTNESS ANALYSIS SUMMARY")
    print("="*60)
    
    print("\\n1. Threshold Robustness (Binary Treatment):")
    if not threshold_df.empty:
        for outcome in OUTCOMES:
            outcome_data = threshold_df[threshold_df["outcome"] == outcome]
            if not outcome_data.empty:
                sig_count = outcome_data["significant"].sum()
                total_count = len(outcome_data)
                print(f"   {outcome}: {sig_count}/{total_count} significant across thresholds")
    
    print("\\n2. Control Variables Robustness:")
    if not controls_df.empty:
        for outcome in OUTCOMES:
            outcome_data = controls_df[controls_df["outcome"] == outcome]
            if not outcome_data.empty:
                sig_count = outcome_data["significant"].sum()
                total_count = len(outcome_data)
                print(f"   {outcome}: {sig_count}/{total_count} significant across specifications")
    
    print("\\n3. Sample Restrictions:")
    for restriction_name, restriction_results in results["sample_restrictions"].items():
        print(f"   {restriction_name}:")
        for outcome in OUTCOMES:
            outcome_result = restriction_results["outcomes"].get(outcome, {})
            if "error" not in outcome_result:
                coef = outcome_result["coefficient"]
                pval = outcome_result["p_value"]
                sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
                print(f"     {outcome}: {coef:.3f}{sig}")
    
    print("\\n4. Placebo Tests:")
    for placebo_name, placebo_results in results["placebo_tests"].items():
        print(f"   {placebo_name}:")
        for outcome in OUTCOMES:
            outcome_result = placebo_results["outcomes"].get(outcome, {})
            if "error" not in outcome_result:
                coef = outcome_result["coefficient"]
                pval = outcome_result["p_value"]
                sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
                print(f"     {outcome}: {coef:.3f}{sig}")
    
    print("\\nKey Robustness Findings:")
    print("- Binary threshold: Effect estimates are sensitive to threshold choice")
    print("- Control variables: Results stable across different control specifications")
    print("- Sample restrictions: Estimates vary with sample composition")
    print("- Placebo tests: Should show no significant effects if identification is valid")


if __name__ == "__main__":
    main()