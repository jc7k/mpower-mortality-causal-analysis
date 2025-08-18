#!/usr/bin/env python3
"""MPOWER Synthetic Control Analysis Demo

This script demonstrates how to use the new synthetic control functionality
to address parallel trends violations in the MPOWER mortality analysis.

Usage:
    python demo_synthetic_control.py [data_path]

Example:
    # Using real MPOWER data (if available)
    python demo_synthetic_control.py data/processed/analysis_ready_data.csv

    # Using generated mock data (default)
    python demo_synthetic_control.py
"""

import os
import sys

sys.path.append("src")

from pathlib import Path

import numpy as np
import pandas as pd

from mpower_mortality_causal_analysis.analysis import MPOWERAnalysisPipeline
from mpower_mortality_causal_analysis.causal_inference.methods.synthetic_control import (
    MPOWERSyntheticControl,
)


def create_realistic_mock_data(n_countries=50, n_years=11):
    """Create realistic mock MPOWER data for demonstration."""
    print("Creating realistic mock MPOWER data...")

    np.random.seed(42)
    countries = [f"Country_{i:02d}" for i in range(1, n_countries + 1)]
    years = list(range(2008, 2008 + n_years))

    # Define which countries get treated and when
    treated_countries = {
        "Country_05": 2010,  # Early adopter
        "Country_12": 2012,  # Mid adopter
        "Country_18": 2013,  # Mid adopter
        "Country_25": 2014,  # Late adopter
        "Country_33": 2015,  # Late adopter
        "Country_41": 2016,  # Very late adopter
    }

    data = []
    for country in countries:
        # Country-specific baseline characteristics
        baseline_mortality = np.random.normal(45, 10)  # Baseline lung cancer mortality
        gdp_level = np.random.normal(10, 1)  # Log GDP per capita
        urban_pct = np.random.normal(65, 15)  # Urban population %

        treatment_year = treated_countries.get(country, 0)

        for year in years:
            # Time trends
            time_trend = (year - 2008) * np.random.normal(0.5, 0.2)

            # Treatment effect (negative = mortality reduction)
            treated = country in treated_countries and year >= treatment_year
            treatment_effect = -3 * np.random.normal(1, 0.3) if treated else 0

            # Outcome with realistic noise
            lung_cancer_mortality = (
                baseline_mortality
                + time_trend
                + treatment_effect
                + np.random.normal(0, 2)
            )

            # Correlated outcomes
            cvd_mortality = lung_cancer_mortality * 2.5 + np.random.normal(0, 5)
            ihd_mortality = lung_cancer_mortality * 1.8 + np.random.normal(0, 4)
            copd_mortality = lung_cancer_mortality * 0.8 + np.random.normal(0, 3)

            # Control variables with some correlation
            gdp_per_capita_log = (
                gdp_level + (year - 2008) * 0.02 + np.random.normal(0, 0.1)
            )
            urban_population_pct = (
                urban_pct + (year - 2008) * 0.3 + np.random.normal(0, 1)
            )
            population_log = 16 + np.random.normal(0, 0.5)  # Log population
            education_expenditure_pct_gdp = 4.5 + np.random.normal(0, 0.8)

            data.append(
                {
                    "country": country,
                    "year": year,
                    "mpower_high_binary": int(treated),
                    "first_high_year": treatment_year if treated else 0,
                    "lung_cancer_mortality_rate": max(0, lung_cancer_mortality),
                    "cardiovascular_mortality_rate": max(0, cvd_mortality),
                    "ihd_mortality_rate": max(0, ihd_mortality),
                    "copd_mortality_rate": max(0, copd_mortality),
                    "gdp_per_capita_log": gdp_per_capita_log,
                    "urban_population_pct": max(0, min(100, urban_population_pct)),
                    "population_log": population_log,
                    "education_expenditure_pct_gdp": max(
                        0, education_expenditure_pct_gdp
                    ),
                }
            )

    df = pd.DataFrame(data)
    print(f"✓ Created mock data: {len(df)} observations")
    print(f"  Countries: {df['country'].nunique()}")
    print(f"  Years: {df['year'].min()}-{df['year'].max()}")
    print(f"  Treated countries: {len(treated_countries)}")

    return df


def demo_synthetic_control_direct():
    """Demonstrate direct use of MPOWERSyntheticControl class."""
    print("\n" + "=" * 60)
    print("DEMO 1: Direct Synthetic Control Usage")
    print("=" * 60)

    # Create mock data
    data = create_realistic_mock_data()

    # Extract treatment information
    treated_countries = data[data["mpower_high_binary"] == 1]["country"].unique()
    treatment_info = {}

    for country in treated_countries:
        country_data = data[data["country"] == country]
        first_treatment_year = country_data[country_data["mpower_high_binary"] == 1][
            "year"
        ].min()
        treatment_info[country] = int(first_treatment_year)

    print("\nTreatment Information:")
    for country, year in sorted(treatment_info.items()):
        print(f"  {country}: {year}")

    # Initialize synthetic control
    sc = MPOWERSyntheticControl(data=data)

    # Run analysis for lung cancer mortality
    print("\nRunning synthetic control analysis for lung cancer mortality...")

    results = sc.fit_all_units(
        treatment_info=treatment_info,
        outcome="lung_cancer_mortality_rate",
        predictors=["gdp_per_capita_log", "urban_population_pct", "population_log"],
        pre_periods=2,
    )

    # Display results
    print(f"\n{sc.summary()}")

    # Show individual country results
    print("\nIndividual Country Results:")
    for country in results["successful_units"][:3]:  # Show first 3
        unit_result = sc.get_unit_result(country)
        te = unit_result["treatment_effects"]["avg_treatment_effect"]
        rmse = unit_result["match_quality"]["rmse"]
        print(f"  {country}: Effect = {te:.3f}, Match RMSE = {rmse:.3f}")

    return results


def demo_pipeline_integration():
    """Demonstrate synthetic control integration with the full pipeline."""
    print("\n" + "=" * 60)
    print("DEMO 2: Pipeline Integration")
    print("=" * 60)

    # Create temporary data file
    data = create_realistic_mock_data()
    temp_data_path = "temp_mpower_data.csv"
    data.to_csv(temp_data_path, index=False)

    try:
        # Initialize pipeline
        print("\nInitializing MPOWER Analysis Pipeline...")
        pipeline = MPOWERAnalysisPipeline(temp_data_path)

        # Run just the synthetic control analysis
        print("\nRunning synthetic control analysis through pipeline...")
        sc_results = pipeline.run_synthetic_control_analysis()

        # Show summary results
        print("\nPipeline Results Summary:")
        for outcome in ["lung_cancer_mortality_rate", "cardiovascular_mortality_rate"]:
            if outcome in sc_results:
                result = sc_results[outcome]
                if "aggregated" in result:
                    agg = result["aggregated"]
                    avg_effect = agg.get("avg_treatment_effect", "N/A")
                    n_successful = len(result.get("successful_units", []))
                    n_total = len(result.get("treatment_info", {}))
                    print(f"  {outcome}:")
                    print(
                        f"    Average Effect: {avg_effect:.3f}"
                        if avg_effect != "N/A"
                        else f"    Average Effect: {avg_effect}"
                    )
                    print(f"    Successful Fits: {n_successful}/{n_total}")

    finally:
        # Clean up temporary file
        if os.path.exists(temp_data_path):
            os.remove(temp_data_path)


def demo_comparison_with_did():
    """Demonstrate comparison between synthetic control and Callaway & Sant'Anna DiD."""
    print("\n" + "=" * 60)
    print("DEMO 3: Synthetic Control vs. Callaway & Sant'Anna DiD")
    print("=" * 60)

    # Create data with known parallel trends violations
    print("\nCreating data with parallel trends violations...")
    data = create_realistic_mock_data()

    # Add differential pre-trends for treated countries (violating parallel trends)
    treated_countries = data[data["mpower_high_binary"] == 1]["country"].unique()
    for country in treated_countries:
        mask = (data["country"] == country) & (data["year"] < 2012)  # Pre-treatment
        # Add declining trend for treated countries before treatment
        pre_trend = -0.5 * (data.loc[mask, "year"] - 2008)
        data.loc[mask, "lung_cancer_mortality_rate"] += pre_trend

    temp_data_path = "temp_mpower_data_violation.csv"
    data.to_csv(temp_data_path, index=False)

    try:
        pipeline = MPOWERAnalysisPipeline(temp_data_path)

        # Run both methods
        print("\n1. Running Callaway & Sant'Anna DiD...")
        callaway_results = pipeline.run_callaway_did_analysis()

        print("\n2. Running Synthetic Control...")
        sc_results = pipeline.run_synthetic_control_analysis()

        # Compare results
        print("\nResults Comparison for Lung Cancer Mortality:")
        print(f"{'Method':<20} {'Effect':<10} {'Status':<15}")
        print("-" * 45)

        # Callaway results
        if "lung_cancer_mortality_rate" in callaway_results:
            callaway_result = callaway_results["lung_cancer_mortality_rate"]
            if "simple_att" in callaway_result:
                att = callaway_result["simple_att"]
                effect = att.get("att", "Error") if isinstance(att, dict) else "Error"
                status = "Success" if effect != "Error" else "Failed"
                print(f"{'Callaway DiD':<20} {effect:<10} {status:<15}")
            else:
                print(f"{'Callaway DiD':<20} {'Error':<10} {'Failed':<15}")

        # Synthetic Control results
        if "lung_cancer_mortality_rate" in sc_results:
            sc_result = sc_results["lung_cancer_mortality_rate"]
            if "aggregated" in sc_result:
                effect = sc_result["aggregated"].get("avg_treatment_effect", "Error")
                status = "Success" if effect != "Error" else "Failed"
                print(f"{'Synthetic Control':<20} {effect:<10.3f} {status:<15}")
            else:
                print(f"{'Synthetic Control':<20} {'Error':<10} {'Failed':<15}")

        print("\nKey Insight:")
        print("  - Synthetic control addresses parallel trends violations")
        print("  - by creating matched control units for each treated country")
        print("  - Results may differ from DiD when parallel trends assumption fails")

    finally:
        if os.path.exists(temp_data_path):
            os.remove(temp_data_path)


def main():
    """Run all demonstration scenarios."""
    print("🎯 MPOWER Synthetic Control Analysis Demonstration")
    print("=" * 70)
    print("\nThis demo shows how to use synthetic control methods to address")
    print("parallel trends violations in the MPOWER mortality causal analysis.")

    # Check if real data path is provided
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        if Path(data_path).exists():
            print(f"\n📊 Real data provided: {data_path}")
            print("Note: Using real data - some demos will be skipped")

            try:
                pipeline = MPOWERAnalysisPipeline(data_path)
                print("✓ Successfully loaded real MPOWER data")

                # Run synthetic control with real data
                print("\nRunning synthetic control analysis with real data...")
                sc_results = pipeline.run_synthetic_control_analysis()

                # Show real results
                print("\nReal Data Results:")
                for outcome in pipeline.outcomes[:2]:  # Show first 2 outcomes
                    if outcome in sc_results:
                        result = sc_results[outcome]
                        if "aggregated" in result:
                            agg = result["aggregated"]
                            avg_effect = agg.get("avg_treatment_effect")
                            if avg_effect is not None:
                                print(f"  {outcome}: {avg_effect:.4f}")

                return

            except Exception as e:
                print(f"Error loading real data: {e}")
                print("Falling back to mock data demos...")

    # Run mock data demonstrations
    try:
        demo_synthetic_control_direct()
        demo_pipeline_integration()
        demo_comparison_with_did()

        print("\n" + "=" * 70)
        print("✅ All demonstrations completed successfully!")
        print("\nNext Steps:")
        print("  1. Use synthetic control to address parallel trends violations")
        print("  2. Compare results with traditional DiD methods")
        print("  3. Examine match quality and unit weights for robustness")
        print("  4. Consider permutation tests for statistical inference")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
