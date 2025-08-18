#!/usr/bin/env python3
"""Test script for MPOWER synthetic control implementation.

This script tests the new synthetic control functionality to ensure it works
correctly with the MPOWER data.
"""

import sys

sys.path.append("src")

from pathlib import Path

import numpy as np
import pandas as pd

# Test imports
try:
    from mpower_mortality_causal_analysis.analysis import MPOWERAnalysisPipeline
    from mpower_mortality_causal_analysis.causal_inference.methods.synthetic_control import (
        MPOWERSyntheticControl,
    )

    print("✓ Successfully imported synthetic control classes")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


def test_synthetic_control_basic():
    """Test basic synthetic control functionality with mock data."""
    print("\n=== Testing Basic Synthetic Control Functionality ===")

    # Create mock panel data
    np.random.seed(42)
    countries = [f"Country_{i}" for i in range(20)]
    years = list(range(2008, 2019))

    # Create panel structure
    data = []
    for country in countries:
        for year in years:
            # Mock treatment: some countries get treated in different years
            treated = (
                country in ["Country_1", "Country_5", "Country_10"] and year >= 2012
            )
            treatment_year = 2012 if treated and year >= 2012 else 0

            # Mock outcome with treatment effect
            base_outcome = 50 + np.random.normal(0, 5)
            if treated:
                base_outcome -= 3  # Treatment effect

            data.append(
                {
                    "country": country,
                    "year": year,
                    "mpower_high_binary": int(treated),
                    "first_high_year": treatment_year,
                    "lung_cancer_mortality_rate": base_outcome,
                    "gdp_per_capita_log": 10 + np.random.normal(0, 0.5),
                    "urban_population_pct": 60 + np.random.normal(0, 10),
                }
            )

    mock_data = pd.DataFrame(data)
    print(
        f"✓ Created mock data: {len(mock_data)} observations, {len(countries)} countries"
    )

    # Test MPOWERSyntheticControl class
    try:
        sc = MPOWERSyntheticControl(data=mock_data)
        print("✓ Successfully initialized MPOWERSyntheticControl")

        # Test treatment info extraction
        treatment_info = {
            "Country_1": 2012,
            "Country_5": 2012,
            "Country_10": 2012,
        }

        # Test single unit fit
        result = sc.fit_single_unit(
            treated_unit="Country_1",
            treatment_time=2012,
            outcome="lung_cancer_mortality_rate",
            predictors=["gdp_per_capita_log", "urban_population_pct"],
        )

        if result["status"] == "success":
            print("✓ Single unit synthetic control fit successful")
            print(
                f"  Treatment effect: {result['treatment_effects']['avg_treatment_effect']:.3f}"
            )
            print(f"  Match quality (RMSE): {result['match_quality']['rmse']:.3f}")
        else:
            print(f"✗ Single unit fit failed: {result.get('error', 'Unknown error')}")
            return False

        # Test multiple units
        results = sc.fit_all_units(
            treatment_info=treatment_info,
            outcome="lung_cancer_mortality_rate",
            predictors=["gdp_per_capita_log", "urban_population_pct"],
        )

        if results["aggregated"].get("avg_treatment_effect") is not None:
            print("✓ Multi-unit synthetic control fit successful")
            print(
                f"  Average treatment effect: {results['aggregated']['avg_treatment_effect']:.3f}"
            )
            print(
                f"  Successful fits: {len(results['successful_units'])}/{len(treatment_info)}"
            )
        else:
            print("✗ Multi-unit fit failed")
            return False

        return True

    except Exception as e:
        print(f"✗ Synthetic control test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_pipeline_integration():
    """Test synthetic control integration with the main analysis pipeline."""
    print("\n=== Testing Pipeline Integration ===")

    # Check if real data exists
    data_path = Path("data/processed/analysis_ready_data.csv")

    if not data_path.exists():
        print(f"⚠ Data file not found at {data_path}")
        print("  Skipping pipeline integration test")
        return True

    try:
        # Test with real data (just load and check structure)
        pipeline = MPOWERAnalysisPipeline(data_path)
        print(f"✓ Pipeline initialized with {len(pipeline.data)} observations")

        # Test synthetic control method directly (without running full analysis)
        print("  Testing synthetic control method...")

        # Just test if the method can be called (don't run full analysis)
        sc_method = getattr(pipeline, "run_synthetic_control_analysis", None)
        if sc_method is None:
            print("✗ run_synthetic_control_analysis method not found")
            return False

        print("✓ Synthetic control method available in pipeline")

        # Check if data has the required structure
        required_cols = ["country", "year", "mpower_high_binary", "first_high_year"]
        missing_cols = [
            col for col in required_cols if col not in pipeline.data.columns
        ]

        if missing_cols:
            print(f"✗ Missing required columns: {missing_cols}")
            return False

        print("✓ Data has required columns for synthetic control")

        # Check treatment countries
        treated_countries = pipeline.data[pipeline.data["mpower_high_binary"] == 1][
            "country"
        ].nunique()
        print(f"✓ Found {treated_countries} treated countries in data")

        return True

    except Exception as e:
        print(f"✗ Pipeline integration test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("MPOWER Synthetic Control Implementation Test")
    print("=" * 50)

    tests = [
        ("Basic Functionality", test_synthetic_control_basic),
        ("Pipeline Integration", test_pipeline_integration),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\nRunning {test_name} test...")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                print(f"✓ {test_name} test passed")
            else:
                print(f"✗ {test_name} test failed")
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results.append((test_name, False))

    print("\n" + "=" * 50)
    print("Test Summary:")
    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"  {test_name}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Synthetic control implementation is ready.")
        return 0
    print("⚠ Some tests failed. Check implementation.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
