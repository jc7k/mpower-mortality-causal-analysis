#!/usr/bin/env python3
"""Test the fixed event study analysis."""

import sys

sys.path.insert(0, "src")


def test_event_study_fix():
    """Test if the event study analysis works with the dtype fix."""
    # Import after setting up the path
    from mpower_mortality_causal_analysis.analysis import MPOWERAnalysisPipeline

    # Initialize pipeline
    pipeline = MPOWERAnalysisPipeline(
        data_path="/home/user/projects/mpower-mortality-causal-analysis/data/processed/analysis_ready_data.csv",
        outcomes=["mort_lung_cancer_asr"],
        treatment_col="ever_treated",
        treatment_year_col="treatment_cohort",
        unit_col="country_name",
        time_col="year",
        control_vars=["gdp_pc_constant_log", "urban_pop_pct"],
    )

    print("Testing single outcome parallel trends analysis...")

    try:
        # Test just one outcome
        parallel_trends_results = pipeline.run_parallel_trends_analysis()
        print("✓ Parallel trends analysis completed successfully!")

        for outcome, result in parallel_trends_results.items():
            conclusion = result["parallel_trends_tests"]["conclusion"]
            print(f"  - {outcome}: {conclusion}")

        return True

    except Exception as e:
        print(f"✗ Parallel trends analysis failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_event_study_fix()
    if success:
        print("\n🎉 Event study fix successful!")
    else:
        print("\n❌ Event study still has issues.")
