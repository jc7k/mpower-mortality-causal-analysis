#!/usr/bin/env python3
"""MPOWER Mortality Causal Analysis - Full Pipeline
Implements Callaway & Sant'Anna (2021) staggered difference-in-differences
and comprehensive robustness checks.
"""

import sys

sys.path.insert(0, "src")

import logging
import warnings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("analysis.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def run_full_analysis():
    """Run the complete MPOWER causal analysis pipeline."""
    try:
        # Import after setting up the path
        from mpower_mortality_causal_analysis.analysis import MPOWERAnalysisPipeline

        logger.info("=" * 60)
        logger.info("MPOWER MORTALITY CAUSAL ANALYSIS")
        logger.info("Callaway & Sant'Anna (2021) Staggered Difference-in-Differences")
        logger.info("=" * 60)

        # Initialize pipeline with correct column mappings
        logger.info("Initializing analysis pipeline...")
        pipeline = MPOWERAnalysisPipeline(
            data_path="/home/user/projects/mpower-mortality-causal-analysis/data/processed/analysis_ready_data.csv",
            outcomes=[
                "mort_lung_cancer_asr",
                "mort_cvd_asr",
                "mort_ihd_asr",
                "mort_copd_asr",
            ],
            treatment_col="ever_treated",
            treatment_year_col="treatment_cohort",
            unit_col="country_name",
            time_col="year",
            control_vars=["gdp_pc_constant_log", "urban_pop_pct", "population_total"],
        )

        logger.info(
            f"Data loaded: {pipeline.data.shape[0]} observations, {pipeline.data.shape[1]} variables"
        )
        logger.info(f"Countries: {pipeline.data[pipeline.unit_col].nunique()}")
        logger.info(
            f"Time period: {pipeline.data[pipeline.time_col].min()}-{pipeline.data[pipeline.time_col].max()}"
        )

        # Step 1: Descriptive Analysis
        logger.info("\n" + "=" * 40)
        logger.info("STEP 1: DESCRIPTIVE ANALYSIS")
        logger.info("=" * 40)

        descriptive_results = pipeline.run_descriptive_analysis()
        logger.info("✓ Descriptive analysis completed")
        logger.info("  - Treatment adoption timeline created")
        logger.info("  - Outcome trends by cohort plotted")
        logger.info("  - Treatment balance checked")

        # Step 2: Parallel Trends Testing
        logger.info("\n" + "=" * 40)
        logger.info("STEP 2: PARALLEL TRENDS TESTING")
        logger.info("=" * 40)

        parallel_trends_results = pipeline.run_parallel_trends_analysis()
        logger.info("✓ Parallel trends testing completed")

        for outcome, result in parallel_trends_results.items():
            conclusion = result["parallel_trends_tests"]["conclusion"]
            logger.info(f"  - {outcome}: {conclusion}")

        # Step 3: Main Causal Analysis (Callaway & Sant'Anna)
        logger.info("\n" + "=" * 40)
        logger.info("STEP 3: CALLAWAY & SANT'ANNA ANALYSIS")
        logger.info("=" * 40)

        callaway_results = pipeline.run_callaway_did_analysis()
        logger.info("✓ Callaway & Sant'Anna analysis completed")

        for outcome, result in callaway_results.items():
            att = result.get("overall_att", "N/A")
            logger.info(f"  - {outcome}: Overall ATT = {att}")

        # Step 4: Event Study Analysis
        logger.info("\n" + "=" * 40)
        logger.info("STEP 4: EVENT STUDY ANALYSIS")
        logger.info("=" * 40)

        event_study_results = pipeline.run_event_study_analysis()
        logger.info("✓ Event study analysis completed")
        logger.info("  - Dynamic treatment effects estimated")
        logger.info("  - Event study plots generated")

        # Step 5: Robustness Checks
        logger.info("\n" + "=" * 40)
        logger.info("STEP 5: ROBUSTNESS CHECKS")
        logger.info("=" * 40)

        robustness_results = pipeline.run_robustness_checks()
        logger.info("✓ Robustness checks completed")
        logger.info("  - Alternative estimators tested")
        logger.info("  - Sample robustness verified")
        logger.info("  - Placebo tests conducted")

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("ANALYSIS COMPLETE!")
        logger.info("=" * 60)
        logger.info("Results saved to:")
        logger.info("  - results/descriptive/")
        logger.info("  - results/callaway_did/")
        logger.info("  - results/event_study/")
        logger.info("  - results/robustness/")
        logger.info("  - results/summary_report.md")

        return {
            "descriptive": descriptive_results,
            "parallel_trends": parallel_trends_results,
            "callaway": callaway_results,
            "event_study": event_study_results,
            "robustness": robustness_results,
        }

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return None


if __name__ == "__main__":
    # Suppress non-critical warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*rpy2.*")
    warnings.filterwarnings("ignore", message=".*differences.*")

    results = run_full_analysis()

    if results:
        print("\n🎉 MPOWER causal analysis completed successfully!")
        print("Check analysis.log for detailed progress and results/ for outputs.")
    else:
        print("\n❌ Analysis failed. Check analysis.log for details.")
        sys.exit(1)
