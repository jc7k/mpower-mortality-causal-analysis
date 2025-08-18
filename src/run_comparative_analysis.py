#!/usr/bin/env python3
"""Demonstration Script for Comprehensive Comparative Analysis.

This script demonstrates how to run the complete comparative analysis framework
across all four research extensions, generating unprecedented methodological
comparison for tobacco control policy evaluation.

Usage:
    python src/run_comparative_analysis.py [data_path] [output_dir]

Example:
    python src/run_comparative_analysis.py data/processed/analysis_ready_data.csv results/comparative_analysis/
"""

import sys
import time
import logging
from pathlib import Path

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from mpower_mortality_causal_analysis.comparative_analysis import (
    ComparativeAnalysisPipeline,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Run comprehensive comparative analysis demonstration."""
    # Default paths
    default_data_path = (
        Path(__file__).parent.parent / "data" / "processed" / "analysis_ready_data.csv"
    )
    default_output_dir = (
        Path(__file__).parent.parent / "results" / "comparative_analysis"
    )

    # Parse command line arguments
    if len(sys.argv) >= 2:
        data_path = Path(sys.argv[1])
    else:
        data_path = default_data_path

    if len(sys.argv) >= 3:
        output_dir = Path(sys.argv[2])
    else:
        output_dir = default_output_dir

    # Validate data path
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        logger.info("Please ensure the analysis-ready dataset exists.")
        logger.info("You may need to run data preparation scripts first.")
        return 1

    logger.info("=" * 80)
    logger.info("MPOWER COMPARATIVE ANALYSIS DEMONSTRATION")
    logger.info("=" * 80)
    logger.info(f"Data path: {data_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")

    try:
        # Initialize comparative analysis pipeline
        logger.info("Initializing Comparative Analysis Pipeline...")
        start_time = time.time()

        pipeline = ComparativeAnalysisPipeline(
            data_path=data_path,
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
            control_vars=["gdp_pc_constant_log", "urban_pop_pct", "edu_exp_pct_gdp"],
        )

        init_time = time.time() - start_time
        logger.info(f"Pipeline initialized successfully in {init_time:.2f} seconds")
        logger.info("")

        # Run comprehensive analysis
        logger.info("Starting Comprehensive Comparative Analysis...")
        logger.info("This analysis includes:")
        logger.info(
            "  1. Core causal inference (Callaway & Sant'Anna, Synthetic Control)"
        )
        logger.info(
            "  2. Advanced DiD methods (Sun & Abraham, Borusyak, DCDH, Doubly Robust)"
        )
        logger.info(
            "  3. Cost-effectiveness analysis (Health economics, ICERs, budget optimization)"
        )
        logger.info("  4. Spillover analysis (Spatial econometrics, network diffusion)")
        logger.info(
            "  5. Policy optimization (Component interactions, sequential implementation)"
        )
        logger.info("")

        analysis_start = time.time()

        # Run with error handling for individual extensions
        results = pipeline.run_comprehensive_analysis(
            skip_advanced_did=False,  # Include advanced DiD methods
            skip_cost_effectiveness=False,  # Include cost-effectiveness analysis
            skip_spillover=False,  # Include spillover analysis
            skip_optimization=False,  # Include policy optimization
        )

        analysis_time = time.time() - analysis_start
        logger.info(f"Comprehensive analysis completed in {analysis_time:.2f} seconds")
        logger.info("")

        # Export comprehensive report
        logger.info("Exporting Comprehensive Report...")
        export_start = time.time()

        pipeline.export_comparative_report(
            output_dir=output_dir, format="comprehensive"
        )

        export_time = time.time() - export_start
        logger.info(f"Report exported successfully in {export_time:.2f} seconds")
        logger.info("")

        # Analysis summary
        total_time = time.time() - start_time
        logger.info("=" * 80)
        logger.info("ANALYSIS COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Total execution time: {total_time:.2f} seconds")
        logger.info("")

        # Results summary
        logger.info("Results Summary:")
        methods_executed = results.get("method_comparison", {}).get(
            "methods_executed", []
        )
        logger.info(f"  • Methods executed: {len(methods_executed)}")
        for method in methods_executed:
            logger.info(f"    - {method.replace('_', ' ').title()}")

        logger.info(f"  • Mortality outcomes analyzed: {len(pipeline.outcomes)}")
        logger.info(
            f"  • Countries in analysis: {pipeline.data[pipeline.unit_col].nunique()}"
        )
        logger.info(
            f"  • Time periods: {pipeline.data[pipeline.time_col].min()}-{pipeline.data[pipeline.time_col].max()}"
        )
        logger.info("")

        # Output files
        logger.info("Output Files Generated:")
        logger.info(f"  • Main results: {output_dir}/comparative_analysis_results.json")
        logger.info(
            f"  • Method comparison: {output_dir}/method_comparison_summary.xlsx"
        )
        logger.info(f"  • Academic tables: {output_dir}/academic_tables/")
        logger.info(f"  • Visualizations: {output_dir}/visualizations/")
        logger.info(f"  • Policy brief: {output_dir}/policy_brief.md")
        logger.info("")

        # Method comparison insights
        if "method_agreement" in results.get("method_comparison", {}):
            logger.info("Method Agreement Analysis:")
            for outcome, stats in results["method_comparison"][
                "method_agreement"
            ].items():
                cv = stats.get("coefficient_of_variation", float("nan"))
                if cv == cv:  # Check for NaN without importing pandas
                    agreement_level = (
                        "High" if cv < 0.2 else "Moderate" if cv < 0.5 else "Low"
                    )
                    logger.info(
                        f"  • {outcome}: {agreement_level} agreement (CV = {cv:.3f})"
                    )
            logger.info("")

        # Policy implications
        logger.info("Key Policy Insights:")
        logger.info(
            "  • Multiple methodological approaches provide robust causal evidence"
        )
        logger.info(
            "  • Component-specific analysis enables targeted policy implementation"
        )
        logger.info(
            "  • Cost-effectiveness analysis guides resource allocation priorities"
        )
        logger.info("  • Spillover analysis reveals cross-country policy externalities")
        logger.info("  • Optimization framework provides implementation roadmaps")
        logger.info("")

        logger.info("Next Steps:")
        logger.info("  1. Review detailed results in output files")
        logger.info("  2. Examine method comparison and agreement statistics")
        logger.info("  3. Use policy brief for stakeholder engagement")
        logger.info(
            "  4. Consider academic publication based on comprehensive findings"
        )
        logger.info("")

        return 0

    except Exception as e:
        logger.error(f"Analysis failed with error: {e}")
        logger.error("Please check the error message and data requirements.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
