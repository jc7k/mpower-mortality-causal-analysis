#!/usr/bin/env python3
"""Demonstration Script for MPOWER Mechanism Analysis.

This script demonstrates how to run the comprehensive mechanism analysis
to decompose MPOWER effects by individual components (M,P,O,W,E,R).

The mechanism analysis helps answer key policy questions:
1. Which MPOWER components drive mortality reductions?
2. What is the relative importance of different tobacco control policies?
3. How should policymakers prioritize MPOWER implementation?

Usage:
    python demo_mechanism_analysis.py [data_path] [output_dir]

Example:
    python demo_mechanism_analysis.py data/processed/analysis_ready_data.csv results/
"""

import sys

from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from mpower_mortality_causal_analysis.analysis import MPOWERAnalysisPipeline
from mpower_mortality_causal_analysis.causal_inference.utils.mechanism_analysis import (
    MPOWER_COMPONENTS,
    MPOWERMechanismAnalysis,
)


def create_demonstration_data() -> pd.DataFrame:
    """Create realistic demonstration data for mechanism analysis."""
    np.random.seed(42)  # For reproducibility

    print("Creating demonstration data with realistic MPOWER components...")

    # Create country-year panel
    countries = [f"Country_{i:02d}" for i in range(1, 31)]  # 30 countries
    years = [2008, 2010, 2012, 2014, 2016, 2018]  # 6 time periods

    data = []

    for country in countries:
        # Country-specific characteristics
        country_id = int(country.split("_")[1])
        is_developed = country_id <= 15  # First 15 are "developed"
        base_mortality = 45 if is_developed else 65

        # Country-specific MPOWER progression
        country_mpower_start = {}
        country_improvement_rates = {}

        for component, info in MPOWER_COMPONENTS.items():
            # Starting score (developed countries start higher)
            start_score = np.random.randint(
                1 if is_developed else 0,
                info["max_score"] // 2 + (2 if is_developed else 1),
            )
            country_mpower_start[component] = start_score

            # Improvement rate (some countries improve faster)
            improvement_rate = 0.15 if is_developed else 0.10
            if np.random.random() < 0.3:  # 30% are "fast improvers"
                improvement_rate *= 2
            country_improvement_rates[component] = improvement_rate

        # Generate yearly data
        for year in years:
            year_index = years.index(year)

            # MPOWER component evolution
            mpower_scores = {}
            total_mpower = 0

            for component, info in MPOWER_COMPONENTS.items():
                base_score = country_mpower_start[component]
                improvement_rate = country_improvement_rates[component]

                # Progressive improvement over time
                max_improvement = year_index * improvement_rate * info["max_score"]
                current_score = min(
                    base_score + np.random.poisson(max_improvement), info["max_score"]
                )

                mpower_scores[f"mpower_{component.lower()}_score"] = current_score
                total_mpower += current_score

            # Create realistic treatment variables
            total_possible = sum(
                info["max_score"] for info in MPOWER_COMPONENTS.values()
            )
            mpower_total_score = total_mpower
            mpower_high_binary = 1 if mpower_total_score >= 25 else 0

            # Mortality outcomes (negatively related to MPOWER with realistic noise)
            mpower_effect = -0.8 * mpower_total_score
            time_trend = -1.2 * year_index  # General improvement over time
            country_effect = np.random.normal(0, 8)  # Country fixed effect
            noise = np.random.normal(0, 6)

            base_mortality_adjusted = (
                base_mortality + mpower_effect + time_trend + country_effect + noise
            )

            # Different outcomes with correlated but distinct patterns
            lung_cancer = max(base_mortality_adjusted * 0.3 + np.random.normal(0, 3), 5)
            cardiovascular = max(
                base_mortality_adjusted * 1.8 + np.random.normal(0, 8), 15
            )
            ihd = max(base_mortality_adjusted * 1.2 + np.random.normal(0, 6), 10)
            copd = max(base_mortality_adjusted * 0.9 + np.random.normal(0, 5), 8)

            # Control variables
            gdp_base = 9.5 if is_developed else 7.8
            gdp_growth = 0.02 * year_index + np.random.normal(0, 0.1)

            urban_base = 75 if is_developed else 45
            urban_growth = 1.5 * year_index + np.random.normal(0, 2)

            pop_base = 15.5 + np.random.normal(0, 1.5)

            education_base = 5.2 if is_developed else 3.8
            education_trend = 0.1 * year_index + np.random.normal(0, 0.3)

            # Create first treatment year (for countries that reach high MPOWER)
            # This will be calculated properly by the analysis pipeline
            first_high_year = year if mpower_high_binary == 1 else 0

            # Compile row
            row = {
                "country": country,
                "year": year,
                "lung_cancer_mortality_rate": lung_cancer,
                "cardiovascular_mortality_rate": cardiovascular,
                "ihd_mortality_rate": ihd,
                "copd_mortality_rate": copd,
                "mpower_total_score": mpower_total_score,
                "mpower_high_binary": mpower_high_binary,
                "first_high_year": first_high_year,
                "gdp_per_capita_log": gdp_base + gdp_growth,
                "urban_population_pct": min(max(urban_base + urban_growth, 10), 95),
                "population_log": pop_base,
                "education_expenditure_pct_gdp": max(
                    education_base + education_trend, 1
                ),
                **mpower_scores,
            }

            data.append(row)

    df = pd.DataFrame(data)

    # Fix first_high_year calculation (sustained treatment)
    def get_first_sustained_high(group):
        high_years = group[group["mpower_high_binary"] == 1]["year"].tolist()
        if len(high_years) >= 2:  # Require at least 2 consecutive periods
            for i in range(len(high_years) - 1):
                if (
                    high_years[i + 1] == high_years[i] + 2
                ):  # Consecutive biennial periods
                    return high_years[i]
        return 0

    first_high = df.groupby("country").apply(get_first_sustained_high).reset_index()
    first_high.columns = ["country", "first_high_year_corrected"]

    df = df.merge(first_high, on="country")
    df["first_high_year"] = df["first_high_year_corrected"]
    df = df.drop("first_high_year_corrected", axis=1)

    print(f"Created demonstration data: {len(df)} observations")
    print(f"Countries: {df['country'].nunique()}")
    print(f"Years: {sorted(df['year'].unique())}")
    print(f"Treated countries: {df[df['first_high_year'] > 0]['country'].nunique()}")

    return df


def run_mechanism_analysis_demo(data_path=None, output_dir="results/mechanism_demo"):
    """Run comprehensive mechanism analysis demonstration."""
    print("=" * 80)
    print("MPOWER MECHANISM ANALYSIS DEMONSTRATION")
    print("=" * 80)
    print("Analyzing which MPOWER components drive mortality reductions")

    # Load or create data
    if data_path and Path(data_path).exists():
        print(f"\nLoading data from: {data_path}")
        data = pd.read_csv(data_path)
    else:
        print("\nCreating demonstration data...")
        data = create_demonstration_data()

        # Save demonstration data
        demo_data_path = Path(output_dir) / "demonstration_data.csv"
        demo_data_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(demo_data_path, index=False)
        print(f"Demonstration data saved to: {demo_data_path}")

    print("\nData summary:")
    print(f"- Observations: {len(data)}")
    print(f"- Countries: {data['country'].nunique()}")
    print(f"- Time range: {data['year'].min()}-{data['year'].max()}")

    # Initialize analysis pipeline
    print("\nInitializing MPOWER Analysis Pipeline...")

    # Create temporary data file if needed
    if data_path is None:
        temp_data_path = Path(output_dir) / "temp_analysis_data.csv"
        data.to_csv(temp_data_path, index=False)
        data_path = temp_data_path

    pipeline = MPOWERAnalysisPipeline(
        data_path=data_path,
        outcomes=[
            "lung_cancer_mortality_rate",
            "cardiovascular_mortality_rate",
        ],  # Focus on 2 outcomes for demo
        control_vars=[
            "gdp_per_capita_log",
            "urban_population_pct",
        ],
    )

    # Run mechanism analysis
    print("\n" + "-" * 60)
    print("RUNNING MECHANISM ANALYSIS")
    print("-" * 60)

    try:
        mechanism_results = pipeline.run_mechanism_analysis(
            methods=["callaway_did"],  # Use only one method for demo
        )

        print("\n✅ Mechanism analysis completed successfully!")

        # Display key findings
        print("\n" + "=" * 60)
        print("KEY FINDINGS SUMMARY")
        print("=" * 60)

        for outcome, outcome_results in mechanism_results.items():
            print(f"\n📊 OUTCOME: {outcome.replace('_', ' ').title()}")
            print("-" * 50)

            if "summary" in outcome_results:
                summary = outcome_results["summary"]

                # Treatment coverage
                coverage = summary.get("treatment_coverage", {})
                if coverage:
                    print("\n🎯 Treatment Coverage by Component:")
                    for comp, cov_info in coverage.items():
                        comp_name = MPOWER_COMPONENTS.get(comp, {}).get("name", comp)
                        countries = cov_info["countries"]
                        rate = cov_info["rate"] * 100
                        print(
                            f"  • {comp} ({comp_name}): {countries} countries ({rate:.1f}%)"
                        )

                # Effect rankings
                rankings = summary.get("policy_rankings", {})
                for method, ranking_list in rankings.items():
                    if ranking_list:
                        print(
                            f"\n🏆 Policy Effectiveness Ranking ({method.replace('_', ' ').title()}):"
                        )
                        for i, item in enumerate(ranking_list[:4], 1):  # Top 4
                            comp = item["component"]
                            name = item["component_name"]
                            effect = item["effect"]
                            print(f"  {i}. {comp} ({name}): {effect:.3f} effect")

        # Export results
        print("\n" + "-" * 60)
        print("EXPORTING RESULTS")
        print("-" * 60)

        pipeline.export_results(output_dir)
        print(f"✅ Results exported to: {output_dir}")
        print("📁 Key files:")
        print("  • analysis_results.json - Complete results")
        print(
            "  • analysis_summary.xlsx - Summary tables (including Mechanism_Analysis sheet)"
        )

        # Also export dedicated mechanism analysis
        if mechanism_results:
            mechanism = MPOWERMechanismAnalysis(
                data=data,
                control_vars=pipeline.control_vars,
            )

            # Export dedicated mechanism results
            mechanism.export_mechanism_results(
                results=mechanism_results[list(mechanism_results.keys())[0]],
                output_dir=f"{output_dir}/detailed_mechanism",
            )
            print("  • detailed_mechanism/ - Detailed mechanism analysis results")

    except Exception as e:
        print(f"\n❌ Error in mechanism analysis: {str(e)}")
        print("This might be due to missing dependencies (R packages)")
        print(
            "The framework is implemented and ready - install R dependencies for full functionality"
        )
        return False

    # Interpretation guide
    print("\n" + "=" * 60)
    print("INTERPRETATION GUIDE")
    print("=" * 60)
    print("""
📖 How to Interpret Results:

1. TREATMENT COVERAGE: Shows how many countries implemented each component
   • Higher coverage = more common policy implementation
   • Low coverage might indicate implementation barriers

2. EFFECT RANKINGS: Shows which components have largest mortality impact
   • Negative effects = mortality reduction (good)
   • Larger absolute effects = more important for policy
   • Rankings help prioritize implementation

3. STATISTICAL SIGNIFICANCE: P-values indicate confidence in estimates
   • p < 0.05 = statistically significant effect
   • p < 0.01 = highly significant effect

4. POLICY IMPLICATIONS:
   • Top-ranked components should be implementation priorities
   • High-coverage + high-effect = successful widespread policies
   • Low-coverage + high-effect = expansion opportunities
   • High-coverage + low-effect = reconsider implementation approaches
""")

    return True


def main():
    """Main function for demonstration script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Demonstrate MPOWER mechanism analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "data_path",
        nargs="?",
        help="Path to analysis-ready data (optional - will create demo data if not provided)",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="results/mechanism_demo",
        help="Output directory for results (default: results/mechanism_demo)",
    )
    parser.add_argument(
        "--create-data-only",
        action="store_true",
        help="Only create demonstration data without running analysis",
    )

    args = parser.parse_args()

    if args.create_data_only:
        print("Creating demonstration data only...")
        data = create_demonstration_data()
        output_path = Path(args.output_dir) / "demonstration_data.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(output_path, index=False)
        print(f"Demonstration data saved to: {output_path}")
        return

    # Run full demonstration
    success = run_mechanism_analysis_demo(
        data_path=args.data_path, output_dir=args.output_dir
    )

    if success:
        print("\n🎉 Mechanism analysis demonstration completed successfully!")
        print(f"📁 Check {args.output_dir} for detailed results")
    else:
        print("\n⚠️ Demonstration completed with limitations")
        print("Install R and the 'did' package for full functionality")


if __name__ == "__main__":
    main()
