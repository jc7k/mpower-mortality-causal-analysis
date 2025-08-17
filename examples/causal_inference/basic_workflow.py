"""Basic Workflow Example for MPOWER Causal Inference Analysis.

This example demonstrates a complete workflow for analyzing the causal impact
of WHO MPOWER tobacco control policies on mortality outcomes using the
causal inference framework.
"""

import numpy as np
import pandas as pd

# Import the causal inference framework
from mpower_mortality_causal_analysis.causal_inference import (
    CallawayDiD,
    EventStudyAnalysis,
    MPOWERDataPrep,
    PanelFixedEffects,
    RobustnessTests,
    SyntheticControl,
)


def create_sample_data() -> pd.DataFrame:
    """Create sample MPOWER data for demonstration purposes.

    Returns:
        DataFrame with sample MPOWER data
    """
    # Set random seed for reproducibility
    np.random.seed(42)

    # Define countries and years
    countries = [
        "Argentina",
        "Brazil",
        "Chile",
        "Colombia",
        "Mexico",
        "Uruguay",
        "Australia",
        "Canada",
        "France",
        "Germany",
        "Italy",
        "Spain",
        "UK",
        "USA",
        "China",
        "India",
        "Japan",
        "South Korea",
        "Thailand",
        "Turkey",
    ]
    years = list(range(2008, 2020))

    # Create panel structure
    data = []
    for country in countries:
        for year in years:
            data.append({"country": country, "year": year})

    df = pd.DataFrame(data)

    # Create MPOWER scores (0-37 scale, higher = stronger policies)
    # Some countries adopt strong policies earlier than others
    treatment_countries = ["Uruguay", "Australia", "UK", "Brazil", "Turkey"]
    treatment_years = {
        "Uruguay": 2010,
        "Australia": 2012,
        "UK": 2011,
        "Brazil": 2014,
        "Turkey": 2013,
    }

    df["mpower_total_score"] = 0

    for _, row in df.iterrows():
        country = row["country"]
        year = row["year"]

        if country in treatment_countries:
            treatment_year = treatment_years[country]
            if year < treatment_year:
                # Pre-treatment: gradually increasing score
                base_score = 10 + np.random.normal(2, 1)
                df.loc[
                    (df["country"] == country) & (df["year"] == year),
                    "mpower_total_score",
                ] = base_score
            else:
                # Post-treatment: high score with some variation
                high_score = 28 + np.random.normal(3, 2)
                df.loc[
                    (df["country"] == country) & (df["year"] == year),
                    "mpower_total_score",
                ] = high_score
        else:
            # Never-treated or late-treated countries
            base_score = (
                8 + np.random.normal(3, 2) + 0.5 * (year - 2008)
            )  # Gradual improvement
            df.loc[
                (df["country"] == country) & (df["year"] == year), "mpower_total_score"
            ] = base_score

    # Ensure scores are within reasonable bounds
    df["mpower_total_score"] = df["mpower_total_score"].clip(0, 37)

    # Create individual MPOWER components (6 components, each 0-6 scale)
    component_names = [
        "M_monitor",
        "P_protect",
        "O_offer",
        "W_warn",
        "E_enforce",
        "R_raise",
    ]

    for component in component_names:
        # Distribute total score across components with some randomness
        df[component] = (df["mpower_total_score"] / 6) + np.random.normal(
            0, 0.5, len(df)
        )
        df[component] = df[component].clip(0, 6)

    # Create mortality outcome (deaths per 100,000)
    # Treatment should reduce mortality with some lag
    df["mortality_rate"] = 0

    for country in countries:
        country_data = df[df["country"] == country].copy()

        # Base mortality rate with country-specific effects
        base_rate = 50 + np.random.normal(0, 10)  # Country fixed effect

        for i, (_, row) in enumerate(country_data.iterrows()):
            year = row["year"]
            mpower_score = row["mpower_total_score"]

            # Time trend (slight decrease over time)
            time_effect = -0.5 * (year - 2008)

            # Treatment effect (stronger MPOWER policies reduce mortality)
            if country in treatment_countries:
                treatment_year = treatment_years[country]
                if year >= treatment_year + 1:  # 1-year lag
                    treatment_effect = -0.3 * (
                        mpower_score - 15
                    )  # Effect proportional to policy strength
                else:
                    treatment_effect = 0
            else:
                treatment_effect = -0.1 * max(
                    0, mpower_score - 15
                )  # Smaller effect for gradual adopters

            # Random noise
            noise = np.random.normal(0, 3)

            mortality_rate = base_rate + time_effect + treatment_effect + noise
            df.loc[
                (df["country"] == country) & (df["year"] == year), "mortality_rate"
            ] = max(0, mortality_rate)

    # Add control variables
    df["gdp_per_capita"] = 20000 + np.random.normal(5000, 2000, len(df))
    df["gdp_log"] = np.log(df["gdp_per_capita"])
    df["urban_percentage"] = 60 + np.random.normal(15, 5, len(df))
    df["healthcare_spending"] = 1000 + np.random.normal(300, 100, len(df))

    # Add some correlation between controls and both treatment and outcome
    # Countries with higher GDP more likely to have strong tobacco policies
    for country in countries:
        country_mask = df["country"] == country
        if country in treatment_countries:
            df.loc[country_mask, "gdp_per_capita"] *= 1.2
            df.loc[country_mask, "healthcare_spending"] *= 1.3

    df["gdp_log"] = np.log(df["gdp_per_capita"])

    return df


def run_basic_workflow():
    """Run the complete basic workflow for MPOWER causal analysis."""
    print("=" * 80)
    print("MPOWER CAUSAL INFERENCE ANALYSIS - BASIC WORKFLOW")
    print("=" * 80)

    # Step 1: Create or load data
    print("\n1. Data Preparation")
    print("-" * 50)

    # For this example, create sample data
    # In practice, you would load your actual MPOWER data
    data = create_sample_data()
    print(
        f"Loaded data: {len(data)} observations, {data['country'].nunique()} countries"
    )
    print(f"Time period: {data['year'].min()}-{data['year'].max()}")

    # Step 2: Prepare data for analysis
    print("\n2. Data Preprocessing")
    print("-" * 50)

    prep = MPOWERDataPrep(data, country_col="country", year_col="year")

    # Create treatment cohorts based on MPOWER score threshold
    data_with_cohorts = prep.create_treatment_cohorts(
        mpower_col="mpower_total_score",
        treatment_definition="binary_threshold",
        threshold=25.0,
        min_years_high=2,
    )

    print("Treatment cohorts created:")
    cohort_summary = data_with_cohorts.groupby("treatment_cohort")["country"].nunique()
    for cohort, n_countries in cohort_summary.items():
        if cohort == 0:
            print(f"  Never treated: {n_countries} countries")
        else:
            print(f"  Treated in {cohort}: {n_countries} countries")

    # Balance panel and prepare for analysis
    balanced_data = prep.balance_panel(method="drop_unbalanced", min_years=10)
    analysis_data = prep.prepare_for_analysis(
        outcome_cols=["mortality_rate"],
        control_cols=["gdp_log", "urban_percentage"],
        log_transform=["healthcare_spending"],
    )

    print(f"Final analysis dataset: {len(analysis_data)} observations")

    # Step 3: Callaway & Sant'Anna Difference-in-Differences
    print("\n3. Callaway & Sant'Anna Difference-in-Differences")
    print("-" * 50)

    try:
        did = CallawayDiD(
            data=analysis_data,
            cohort_col="treatment_cohort",
            unit_col="country",
            time_col="year",
        )

        did.fit(outcome="mortality_rate", covariates=["gdp_log", "urban_percentage"])

        # Get aggregated results
        simple_att = did.aggregate("simple")
        print(f"Overall ATT: {simple_att}")

        # Print summary
        print("\nCallaway & Sant'Anna Results:")
        print(did.summary())

    except Exception as e:
        print(f"Callaway DiD failed: {e}")
        print("This may be due to the differences package dependency conflict.")

    # Step 4: Panel Fixed Effects Analysis
    print("\n4. Panel Fixed Effects Analysis")
    print("-" * 50)

    try:
        panel = PanelFixedEffects(
            data=analysis_data, unit_col="country", time_col="year"
        )

        # Two-way fixed effects
        panel.fit(
            outcome="mortality_rate",
            covariates=["mpower_total_score", "gdp_log", "urban_percentage"],
            method="twfe",
        )

        print("Panel Fixed Effects Results:")
        coeffs = panel.get_coefficients()
        print(coeffs.round(4))

    except Exception as e:
        print(f"Panel fixed effects failed: {e}")

    # Step 5: Event Study Analysis
    print("\n5. Event Study Analysis")
    print("-" * 50)

    try:
        event_study = EventStudyAnalysis(
            data=analysis_data,
            unit_col="country",
            time_col="year",
            treatment_col="treatment_cohort",
        )

        event_results = event_study.estimate(
            outcome="mortality_rate",
            covariates=["gdp_log", "urban_percentage"],
            max_lag=3,
            max_lead=3,
        )

        print("Event Study Results:")
        print(
            f"Pre-treatment coefficients: {len([k for k in event_results['event_time_coefficients'] if 'lead' in k])}"
        )
        print(
            f"Post-treatment coefficients: {len([k for k in event_results['event_time_coefficients'] if 'lag' in k])}"
        )

        # Test parallel trends
        parallel_trends = event_study.test_parallel_trends(event_results)
        print(f"\nParallel trends test: {parallel_trends['conclusion']}")

    except Exception as e:
        print(f"Event study failed: {e}")

    # Step 6: Synthetic Control (for a specific country)
    print("\n6. Synthetic Control Analysis")
    print("-" * 50)

    try:
        # Analyze Uruguay as treated unit (early MPOWER adopter)
        sc = SyntheticControl(
            data=analysis_data,
            unit_col="country",
            time_col="year",
            treatment_time=2010,
            treated_unit="Uruguay",
        )

        sc.fit(
            outcome="mortality_rate",
            predictors=["gdp_log", "urban_percentage", "healthcare_spending_log"],
        )

        print("Synthetic Control Results:")
        print(sc.summary())

        # Get treatment effect
        treatment_effect = sc.get_treatment_effect()
        print(f"\nTreatment effect: {treatment_effect}")

    except Exception as e:
        print(f"Synthetic control failed: {e}")

    # Step 7: Robustness Tests
    print("\n7. Robustness Tests")
    print("-" * 50)

    try:
        robustness = RobustnessTests(
            data=analysis_data, unit_col="country", time_col="year"
        )

        # Define a simple estimator function for robustness testing
        def simple_estimator(data):
            """Simple difference-in-differences estimator for robustness testing."""
            # Create treatment indicators
            data = data.copy()
            data["treated"] = (data["treatment_cohort"] > 0).astype(int)
            data["post"] = 0

            # Mark post-treatment periods
            for country in data["country"].unique():
                country_data = data[data["country"] == country]
                if country_data["treatment_cohort"].iloc[0] > 0:
                    treatment_year = country_data["treatment_cohort"].iloc[0]
                    post_mask = (data["country"] == country) & (
                        data["year"] >= treatment_year
                    )
                    data.loc[post_mask, "post"] = 1

            data["treated_post"] = data["treated"] * data["post"]

            # Simple regression
            try:
                import statsmodels.api as sm

                y = data["mortality_rate"]
                X = data[["treated", "post", "treated_post", "gdp_log"]]
                X = sm.add_constant(X)
                model = sm.OLS(y, X).fit()
                return {"att": model.params.get("treated_post", np.nan)}
            except:
                return {"att": np.nan}

        # Sample sensitivity analysis
        sensitivity = robustness.sample_sensitivity(
            estimator_func=simple_estimator,
            exclude_units=["Uruguay", "Australia"],  # Exclude key treatment countries
            random_exclusions=True,
            bootstrap_samples=50,
        )

        print("Sample Sensitivity Results:")
        summary = sensitivity.get("summary", {})
        print(f"Baseline effect: {summary.get('baseline_effect', 'N/A')}")
        print(f"Effect stability: {summary.get('robust_to_exclusions', 'N/A')}")

    except Exception as e:
        print(f"Robustness tests failed: {e}")

    # Step 8: Generate Summary Report
    print("\n8. Summary Report")
    print("-" * 50)

    try:
        summary_report = prep.generate_summary_report()

        print("Data Preparation Summary:")
        print(f"- Countries: {summary_report['basic_structure']['n_countries']}")
        print(f"- Years: {summary_report['basic_structure']['year_range']}")
        print(f"- Observations: {summary_report['basic_structure']['n_observations']}")

        if "treatment_cohorts" in summary_report:
            cohorts = summary_report["treatment_cohorts"]
            print(f"- Treated countries: {cohorts.get('n_treated_countries', 0)}")
            print(
                f"- Never-treated countries: {cohorts.get('n_never_treated_countries', 0)}"
            )
            print(f"- Treatment years: {cohorts.get('treatment_years', [])}")

    except Exception as e:
        print(f"Summary report failed: {e}")

    print("\n" + "=" * 80)
    print("WORKFLOW COMPLETED")
    print("=" * 80)

    return analysis_data


if __name__ == "__main__":
    # Run the basic workflow
    final_data = run_basic_workflow()

    print(f"\nFinal dataset shape: {final_data.shape}")
    print(f"Columns: {list(final_data.columns)}")

    # Optionally save the results
    # final_data.to_csv('examples/output/mpower_analysis_results.csv', index=False)
