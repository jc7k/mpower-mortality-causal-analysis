#!/usr/bin/env python3
"""Data Processing Script for MPOWER Causal Analysis.

This script implements the complete data preparation pipeline for causal inference analysis
of WHO MPOWER tobacco control policies' impact on mortality outcomes. The pipeline:

1. Loads and cleans WHO MPOWER policy data (M,P,O,W,E,R component scores)
2. Processes IHME Global Burden of Disease mortality data (lung cancer, CVD, etc.)
3. Integrates World Bank economic and demographic control variables
4. Creates treatment cohorts for staggered difference-in-differences analysis
5. Generates analysis-ready panel dataset with proper variable transformations

Data Sources:
- WHO MPOWER: Tobacco control policy implementation scores by country-year
- IHME GBD: Age-standardized mortality rates for tobacco-related diseases
- World Bank WDI: GDP per capita, urbanization, population, education expenditure

Output:
- data/processed/integrated_panel.csv: Merged raw data from all sources
- data/processed/analysis_ready_data.csv: Final dataset with treatment cohorts

Follows the data preparation pipeline outlined in PRPs/mpower_causal_analysis.md
"""

import sys

from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.append("src")
from mpower_mortality_causal_analysis.causal_inference.data.preparation import (
    MPOWERDataPrep,
)


def load_and_clean_mpower_data() -> pd.DataFrame:
    """Load and clean WHO MPOWER data from Global Health Observatory (GHO) format.

    The MPOWER package represents six tobacco control measures:
    - M: Monitor tobacco use and prevention policies (0-4 scale)
    - P: Protect people from tobacco smoke (0-4 scale)
    - O: Offer help to quit tobacco use (0-4 scale)
    - W: Warn about dangers of tobacco (0-4 scale)
    - E: Enforce bans on tobacco advertising/promotion (0-4 scale)
    - R: Raise taxes on tobacco (0-5 scale)

    Total possible score: 29 points (used as treatment intensity measure)

    Returns:
        DataFrame with cleaned MPOWER data in country-year panel format
    """
    print("Loading MPOWER data...")
    mpower_raw = pd.read_csv("data/raw/mpower_gho/mpower_gho_data.csv")

    # Filter to numeric values only (exclude text/categorical entries)
    mpower_clean = mpower_raw[mpower_raw["ValueType"] == "numeric"].copy()

    # Rename columns for consistency with causal inference standards
    mpower_clean = mpower_clean.rename(
        columns={
            "Location": "country_name",  # WHO country names
            "Period": "year",  # Survey years (biennial 2007-2022)
            "FactValueNumeric": "value",  # Policy implementation score
            "IndicatorCode": "indicator",  # MPOWER component identifier
        }
    )

    # Filter to main MPOWER policy indicators (exclude sub-components)
    mpower_indicators = [
        "M_Group",
        "P_Group",
        "O_Group",
        "W_Group",
        "E_Group",
        "R_Group",
    ]
    mpower_clean = mpower_clean[mpower_clean["indicator"].isin(mpower_indicators)]

    # Pivot from long to wide format (one row per country-year, columns for each MPOWER component)
    mpower_wide = mpower_clean.pivot_table(
        index=["country_name", "year"],
        columns="indicator",
        values="value",
        aggfunc="first",  # Take first value if duplicates exist
    ).reset_index()

    # Rename columns to match academic convention and PRP specification
    column_mapping = {
        "M_Group": "m_score",  # Monitor
        "P_Group": "p_score",  # Protect
        "O_Group": "o_score",  # Offer
        "W_Group": "w_score",  # Warn
        "E_Group": "e_score",  # Enforce
        "R_Group": "r_score",  # Raise taxes
    }
    mpower_wide = mpower_wide.rename(columns=column_mapping)

    # Calculate total MPOWER score (primary treatment intensity measure)
    # This will be used to define high vs. low policy implementation
    score_cols = ["m_score", "p_score", "o_score", "w_score", "e_score", "r_score"]
    mpower_wide["mpower_total"] = mpower_wide[score_cols].sum(axis=1, skipna=False)

    # Add temporary country code (will be properly standardized later)
    mpower_wide["iso3"] = mpower_wide["country_name"].str.upper().str[:3]

    print(f"MPOWER data shape: {mpower_wide.shape}")
    print(f"Years covered: {sorted(mpower_wide.year.unique())}")
    print(f"Countries: {mpower_wide.country_name.nunique()}")

    return mpower_wide


def load_and_clean_gbd_data() -> pd.DataFrame:
    """Load and clean IHME GBD mortality data.

    Returns:
        DataFrame with relevant mortality outcomes
    """
    print("Loading IHME GBD data...")
    gbd_raw = pd.read_csv(
        "data/raw/gbd/IHME-GBD_2008-2019_DATA-AllCountries-Tobacco.csv"
    )

    # Filter to relevant causes and all ages/both sexes
    target_causes = [
        "Tracheal, bronchus, and lung cancer",
        "Cardiovascular diseases",
        "Ischemic heart disease",
        "Chronic obstructive pulmonary disease",
    ]

    gbd_filtered = gbd_raw[
        (gbd_raw["cause_name"].isin(target_causes))
        & (gbd_raw["age_name"] == "Age-standardized")
        & (gbd_raw["sex_name"] == "Both")
        & (gbd_raw["metric_name"] == "Rate")
        & (gbd_raw["measure_name"] == "Deaths")
    ].copy()

    # Rename columns
    gbd_filtered = gbd_filtered.rename(
        columns={
            "location_name": "country_name",
            "val": "mortality_rate",
            "cause_name": "cause",
        }
    )

    # Create outcome variables
    outcome_mapping = {
        "Tracheal, bronchus, and lung cancer": "mort_lung_cancer_asr",
        "Cardiovascular diseases": "mort_cvd_asr",
        "Ischemic heart disease": "mort_ihd_asr",
        "Chronic obstructive pulmonary disease": "mort_copd_asr",
    }

    gbd_filtered["outcome_var"] = gbd_filtered["cause"].map(outcome_mapping)

    # Pivot to wide format
    gbd_wide = gbd_filtered.pivot_table(
        index=["country_name", "year"],
        columns="outcome_var",
        values="mortality_rate",
        aggfunc="first",
    ).reset_index()

    print(f"GBD data shape: {gbd_wide.shape}")
    print(f"Years covered: {sorted(gbd_wide.year.unique())}")
    print(f"Countries: {gbd_wide.country_name.nunique()}")

    return gbd_wide


def load_and_clean_worldbank_data() -> pd.DataFrame:
    """Load and clean World Bank control variables.

    Returns:
        DataFrame with control variables in panel format
    """
    print("Loading World Bank data...")

    # Load both WB files
    wb_manual = pd.read_csv("data/raw/worldbank/worldbank_wdi_manual_20250814.csv")
    wb_additional = pd.read_csv(
        "data/raw/worldbank/worldbank_wdi_additional_20250815.csv"
    )

    # Combine datasets
    wb_combined = pd.concat([wb_manual, wb_additional], ignore_index=True)

    # Key control variables we want
    target_indicators = [
        "NY.GDP.PCAP.KD",  # GDP per capita (constant 2015 US$)
        "SP.URB.TOTL.IN.ZS",  # Urban population (% of total)
        "SP.POP.TOTL",  # Population, total
        "SE.XPD.TOTL.GD.ZS",  # Government expenditure on education, total (% of GDP)
    ]

    # Filter to target indicators
    wb_filtered = wb_combined[wb_combined["Series Code"].isin(target_indicators)].copy()

    # Melt from wide to long format
    year_cols = [col for col in wb_filtered.columns if "[YR" in col]

    wb_long = wb_filtered.melt(
        id_vars=["Country Name", "Country Code", "Series Name", "Series Code"],
        value_vars=year_cols,
        var_name="year_raw",
        value_name="value",
    )

    # Extract year from column name
    wb_long["year"] = wb_long["year_raw"].str.extract(r"(\d{4})").astype(int)

    # Clean values (convert '..' to NaN)
    wb_long["value"] = pd.to_numeric(wb_long["value"], errors="coerce")

    # Create variable names
    var_mapping = {
        "NY.GDP.PCAP.KD": "gdp_pc_constant",
        "SP.URB.TOTL.IN.ZS": "urban_pop_pct",
        "SP.POP.TOTL": "population_total",
        "SE.XPD.TOTL.GD.ZS": "edu_exp_pct_gdp",
    }

    wb_long["variable"] = wb_long["Series Code"].map(var_mapping)

    # Pivot to wide format
    wb_wide = wb_long.pivot_table(
        index=["Country Name", "Country Code", "year"],
        columns="variable",
        values="value",
        aggfunc="first",
    ).reset_index()

    # Rename columns
    wb_wide = wb_wide.rename(columns={"Country Name": "country_name"})

    # Log transform GDP per capita
    wb_wide["gdp_log"] = np.log(wb_wide["gdp_pc_constant"] + 1)

    print(f"World Bank data shape: {wb_wide.shape}")
    print(f"Years covered: {sorted(wb_wide.year.unique())}")
    print(f"Countries: {wb_wide.country_name.nunique()}")

    return wb_wide


def create_country_mapping() -> dict[str, str]:
    """Create mapping between different country name formats.

    Returns:
        Dictionary mapping country names to ISO3 codes
    """
    # This is a simplified mapping - in practice would need comprehensive country name standardization
    # For now, we'll use a basic approach and handle mismatches

    country_mapping = {
        # Add specific mappings as needed
        "United States of America": "USA",
        "United Kingdom of Great Britain and Northern Ireland": "GBR",
        "Russian Federation": "RUS",
        "Iran (Islamic Republic of)": "IRN",
        "Venezuela (Bolivarian Republic of)": "VEN",
        "Bolivia (Plurinational State of)": "BOL",
        "Republic of Korea": "KOR",
        "Democratic People's Republic of Korea": "PRK",
        "Republic of Moldova": "MDA",
        "The former Yugoslav Republic of Macedonia": "MKD",
        "Türkiye": "TUR",
        "Turkey": "TUR",
    }

    return country_mapping


def standardize_country_names(
    df: pd.DataFrame, country_col: str = "country_name"
) -> pd.DataFrame:
    """Standardize country names across datasets.

    Args:
        df: DataFrame with country names
        country_col: Name of country column

    Returns:
        DataFrame with standardized country names and ISO3 codes
    """
    df_clean = df.copy()

    # Apply country mapping
    country_mapping = create_country_mapping()
    df_clean[country_col] = df_clean[country_col].replace(country_mapping)

    # Create simple ISO3 codes (this is simplified - would need proper ISO3 mapping)
    # For demonstration, we'll create consistent identifiers
    unique_countries = df_clean[country_col].unique()
    iso3_mapping = {
        country: f"C{i:03d}" for i, country in enumerate(sorted(unique_countries))
    }

    # Use actual ISO3 codes where available
    actual_iso3 = {
        "United States": "USA",
        "United Kingdom": "GBR",
        "Germany": "DEU",
        "France": "FRA",
        "Italy": "ITA",
        "Spain": "ESP",
        "Japan": "JPN",
        "China": "CHN",
        "India": "IND",
        "Brazil": "BRA",
        "Canada": "CAN",
        "Australia": "AUS",
        "Mexico": "MEX",
    }

    iso3_mapping.update(actual_iso3)
    df_clean["iso3"] = df_clean[country_col].map(iso3_mapping)

    return df_clean


def integrate_datasets() -> pd.DataFrame:
    """Integrate all datasets into unified panel.

    Returns:
        Integrated panel dataset ready for analysis
    """
    print("Integrating datasets...")

    # Load and clean each dataset
    mpower_data = load_and_clean_mpower_data()
    gbd_data = load_and_clean_gbd_data()
    wb_data = load_and_clean_worldbank_data()

    # Standardize country names
    mpower_data = standardize_country_names(mpower_data)
    gbd_data = standardize_country_names(gbd_data)
    wb_data = standardize_country_names(wb_data)

    # Merge datasets
    # Start with MPOWER as base (treatment data)
    panel_data = mpower_data.copy()

    # Merge GBD outcomes
    panel_data = panel_data.merge(
        gbd_data, on=["country_name", "year"], how="left", suffixes=("", "_gbd")
    )

    # Merge World Bank controls
    panel_data = panel_data.merge(
        wb_data, on=["country_name", "year"], how="left", suffixes=("", "_wb")
    )

    # Filter to common years (2008-2019 intersection)
    common_years = list(range(2008, 2020))
    panel_data = panel_data[panel_data["year"].isin(common_years)]

    # Remove countries with too little data
    country_counts = panel_data.groupby("country_name").size()
    valid_countries = country_counts[country_counts >= 5].index  # At least 5 years
    panel_data = panel_data[panel_data["country_name"].isin(valid_countries)]

    # Sort by country and year
    panel_data = panel_data.sort_values(["country_name", "year"]).reset_index(drop=True)

    print(f"Integrated panel shape: {panel_data.shape}")
    print(f"Countries: {panel_data.country_name.nunique()}")
    print(f"Years: {sorted(panel_data.year.unique())}")
    print(f"Missing MPOWER data: {panel_data.mpower_total.isnull().sum()}")
    print(
        f"Missing outcome data: {panel_data[['mort_lung_cancer_asr', 'mort_cvd_asr']].isnull().sum().sum()}"
    )

    return panel_data


def create_treatment_cohorts(panel_data: pd.DataFrame) -> pd.DataFrame:
    """Create treatment cohorts for staggered DiD analysis.

    Args:
        panel_data: Integrated panel dataset

    Returns:
        Panel data with treatment cohorts defined
    """
    print("Creating treatment cohorts...")

    # Use the MPOWERDataPrep class
    prep = MPOWERDataPrep(data=panel_data, country_col="country_name", year_col="year")

    # Create treatment cohorts based on MPOWER total score threshold
    data_with_cohorts = prep.create_treatment_cohorts(
        mpower_col="mpower_total",
        treatment_definition="binary_threshold",
        threshold=25.0,  # High MPOWER score threshold
        min_years_high=2,  # Must sustain for 2+ years
    )

    # Balance the panel
    balanced_data = prep.balance_panel(
        data=data_with_cohorts, method="drop_unbalanced", min_years=5
    )

    # Prepare for analysis with additional variables
    analysis_data = prep.prepare_for_analysis(
        outcome_cols=["mort_lung_cancer_asr", "mort_cvd_asr"],
        control_cols=["gdp_pc_constant", "urban_pop_pct"],
        log_transform=["gdp_pc_constant"],
        create_lags={"mpower_total": 3},
    )

    # Generate summary report
    summary = prep.generate_summary_report()
    print("Data preparation summary:")
    print(f"- Countries: {summary['basic_structure']['n_countries']}")
    print(f"- Years: {summary['basic_structure']['year_range']}")
    print(f"- Observations: {summary['basic_structure']['n_observations']}")

    if "treatment_cohorts" in summary:
        print(
            f"- Treated countries: {summary['treatment_cohorts']['n_treated_countries']}"
        )
        print(
            f"- Never treated: {summary['treatment_cohorts']['n_never_treated_countries']}"
        )
        print(f"- Treatment years: {summary['treatment_cohorts']['treatment_years']}")

    return analysis_data


def main():
    """Main data processing pipeline."""
    print("Starting MPOWER data processing pipeline...")

    # Create output directory
    Path("data/processed").mkdir(exist_ok=True)

    try:
        # Integrate all datasets
        panel_data = integrate_datasets()

        # Save intermediate result
        panel_data.to_csv("data/processed/integrated_panel.csv", index=False)
        print("Saved integrated panel data")

        # Create treatment cohorts and prepare for analysis
        analysis_data = create_treatment_cohorts(panel_data)

        # Save final analysis dataset
        analysis_data.to_csv("data/processed/analysis_ready_data.csv", index=False)
        print("Saved analysis-ready dataset")

        # Basic validation
        print("\n=== FINAL DATASET VALIDATION ===")
        print(f"Shape: {analysis_data.shape}")
        print(f"Columns: {list(analysis_data.columns)}")
        print("Treatment cohort distribution:")
        print(analysis_data["treatment_cohort"].value_counts().sort_index())

        print("\\nData processing completed successfully!")
        print("Files saved:")
        print("- data/processed/integrated_panel.csv")
        print("- data/processed/analysis_ready_data.csv")

    except Exception as e:
        print(f"Error in data processing: {e}")
        raise


if __name__ == "__main__":
    main()
