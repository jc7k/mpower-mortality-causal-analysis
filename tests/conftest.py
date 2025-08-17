"""Pytest configuration and fixtures for causal inference tests.

This module provides shared fixtures and configuration for testing
the MPOWER causal inference framework.
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_mpower_data() -> pd.DataFrame:
    """Create sample MPOWER data for testing."""
    np.random.seed(42)

    # Define panel structure
    countries = ["Country_A", "Country_B", "Country_C", "Country_D", "Country_E"]
    years = list(range(2008, 2020))

    data = []
    for country in countries:
        for year in years:
            data.append({"country": country, "year": year})

    df = pd.DataFrame(data)

    # Create treatment timing
    # Countries A and B are treated in 2012 and 2014
    treatment_timing = {
        "Country_A": 2012,
        "Country_B": 2014,
        "Country_C": 0,  # Never treated
        "Country_D": 0,  # Never treated
        "Country_E": 0,  # Never treated
    }

    df["treatment_cohort"] = df["country"].map(treatment_timing)

    # Create MPOWER scores
    df["mpower_total_score"] = 0
    for country in countries:
        country_mask = df["country"] == country
        if treatment_timing[country] > 0:
            # Treated countries: low pre-treatment, high post-treatment
            treatment_year = treatment_timing[country]
            pre_mask = country_mask & (df["year"] < treatment_year)
            post_mask = country_mask & (df["year"] >= treatment_year)

            df.loc[pre_mask, "mpower_total_score"] = np.random.normal(
                15, 3, pre_mask.sum()
            )
            df.loc[post_mask, "mpower_total_score"] = np.random.normal(
                30, 2, post_mask.sum()
            )
        else:
            # Never treated: consistently low scores
            df.loc[country_mask, "mpower_total_score"] = np.random.normal(
                12, 2, country_mask.sum()
            )

    # Ensure scores are within bounds
    df["mpower_total_score"] = df["mpower_total_score"].clip(0, 37)

    # Create outcome variable with treatment effect
    df["mortality_rate"] = 0
    for country in countries:
        country_mask = df["country"] == country
        base_rate = 50 + np.random.normal(0, 5)  # Country fixed effect

        for idx in df[country_mask].index:
            year = df.loc[idx, "year"]

            # Time trend
            time_effect = -0.5 * (year - 2008)

            # Treatment effect (with lag)
            treatment_effect = 0
            if treatment_timing[country] > 0 and year >= treatment_timing[country] + 1:
                treatment_effect = -8  # Reduction in mortality rate

            # Random noise
            noise = np.random.normal(0, 2)

            mortality_rate = base_rate + time_effect + treatment_effect + noise
            df.loc[idx, "mortality_rate"] = max(0, mortality_rate)

    # Add control variables
    df["gdp_per_capita"] = np.random.normal(25000, 5000, len(df))
    df["gdp_log"] = np.log(df["gdp_per_capita"])
    df["urban_percentage"] = np.random.normal(65, 10, len(df))
    df["healthcare_spending"] = np.random.normal(1500, 300, len(df))

    return df


@pytest.fixture
def balanced_panel_data(sample_mpower_data) -> pd.DataFrame:
    """Create balanced panel data for testing."""
    # Ensure all countries have observations for all years
    return sample_mpower_data.copy()


@pytest.fixture
def unbalanced_panel_data(sample_mpower_data) -> pd.DataFrame:
    """Create unbalanced panel data for testing."""
    data = sample_mpower_data.copy()

    # Remove some observations to create unbalanced panel
    # Remove last 2 years for Country_C
    drop_mask = (data["country"] == "Country_C") & (data["year"] >= 2018)
    data = data[~drop_mask]

    # Remove first 2 years for Country_D
    drop_mask = (data["country"] == "Country_D") & (data["year"] <= 2009)
    data = data[~drop_mask]

    return data


@pytest.fixture
def treatment_cohort_data(sample_mpower_data) -> pd.DataFrame:
    """Data with treatment cohorts already defined."""
    return sample_mpower_data.copy()


@pytest.fixture
def synthetic_control_data(sample_mpower_data) -> pd.DataFrame:
    """Data prepared for synthetic control analysis."""
    data = sample_mpower_data.copy()

    # Ensure we have a clear treated unit
    treated_unit = "Country_A"
    treatment_time = 2012

    return data


@pytest.fixture
def event_study_data(sample_mpower_data) -> pd.DataFrame:
    """Data prepared for event study analysis."""
    return sample_mpower_data.copy()


@pytest.fixture
def sample_config() -> dict[str, Any]:
    """Sample configuration for testing."""
    return {
        "country_col": "country",
        "year_col": "year",
        "outcome_col": "mortality_rate",
        "treatment_col": "treatment_cohort",
        "mpower_col": "mpower_total_score",
        "control_cols": ["gdp_log", "urban_percentage"],
        "threshold": 25.0,
        "never_treated_value": 0,
    }


@pytest.fixture
def minimal_data() -> pd.DataFrame:
    """Minimal dataset for edge case testing."""
    return pd.DataFrame(
        {
            "country": ["A", "A", "B", "B"],
            "year": [2010, 2011, 2010, 2011],
            "treatment_cohort": [2011, 2011, 0, 0],
            "outcome": [50, 45, 52, 51],
            "control": [1.0, 1.1, 0.9, 1.0],
        }
    )


@pytest.fixture
def empty_data() -> pd.DataFrame:
    """Empty DataFrame for error testing."""
    return pd.DataFrame()


@pytest.fixture
def malformed_data() -> pd.DataFrame:
    """Malformed data for error handling tests."""
    return pd.DataFrame({"wrong_col": [1, 2, 3], "another_col": ["a", "b", "c"]})
