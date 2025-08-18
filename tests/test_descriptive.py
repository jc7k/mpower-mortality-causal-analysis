"""Unit tests for descriptive analysis utilities - Fixed version.

Tests cover:
- Class initialization
- Summary statistics generation
- Plotting functionality with actual methods
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from mpower_mortality_causal_analysis.causal_inference.utils.descriptive import (
    MPOWERDescriptives,
)


class TestMPOWERDescriptives:
    """Test suite for MPOWER descriptive analysis utilities."""

    @pytest.fixture
    def sample_data(self):
        """Create sample MPOWER data for testing."""
        np.random.seed(42)

        countries = [f"Country_{i}" for i in range(15)]
        years = list(range(2008, 2019))

        data_list = []

        for country in countries:
            # Assign treatment timing
            if country in countries[:8]:  # Some treated
                treatment_year = np.random.choice(
                    [2012, 2014, 2016, 0], p=[0.25, 0.25, 0.25, 0.25]
                )
            else:
                treatment_year = 0

            for year in years:
                # MPOWER scores and treatment status
                if treatment_year > 0 and year >= treatment_year:
                    mpower_total = np.random.uniform(25, 29)
                    mpower_high = 1
                else:
                    mpower_total = np.random.uniform(10, 24)
                    mpower_high = 0

                # Individual MPOWER components
                m_score = np.random.randint(0, 5)
                p_score = np.random.randint(0, 5)
                o_score = np.random.randint(0, 5)
                w_score = np.random.randint(0, 5)
                e_score = np.random.randint(0, 5)
                r_score = np.random.randint(0, 4)

                # Mortality outcomes (with treatment effect)
                base_mortality = 45 + np.random.normal(0, 5)
                if mpower_high:
                    treatment_effect = -2 + np.random.normal(0, 1)
                    mortality = base_mortality + treatment_effect
                else:
                    mortality = base_mortality

                # Control variables
                gdp_log = 8 + np.random.normal(0, 0.5)
                urban_pct = 55 + np.random.normal(0, 15)
                population_log = 15 + np.random.normal(0, 1)
                education_pct = 4 + np.random.normal(0, 1)

                data_list.append(
                    {
                        "country": country,
                        "year": year,
                        "mpower_total": mpower_total,
                        "mpower_high_binary": mpower_high,
                        "first_high_year": treatment_year if treatment_year > 0 else 0,
                        "mpower_m": m_score,
                        "mpower_p": p_score,
                        "mpower_o": o_score,
                        "mpower_w": w_score,
                        "mpower_e": e_score,
                        "mpower_r": r_score,
                        "lung_cancer_mortality_rate": mortality,
                        "cardiovascular_mortality_rate": mortality
                        + np.random.normal(0, 3),
                        "gdp_per_capita_log": gdp_log,
                        "urban_population_pct": urban_pct,
                        "population_log": population_log,
                        "education_expenditure_pct_gdp": education_pct,
                    }
                )

        return pd.DataFrame(data_list)

    def test_initialization(self, sample_data):
        """Test proper initialization of MPOWERDescriptives."""
        descriptives = MPOWERDescriptives(
            data=sample_data,
            country_col="country",
            year_col="year",
            cohort_col="first_high_year",
        )

        assert descriptives.country_col == "country"
        assert descriptives.year_col == "year"
        assert descriptives.cohort_col == "first_high_year"

    def test_summary_statistics(self, sample_data):
        """Test generation of summary statistics."""
        descriptives = MPOWERDescriptives(
            data=sample_data,
            country_col="country",
            year_col="year",
            cohort_col="first_high_year",
        )

        summary = descriptives.generate_summary_statistics()

        assert isinstance(summary, dict)
        assert "dataset_structure" in summary
        assert "treatment_cohorts" in summary
        assert "outcomes" in summary

        # Check dataset structure
        dataset_structure = summary["dataset_structure"]
        assert dataset_structure["n_countries"] == 15
        assert dataset_structure["n_observations"] > 0
        assert len(dataset_structure["countries_list"]) == 15

    @patch("matplotlib.pyplot.show")
    def test_plotting_methods(self, mock_show, sample_data):
        """Test all plotting methods work without errors."""
        descriptives = MPOWERDescriptives(
            data=sample_data,
            country_col="country",
            year_col="year",
            cohort_col="first_high_year",
        )

        # Test treatment adoption timeline plot
        fig1 = descriptives.plot_treatment_adoption_timeline()
        assert fig1 is not None

        # Test outcome trends plot
        fig2 = descriptives.plot_outcome_trends_by_cohort()
        assert fig2 is not None

        # Test MPOWER score distributions
        fig3 = descriptives.plot_mpower_score_distributions()
        assert fig3 is not None

        # Test correlation heatmap
        fig4 = descriptives.plot_correlation_heatmap()
        assert fig4 is not None

        # Test treatment balance check
        fig5 = descriptives.plot_treatment_balance_check()
        assert fig5 is not None

    def test_edge_cases(self, sample_data):
        """Test edge cases and error handling."""
        descriptives = MPOWERDescriptives(
            data=sample_data,
            country_col="country",
            year_col="year",
            cohort_col="first_high_year",
        )

        # Test with minimal data
        minimal_data = sample_data.head(10)
        minimal_descriptives = MPOWERDescriptives(
            data=minimal_data,
            country_col="country",
            year_col="year",
            cohort_col="first_high_year",
        )

        summary = minimal_descriptives.generate_summary_statistics()
        assert isinstance(summary, dict)

    def test_data_validation(self, sample_data):
        """Test data validation during initialization."""
        # Test with missing required columns
        invalid_data = sample_data.drop("country", axis=1)

        with pytest.raises(ValueError):
            MPOWERDescriptives(
                data=invalid_data,
                country_col="country",
                year_col="year",
                cohort_col="first_high_year",
            )

    def test_export_functionality(self, sample_data, tmp_path):
        """Test export functionality."""
        descriptives = MPOWERDescriptives(
            data=sample_data,
            country_col="country",
            year_col="year",
            cohort_col="first_high_year",
        )

        # Test export to file
        export_path = tmp_path / "test_report.md"
        descriptives.export_summary_report(
            filepath=str(export_path),
            include_plots=False,  # Skip plots to avoid display issues in test
        )

        assert export_path.exists()

        # Check that file has some content
        content = export_path.read_text()
        assert len(content) > 0
        assert "MPOWER Causal Analysis - Descriptive Statistics Report" in content
