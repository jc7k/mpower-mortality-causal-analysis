"""Unit tests for descriptive analysis utilities.

Tests cover:
- Treatment adoption timeline analysis
- Outcome trends by treatment status
- MPOWER score distributions
- Treatment balance checking
- Correlation analysis
- Visualization functions
"""

from unittest.mock import Mock, patch

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
            unit_col="country",
            time_col="year",
            treatment_col="mpower_high_binary",
            treatment_year_col="first_high_year",
        )

        assert descriptives.unit_col == "country"
        assert descriptives.time_col == "year"
        assert descriptives.treatment_col == "mpower_high_binary"
        assert descriptives.treatment_year_col == "first_high_year"

        # Test with missing columns
        invalid_data = sample_data.drop("country", axis=1)
        with pytest.raises(ValueError, match="Missing required columns"):
            MPOWERDescriptives(
                data=invalid_data,
                unit_col="country",
                time_col="year",
                treatment_col="mpower_high_binary",
                treatment_year_col="first_high_year",
            )

    def test_treatment_adoption_timeline(self, sample_data):
        """Test treatment adoption timeline analysis."""
        descriptives = MPOWERDescriptives(
            data=sample_data,
            unit_col="country",
            time_col="year",
            treatment_col="mpower_high_binary",
            treatment_year_col="first_high_year",
        )

        timeline = descriptives.treatment_adoption_timeline()

        # Check structure
        assert isinstance(timeline, dict)
        assert "adoption_by_year" in timeline
        assert "never_treated_count" in timeline
        assert "total_countries" in timeline
        assert "treatment_summary" in timeline

        # Check that numbers make sense
        assert timeline["total_countries"] == sample_data["country"].nunique()
        assert timeline["never_treated_count"] >= 0
        assert timeline["never_treated_count"] <= timeline["total_countries"]

        # Check adoption by year
        adoption_by_year = timeline["adoption_by_year"]
        total_adopted = sum(adoption_by_year.values())
        expected_treated = timeline["total_countries"] - timeline["never_treated_count"]
        assert total_adopted == expected_treated

    def test_outcome_trends_by_treatment(self, sample_data):
        """Test outcome trends analysis by treatment status."""
        descriptives = MPOWERDescriptives(
            data=sample_data,
            unit_col="country",
            time_col="year",
            treatment_col="mpower_high_binary",
            treatment_year_col="first_high_year",
        )

        trends = descriptives.outcome_trends_by_treatment(
            outcome="lung_cancer_mortality_rate"
        )

        # Check structure
        assert isinstance(trends, dict)
        assert "treated_trend" in trends
        assert "control_trend" in trends
        assert "difference_trend" in trends
        assert "pre_treatment_comparison" in trends

        # Check that trends are DataFrames/Series
        assert hasattr(trends["treated_trend"], "index")
        assert hasattr(trends["control_trend"], "index")

        # Test with invalid outcome
        with pytest.raises(
            ValueError, match="Outcome variable 'nonexistent' not found"
        ):
            descriptives.outcome_trends_by_treatment(outcome="nonexistent")

    def test_mpower_score_distribution(self, sample_data):
        """Test MPOWER score distribution analysis."""
        descriptives = MPOWERDescriptives(
            data=sample_data,
            unit_col="country",
            time_col="year",
            treatment_col="mpower_high_binary",
            treatment_year_col="first_high_year",
        )

        distribution = descriptives.mpower_score_distribution()

        # Check structure
        assert isinstance(distribution, dict)
        assert "total_score_stats" in distribution
        assert "component_stats" in distribution
        assert "score_by_year" in distribution
        assert "treatment_threshold_analysis" in distribution

        # Check component stats
        component_stats = distribution["component_stats"]
        expected_components = [
            "mpower_m",
            "mpower_p",
            "mpower_o",
            "mpower_w",
            "mpower_e",
            "mpower_r",
        ]
        for component in expected_components:
            if component in sample_data.columns:
                assert component in component_stats

    def test_check_treatment_balance(self, sample_data):
        """Test treatment balance checking."""
        descriptives = MPOWERDescriptives(
            data=sample_data,
            unit_col="country",
            time_col="year",
            treatment_col="mpower_high_binary",
            treatment_year_col="first_high_year",
        )

        covariates = ["gdp_per_capita_log", "urban_population_pct", "population_log"]
        balance_results = descriptives.check_treatment_balance(covariates=covariates)

        # Check structure
        assert isinstance(balance_results, dict)
        assert "balance_table" in balance_results
        assert "statistical_tests" in balance_results
        assert "standardized_differences" in balance_results

        # Check balance table
        balance_table = balance_results["balance_table"]
        assert isinstance(balance_table, pd.DataFrame)
        assert "variable" in balance_table.columns
        assert "treated_mean" in balance_table.columns
        assert "control_mean" in balance_table.columns

        # Test with missing covariates
        with pytest.raises(ValueError, match="Covariates not found in data"):
            descriptives.check_treatment_balance(covariates=["nonexistent_var"])

    def test_correlation_analysis(self, sample_data):
        """Test correlation analysis."""
        descriptives = MPOWERDescriptives(
            data=sample_data,
            unit_col="country",
            time_col="year",
            treatment_col="mpower_high_binary",
            treatment_year_col="first_high_year",
        )

        variables = [
            "mpower_total",
            "lung_cancer_mortality_rate",
            "gdp_per_capita_log",
            "urban_population_pct",
        ]
        corr_results = descriptives.correlation_analysis(variables=variables)

        # Check structure
        assert isinstance(corr_results, dict)
        assert "correlation_matrix" in corr_results
        assert "significant_correlations" in corr_results

        # Check correlation matrix
        corr_matrix = corr_results["correlation_matrix"]
        assert isinstance(corr_matrix, pd.DataFrame)
        assert corr_matrix.shape[0] == corr_matrix.shape[1]  # Square matrix

        # Diagonal should be 1.0
        np.testing.assert_array_almost_equal(np.diag(corr_matrix), 1.0)

        # Test with missing variables
        with pytest.raises(ValueError, match="Variables not found in data"):
            descriptives.correlation_analysis(variables=["nonexistent_var"])

    def test_pre_post_comparison(self, sample_data):
        """Test pre-post treatment comparison."""
        descriptives = MPOWERDescriptives(
            data=sample_data,
            unit_col="country",
            time_col="year",
            treatment_col="mpower_high_binary",
            treatment_year_col="first_high_year",
        )

        comparison = descriptives.pre_post_comparison(
            outcome="lung_cancer_mortality_rate", pre_periods=2, post_periods=2
        )

        # Check structure
        assert isinstance(comparison, dict)
        assert "country_level_changes" in comparison
        assert "aggregate_effects" in comparison
        assert "statistical_test" in comparison

        # Check country-level changes
        country_changes = comparison["country_level_changes"]
        assert isinstance(country_changes, pd.DataFrame)
        assert "country" in country_changes.columns
        assert "pre_mean" in country_changes.columns
        assert "post_mean" in country_changes.columns
        assert "change" in country_changes.columns

    def test_plotting_functionality_mock(self, sample_data):
        """Test plotting functionality with mocked matplotlib."""
        descriptives = MPOWERDescriptives(
            data=sample_data,
            unit_col="country",
            time_col="year",
            treatment_col="mpower_high_binary",
            treatment_year_col="first_high_year",
        )

        # Mock matplotlib
        with patch(
            "mpower_mortality_causal_analysis.causal_inference.utils.descriptive.PLOTTING_AVAILABLE",
            True,
        ):
            with patch(
                "mpower_mortality_causal_analysis.causal_inference.utils.descriptive.plt"
            ) as mock_plt:
                mock_fig = Mock()
                mock_plt.subplots.return_value = (mock_fig, Mock())

                # Test outcome trends with plotting
                trends = descriptives.outcome_trends_by_treatment(
                    outcome="lung_cancer_mortality_rate", save_path="test_plot.png"
                )

                # Check that plotting functions were called
                mock_plt.subplots.assert_called()

        # Test without plotting available
        with patch(
            "mpower_mortality_causal_analysis.causal_inference.utils.descriptive.PLOTTING_AVAILABLE",
            False,
        ):
            # Should still work but without plots
            trends = descriptives.outcome_trends_by_treatment(
                outcome="lung_cancer_mortality_rate"
            )
            assert isinstance(trends, dict)

    def test_statistical_tests(self, sample_data):
        """Test statistical testing functionality."""
        descriptives = MPOWERDescriptives(
            data=sample_data,
            unit_col="country",
            time_col="year",
            treatment_col="mpower_high_binary",
            treatment_year_col="first_high_year",
        )

        # Test t-test functionality in balance checking
        balance_results = descriptives.check_treatment_balance(
            covariates=["gdp_per_capita_log"]
        )

        statistical_tests = balance_results["statistical_tests"]
        assert isinstance(statistical_tests, dict)
        assert "gdp_per_capita_log" in statistical_tests

        test_result = statistical_tests["gdp_per_capita_log"]
        assert "statistic" in test_result
        assert "p_value" in test_result
        assert isinstance(test_result["p_value"], (int, float))
        assert 0 <= test_result["p_value"] <= 1

    def test_edge_cases(self, sample_data):
        """Test handling of edge cases."""
        descriptives = MPOWERDescriptives(
            data=sample_data,
            unit_col="country",
            time_col="year",
            treatment_col="mpower_high_binary",
            treatment_year_col="first_high_year",
        )

        # Test with no treated units
        no_treatment_data = sample_data.copy()
        no_treatment_data["mpower_high_binary"] = 0
        no_treatment_data["first_high_year"] = 0

        descriptives_no_treat = MPOWERDescriptives(
            data=no_treatment_data,
            unit_col="country",
            time_col="year",
            treatment_col="mpower_high_binary",
            treatment_year_col="first_high_year",
        )

        # Should handle gracefully
        timeline = descriptives_no_treat.treatment_adoption_timeline()
        assert timeline["never_treated_count"] == timeline["total_countries"]

        # Test with all treated units
        all_treatment_data = sample_data.copy()
        all_treatment_data["mpower_high_binary"] = 1
        all_treatment_data["first_high_year"] = 2012

        descriptives_all_treat = MPOWERDescriptives(
            data=all_treatment_data,
            unit_col="country",
            time_col="year",
            treatment_col="mpower_high_binary",
            treatment_year_col="first_high_year",
        )

        timeline = descriptives_all_treat.treatment_adoption_timeline()
        assert timeline["never_treated_count"] == 0

    def test_data_validation(self):
        """Test data validation functionality."""
        # Create minimal valid data
        valid_data = pd.DataFrame(
            {
                "country": ["A", "B"],
                "year": [2010, 2010],
                "mpower_high_binary": [0, 1],
                "first_high_year": [0, 2010],
                "outcome": [1, 2],
            }
        )

        # Should initialize successfully
        descriptives = MPOWERDescriptives(
            data=valid_data,
            unit_col="country",
            time_col="year",
            treatment_col="mpower_high_binary",
            treatment_year_col="first_high_year",
        )

        assert descriptives is not None

        # Test empty data
        empty_data = pd.DataFrame()
        with pytest.raises(ValueError):
            MPOWERDescriptives(
                data=empty_data,
                unit_col="country",
                time_col="year",
                treatment_col="mpower_high_binary",
                treatment_year_col="first_high_year",
            )

    def test_summary_statistics(self, sample_data):
        """Test summary statistics generation."""
        descriptives = MPOWERDescriptives(
            data=sample_data,
            unit_col="country",
            time_col="year",
            treatment_col="mpower_high_binary",
            treatment_year_col="first_high_year",
        )

        # Test that all main methods return valid results
        timeline = descriptives.treatment_adoption_timeline()
        assert isinstance(timeline, dict)

        trends = descriptives.outcome_trends_by_treatment("lung_cancer_mortality_rate")
        assert isinstance(trends, dict)

        distribution = descriptives.mpower_score_distribution()
        assert isinstance(distribution, dict)

        balance = descriptives.check_treatment_balance(["gdp_per_capita_log"])
        assert isinstance(balance, dict)

        correlation = descriptives.correlation_analysis(
            ["mpower_total", "lung_cancer_mortality_rate"]
        )
        assert isinstance(correlation, dict)

        # All should complete without errors
        assert True

    def test_missing_data_handling(self, sample_data):
        """Test handling of missing data."""
        # Add some missing values
        missing_data = sample_data.copy()
        missing_data.loc[0:5, "lung_cancer_mortality_rate"] = np.nan
        missing_data.loc[10:15, "gdp_per_capita_log"] = np.nan

        descriptives = MPOWERDescriptives(
            data=missing_data,
            unit_col="country",
            time_col="year",
            treatment_col="mpower_high_binary",
            treatment_year_col="first_high_year",
        )

        # Should handle missing data gracefully
        trends = descriptives.outcome_trends_by_treatment("lung_cancer_mortality_rate")
        assert isinstance(trends, dict)

        balance = descriptives.check_treatment_balance(["gdp_per_capita_log"])
        assert isinstance(balance, dict)

        # Methods should complete without errors despite missing data
        assert True
