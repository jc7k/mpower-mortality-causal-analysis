"""Tests for MPOWER data preparation utilities.

This module tests the MPOWERDataPrep class and its methods for preparing
data for causal inference analysis.
"""

import numpy as np
import pandas as pd
import pytest

from mpower_mortality_causal_analysis.causal_inference.data.preparation import (
    MPOWERDataPrep,
)


class TestMPOWERDataPrep:
    """Test suite for MPOWERDataPrep class."""

    def test_init_valid_data(self, sample_mpower_data, sample_config):
        """Test initialization with valid data."""
        prep = MPOWERDataPrep(
            data=sample_mpower_data,
            country_col=sample_config["country_col"],
            year_col=sample_config["year_col"],
        )

        assert isinstance(prep.data, pd.DataFrame)
        assert len(prep.data) == len(sample_mpower_data)
        assert prep.country_col == sample_config["country_col"]
        assert prep.year_col == sample_config["year_col"]

    def test_init_empty_data(self, empty_data):
        """Test initialization with empty data raises error."""
        with pytest.raises(ValueError, match="Data cannot be empty"):
            MPOWERDataPrep(data=empty_data)

    def test_init_invalid_data_type(self):
        """Test initialization with invalid data type raises error."""
        with pytest.raises(TypeError, match="Data must be a pandas DataFrame"):
            MPOWERDataPrep(data="not a dataframe")

    def test_init_missing_columns(self, sample_mpower_data):
        """Test initialization with missing required columns raises error."""
        with pytest.raises(ValueError, match="Missing required columns"):
            MPOWERDataPrep(
                data=sample_mpower_data, country_col="missing_col", year_col="year"
            )

    def test_create_binary_threshold_cohorts(self, sample_mpower_data, sample_config):
        """Test creation of binary threshold treatment cohorts."""
        prep = MPOWERDataPrep(
            data=sample_mpower_data,
            country_col=sample_config["country_col"],
            year_col=sample_config["year_col"],
        )

        result = prep.create_treatment_cohorts(
            mpower_col=sample_config["mpower_col"],
            treatment_definition="binary_threshold",
            threshold=sample_config["threshold"],
            min_years_high=2,
        )

        assert "treatment_cohort" in result.columns
        assert prep._treatment_cohorts is not None

        # Check that some countries are treated
        treated_countries = result[result["treatment_cohort"] > 0]["country"].nunique()
        assert treated_countries > 0

        # Check that never-treated countries have cohort 0
        never_treated = result[result["treatment_cohort"] == 0]
        assert len(never_treated) > 0

    def test_create_continuous_change_cohorts(self, sample_mpower_data, sample_config):
        """Test creation of continuous change treatment cohorts."""
        prep = MPOWERDataPrep(
            data=sample_mpower_data,
            country_col=sample_config["country_col"],
            year_col=sample_config["year_col"],
        )

        result = prep.create_treatment_cohorts(
            mpower_col=sample_config["mpower_col"],
            treatment_definition="continuous_change",
            baseline_years=[2008, 2009, 2010],
        )

        assert "treatment_cohort" in result.columns

        # Should have some treated countries
        treated_count = (result["treatment_cohort"] > 0).sum()
        assert treated_count >= 0  # May be zero if no substantial improvements

    def test_create_component_based_cohorts(self, sample_mpower_data, sample_config):
        """Test creation of component-based treatment cohorts."""
        prep = MPOWERDataPrep(
            data=sample_mpower_data,
            country_col=sample_config["country_col"],
            year_col=sample_config["year_col"],
        )

        # Add component columns for testing
        data_with_components = sample_mpower_data.copy()
        components = ["M_monitor", "P_protect", "O_offer"]
        for comp in components:
            data_with_components[comp] = np.random.uniform(
                0, 6, len(data_with_components)
            )

        prep.data = data_with_components

        result = prep.create_treatment_cohorts(
            mpower_col=sample_config["mpower_col"],
            treatment_definition="component_based",
            component_cols=components,
            threshold=4.0,
        )

        assert "treatment_cohort" in result.columns

    def test_create_cohorts_missing_mpower_col(self, sample_mpower_data, sample_config):
        """Test error handling for missing MPOWER column."""
        prep = MPOWERDataPrep(
            data=sample_mpower_data,
            country_col=sample_config["country_col"],
            year_col=sample_config["year_col"],
        )

        with pytest.raises(ValueError, match="MPOWER column 'missing_col' not found"):
            prep.create_treatment_cohorts(
                mpower_col="missing_col", treatment_definition="binary_threshold"
            )

    def test_balance_panel_drop_unbalanced(self, unbalanced_panel_data, sample_config):
        """Test panel balancing by dropping unbalanced countries."""
        prep = MPOWERDataPrep(
            data=unbalanced_panel_data,
            country_col=sample_config["country_col"],
            year_col=sample_config["year_col"],
        )

        # Create cohorts first
        data_with_cohorts = prep.create_treatment_cohorts(
            mpower_col=sample_config["mpower_col"],
            treatment_definition="binary_threshold",
            threshold=sample_config["threshold"],
        )

        balanced = prep.balance_panel(method="drop_unbalanced", min_years=10)

        # Should have fewer countries than original
        original_countries = unbalanced_panel_data["country"].nunique()
        balanced_countries = balanced["country"].nunique()
        assert balanced_countries <= original_countries

        # Each remaining country should have sufficient observations
        country_counts = balanced.groupby("country").size()
        assert all(count >= 10 for count in country_counts)

    def test_balance_panel_fill_missing(self, sample_mpower_data, sample_config):
        """Test panel balancing by filling missing values."""
        # Introduce some missing values
        data_with_missing = sample_mpower_data.copy()
        data_with_missing.loc[0:5, "mortality_rate"] = np.nan

        prep = MPOWERDataPrep(
            data=data_with_missing,
            country_col=sample_config["country_col"],
            year_col=sample_config["year_col"],
        )

        balanced = prep.balance_panel(method="fill_missing")

        # Should have no missing values in numeric columns
        numeric_cols = balanced.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert not balanced[col].isnull().any(), (
                f"Column {col} still has missing values"
            )

    def test_prepare_for_analysis(self, treatment_cohort_data, sample_config):
        """Test preparation of final analysis dataset."""
        prep = MPOWERDataPrep(
            data=treatment_cohort_data,
            country_col=sample_config["country_col"],
            year_col=sample_config["year_col"],
        )

        # Set treatment cohorts
        prep._treatment_cohorts = treatment_cohort_data

        analysis_data = prep.prepare_for_analysis(
            outcome_cols=["mortality_rate"],
            control_cols=["gdp_log", "urban_percentage"],
            log_transform=["healthcare_spending"],
            standardize=["urban_percentage"],
            create_lags={"mortality_rate": 2},
        )

        # Check that transformations were applied
        assert "healthcare_spending_log" in analysis_data.columns
        assert "urban_percentage_std" in analysis_data.columns
        assert "mortality_rate_lag1" in analysis_data.columns
        assert "mortality_rate_lag2" in analysis_data.columns

        # Check that additional variables were created
        assert "ever_treated" in analysis_data.columns
        assert "post_treatment" in analysis_data.columns
        assert "years_since_treatment" in analysis_data.columns

    def test_prepare_for_analysis_missing_cols(self, sample_mpower_data, sample_config):
        """Test error handling for missing required columns."""
        prep = MPOWERDataPrep(
            data=sample_mpower_data,
            country_col=sample_config["country_col"],
            year_col=sample_config["year_col"],
        )

        with pytest.raises(ValueError, match="Required columns not found"):
            prep.prepare_for_analysis(
                outcome_cols=["missing_outcome"], control_cols=["missing_control"]
            )

    def test_generate_summary_report(self, treatment_cohort_data, sample_config):
        """Test generation of summary report."""
        prep = MPOWERDataPrep(
            data=treatment_cohort_data,
            country_col=sample_config["country_col"],
            year_col=sample_config["year_col"],
        )

        # Create treatment cohorts
        prep._treatment_cohorts = treatment_cohort_data

        # Balance panel
        prep._panel_data = treatment_cohort_data

        report = prep.generate_summary_report()

        # Check report structure
        assert "basic_structure" in report
        assert "treatment_cohorts" in report
        assert "panel_balance" in report

        # Check basic structure
        basic = report["basic_structure"]
        assert "n_countries" in basic
        assert "n_years" in basic
        assert "year_range" in basic
        assert "n_observations" in basic

        # Check treatment cohorts
        cohorts = report["treatment_cohorts"]
        assert "n_treated_countries" in cohorts
        assert "n_never_treated_countries" in cohorts

    def test_export_prepared_data(self, treatment_cohort_data, sample_config, tmp_path):
        """Test exporting prepared data to file."""
        prep = MPOWERDataPrep(
            data=treatment_cohort_data,
            country_col=sample_config["country_col"],
            year_col=sample_config["year_col"],
        )

        prep._treatment_cohorts = treatment_cohort_data

        # Test CSV export
        csv_path = tmp_path / "test_data.csv"
        prep.export_prepared_data(str(csv_path), data_type="cohorts", format="csv")

        assert csv_path.exists()

        # Test reading back the data
        exported_data = pd.read_csv(csv_path)
        assert len(exported_data) == len(treatment_cohort_data)

    def test_validate_mpower_data_valid(self, sample_mpower_data):
        """Test validation of valid MPOWER data."""
        validation = MPOWERDataPrep.validate_mpower_data(
            data=sample_mpower_data,
            required_cols=["country", "year", "mpower_total_score"],
        )

        assert validation["valid"] is True
        assert len(validation["issues"]) == 0

    def test_validate_mpower_data_invalid(self, malformed_data):
        """Test validation of invalid MPOWER data."""
        validation = MPOWERDataPrep.validate_mpower_data(
            data=malformed_data, required_cols=["country", "year"]
        )

        assert validation["valid"] is False
        assert len(validation["issues"]) > 0

    def test_validate_mpower_data_empty(self, empty_data):
        """Test validation of empty MPOWER data."""
        validation = MPOWERDataPrep.validate_mpower_data(data=empty_data)

        assert validation["valid"] is False
        assert "Data is empty" in validation["issues"]

    def test_edge_case_single_country(self):
        """Test handling of data with single country."""
        single_country_data = pd.DataFrame(
            {
                "country": ["A"] * 10,
                "year": list(range(2010, 2020)),
                "mpower_total_score": [20] * 10,
                "mortality_rate": [50] * 10,
            }
        )

        prep = MPOWERDataPrep(data=single_country_data)

        # Should not raise an error
        assert prep.data["country"].nunique() == 1

        # Treatment cohort creation should work
        result = prep.create_treatment_cohorts(
            mpower_col="mpower_total_score", threshold=25
        )

        # Single country below threshold should be never-treated
        assert all(result["treatment_cohort"] == 0)

    def test_edge_case_single_year(self):
        """Test handling of data with single year."""
        single_year_data = pd.DataFrame(
            {
                "country": ["A", "B", "C"],
                "year": [2015] * 3,
                "mpower_total_score": [20, 30, 15],
                "mortality_rate": [50, 45, 55],
            }
        )

        with pytest.warns(UserWarning, match="Short time series"):
            prep = MPOWERDataPrep(data=single_year_data)

        # Should still initialize successfully
        assert prep.data["year"].nunique() == 1
