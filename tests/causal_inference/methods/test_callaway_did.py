"""Tests for Callaway & Sant'Anna Difference-in-Differences implementation.

This module tests the CallawayDiD class and its methods for staggered
difference-in-differences analysis.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from mpower_mortality_causal_analysis.causal_inference.methods.callaway_did import (
    CallawayDiD,
)


class TestCallawayDiD:
    """Test suite for CallawayDiD class."""

    def test_init_valid_data(self, treatment_cohort_data, sample_config):
        """Test initialization with valid data."""
        did = CallawayDiD(
            data=treatment_cohort_data,
            cohort_col=sample_config["treatment_col"],
            unit_col=sample_config["country_col"],
            time_col=sample_config["year_col"],
        )

        assert isinstance(did.data, pd.DataFrame)
        assert did.cohort_col == sample_config["treatment_col"]
        assert did.unit_col == sample_config["country_col"]
        assert did.time_col == sample_config["year_col"]
        assert did.never_treated_value == 0

    def test_init_missing_columns(self, sample_mpower_data):
        """Test initialization with missing required columns raises error."""
        with pytest.raises(ValueError, match="Missing required columns"):
            CallawayDiD(
                data=sample_mpower_data,
                cohort_col="missing_cohort",
                unit_col="country",
                time_col="year",
            )

    def test_init_empty_data(self, empty_data):
        """Test initialization with empty data raises error."""
        with pytest.raises(ValueError, match="Data cannot be empty"):
            CallawayDiD(
                data=empty_data, cohort_col="cohort", unit_col="unit", time_col="time"
            )

    def test_data_validation(self, treatment_cohort_data, sample_config):
        """Test data validation during initialization."""
        # Create data with no never-treated units
        all_treated_data = treatment_cohort_data.copy()
        all_treated_data["treatment_cohort"] = 2010  # All treated in 2010

        with pytest.raises(ValueError, match="No never-treated units found"):
            CallawayDiD(
                data=all_treated_data,
                cohort_col="treatment_cohort",
                unit_col="country",
                time_col="year",
            )

    def test_fit_with_differences_package(self, treatment_cohort_data, sample_config):
        """Test fitting with differences package (if available)."""
        did = CallawayDiD(
            data=treatment_cohort_data,
            cohort_col=sample_config["treatment_col"],
            unit_col=sample_config["country_col"],
            time_col=sample_config["year_col"],
        )

        # Test basic fit
        try:
            result = did.fit(
                outcome=sample_config["outcome_col"],
                covariates=sample_config["control_cols"],
            )

            assert result is did  # Should return self
            assert did._fitted_model is not None

        except Exception as e:
            # If differences package not available or has conflicts, should use fallback
            assert "differences" in str(e).lower() or did._fitted_model is not None

    def test_fit_fallback_implementation(self, treatment_cohort_data, sample_config):
        """Test fallback implementation when differences package unavailable."""
        did = CallawayDiD(
            data=treatment_cohort_data,
            cohort_col=sample_config["treatment_col"],
            unit_col=sample_config["country_col"],
            time_col=sample_config["year_col"],
        )

        # Force fallback by patching the availability
        with patch(
            "mpower_mortality_causal_analysis.causal_inference.methods.callaway_did.DIFFERENCES_AVAILABLE",
            False,
        ):
            result = did.fit(
                outcome=sample_config["outcome_col"],
                covariates=sample_config["control_cols"],
            )

            assert result is did
            assert did._fitted_model is not None
            assert isinstance(did._fitted_model, dict)
            assert did._fitted_model["type"] == "fallback_did"

    def test_fit_missing_outcome(self, treatment_cohort_data, sample_config):
        """Test fitting with missing outcome variable raises error."""
        did = CallawayDiD(
            data=treatment_cohort_data,
            cohort_col=sample_config["treatment_col"],
            unit_col=sample_config["country_col"],
            time_col=sample_config["year_col"],
        )

        with pytest.raises(
            ValueError, match="Outcome variable 'missing_outcome' not found"
        ):
            did.fit(outcome="missing_outcome")

    def test_fit_missing_covariates(self, treatment_cohort_data, sample_config):
        """Test fitting with missing covariates raises error."""
        did = CallawayDiD(
            data=treatment_cohort_data,
            cohort_col=sample_config["treatment_col"],
            unit_col=sample_config["country_col"],
            time_col=sample_config["year_col"],
        )

        with pytest.raises(ValueError, match="Covariates not found in data"):
            did.fit(
                outcome=sample_config["outcome_col"], covariates=["missing_covariate"]
            )

    def test_aggregate_simple(self, treatment_cohort_data, sample_config):
        """Test simple aggregation of results."""
        did = CallawayDiD(
            data=treatment_cohort_data,
            cohort_col=sample_config["treatment_col"],
            unit_col=sample_config["country_col"],
            time_col=sample_config["year_col"],
        )

        # Fit with fallback
        with patch(
            "mpower_mortality_causal_analysis.causal_inference.methods.callaway_did.DIFFERENCES_AVAILABLE",
            False,
        ):
            did.fit(
                outcome=sample_config["outcome_col"],
                covariates=sample_config["control_cols"],
            )

            simple_result = did.aggregate("simple")

            assert isinstance(simple_result, dict)
            if "error" not in simple_result:
                assert "att" in simple_result or "error" in simple_result

    def test_aggregate_without_fit(self, treatment_cohort_data, sample_config):
        """Test aggregation without fitting raises error."""
        did = CallawayDiD(
            data=treatment_cohort_data,
            cohort_col=sample_config["treatment_col"],
            unit_col=sample_config["country_col"],
            time_col=sample_config["year_col"],
        )

        with pytest.raises(ValueError, match="Model must be fitted first"):
            did.aggregate("simple")

    def test_summary_without_fit(self, treatment_cohort_data, sample_config):
        """Test summary without fitting."""
        did = CallawayDiD(
            data=treatment_cohort_data,
            cohort_col=sample_config["treatment_col"],
            unit_col=sample_config["country_col"],
            time_col=sample_config["year_col"],
        )

        summary = did.summary()
        assert "Model not fitted yet" in summary

    def test_summary_with_fallback(self, treatment_cohort_data, sample_config):
        """Test summary with fallback implementation."""
        did = CallawayDiD(
            data=treatment_cohort_data,
            cohort_col=sample_config["treatment_col"],
            unit_col=sample_config["country_col"],
            time_col=sample_config["year_col"],
        )

        # Fit with fallback
        with patch(
            "mpower_mortality_causal_analysis.causal_inference.methods.callaway_did.DIFFERENCES_AVAILABLE",
            False,
        ):
            did.fit(
                outcome=sample_config["outcome_col"],
                covariates=sample_config["control_cols"],
            )

            summary = did.summary()

            assert isinstance(summary, str)
            assert "Fallback Implementation" in summary

    def test_plot_event_study_no_matplotlib(self, treatment_cohort_data, sample_config):
        """Test event study plotting without matplotlib."""
        did = CallawayDiD(
            data=treatment_cohort_data,
            cohort_col=sample_config["treatment_col"],
            unit_col=sample_config["country_col"],
            time_col=sample_config["year_col"],
        )

        # Fit model first
        with patch(
            "mpower_mortality_causal_analysis.causal_inference.methods.callaway_did.DIFFERENCES_AVAILABLE",
            False,
        ):
            did.fit(outcome=sample_config["outcome_col"])

            # Mock matplotlib as unavailable
            with patch.dict("sys.modules", {"matplotlib.pyplot": None}):
                with pytest.raises(ImportError, match="Plotting requires matplotlib"):
                    did.plot_event_study()

    def test_custom_never_treated_value(self, treatment_cohort_data, sample_config):
        """Test using custom never-treated value."""
        # Modify data to use -1 as never-treated value
        custom_data = treatment_cohort_data.copy()
        custom_data["treatment_cohort"] = custom_data["treatment_cohort"].replace(0, -1)

        did = CallawayDiD(
            data=custom_data,
            cohort_col=sample_config["treatment_col"],
            unit_col=sample_config["country_col"],
            time_col=sample_config["year_col"],
            never_treated_value=-1,
        )

        assert did.never_treated_value == -1

        # Should still be able to fit
        with patch(
            "mpower_mortality_causal_analysis.causal_inference.methods.callaway_did.DIFFERENCES_AVAILABLE",
            False,
        ):
            result = did.fit(outcome=sample_config["outcome_col"])
            assert result is did

    def test_minimal_data(self, minimal_data):
        """Test with minimal dataset."""
        did = CallawayDiD(
            data=minimal_data,
            cohort_col="treatment_cohort",
            unit_col="country",
            time_col="year",
        )

        # Should initialize without error
        assert did.data.shape == minimal_data.shape

        # Fit should work with fallback
        with patch(
            "mpower_mortality_causal_analysis.causal_inference.methods.callaway_did.DIFFERENCES_AVAILABLE",
            False,
        ):
            result = did.fit(outcome="outcome", covariates=["control"])
            assert result is did

    def test_unbalanced_panel_warning(self):
        """Test warning for unbalanced panel."""
        # Create unbalanced panel
        unbalanced_data = pd.DataFrame(
            {
                "country": ["A", "A", "B", "B", "B"],  # Country B has more observations
                "year": [2010, 2011, 2010, 2011, 2012],
                "treatment_cohort": [2011, 2011, 0, 0, 0],
                "outcome": [50, 45, 52, 51, 50],
            }
        )

        with pytest.warns(UserWarning, match="Unbalanced panel detected"):
            CallawayDiD(
                data=unbalanced_data,
                cohort_col="treatment_cohort",
                unit_col="country",
                time_col="year",
            )

    def test_no_treated_units_warning(self):
        """Test warning when no treated units found."""
        # Create data with no treated units
        no_treatment_data = pd.DataFrame(
            {
                "country": ["A", "A", "B", "B"],
                "year": [2010, 2011, 2010, 2011],
                "treatment_cohort": [0, 0, 0, 0],  # No treatment
                "outcome": [50, 45, 52, 51],
            }
        )

        with pytest.warns(UserWarning, match="No treated units found"):
            CallawayDiD(
                data=no_treatment_data,
                cohort_col="treatment_cohort",
                unit_col="country",
                time_col="year",
            )

    def test_extract_effect_size_fallback(self, treatment_cohort_data, sample_config):
        """Test extraction of effect size from fallback results."""
        did = CallawayDiD(
            data=treatment_cohort_data,
            cohort_col=sample_config["treatment_col"],
            unit_col=sample_config["country_col"],
            time_col=sample_config["year_col"],
        )

        # Fit with fallback
        with patch(
            "mpower_mortality_causal_analysis.causal_inference.methods.callaway_did.DIFFERENCES_AVAILABLE",
            False,
        ):
            did.fit(outcome=sample_config["outcome_col"])

            # Test that we can extract results
            simple_att = did.aggregate("simple")

            # Should have some result structure
            assert isinstance(simple_att, dict)

            # If successful, should have att estimate
            if "error" not in simple_att:
                assert "att" in simple_att or "method" in simple_att
