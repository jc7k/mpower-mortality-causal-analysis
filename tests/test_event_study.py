"""Unit tests for event study analysis utilities - Fixed version.

Tests cover:
- Event study initialization
- Event time dummy creation
- Basic functionality without complex mocking
"""

import numpy as np
import pandas as pd
import pytest

from mpower_mortality_causal_analysis.causal_inference.utils.event_study import (
    EventStudyAnalysis,
)


class TestEventStudyAnalysis:
    """Test suite for event study analysis utilities."""

    @pytest.fixture
    def sample_data(self):
        """Create sample event study data."""
        np.random.seed(42)
        countries = [f"Country_{i}" for i in range(12)]
        years = list(range(2008, 2019))

        data_list = []
        for country in countries:
            # Some countries get treated in different years
            if country in countries[:6]:
                treatment_year = np.random.choice([2012, 2014, 2016])
            else:
                treatment_year = 0  # Never treated

            for year in years:
                outcome = 45 + np.random.normal(0, 5)
                # Add treatment effect
                if treatment_year > 0 and year >= treatment_year:
                    outcome += np.random.normal(-3, 1)

                data_list.append(
                    {
                        "country": country,
                        "year": year,
                        "treatment_year": treatment_year,
                        "outcome": outcome,
                        "gdp_per_capita_log": 8 + np.random.normal(0, 0.5),
                        "urban_population_pct": 55 + np.random.normal(0, 10),
                    }
                )

        return pd.DataFrame(data_list)

    def test_initialization(self, sample_data):
        """Test proper initialization of EventStudyAnalysis."""
        event_study = EventStudyAnalysis(
            data=sample_data,
            unit_col="country",
            time_col="year",
            treatment_col="treatment_year",
            never_treated_value=0,
        )

        assert event_study.unit_col == "country"
        assert event_study.time_col == "year"
        assert event_study.treatment_col == "treatment_year"
        assert event_study.never_treated_value == 0

    def test_event_time_creation(self, sample_data):
        """Test event time variable creation."""
        event_study = EventStudyAnalysis(
            data=sample_data,
            unit_col="country",
            time_col="year",
            treatment_col="treatment_year",
            never_treated_value=0,
        )

        # Should have event_time column after initialization
        assert "event_time" in event_study.data.columns

        # Check event time calculation
        treated_data = event_study.data[event_study.data["treatment_year"] > 0]
        for _, row in treated_data.head(5).iterrows():  # Check a few rows
            expected_event_time = row["year"] - row["treatment_year"]
            assert row["event_time"] == expected_event_time

    def test_create_event_time_dummies(self, sample_data):
        """Test creation of event time dummy variables."""
        event_study = EventStudyAnalysis(
            data=sample_data,
            unit_col="country",
            time_col="year",
            treatment_col="treatment_year",
            never_treated_value=0,
        )

        result = event_study.create_event_time_dummies()

        assert isinstance(result, pd.DataFrame)

        # Should have dummy variables for different event times
        dummy_cols = [col for col in result.columns if col.startswith("event_time_")]
        assert len(dummy_cols) > 0

        # All dummy variables should be 0 or 1
        for col in dummy_cols:
            assert result[col].isin([0, 1]).all()

    def test_basic_functionality(self, sample_data):
        """Test basic functionality without complex method calls."""
        event_study = EventStudyAnalysis(
            data=sample_data,
            unit_col="country",
            time_col="year",
            treatment_col="treatment_year",
            never_treated_value=0,
        )

        # Test that data is processed correctly
        assert hasattr(event_study, "data")
        assert len(event_study.data) > 0

        # Test that event time column exists
        assert "event_time" in event_study.data.columns

        # Test that the class maintains its configuration
        assert event_study.unit_col == "country"
        assert event_study.time_col == "year"
        assert event_study.treatment_col == "treatment_year"

    def test_error_handling(self, sample_data):
        """Test error handling for invalid inputs."""
        # Test with missing required columns
        invalid_data = sample_data.drop("country", axis=1)

        with pytest.raises(ValueError):
            EventStudyAnalysis(
                data=invalid_data,
                unit_col="country",
                time_col="year",
                treatment_col="treatment_year",
                never_treated_value=0,
            )

    def test_edge_case_no_treatment(self):
        """Test with data containing no treated units."""
        # Create data with no treatment
        data = pd.DataFrame(
            {
                "country": ["A", "B", "C"] * 3,
                "year": [2010, 2010, 2010, 2011, 2011, 2011, 2012, 2012, 2012],
                "treatment_year": [0] * 9,
                "outcome": np.random.normal(45, 5, 9),
            }
        )

        event_study = EventStudyAnalysis(
            data=data,
            unit_col="country",
            time_col="year",
            treatment_col="treatment_year",
            never_treated_value=0,
        )

        # Just test that initialization works
        assert len(event_study.data) == 9
        assert "event_time" in event_study.data.columns

    def test_edge_case_all_treatment(self):
        """Test with data where all units are treated."""
        # Create data where all units are treated
        data = pd.DataFrame(
            {
                "country": ["A", "B", "C"] * 3,
                "year": [2010, 2010, 2010, 2011, 2011, 2011, 2012, 2012, 2012],
                "treatment_year": [2011, 2011, 2011] * 3,
                "outcome": np.random.normal(45, 5, 9),
            }
        )

        event_study = EventStudyAnalysis(
            data=data,
            unit_col="country",
            time_col="year",
            treatment_col="treatment_year",
            never_treated_value=0,
        )

        # Just test that initialization works
        assert len(event_study.data) == 9
        assert "event_time" in event_study.data.columns
