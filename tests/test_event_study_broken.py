"""Unit tests for event study analysis utilities.

Tests cover:
- Event time variable creation
- Event time dummy generation
- Event study estimation
- Parallel trends testing
- Plotting functionality
- Edge cases and error handling
"""

from unittest.mock import Mock, patch

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
        """Create sample panel data for event study testing."""
        np.random.seed(42)

        countries = [f"Country_{i}" for i in range(12)]
        years = list(range(2008, 2019))

        data_list = []

        for country in countries:
            # Assign treatment timing
            if country in countries[:8]:  # Some treated
                treatment_year = np.random.choice(
                    [2012, 2014, 2016, 0], p=[0.3, 0.3, 0.3, 0.1]
                )
            else:
                treatment_year = 0  # Never treated

            for year in years:
                # Simulate outcome with pre-trends and treatment effects
                base_outcome = 50 + 0.5 * (year - 2008)  # Linear time trend

                # Add treatment effect
                if treatment_year > 0 and year >= treatment_year:
                    # Dynamic treatment effect
                    years_since_treatment = year - treatment_year
                    treatment_effect = (
                        -2 - 0.5 * years_since_treatment + np.random.normal(0, 0.5)
                    )
                    outcome = base_outcome + treatment_effect
                else:
                    outcome = base_outcome + np.random.normal(0, 2)

                # Add control variables
                gdp_log = 8 + 0.02 * (year - 2008) + np.random.normal(0, 0.3)
                urban_pct = 60 + 0.5 * (year - 2008) + np.random.normal(0, 5)

                data_list.append(
                    {
                        "country": country,
                        "year": year,
                        "treatment_year": treatment_year,
                        "mortality_rate": outcome,
                        "gdp_per_capita_log": gdp_log,
                        "urban_population_pct": urban_pct,
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

        # Check that event time was created
        assert "event_time" in event_study.data.columns

        # Test with missing required columns
        invalid_data = sample_data.drop("country", axis=1)
        with pytest.raises(ValueError, match="Missing required columns"):
            EventStudyAnalysis(
                data=invalid_data,
                unit_col="country",
                time_col="year",
                treatment_col="treatment_year",
            )

    def test_event_time_creation(self, sample_data):
        """Test event time variable creation."""
        event_study = EventStudyAnalysis(
            data=sample_data,
            unit_col="country",
            time_col="year",
            treatment_col="treatment_year",
            never_treated_value=0,
        )

        # Check event time for treated units
        treated_data = event_study.data[event_study.data["treatment_year"] > 0]
        for _, row in treated_data.iterrows():
            expected_event_time = row["year"] - row["treatment_year"]
            assert row["event_time"] == expected_event_time

        # Check event time for never-treated units
        never_treated_data = event_study.data[event_study.data["treatment_year"] == 0]
        assert all(never_treated_data["event_time"] == np.inf)

    def test_create_event_time_dummies(self, sample_data):
        """Test event time dummy variable creation."""
        event_study = EventStudyAnalysis(
            data=sample_data,
            unit_col="country",
            time_col="year",
            treatment_col="treatment_year",
            never_treated_value=0,
        )

        data_with_dummies = event_study.create_event_time_dummies(
            max_lag=3, max_lead=3, reference_period=-1
        )

        # Check that dummy variables were created
        expected_dummies = [
            "event_time_lead_3",
            "event_time_lead_2",
            "event_time_0",
            "event_time_lag_1",
            "event_time_lag_2",
            "event_time_lag_3",
        ]

        for dummy in expected_dummies:
            assert dummy in data_with_dummies.columns
            # Check that dummies are binary
            assert data_with_dummies[dummy].isin([0, 1]).all()

        # Reference period should not be included
        assert "event_time_lead_1" not in data_with_dummies.columns

        # Check endpoint bins
        assert "event_time_lead_3_plus" in data_with_dummies.columns
        assert "event_time_lag_3_plus" in data_with_dummies.columns

    def test_estimate_fixed_effects(self, sample_data):
        """Test event study estimation with fixed effects."""
        event_study = EventStudyAnalysis(
            data=sample_data,
            unit_col="country",
            time_col="year",
            treatment_col="treatment_year",
            never_treated_value=0,
        )

        # Mock statsmodels for testing
        with patch("statsmodels.api") as mock_sm:
            # Mock OLS model
            mock_model = Mock()
            mock_model.fit.return_value = mock_model
            mock_model.params = pd.Series(
                {
                    "event_time_lead_2": 0.5,
                    "event_time_0": -1.5,
                    "event_time_lag_1": -2.0,
                    "event_time_lag_2": -2.5,
                }
            )
            mock_model.bse = pd.Series(
                {
                    "event_time_lead_2": 0.3,
                    "event_time_0": 0.4,
                    "event_time_lag_1": 0.5,
                    "event_time_lag_2": 0.6,
                }
            )
            mock_model.pvalues = pd.Series(
                {
                    "event_time_lead_2": 0.1,
                    "event_time_0": 0.001,
                    "event_time_lag_1": 0.0001,
                    "event_time_lag_2": 0.0001,
                }
            )
            mock_model.rsquared = 0.75
            mock_model.nobs = 100

            mock_sm.OLS.return_value = mock_model
            mock_sm.add_constant.return_value = pd.DataFrame()

            results = event_study.estimate(
                outcome="mortality_rate",
                covariates=["gdp_per_capita_log"],
                max_lag=3,
                max_lead=3,
                method="fixed_effects",
            )

            # Check results structure
            assert isinstance(results, dict)
            assert "event_time_coefficients" in results
            assert "event_time_std_errors" in results
            assert "event_time_pvalues" in results
            assert "method" in results
            assert results["method"] == "fixed_effects"

            # Check specific coefficients
            coeffs = results["event_time_coefficients"]
            assert "event_time_lead_2" in coeffs
            assert "event_time_0" in coeffs
            assert coeffs["event_time_0"] == -1.5

    def test_estimate_ols(self, sample_data):
        """Test event study estimation with OLS."""
        event_study = EventStudyAnalysis(
            data=sample_data,
            unit_col="country",
            time_col="year",
            treatment_col="treatment_year",
            never_treated_value=0,
        )

        # Mock statsmodels for testing
        with patch("statsmodels.api") as mock_sm:
            mock_model = Mock()
            mock_model.fit.return_value = mock_model
            mock_model.params = pd.Series({"event_time_0": -1.5})
            mock_model.bse = pd.Series({"event_time_0": 0.4})
            mock_model.pvalues = pd.Series({"event_time_0": 0.001})
            mock_model.rsquared = 0.65
            mock_model.nobs = 100

            mock_sm.OLS.return_value = mock_model
            mock_sm.add_constant.return_value = pd.DataFrame()

            results = event_study.estimate(outcome="mortality_rate", method="ols")

            assert results["method"] == "ols"
            assert "event_time_coefficients" in results

    def test_parallel_trends_testing(self, sample_data):
        """Test parallel trends testing functionality."""
        event_study = EventStudyAnalysis(
            data=sample_data,
            unit_col="country",
            time_col="year",
            treatment_col="treatment_year",
            never_treated_value=0,
        )

        # Mock estimation results
        mock_results = {
            "event_time_coefficients": {
                "event_time_lead_3": 0.2,
                "event_time_lead_2": 0.1,
                "event_time_0": -1.5,
                "event_time_lag_1": -2.0,
            },
            "event_time_pvalues": {
                "event_time_lead_3": 0.4,
                "event_time_lead_2": 0.8,
                "event_time_0": 0.001,
                "event_time_lag_1": 0.0001,
            },
            "event_time_std_errors": {
                "event_time_lead_3": 0.3,
                "event_time_lead_2": 0.4,
                "event_time_0": 0.4,
                "event_time_lag_1": 0.5,
            },
            "model": Mock(),
        }

        # Mock the F-test and trend test
        with patch.object(event_study, "_conduct_joint_f_test", return_value=0.6):
            with patch.object(
                event_study,
                "_test_linear_pre_trend",
                return_value={
                    "test": "linear_trend",
                    "pvalue": 0.3,
                    "significant": False,
                    "trend_coefficient": 0.05,
                },
            ):
                pt_results = event_study.test_parallel_trends(mock_results, max_lead=3)

                # Check results structure
                assert isinstance(pt_results, dict)
                assert "n_pre_treatment_periods" in pt_results
                assert "significant_individual_tests" in pt_results
                assert "joint_f_test_pvalue" in pt_results
                assert "linear_trend_test" in pt_results
                assert "conclusion" in pt_results

                # Check specific values
                assert pt_results["n_pre_treatment_periods"] == 2
                assert (
                    pt_results["significant_individual_tests"] == 0
                )  # No significant pre-trends
                assert pt_results["joint_f_test_pvalue"] == 0.6

    def test_comprehensive_parallel_trends_analysis(self, sample_data):
        """Test comprehensive parallel trends analysis."""
        event_study = EventStudyAnalysis(
            data=sample_data,
            unit_col="country",
            time_col="year",
            treatment_col="treatment_year",
            never_treated_value=0,
        )

        # Mock the estimate method
        mock_estimate_results = {
            "event_time_coefficients": {"event_time_lead_2": 0.1, "event_time_0": -1.5},
            "event_time_pvalues": {"event_time_lead_2": 0.7, "event_time_0": 0.001},
            "event_time_std_errors": {"event_time_lead_2": 0.3, "event_time_0": 0.4},
            "model": Mock(),
        }

        with patch.object(event_study, "estimate", return_value=mock_estimate_results):
            with patch.object(event_study, "test_parallel_trends") as mock_test_pt:
                mock_test_pt.return_value = {
                    "conclusion": "Parallel trends assumption supported",
                    "joint_f_test_pvalue": 0.8,
                }

                comprehensive_results = (
                    event_study.comprehensive_parallel_trends_analysis(
                        outcome="mortality_rate", max_lead=3
                    )
                )

                # Check structure
                assert isinstance(comprehensive_results, dict)
                assert "event_study_results" in comprehensive_results
                assert "parallel_trends_tests" in comprehensive_results
                assert "visual_inspection_data" in comprehensive_results
                assert "placebo_test_suggestions" in comprehensive_results
                assert "overall_assessment" in comprehensive_results

    def test_plot_event_study_mock(self, sample_data):
        """Test event study plotting with mocked matplotlib."""
        event_study = EventStudyAnalysis(
            data=sample_data,
            unit_col="country",
            time_col="year",
            treatment_col="treatment_year",
            never_treated_value=0,
        )

        mock_results = {
            "event_time_coefficients": {
                "event_time_lead_2": 0.1,
                "event_time_0": -1.5,
                "event_time_lag_1": -2.0,
            },
            "event_time_std_errors": {
                "event_time_lead_2": 0.3,
                "event_time_0": 0.4,
                "event_time_lag_1": 0.5,
            },
            "outcome": "mortality_rate",
        }

        # Mock matplotlib
        with patch(
            "mpower_mortality_causal_analysis.causal_inference.utils.event_study.PLOTTING_AVAILABLE",
            True,
        ):
            with patch(
                "mpower_mortality_causal_analysis.causal_inference.utils.event_study.plt"
            ) as mock_plt:
                mock_fig = Mock()
                mock_ax = Mock()
                mock_plt.subplots.return_value = (mock_fig, mock_ax)

                fig = event_study.plot_event_study(mock_results)

                # Check that plotting functions were called
                mock_plt.subplots.assert_called()
                mock_ax.plot.assert_called()

        # Test without plotting available
        with patch(
            "mpower_mortality_causal_analysis.causal_inference.utils.event_study.PLOTTING_AVAILABLE",
            False,
        ):
            with pytest.raises(ImportError, match="Plotting requires matplotlib"):
                event_study.plot_event_study(mock_results)

    def test_joint_f_test(self, sample_data):
        """Test joint F-test implementation."""
        event_study = EventStudyAnalysis(
            data=sample_data,
            unit_col="country",
            time_col="year",
            treatment_col="treatment_year",
            never_treated_value=0,
        )

        # Mock model with proper structure
        mock_model = Mock()
        mock_model.params = pd.Series(
            {
                "const": 50.0,
                "event_time_lead_2": 0.1,
                "event_time_lead_1": 0.05,
                "event_time_0": -1.5,
            }
        )
        mock_model.params.index = [
            "const",
            "event_time_lead_2",
            "event_time_lead_1",
            "event_time_0",
        ]
        # Cannot set values directly on Series, so just ensure params has the right shape
        # mock_model.params.values = np.array([50.0, 0.1, 0.05, -1.5])

        # Mock covariance matrix
        cov_matrix = np.eye(4) * 0.25  # Simple diagonal covariance
        mock_model.cov_params.return_value = pd.DataFrame(
            cov_matrix, index=mock_model.params.index, columns=mock_model.params.index
        )
        mock_model.df_resid = 95

        mock_results = {"model": mock_model}
        pre_treatment_vars = ["event_time_lead_2", "event_time_lead_1"]

        # Test F-test
        with patch("scipy.stats") as mock_stats:
            mock_stats.f.cdf.return_value = 0.3  # Mock CDF value

            p_value = event_study._conduct_joint_f_test(
                mock_results, pre_treatment_vars
            )

            assert p_value is not None
            assert isinstance(p_value, (int, float))
            assert 0 <= p_value <= 1

    def test_linear_trend_test(self, sample_data):
        """Test linear pre-trend testing."""
        event_study = EventStudyAnalysis(
            data=sample_data,
            unit_col="country",
            time_col="year",
            treatment_col="treatment_year",
            never_treated_value=0,
        )

        # Test with sufficient data
        coeffs = [0.1, 0.05, -0.02]  # Slight downward trend
        ses = [0.3, 0.25, 0.2]

        with patch("scipy.stats") as mock_stats:
            mock_stats.t.cdf.return_value = 0.7  # Mock CDF value

            trend_result = event_study._test_linear_pre_trend(coeffs, ses)

            assert isinstance(trend_result, dict)
            assert "test" in trend_result
            assert "trend_coefficient" in trend_result
            assert "pvalue" in trend_result
            assert trend_result["test"] == "linear_trend"

        # Test with insufficient data
        insufficient_coeffs = [0.1, 0.05]  # Only 2 points
        insufficient_ses = [0.3, 0.25]

        trend_result = event_study._test_linear_pre_trend(
            insufficient_coeffs, insufficient_ses
        )
        assert trend_result["test"] == "insufficient_data"
        assert trend_result["pvalue"] is None

    def test_visual_inspection_data(self, sample_data):
        """Test visual inspection data preparation."""
        event_study = EventStudyAnalysis(
            data=sample_data,
            unit_col="country",
            time_col="year",
            treatment_col="treatment_year",
            never_treated_value=0,
        )

        visual_data = event_study._prepare_visual_inspection_data(
            "mortality_rate", max_lead=3
        )

        # Check structure
        assert isinstance(visual_data, dict)
        assert "cohort_time_means" in visual_data
        assert "outcome_variable" in visual_data
        assert "time_range" in visual_data
        assert "n_cohorts" in visual_data

        # Check cohort data
        cohort_data = visual_data["cohort_time_means"]
        assert isinstance(cohort_data, list)
        assert len(cohort_data) > 0

        # Each cohort should have required fields
        for cohort in cohort_data:
            assert "cohort" in cohort
            assert "is_treated" in cohort
            assert "yearly_means" in cohort

    def test_placebo_test_suggestions(self, sample_data):
        """Test placebo test suggestions."""
        event_study = EventStudyAnalysis(
            data=sample_data,
            unit_col="country",
            time_col="year",
            treatment_col="treatment_year",
            never_treated_value=0,
        )

        suggestions = event_study._suggest_placebo_tests()

        # Check structure
        assert isinstance(suggestions, dict)
        assert "placebo_tests" in suggestions
        assert "recommendation" in suggestions

        # Check that suggestions were generated
        placebo_tests = suggestions["placebo_tests"]
        assert isinstance(placebo_tests, list)
        assert len(placebo_tests) > 0

        # Check suggestion structure
        for test in placebo_tests:
            assert "type" in test
            assert "description" in test

    def test_error_handling(self, sample_data):
        """Test error handling for various edge cases."""
        event_study = EventStudyAnalysis(
            data=sample_data,
            unit_col="country",
            time_col="year",
            treatment_col="treatment_year",
            never_treated_value=0,
        )

        # Test with missing outcome variable
        with pytest.raises(
            ValueError, match="Outcome variable 'nonexistent' not found"
        ):
            event_study.estimate(outcome="nonexistent")

        # Test with missing covariates
        with pytest.raises(ValueError, match="Covariates not found in data"):
            event_study.estimate(
                outcome="mortality_rate", covariates=["nonexistent_var"]
            )

        # Test with unknown method
        with pytest.raises(ValueError, match="Unknown estimation method"):
            event_study.estimate(outcome="mortality_rate", method="invalid_method")

    def test_summary_statistics(self, sample_data):
        """Test summary statistics generation."""
        event_study = EventStudyAnalysis(
            data=sample_data,
            unit_col="country",
            time_col="year",
            treatment_col="treatment_year",
            never_treated_value=0,
        )

        summary = event_study.summary_statistics()

        # Check that it returns a DataFrame
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) > 0

        # Check expected columns
        expected_cols = [
            "total_units",
            "total_periods",
            "treated_observations",
            "never_treated_observations",
            "treatment_years",
        ]

        for col in expected_cols:
            assert col in summary.columns

    def test_edge_case_no_treatment(self):
        """Test handling of data with no treatment variation."""
        # Create data with no treated units
        no_treatment_data = pd.DataFrame(
            {
                "country": ["A", "A", "B", "B"],
                "year": [2010, 2011, 2010, 2011],
                "treatment_year": [0, 0, 0, 0],
                "mortality_rate": [50, 51, 52, 53],
            }
        )

        event_study = EventStudyAnalysis(
            data=no_treatment_data,
            unit_col="country",
            time_col="year",
            treatment_col="treatment_year",
            never_treated_value=0,
        )

        # All event times should be infinite (never treated)
        assert all(event_study.data["event_time"] == np.inf)

        # Creating dummies should still work
        data_with_dummies = event_study.create_event_time_dummies()
        assert isinstance(data_with_dummies, pd.DataFrame)

    def test_edge_case_all_treatment(self):
        """Test handling of data where all units are treated."""
        # Create data with all treated units
        all_treatment_data = pd.DataFrame(
            {
                "country": ["A", "A", "B", "B"],
                "year": [2010, 2011, 2010, 2011],
                "treatment_year": [2010, 2010, 2010, 2010],
                "mortality_rate": [50, 51, 52, 53],
            }
        )

        event_study = EventStudyAnalysis(
            data=all_treatment_data,
            unit_col="country",
            time_col="year",
            treatment_col="treatment_year",
            never_treated_value=0,
        )

        # Event times should be calculated correctly
        expected_event_times = [0, 1, 0, 1]  # Years relative to 2010 treatment
        actual_event_times = event_study.data["event_time"].tolist()
        assert actual_event_times == expected_event_times
