"""Comprehensive tests for Advanced DiD Methods Extension.

This module tests all components of the advanced DiD implementation,
including Sun & Abraham, Borusyak, DCDH, and Doubly Robust estimators.
"""

import time

import numpy as np
import pandas as pd
import pytest

from mpower_mortality_causal_analysis.extensions.advanced_did import (
    BorusyakImputation,
    DCDHEstimator,
    DoublyRobustDiD,
    MethodComparison,
    SunAbrahamEstimator,
)
from mpower_mortality_causal_analysis.extensions.advanced_did.diagnostics import (
    check_common_support,
    compute_effective_sample_size,
    test_no_anticipation,
    test_parallel_trends,
)

# Test constants
RANDOM_SEED = 42
N_UNITS = 100
N_PERIODS = 10
SIGNIFICANCE_LEVEL = 0.05
NEVER_TREATED_VALUE = 9999
TREATMENT_CUTOFF = 50
MAX_TEST_TIME_SECONDS = 30


@pytest.fixture
def sample_panel_data():
    """Generate sample panel data for testing."""
    np.random.seed(RANDOM_SEED)

    # Create panel structure
    units = range(N_UNITS)
    periods = range(2008, 2008 + N_PERIODS)

    data = []
    for unit in units:
        for period in periods:
            data.append(
                {
                    "unit": unit,
                    "time": period,
                    "cohort": np.random.choice([2010, 2012, NEVER_TREATED_VALUE])
                    if unit < TREATMENT_CUTOFF
                    else NEVER_TREATED_VALUE,
                    "outcome": np.random.normal(50, 10) + 0.5 * (period - 2008),
                    "covariate1": np.random.normal(0, 1),
                    "covariate2": np.random.uniform(0, 10),
                    "treatment": 0,
                }
            )

    df = pd.DataFrame(data)

    # Add treatment effects
    treated_mask = (df["cohort"] != NEVER_TREATED_VALUE) & (df["time"] >= df["cohort"])
    df.loc[treated_mask, "treatment"] = 1
    df.loc[treated_mask, "outcome"] += np.random.normal(5, 2, treated_mask.sum())

    return df


@pytest.fixture
def simple_did_data():
    """Generate simple 2x2 DiD data for basic tests."""
    np.random.seed(RANDOM_SEED)

    data = []
    # Control group (never treated)
    for unit in range(25):
        data.append(
            {
                "unit": unit,
                "time": 0,
                "treatment": 0,
                "outcome": 10 + np.random.normal(0, 1),
            }
        )
        data.append(
            {
                "unit": unit,
                "time": 1,
                "treatment": 0,
                "outcome": 12 + np.random.normal(0, 1),
            }
        )

    # Treatment group
    for unit in range(25, 50):
        data.append(
            {
                "unit": unit,
                "time": 0,
                "treatment": 0,
                "outcome": 10 + np.random.normal(0, 1),
            }
        )
        data.append(
            {
                "unit": unit,
                "time": 1,
                "treatment": 1,
                "outcome": 17 + np.random.normal(0, 1),
            }
        )  # Effect = 5

    df = pd.DataFrame(data)
    df["cohort"] = df["unit"].map(lambda x: 1 if x >= 25 else 9999)
    df["covariate1"] = np.random.normal(0, 1, len(df))

    return df


class TestSunAbrahamEstimator:
    """Test Sun & Abraham (2021) estimator."""

    def test_initialization(self, sample_panel_data):
        """Test estimator initialization."""
        estimator = SunAbrahamEstimator(
            sample_panel_data, cohort_col="cohort", time_col="time", unit_col="unit"
        )

        assert estimator.data is not None, "Data should be initialized"
        assert len(estimator.units) == N_UNITS, f"Expected {N_UNITS} units"
        assert len(estimator.time_periods) == N_PERIODS, f"Expected {N_PERIODS} periods"
        assert estimator.never_treated is not None, "Never treated value should be set"

    def test_invalid_data(self):
        """Test error handling for invalid data."""
        invalid_data = pd.DataFrame({"x": [1, 2], "y": [3, 4]})

        with pytest.raises(ValueError):
            SunAbrahamEstimator(invalid_data, "missing", "time", "unit")

    def test_estimate_basic(self, sample_panel_data):
        """Test basic estimation functionality."""
        estimator = SunAbrahamEstimator(
            sample_panel_data, cohort_col="cohort", time_col="time", unit_col="unit"
        )

        results = estimator.estimate("outcome")

        assert "cohort_effects" in results, "Results should include cohort effects"
        assert "aggregate_effects" in results, "Results should include aggregate effects"

        if results["aggregate_effects"]:
            assert "att" in results["aggregate_effects"], "ATT should be present"
            assert "se" in results["aggregate_effects"], "SE should be present"
            assert isinstance(results["aggregate_effects"]["att"], (int, float)), "ATT should be numeric"

    def test_event_study(self, sample_panel_data):
        """Test event study analysis."""
        estimator = SunAbrahamEstimator(
            sample_panel_data, cohort_col="cohort", time_col="time", unit_col="unit"
        )

        event_results = estimator.event_study("outcome", horizon=3)

        assert isinstance(event_results, pd.DataFrame), "Event results should be DataFrame"
        assert "relative_time" in event_results.columns, "Relative time column missing"
        assert "coefficient" in event_results.columns, "Coefficient column missing"
        assert "std_error" in event_results.columns, "Std error column missing"

    def test_parallel_trends_test(self, sample_panel_data):
        """Test parallel trends testing."""
        estimator = SunAbrahamEstimator(
            sample_panel_data, cohort_col="cohort", time_col="time", unit_col="unit"
        )

        pt_results = estimator.test_parallel_trends("outcome")

        assert "test_statistic" in pt_results, "Test statistic should be present"
        assert "p_value" in pt_results, "P-value should be present"
        assert "reject_parallel" in pt_results, "Reject parallel should be present"
        assert isinstance(pt_results["reject_parallel"], bool), "Reject parallel should be boolean"

    def test_with_covariates(self, sample_panel_data):
        """Test estimation with covariates."""
        estimator = SunAbrahamEstimator(
            sample_panel_data, cohort_col="cohort", time_col="time", unit_col="unit"
        )

        results = estimator.estimate("outcome", covariates=["covariate1", "covariate2"])

        assert results["covariates"] == ["covariate1", "covariate2"], "Covariates should match"


class TestBorusyakImputation:
    """Test Borusyak et al. (2021) imputation estimator."""

    def test_initialization(self, sample_panel_data):
        """Test estimator initialization."""
        estimator = BorusyakImputation(
            sample_panel_data,
            unit_col="unit",
            time_col="time",
            treatment_col="treatment",
        )

        assert estimator.data is not None
        assert len(estimator.units) == N_UNITS
        assert len(estimator.time_periods) == N_PERIODS

    def test_invalid_initialization(self):
        """Test error handling for invalid initialization."""
        invalid_data = pd.DataFrame({"x": [1, 2]})

        with pytest.raises(ValueError):
            BorusyakImputation(invalid_data, "missing", "time")

    def test_imputation_methods(self, sample_panel_data):
        """Test counterfactual imputation methods."""
        estimator = BorusyakImputation(
            sample_panel_data,
            unit_col="unit",
            time_col="time",
            treatment_col="treatment",
        )

        # Test fixed effects imputation
        imputed_df = estimator.impute_counterfactuals("outcome", method="fe")

        assert "Y0_imputed" in imputed_df.columns
        assert "is_treated" in imputed_df.columns
        assert "treatment_effect" in imputed_df.columns

        # Test linear imputation
        imputed_df_linear = estimator.impute_counterfactuals("outcome", method="linear")

        assert "Y0_imputed" in imputed_df_linear.columns

    def test_effect_estimation(self, sample_panel_data):
        """Test treatment effect estimation."""
        estimator = BorusyakImputation(
            sample_panel_data,
            unit_col="unit",
            time_col="time",
            treatment_col="treatment",
        )

        estimator.impute_counterfactuals("outcome")
        results = estimator.estimate_effects(bootstrap=False)

        assert "att" in results
        assert "se" in results
        assert "n_treated" in results
        assert isinstance(results["att"], (int, float))

    def test_aggregation_levels(self, sample_panel_data):
        """Test different aggregation levels."""
        estimator = BorusyakImputation(
            sample_panel_data,
            unit_col="unit",
            time_col="time",
            treatment_col="treatment",
        )

        estimator.impute_counterfactuals("outcome")

        # Test unit-level aggregation
        unit_results = estimator.estimate_effects(level="unit")
        assert isinstance(unit_results, dict)

        # Test time-level aggregation
        time_results = estimator.estimate_effects(level="time")
        assert isinstance(time_results, dict)

    def test_pre_trends_testing(self, sample_panel_data):
        """Test pre-treatment trends analysis."""
        estimator = BorusyakImputation(
            sample_panel_data,
            unit_col="unit",
            time_col="time",
            treatment_col="treatment",
        )

        pre_trend_results = estimator.test_pre_trends("outcome")

        assert "test_statistic" in pre_trend_results
        assert "p_value" in pre_trend_results
        assert "reject_parallel" in pre_trend_results

    def test_invalid_method(self, sample_panel_data):
        """Test error handling for invalid methods."""
        estimator = BorusyakImputation(
            sample_panel_data,
            unit_col="unit",
            time_col="time",
            treatment_col="treatment",
        )

        with pytest.raises(ValueError):
            estimator.impute_counterfactuals("outcome", method="invalid")


class TestDCDHEstimator:
    """Test de Chaisemartin & D'Haultfœuille estimator."""

    def test_initialization(self, sample_panel_data):
        """Test estimator initialization."""
        estimator = DCDHEstimator(sample_panel_data, unit_col="unit", time_col="time")

        assert estimator.data is not None
        assert len(estimator.units) == N_UNITS
        assert len(estimator.time_periods) == N_PERIODS

    def test_fuzzy_did_continuous(self, sample_panel_data):
        """Test fuzzy DiD with continuous treatment."""
        # Add treatment intensity
        sample_panel_data["treatment_intensity"] = sample_panel_data[
            "treatment"
        ] * np.random.uniform(0.5, 1.5, len(sample_panel_data))

        estimator = DCDHEstimator(sample_panel_data, unit_col="unit", time_col="time")

        results = estimator.fuzzy_did("outcome", "treatment_intensity", continuous=True)

        assert "att" in results
        assert "se" in results
        assert "continuous" in results
        assert results["continuous"] is True

    def test_fuzzy_did_binary(self, sample_panel_data):
        """Test fuzzy DiD with binary treatment."""
        estimator = DCDHEstimator(sample_panel_data, unit_col="unit", time_col="time")

        results = estimator.fuzzy_did("outcome", "treatment", continuous=False)

        assert "att" in results
        assert "se" in results
        assert "continuous" in results
        assert results["continuous"] is False

    def test_placebo_tests(self, simple_did_data):
        """Test placebo testing framework."""
        estimator = DCDHEstimator(simple_did_data, unit_col="unit", time_col="time")

        placebo_results = estimator.placebo_tests(
            "outcome", "treatment", n_placebos=10, seed=RANDOM_SEED
        )

        assert isinstance(placebo_results, pd.DataFrame)
        assert len(placebo_results) == 10
        assert "placebo_att" in placebo_results.columns

    def test_heterogeneity_analysis(self, sample_panel_data):
        """Test treatment effect heterogeneity analysis."""
        estimator = DCDHEstimator(sample_panel_data, unit_col="unit", time_col="time")

        het_results = estimator.heterogeneity_analysis(
            "outcome", "treatment", "covariate1", n_groups=3
        )

        assert isinstance(het_results, dict)
        assert "group_0" in het_results
        assert "group_1" in het_results
        assert "group_2" in het_results

    def test_compare_specifications(self, sample_panel_data):
        """Test comparison of binary vs continuous specifications."""
        sample_panel_data["treatment_continuous"] = sample_panel_data[
            "treatment"
        ] * np.random.uniform(0, 2, len(sample_panel_data))

        estimator = DCDHEstimator(sample_panel_data, unit_col="unit", time_col="time")

        comparison = estimator.compare_binary_continuous(
            "outcome", "treatment_continuous"
        )

        assert "continuous" in comparison
        assert "binary" in comparison
        assert (
            "difference" in comparison
            or len([k for k in comparison.values() if "att" in k]) >= 2
        )


class TestDoublyRobustDiD:
    """Test Doubly Robust DiD estimator."""

    def test_initialization(self, sample_panel_data):
        """Test estimator initialization."""
        estimator = DoublyRobustDiD(sample_panel_data, unit_col="unit", time_col="time")

        assert estimator.data is not None
        assert len(estimator.units) == N_UNITS

    def test_propensity_score_estimation(self, sample_panel_data):
        """Test propensity score estimation."""
        estimator = DoublyRobustDiD(sample_panel_data, unit_col="unit", time_col="time")

        ps_scores = estimator.estimate_propensity(
            "treatment", ["covariate1", "covariate2"], method="logistic"
        )

        assert isinstance(ps_scores, pd.Series)
        assert len(ps_scores) == len(sample_panel_data)
        assert ps_scores.min() >= 0.01
        assert ps_scores.max() <= 0.99

    def test_outcome_regression(self, sample_panel_data):
        """Test outcome regression models."""
        estimator = DoublyRobustDiD(sample_panel_data, unit_col="unit", time_col="time")

        or_results = estimator.outcome_regression(
            "outcome", ["covariate1", "covariate2"], treatment="treatment"
        )

        assert "predictions" in or_results
        assert "residuals" in or_results
        assert "r_squared" in or_results
        assert "rmse" in or_results

    def test_doubly_robust_att(self, sample_panel_data):
        """Test doubly robust ATT estimation."""
        estimator = DoublyRobustDiD(sample_panel_data, unit_col="unit", time_col="time")

        dr_results = estimator.doubly_robust_att(
            "outcome", "treatment", ["covariate1", "covariate2"], bootstrap=False
        )

        assert "dr_att" in dr_results
        assert "se" in dr_results
        assert "ipw_att" in dr_results
        assert "regression_att" in dr_results
        assert "n_treated" in dr_results
        assert "n_control" in dr_results

    def test_sensitivity_analysis(self, sample_panel_data):
        """Test sensitivity to model specification."""
        estimator = DoublyRobustDiD(sample_panel_data, unit_col="unit", time_col="time")

        sensitivity_results = estimator.sensitivity_analysis(
            "outcome", "treatment", ["covariate1", "covariate2"]
        )

        assert isinstance(sensitivity_results, pd.DataFrame)
        assert "ps_method" in sensitivity_results.columns
        assert "or_method" in sensitivity_results.columns
        assert "dr_att" in sensitivity_results.columns

    def test_covariate_balance(self, sample_panel_data):
        """Test covariate balance checking."""
        estimator = DoublyRobustDiD(sample_panel_data, unit_col="unit", time_col="time")

        # Estimate propensity scores first
        estimator.estimate_propensity("treatment", ["covariate1", "covariate2"])

        balance_results = estimator.covariate_balance_check(
            "treatment", ["covariate1", "covariate2"], weighted=True
        )

        assert isinstance(balance_results, pd.DataFrame)
        assert "covariate" in balance_results.columns
        assert "std_diff" in balance_results.columns
        assert "balanced" in balance_results.columns


class TestMethodComparison:
    """Test method comparison framework."""

    def test_initialization(self, sample_panel_data):
        """Test comparison framework initialization."""
        comparison = MethodComparison(
            sample_panel_data, unit_col="unit", time_col="time", cohort_col="cohort"
        )

        assert comparison.data is not None
        assert len(comparison.methods) > 0
        assert "sun_abraham" in comparison.methods
        assert "borusyak" in comparison.methods

    def test_run_individual_methods(self, sample_panel_data):
        """Test running individual methods."""
        comparison = MethodComparison(
            sample_panel_data, unit_col="unit", time_col="time", cohort_col="cohort"
        )

        # Test Sun & Abraham
        sa_result = comparison.run_sun_abraham("outcome")
        assert sa_result.method_name == "Sun & Abraham (2021)"
        assert hasattr(sa_result, "att")
        assert hasattr(sa_result, "se")

        # Test Borusyak
        b_result = comparison.run_borusyak("outcome", "treatment")
        assert b_result.method_name == "Borusyak et al. (2021)"

        # Test DCDH
        dcdh_result = comparison.run_dcdh("outcome", "treatment")
        assert dcdh_result.method_name == "de Chaisemartin & D'Haultfœuille"

        # Test Doubly Robust
        dr_result = comparison.run_doubly_robust(
            "outcome", "treatment", ["covariate1", "covariate2"]
        )
        assert dr_result.method_name == "Doubly Robust DiD"

    def test_run_all_methods(self, sample_panel_data):
        """Test running all methods together."""
        comparison = MethodComparison(
            sample_panel_data, unit_col="unit", time_col="time", cohort_col="cohort"
        )

        results_df = comparison.run_all_methods(
            "outcome",
            treatment="treatment",
            covariates=["covariate1", "covariate2"],
            methods_to_run=["sun_abraham", "borusyak", "dcdh"],
        )

        assert isinstance(results_df, pd.DataFrame)
        assert "method" in results_df.columns
        assert "att" in results_df.columns
        assert "se" in results_df.columns
        assert len(results_df) <= 3

    def test_diagnostic_suite(self, sample_panel_data):
        """Test comprehensive diagnostics."""
        comparison = MethodComparison(
            sample_panel_data, unit_col="unit", time_col="time", cohort_col="cohort"
        )

        diagnostics = comparison.diagnostic_suite(
            "outcome", treatment="treatment", covariates=["covariate1", "covariate2"]
        )

        assert isinstance(diagnostics, dict)
        assert "sample_size" in diagnostics
        assert "treatment_timing" in diagnostics

    def test_generate_report(self, sample_panel_data):
        """Test report generation."""
        comparison = MethodComparison(
            sample_panel_data, unit_col="unit", time_col="time", cohort_col="cohort"
        )

        # Run some methods first
        comparison.run_all_methods("outcome", methods_to_run=["sun_abraham"])
        comparison.diagnostic_suite("outcome")

        report = comparison.generate_report()

        assert isinstance(report, str)
        assert "COMPARISON REPORT" in report
        assert len(report) > 100


class TestDiagnostics:
    """Test diagnostic functions."""

    def test_parallel_trends_test(self, sample_panel_data):
        """Test parallel trends testing function."""
        result = test_parallel_trends(
            sample_panel_data, "outcome", "unit", "time", "treatment"
        )

        assert "test_statistic" in result
        assert "p_value" in result
        assert "reject_parallel" in result

    def test_common_support_propensity(self, sample_panel_data):
        """Test common support using propensity scores."""
        result = check_common_support(
            sample_panel_data,
            "treatment",
            ["covariate1", "covariate2"],
            method="propensity",
        )

        assert result["method"] == "propensity_score"
        assert "overlap_proportion" in result
        assert "good_support" in result

    def test_common_support_covariate(self, sample_panel_data):
        """Test common support using covariate ranges."""
        result = check_common_support(
            sample_panel_data,
            "treatment",
            ["covariate1", "covariate2"],
            method="covariate",
        )

        assert result["method"] == "covariate_ranges"
        assert "covariate_support" in result
        assert "good_support" in result

    def test_no_anticipation_test(self, sample_panel_data):
        """Test no anticipation assumption."""
        result = test_no_anticipation(
            sample_panel_data, "outcome", "unit", "time", "treatment"
        )

        assert "test_statistic" in result
        assert "p_value" in result
        assert "reject_no_anticipation" in result

    def test_effective_sample_size(self, sample_panel_data):
        """Test effective sample size computation."""
        result = compute_effective_sample_size(sample_panel_data, "treatment")

        assert "n_treated" in result
        assert "n_control" in result
        assert "ess_treated" in result
        assert "ess_control" in result
        assert "total_ess" in result


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_data(self):
        """Test behavior with empty data."""
        empty_data = pd.DataFrame()

        with pytest.raises((ValueError, KeyError)):
            SunAbrahamEstimator(empty_data, "cohort", "time", "unit")

    def test_single_period_data(self):
        """Test behavior with single time period."""
        single_period = pd.DataFrame(
            {
                "unit": [1, 2, 3],
                "time": [2010, 2010, 2010],
                "cohort": [2010, 2010, 9999],
                "outcome": [10, 12, 11],
                "treatment": [1, 1, 0],
            }
        )

        estimator = SunAbrahamEstimator(single_period, "cohort", "time", "unit")

        # Should handle gracefully but may return NaN results
        results = estimator.estimate("outcome")
        # Just ensure it doesn't crash
        assert "aggregate_effects" in results

    def test_no_treated_units(self):
        """Test behavior with no treated units."""
        no_treated = pd.DataFrame(
            {
                "unit": [1, 2, 3, 1, 2, 3],
                "time": [2010, 2010, 2010, 2011, 2011, 2011],
                "cohort": [9999, 9999, 9999, 9999, 9999, 9999],
                "outcome": [10, 12, 11, 11, 13, 12],
                "treatment": [0, 0, 0, 0, 0, 0],
            }
        )

        estimator = SunAbrahamEstimator(no_treated, "cohort", "time", "unit")
        results = estimator.estimate("outcome")

        # Should return None or NaN for aggregate effects
        assert results["aggregate_effects"] is None or np.isnan(
            results["aggregate_effects"]["att"]
        )

    def test_missing_outcome_values(self, sample_panel_data):
        """Test behavior with missing outcome values."""
        data_with_missing = sample_panel_data.copy()
        # Introduce some missing values
        data_with_missing.loc[data_with_missing.index[:10], "outcome"] = np.nan

        estimator = SunAbrahamEstimator(data_with_missing, "cohort", "time", "unit")

        # Should handle missing data gracefully
        results = estimator.estimate("outcome")
        assert "aggregate_effects" in results

    def test_performance_timeout(self, sample_panel_data):
        """Test that methods complete within reasonable time."""

        comparison = MethodComparison(
            sample_panel_data, unit_col="unit", time_col="time", cohort_col="cohort"
        )

        start_time = time.time()
        results = comparison.run_all_methods("outcome", methods_to_run=["sun_abraham"])
        end_time = time.time()

        # Should complete within reasonable time for this sample size
        assert (end_time - start_time) < MAX_TEST_TIME_SECONDS
        assert len(results) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
