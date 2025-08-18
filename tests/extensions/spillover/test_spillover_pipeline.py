"""Tests for the main spillover analysis pipeline."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from mpower_mortality_causal_analysis.extensions.spillover.spillover_pipeline import (
    SpilloverPipeline,
)


class TestSpilloverPipeline:
    """Test the main spillover analysis pipeline."""

    @pytest.fixture
    def sample_data(self):
        """Create sample panel data for testing."""
        np.random.seed(42)

        countries = ["USA", "Canada", "Mexico", "Brazil", "Argentina", "Chile", "Peru"]
        years = [2010, 2012, 2014, 2016, 2018]

        # Create panel data
        data_rows = []
        for country in countries:
            for year in years:
                # Create treatment variable (some countries adopt in later years)
                treatment = 1 if country in ["USA", "Canada"] and year >= 2014 else 0

                # Create outcomes with treatment effect and spatial correlation
                base_mortality = 80 + np.random.normal(0, 10)
                treatment_effect = -15 if treatment else 0

                data_rows.append(
                    {
                        "country": country,
                        "year": year,
                        "lung_cancer_mortality_rate": base_mortality
                        + treatment_effect
                        + np.random.normal(0, 5),
                        "cardiovascular_mortality_rate": base_mortality * 1.2
                        + treatment_effect
                        + np.random.normal(0, 7),
                        "mpower_high": treatment,
                        "gdp_per_capita_log": 9 + np.random.normal(0, 1),
                        "urban_population_pct": 60 + np.random.normal(0, 15),
                        "total_population_log": 16 + np.random.normal(0, 2),
                    }
                )

        return pd.DataFrame(data_rows)

    @pytest.fixture
    def pipeline(self, sample_data):
        """Initialize spillover pipeline with sample data."""
        return SpilloverPipeline(
            data=sample_data,
            outcomes=["lung_cancer_mortality_rate", "cardiovascular_mortality_rate"],
            treatment_col="mpower_high",
            covariates=["gdp_per_capita_log", "urban_population_pct"],
        )

    def test_initialization(self, sample_data):
        """Test proper initialization of pipeline."""
        pipeline = SpilloverPipeline(sample_data)

        assert len(pipeline.countries) == 7
        assert len(pipeline.years) == 5
        assert pipeline.n_countries == 7
        assert pipeline.n_years == 5
        assert pipeline.unit_col == "country"
        assert pipeline.time_col == "year"
        assert len(pipeline.outcomes) >= 1  # At least one outcome should be found

    def test_initialization_with_custom_parameters(self, sample_data):
        """Test initialization with custom parameters."""
        custom_outcomes = ["lung_cancer_mortality_rate"]
        custom_covariates = ["gdp_per_capita_log"]

        pipeline = SpilloverPipeline(
            data=sample_data,
            outcomes=custom_outcomes,
            covariates=custom_covariates,
            treatment_col="mpower_high",
        )

        assert pipeline.outcomes == custom_outcomes
        assert pipeline.covariates == custom_covariates
        assert pipeline.treatment_col == "mpower_high"

    def test_initialization_missing_columns(self, sample_data):
        """Test initialization gracefully handles missing columns."""
        missing_outcomes = ["nonexistent_outcome", "lung_cancer_mortality_rate"]
        missing_covariates = ["nonexistent_covariate", "gdp_per_capita_log"]

        pipeline = SpilloverPipeline(
            data=sample_data, outcomes=missing_outcomes, covariates=missing_covariates
        )

        # Should filter out missing columns
        assert "nonexistent_outcome" not in pipeline.outcomes
        assert "lung_cancer_mortality_rate" in pipeline.outcomes
        assert "nonexistent_covariate" not in pipeline.covariates
        assert "gdp_per_capita_log" in pipeline.covariates

    def test_weight_matrices_creation(self, pipeline):
        """Test creation of spatial weight matrices."""
        pipeline._create_weight_matrices()

        assert "weight_matrices" in pipeline.results

        # Check that required matrices are created
        expected_matrices = ["contiguity", "distance", "knn", "hybrid"]
        for matrix_name in expected_matrices:
            assert matrix_name in pipeline.results["weight_matrices"]

            matrix_info = pipeline.results["weight_matrices"][matrix_name]
            assert "matrix" in matrix_info
            assert "statistics" in matrix_info

            # Check matrix properties
            W = matrix_info["matrix"]
            assert W.shape == (pipeline.n_countries, pipeline.n_countries)
            assert np.all(W >= 0)  # Non-negative weights

            # Row-standardized matrices are not necessarily symmetric
            stats = matrix_info["statistics"]
            if not stats.get("is_row_standardized", False):
                assert np.allclose(W, W.T)  # Symmetry only for non-row-standardized

    def test_spatial_models_execution(self, pipeline):
        """Test execution of spatial econometric models."""
        # Need weight matrices first
        pipeline._create_weight_matrices()

        # Run spatial models
        pipeline._run_spatial_models()

        assert "spatial_models" in pipeline.results

        # Check that models are run for each outcome
        for outcome in pipeline.outcomes:
            assert outcome in pipeline.results["spatial_models"]

            outcome_results = pipeline.results["spatial_models"][outcome]

            # Check that expected models are attempted
            expected_models = ["sar", "sem", "sdm", "lm_tests"]
            for model in expected_models:
                assert model in outcome_results

    def test_diffusion_analysis(self, pipeline):
        """Test policy diffusion analysis."""
        # Need weight matrices first
        pipeline._create_weight_matrices()

        # Run diffusion analysis
        pipeline._analyze_diffusion()

        assert "diffusion_analysis" in pipeline.results

        diffusion_results = pipeline.results["diffusion_analysis"]

        # Check expected components
        expected_components = ["threshold_model", "cascade_model", "influencers"]
        for component in expected_components:
            assert component in diffusion_results

    def test_border_analysis(self, pipeline):
        """Test border discontinuity analysis."""
        pipeline._analyze_borders()

        assert "border_analysis" in pipeline.results

        border_results = pipeline.results["border_analysis"]

        # Check that analysis is run for each outcome
        for outcome in pipeline.outcomes:
            assert outcome in border_results

        # Check for additional components
        assert "heterogeneity" in border_results or "all_borders" in border_results

    def test_model_comparison(self, pipeline):
        """Test model comparison functionality."""
        # Set up some mock results
        pipeline.results["spatial_models"] = {}
        for outcome in pipeline.outcomes:
            pipeline.results["spatial_models"][outcome] = {
                "sar": {"aic": 100, "bic": 110, "log_likelihood": -45, "n_obs": 35},
                "sem": {"aic": 105, "bic": 115, "log_likelihood": -47, "n_obs": 35},
                "sdm": {"aic": 98, "bic": 112, "log_likelihood": -44, "n_obs": 35},
            }

        pipeline._compare_models()

        assert "model_comparison" in pipeline.results

        # Check that best models are identified
        for outcome in pipeline.outcomes:
            if outcome in pipeline.results["model_comparison"]:
                comparison = pipeline.results["model_comparison"][outcome]
                assert "best_model" in comparison
                assert "best_aic" in comparison
                assert "model_statistics" in comparison

    def test_summary_generation(self, pipeline):
        """Test summary generation."""
        # Set up some mock results for summary
        pipeline.results["spatial_models"] = {
            pipeline.outcomes[0]: {
                "sar": {"rho": 0.3, "rho_pvalue": 0.01},
                "lm_tests": {
                    "LM_lag": {"p_value": 0.02},
                    "LM_error": {"p_value": 0.15},
                },
            }
        }

        pipeline._generate_summary()

        assert "summary" in pipeline.results

        summary = pipeline.results["summary"]

        # Check required sections
        assert "analysis_overview" in summary
        assert "key_findings" in summary

        # Check analysis overview
        overview = summary["analysis_overview"]
        assert overview["n_countries"] == pipeline.n_countries
        assert overview["n_years"] == pipeline.n_years
        assert overview["n_outcomes"] == len(pipeline.outcomes)

        # Check key findings
        assert isinstance(summary["key_findings"], list)

    def test_adoption_data_preparation(self, pipeline):
        """Test preparation of adoption data for diffusion analysis."""
        adoption_data = pipeline._prepare_adoption_data()

        assert isinstance(adoption_data, pd.DataFrame)
        assert "country" in adoption_data.columns
        assert "year" in adoption_data.columns
        assert "adopted" in adoption_data.columns

        # Check data types
        assert adoption_data["adopted"].dtype in [np.int64, np.int32, int]
        assert adoption_data["adopted"].isin([0, 1]).all()

    def test_border_data_preparation(self, pipeline):
        """Test preparation of border data."""
        border_data = pipeline._prepare_border_data()

        assert isinstance(border_data, pd.DataFrame)
        assert "distance_to_border" in border_data.columns
        assert pipeline.treatment_col in border_data.columns

        # Check that outcomes are added
        for outcome in pipeline.outcomes:
            assert outcome in border_data.columns

    def test_full_analysis_execution(self, pipeline):
        """Test execution of full analysis pipeline."""
        import matplotlib

        matplotlib.use("Agg")  # Use non-interactive backend for testing

        with tempfile.TemporaryDirectory() as temp_dir:
            results = pipeline.run_full_analysis(save_results=True, output_dir=temp_dir)

            # Check that all main components are present
            expected_components = [
                "weight_matrices",
                "spatial_models",
                "diffusion_analysis",
                "border_analysis",
                "model_comparison",
                "summary",
            ]

            for component in expected_components:
                assert component in results

            # Check that files are saved
            assert os.path.exists(os.path.join(temp_dir, "spillover_results.json"))
            assert os.path.exists(os.path.join(temp_dir, "weight_matrices"))
            # Visualization may fail due to display issues in CI, so don't require it
            # assert os.path.exists(os.path.join(temp_dir, "visualizations"))

    def test_results_json_conversion(self, pipeline):
        """Test conversion of results for JSON serialization."""
        # Create sample results with numpy arrays
        test_results = {
            "array": np.array([1, 2, 3]),
            "scalar": np.float64(1.5),
            "bool": np.bool_(True),
            "nested": {
                "inner_array": np.array([[1, 2], [3, 4]]),
                "normal_list": [1, 2, 3],
            },
        }

        json_results = pipeline._convert_for_json(test_results)

        # Check conversions
        assert isinstance(json_results["array"], list)
        assert json_results["array"] == [1, 2, 3]
        assert isinstance(json_results["scalar"], float)
        assert isinstance(json_results["bool"], bool)
        assert isinstance(json_results["nested"]["inner_array"], list)
        assert json_results["nested"]["inner_array"] == [[1, 2], [3, 4]]

    def test_dashboard_data_preparation(self, pipeline):
        """Test preparation of data for summary dashboard."""
        # Set up some mock results
        pipeline.results = {
            "summary": {"key_findings": ["Finding 1", "Finding 2"]},
            "spatial_models": {
                pipeline.outcomes[0]: {
                    "sar": {
                        "direct_effects": np.array([1.5]),
                        "indirect_effects": np.array([0.5]),
                        "total_effects": np.array([2.0]),
                    }
                }
            },
        }

        dashboard_data = pipeline._prepare_dashboard_data()

        assert "key_findings" in dashboard_data
        assert dashboard_data["key_findings"] == ["Finding 1", "Finding 2"]

        if "spillover_effects" in dashboard_data:
            effects = dashboard_data["spillover_effects"]
            assert "direct" in effects
            assert "indirect" in effects
            assert "total" in effects

    def test_error_handling_in_spatial_models(self, pipeline):
        """Test error handling in spatial model estimation."""
        # Create weight matrices
        pipeline._create_weight_matrices()

        # Create invalid data scenario (e.g., perfect collinearity)
        invalid_data = pipeline.data.copy()
        invalid_data["perfect_collinear"] = invalid_data[pipeline.covariates[0]]

        # Update pipeline covariates to include problematic variable
        pipeline.covariates.append("perfect_collinear")

        # Should handle errors gracefully
        pipeline._run_spatial_models()

        # Check that results contain error information
        assert "spatial_models" in pipeline.results

    def test_empty_data_handling(self):
        """Test handling of empty or insufficient data."""
        # Create minimal data
        minimal_data = pd.DataFrame(
            {
                "country": ["A", "B"],
                "year": [2010, 2010],
                "outcome": [1, 2],
                "treatment": [0, 1],
            }
        )

        pipeline = SpilloverPipeline(
            minimal_data, outcomes=["outcome"], treatment_col="treatment", covariates=[]
        )

        # Should initialize without error
        assert pipeline.n_countries == 2
        assert len(pipeline.outcomes) == 1

    def test_missing_treatment_column(self, sample_data):
        """Test handling when treatment column is missing."""
        data_no_treatment = sample_data.drop(columns=["mpower_high"])

        pipeline = SpilloverPipeline(
            data_no_treatment, treatment_col="nonexistent_treatment"
        )

        # Should handle gracefully in diffusion analysis
        adoption_data = pipeline._prepare_adoption_data()
        assert "adopted" in adoption_data.columns

    def test_single_outcome_analysis(self, sample_data):
        """Test pipeline with single outcome variable."""
        pipeline = SpilloverPipeline(
            sample_data, outcomes=["lung_cancer_mortality_rate"]
        )

        assert len(pipeline.outcomes) == 1

        # Should run without errors
        pipeline._create_weight_matrices()
        pipeline._run_spatial_models()

        assert "spatial_models" in pipeline.results
        assert "lung_cancer_mortality_rate" in pipeline.results["spatial_models"]
