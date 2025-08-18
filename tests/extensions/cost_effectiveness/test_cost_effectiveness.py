"""Comprehensive unit tests for cost-effectiveness framework."""

import pandas as pd
import pytest

from mpower_mortality_causal_analysis.extensions.cost_effectiveness import (
    BudgetOptimizer,
    CEPipeline,
    CEReporting,
    CostEstimator,
    HealthOutcomeModel,
    ICERAnalysis,
)


class TestHealthOutcomes:
    """Test health outcome calculations."""

    def test_qaly_calculation(self):
        """Validate QALY computation."""
        # Create test data
        mortality_data = pd.DataFrame(
            {"country": ["TestCountry"], "mortality_rate": [100]}
        )

        model = HealthOutcomeModel(mortality_data)

        # Test QALY calculation
        results = model.calculate_qalys(mortality_reduction=10.0)

        assert "total_qalys" in results
        assert "discounted_qalys" in results
        assert results["total_qalys"] > 0
        assert results["discounted_qalys"] > 0
        assert results["discounted_qalys"] <= results["total_qalys"]

    def test_daly_calculation(self):
        """Validate DALY computation."""
        mortality_data = pd.DataFrame({"country": ["Test"], "rate": [100]})
        model = HealthOutcomeModel(mortality_data)

        results = model.calculate_dalys(
            mortality_reduction=10.0, disease="cardiovascular"
        )

        assert "yll_averted" in results
        assert "yld_averted" in results
        assert "total_dalys_averted" in results
        assert results["total_dalys_averted"] > 0

    def test_markov_convergence(self):
        """Test Markov model steady state."""
        mortality_data = pd.DataFrame({"country": ["Test"], "rate": [100]})
        model = HealthOutcomeModel(mortality_data)

        # Run Markov model
        results = model.markov_model(
            transition_probs={},  # Will use defaults
            initial_state={},  # Will use defaults
            time_steps=100,
        )

        assert len(results) == 100
        assert "time" in results.columns
        assert "healthy_baseline" in results.columns
        assert "healthy_intervention" in results.columns

        # Check convergence (steady state)
        final_state = results.iloc[-1]
        penultimate_state = results.iloc[-2]

        # States should be relatively stable at end
        for col in ["healthy_baseline", "at_risk_baseline", "diseased_baseline"]:
            if col in results.columns:
                diff = abs(final_state[col] - penultimate_state[col])
                assert diff < 0.01  # Less than 1% change


class TestCostModels:
    """Test cost estimation models."""

    def test_implementation_costs(self):
        """Test policy implementation cost calculation."""
        estimator = CostEstimator()

        costs = estimator.implementation_costs(policy="M", country="Example", years=5)

        assert "total_cost" in costs
        assert "discounted_cost" in costs
        assert "annual_costs" in costs
        assert costs["total_cost"] > 0
        assert len(costs["annual_costs"]) == 5

    def test_healthcare_savings(self):
        """Test healthcare savings calculation."""
        estimator = CostEstimator()

        cases_prevented = {"lung_cancer": 10, "cardiovascular": 20}

        savings = estimator.healthcare_savings(
            cases_prevented=cases_prevented, country="Example"
        )

        assert "total_savings" in savings
        assert "discounted_savings" in savings
        assert savings["total_savings"] > 0

    def test_net_costs(self):
        """Test net cost calculation with all offsets."""
        estimator = CostEstimator()

        net_costs = estimator.calculate_net_costs(
            policy="P",
            country="Example",
            mortality_reduction=10.0,
            cases_prevented={"cardiovascular": 5},
        )

        assert "implementation_costs" in net_costs
        assert "healthcare_savings" in net_costs
        assert "productivity_gains" in net_costs
        assert "net_cost" in net_costs
        assert "roi" in net_costs


class TestICER:
    """Test ICER analysis."""

    def test_icer_calculation(self):
        """Test basic ICER calculation."""
        costs = {"intervention": 10000, "control": 5000}
        effects = {"intervention": 10, "control": 5}

        analyzer = ICERAnalysis(costs=costs, effects=effects)

        result = analyzer.calculate_icer("intervention", "control")

        assert result["incremental_cost"] == 5000
        assert result["incremental_effect"] == 5
        assert result["icer"] == 1000

    def test_dominance_detection(self):
        """Identify dominated strategies."""
        costs = {
            "A": 1000,
            "B": 2000,  # Dominated (more cost, less effect)
            "C": 500,  # Dominant (less cost, more effect)
        }
        effects = {
            "A": 10,
            "B": 5,
            "C": 15,
        }

        analyzer = ICERAnalysis(costs=costs, effects=effects)

        # Test dominance
        result_b = analyzer.calculate_icer("B", "A")
        assert result_b["dominance_status"] == "dominated"

        result_c = analyzer.calculate_icer("C", "A")
        assert result_c["dominance_status"] == "dominant"

    def test_psa_convergence(self):
        """PSA stability with iterations."""
        costs = {"intervention": 10000, "control": 5000}
        effects = {"intervention": 10, "control": 5}

        analyzer = ICERAnalysis(costs=costs, effects=effects)

        psa_results = analyzer.probabilistic_sensitivity(n_simulations=100)

        assert len(psa_results) > 0
        assert "icer" in psa_results.columns
        assert psa_results["icer"].notna().sum() > 0  # Some valid ICERs


class TestBudgetOptimizer:
    """Test budget optimization."""

    def test_linear_optimization(self):
        """Test linear programming optimization."""
        optimizer = BudgetOptimizer(
            budget=1000000,
            policies=["M", "P", "O"],
            costs={"M": 100000, "P": 200000, "O": 300000},
            effects={"M": 100, "P": 150, "O": 200},
        )

        result = optimizer.optimize_allocation(method="linear")

        assert result["success"]
        assert "allocation" in result
        assert sum(result["allocation"].values()) <= len(optimizer.policies)
        assert result["total_cost"] <= optimizer.budget

    def test_portfolio_optimization(self):
        """Test portfolio optimization."""
        optimizer = BudgetOptimizer(
            budget=1000000,
            policies=["M", "P"],
            costs={"M": 100000, "P": 200000},
            effects={"M": 100, "P": 150},
        )

        result = optimizer.portfolio_optimization()

        if result["success"]:
            assert "allocation" in result
            assert "expected_return" in result
            assert "risk" in result
            assert (
                abs(sum(result["allocation"].values()) - 1.0) < 0.01
            )  # Weights sum to 1

    def test_incremental_allocation(self):
        """Test incremental allocation strategy."""
        optimizer = BudgetOptimizer(
            budget=500000,
            policies=["M", "P"],
            costs={"M": 100000, "P": 200000},
            effects={"M": 100, "P": 150},
        )

        steps = optimizer.incremental_allocation()

        assert len(steps) > 0
        assert steps[-1]["total_cost"] <= optimizer.budget
        # Check that allocation increases over steps
        for i in range(1, len(steps)):
            assert steps[i]["budget"] >= steps[i - 1]["budget"]


class TestCEPipeline:
    """Test main pipeline orchestration."""

    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = CEPipeline()

        assert pipeline.mortality_data is None
        assert pipeline.health_model is None

        pipeline.initialize_components()

        assert pipeline.health_model is not None
        assert pipeline.cost_estimator is not None
        assert pipeline.icer_analyzer is not None

    def test_run_analysis(self):
        """Test complete analysis run."""
        # Create test data
        mortality_data = pd.DataFrame(
            {
                "country": ["TestCountry"],
                "lung_cancer_reduction": [10],
                "cardiovascular_reduction": [8],
            }
        )

        pipeline = CEPipeline(mortality_data=mortality_data)

        results = pipeline.run_analysis(
            country="TestCountry",
            policies=["M", "P"],
            budget=1000000,
            sensitivity=False,  # Skip for speed
        )

        assert "health_outcomes" in results
        assert "costs" in results
        assert "icers" in results
        assert "optimization" in results

    def test_export_results(self):
        """Test results export functionality."""
        import json
        import tempfile

        from pathlib import Path

        pipeline = CEPipeline()
        pipeline.results = {"test": "data", "health_outcomes": {"M": {"qalys": 100}}}

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline.export_results(tmpdir, formats=["json"])

            # Check that file was created
            json_file = Path(tmpdir) / "ce_results.json"
            assert json_file.exists()

            # Load and verify content
            with open(json_file) as f:
                data = json.load(f)
            assert data["test"] == "data"


class TestCEReporting:
    """Test reporting functionality."""

    def test_report_generation(self):
        """Test report generation."""
        import tempfile

        from pathlib import Path

        reporter = CEReporting()

        test_results = {
            "country": "TestCountry",
            "policies": ["M", "P"],
            "health_outcomes": {
                "M": {"qalys": 100, "dalys": 80},
                "combined": {"qalys": 200, "dalys": 160},
            },
            "costs": {"M": {"net_cost": -50000}, "combined": {"net_cost": -100000}},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.md"
            reporter.generate_report(test_results, output_path, format="summary")

            assert output_path.exists()

            # Check content
            with open(output_path) as f:
                content = f.read()
            assert "TestCountry" in content
            assert "M, P" in content

    def test_visualization_generation(self):
        """Test that visualization methods don't crash."""
        reporter = CEReporting()

        test_results = {
            "icers": {
                "icer_table": [
                    {
                        "intervention": "M",
                        "comparator": "control",
                        "incremental_cost": 1000,
                        "incremental_effect": 10,
                        "icer": 100,
                    }
                ],
                "ceac_data": [
                    {
                        "threshold": 1000,
                        "strategy": "M",
                        "probability_cost_effective": 0.8,
                    }
                ],
            }
        }

        import tempfile

        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            # This should not crash even without matplotlib
            reporter._create_visualizations(test_results, output_dir)


# Integration test
class TestIntegration:
    """Integration tests for the complete framework."""

    def test_full_workflow(self):
        """Test complete cost-effectiveness analysis workflow."""
        # Step 1: Create mortality data
        mortality_data = pd.DataFrame(
            {
                "country": ["Country1", "Country2"],
                "lung_cancer_reduction": [10, 12],
                "cardiovascular_reduction": [8, 9],
                "copd_reduction": [6, 7],
                "ihd_reduction": [7, 8],
            }
        )

        # Step 2: Create cost data
        cost_data = pd.DataFrame(
            {
                "country": ["Country1", "Country2"],
                "gdp_per_capita": [15000, 20000],
                "income_group": ["middle_income", "middle_income"],
                "population": [10000000, 15000000],
                "health_expenditure_per_capita": [600, 800],
                "avg_wage": [18000, 22000],
            }
        )

        # Step 3: Initialize pipeline
        pipeline = CEPipeline(
            mortality_data=mortality_data, cost_data=cost_data, wtp_threshold=30000
        )

        # Step 4: Run analysis
        results = pipeline.run_analysis(
            country="Country1",
            policies=["M", "P", "O"],
            budget=2000000,
            sensitivity=True,
        )

        # Step 5: Verify results structure
        assert "health_outcomes" in results
        assert "costs" in results
        assert "icers" in results
        assert "optimization" in results
        assert "sensitivity" in results

        # Check health outcomes
        assert "M" in results["health_outcomes"]
        assert "combined" in results["health_outcomes"]
        assert results["health_outcomes"]["combined"]["qalys"] > 0

        # Check costs
        assert "M" in results["costs"]
        assert "net_cost" in results["costs"]["M"]

        # Check ICERs
        assert "icer_table" in results["icers"]
        assert "efficient_frontier" in results["icers"]

        # Check optimization
        assert "linear_optimization" in results["optimization"]
        if results["optimization"]["linear_optimization"]["success"]:
            assert "allocation" in results["optimization"]["linear_optimization"]

        # Step 6: Test export
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline.export_results(tmpdir, formats=["json", "excel"])

            from pathlib import Path

            assert (Path(tmpdir) / "ce_results.json").exists()
            assert (Path(tmpdir) / "ce_results.xlsx").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
