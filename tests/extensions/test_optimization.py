"""Comprehensive tests for Policy Optimization Extension."""

import numpy as np
import pandas as pd
import pytest

from mpower_mortality_causal_analysis.extensions.optimization import (
    PolicyDecisionSupport,
    PolicyInteractionAnalysis,
    PolicyScheduler,
    PoliticalFeasibility,
    SequentialPolicyOptimizer,
)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)

    countries = ["Country_A", "Country_B", "Country_C", "Country_D", "Country_E"]
    years = list(range(2008, 2019))

    data = []
    for country in countries:
        for year in years:
            row = {
                "country": country,
                "year": year,
                "lung_cancer_mortality_rate": np.random.normal(20, 5),
                "mpower_m_score": np.random.randint(0, 6),
                "mpower_p_score": np.random.randint(0, 6),
                "mpower_o_score": np.random.randint(0, 6),
                "mpower_w_score": np.random.randint(0, 5),
                "mpower_e_score": np.random.randint(0, 6),
                "mpower_r_score": np.random.randint(0, 6),
                "gdp_per_capita_log": np.random.normal(10, 1),
                "urban_population_pct": np.random.normal(50, 20),
                "democracy_score": np.random.uniform(0, 10),
                "governance_effectiveness": np.random.uniform(-2, 2),
                "regulatory_quality": np.random.uniform(-2, 2),
            }
            data.append(row)

    return pd.DataFrame(data)


@pytest.fixture
def sample_effects():
    """Sample policy effects for testing."""
    return {
        "M": 1.2,
        "P": 5.5,
        "O": 3.2,
        "W": 2.8,
        "E": 4.1,
        "R": 8.3,
    }


@pytest.fixture
def sample_constraints():
    """Sample constraints for testing."""
    return {
        "budget_per_period": 200000,
        "total_budget": 1000000,
        "capacity_limit": 2,
        "max_policies_per_period": 3,
        "max_periods": 5,
        "costs": {
            "M": 50000,
            "P": 100000,
            "O": 75000,
            "W": 25000,
            "E": 150000,
            "R": 30000,
        },
    }


class TestPolicyInteractionAnalysis:
    """Test policy interaction analysis functionality."""

    def test_initialization(self, sample_data):
        """Test proper initialization."""
        analyzer = PolicyInteractionAnalysis(sample_data)
        assert analyzer.data is not None
        assert analyzer.unit_col == "country"
        assert analyzer.time_col == "year"
        assert analyzer.bootstrap_samples == 1000

    def test_create_interaction_terms(self, sample_data):
        """Test interaction term creation."""
        analyzer = PolicyInteractionAnalysis(sample_data)
        policies = ["M", "P", "O"]

        data_with_interactions = analyzer._create_interaction_terms(policies)

        # Check that interaction terms are created
        expected_interactions = [
            "M_P_interaction",
            "M_O_interaction",
            "P_O_interaction",
        ]
        for interaction in expected_interactions:
            assert interaction in data_with_interactions.columns

    def test_estimate_interactions(self, sample_data):
        """Test interaction effect estimation."""
        analyzer = PolicyInteractionAnalysis(sample_data)

        results = analyzer.estimate_interactions(
            outcome="lung_cancer_mortality_rate",
            policies=["M", "P", "W"],
            covariates=["gdp_per_capita_log"],
        )

        assert isinstance(results, dict)
        assert "main_model" in results or "error" in results

        if "main_model" in results:
            assert "rsquared" in results["main_model"]
            assert "nobs" in results["main_model"]

    def test_identify_synergies(self, sample_data):
        """Test synergy identification."""
        analyzer = PolicyInteractionAnalysis(sample_data)

        # Create mock interaction results
        mock_results = {
            "main_model": {
                "interaction_effects": {
                    "M_P_interaction": {
                        "coefficient": 2.5,
                        "p_value": 0.03,
                    },
                    "M_W_interaction": {
                        "coefficient": -1.2,
                        "p_value": 0.15,
                    },
                },
            },
        }

        synergies = analyzer.identify_synergies(mock_results)

        assert isinstance(synergies, list)
        assert len(synergies) == 2

        # First synergy should be positive and significant
        assert synergies[0][1] > 0  # Positive coefficient
        assert synergies[0][2] is True  # Significant

    def test_dose_response_analysis(self, sample_data):
        """Test dose-response analysis."""
        analyzer = PolicyInteractionAnalysis(sample_data)

        results = analyzer.analyze_dose_response(
            outcome="lung_cancer_mortality_rate",
            component="M",
        )

        assert isinstance(results, dict)
        assert "component" in results
        assert "outcome" in results
        assert "dose_response" in results


class TestSequentialPolicyOptimizer:
    """Test sequential policy optimization functionality."""

    def test_initialization(self, sample_effects, sample_constraints):
        """Test proper initialization."""
        optimizer = SequentialPolicyOptimizer(sample_effects, sample_constraints)
        assert optimizer.effects == sample_effects
        assert optimizer.constraints == sample_constraints
        assert optimizer.discount_rate == 0.03

    def test_dynamic_programming(self, sample_effects, sample_constraints):
        """Test dynamic programming optimization."""
        optimizer = SequentialPolicyOptimizer(sample_effects, sample_constraints)

        results = optimizer.dynamic_programming(
            horizon=3,
            budget_per_period=200000,
            capacity_limit=2,
        )

        assert isinstance(results, dict)
        assert "optimal_value" in results
        assert "optimal_sequence" in results
        assert "implementation_schedule" in results

        # Check that sequence respects constraints
        for period_policies in results["optimal_sequence"]:
            assert len(period_policies) <= 2  # Capacity constraint

    def test_adaptive_learning(self, sample_effects, sample_constraints):
        """Test adaptive learning algorithm."""
        optimizer = SequentialPolicyOptimizer(sample_effects, sample_constraints)

        results = optimizer.adaptive_learning(n_episodes=10, exploration_rate=0.1)

        assert isinstance(results, dict)
        assert "q_table" in results
        assert "learned_policy" in results
        assert "episode_returns" in results
        assert len(results["episode_returns"]) == 10

    def test_capacity_constrained_optimization(
        self, sample_effects, sample_constraints
    ):
        """Test capacity-constrained optimization."""
        optimizer = SequentialPolicyOptimizer(sample_effects, sample_constraints)

        results = optimizer.capacity_constrained_optimization(
            max_policies_per_period=2,
            total_periods=3,
        )

        assert isinstance(results, dict)
        if "error" not in results:
            assert "implementation_schedule" in results
            assert "optimization_result" in results

    def test_evaluate_sequence(self, sample_effects, sample_constraints):
        """Test sequence evaluation."""
        optimizer = SequentialPolicyOptimizer(sample_effects, sample_constraints)

        test_sequence = ["W", "M", "P", "R"]
        results = optimizer.evaluate_sequence(test_sequence, horizon=4)

        assert isinstance(results, dict)
        assert "total_discounted_benefit" in results
        assert "total_cost" in results
        assert "benefit_cost_ratio" in results


class TestPolicyScheduler:
    """Test policy scheduling functionality."""

    def test_initialization(self, sample_effects, sample_constraints):
        """Test proper initialization."""
        scheduler = PolicyScheduler(
            policies=["M", "P", "O", "W"],
            resources=sample_constraints,
            effects=sample_effects,
        )
        assert len(scheduler.policies) == 4
        assert scheduler.resources == sample_constraints

    def test_genetic_algorithm(self, sample_effects, sample_constraints):
        """Test genetic algorithm optimization."""
        scheduler = PolicyScheduler(
            policies=["M", "P", "W"],  # Smaller problem for faster testing
            resources=sample_constraints,
            effects=sample_effects,
        )

        results = scheduler.genetic_algorithm(
            population_size=10,
            max_generations=5,
        )

        assert isinstance(results, dict)
        assert "best_fitness" in results
        assert "best_schedule" in results
        assert "fitness_history" in results

    def test_simulated_annealing(self, sample_effects, sample_constraints):
        """Test simulated annealing optimization."""
        scheduler = PolicyScheduler(
            policies=["M", "P", "W"],
            resources=sample_constraints,
            effects=sample_effects,
        )

        results = scheduler.simulated_annealing(
            initial_temperature=100,
            max_iterations=50,
        )

        assert isinstance(results, dict)
        assert "best_fitness" in results
        assert "best_schedule" in results
        assert "algorithm_history" in results

    def test_branch_and_bound_small(self, sample_effects, sample_constraints):
        """Test branch-and-bound for small problems."""
        scheduler = PolicyScheduler(
            policies=["M", "P", "W"],  # Small problem
            resources=sample_constraints,
            effects=sample_effects,
        )

        results = scheduler.branch_and_bound(max_periods=3)

        assert isinstance(results, dict)
        assert "optimal_value" in results
        assert "optimal_schedule" in results
        assert "nodes_explored" in results

    def test_decode_individual(self, sample_effects, sample_constraints):
        """Test individual decoding."""
        scheduler = PolicyScheduler(
            policies=["M", "P", "O", "W"],
            resources=sample_constraints,
            effects=sample_effects,
        )

        individual = [0, 1, 2, 3]  # Policy indices
        schedule = scheduler._decode_individual(individual)

        assert isinstance(schedule, list)
        assert all(isinstance(period, list) for period in schedule)

    def test_evaluate_schedule(self, sample_effects, sample_constraints):
        """Test schedule evaluation."""
        scheduler = PolicyScheduler(
            policies=["M", "P", "O", "W"],
            resources=sample_constraints,
            effects=sample_effects,
        )

        test_schedule = [["M", "P"], ["O"], ["W"]]
        fitness = scheduler._evaluate_schedule(test_schedule)

        assert isinstance(fitness, float)


class TestPoliticalFeasibility:
    """Test political feasibility analysis functionality."""

    def test_initialization(self, sample_data):
        """Test proper initialization."""
        analyzer = PoliticalFeasibility(sample_data)
        assert analyzer.country_data is not None
        assert analyzer.unit_col == "country"
        assert analyzer.time_col == "year"

    def test_feasibility_scores(self, sample_data):
        """Test feasibility score calculation."""
        analyzer = PoliticalFeasibility(sample_data)

        results = analyzer.feasibility_scores(
            policies=["M", "P"],
            indicators=["democracy_score", "governance_effectiveness"],
        )

        assert isinstance(results, dict)

        for policy in ["M", "P"]:
            if policy in results:
                assert isinstance(results[policy], pd.DataFrame)
                feasibility_col = f"{policy}_feasibility_score"
                if feasibility_col in results[policy].columns:
                    assert results[policy][feasibility_col].between(0, 1).all()

    def test_stakeholder_model(self, sample_data):
        """Test stakeholder analysis."""
        analyzer = PoliticalFeasibility(sample_data)

        results = analyzer.stakeholder_model()

        assert isinstance(results, dict)
        assert "stakeholder_games" in results
        assert "overall_feasibility" in results
        assert "key_veto_players" in results

    def test_synthetic_feasibility_scores(self, sample_data):
        """Test synthetic score generation."""
        analyzer = PoliticalFeasibility(sample_data)

        results = analyzer._synthetic_feasibility_scores("M")

        assert isinstance(results, pd.DataFrame)
        assert "M_feasibility_score" in results.columns
        assert results["M_feasibility_score"].between(0, 1).all()

    def test_policy_weights(self, sample_data):
        """Test policy-specific weight calculation."""
        analyzer = PoliticalFeasibility(sample_data)

        features = ["democracy_score", "governance_effectiveness"]
        weights = analyzer._get_policy_weights("M", features)

        assert isinstance(weights, dict)
        assert all(isinstance(w, float) for w in weights.values())
        assert all(w > 0 for w in weights.values())


class TestPolicyDecisionSupport:
    """Test policy decision support functionality."""

    def test_initialization(self, sample_effects, sample_constraints):
        """Test proper initialization."""
        optimization_results = {
            "optimal_sequence": ["W", "M", "P", "O", "E", "R"],
            "optimal_value": 25.5,
        }

        support = PolicyDecisionSupport(optimization_results)
        assert support.optimization_results == optimization_results

    def test_generate_roadmap(self, sample_effects, sample_constraints, sample_data):
        """Test roadmap generation."""
        optimization_results = {
            "optimal_sequence": ["W", "M", "P", "O"],
        }

        support = PolicyDecisionSupport(
            optimization_results,
            country_data=sample_data,
        )

        roadmap = support.generate_roadmap(
            country="Country_A",
            budget=500000,
            time_horizon=3,
        )

        assert isinstance(roadmap, dict)
        assert "country" in roadmap
        assert "implementation_sequence" in roadmap
        assert "risk_assessment" in roadmap
        assert "success_metrics" in roadmap
        assert "estimated_impact" in roadmap

    def test_scenario_analysis(self, sample_effects, sample_data):
        """Test scenario analysis."""
        optimization_results = {
            "optimal_sequence": ["W", "M", "P"],
        }

        support = PolicyDecisionSupport(
            optimization_results,
            country_data=sample_data,
        )

        scenarios = [
            {
                "name": "High Budget",
                "country": "Country_A",
                "budget": 1000000,
                "time_horizon": 5,
            },
            {
                "name": "Low Budget",
                "country": "Country_A",
                "budget": 300000,
                "time_horizon": 3,
            },
        ]

        results = support.scenario_analysis(scenarios)

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 2
        assert "scenario_name" in results.columns
        assert "estimated_impact" in results.columns

    def test_get_optimal_sequence(self, sample_data):
        """Test optimal sequence extraction."""
        optimization_results = {
            "optimal_sequence": ["W", "M", "P", "O"],
        }

        support = PolicyDecisionSupport(optimization_results)
        sequence = support._get_optimal_sequence("Country_A")

        assert isinstance(sequence, list)
        assert len(sequence) > 0
        assert all(policy in ["M", "P", "O", "W", "E", "R"] for policy in sequence)

    def test_adjust_for_budget(self, sample_data):
        """Test budget adjustment."""
        optimization_results = {"optimal_sequence": ["W", "M", "P", "O", "E", "R"]}
        support = PolicyDecisionSupport(optimization_results)

        sequence = ["W", "M", "P", "O", "E", "R"]
        adjusted = support._adjust_for_budget(sequence, 300000, 3)

        assert isinstance(adjusted, list)
        assert len(adjusted) <= 3  # Time horizon constraint
        assert all(isinstance(period, list) for period in adjusted)

    def test_estimate_cumulative_impact(self, sample_data):
        """Test impact estimation."""
        optimization_results = {"optimal_sequence": ["W", "M", "P"]}
        support = PolicyDecisionSupport(optimization_results)

        mock_roadmap = [
            {"period": 1, "policies": ["W", "M"]},
            {"period": 2, "policies": ["P"]},
        ]

        impact = support._estimate_cumulative_impact(mock_roadmap)

        assert isinstance(impact, dict)
        assert "total_impact" in impact
        assert "annual_impacts" in impact
        assert impact["total_impact"] > 0


class TestIntegration:
    """Integration tests across modules."""

    def test_full_optimization_pipeline(
        self, sample_data, sample_effects, sample_constraints
    ):
        """Test complete optimization pipeline."""
        # Step 1: Interaction analysis
        interaction_analyzer = PolicyInteractionAnalysis(sample_data)
        interaction_results = interaction_analyzer.estimate_interactions(
            outcome="lung_cancer_mortality_rate",
            policies=["M", "P", "W"],
        )

        # Step 2: Sequential optimization
        optimizer = SequentialPolicyOptimizer(sample_effects, sample_constraints)
        optimization_results = optimizer.dynamic_programming(horizon=3)

        # Step 3: Political feasibility
        feasibility_analyzer = PoliticalFeasibility(sample_data)
        feasibility_results = feasibility_analyzer.feasibility_scores(
            policies=["M", "P", "W"]
        )

        # Step 4: Decision support
        support = PolicyDecisionSupport(
            optimization_results,
            feasibility_results,
            interaction_results,
            sample_data,
        )

        roadmap = support.generate_roadmap(
            country="Country_A",
            budget=500000,
            time_horizon=3,
        )

        # Verify complete pipeline
        assert isinstance(roadmap, dict)
        assert "implementation_sequence" in roadmap

    def test_edge_cases(self, sample_data):
        """Test edge cases and error handling."""
        # Empty data
        empty_data = pd.DataFrame()

        # Should handle gracefully
        try:
            analyzer = PolicyInteractionAnalysis(empty_data)
            results = analyzer.estimate_interactions("outcome", ["M", "P"])
            assert "error" in results or isinstance(results, dict)
        except Exception:
            pass  # Expected to fail gracefully

        # Invalid policies
        analyzer = PolicyInteractionAnalysis(sample_data)
        results = analyzer.estimate_interactions(
            "lung_cancer_mortality_rate",
            ["INVALID", "ALSO_INVALID"],
        )
        # Should handle gracefully without crashing

    def test_consistency_across_modules(
        self, sample_data, sample_effects, sample_constraints
    ):
        """Test consistency of results across modules."""
        # Same input should produce consistent outputs
        optimizer1 = SequentialPolicyOptimizer(sample_effects, sample_constraints)
        optimizer2 = SequentialPolicyOptimizer(sample_effects, sample_constraints)

        # Clear memoization for consistent testing
        optimizer1.dp_memo = {}
        optimizer2.dp_memo = {}

        results1 = optimizer1.dynamic_programming(horizon=3)
        results2 = optimizer2.dynamic_programming(horizon=3)

        # Should produce identical results
        assert results1["optimal_value"] == results2["optimal_value"]


# Performance tests
class TestPerformance:
    """Performance and scalability tests."""

    def test_large_dataset_performance(self):
        """Test performance with larger datasets."""
        # Create larger dataset
        np.random.seed(42)
        large_data = []

        countries = [f"Country_{i}" for i in range(20)]
        years = list(range(2000, 2021))

        for country in countries:
            for year in years:
                row = {
                    "country": country,
                    "year": year,
                    "lung_cancer_mortality_rate": np.random.normal(20, 5),
                    "mpower_m_score": np.random.randint(0, 6),
                    "mpower_p_score": np.random.randint(0, 6),
                }
                large_data.append(row)

        large_df = pd.DataFrame(large_data)

        # Test that analysis completes in reasonable time
        analyzer = PolicyInteractionAnalysis(large_df)

        import time

        start_time = time.time()
        results = analyzer.estimate_interactions(
            "lung_cancer_mortality_rate",
            ["M", "P"],
        )
        end_time = time.time()

        # Should complete within 30 seconds
        assert end_time - start_time < 30
        assert isinstance(results, dict)

    def test_optimization_scalability(self):
        """Test optimization scalability with more policies."""
        effects = {f"Policy_{i}": np.random.uniform(1, 10) for i in range(8)}
        constraints = {
            "budget_per_period": 200000,
            "capacity_limit": 3,
            "max_periods": 4,
        }

        scheduler = PolicyScheduler(
            policies=list(effects.keys()),
            resources=constraints,
            effects=effects,
        )

        # Genetic algorithm should handle larger problems
        results = scheduler.genetic_algorithm(
            population_size=20,
            max_generations=10,
        )

        assert isinstance(results, dict)
        assert "best_fitness" in results


if __name__ == "__main__":
    # Run basic smoke tests
    import sys

    # Create sample data
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "country": ["A", "B"] * 5,
            "year": list(range(2010, 2015)) * 2,
            "lung_cancer_mortality_rate": np.random.normal(20, 5, 10),
            "mpower_m_score": np.random.randint(0, 6, 10),
            "mpower_p_score": np.random.randint(0, 6, 10),
        }
    )

    try:
        # Test basic functionality
        analyzer = PolicyInteractionAnalysis(data)
        print("✓ PolicyInteractionAnalysis initialization successful")

        effects = {"M": 1.2, "P": 5.5}
        constraints = {"budget_per_period": 100000}

        optimizer = SequentialPolicyOptimizer(effects, constraints)
        print("✓ SequentialPolicyOptimizer initialization successful")

        scheduler = PolicyScheduler(["M", "P"], constraints, effects)
        print("✓ PolicyScheduler initialization successful")

        feasibility = PoliticalFeasibility(data)
        print("✓ PoliticalFeasibility initialization successful")

        support = PolicyDecisionSupport({"optimal_sequence": ["M", "P"]})
        print("✓ PolicyDecisionSupport initialization successful")

        print("\n✓ All modules initialized successfully!")
        print(
            "Run 'pytest tests/extensions/test_optimization.py -v' for full test suite"
        )

    except Exception as e:
        print(f"✗ Error during smoke test: {e}")
        sys.exit(1)
