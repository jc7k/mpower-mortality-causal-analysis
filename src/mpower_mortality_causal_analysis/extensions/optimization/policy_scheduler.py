"""Policy Scheduler for Combinatorial Optimization of MPOWER Implementation.

This module provides algorithms for scheduling policy implementation optimally
using exact and heuristic optimization methods.
"""

import random
import warnings

from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    warnings.warn(
        "Plotting packages not available. Visualization disabled.",
        stacklevel=2,
    )

# Constants
MPOWER_COMPONENTS = ["M", "P", "O", "W", "E", "R"]
MAX_GENERATIONS = 100
POPULATION_SIZE = 50


class PolicyScheduler:
    """Schedules policy implementation using combinatorial optimization.

    This class implements branch-and-bound and genetic algorithms to find
    optimal implementation schedules under resource constraints.

    Parameters:
        policies (list[str]): List of MPOWER policies to schedule
        resources (dict): Resource constraints and costs
        effects (dict): Expected effects of each policy
        dependencies (dict): Policy dependency relationships
    """

    def __init__(
        self,
        policies: list[str],
        resources: dict[str, Any],
        effects: dict[str, float] | None = None,
        dependencies: dict[str, list[str]] | None = None,
    ) -> None:
        self.policies = policies if policies else MPOWER_COMPONENTS
        self.resources = resources
        self.effects = effects or {}
        self.dependencies = dependencies or {}
        self.best_solution = None
        self.optimization_history = []

    def branch_and_bound(
        self,
        max_periods: int = 5,
        budget_per_period: float | None = None,
    ) -> dict[str, Any]:
        """Exact solution using branch-and-bound algorithm.

        Finds the optimal schedule for small to medium-sized problems
        by systematically exploring the solution space.

        Args:
            max_periods: Maximum number of implementation periods
            budget_per_period: Budget constraint per period

        Returns:
            Optimal schedule and solution metrics
        """
        if budget_per_period is None:
            budget_per_period = self.resources.get("budget_per_period", float("inf"))

        # State: (period, implemented_policies, remaining_budget)
        best_value = -float("inf")
        best_schedule = []
        nodes_explored = 0

        def upper_bound(
            period: int,
            implemented: set[str],
            remaining_budget: float,
        ) -> float:
            """Calculate upper bound for remaining policies."""
            if period >= max_periods:
                return 0.0

            # Greedy upper bound: implement best remaining policies
            remaining_policies = [p for p in self.policies if p not in implemented]

            # Sort by effect-to-cost ratio
            policy_values = []
            for policy in remaining_policies:
                effect = self.effects.get(policy, 0.0)
                cost = self.resources.get("costs", {}).get(policy, 1.0)
                ratio = effect / cost if cost > 0 else effect
                policy_values.append((ratio, policy, effect))

            policy_values.sort(reverse=True)

            # Simulate greedy implementation
            temp_budget = remaining_budget
            temp_implemented = implemented.copy()
            total_value = 0.0

            for remaining_period in range(period, max_periods):
                period_budget = min(temp_budget, budget_per_period)

                for ratio, policy, effect in policy_values:
                    if policy not in temp_implemented:
                        cost = self.resources.get("costs", {}).get(policy, 1.0)
                        if cost <= period_budget:
                            total_value += effect
                            temp_implemented.add(policy)
                            period_budget -= cost
                            temp_budget -= cost

                            if period_budget <= 0:
                                break

                if len(temp_implemented) == len(self.policies):
                    break

            return total_value

        def branch_and_bound_recursive(
            period: int,
            implemented: set[str],
            current_schedule: list[list[str]],
            current_value: float,
            remaining_budget: float,
        ) -> None:
            """Recursive branch-and-bound exploration."""
            nonlocal best_value, best_schedule, nodes_explored
            nodes_explored += 1

            # Pruning: check upper bound
            ub = current_value + upper_bound(period, implemented, remaining_budget)
            if ub <= best_value:
                return

            # Terminal condition
            if period >= max_periods or len(implemented) == len(self.policies):
                if current_value > best_value:
                    best_value = current_value
                    best_schedule = current_schedule.copy()
                return

            # Generate feasible subsets for this period
            remaining_policies = [p for p in self.policies if p not in implemented]

            # Check dependencies
            feasible_policies = []
            for policy in remaining_policies:
                deps = self.dependencies.get(policy, [])
                if all(dep in implemented for dep in deps):
                    feasible_policies.append(policy)

            # Try all feasible subsets (up to reasonable size)
            max_subset_size = min(
                len(feasible_policies),
                self.resources.get("max_policies_per_period", 3),
            )

            # Include empty subset (do nothing this period)
            subsets_to_try = [[]]

            # Add single policies
            for policy in feasible_policies:
                subsets_to_try.append([policy])

            # Add pairs and larger if manageable
            if len(feasible_policies) <= 6:  # Manageable for exact enumeration
                from itertools import combinations

                for size in range(
                    2, min(max_subset_size + 1, len(feasible_policies) + 1)
                ):
                    for subset in combinations(feasible_policies, size):
                        subsets_to_try.append(list(subset))

            for subset in subsets_to_try:
                # Check budget constraint
                subset_cost = sum(
                    self.resources.get("costs", {}).get(policy, 1.0)
                    for policy in subset
                )

                if subset_cost <= min(budget_per_period, remaining_budget):
                    # Calculate value of this subset
                    subset_value = sum(
                        self.effects.get(policy, 0.0) for policy in subset
                    )

                    # Update state
                    new_implemented = implemented | set(subset)
                    new_schedule = current_schedule + [subset]
                    new_value = current_value + subset_value
                    new_budget = remaining_budget - subset_cost

                    # Recurse
                    branch_and_bound_recursive(
                        period + 1,
                        new_implemented,
                        new_schedule,
                        new_value,
                        new_budget,
                    )

        # Start branch-and-bound
        total_budget = self.resources.get("total_budget", float("inf"))
        branch_and_bound_recursive(0, set(), [], 0.0, total_budget)

        return {
            "optimal_value": best_value,
            "optimal_schedule": best_schedule,
            "nodes_explored": nodes_explored,
            "schedule_dict": {
                f"period_{i}": period_policies
                for i, period_policies in enumerate(best_schedule)
            },
        }

    def genetic_algorithm(
        self,
        population_size: int = POPULATION_SIZE,
        max_generations: int = MAX_GENERATIONS,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
    ) -> dict[str, Any]:
        """Genetic algorithm for large problem instances.

        Uses evolutionary computation to find near-optimal solutions
        for complex scheduling problems.

        Args:
            population_size: Size of the population
            max_generations: Maximum number of generations
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover

        Returns:
            Best solution found and algorithm statistics
        """

        def create_individual() -> list[int]:
            """Create a random valid schedule representation.

            Individual encoding: list of policy indices in implementation order
            """
            return random.sample(range(len(self.policies)), len(self.policies))

        def evaluate_fitness(individual: list[int]) -> float:
            """Evaluate fitness of an individual."""
            schedule = self._decode_individual(individual)
            return self._evaluate_schedule(schedule)

        def tournament_selection(population: list[list[int]], k: int = 3) -> list[int]:
            """Tournament selection for parent selection."""
            tournament = random.sample(population, k)
            return max(tournament, key=evaluate_fitness)

        def order_crossover(parent1: list[int], parent2: list[int]) -> list[int]:
            """Order crossover (OX) for permutation encoding."""
            size = len(parent1)
            start, end = sorted(random.sample(range(size), 2))

            child = [-1] * size
            child[start:end] = parent1[start:end]

            pointer = end
            for city in parent2[end:] + parent2[:end]:
                if city not in child:
                    child[pointer % size] = city
                    pointer += 1

            return child

        def mutate(individual: list[int]) -> list[int]:
            """Swap mutation for permutation encoding."""
            mutated = individual.copy()
            if random.random() < mutation_rate:
                i, j = random.sample(range(len(mutated)), 2)
                mutated[i], mutated[j] = mutated[j], mutated[i]
            return mutated

        # Initialize population
        population = [create_individual() for _ in range(population_size)]

        best_fitness_history = []
        avg_fitness_history = []
        best_individual = None
        best_fitness = -float("inf")

        for generation in range(max_generations):
            # Evaluate population
            fitness_values = [evaluate_fitness(ind) for ind in population]

            # Track best solution
            gen_best_idx = np.argmax(fitness_values)
            gen_best_fitness = fitness_values[gen_best_idx]

            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_individual = population[gen_best_idx].copy()

            # Record statistics
            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(np.mean(fitness_values))

            # Create next generation
            new_population = []

            # Elitism: keep best individual
            new_population.append(best_individual.copy())

            # Generate offspring
            while len(new_population) < population_size:
                parent1 = tournament_selection(population)
                parent2 = tournament_selection(population)

                if random.random() < crossover_rate:
                    child = order_crossover(parent1, parent2)
                else:
                    child = parent1.copy()

                child = mutate(child)
                new_population.append(child)

            population = new_population

        # Decode best solution
        best_schedule = self._decode_individual(best_individual)

        self.optimization_history = {
            "best_fitness": best_fitness_history,
            "avg_fitness": avg_fitness_history,
        }

        return {
            "best_fitness": best_fitness,
            "best_schedule": best_schedule,
            "best_individual": best_individual,
            "generations": max_generations,
            "final_population_size": len(population),
            "fitness_history": {
                "best": best_fitness_history,
                "average": avg_fitness_history,
            },
        }

    def simulated_annealing(
        self,
        initial_temperature: float = 1000.0,
        cooling_rate: float = 0.95,
        min_temperature: float = 1.0,
        max_iterations: int = 1000,
    ) -> dict[str, Any]:
        """Simulated annealing optimization.

        Uses simulated annealing to find good solutions by accepting
        worse solutions with decreasing probability.

        Args:
            initial_temperature: Starting temperature
            cooling_rate: Temperature reduction factor
            min_temperature: Minimum temperature threshold
            max_iterations: Maximum iterations

        Returns:
            Best solution and algorithm statistics
        """

        def get_neighbor(solution: list[int]) -> list[int]:
            """Generate neighbor by swapping two random positions."""
            neighbor = solution.copy()
            i, j = random.sample(range(len(neighbor)), 2)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            return neighbor

        # Initialize
        current_solution = list(range(len(self.policies)))
        random.shuffle(current_solution)

        current_fitness = self._evaluate_schedule(
            self._decode_individual(current_solution)
        )

        best_solution = current_solution.copy()
        best_fitness = current_fitness

        temperature = initial_temperature
        temperature_history = []
        fitness_history = []
        acceptance_history = []

        iteration = 0
        while temperature > min_temperature and iteration < max_iterations:
            # Generate neighbor
            neighbor = get_neighbor(current_solution)
            neighbor_fitness = self._evaluate_schedule(
                self._decode_individual(neighbor)
            )

            # Acceptance criterion
            delta = neighbor_fitness - current_fitness

            if delta > 0 or random.random() < np.exp(delta / temperature):
                # Accept the neighbor
                current_solution = neighbor
                current_fitness = neighbor_fitness
                acceptance_history.append(1)

                # Update best if improved
                if current_fitness > best_fitness:
                    best_solution = current_solution.copy()
                    best_fitness = current_fitness
            else:
                acceptance_history.append(0)

            # Record statistics
            temperature_history.append(temperature)
            fitness_history.append(current_fitness)

            # Cool down
            temperature *= cooling_rate
            iteration += 1

        best_schedule = self._decode_individual(best_solution)

        return {
            "best_fitness": best_fitness,
            "best_schedule": best_schedule,
            "best_solution": best_solution,
            "iterations": iteration,
            "final_temperature": temperature,
            "acceptance_rate": np.mean(acceptance_history)
            if acceptance_history
            else 0.0,
            "algorithm_history": {
                "temperature": temperature_history,
                "fitness": fitness_history,
                "acceptance": acceptance_history,
            },
        }

    def _decode_individual(self, individual: list[int]) -> list[list[str]]:
        """Decode individual to schedule format.

        Args:
            individual: List of policy indices in implementation order

        Returns:
            Schedule as list of periods, each containing list of policies
        """
        max_periods = self.resources.get("max_periods", 5)
        budget_per_period = self.resources.get("budget_per_period", float("inf"))
        max_policies_per_period = self.resources.get(
            "max_policies_per_period", len(self.policies)
        )

        schedule = []
        current_period = []
        current_budget = budget_per_period

        for policy_idx in individual:
            policy = self.policies[policy_idx]
            cost = self.resources.get("costs", {}).get(policy, 1.0)

            # Check if policy fits in current period
            if (
                len(current_period) < max_policies_per_period
                and cost <= current_budget
                and len(schedule) < max_periods
            ):
                current_period.append(policy)
                current_budget -= cost
            else:
                # Start new period
                if current_period:
                    schedule.append(current_period)

                if len(schedule) >= max_periods:
                    break

                current_period = [policy]
                current_budget = budget_per_period - cost

        # Add final period if not empty
        if current_period and len(schedule) < max_periods:
            schedule.append(current_period)

        return schedule

    def _evaluate_schedule(self, schedule: list[list[str]]) -> float:
        """Evaluate the quality of a schedule.

        Args:
            schedule: List of periods with policies

        Returns:
            Schedule quality score
        """
        total_value = 0.0
        implemented = set()
        discount_rate = 0.03  # Annual discount rate

        for period, policies in enumerate(schedule):
            period_value = 0.0

            for policy in policies:
                # Check dependencies
                deps = self.dependencies.get(policy, [])
                if all(dep in implemented for dep in deps):
                    effect = self.effects.get(policy, 0.0)
                    discounted_effect = effect / ((1 + discount_rate) ** period)
                    period_value += discounted_effect
                    implemented.add(policy)
                else:
                    # Penalty for unmet dependencies
                    period_value -= 10.0

            total_value += period_value

        # Bonus for implementing all policies
        if len(implemented) == len(self.policies):
            total_value += 5.0

        return total_value

    def compare_algorithms(
        self,
        max_periods: int = 5,
        budget_per_period: float | None = None,
    ) -> dict[str, Any]:
        """Compare performance of different optimization algorithms.

        Args:
            max_periods: Maximum implementation periods
            budget_per_period: Budget per period

        Returns:
            Comparison results across algorithms
        """
        results = {}

        # Branch and bound (for small problems)
        if len(self.policies) <= 6:
            try:
                bnb_result = self.branch_and_bound(max_periods, budget_per_period)
                results["branch_and_bound"] = bnb_result
            except Exception as e:
                results["branch_and_bound"] = {"error": str(e)}

        # Genetic algorithm
        try:
            ga_result = self.genetic_algorithm()
            results["genetic_algorithm"] = ga_result
        except Exception as e:
            results["genetic_algorithm"] = {"error": str(e)}

        # Simulated annealing
        try:
            sa_result = self.simulated_annealing()
            results["simulated_annealing"] = sa_result
        except Exception as e:
            results["simulated_annealing"] = {"error": str(e)}

        # Find best algorithm
        best_algorithm = None
        best_value = -float("inf")

        for alg_name, alg_result in results.items():
            if "error" not in alg_result:
                value_key = (
                    "optimal_value" if "optimal_value" in alg_result else "best_fitness"
                )
                if value_key in alg_result and alg_result[value_key] > best_value:
                    best_value = alg_result[value_key]
                    best_algorithm = alg_name

        return {
            "algorithm_results": results,
            "best_algorithm": best_algorithm,
            "best_value": best_value,
            "comparison_summary": {
                alg: {"value": res.get("optimal_value", res.get("best_fitness", "N/A"))}
                for alg, res in results.items()
                if "error" not in res
            },
        }

    def plot_optimization_history(self, save_path: str | None = None) -> None:
        """Plot optimization algorithm performance history.

        Args:
            save_path: Path to save plot (optional)
        """
        if not PLOTTING_AVAILABLE:
            warnings.warn("Plotting not available. Install matplotlib.")
            return

        if not self.optimization_history:
            warnings.warn("No optimization history available.")
            return

        plt.figure(figsize=(12, 4))

        # Plot fitness evolution
        plt.subplot(1, 2, 1)
        if "best_fitness" in self.optimization_history:
            plt.plot(
                self.optimization_history["best_fitness"],
                label="Best Fitness",
                linewidth=2,
            )
        if "avg_fitness" in self.optimization_history:
            plt.plot(
                self.optimization_history["avg_fitness"],
                label="Average Fitness",
                alpha=0.7,
            )

        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Genetic Algorithm Performance")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot convergence
        plt.subplot(1, 2, 2)
        if "best_fitness" in self.optimization_history:
            best_fitness = self.optimization_history["best_fitness"]
            improvements = [
                i
                for i in range(1, len(best_fitness))
                if best_fitness[i] > best_fitness[i - 1]
            ]

            plt.scatter(
                improvements,
                [best_fitness[i] for i in improvements],
                color="red",
                s=50,
                alpha=0.7,
                label="Improvements",
            )
            plt.plot(best_fitness, color="blue", alpha=0.5, label="Best Fitness")

        plt.xlabel("Generation")
        plt.ylabel("Best Fitness")
        plt.title("Solution Improvement Over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()
