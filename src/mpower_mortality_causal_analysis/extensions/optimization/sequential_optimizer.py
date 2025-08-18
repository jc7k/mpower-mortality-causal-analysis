"""Sequential Policy Optimization for MPOWER Implementation Timing.

This module optimizes the timing and sequence of MPOWER component implementation
using dynamic programming and adaptive learning approaches.
"""

import warnings

from typing import Any

import numpy as np

from scipy.optimize import minimize

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
DISCOUNT_RATE = 0.03  # Annual discount rate for future benefits


class SequentialPolicyOptimizer:
    """Optimizes policy implementation timing using dynamic programming.

    This class determines the optimal sequence and timing for implementing
    MPOWER components to maximize cumulative health benefits under constraints.

    Parameters:
        effects (dict): Estimated treatment effects for each component
        constraints (dict): Implementation constraints (budget, capacity, etc.)
        discount_rate (float): Discount rate for future benefits
        learning_rate (float): Learning rate for adaptive algorithms
    """

    def __init__(
        self,
        effects: dict[str, float],
        constraints: dict[str, Any],
        discount_rate: float = DISCOUNT_RATE,
        learning_rate: float = 0.1,
    ) -> None:
        self.effects = effects
        self.constraints = constraints
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.dp_memo = {}
        self.learning_history = []

    def dynamic_programming(
        self,
        horizon: int,
        budget_per_period: float | None = None,
        capacity_limit: int | None = None,
    ) -> dict[str, Any]:
        """Dynamic programming solution for optimal implementation sequence.

        Args:
            horizon: Number of time periods to optimize over
            budget_per_period: Budget constraint per period
            capacity_limit: Maximum policies per period

        Returns:
            Optimal sequence and value function
        """
        if budget_per_period is None:
            budget_per_period = self.constraints.get("budget_per_period", float("inf"))

        if capacity_limit is None:
            capacity_limit = self.constraints.get(
                "capacity_limit", len(MPOWER_COMPONENTS)
            )

        # State representation: (period, implemented_policies_bitmask)
        # Action: subset of remaining policies to implement this period

        def state_value(period: int, implemented_mask: int) -> tuple[float, list[str]]:
            """Calculate optimal value and policy from given state.

            Args:
                period: Current time period (0 to horizon-1)
                implemented_mask: Bitmask of already implemented policies

            Returns:
                Tuple of (optimal_value, optimal_action_sequence)
            """
            # Memoization key
            state_key = (period, implemented_mask)
            if state_key in self.dp_memo:
                return self.dp_memo[state_key]

            # Base case: last period or all policies implemented
            if period >= horizon:
                self.dp_memo[state_key] = (0.0, [])
                return (0.0, [])

            # Get remaining policies
            remaining_policies = [
                comp
                for i, comp in enumerate(MPOWER_COMPONENTS)
                if not (implemented_mask & (1 << i))
            ]

            if not remaining_policies:
                # All policies implemented
                self.dp_memo[state_key] = (0.0, [])
                return (0.0, [])

            best_value = 0.0
            best_sequence = []

            # Try all feasible subsets of remaining policies
            for subset_mask in range(1, 1 << len(remaining_policies)):
                subset_policies = [
                    remaining_policies[i]
                    for i in range(len(remaining_policies))
                    if subset_mask & (1 << i)
                ]

                # Check constraints
                if len(subset_policies) > capacity_limit:
                    continue

                subset_cost = sum(
                    self.constraints.get("costs", {}).get(policy, 1.0)
                    for policy in subset_policies
                )
                if subset_cost > budget_per_period:
                    continue

                # Calculate immediate benefit
                immediate_benefit = sum(
                    self.effects.get(policy, 0.0) for policy in subset_policies
                )

                # Update implemented mask
                new_mask = implemented_mask
                for policy in subset_policies:
                    policy_idx = MPOWER_COMPONENTS.index(policy)
                    new_mask |= 1 << policy_idx

                # Calculate future value
                future_value, future_sequence = state_value(period + 1, new_mask)

                # Total discounted value
                total_value = immediate_benefit + (
                    future_value / (1 + self.discount_rate)
                )

                if total_value > best_value:
                    best_value = total_value
                    best_sequence = [subset_policies] + future_sequence

            self.dp_memo[state_key] = (best_value, best_sequence)
            return (best_value, best_sequence)

        # Solve from initial state
        optimal_value, optimal_sequence = state_value(0, 0)

        return {
            "optimal_value": optimal_value,
            "optimal_sequence": optimal_sequence,
            "horizon": horizon,
            "total_policies": sum(
                len(period_policies) for period_policies in optimal_sequence
            ),
            "implementation_schedule": {
                f"period_{i}": period_policies
                for i, period_policies in enumerate(optimal_sequence)
            },
        }

    def adaptive_learning(
        self,
        n_episodes: int = 100,
        exploration_rate: float = 0.1,
    ) -> dict[str, Any]:
        """Adaptive learning model for sequential policy decisions.

        Uses Q-learning to adapt implementation strategy based on observed outcomes.

        Args:
            n_episodes: Number of learning episodes
            exploration_rate: Epsilon for epsilon-greedy exploration

        Returns:
            Learned policy and performance metrics
        """
        # Initialize Q-table: Q[state][action] = expected return
        # State: (period, implemented_policies_bitmask, budget_remaining)
        # Action: policy to implement next

        n_states = 2 ** len(MPOWER_COMPONENTS)  # All possible implementation states
        n_actions = len(MPOWER_COMPONENTS)

        # Simplified state space for tractability
        q_table = np.zeros((n_states, n_actions))

        # Track learning performance
        episode_returns = []
        exploration_counts = np.zeros((n_states, n_actions))

        def get_state_index(implemented_mask: int) -> int:
            """Convert implemented policies bitmask to state index."""
            return implemented_mask

        def simulate_episode() -> float:
            """Simulate one episode of policy implementation."""
            implemented_mask = 0
            total_return = 0.0
            period = 0
            budget_remaining = self.constraints.get("total_budget", 10.0)

            while period < 5 and implemented_mask != (2 ** len(MPOWER_COMPONENTS) - 1):
                state_idx = get_state_index(implemented_mask)

                # Epsilon-greedy action selection
                if np.random.random() < exploration_rate:
                    # Explore: random action from available policies
                    available_actions = [
                        i
                        for i, comp in enumerate(MPOWER_COMPONENTS)
                        if not (implemented_mask & (1 << i))
                    ]
                    if not available_actions:
                        break
                    action = np.random.choice(available_actions)
                else:
                    # Exploit: best known action
                    available_actions = [
                        i
                        for i, comp in enumerate(MPOWER_COMPONENTS)
                        if not (implemented_mask & (1 << i))
                    ]
                    if not available_actions:
                        break

                    # Choose best available action
                    action = max(available_actions, key=lambda a: q_table[state_idx, a])

                # Implement policy and observe reward
                policy = MPOWER_COMPONENTS[action]
                cost = self.constraints.get("costs", {}).get(policy, 1.0)

                if cost <= budget_remaining:
                    # Valid action
                    reward = self.effects.get(policy, 0.0)
                    budget_remaining -= cost
                    implemented_mask |= 1 << action

                    # Update Q-value
                    next_state_idx = get_state_index(implemented_mask)

                    # Bellman update
                    old_q = q_table[state_idx, action]
                    if implemented_mask == (2 ** len(MPOWER_COMPONENTS) - 1):
                        # Terminal state
                        max_next_q = 0.0
                    else:
                        available_next = [
                            i
                            for i, comp in enumerate(MPOWER_COMPONENTS)
                            if not (implemented_mask & (1 << i))
                        ]
                        if available_next:
                            max_next_q = max(
                                q_table[next_state_idx, a] for a in available_next
                            )
                        else:
                            max_next_q = 0.0

                    new_q = old_q + self.learning_rate * (
                        reward + self.discount_rate * max_next_q - old_q
                    )
                    q_table[state_idx, action] = new_q

                    total_return += reward * (self.discount_rate**period)
                    exploration_counts[state_idx, action] += 1
                else:
                    # Invalid action (budget constraint)
                    q_table[state_idx, action] = -1000  # Large penalty
                    break

                period += 1

            return total_return

        # Run learning episodes
        for episode in range(n_episodes):
            episode_return = simulate_episode()
            episode_returns.append(episode_return)

            # Decay exploration rate
            if episode > 0 and episode % 20 == 0:
                exploration_rate *= 0.95

        # Extract learned policy
        learned_policy = {}
        for state in range(n_states):
            available_actions = [
                i for i, comp in enumerate(MPOWER_COMPONENTS) if not (state & (1 << i))
            ]
            if available_actions:
                best_action = max(available_actions, key=lambda a: q_table[state, a])
                learned_policy[state] = MPOWER_COMPONENTS[best_action]

        self.learning_history.extend(episode_returns)

        return {
            "q_table": q_table,
            "learned_policy": learned_policy,
            "episode_returns": episode_returns,
            "final_exploration_rate": exploration_rate,
            "average_return": np.mean(episode_returns[-20:])
            if len(episode_returns) >= 20
            else np.mean(episode_returns),
            "exploration_counts": exploration_counts,
        }

    def capacity_constrained_optimization(
        self,
        max_policies_per_period: int,
        total_periods: int,
    ) -> dict[str, Any]:
        """Optimize under capacity constraints using integer programming.

        Args:
            max_policies_per_period: Maximum policies implementable per period
            total_periods: Total number of implementation periods

        Returns:
            Optimal implementation schedule under capacity constraints
        """
        # Decision variables: x[policy][period] = 1 if policy implemented in period
        # Objective: maximize sum of discounted benefits
        # Constraints:
        #   - Each policy implemented at most once
        #   - Capacity constraint per period

        n_policies = len(MPOWER_COMPONENTS)

        # Use scipy.optimize for simplified problem
        def objective(x: np.ndarray) -> float:
            """Objective function: negative of total discounted benefits."""
            x_matrix = x.reshape((n_policies, total_periods))
            total_benefit = 0.0

            for policy_idx, policy in enumerate(MPOWER_COMPONENTS):
                effect = self.effects.get(policy, 0.0)
                for period in range(total_periods):
                    if x_matrix[policy_idx, period] > 0.5:  # Binary threshold
                        discount_factor = (1 + self.discount_rate) ** (-period)
                        total_benefit += effect * discount_factor

            return -total_benefit  # Minimize negative

        def constraints(x: np.ndarray) -> list[float]:
            """Constraint functions."""
            x_matrix = x.reshape((n_policies, total_periods))
            constraints_list = []

            # Each policy implemented at most once
            for policy_idx in range(n_policies):
                constraints_list.append(1.0 - np.sum(x_matrix[policy_idx, :]))

            # Capacity constraint per period
            for period in range(total_periods):
                constraints_list.append(
                    max_policies_per_period - np.sum(x_matrix[:, period])
                )

            return constraints_list

        # Initial guess: spread policies evenly
        x0 = np.zeros((n_policies, total_periods))
        policies_per_period = min(
            max_policies_per_period, n_policies // total_periods + 1
        )

        policy_idx = 0
        for period in range(total_periods):
            for _ in range(policies_per_period):
                if policy_idx < n_policies:
                    x0[policy_idx, period] = 1.0
                    policy_idx += 1

        x0 = x0.flatten()

        # Bounds: binary variables (relaxed to [0,1])
        bounds = [(0, 1) for _ in range(n_policies * total_periods)]

        # Solve optimization
        try:
            result = minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints={"type": "ineq", "fun": lambda x: constraints(x)},
                options={"maxiter": 1000},
            )

            # Round to binary solution
            x_optimal = result.x.reshape((n_policies, total_periods))
            x_binary = np.round(x_optimal)

            # Create implementation schedule
            schedule = {}
            for period in range(total_periods):
                period_policies = [
                    MPOWER_COMPONENTS[policy_idx]
                    for policy_idx in range(n_policies)
                    if x_binary[policy_idx, period] > 0.5
                ]
                schedule[f"period_{period}"] = period_policies

            return {
                "optimization_result": {
                    "success": result.success,
                    "optimal_value": -result.fun,
                    "message": result.message,
                },
                "implementation_schedule": schedule,
                "total_periods": total_periods,
                "capacity_limit": max_policies_per_period,
                "solution_matrix": x_binary.tolist(),
            }

        except Exception as e:
            return {"error": f"Optimization failed: {str(e)}"}

    def plot_learning_curve(self, save_path: str | None = None) -> None:
        """Plot learning performance over episodes.

        Args:
            save_path: Path to save plot (optional)
        """
        if not PLOTTING_AVAILABLE:
            warnings.warn("Plotting not available. Install matplotlib.")
            return

        if not self.learning_history:
            warnings.warn("No learning history available. Run adaptive_learning first.")
            return

        plt.figure(figsize=(10, 6))

        # Plot episode returns
        plt.subplot(1, 2, 1)
        plt.plot(self.learning_history, alpha=0.7)

        # Moving average
        window_size = min(20, len(self.learning_history) // 5)
        if window_size > 1:
            moving_avg = np.convolve(
                self.learning_history, np.ones(window_size) / window_size, mode="valid"
            )
            plt.plot(
                range(window_size - 1, len(self.learning_history)),
                moving_avg,
                "r-",
                linewidth=2,
            )

        plt.xlabel("Episode")
        plt.ylabel("Total Return")
        plt.title("Learning Performance")
        plt.grid(True, alpha=0.3)

        # Plot performance distribution
        plt.subplot(1, 2, 2)
        plt.hist(self.learning_history, bins=20, alpha=0.7, edgecolor="black")
        plt.xlabel("Total Return")
        plt.ylabel("Frequency")
        plt.title("Return Distribution")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def evaluate_sequence(self, sequence: list[str], horizon: int) -> dict[str, float]:
        """Evaluate the value of a given implementation sequence.

        Args:
            sequence: List of policies in implementation order
            horizon: Number of periods to evaluate over

        Returns:
            Evaluation metrics for the sequence
        """
        total_discounted_benefit = 0.0
        total_cost = 0.0

        for period, policy in enumerate(sequence):
            if period >= horizon:
                break

            # Calculate discounted benefit
            effect = self.effects.get(policy, 0.0)
            discount_factor = (1 + self.discount_rate) ** (-period)
            discounted_benefit = effect * discount_factor
            total_discounted_benefit += discounted_benefit

            # Add cost
            cost = self.constraints.get("costs", {}).get(policy, 1.0)
            total_cost += cost

        return {
            "total_discounted_benefit": total_discounted_benefit,
            "total_cost": total_cost,
            "benefit_cost_ratio": total_discounted_benefit / total_cost
            if total_cost > 0
            else float("inf"),
            "sequence_length": min(len(sequence), horizon),
        }
