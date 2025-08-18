"""Budget Optimization for MPOWER Policy Resource Allocation.

This module provides optimization methods for allocating limited budgets
across MPOWER policy components to maximize health outcomes.
"""

from typing import Any

import numpy as np
import pandas as pd

from scipy.optimize import linprog, minimize


class BudgetOptimizer:
    """Optimizes resource allocation across MPOWER policies.

    This class provides linear programming and portfolio optimization methods
    for optimal budget allocation under various constraints.

    Args:
        budget: Total available budget
        policies: List of available MPOWER policy components
        costs: Dictionary of policy implementation costs
        effects: Dictionary of policy health effects
    """

    def __init__(
        self,
        budget: float,
        policies: list[str] | None = None,
        costs: dict[str, float] | None = None,
        effects: dict[str, float] | None = None,
    ):
        """Initialize the budget optimizer."""
        self.budget = budget
        self.policies = (
            policies if policies is not None else ["M", "P", "O", "W", "E", "R"]
        )
        self.costs = costs if costs is not None else self._generate_default_costs()
        self.effects = (
            effects if effects is not None else self._generate_default_effects()
        )

    def _generate_default_costs(self) -> dict[str, float]:
        """Generate default policy costs if not provided."""
        default_costs = {
            "M": 500000,  # Monitoring
            "P": 1000000,  # Protect from smoke
            "O": 2500000,  # Offer help to quit
            "W": 800000,  # Warn about dangers
            "E": 1200000,  # Enforce bans
            "R": 300000,  # Raise taxes
        }
        return {p: default_costs.get(p, 1000000) for p in self.policies}

    def _generate_default_effects(self) -> dict[str, float]:
        """Generate default policy effects (QALYs) if not provided."""
        default_effects = {
            "M": 100,  # Monitoring
            "P": 500,  # Protect from smoke
            "O": 800,  # Offer help to quit
            "W": 300,  # Warn about dangers
            "E": 400,  # Enforce bans
            "R": 600,  # Raise taxes
        }
        return {p: default_effects.get(p, 400) for p in self.policies}

    def optimize_allocation(
        self, constraints: dict[str, Any] | None = None, method: str = "linear"
    ) -> dict[str, Any]:
        """Optimize budget allocation using linear programming.

        Args:
            constraints: Additional constraints (min/max allocations, capacity)
            method: Optimization method ('linear' or 'nonlinear')

        Returns:
            Dictionary with optimal allocation and outcomes
        """
        if constraints is None:
            constraints = {}

        len(self.policies)

        if method == "linear":
            return self._linear_optimization(constraints)
        return self._nonlinear_optimization(constraints)

    def _linear_optimization(self, constraints: dict[str, Any]) -> dict[str, Any]:
        """Linear programming optimization for budget allocation.

        Maximize: sum(effects[i] * x[i])
        Subject to: sum(costs[i] * x[i]) <= budget
                   0 <= x[i] <= 1 (fraction of policy implemented)
        """
        n = len(self.policies)

        # Objective: maximize effects (negative for minimization in linprog)
        c = [-self.effects[p] for p in self.policies]

        # Budget constraint: sum(costs * x) <= budget
        A_ub = [[self.costs[p] for p in self.policies]]
        b_ub = [self.budget]

        # Bounds: 0 <= x <= 1 (fraction of implementation)
        bounds = []
        for policy in self.policies:
            min_val = constraints.get(f"{policy}_min", 0)
            max_val = constraints.get(f"{policy}_max", 1)
            bounds.append((min_val, max_val))

        # Additional inequality constraints
        if "min_policies" in constraints:
            # At least min_policies should be partially implemented
            A_ub.append([-1] * n)
            b_ub.append(-constraints["min_policies"])

        if "synergy_pairs" in constraints:
            # Ensure synergistic policies are implemented together
            for pair in constraints["synergy_pairs"]:
                idx1 = self.policies.index(pair[0])
                idx2 = self.policies.index(pair[1])
                # x[idx1] - x[idx2] <= 0.2 (similar implementation levels)
                constraint = [0] * n
                constraint[idx1] = 1
                constraint[idx2] = -1
                A_ub.append(constraint)
                b_ub.append(0.2)
                # Reverse constraint
                constraint = [0] * n
                constraint[idx1] = -1
                constraint[idx2] = 1
                A_ub.append(constraint)
                b_ub.append(0.2)

        # Solve
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

        if result.success:
            allocation = {policy: result.x[i] for i, policy in enumerate(self.policies)}

            # Calculate outcomes
            total_cost = sum(self.costs[p] * allocation[p] for p in self.policies)
            total_effect = sum(self.effects[p] * allocation[p] for p in self.policies)

            return {
                "success": True,
                "allocation": allocation,
                "total_cost": total_cost,
                "total_effect": total_effect,
                "budget_utilization": total_cost / self.budget,
                "cost_per_effect": total_cost / total_effect
                if total_effect > 0
                else float("inf"),
                "method": "linear_programming",
            }
        return {
            "success": False,
            "message": result.message,
            "method": "linear_programming",
        }

    def _nonlinear_optimization(self, constraints: dict[str, Any]) -> dict[str, Any]:
        """Nonlinear optimization considering diminishing returns."""
        n = len(self.policies)

        # Objective function with diminishing returns
        def objective(x):
            total_effect = 0
            for i, policy in enumerate(self.policies):
                # Diminishing returns: sqrt for sub-linear growth
                total_effect += self.effects[policy] * np.sqrt(x[i])
            return -total_effect  # Negative for minimization

        # Budget constraint
        def budget_constraint(x):
            return self.budget - sum(
                self.costs[p] * x[i] for i, p in enumerate(self.policies)
            )

        # Initial guess (equal allocation)
        x0 = [0.5] * n

        # Bounds
        bounds = []
        for policy in self.policies:
            min_val = constraints.get(f"{policy}_min", 0)
            max_val = constraints.get(f"{policy}_max", 1)
            bounds.append((min_val, max_val))

        # Constraints
        cons = [{"type": "ineq", "fun": budget_constraint}]

        # Add minimum implementation constraint
        if "min_total" in constraints:
            cons.append(
                {"type": "ineq", "fun": lambda x: sum(x) - constraints["min_total"]}
            )

        # Solve
        result = minimize(
            objective, x0, method="SLSQP", bounds=bounds, constraints=cons
        )

        if result.success:
            allocation = {policy: result.x[i] for i, policy in enumerate(self.policies)}

            # Calculate outcomes
            total_cost = sum(self.costs[p] * allocation[p] for p in self.policies)
            total_effect = -result.fun  # Reverse sign

            return {
                "success": True,
                "allocation": allocation,
                "total_cost": total_cost,
                "total_effect": total_effect,
                "budget_utilization": total_cost / self.budget,
                "cost_per_effect": total_cost / total_effect
                if total_effect > 0
                else float("inf"),
                "method": "nonlinear_optimization",
            }
        return {
            "success": False,
            "message": result.message,
            "method": "nonlinear_optimization",
        }

    def portfolio_optimization(
        self, returns: pd.DataFrame | None = None, risk_aversion: float = 1.0
    ) -> dict[str, Any]:
        """Markowitz-style portfolio optimization for policy mix.

        Balances expected returns (health effects) with risk (uncertainty).

        Args:
            returns: Historical returns data for policies (for covariance)
            risk_aversion: Risk aversion parameter (higher = more conservative)

        Returns:
            Dictionary with optimal portfolio allocation
        """
        n = len(self.policies)

        # Generate synthetic returns data if not provided
        if returns is None:
            returns = self._generate_synthetic_returns()

        # Calculate expected returns and covariance
        expected_returns = returns.mean().values
        cov_matrix = returns.cov().values

        # Objective: maximize returns - risk_aversion * variance
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            return -(portfolio_return - risk_aversion * portfolio_variance)

        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1},  # Weights sum to 1
            {
                "type": "ineq",
                "fun": lambda x: self.budget
                - np.dot(x, [self.costs[p] for p in self.policies]),
            },  # Budget
        ]

        # Bounds: 0 <= weight <= 1
        bounds = [(0, 1) for _ in range(n)]

        # Initial guess
        x0 = np.array([1 / n] * n)

        # Optimize
        result = minimize(
            objective, x0, method="SLSQP", bounds=bounds, constraints=constraints
        )

        if result.success:
            weights = result.x
            allocation = {policy: weights[i] for i, policy in enumerate(self.policies)}

            # Calculate portfolio metrics
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            sharpe_ratio = (
                portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
            )

            return {
                "success": True,
                "allocation": allocation,
                "expected_return": portfolio_return,
                "risk": portfolio_risk,
                "sharpe_ratio": sharpe_ratio,
                "risk_aversion": risk_aversion,
                "method": "portfolio_optimization",
            }
        return {
            "success": False,
            "message": result.message,
            "method": "portfolio_optimization",
        }

    def _generate_synthetic_returns(self, n_periods: int = 100) -> pd.DataFrame:
        """Generate synthetic returns data for portfolio optimization."""
        np.random.seed(42)  # For reproducibility

        # Base returns with some correlation
        returns_data = {}
        base_returns = np.random.randn(n_periods) * 0.1

        for policy in self.policies:
            # Each policy has base returns plus individual variation
            policy_specific = np.random.randn(n_periods) * 0.05
            returns_data[policy] = (
                base_returns * 0.5 + policy_specific + self.effects[policy] / 1000
            )

        return pd.DataFrame(returns_data)

    def sensitivity_to_budget(
        self, budget_range: tuple[float, float] | None = None, n_points: int = 20
    ) -> pd.DataFrame:
        """Analyze how optimal allocation changes with budget.

        Args:
            budget_range: (min, max) budget to test
            n_points: Number of budget levels to test

        Returns:
            DataFrame with allocations at different budget levels
        """
        if budget_range is None:
            budget_range = (self.budget * 0.5, self.budget * 2.0)

        budgets = np.linspace(budget_range[0], budget_range[1], n_points)
        results = []

        original_budget = self.budget

        for budget in budgets:
            self.budget = budget
            opt_result = self.optimize_allocation()

            if opt_result["success"]:
                result_row = {
                    "budget": budget,
                    "total_effect": opt_result["total_effect"],
                    "cost_per_effect": opt_result["cost_per_effect"],
                }
                # Add allocation for each policy
                for policy, allocation in opt_result["allocation"].items():
                    result_row[f"{policy}_allocation"] = allocation

                results.append(result_row)

        self.budget = original_budget
        return pd.DataFrame(results)

    def multi_objective_optimization(
        self,
        objectives: dict[str, dict[str, float]] | None = None,
        weights: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Multi-objective optimization balancing different outcomes.

        Args:
            objectives: Dict of objective names to policy effects
            weights: Importance weights for each objective

        Returns:
            Dictionary with optimal allocation balancing objectives
        """
        if objectives is None:
            # Default: optimize for both QALYs and equity
            objectives = {
                "qalys": self.effects,
                "equity": dict.fromkeys(self.policies, 1.0),  # Equal weight for equity
            }

        if weights is None:
            weights = dict.fromkeys(objectives, 1.0)

        n = len(self.policies)

        # Combined objective function
        def objective(x):
            total_value = 0
            for obj_name, obj_effects in objectives.items():
                obj_value = sum(
                    obj_effects[p] * x[i] for i, p in enumerate(self.policies)
                )
                total_value += weights[obj_name] * obj_value
            return -total_value  # Negative for minimization

        # Budget constraint
        def budget_constraint(x):
            return self.budget - sum(
                self.costs[p] * x[i] for i, p in enumerate(self.policies)
            )

        # Bounds
        bounds = [(0, 1) for _ in range(n)]

        # Constraints
        cons = [{"type": "ineq", "fun": budget_constraint}]

        # Initial guess
        x0 = [0.5] * n

        # Optimize
        result = minimize(
            objective, x0, method="SLSQP", bounds=bounds, constraints=cons
        )

        if result.success:
            allocation = {policy: result.x[i] for i, policy in enumerate(self.policies)}

            # Calculate outcomes for each objective
            objective_values = {}
            for obj_name, obj_effects in objectives.items():
                obj_value = sum(obj_effects[p] * allocation[p] for p in self.policies)
                objective_values[obj_name] = obj_value

            total_cost = sum(self.costs[p] * allocation[p] for p in self.policies)

            return {
                "success": True,
                "allocation": allocation,
                "total_cost": total_cost,
                "objective_values": objective_values,
                "weighted_value": -result.fun,
                "budget_utilization": total_cost / self.budget,
                "method": "multi_objective",
            }
        return {
            "success": False,
            "message": result.message,
            "method": "multi_objective",
        }

    def incremental_allocation(
        self, step_size: float | None = None
    ) -> list[dict[str, Any]]:
        """Determine optimal incremental allocation strategy.

        Shows which policies to implement first with limited budget.

        Args:
            step_size: Budget increment size (default: 10% of total)

        Returns:
            List of allocation steps with cumulative effects
        """
        if step_size is None:
            step_size = self.budget * 0.1

        steps = []
        current_budget = 0
        allocated = dict.fromkeys(self.policies, 0)

        while current_budget < self.budget:
            current_budget = min(current_budget + step_size, self.budget)

            # Find best marginal investment
            best_policy = None
            best_ratio = 0

            for policy in self.policies:
                if allocated[policy] < 1:  # Not fully allocated
                    # Marginal cost and effect
                    marginal_cost = self.costs[policy] * 0.1  # 10% increment
                    marginal_effect = self.effects[policy] * 0.1

                    # Check if affordable
                    total_cost = sum(
                        self.costs[p] * allocated[p] for p in self.policies
                    )
                    if total_cost + marginal_cost <= current_budget:
                        ratio = marginal_effect / marginal_cost
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_policy = policy

            if best_policy:
                allocated[best_policy] = min(allocated[best_policy] + 0.1, 1.0)

            # Record step
            total_cost = sum(self.costs[p] * allocated[p] for p in self.policies)
            total_effect = sum(self.effects[p] * allocated[p] for p in self.policies)

            steps.append(
                {
                    "budget": current_budget,
                    "allocation": allocated.copy(),
                    "total_cost": total_cost,
                    "total_effect": total_effect,
                    "last_added": best_policy,
                    "cost_effectiveness": total_effect / total_cost
                    if total_cost > 0
                    else 0,
                }
            )

        return steps
