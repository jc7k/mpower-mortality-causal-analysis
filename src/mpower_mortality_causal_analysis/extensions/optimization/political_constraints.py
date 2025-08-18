"""Political Feasibility Analysis for MPOWER Policy Implementation.

This module models political economy constraints and feasibility
for tobacco control policy implementation.
"""

import warnings

from typing import Any

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    warnings.warn(
        "Plotting packages not available. Visualization disabled.",
        stacklevel=2,
    )

# Constants
MPOWER_COMPONENTS = ["M", "P", "O", "W", "E", "R"]
POLITICAL_INDICATORS = [
    "democracy_score",
    "governance_effectiveness",
    "regulatory_quality",
    "rule_of_law",
    "control_corruption",
    "voice_accountability",
]


class PoliticalFeasibility:
    """Models political economy constraints for policy implementation.

    This class analyzes implementation feasibility based on political
    and institutional factors that affect tobacco control policy adoption.

    Parameters:
        country_data (pd.DataFrame): Country-level political and economic indicators
        policy_data (pd.DataFrame): Historical policy implementation data
        unit_col (str): Column name for country identifier
        time_col (str): Column name for time identifier
    """

    def __init__(
        self,
        country_data: pd.DataFrame,
        policy_data: pd.DataFrame | None = None,
        unit_col: str = "country",
        time_col: str = "year",
    ) -> None:
        self.country_data = country_data.copy()
        self.policy_data = policy_data.copy() if policy_data is not None else None
        self.unit_col = unit_col
        self.time_col = time_col
        self.feasibility_models = {}

    def feasibility_scores(
        self,
        policies: list[str] | None = None,
        indicators: list[str] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Estimate implementation feasibility scores for policies.

        Uses political and institutional indicators to predict the likelihood
        of successful policy implementation.

        Args:
            policies: List of MPOWER policies to analyze (default: all)
            indicators: Political indicators to use (default: standard set)

        Returns:
            Dictionary mapping policies to feasibility score DataFrames
        """
        if policies is None:
            policies = MPOWER_COMPONENTS

        if indicators is None:
            # Use available indicators
            available_indicators = [
                col for col in POLITICAL_INDICATORS if col in self.country_data.columns
            ]
            if not available_indicators:
                # Fallback to common governance indicators
                available_indicators = [
                    col
                    for col in self.country_data.columns
                    if any(
                        keyword in col.lower()
                        for keyword in [
                            "democracy",
                            "governance",
                            "quality",
                            "corruption",
                        ]
                    )
                ]
            indicators = available_indicators

        if not indicators:
            return {"error": "No political indicators available in data"}

        feasibility_results = {}

        for policy in policies:
            policy_scores = self._estimate_policy_feasibility(policy, indicators)
            feasibility_results[policy] = policy_scores

        return feasibility_results

    def _estimate_policy_feasibility(
        self,
        policy: str,
        indicators: list[str],
    ) -> pd.DataFrame:
        """Estimate feasibility for a specific policy.

        Args:
            policy: MPOWER component to analyze
            indicators: Political indicators to use

        Returns:
            DataFrame with feasibility scores by country and time
        """
        # Create feasibility model based on available data
        analysis_data = self.country_data.copy()

        # Add policy implementation indicator if available
        policy_col = f"mpower_{policy.lower()}_score"
        has_policy_data = policy_col in analysis_data.columns

        if has_policy_data:
            # Binary implementation indicator (high vs low implementation)
            policy_threshold = 3 if policy == "W" else 4  # W has max score of 4
            analysis_data[f"{policy}_implemented"] = (
                analysis_data[policy_col] >= policy_threshold
            ).astype(int)

        # Prepare features
        feature_cols = [col for col in indicators if col in analysis_data.columns]

        if not feature_cols:
            # Create synthetic feasibility based on basic indicators
            return self._synthetic_feasibility_scores(policy)

        # Remove missing values for modeling
        model_data = analysis_data[
            [self.unit_col, self.time_col]
            + feature_cols
            + ([f"{policy}_implemented"] if has_policy_data else [])
        ].dropna()

        if len(model_data) == 0:
            return self._synthetic_feasibility_scores(policy)

        if has_policy_data and len(model_data) > 20:
            # Supervised learning approach
            feasibility_scores = self._supervised_feasibility_model(
                model_data, policy, feature_cols
            )
        else:
            # Unsupervised scoring approach
            feasibility_scores = self._unsupervised_feasibility_model(
                model_data, policy, feature_cols
            )

        return feasibility_scores

    def _supervised_feasibility_model(
        self,
        data: pd.DataFrame,
        policy: str,
        features: list[str],
    ) -> pd.DataFrame:
        """Build supervised model for feasibility prediction.

        Args:
            data: Analysis dataset
            policy: Policy component
            features: Feature columns

        Returns:
            DataFrame with feasibility predictions
        """
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_score
            from sklearn.preprocessing import StandardScaler

            # Prepare features and target
            X = data[features].values
            y = data[f"{policy}_implemented"].values

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Fit logistic regression
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_scaled, y)

            # Cross-validation score
            cv_scores = cross_val_score(model, X_scaled, y, cv=min(5, len(data) // 3))

            # Predict probabilities for all data
            all_data = self.country_data[
                [self.unit_col, self.time_col] + features
            ].dropna()

            if len(all_data) > 0:
                X_all = scaler.transform(all_data[features].values)
                feasibility_probs = model.predict_proba(X_all)[
                    :, 1
                ]  # Probability of implementation

                results = all_data[[self.unit_col, self.time_col]].copy()
                results[f"{policy}_feasibility_score"] = feasibility_probs
                results[f"{policy}_feasibility_category"] = pd.cut(
                    feasibility_probs,
                    bins=[0, 0.3, 0.7, 1.0],
                    labels=["Low", "Medium", "High"],
                )

                # Store model performance
                results.attrs["model_performance"] = {
                    "cv_mean": np.mean(cv_scores),
                    "cv_std": np.std(cv_scores),
                    "feature_importance": dict(
                        zip(features, abs(model.coef_[0]), strict=False)
                    ),
                }

                return results
            return self._synthetic_feasibility_scores(policy)

        except ImportError:
            # Fallback if sklearn not available
            return self._unsupervised_feasibility_model(data, policy, features)
        except Exception:
            # Fallback for any other issues
            return self._unsupervised_feasibility_model(data, policy, features)

    def _unsupervised_feasibility_model(
        self,
        data: pd.DataFrame,
        policy: str,
        features: list[str],
    ) -> pd.DataFrame:
        """Unsupervised feasibility scoring using composite indicators.

        Args:
            data: Analysis dataset
            policy: Policy component
            features: Feature columns

        Returns:
            DataFrame with feasibility scores
        """
        # Normalize features to [0, 1] scale
        normalized_data = data.copy()

        for feature in features:
            values = data[feature].values
            min_val, max_val = np.nanmin(values), np.nanmax(values)
            if max_val > min_val:
                normalized_data[f"{feature}_norm"] = (values - min_val) / (
                    max_val - min_val
                )
            else:
                normalized_data[f"{feature}_norm"] = 0.5  # Neutral score

        # Create composite feasibility score
        normalized_features = [f"{f}_norm" for f in features]

        # Policy-specific weights (can be refined based on literature)
        policy_weights = self._get_policy_weights(policy, features)

        # Weighted average
        feasibility_scores = np.zeros(len(normalized_data))
        for i, feature in enumerate(normalized_features):
            weight = policy_weights.get(features[i], 1.0)
            feasibility_scores += weight * normalized_data[feature].fillna(0.5).values

        feasibility_scores /= sum(policy_weights.values())  # Normalize by total weight

        # Create results DataFrame
        results = data[[self.unit_col, self.time_col]].copy()
        results[f"{policy}_feasibility_score"] = feasibility_scores
        results[f"{policy}_feasibility_category"] = pd.cut(
            feasibility_scores,
            bins=[0, 0.3, 0.7, 1.0],
            labels=["Low", "Medium", "High"],
        )

        return results

    def _get_policy_weights(self, policy: str, features: list[str]) -> dict[str, float]:
        """Get policy-specific weights for political indicators.

        Args:
            policy: MPOWER component
            features: Available features

        Returns:
            Dictionary of feature weights
        """
        # Policy-specific weight mappings based on policy literature
        weight_mappings = {
            "M": {  # Monitor - requires institutional capacity
                "governance_effectiveness": 1.5,
                "regulatory_quality": 1.3,
                "rule_of_law": 1.2,
                "democracy_score": 1.0,
                "control_corruption": 1.1,
                "voice_accountability": 0.9,
            },
            "P": {  # Protect - requires enforcement capacity
                "regulatory_quality": 1.5,
                "rule_of_law": 1.4,
                "governance_effectiveness": 1.2,
                "control_corruption": 1.1,
                "democracy_score": 1.0,
                "voice_accountability": 0.8,
            },
            "O": {  # Offer - requires service delivery capacity
                "governance_effectiveness": 1.4,
                "regulatory_quality": 1.2,
                "democracy_score": 1.1,
                "voice_accountability": 1.0,
                "rule_of_law": 1.0,
                "control_corruption": 0.9,
            },
            "W": {  # Warn - requires regulatory framework
                "regulatory_quality": 1.5,
                "rule_of_law": 1.3,
                "democracy_score": 1.1,
                "governance_effectiveness": 1.0,
                "voice_accountability": 1.0,
                "control_corruption": 0.9,
            },
            "E": {  # Enforce - requires strong enforcement
                "rule_of_law": 1.5,
                "governance_effectiveness": 1.3,
                "regulatory_quality": 1.2,
                "control_corruption": 1.2,
                "democracy_score": 1.0,
                "voice_accountability": 0.8,
            },
            "R": {  # Raise - requires fiscal capacity and political will
                "governance_effectiveness": 1.4,
                "democracy_score": 1.3,
                "regulatory_quality": 1.2,
                "voice_accountability": 1.1,
                "rule_of_law": 1.0,
                "control_corruption": 1.0,
            },
        }

        policy_weights = weight_mappings.get(policy, {})

        # Return weights only for available features
        return {feature: policy_weights.get(feature, 1.0) for feature in features}

    def _synthetic_feasibility_scores(self, policy: str) -> pd.DataFrame:
        """Generate synthetic feasibility scores when data is insufficient.

        Args:
            policy: Policy component

        Returns:
            DataFrame with synthetic scores
        """
        # Create basic results structure
        countries = self.country_data[self.unit_col].unique()
        years = (
            self.country_data[self.time_col].unique()
            if self.time_col in self.country_data.columns
            else [2020]
        )

        results = []
        np.random.seed(42)  # For reproducibility

        for country in countries:
            for year in years:
                # Generate synthetic score based on policy type
                base_score = {
                    "M": 0.6,  # Monitoring relatively easier
                    "P": 0.5,  # Protection moderate difficulty
                    "O": 0.7,  # Offering help generally feasible
                    "W": 0.8,  # Warning relatively easy
                    "E": 0.4,  # Enforcement challenging
                    "R": 0.5,  # Tax policy moderate difficulty
                }.get(policy, 0.5)

                # Add random variation
                score = np.clip(base_score + np.random.normal(0, 0.15), 0.0, 1.0)

                category = "Low" if score < 0.3 else "Medium" if score < 0.7 else "High"

                results.append(
                    {
                        self.unit_col: country,
                        self.time_col: year,
                        f"{policy}_feasibility_score": score,
                        f"{policy}_feasibility_category": category,
                    }
                )

        return pd.DataFrame(results)

    def stakeholder_model(
        self,
        stakeholder_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Game-theoretic stakeholder analysis for policy adoption.

        Models the interaction between different stakeholders (government,
        tobacco industry, health advocates, public) in policy adoption.

        Args:
            stakeholder_data: Data on stakeholder preferences and power

        Returns:
            Game-theoretic analysis results
        """
        if stakeholder_data is None:
            stakeholder_data = self._default_stakeholder_preferences()

        # Define stakeholders and their utilities
        stakeholders = list(stakeholder_data.keys())
        policies = MPOWER_COMPONENTS

        # Create payoff matrix for each policy
        policy_games = {}

        for policy in policies:
            payoff_matrix = self._create_payoff_matrix(policy, stakeholder_data)
            nash_equilibria = self._find_nash_equilibria(payoff_matrix, stakeholders)
            coalition_outcomes = self._analyze_coalitions(policy, stakeholder_data)

            policy_games[policy] = {
                "payoff_matrix": payoff_matrix,
                "nash_equilibria": nash_equilibria,
                "coalition_analysis": coalition_outcomes,
                "implementation_probability": self._calculate_implementation_probability(
                    nash_equilibria, coalition_outcomes
                ),
            }

        return {
            "stakeholder_games": policy_games,
            "overall_feasibility": self._aggregate_stakeholder_feasibility(
                policy_games
            ),
            "key_veto_players": self._identify_veto_players(policy_games),
            "coalition_recommendations": self._recommend_coalitions(policy_games),
        }

    def _default_stakeholder_preferences(self) -> dict[str, dict[str, float]]:
        """Default stakeholder preference structure.

        Returns:
            Dictionary of stakeholder preferences by policy
        """
        return {
            "government": {
                "power": 0.4,
                "M": 0.7,
                "P": 0.6,
                "O": 0.8,
                "W": 0.9,
                "E": 0.5,
                "R": 0.3,
            },
            "health_advocates": {
                "power": 0.2,
                "M": 0.9,
                "P": 0.95,
                "O": 0.85,
                "W": 0.9,
                "E": 0.95,
                "R": 0.8,
            },
            "tobacco_industry": {
                "power": 0.25,
                "M": 0.1,
                "P": 0.05,
                "O": 0.3,
                "W": 0.1,
                "E": 0.05,
                "R": 0.05,
            },
            "public": {
                "power": 0.15,
                "M": 0.6,
                "P": 0.8,
                "O": 0.7,
                "W": 0.8,
                "E": 0.6,
                "R": 0.4,
            },
        }

    def _create_payoff_matrix(
        self,
        policy: str,
        stakeholder_data: dict[str, dict[str, float]],
    ) -> dict[str, dict[str, float]]:
        """Create payoff matrix for policy game.

        Args:
            policy: MPOWER component
            stakeholder_data: Stakeholder preferences and power

        Returns:
            Payoff matrix for the policy
        """
        payoffs = {}

        for stakeholder, data in stakeholder_data.items():
            if stakeholder != "power":
                base_utility = data.get(policy, 0.5)
                power = data.get("power", 0.25)

                # Payoffs for support vs opposition
                payoffs[stakeholder] = {
                    "support": base_utility * power,
                    "oppose": (1 - base_utility) * power,
                }

        return payoffs

    def _find_nash_equilibria(
        self,
        payoff_matrix: dict[str, dict[str, float]],
        stakeholders: list[str],
    ) -> list[dict[str, str]]:
        """Find Nash equilibria for the policy game.

        Args:
            payoff_matrix: Payoffs for each stakeholder action
            stakeholders: List of stakeholder names

        Returns:
            List of Nash equilibrium strategy profiles
        """
        from itertools import product

        # All possible strategy combinations
        strategies = ["support", "oppose"]
        strategy_combinations = list(product(strategies, repeat=len(stakeholders)))

        nash_equilibria = []

        for combination in strategy_combinations:
            is_nash = True
            strategy_profile = dict(zip(stakeholders, combination, strict=False))

            # Check if any player wants to deviate
            for i, stakeholder in enumerate(stakeholders):
                current_strategy = combination[i]
                current_payoff = payoff_matrix[stakeholder][current_strategy]

                # Check alternative strategy
                alternative = "oppose" if current_strategy == "support" else "support"
                alternative_payoff = payoff_matrix[stakeholder][alternative]

                if alternative_payoff > current_payoff:
                    is_nash = False
                    break

            if is_nash:
                nash_equilibria.append(strategy_profile)

        return nash_equilibria

    def _analyze_coalitions(
        self,
        policy: str,
        stakeholder_data: dict[str, dict[str, float]],
    ) -> dict[str, Any]:
        """Analyze potential coalitions for policy support.

        Args:
            policy: MPOWER component
            stakeholder_data: Stakeholder data

        Returns:
            Coalition analysis results
        """
        from itertools import combinations

        stakeholders = [s for s in stakeholder_data if s != "power"]

        coalition_analysis = {
            "winning_coalitions": [],
            "blocking_coalitions": [],
            "coalition_values": {},
        }

        # Threshold for policy adoption (majority of weighted power)
        adoption_threshold = 0.5

        # Analyze all possible coalitions
        for r in range(1, len(stakeholders) + 1):
            for coalition in combinations(stakeholders, r):
                coalition_power = sum(
                    stakeholder_data[s].get("power", 0.25) for s in coalition
                )
                coalition_support = np.mean(
                    [stakeholder_data[s].get(policy, 0.5) for s in coalition]
                )

                coalition_value = coalition_power * coalition_support
                coalition_analysis["coalition_values"][coalition] = coalition_value

                if coalition_power >= adoption_threshold and coalition_support > 0.6:
                    coalition_analysis["winning_coalitions"].append(
                        {
                            "members": list(coalition),
                            "power": coalition_power,
                            "support": coalition_support,
                            "value": coalition_value,
                        }
                    )

                # Check if coalition can block
                opposition_power = sum(
                    stakeholder_data[s].get("power", 0.25)
                    for s in stakeholders
                    if s not in coalition
                )
                if (
                    coalition_power >= adoption_threshold
                    and opposition_power < adoption_threshold
                ):
                    coalition_analysis["blocking_coalitions"].append(
                        {
                            "members": list(coalition),
                            "power": coalition_power,
                            "opposition_power": opposition_power,
                        }
                    )

        return coalition_analysis

    def _calculate_implementation_probability(
        self,
        nash_equilibria: list[dict[str, str]],
        coalition_analysis: dict[str, Any],
    ) -> float:
        """Calculate probability of policy implementation.

        Args:
            nash_equilibria: Nash equilibrium outcomes
            coalition_analysis: Coalition analysis results

        Returns:
            Implementation probability (0-1)
        """
        # Base probability from Nash equilibria
        if not nash_equilibria:
            nash_prob = 0.1  # Low if no stable equilibrium
        else:
            support_equilibria = sum(
                1
                for eq in nash_equilibria
                if sum(1 for action in eq.values() if action == "support") > len(eq) / 2
            )
            nash_prob = support_equilibria / len(nash_equilibria)

        # Adjustment from coalition analysis
        winning_coalitions = len(coalition_analysis["winning_coalitions"])
        coalition_bonus = min(0.3, 0.1 * winning_coalitions)

        return min(1.0, nash_prob + coalition_bonus)

    def _aggregate_stakeholder_feasibility(
        self,
        policy_games: dict[str, dict[str, Any]],
    ) -> dict[str, float]:
        """Aggregate feasibility across policies.

        Args:
            policy_games: Game analysis results for each policy

        Returns:
            Aggregated feasibility metrics
        """
        implementation_probs = [
            game["implementation_probability"] for game in policy_games.values()
        ]

        return {
            "mean_implementation_prob": np.mean(implementation_probs),
            "min_implementation_prob": np.min(implementation_probs),
            "max_implementation_prob": np.max(implementation_probs),
            "implementation_variance": np.var(implementation_probs),
        }

    def _identify_veto_players(
        self,
        policy_games: dict[str, dict[str, Any]],
    ) -> list[str]:
        """Identify key veto players across policies.

        Args:
            policy_games: Game analysis results

        Returns:
            List of key veto players
        """
        veto_counts = {}

        for policy, game in policy_games.items():
            for coalition in game["coalition_analysis"]["blocking_coalitions"]:
                for member in coalition["members"]:
                    veto_counts[member] = veto_counts.get(member, 0) + 1

        # Sort by veto frequency
        veto_players = sorted(veto_counts.items(), key=lambda x: x[1], reverse=True)

        return [
            player for player, count in veto_players if count >= len(policy_games) / 2
        ]

    def _recommend_coalitions(
        self,
        policy_games: dict[str, dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        """Recommend optimal coalitions for each policy.

        Args:
            policy_games: Game analysis results

        Returns:
            Coalition recommendations by policy
        """
        recommendations = {}

        for policy, game in policy_games.items():
            winning_coalitions = game["coalition_analysis"]["winning_coalitions"]

            if winning_coalitions:
                # Rank by coalition value
                best_coalitions = sorted(
                    winning_coalitions,
                    key=lambda x: x["value"],
                    reverse=True,
                )[:3]  # Top 3 coalitions

                recommendations[policy] = best_coalitions
            else:
                recommendations[policy] = []

        return recommendations

    def plot_feasibility_heatmap(
        self,
        feasibility_results: dict[str, pd.DataFrame],
        save_path: str | None = None,
    ) -> None:
        """Plot feasibility scores as a heatmap.

        Args:
            feasibility_results: Results from feasibility_scores()
            save_path: Path to save plot (optional)
        """
        if not PLOTTING_AVAILABLE:
            warnings.warn("Plotting not available. Install matplotlib and seaborn.")
            return

        # Aggregate feasibility scores by policy and country
        feasibility_matrix = []
        countries = set()

        for policy, scores_df in feasibility_results.items():
            if isinstance(scores_df, pd.DataFrame) and not scores_df.empty:
                score_col = f"{policy}_feasibility_score"
                if score_col in scores_df.columns:
                    country_scores = scores_df.groupby(self.unit_col)[score_col].mean()
                    countries.update(country_scores.index)
                    feasibility_matrix.append(country_scores)

        if not feasibility_matrix:
            warnings.warn("No feasibility data to plot.")
            return

        # Create matrix
        countries = sorted(countries)
        policies = list(feasibility_results.keys())

        matrix = np.zeros((len(policies), len(countries)))

        for i, (policy, scores_df) in enumerate(feasibility_results.items()):
            if isinstance(scores_df, pd.DataFrame) and not scores_df.empty:
                score_col = f"{policy}_feasibility_score"
                if score_col in scores_df.columns:
                    country_scores = scores_df.groupby(self.unit_col)[score_col].mean()
                    for j, country in enumerate(countries):
                        if country in country_scores.index:
                            matrix[i, j] = country_scores[country]

        # Plot heatmap
        plt.figure(figsize=(max(8, len(countries) * 0.3), 6))

        sns.heatmap(
            matrix,
            xticklabels=countries,
            yticklabels=policies,
            annot=False,
            cmap="RdYlGn",
            cbar_kws={"label": "Feasibility Score"},
            vmin=0,
            vmax=1,
        )

        plt.title("Political Feasibility Scores by Policy and Country")
        plt.xlabel("Country")
        plt.ylabel("MPOWER Component")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()
