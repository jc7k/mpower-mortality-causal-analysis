"""Network diffusion analysis for MPOWER policy adoption.

This module analyzes how MPOWER policies diffuse through country networks,
identifying influencers and modeling contagion effects.
"""

import logging

import numpy as np
import pandas as pd

from scipy.stats import norm

logger = logging.getLogger(__name__)


class PolicyDiffusionNetwork:
    """Analyzes MPOWER policy diffusion through networks.

    Models how policy adoption spreads through networks of countries,
    identifying key influencers and cascade patterns.

    Args:
        adoption_data: DataFrame with columns for country, year, and adoption status
        W: Optional spatial weight matrix representing network connections
    """

    def __init__(self, adoption_data: pd.DataFrame, W: np.ndarray | None = None):
        """Initialize policy diffusion network.

        Args:
            adoption_data: DataFrame with adoption information
            W: Optional weight matrix for network structure
        """
        self.adoption_data = adoption_data.copy()
        self.W = W

        # Extract key information
        self.countries = sorted(adoption_data["country"].unique())
        self.n_countries = len(self.countries)
        self.years = sorted(adoption_data["year"].unique())
        self.n_years = len(self.years)

        # Create adoption matrix (countries x time)
        self._create_adoption_matrix()

        # Network metrics storage
        self.network_metrics = {}

    def _create_adoption_matrix(self):
        """Create binary adoption matrix from data."""
        # Pivot data to create adoption matrix
        if "adopted" in self.adoption_data.columns:
            self.adoption_matrix = (
                self.adoption_data.pivot(
                    index="country", columns="year", values="adopted"
                )
                .fillna(0)
                .values
            )
        elif "mpower_score" in self.adoption_data.columns:
            # Create binary adoption based on threshold
            threshold = 25  # Default MPOWER threshold
            pivot = self.adoption_data.pivot(
                index="country", columns="year", values="mpower_score"
            ).fillna(0)
            self.adoption_matrix = (pivot >= threshold).astype(int).values
        else:
            # Create mock adoption pattern for demonstration
            np.random.seed(42)
            self.adoption_matrix = np.zeros((self.n_countries, self.n_years))

            # Start with a few early adopters
            early_adopters = np.random.choice(self.n_countries, size=3, replace=False)
            for adopter in early_adopters:
                start_year = np.random.randint(0, 3)
                self.adoption_matrix[adopter, start_year:] = 1

            # Simulate diffusion
            for t in range(1, self.n_years):
                for i in range(self.n_countries):
                    if self.adoption_matrix[i, t - 1] == 0:  # Not yet adopted
                        # Probability increases with number of adopting neighbors
                        if self.W is not None:
                            n_adopted_neighbors = (
                                self.W[i, :] * self.adoption_matrix[:, t - 1]
                            ).sum()
                            prob = 1 / (1 + np.exp(-2 * (n_adopted_neighbors - 1)))
                        else:
                            # Random diffusion
                            n_adopted = self.adoption_matrix[:, t - 1].sum()
                            prob = n_adopted / (2 * self.n_countries)

                        if np.random.random() < prob:
                            self.adoption_matrix[i, t:] = 1

    def estimate_contagion(self, method: str = "threshold") -> dict:
        """Estimate peer influence on adoption decisions.

        Models how neighboring countries' adoption influences own adoption
        probability using various diffusion models.

        Args:
            method: Diffusion model type ("threshold", "cascade", "independent")

        Returns:
            Dictionary with contagion estimates
        """
        logger.info(f"Estimating contagion effects using {method} model")

        if method == "threshold":
            return self._threshold_model()
        if method == "cascade":
            return self._cascade_model()
        if method == "independent":
            return self._independent_cascade_model()
        raise ValueError(f"Unknown method: {method}")

    def _threshold_model(self) -> dict:
        """Linear threshold model of adoption.

        Countries adopt when fraction of neighbors adopting exceeds threshold.

        Returns:
            Dictionary with threshold estimates
        """
        thresholds = []
        adoption_times = []

        for i in range(self.n_countries):
            # Find adoption time
            adopted = np.where(self.adoption_matrix[i, :] == 1)[0]
            if len(adopted) > 0:
                t_adopt = adopted[0]
                adoption_times.append(t_adopt)

                # Calculate fraction of neighbors adopted at t-1
                if t_adopt > 0 and self.W is not None:
                    neighbors_adopted = (
                        self.W[i, :] * self.adoption_matrix[:, t_adopt - 1]
                    ).sum()
                    total_neighbors = self.W[i, :].sum()
                    if total_neighbors > 0:
                        threshold = neighbors_adopted / total_neighbors
                        thresholds.append(threshold)
                    else:
                        thresholds.append(np.nan)
                else:
                    thresholds.append(0)
            else:
                adoption_times.append(np.inf)
                thresholds.append(np.inf)

        # Estimate distribution of thresholds
        valid_thresholds = [t for t in thresholds if not np.isnan(t) and t != np.inf]

        return {
            "model": "Linear Threshold",
            "mean_threshold": np.mean(valid_thresholds) if valid_thresholds else np.nan,
            "std_threshold": np.std(valid_thresholds) if valid_thresholds else np.nan,
            "median_threshold": np.median(valid_thresholds)
            if valid_thresholds
            else np.nan,
            "thresholds": thresholds,
            "adoption_times": adoption_times,
            "n_adopted": sum(1 for t in adoption_times if t != np.inf),
            "avg_adoption_time": np.mean([t for t in adoption_times if t != np.inf]),
        }

    def _cascade_model(self) -> dict:
        """Information cascade model.

        Models adoption as cascading through network based on influence probabilities.

        Returns:
            Dictionary with cascade statistics
        """
        # Track cascade sizes and depths
        cascades = []

        # Identify initial adopters (seeds)
        seeds = []
        for i in range(self.n_countries):
            if self.adoption_matrix[i, 0] == 1:
                seeds.append(i)
            elif any(self.adoption_matrix[i, :] == 1):
                # Find first adoption time
                t_adopt = np.where(self.adoption_matrix[i, :] == 1)[0][0]
                # Check if influenced by neighbors
                if t_adopt > 0 and self.W is not None:
                    neighbors_adopted = (
                        self.W[i, :] * self.adoption_matrix[:, t_adopt - 1]
                    ).sum()
                    if neighbors_adopted == 0:
                        seeds.append(i)

        # Trace cascades from each seed
        for seed in seeds:
            cascade_size = 1
            cascade_depth = 0
            influenced = {seed}
            current_wave = {seed}

            while current_wave:
                next_wave = set()
                for node in current_wave:
                    # Find nodes influenced by current node
                    if self.W is not None:
                        for j in range(self.n_countries):
                            if j not in influenced and self.W[j, node] > 0:
                                # Check if j adopted after node
                                node_time = np.where(
                                    self.adoption_matrix[node, :] == 1
                                )[0]
                                j_time = np.where(self.adoption_matrix[j, :] == 1)[0]

                                if len(node_time) > 0 and len(j_time) > 0:
                                    if j_time[0] > node_time[0]:
                                        next_wave.add(j)
                                        influenced.add(j)
                                        cascade_size += 1

                current_wave = next_wave
                if current_wave:
                    cascade_depth += 1

            cascades.append(
                {
                    "seed": seed,
                    "size": cascade_size,
                    "depth": cascade_depth,
                    "fraction": cascade_size / self.n_countries,
                }
            )

        # Calculate cascade statistics
        sizes = [c["size"] for c in cascades]
        depths = [c["depth"] for c in cascades]

        return {
            "model": "Information Cascade",
            "n_cascades": len(cascades),
            "mean_cascade_size": np.mean(sizes) if sizes else 0,
            "max_cascade_size": max(sizes) if sizes else 0,
            "mean_cascade_depth": np.mean(depths) if depths else 0,
            "max_cascade_depth": max(depths) if depths else 0,
            "largest_cascade": max(cascades, key=lambda x: x["size"])
            if cascades
            else None,
            "cascades": cascades,
        }

    def _independent_cascade_model(self) -> dict:
        """Independent cascade model with estimated influence probabilities.

        Returns:
            Dictionary with influence probability estimates
        """
        # Estimate pairwise influence probabilities
        influence_probs = np.zeros((self.n_countries, self.n_countries))

        if self.W is not None:
            for i in range(self.n_countries):
                for j in range(self.n_countries):
                    if i != j and self.W[i, j] > 0:
                        # Count successful influences
                        successes = 0
                        attempts = 0

                        i_adopt_time = np.where(self.adoption_matrix[i, :] == 1)[0]
                        j_adopt_time = np.where(self.adoption_matrix[j, :] == 1)[0]

                        if len(i_adopt_time) > 0 and len(j_adopt_time) > 0:
                            if j_adopt_time[0] == i_adopt_time[0] + 1:
                                # j adopted right after i
                                successes = 1
                                attempts = 1
                            elif j_adopt_time[0] > i_adopt_time[0]:
                                attempts = 1

                        if attempts > 0:
                            influence_probs[i, j] = successes / attempts
                        else:
                            influence_probs[i, j] = 0.1  # Default small probability

        # Simulate cascade with estimated probabilities
        simulated_adoptions = self._simulate_cascade(influence_probs)

        return {
            "model": "Independent Cascade",
            "mean_influence_prob": influence_probs[influence_probs > 0].mean()
            if (influence_probs > 0).any()
            else 0,
            "max_influence_prob": influence_probs.max(),
            "influence_matrix": influence_probs,
            "simulated_adoption_rate": simulated_adoptions / self.n_countries,
            "actual_adoption_rate": (self.adoption_matrix[:, -1] == 1).sum()
            / self.n_countries,
        }

    def _simulate_cascade(
        self, influence_probs: np.ndarray, n_simulations: int = 100
    ) -> float:
        """Simulate cascade with given influence probabilities.

        Args:
            influence_probs: Matrix of pairwise influence probabilities
            n_simulations: Number of simulations to run

        Returns:
            Average number of adoptions
        """
        total_adoptions = 0

        for _ in range(n_simulations):
            # Start with initial adopters
            adopted = set()
            for i in range(self.n_countries):
                if self.adoption_matrix[i, 0] == 1:
                    adopted.add(i)

            # Simulate cascade
            newly_adopted = adopted.copy()

            while newly_adopted:
                next_adopted = set()
                for i in newly_adopted:
                    for j in range(self.n_countries):
                        if (
                            j not in adopted
                            and np.random.random() < influence_probs[i, j]
                        ):
                            next_adopted.add(j)

                adopted.update(next_adopted)
                newly_adopted = next_adopted

            total_adoptions += len(adopted)

        return total_adoptions / n_simulations

    def identify_influencers(self, top_k: int = 5) -> list[dict]:
        """Identify key countries in diffusion process.

        Uses various centrality measures to identify influential countries
        in the policy diffusion network.

        Args:
            top_k: Number of top influencers to return

        Returns:
            List of top influencer countries with metrics
        """
        logger.info(f"Identifying top {top_k} influencers in diffusion network")

        influencers = []

        for i, country in enumerate(self.countries):
            metrics = {}

            # Adoption timing (early adopters are influencers)
            adopt_time = np.where(self.adoption_matrix[i, :] == 1)[0]
            if len(adopt_time) > 0:
                metrics["adoption_time"] = adopt_time[0]
            else:
                metrics["adoption_time"] = np.inf

            # Network centrality (if network available)
            if self.W is not None:
                # Degree centrality
                metrics["degree"] = self.W[i, :].sum()

                # Eigenvector centrality (simplified)
                try:
                    eigenvalues, eigenvectors = np.linalg.eig(self.W)
                    largest_idx = np.argmax(np.abs(eigenvalues))
                    centrality = np.abs(eigenvectors[:, largest_idx])
                    metrics["eigenvector_centrality"] = centrality[i]
                except:
                    metrics["eigenvector_centrality"] = 0

                # Influence reach (countries adopted after this one)
                if metrics["adoption_time"] != np.inf:
                    influenced_count = 0
                    for j in range(self.n_countries):
                        if i != j and self.W[j, i] > 0:
                            j_adopt = np.where(self.adoption_matrix[j, :] == 1)[0]
                            if (
                                len(j_adopt) > 0
                                and j_adopt[0] > metrics["adoption_time"]
                            ):
                                influenced_count += 1
                    metrics["influence_reach"] = influenced_count
                else:
                    metrics["influence_reach"] = 0
            else:
                metrics["degree"] = 0
                metrics["eigenvector_centrality"] = 0
                metrics["influence_reach"] = 0

            # Combine metrics into influence score
            if metrics["adoption_time"] != np.inf:
                # Earlier adoption = higher score
                time_score = 1 / (1 + metrics["adoption_time"])
            else:
                time_score = 0

            metrics["influence_score"] = (
                0.3 * time_score
                + 0.3 * metrics["eigenvector_centrality"]
                + 0.2 * (metrics["degree"] / max(1, self.n_countries))
                + 0.2 * (metrics["influence_reach"] / max(1, self.n_countries))
            )

            influencers.append(
                {
                    "country": country,
                    "rank": 0,  # Will be set after sorting
                    **metrics,
                }
            )

        # Sort by influence score
        influencers.sort(key=lambda x: x["influence_score"], reverse=True)

        # Add ranks
        for rank, inf in enumerate(influencers):
            inf["rank"] = rank + 1

        return influencers[:top_k]

    def estimate_peer_effects(
        self, outcome_data: pd.DataFrame, covariates: list[str] | None = None
    ) -> dict:
        """Estimate peer effects on outcomes using adoption network.

        Args:
            outcome_data: DataFrame with outcome variables
            covariates: Optional list of control variables

        Returns:
            Dictionary with peer effect estimates
        """
        logger.info("Estimating peer effects on outcomes")

        # Merge adoption and outcome data
        merged = pd.merge(
            self.adoption_data, outcome_data, on=["country", "year"], how="inner"
        )

        # Create lagged peer adoption variable
        peer_adoption = np.zeros(len(merged))

        if self.W is not None:
            for idx, row in merged.iterrows():
                country_idx = self.countries.index(row["country"])
                year_idx = self.years.index(row["year"])

                if year_idx > 0:
                    # Average adoption among neighbors in previous period
                    neighbor_adoption = (
                        self.W[country_idx, :] * self.adoption_matrix[:, year_idx - 1]
                    ).sum()
                    neighbor_weight = self.W[country_idx, :].sum()

                    if neighbor_weight > 0:
                        peer_adoption[idx] = neighbor_adoption / neighbor_weight

        merged["peer_adoption"] = peer_adoption

        # Simple regression for peer effects
        from sklearn.linear_model import LinearRegression

        # Prepare variables
        X_vars = ["peer_adoption"]
        if covariates:
            X_vars.extend(covariates)

        # Check which variables exist
        available_vars = [v for v in X_vars if v in merged.columns]

        if "mortality_rate" in merged.columns:
            y = merged["mortality_rate"].values
        else:
            # Create mock outcome for demonstration
            np.random.seed(42)
            y = 100 - 5 * merged["peer_adoption"] + np.random.normal(0, 10, len(merged))

        X = merged[available_vars].values

        # Fit model
        model = LinearRegression()
        model.fit(X, y)

        # Calculate statistics
        y_pred = model.predict(X)
        residuals = y - y_pred
        sse = (residuals**2).sum()
        sst = ((y - y.mean()) ** 2).sum()
        r_squared = 1 - (sse / sst) if sst > 0 else 0

        # Standard errors (simplified)
        n = len(y)
        k = X.shape[1]
        se = np.sqrt(
            sse / (n - k - 1) * np.diag(np.linalg.inv(X.T @ X + 1e-10 * np.eye(k)))
        )

        return {
            "peer_effect": model.coef_[0] if len(model.coef_) > 0 else 0,
            "peer_effect_se": se[0] if len(se) > 0 else np.nan,
            "peer_effect_pvalue": 2 * (1 - norm.cdf(abs(model.coef_[0] / se[0])))
            if len(se) > 0 and se[0] > 0
            else np.nan,
            "coefficients": dict(zip(available_vars, model.coef_, strict=False)),
            "standard_errors": dict(zip(available_vars, se, strict=False)),
            "r_squared": r_squared,
            "n_obs": n,
        }

    def predict_future_adoption(self, n_periods: int = 5) -> np.ndarray:
        """Predict future adoption patterns based on current network.

        Args:
            n_periods: Number of future periods to predict

        Returns:
            Predicted adoption matrix for future periods
        """
        # Get current adoption state
        current_adopted = self.adoption_matrix[:, -1]
        predictions = np.zeros((self.n_countries, n_periods))
        predictions[:, 0] = current_adopted

        # Estimate adoption probability model
        contagion = self.estimate_contagion("threshold")
        mean_threshold = contagion.get("mean_threshold", 0.3)

        if self.W is not None:
            for t in range(1, n_periods):
                for i in range(self.n_countries):
                    if predictions[i, t - 1] == 0:  # Not yet adopted
                        # Calculate fraction of neighbors adopted
                        neighbors_adopted = (self.W[i, :] * predictions[:, t - 1]).sum()
                        total_neighbors = self.W[i, :].sum()

                        if total_neighbors > 0:
                            fraction_adopted = neighbors_adopted / total_neighbors
                            if fraction_adopted >= mean_threshold:
                                predictions[i, t:] = 1
        else:
            # Simple logistic growth model
            for t in range(1, n_periods):
                current_fraction = predictions[:, t - 1].mean()
                growth_rate = 0.2 * current_fraction * (1 - current_fraction)

                for i in range(self.n_countries):
                    if predictions[i, t - 1] == 0:
                        if np.random.random() < growth_rate:
                            predictions[i, t:] = 1

        return predictions
