"""Spatial weight matrix construction for spillover analysis.

This module provides methods to construct various spatial weight matrices
used in spatial econometric analysis of policy spillovers.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SpatialWeightMatrix:
    """Constructs various spatial weight matrices for spillover analysis.

    Spatial weight matrices capture the spatial relationships between units
    (countries) and are fundamental to spatial econometric modeling.

    Args:
        countries: List of country names/codes
        geography_data: DataFrame with geographic information (borders, distances)
    """

    def __init__(
        self, countries: list[str], geography_data: pd.DataFrame | None = None
    ):
        """Initialize spatial weight matrix constructor.

        Args:
            countries: List of country names/codes
            geography_data: Optional DataFrame with geographic data
        """
        self.countries = countries
        self.n_countries = len(countries)
        self.country_to_idx = {c: i for i, c in enumerate(countries)}
        self.geography = geography_data

        # Cache for computed matrices
        self._contiguity_cache: np.ndarray | None = None
        self._distance_cache: np.ndarray | None = None

    def contiguity_matrix(self, row_standardize: bool = True) -> np.ndarray:
        """Construct binary contiguity matrix for shared borders.

        Creates a binary matrix where w_ij = 1 if countries i and j share
        a border, and 0 otherwise.

        Args:
            row_standardize: Whether to row-standardize the matrix

        Returns:
            Spatial weight matrix based on contiguity
        """
        if self._contiguity_cache is not None and not row_standardize:
            return self._contiguity_cache.copy()

        W = np.zeros((self.n_countries, self.n_countries))

        if self.geography is not None and "borders" in self.geography.columns:
            # Use actual border data if available
            for idx, row in self.geography.iterrows():
                if (
                    row["country1"] in self.country_to_idx
                    and row["country2"] in self.country_to_idx
                ):
                    i = self.country_to_idx[row["country1"]]
                    j = self.country_to_idx[row["country2"]]
                    if row.get("borders", False):
                        W[i, j] = 1.0
                        W[j, i] = 1.0  # Ensure symmetry
        else:
            # Create mock contiguity for demonstration
            # In practice, this would use actual border data
            np.random.seed(42)
            for i in range(self.n_countries):
                # Each country borders 2-5 neighbors on average
                n_neighbors = min(np.random.poisson(3), self.n_countries - 1)
                neighbors = np.random.choice(
                    [j for j in range(self.n_countries) if j != i],
                    size=n_neighbors,
                    replace=False,
                )
                for j in neighbors:
                    W[i, j] = 1.0
                    W[j, i] = 1.0  # Ensure symmetry

        # Cache the raw matrix
        self._contiguity_cache = W.copy()

        if row_standardize:
            W = self._row_standardize(W)

        return W

    def distance_matrix(
        self,
        cutoff: float | None = None,
        inverse: bool = True,
        row_standardize: bool = True,
    ) -> np.ndarray:
        """Construct distance-based spatial weight matrix.

        Creates weights based on geographic distance between countries,
        typically using inverse distance or a distance decay function.

        Args:
            cutoff: Maximum distance for non-zero weights (in km)
            inverse: Use inverse distance weights (1/d_ij)
            row_standardize: Whether to row-standardize the matrix

        Returns:
            Distance-based spatial weight matrix
        """
        if self._distance_cache is not None and not cutoff and not row_standardize:
            return self._distance_cache.copy()

        W = np.zeros((self.n_countries, self.n_countries))

        if self.geography is not None and "distance_km" in self.geography.columns:
            # Use actual distance data if available
            for idx, row in self.geography.iterrows():
                if (
                    row["country1"] in self.country_to_idx
                    and row["country2"] in self.country_to_idx
                ):
                    i = self.country_to_idx[row["country1"]]
                    j = self.country_to_idx[row["country2"]]
                    dist = row["distance_km"]

                    if cutoff is None or dist <= cutoff:
                        if inverse and dist > 0:
                            W[i, j] = 1.0 / dist
                            W[j, i] = 1.0 / dist
                        elif not inverse and dist > 0:
                            # Distance decay function
                            W[i, j] = np.exp(-dist / 1000)  # Scale by 1000km
                            W[j, i] = W[i, j]
        else:
            # Create mock distance matrix for demonstration
            np.random.seed(42)
            for i in range(self.n_countries):
                for j in range(i + 1, self.n_countries):
                    # Random distances between 100 and 10000 km
                    dist = np.random.uniform(100, 10000)

                    if cutoff is None or dist <= cutoff:
                        if inverse:
                            W[i, j] = 1.0 / dist
                            W[j, i] = 1.0 / dist
                        else:
                            W[i, j] = np.exp(-dist / 1000)
                            W[j, i] = W[i, j]

        # Cache the raw matrix
        if not cutoff:
            self._distance_cache = W.copy()

        if row_standardize:
            W = self._row_standardize(W)

        return W

    def economic_proximity(
        self,
        trade_data: pd.DataFrame,
        normalize: bool = True,
        row_standardize: bool = True,
    ) -> np.ndarray:
        """Construct economic proximity matrix based on trade flows.

        Creates weights based on bilateral trade intensity between countries,
        capturing economic interdependence.

        Args:
            trade_data: DataFrame with columns [exporter, importer, trade_value]
            normalize: Normalize trade values by total trade
            row_standardize: Whether to row-standardize the matrix

        Returns:
            Trade-weighted proximity matrix
        """
        W = np.zeros((self.n_countries, self.n_countries))

        if trade_data is not None and not trade_data.empty:
            # Aggregate trade flows
            for idx, row in trade_data.iterrows():
                if (
                    row["exporter"] in self.country_to_idx
                    and row["importer"] in self.country_to_idx
                ):
                    i = self.country_to_idx[row["exporter"]]
                    j = self.country_to_idx[row["importer"]]
                    W[i, j] += row["trade_value"]

            if normalize:
                # Normalize by total trade for each country
                row_sums = W.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1  # Avoid division by zero
                W = W / row_sums
        else:
            # Create mock trade matrix for demonstration
            np.random.seed(42)
            for i in range(self.n_countries):
                for j in range(self.n_countries):
                    if i != j:
                        # Random trade values with some structure
                        base_trade = np.random.exponential(1000)
                        W[i, j] = base_trade * np.random.uniform(0.1, 2.0)

            if normalize:
                row_sums = W.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1
                W = W / row_sums

        if row_standardize and not normalize:
            W = self._row_standardize(W)

        return W

    def k_nearest_neighbors(
        self, k: int = 5, distance_based: bool = True, row_standardize: bool = True
    ) -> np.ndarray:
        """Construct k-nearest neighbors spatial weight matrix.

        Each country is connected to its k nearest neighbors based on
        geographic distance or other proximity measures.

        Args:
            k: Number of nearest neighbors
            distance_based: Use geographic distance (vs random for demo)
            row_standardize: Whether to row-standardize the matrix

        Returns:
            k-nearest neighbors weight matrix
        """
        W = np.zeros((self.n_countries, self.n_countries))

        if distance_based:
            # Get distance matrix
            dist_matrix = self.distance_matrix(inverse=False, row_standardize=False)

            for i in range(self.n_countries):
                # Get distances to all other countries
                distances = dist_matrix[i, :].copy()
                distances[i] = np.inf  # Exclude self

                # Find k nearest neighbors
                k_actual = min(k, self.n_countries - 1)
                nearest_idx = np.argpartition(distances, k_actual)[:k_actual]

                # Set weights for nearest neighbors
                for j in nearest_idx:
                    W[i, j] = 1.0
        else:
            # Random k-nearest for demonstration
            np.random.seed(42)
            for i in range(self.n_countries):
                k_actual = min(k, self.n_countries - 1)
                neighbors = np.random.choice(
                    [j for j in range(self.n_countries) if j != i],
                    size=k_actual,
                    replace=False,
                )
                for j in neighbors:
                    W[i, j] = 1.0

        if row_standardize:
            W = self._row_standardize(W)

        return W

    def hybrid_matrix(
        self,
        contiguity_weight: float = 0.5,
        distance_weight: float = 0.3,
        economic_weight: float = 0.2,
        trade_data: pd.DataFrame | None = None,
        row_standardize: bool = True,
    ) -> np.ndarray:
        """Construct hybrid spatial weight matrix combining multiple criteria.

        Combines contiguity, distance, and economic proximity into a single
        weight matrix with specified weights.

        Args:
            contiguity_weight: Weight for contiguity component
            distance_weight: Weight for distance component
            economic_weight: Weight for economic proximity component
            trade_data: Trade data for economic proximity
            row_standardize: Whether to row-standardize the final matrix

        Returns:
            Hybrid spatial weight matrix
        """
        # Normalize weights to sum to 1
        total_weight = contiguity_weight + distance_weight + economic_weight
        if total_weight > 0:
            contiguity_weight /= total_weight
            distance_weight /= total_weight
            economic_weight /= total_weight
        else:
            contiguity_weight = 1.0
            distance_weight = 0.0
            economic_weight = 0.0

        # Get component matrices (not row-standardized yet)
        W = np.zeros((self.n_countries, self.n_countries))

        if contiguity_weight > 0:
            W_cont = self.contiguity_matrix(row_standardize=False)
            W += contiguity_weight * W_cont

        if distance_weight > 0:
            W_dist = self.distance_matrix(row_standardize=False)
            # Normalize distance matrix to [0, 1]
            if W_dist.max() > 0:
                W_dist = W_dist / W_dist.max()
            W += distance_weight * W_dist

        if economic_weight > 0 and trade_data is not None:
            W_econ = self.economic_proximity(
                trade_data, normalize=True, row_standardize=False
            )
            W += economic_weight * W_econ

        if row_standardize:
            W = self._row_standardize(W)

        return W

    def _row_standardize(self, W: np.ndarray) -> np.ndarray:
        """Row-standardize a weight matrix.

        Each row sums to 1 (or 0 if no neighbors).

        Args:
            W: Input weight matrix

        Returns:
            Row-standardized weight matrix
        """
        row_sums = W.sum(axis=1, keepdims=True)
        # Avoid division by zero for isolated units
        row_sums[row_sums == 0] = 1
        return W / row_sums

    def get_connectivity_stats(self, W: np.ndarray) -> dict:
        """Compute connectivity statistics for a weight matrix.

        Args:
            W: Spatial weight matrix

        Returns:
            Dictionary with connectivity statistics
        """
        # Number of neighbors for each unit
        n_neighbors = (W > 0).sum(axis=1)

        # Check for symmetry
        is_symmetric = np.allclose(W, W.T)

        # Check if row-standardized
        row_sums = W.sum(axis=1)
        is_row_standardized = np.allclose(row_sums[row_sums > 0], 1.0)

        return {
            "n_units": self.n_countries,
            "mean_neighbors": n_neighbors.mean(),
            "min_neighbors": n_neighbors.min(),
            "max_neighbors": n_neighbors.max(),
            "n_isolated": (n_neighbors == 0).sum(),
            "is_symmetric": is_symmetric,
            "is_row_standardized": is_row_standardized,
            "density": (W > 0).sum() / (self.n_countries**2 - self.n_countries),
        }

    from typing import Any

    def to_sparse(self, W: np.ndarray) -> Any:
        """Convert weight matrix to sparse format for efficiency.

        Args:
            W: Dense weight matrix

        Returns:
            Sparse CSR matrix
        """
        try:
            from scipy import sparse

            return sparse.csr_matrix(W)
        except ImportError:
            logger.warning("scipy not available, returning dense matrix")
            return W
