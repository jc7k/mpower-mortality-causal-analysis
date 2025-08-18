"""Tests for spatial weight matrix construction."""

import numpy as np
import pandas as pd
import pytest

from mpower_mortality_causal_analysis.extensions.spillover.spatial_weights import (
    SpatialWeightMatrix,
)


class TestSpatialWeightMatrix:
    """Test spatial weight matrix construction."""

    @pytest.fixture
    def sample_countries(self):
        """Sample country list for testing."""
        return ["USA", "Canada", "Mexico", "Brazil", "Argentina"]

    @pytest.fixture
    def sample_geography_data(self):
        """Sample geography data for testing."""
        return pd.DataFrame(
            {
                "country1": ["USA", "USA", "Canada", "Brazil"],
                "country2": ["Canada", "Mexico", "USA", "Argentina"],
                "borders": [True, True, True, True],
                "distance_km": [1000, 2000, 1000, 1500],
            }
        )

    @pytest.fixture
    def weight_matrix(self, sample_countries):
        """Initialize weight matrix constructor."""
        return SpatialWeightMatrix(sample_countries)

    def test_initialization(self, sample_countries):
        """Test proper initialization of SpatialWeightMatrix."""
        swm = SpatialWeightMatrix(sample_countries)

        assert swm.countries == sample_countries
        assert swm.n_countries == 5
        assert len(swm.country_to_idx) == 5
        assert swm.country_to_idx["USA"] == 0
        assert swm.country_to_idx["Argentina"] == 4

    def test_contiguity_matrix_properties(self, weight_matrix):
        """Test contiguity matrix has correct properties."""
        W = weight_matrix.contiguity_matrix(row_standardize=False)

        # Check dimensions
        assert W.shape == (5, 5)

        # Check symmetry
        assert np.allclose(W, W.T)

        # Check diagonal is zero
        assert np.all(np.diag(W) == 0)

        # Check binary values
        assert np.all((W == 0) | (W == 1))

    def test_contiguity_matrix_row_standardization(self, weight_matrix):
        """Test row standardization of contiguity matrix."""
        W = weight_matrix.contiguity_matrix(row_standardize=True)

        # Check row sums
        row_sums = W.sum(axis=1)
        non_zero_rows = row_sums > 0

        if non_zero_rows.any():
            assert np.allclose(row_sums[non_zero_rows], 1.0)

    def test_distance_matrix_properties(self, weight_matrix):
        """Test distance matrix properties."""
        W = weight_matrix.distance_matrix(row_standardize=False)

        # Check dimensions
        assert W.shape == (5, 5)

        # Check symmetry
        assert np.allclose(W, W.T)

        # Check diagonal is zero
        assert np.all(np.diag(W) == 0)

        # Check non-negative values
        assert np.all(W >= 0)

    def test_distance_matrix_cutoff(self, weight_matrix):
        """Test distance matrix with cutoff."""
        W_no_cutoff = weight_matrix.distance_matrix(cutoff=None, row_standardize=False)
        W_cutoff = weight_matrix.distance_matrix(cutoff=1000, row_standardize=False)

        # Cutoff version should have fewer or equal non-zero elements
        assert (W_cutoff > 0).sum() <= (W_no_cutoff > 0).sum()

    def test_distance_matrix_inverse_vs_decay(self, weight_matrix):
        """Test inverse distance vs decay function."""
        W_inverse = weight_matrix.distance_matrix(inverse=True, row_standardize=False)
        W_decay = weight_matrix.distance_matrix(inverse=False, row_standardize=False)

        # Both should be non-negative
        assert np.all(W_inverse >= 0)
        assert np.all(W_decay >= 0)

        # Both should be symmetric
        assert np.allclose(W_inverse, W_inverse.T)
        assert np.allclose(W_decay, W_decay.T)

    def test_economic_proximity_normalization(self, weight_matrix):
        """Test economic proximity matrix normalization."""
        # Create mock trade data
        trade_data = pd.DataFrame(
            {
                "exporter": ["USA", "USA", "Canada"],
                "importer": ["Canada", "Mexico", "USA"],
                "trade_value": [1000, 500, 800],
            }
        )

        W_norm = weight_matrix.economic_proximity(
            trade_data, normalize=True, row_standardize=False
        )
        W_raw = weight_matrix.economic_proximity(
            trade_data, normalize=False, row_standardize=False
        )

        # Check dimensions
        assert W_norm.shape == (5, 5)
        assert W_raw.shape == (5, 5)

        # Normalized version should have smaller values
        assert W_norm.max() <= 1.0

    def test_k_nearest_neighbors(self, weight_matrix):
        """Test k-nearest neighbors matrix."""
        k = 3
        W = weight_matrix.k_nearest_neighbors(k=k, row_standardize=False)

        # Check dimensions
        assert W.shape == (5, 5)

        # Check that each row has at most k non-zero elements
        for i in range(5):
            assert (W[i, :] > 0).sum() <= k

        # Check binary values
        assert np.all((W == 0) | (W == 1))

    def test_hybrid_matrix_weights(self, weight_matrix):
        """Test hybrid matrix with different weight combinations."""
        # Test with equal weights
        W_equal = weight_matrix.hybrid_matrix(
            contiguity_weight=0.33,
            distance_weight=0.33,
            economic_weight=0.34,
            row_standardize=False,
        )

        # Test with only contiguity
        W_cont_only = weight_matrix.hybrid_matrix(
            contiguity_weight=1.0,
            distance_weight=0.0,
            economic_weight=0.0,
            row_standardize=False,
        )

        # Check dimensions
        assert W_equal.shape == (5, 5)
        assert W_cont_only.shape == (5, 5)

        # Check non-negative
        assert np.all(W_equal >= 0)
        assert np.all(W_cont_only >= 0)

    def test_connectivity_stats(self, weight_matrix):
        """Test connectivity statistics calculation."""
        W = weight_matrix.contiguity_matrix(row_standardize=False)
        stats = weight_matrix.get_connectivity_stats(W)

        # Check required fields
        required_fields = [
            "n_units",
            "mean_neighbors",
            "min_neighbors",
            "max_neighbors",
            "n_isolated",
            "is_symmetric",
            "is_row_standardized",
            "density",
        ]

        for field in required_fields:
            assert field in stats

        # Check value ranges
        assert stats["n_units"] == 5
        assert stats["mean_neighbors"] >= 0
        assert stats["min_neighbors"] >= 0
        assert stats["max_neighbors"] <= 4  # Max possible neighbors
        assert stats["n_isolated"] >= 0
        assert isinstance(stats["is_symmetric"], bool)
        assert isinstance(stats["is_row_standardized"], bool)
        assert 0 <= stats["density"] <= 1

    def test_connectivity_stats_row_standardized(self, weight_matrix):
        """Test connectivity stats correctly identify row standardization."""
        W_raw = weight_matrix.contiguity_matrix(row_standardize=False)
        W_std = weight_matrix.contiguity_matrix(row_standardize=True)

        stats_raw = weight_matrix.get_connectivity_stats(W_raw)
        stats_std = weight_matrix.get_connectivity_stats(W_std)

        # Raw matrix should not be row standardized
        assert not stats_raw["is_row_standardized"]

        # Standardized matrix should be row standardized (if any connections exist)
        if W_std.sum() > 0:
            assert stats_std["is_row_standardized"]

    def test_to_sparse_conversion(self, weight_matrix):
        """Test conversion to sparse matrix format."""
        W = weight_matrix.contiguity_matrix()

        # Test sparse conversion (if scipy available)
        try:
            W_sparse = weight_matrix.to_sparse(W)
            # If scipy is available, should return sparse matrix
            # If not available, should return original matrix
            assert W_sparse is not None
        except ImportError:
            # If scipy not available, should handle gracefully
            pass

    def test_matrix_caching(self, weight_matrix):
        """Test that matrices are cached properly."""
        # First call
        W1 = weight_matrix.contiguity_matrix(row_standardize=False)

        # Second call should use cache
        W2 = weight_matrix.contiguity_matrix(row_standardize=False)

        # Should be identical
        assert np.array_equal(W1, W2)

        # Cache should exist
        assert weight_matrix._contiguity_cache is not None

    def test_empty_geography_data_handling(self):
        """Test handling of missing geography data."""
        countries = ["A", "B", "C"]
        swm = SpatialWeightMatrix(countries, geography_data=None)

        # Should create mock matrices without error
        W_cont = swm.contiguity_matrix()
        W_dist = swm.distance_matrix()

        assert W_cont.shape == (3, 3)
        assert W_dist.shape == (3, 3)

    def test_invalid_trade_data_handling(self, weight_matrix):
        """Test handling of invalid trade data."""
        # Empty trade data
        empty_trade = pd.DataFrame()
        W_empty = weight_matrix.economic_proximity(empty_trade)

        assert W_empty.shape == (5, 5)

        # None trade data
        W_none = weight_matrix.economic_proximity(None)

        assert W_none.shape == (5, 5)

    def test_zero_weight_handling(self, weight_matrix):
        """Test handling of zero weights in hybrid matrix."""
        # All weights zero
        W_zeros = weight_matrix.hybrid_matrix(
            contiguity_weight=0.0, distance_weight=0.0, economic_weight=0.0
        )

        # Should default to contiguity only
        assert W_zeros.shape == (5, 5)

    def test_single_country_edge_case(self):
        """Test edge case with single country."""
        swm = SpatialWeightMatrix(["USA"])

        W = swm.contiguity_matrix()

        assert W.shape == (1, 1)
        assert W[0, 0] == 0  # No self-connection

    def test_matrix_symmetry_preservation(self, weight_matrix):
        """Test that all matrices maintain symmetry."""
        matrices = [
            weight_matrix.contiguity_matrix(row_standardize=False),
            weight_matrix.distance_matrix(row_standardize=False),
            weight_matrix.k_nearest_neighbors(row_standardize=False),
        ]

        for W in matrices:
            # Check symmetry (allowing for small numerical errors)
            assert np.allclose(W, W.T, rtol=1e-10)
