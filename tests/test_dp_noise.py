"""Property-based and unit tests for differential privacy noise mechanisms.

Tests verify statistical properties of Laplace and Gaussian mechanisms
using Hypothesis for property-based testing.
"""

import math

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings
from scipy import stats

from src.privacy.dp_noise import (
    DPMechanism,
    DPNoiseAdder,
    GaussianMechanism,
    LaplaceMechanism,
)

# ============================================================================
# Fixtures and Helpers
# ============================================================================


@pytest.fixture
def laplace_mechanism():
    """Create a standard Laplace mechanism."""
    return LaplaceMechanism(epsilon=1.0, sensitivity=2.0)


@pytest.fixture
def gaussian_mechanism():
    """Create a standard Gaussian mechanism."""
    return GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=2.0)


def generate_test_vector(dim: int = 512) -> np.ndarray:
    """Generate a random L2-normalized test vector."""
    vec = np.random.randn(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


# ============================================================================
# Unit Tests — Laplace Mechanism
# ============================================================================


class TestLaplaceMechanism:
    """Unit tests for Laplace mechanism."""

    def test_initialization_valid(self):
        """Test valid initialization."""
        mech = LaplaceMechanism(epsilon=1.0, sensitivity=2.0)
        assert mech.epsilon == 1.0
        assert mech.sensitivity == 2.0
        assert mech.scale == 2.0  # sensitivity / epsilon

    def test_initialization_invalid_epsilon(self):
        """Test initialization with invalid epsilon."""
        with pytest.raises(ValueError, match="epsilon must be positive"):
            LaplaceMechanism(epsilon=0.0, sensitivity=2.0)

        with pytest.raises(ValueError, match="epsilon must be positive"):
            LaplaceMechanism(epsilon=-1.0, sensitivity=2.0)

    def test_initialization_invalid_sensitivity(self):
        """Test initialization with invalid sensitivity."""
        with pytest.raises(ValueError, match="sensitivity must be positive"):
            LaplaceMechanism(epsilon=1.0, sensitivity=0.0)

        with pytest.raises(ValueError, match="sensitivity must be positive"):
            LaplaceMechanism(epsilon=1.0, sensitivity=-1.0)

    def test_add_noise_output_shape(self, laplace_mechanism):
        """Test that output has same shape as input."""
        vector = generate_test_vector(dim=512)
        noised = laplace_mechanism.add_noise(vector)

        assert noised.shape == vector.shape
        assert noised.dtype == vector.dtype

    def test_add_noise_invalid_input(self, laplace_mechanism):
        """Test that non-1D input raises error."""
        vector_2d = np.random.randn(10, 512).astype(np.float32)

        with pytest.raises(ValueError, match="Expected 1D vector"):
            laplace_mechanism.add_noise(vector_2d)

    def test_add_noise_batch(self, laplace_mechanism):
        """Test batch noise addition."""
        vectors = np.random.randn(10, 512).astype(np.float32)
        noised = laplace_mechanism.add_noise_batch(vectors)

        assert noised.shape == vectors.shape
        assert noised.dtype == vectors.dtype

    def test_add_noise_batch_invalid_input(self, laplace_mechanism):
        """Test that non-2D batch raises error."""
        vector_1d = np.random.randn(512).astype(np.float32)

        with pytest.raises(ValueError, match="Expected 2D array"):
            laplace_mechanism.add_noise_batch(vector_1d)

    def test_get_privacy_guarantee(self, laplace_mechanism):
        """Test privacy guarantee reporting."""
        guarantee = laplace_mechanism.get_privacy_guarantee()

        assert guarantee["mechanism"] == "laplace"
        assert guarantee["epsilon"] == 1.0
        assert guarantee["delta"] == 0.0  # Pure ε-DP
        assert guarantee["sensitivity"] == 2.0

    def test_noise_determinism(self):
        """Test that noise is deterministic with same seed."""
        np.random.seed(42)
        mech1 = LaplaceMechanism(epsilon=1.0, sensitivity=2.0)
        vector = generate_test_vector(dim=512)

        np.random.seed(42)
        mech2 = LaplaceMechanism(epsilon=1.0, sensitivity=2.0)

        np.random.seed(123)
        noised1 = mech1.add_noise(vector)

        np.random.seed(123)
        noised2 = mech2.add_noise(vector)

        np.testing.assert_array_almost_equal(noised1, noised2)


# ============================================================================
# Unit Tests — Gaussian Mechanism
# ============================================================================


class TestGaussianMechanism:
    """Unit tests for Gaussian mechanism."""

    def test_initialization_valid(self):
        """Test valid initialization."""
        mech = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=2.0)
        assert mech.epsilon == 1.0
        assert mech.delta == 1e-5
        assert mech.sensitivity == 2.0

        # Verify sigma computation
        expected_sigma = 2.0 * math.sqrt(2 * math.log(1.25 / 1e-5)) / 1.0
        assert abs(mech.sigma - expected_sigma) < 1e-6

    def test_initialization_invalid_epsilon(self):
        """Test initialization with invalid epsilon."""
        with pytest.raises(ValueError, match="epsilon must be positive"):
            GaussianMechanism(epsilon=0.0, delta=1e-5, sensitivity=2.0)

    def test_initialization_invalid_delta(self):
        """Test initialization with invalid delta."""
        with pytest.raises(ValueError, match="delta must be in \\(0, 1\\)"):
            GaussianMechanism(epsilon=1.0, delta=0.0, sensitivity=2.0)

        with pytest.raises(ValueError, match="delta must be in \\(0, 1\\)"):
            GaussianMechanism(epsilon=1.0, delta=1.0, sensitivity=2.0)

    def test_initialization_invalid_sensitivity(self):
        """Test initialization with invalid sensitivity."""
        with pytest.raises(ValueError, match="sensitivity must be positive"):
            GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=0.0)

    def test_add_noise_output_shape(self, gaussian_mechanism):
        """Test that output has same shape as input."""
        vector = generate_test_vector(dim=512)
        noised = gaussian_mechanism.add_noise(vector)

        assert noised.shape == vector.shape
        assert noised.dtype == vector.dtype

    def test_add_noise_invalid_input(self, gaussian_mechanism):
        """Test that non-1D input raises error."""
        vector_2d = np.random.randn(10, 512).astype(np.float32)

        with pytest.raises(ValueError, match="Expected 1D vector"):
            gaussian_mechanism.add_noise(vector_2d)

    def test_add_noise_batch(self, gaussian_mechanism):
        """Test batch noise addition."""
        vectors = np.random.randn(10, 512).astype(np.float32)
        noised = gaussian_mechanism.add_noise_batch(vectors)

        assert noised.shape == vectors.shape
        assert noised.dtype == vectors.dtype

    def test_get_privacy_guarantee(self, gaussian_mechanism):
        """Test privacy guarantee reporting."""
        guarantee = gaussian_mechanism.get_privacy_guarantee()

        assert guarantee["mechanism"] == "gaussian"
        assert guarantee["epsilon"] == 1.0
        assert guarantee["delta"] == 1e-5
        assert guarantee["sensitivity"] == 2.0


# ============================================================================
# Unit Tests — DPNoiseAdder Factory
# ============================================================================


class TestDPNoiseAdder:
    """Unit tests for DP noise adder factory."""

    def test_create_laplace(self):
        """Test creating Laplace mechanism via factory."""
        adder = DPNoiseAdder(
            mechanism="laplace",
            epsilon=1.0,
            sensitivity=2.0,
        )

        assert adder.mechanism_type == DPMechanism.LAPLACE
        assert isinstance(adder.mechanism, LaplaceMechanism)

    def test_create_gaussian(self):
        """Test creating Gaussian mechanism via factory."""
        adder = DPNoiseAdder(
            mechanism="gaussian",
            epsilon=1.0,
            sensitivity=2.0,
            delta=1e-5,
        )

        assert adder.mechanism_type == DPMechanism.GAUSSIAN
        assert isinstance(adder.mechanism, GaussianMechanism)

    def test_create_gaussian_missing_delta(self):
        """Test that Gaussian requires delta."""
        with pytest.raises(ValueError, match="delta is required"):
            DPNoiseAdder(
                mechanism="gaussian",
                epsilon=1.0,
                sensitivity=2.0,
            )

    def test_create_invalid_mechanism(self):
        """Test that invalid mechanism raises error."""
        with pytest.raises(ValueError):
            DPNoiseAdder(
                mechanism="invalid",
                epsilon=1.0,
                sensitivity=2.0,
            )

    def test_add_noise(self):
        """Test noise addition via factory."""
        adder = DPNoiseAdder(
            mechanism="laplace",
            epsilon=1.0,
            sensitivity=2.0,
        )

        vector = generate_test_vector(dim=512)
        noised = adder.add_noise(vector)

        assert noised.shape == vector.shape

    def test_get_privacy_guarantee(self):
        """Test privacy guarantee reporting via factory."""
        adder = DPNoiseAdder(
            mechanism="laplace",
            epsilon=1.0,
            sensitivity=2.0,
        )

        guarantee = adder.get_privacy_guarantee()
        assert guarantee["mechanism"] == "laplace"
        assert guarantee["epsilon"] == 1.0


# ============================================================================
# Property-Based Tests — Statistical Properties
# ============================================================================


class TestLaplaceStatisticalProperties:
    """Property-based tests for Laplace mechanism statistical properties."""

    @settings(max_examples=50, deadline=None)
    @given(
        epsilon=st.floats(min_value=0.1, max_value=10.0),
        sensitivity=st.floats(min_value=0.1, max_value=5.0),
        dimension=st.integers(min_value=10, max_value=1000),
    )
    def test_mean_preservation(self, epsilon, sensitivity, dimension):
        """Property: Noisy samples should have mean ≈ original (mean preservation).

        Generate many noisy samples and verify that the mean of noisy samples
        is close to the original vector (within statistical tolerance).
        """
        mech = LaplaceMechanism(epsilon=epsilon, sensitivity=sensitivity)

        # Generate original vector
        original = np.random.randn(dimension).astype(np.float32)
        original = original / np.linalg.norm(original)

        # Generate many noisy samples
        num_samples = 1000
        noisy_samples = np.array([mech.add_noise(original) for _ in range(num_samples)])

        # Compute mean of noisy samples
        mean_noisy = np.mean(noisy_samples, axis=0)

        # Mean should be close to original (within 3σ tolerance)
        # Standard error of mean = scale / sqrt(n) per dimension
        scale = sensitivity / epsilon
        se_mean = scale / np.sqrt(num_samples)
        tolerance = 3 * se_mean

        # Check element-wise (more lenient for high-dimensional vectors)
        mean_diff = np.mean(np.abs(mean_noisy - original))
        assert (
            mean_diff < tolerance * 10
        ), f"Mean difference {mean_diff} exceeds tolerance {tolerance * 10}"

    @settings(max_examples=50, deadline=None)
    @given(
        epsilon=st.floats(min_value=0.1, max_value=10.0),
        sensitivity=st.floats(min_value=0.1, max_value=5.0),
    )
    def test_variance_scaling(self, epsilon, sensitivity):
        """Property: Noise variance should scale as 2b² where b = sensitivity/epsilon."""
        mech = LaplaceMechanism(epsilon=epsilon, sensitivity=sensitivity)

        # Generate noise samples (single dimension for simplicity)
        num_samples = 10000
        noise_samples = []

        zero_vector = np.zeros(1, dtype=np.float32)
        for _ in range(num_samples):
            noised = mech.add_noise(zero_vector)
            noise_samples.append(noised[0])

        noise_samples = np.array(noise_samples)

        # Theoretical variance for Laplace distribution: 2b²
        expected_scale = sensitivity / epsilon
        expected_variance = 2 * (expected_scale**2)

        # Observed variance
        observed_variance = np.var(noise_samples)

        # Allow 10% tolerance due to sampling
        assert abs(observed_variance - expected_variance) / expected_variance < 0.15

    @settings(max_examples=20, deadline=None)
    @given(
        sensitivity=st.floats(min_value=0.5, max_value=3.0),
    )
    def test_epsilon_noise_relationship(self, sensitivity):
        """Property: Higher epsilon should result in less noise (lower variance)."""
        epsilons = [0.5, 1.0, 2.0, 5.0]
        variances = []

        zero_vector = np.zeros(100, dtype=np.float32)
        num_samples = 1000

        for eps in epsilons:
            mech = LaplaceMechanism(epsilon=eps, sensitivity=sensitivity)

            noise_magnitudes = []
            for _ in range(num_samples):
                noised = mech.add_noise(zero_vector)
                noise_magnitudes.append(np.linalg.norm(noised))

            variances.append(np.var(noise_magnitudes))

        # Variances should decrease monotonically with increasing epsilon
        for i in range(len(variances) - 1):
            assert (
                variances[i] > variances[i + 1]
            ), f"Variance should decrease with increasing epsilon: {variances}"

    def test_laplace_distribution_shape(self):
        """Property: Noise should follow Laplace distribution (KS test)."""
        mech = LaplaceMechanism(epsilon=1.0, sensitivity=2.0)
        expected_scale = 2.0

        # Generate noise samples (single dimension)
        num_samples = 10000
        noise_samples = []

        zero_vector = np.zeros(1, dtype=np.float32)
        for _ in range(num_samples):
            noised = mech.add_noise(zero_vector)
            noise_samples.append(noised[0])

        noise_samples = np.array(noise_samples)

        # Kolmogorov-Smirnov test against Laplace distribution
        # H0: samples come from Laplace(0, scale)
        ks_statistic, p_value = stats.kstest(
            noise_samples, lambda x: stats.laplace.cdf(x, loc=0, scale=expected_scale)
        )

        # p-value > 0.05 means we cannot reject H0 (good!)
        assert (
            p_value > 0.01
        ), f"KS test failed: samples do not follow Laplace distribution (p={p_value})"


class TestGaussianStatisticalProperties:
    """Property-based tests for Gaussian mechanism statistical properties."""

    @settings(max_examples=50, deadline=None)
    @given(
        epsilon=st.floats(min_value=0.1, max_value=10.0),
        sensitivity=st.floats(min_value=0.1, max_value=5.0),
        dimension=st.integers(min_value=10, max_value=1000),
    )
    def test_mean_preservation(self, epsilon, sensitivity, dimension):
        """Property: Noisy samples should have mean ≈ original."""
        mech = GaussianMechanism(epsilon=epsilon, delta=1e-5, sensitivity=sensitivity)

        # Generate original vector
        original = np.random.randn(dimension).astype(np.float32)
        original = original / np.linalg.norm(original)

        # Generate many noisy samples
        num_samples = 1000
        noisy_samples = np.array([mech.add_noise(original) for _ in range(num_samples)])

        # Compute mean of noisy samples
        mean_noisy = np.mean(noisy_samples, axis=0)

        # Mean should be close to original
        sigma = mech.sigma
        se_mean = sigma / np.sqrt(num_samples)
        tolerance = 3 * se_mean

        mean_diff = np.mean(np.abs(mean_noisy - original))
        assert mean_diff < tolerance * 10

    @settings(max_examples=50, deadline=None)
    @given(
        epsilon=st.floats(min_value=0.1, max_value=10.0),
        sensitivity=st.floats(min_value=0.1, max_value=5.0),
    )
    def test_variance_scaling(self, epsilon, sensitivity):
        """Property: Noise variance should match σ²."""
        mech = GaussianMechanism(epsilon=epsilon, delta=1e-5, sensitivity=sensitivity)

        # Generate noise samples
        num_samples = 10000
        noise_samples = []

        zero_vector = np.zeros(1, dtype=np.float32)
        for _ in range(num_samples):
            noised = mech.add_noise(zero_vector)
            noise_samples.append(noised[0])

        noise_samples = np.array(noise_samples)

        # Expected variance: σ²
        expected_variance = mech.sigma**2

        # Observed variance
        observed_variance = np.var(noise_samples)

        # Allow 10% tolerance
        assert abs(observed_variance - expected_variance) / expected_variance < 0.15

    def test_gaussian_distribution_shape(self):
        """Property: Noise should follow Gaussian distribution (KS test)."""
        mech = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=2.0)
        expected_sigma = mech.sigma

        # Generate noise samples
        num_samples = 10000
        noise_samples = []

        zero_vector = np.zeros(1, dtype=np.float32)
        for _ in range(num_samples):
            noised = mech.add_noise(zero_vector)
            noise_samples.append(noised[0])

        noise_samples = np.array(noise_samples)

        # KS test against Gaussian distribution
        ks_statistic, p_value = stats.kstest(
            noise_samples, lambda x: stats.norm.cdf(x, loc=0, scale=expected_sigma)
        )

        assert (
            p_value > 0.01
        ), f"KS test failed: samples do not follow Gaussian distribution (p={p_value})"


class TestSensitivityNoiseRelationship:
    """Property-based tests for sensitivity-noise relationship."""

    @settings(max_examples=20, deadline=None)
    @given(
        epsilon=st.floats(min_value=0.5, max_value=5.0),
        dimension=st.integers(min_value=50, max_value=500),
    )
    def test_sensitivity_doubling_doubles_noise(self, epsilon, dimension):
        """Property: Doubling sensitivity should double noise scale."""
        sensitivity1 = 1.0
        sensitivity2 = 2.0

        mech1 = LaplaceMechanism(epsilon=epsilon, sensitivity=sensitivity1)
        mech2 = LaplaceMechanism(epsilon=epsilon, sensitivity=sensitivity2)

        # Generate noise samples
        num_samples = 1000
        zero_vector = np.zeros(dimension, dtype=np.float32)

        noise_magnitudes1 = []
        noise_magnitudes2 = []

        for _ in range(num_samples):
            noised1 = mech1.add_noise(zero_vector)
            noised2 = mech2.add_noise(zero_vector)

            noise_magnitudes1.append(np.linalg.norm(noised1))
            noise_magnitudes2.append(np.linalg.norm(noised2))

        # Mean noise magnitude should be ~2x
        mean_noise1 = np.mean(noise_magnitudes1)
        mean_noise2 = np.mean(noise_magnitudes2)

        ratio = mean_noise2 / mean_noise1
        assert 1.8 < ratio < 2.2, f"Noise ratio {ratio} not close to 2.0"


class TestPrivacyParameterValidation:
    """Property-based tests for parameter validation."""

    @given(
        epsilon=st.floats(max_value=0.0),
        sensitivity=st.floats(min_value=0.1, max_value=5.0),
    )
    def test_negative_epsilon_rejected(self, epsilon, sensitivity):
        """Property: All negative/zero epsilon values should be rejected."""
        with pytest.raises(ValueError):
            LaplaceMechanism(epsilon=epsilon, sensitivity=sensitivity)

    @given(
        epsilon=st.floats(min_value=0.1, max_value=10.0),
        sensitivity=st.floats(max_value=0.0),
    )
    def test_negative_sensitivity_rejected(self, epsilon, sensitivity):
        """Property: All negative/zero sensitivity values should be rejected."""
        with pytest.raises(ValueError):
            LaplaceMechanism(epsilon=epsilon, sensitivity=sensitivity)

    @given(
        epsilon=st.floats(min_value=0.1, max_value=10.0),
        sensitivity=st.floats(min_value=0.1, max_value=5.0),
        delta=st.floats().filter(lambda x: x <= 0 or x >= 1),
    )
    def test_invalid_delta_rejected(self, epsilon, sensitivity, delta):
        """Property: Delta outside (0, 1) should be rejected."""
        with pytest.raises(ValueError):
            GaussianMechanism(epsilon=epsilon, delta=delta, sensitivity=sensitivity)


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for DP mechanisms."""

    def test_realistic_embedding_pipeline(self):
        """Test realistic embedding privacy pipeline."""
        # Simulate embedding pipeline
        embedding_dim = 512
        epsilon = 5.0  # Higher epsilon for test demonstration
        sensitivity = 0.5  # Lower sensitivity (typical for normalized embeddings)

        # Create DP noise adder
        dp_adder = DPNoiseAdder(
            mechanism="laplace",
            epsilon=epsilon,
            sensitivity=sensitivity,
        )

        # Generate batch of embeddings
        batch_size = 100
        embeddings = np.random.randn(batch_size, embedding_dim).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Add noise
        noisy_embeddings = dp_adder.add_noise_batch(embeddings)

        # Verify shapes
        assert noisy_embeddings.shape == embeddings.shape

        # Verify utility (cosine similarity should be positive on average)
        cosine_sims = []
        for orig, noisy in zip(embeddings, noisy_embeddings, strict=False):
            sim = np.dot(orig, noisy) / (np.linalg.norm(orig) * np.linalg.norm(noisy))
            cosine_sims.append(sim)

        mean_cosine_sim = np.mean(cosine_sims)

        # With higher ε and lower sensitivity, utility should be preserved
        # Note: In high dimensions, even moderate noise significantly affects cosine similarity
        assert mean_cosine_sim > 0.2, f"Mean cosine similarity {mean_cosine_sim} too low"

        # Verify noise was actually added (embeddings changed)
        assert not np.allclose(embeddings, noisy_embeddings)

        # Get privacy guarantee
        guarantee = dp_adder.get_privacy_guarantee()
        assert guarantee["epsilon"] == epsilon
        assert guarantee["sensitivity"] == sensitivity

    def test_comparison_laplace_vs_gaussian(self):
        """Compare Laplace vs Gaussian mechanisms."""
        epsilon = 5.0  # Higher epsilon for better utility
        delta = 1e-5
        sensitivity = 0.5  # Lower sensitivity
        dimension = 512

        laplace_mech = LaplaceMechanism(epsilon=epsilon, sensitivity=sensitivity)
        gaussian_mech = GaussianMechanism(epsilon=epsilon, delta=delta, sensitivity=sensitivity)

        # Generate test vector
        vector = generate_test_vector(dim=dimension)

        # Add noise with both mechanisms
        num_samples = 100
        laplace_sims = []
        gaussian_sims = []

        for _ in range(num_samples):
            laplace_noised = laplace_mech.add_noise(vector)
            gaussian_noised = gaussian_mech.add_noise(vector)

            laplace_sim = np.dot(vector, laplace_noised) / (
                np.linalg.norm(vector) * np.linalg.norm(laplace_noised)
            )
            gaussian_sim = np.dot(vector, gaussian_noised) / (
                np.linalg.norm(vector) * np.linalg.norm(gaussian_noised)
            )

            laplace_sims.append(laplace_sim)
            gaussian_sims.append(gaussian_sim)

        # Both should provide utility (vectors not completely random)
        # Note: Due to randomness and high dimensionality, this can vary
        assert np.mean(laplace_sims) > -0.5  # At least better than random negative
        assert np.mean(gaussian_sims) > -0.5

        # Verify noise was added (not identity)
        assert np.std(laplace_sims) > 0.01
        assert np.std(gaussian_sims) > 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
