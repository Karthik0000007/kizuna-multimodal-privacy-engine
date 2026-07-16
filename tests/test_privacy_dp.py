"""Unit and property-based tests for differential privacy mechanisms."""

import numpy as np
import pytest
from scipy import stats

from src.privacy import DPMechanism, DPNoiseAdder, GaussianMechanism, LaplaceMechanism


class TestLaplaceMechanism:
    """Test suite for Laplace mechanism."""

    def test_init_success(self):
        """Test successful initialization."""
        mech = LaplaceMechanism(epsilon=1.0, sensitivity=2.0)

        assert mech.epsilon == 1.0
        assert mech.sensitivity == 2.0
        assert mech.scale == 2.0  # sensitivity / epsilon

    def test_init_invalid_epsilon(self):
        """Test initialization fails with invalid epsilon."""
        with pytest.raises(ValueError, match="epsilon must be positive"):
            LaplaceMechanism(epsilon=0.0, sensitivity=2.0)

        with pytest.raises(ValueError, match="epsilon must be positive"):
            LaplaceMechanism(epsilon=-1.0, sensitivity=2.0)

    def test_init_invalid_sensitivity(self):
        """Test initialization fails with invalid sensitivity."""
        with pytest.raises(ValueError, match="sensitivity must be positive"):
            LaplaceMechanism(epsilon=1.0, sensitivity=0.0)

    def test_add_noise_output_shape(self):
        """Test noise addition preserves shape."""
        mech = LaplaceMechanism(epsilon=1.0, sensitivity=2.0)
        vector = np.random.randn(512).astype(np.float32)

        noised = mech.add_noise(vector)

        assert noised.shape == vector.shape
        assert noised.dtype == vector.dtype

    def test_add_noise_changes_vector(self):
        """Test noise actually changes the vector."""
        mech = LaplaceMechanism(epsilon=1.0, sensitivity=2.0)
        vector = np.random.randn(512).astype(np.float32)

        noised = mech.add_noise(vector)

        # Vectors should be different
        assert not np.allclose(vector, noised)

    def test_add_noise_invalid_shape(self):
        """Test noise addition fails with invalid shape."""
        mech = LaplaceMechanism(epsilon=1.0, sensitivity=2.0)
        vector = np.random.randn(10, 512).astype(np.float32)  # 2D

        with pytest.raises(ValueError, match="Expected 1D vector"):
            mech.add_noise(vector)

    def test_add_noise_batch_output_shape(self):
        """Test batch noise addition preserves shape."""
        mech = LaplaceMechanism(epsilon=1.0, sensitivity=2.0)
        vectors = np.random.randn(10, 512).astype(np.float32)

        noised = mech.add_noise_batch(vectors)

        assert noised.shape == vectors.shape
        assert noised.dtype == vectors.dtype

    def test_add_noise_batch_invalid_shape(self):
        """Test batch noise addition fails with invalid shape."""
        mech = LaplaceMechanism(epsilon=1.0, sensitivity=2.0)
        vector = np.random.randn(512).astype(np.float32)  # 1D

        with pytest.raises(ValueError, match="Expected 2D array"):
            mech.add_noise_batch(vector)

    def test_get_privacy_guarantee(self):
        """Test privacy guarantee returns correct parameters."""
        mech = LaplaceMechanism(epsilon=1.0, sensitivity=2.0)

        guarantee = mech.get_privacy_guarantee()

        assert guarantee["mechanism"] == "laplace"
        assert guarantee["epsilon"] == 1.0
        assert guarantee["delta"] == 0.0  # Pure ε-DP
        assert guarantee["sensitivity"] == 2.0

    @pytest.mark.parametrize("epsilon", [0.1, 0.5, 1.0, 2.0, 5.0])
    def test_noise_scale_inversely_proportional_to_epsilon(self, epsilon):
        """Test noise scale is inversely proportional to epsilon."""
        sensitivity = 2.0
        mech = LaplaceMechanism(epsilon=epsilon, sensitivity=sensitivity)

        expected_scale = sensitivity / epsilon
        assert np.isclose(mech.scale, expected_scale)


class TestGaussianMechanism:
    """Test suite for Gaussian mechanism."""

    def test_init_success(self):
        """Test successful initialization."""
        mech = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=2.0)

        assert mech.epsilon == 1.0
        assert mech.delta == 1e-5
        assert mech.sensitivity == 2.0
        assert mech.sigma > 0

    def test_init_invalid_epsilon(self):
        """Test initialization fails with invalid epsilon."""
        with pytest.raises(ValueError, match="epsilon must be positive"):
            GaussianMechanism(epsilon=0.0, delta=1e-5, sensitivity=2.0)

    def test_init_invalid_delta(self):
        """Test initialization fails with invalid delta."""
        with pytest.raises(ValueError, match="delta must be in"):
            GaussianMechanism(epsilon=1.0, delta=0.0, sensitivity=2.0)

        with pytest.raises(ValueError, match="delta must be in"):
            GaussianMechanism(epsilon=1.0, delta=1.0, sensitivity=2.0)

    def test_init_invalid_sensitivity(self):
        """Test initialization fails with invalid sensitivity."""
        with pytest.raises(ValueError, match="sensitivity must be positive"):
            GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=0.0)

    def test_add_noise_output_shape(self):
        """Test noise addition preserves shape."""
        mech = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=2.0)
        vector = np.random.randn(512).astype(np.float32)

        noised = mech.add_noise(vector)

        assert noised.shape == vector.shape
        assert noised.dtype == vector.dtype

    def test_add_noise_changes_vector(self):
        """Test noise actually changes the vector."""
        mech = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=2.0)
        vector = np.random.randn(512).astype(np.float32)

        noised = mech.add_noise(vector)

        assert not np.allclose(vector, noised)

    def test_get_privacy_guarantee(self):
        """Test privacy guarantee returns correct parameters."""
        mech = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=2.0)

        guarantee = mech.get_privacy_guarantee()

        assert guarantee["mechanism"] == "gaussian"
        assert guarantee["epsilon"] == 1.0
        assert guarantee["delta"] == 1e-5
        assert guarantee["sensitivity"] == 2.0


class TestDPNoiseAdder:
    """Test suite for DPNoiseAdder factory."""

    def test_init_laplace(self):
        """Test initialization with Laplace mechanism."""
        adder = DPNoiseAdder(
            mechanism="laplace",
            epsilon=1.0,
            sensitivity=2.0,
        )

        assert adder.mechanism_type == DPMechanism.LAPLACE
        assert isinstance(adder.mechanism, LaplaceMechanism)

    def test_init_gaussian(self):
        """Test initialization with Gaussian mechanism."""
        adder = DPNoiseAdder(
            mechanism="gaussian",
            epsilon=1.0,
            sensitivity=2.0,
            delta=1e-5,
        )

        assert adder.mechanism_type == DPMechanism.GAUSSIAN
        assert isinstance(adder.mechanism, GaussianMechanism)

    def test_init_gaussian_missing_delta(self):
        """Test Gaussian initialization fails without delta."""
        with pytest.raises(ValueError, match="delta is required"):
            DPNoiseAdder(
                mechanism="gaussian",
                epsilon=1.0,
                sensitivity=2.0,
            )

    def test_init_invalid_mechanism(self):
        """Test initialization fails with invalid mechanism."""
        with pytest.raises(ValueError):
            DPNoiseAdder(
                mechanism="invalid",
                epsilon=1.0,
                sensitivity=2.0,
            )

    def test_add_noise(self):
        """Test noise addition works."""
        adder = DPNoiseAdder(
            mechanism="laplace",
            epsilon=1.0,
            sensitivity=2.0,
        )

        vector = np.random.randn(512).astype(np.float32)
        noised = adder.add_noise(vector)

        assert noised.shape == vector.shape
        assert not np.allclose(vector, noised)

    def test_add_noise_batch(self):
        """Test batch noise addition works."""
        adder = DPNoiseAdder(
            mechanism="laplace",
            epsilon=1.0,
            sensitivity=2.0,
        )

        vectors = np.random.randn(10, 512).astype(np.float32)
        noised = adder.add_noise_batch(vectors)

        assert noised.shape == vectors.shape
        assert not np.allclose(vectors, noised)

    def test_get_privacy_guarantee(self):
        """Test privacy guarantee retrieval."""
        adder = DPNoiseAdder(
            mechanism="laplace",
            epsilon=1.0,
            sensitivity=2.0,
        )

        guarantee = adder.get_privacy_guarantee()

        assert "mechanism" in guarantee
        assert "epsilon" in guarantee
        assert "delta" in guarantee
        assert "sensitivity" in guarantee


# Property-based tests
class TestDPNoiseProperties:
    """Property-based tests for DP noise mechanisms.

    These tests verify statistical properties of the noise distributions.
    """

    def test_property_laplace_noise_mean_preservation(self):
        """Property 1: Laplace noise preserves mean approximately.

        With many samples, the noisy mean should be close to the original mean.
        """
        mech = LaplaceMechanism(epsilon=1.0, sensitivity=2.0)
        vector = np.random.randn(512).astype(np.float32)

        num_samples = 10000
        noised_samples = [mech.add_noise(vector) for _ in range(num_samples)]
        noised_mean = np.mean(noised_samples, axis=0)

        # Mean should be preserved (within statistical tolerance)
        mean_error = np.linalg.norm(noised_mean - vector)
        expected_std = mech.scale / np.sqrt(num_samples)

        # 3-sigma test (99.7% confidence)
        assert mean_error < 3 * expected_std * np.sqrt(len(vector))

    def test_property_laplace_noise_variance(self):
        """Property 1: Laplace noise has correct variance.

        Variance of Laplace(0, b) is 2b².
        """
        epsilon = 1.0
        sensitivity = 2.0
        mech = LaplaceMechanism(epsilon=epsilon, sensitivity=sensitivity)

        vector = np.zeros(512, dtype=np.float32)  # Zero vector for simplicity

        num_samples = 10000
        noised_samples = np.array([mech.add_noise(vector) for _ in range(num_samples)])

        # Empirical variance along each dimension
        empirical_var = np.var(noised_samples, axis=0)

        # Expected variance: 2 * scale²
        expected_var = 2 * mech.scale**2

        # Mean empirical variance should be close to expected
        mean_empirical_var = np.mean(empirical_var)

        # Allow 10% tolerance
        assert abs(mean_empirical_var - expected_var) / expected_var < 0.1

    def test_property_gaussian_noise_mean_preservation(self):
        """Property 1: Gaussian noise preserves mean approximately."""
        mech = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=2.0)
        vector = np.random.randn(512).astype(np.float32)

        num_samples = 10000
        noised_samples = [mech.add_noise(vector) for _ in range(num_samples)]
        noised_mean = np.mean(noised_samples, axis=0)

        mean_error = np.linalg.norm(noised_mean - vector)
        expected_std = mech.sigma / np.sqrt(num_samples)

        # 3-sigma test
        assert mean_error < 3 * expected_std * np.sqrt(len(vector))

    def test_property_gaussian_noise_variance(self):
        """Property 1: Gaussian noise has correct variance."""
        epsilon = 1.0
        delta = 1e-5
        sensitivity = 2.0
        mech = GaussianMechanism(epsilon=epsilon, delta=delta, sensitivity=sensitivity)

        vector = np.zeros(512, dtype=np.float32)

        num_samples = 10000
        noised_samples = np.array([mech.add_noise(vector) for _ in range(num_samples)])

        empirical_var = np.var(noised_samples, axis=0)
        expected_var = mech.sigma**2

        mean_empirical_var = np.mean(empirical_var)

        # Allow 10% tolerance
        assert abs(mean_empirical_var - expected_var) / expected_var < 0.1

    def test_property_higher_epsilon_less_noise(self):
        """Property 1: Higher epsilon results in less noise."""
        sensitivity = 2.0
        vector = np.random.randn(512).astype(np.float32)

        # Test with different epsilon values
        epsilons = [0.1, 0.5, 1.0, 2.0, 5.0]
        noise_magnitudes = []

        for epsilon in epsilons:
            mech = LaplaceMechanism(epsilon=epsilon, sensitivity=sensitivity)

            # Generate multiple samples and measure average noise
            samples = [mech.add_noise(vector) for _ in range(100)]
            avg_noise = np.mean([np.linalg.norm(s - vector) for s in samples])
            noise_magnitudes.append(avg_noise)

        # Noise magnitudes should decrease as epsilon increases
        for i in range(len(noise_magnitudes) - 1):
            assert noise_magnitudes[i] > noise_magnitudes[i + 1]

    def test_property_laplace_distribution_shape(self):
        """Property 1: Noise follows Laplace distribution shape.

        Use Kolmogorov-Smirnov test to verify distribution.
        """
        epsilon = 1.0
        sensitivity = 2.0
        mech = LaplaceMechanism(epsilon=epsilon, sensitivity=sensitivity)

        vector = np.zeros(512, dtype=np.float32)

        # Generate noise samples for one dimension
        num_samples = 10000
        noise_samples = []
        for _ in range(num_samples):
            noised = mech.add_noise(vector)
            noise_samples.append(noised[0])  # Just first dimension

        # K-S test against Laplace(0, scale)
        ks_statistic, p_value = stats.kstest(
            noise_samples,
            lambda x: stats.laplace.cdf(x, loc=0, scale=mech.scale),
        )

        # p-value should be reasonably high (not rejecting null hypothesis)
        assert p_value > 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
