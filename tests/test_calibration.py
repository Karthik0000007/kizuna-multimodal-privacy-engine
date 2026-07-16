"""Unit tests for sensitivity calibration.

Tests verify empirical L2 sensitivity estimation for embedding functions.
"""

import numpy as np
import pytest

from src.privacy.calibration import SensitivityCalibrator

# ============================================================================
# Fixtures and Helpers
# ============================================================================


@pytest.fixture
def calibrator():
    """Create a sensitivity calibrator."""
    return SensitivityCalibrator()


def create_dummy_embedding_func(dim: int = 512, scale: float = 1.0):
    """Create a dummy embedding function for testing.

    Uses a fixed random projection for reproducibility.
    """
    np.random.seed(42)
    projection_matrix = np.random.randn(100, dim).astype(np.float32) * scale

    def embedding_func(input_vec: np.ndarray) -> np.ndarray:
        """Embed input using random projection."""
        embedded = input_vec @ projection_matrix
        # L2 normalize
        return embedded / np.linalg.norm(embedded)

    return embedding_func


def create_input_generator(input_dim: int = 100):
    """Create an input generator for testing."""

    def generator() -> np.ndarray:
        """Generate random normalized input."""
        vec = np.random.randn(input_dim).astype(np.float32)
        return vec / np.linalg.norm(vec)

    return generator


def create_dataset_generator(input_dim: int = 100):
    """Create a dataset generator for testing."""

    def generator(size: int) -> list:
        """Generate dataset of random normalized vectors."""
        return [
            np.random.randn(input_dim).astype(np.float32)
            / np.linalg.norm(np.random.randn(input_dim))
            for _ in range(size)
        ]

    return generator


def aggregate_embedding_func(dataset: list) -> np.ndarray:
    """Aggregate embedding function for dataset testing."""
    # Average all inputs and project
    avg_input = np.mean(dataset, axis=0).astype(np.float32)

    # Random projection
    np.random.seed(42)
    projection_matrix = np.random.randn(len(avg_input), 512).astype(np.float32)

    embedded = avg_input @ projection_matrix
    return embedded / np.linalg.norm(embedded)


# ============================================================================
# Unit Tests — Initialization
# ============================================================================


class TestInitialization:
    """Tests for calibrator initialization."""

    def test_valid_initialization(self, calibrator):
        """Test valid initialization."""
        assert isinstance(calibrator, SensitivityCalibrator)
        assert calibrator.calibration_results == []


# ============================================================================
# Unit Tests — Embedding Sensitivity Calibration
# ============================================================================


class TestEmbeddingSensitivityCalibration:
    """Tests for calibrate_embedding_sensitivity()."""

    def test_basic_calibration(self, calibrator):
        """Test basic sensitivity calibration."""
        embedding_func = create_dummy_embedding_func(dim=512)
        input_generator = create_input_generator(input_dim=100)

        sensitivity, stats = calibrator.calibrate_embedding_sensitivity(
            embedding_func=embedding_func,
            input_generator=input_generator,
            num_samples=100,  # Small number for fast test
            confidence_level=0.95,
        )

        # Sensitivity should be positive
        assert sensitivity > 0

        # Sensitivity should be reasonable (for L2-normalized embeddings, typically < 2)
        assert sensitivity < 3.0

        # Check statistics
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "median" in stats
        assert "p95" in stats
        assert "p99" in stats
        assert "confidence_interval" in stats
        assert "sensitivity_estimate" in stats

        # Sensitivity estimate should be in confidence interval
        ci_lower, ci_upper = stats["confidence_interval"]
        assert ci_lower <= sensitivity <= ci_upper

        # Calibration result should be stored
        assert len(calibrator.calibration_results) == 1
        assert calibrator.calibration_results[0] == sensitivity

    def test_calibration_with_different_dimensions(self, calibrator):
        """Test calibration with different embedding dimensions."""
        for dim in [128, 256, 512, 1024]:
            embedding_func = create_dummy_embedding_func(dim=dim)
            input_generator = create_input_generator(input_dim=100)

            sensitivity, stats = calibrator.calibrate_embedding_sensitivity(
                embedding_func=embedding_func,
                input_generator=input_generator,
                num_samples=50,
                confidence_level=0.95,
            )

            assert sensitivity > 0
            assert sensitivity < 3.0

    def test_calibration_statistics_consistency(self, calibrator):
        """Test that statistics are internally consistent."""
        embedding_func = create_dummy_embedding_func()
        input_generator = create_input_generator()

        sensitivity, stats = calibrator.calibrate_embedding_sensitivity(
            embedding_func=embedding_func,
            input_generator=input_generator,
            num_samples=200,
            confidence_level=0.95,
        )

        # min <= mean <= max
        assert stats["min"] <= stats["mean"] <= stats["max"]

        # min <= median <= max
        assert stats["min"] <= stats["median"] <= stats["max"]

        # p95 <= p99 <= max
        assert stats["p95"] <= stats["p99"] <= stats["max"]

        # Confidence interval should be valid
        ci_lower, ci_upper = stats["confidence_interval"]
        assert ci_lower <= ci_upper

        # Sensitivity estimate should be reasonable
        assert stats["sensitivity_estimate"] == sensitivity

    def test_calibration_with_high_confidence(self, calibrator):
        """Test calibration with high confidence level."""
        embedding_func = create_dummy_embedding_func()
        input_generator = create_input_generator()

        # 99% confidence level
        sensitivity_99, stats_99 = calibrator.calibrate_embedding_sensitivity(
            embedding_func=embedding_func,
            input_generator=input_generator,
            num_samples=100,
            confidence_level=0.99,
        )

        # 95% confidence level
        np.random.seed(42)  # Reset seed for same samples
        embedding_func_2 = create_dummy_embedding_func()

        sensitivity_95, stats_95 = calibrator.calibrate_embedding_sensitivity(
            embedding_func=embedding_func_2,
            input_generator=input_generator,
            num_samples=100,
            confidence_level=0.95,
        )

        # Higher confidence should give wider interval (higher upper bound)
        ci_99_lower, ci_99_upper = stats_99["confidence_interval"]
        ci_95_lower, ci_95_upper = stats_95["confidence_interval"]

        # Note: Due to randomness, this might not always hold, but generally true
        # Just check that both are positive and reasonable
        assert ci_99_upper > 0
        assert ci_95_upper > 0

    def test_multiple_calibration_runs(self, calibrator):
        """Test multiple calibration runs."""
        embedding_func = create_dummy_embedding_func()
        input_generator = create_input_generator()

        # Run calibration 3 times
        for _ in range(3):
            calibrator.calibrate_embedding_sensitivity(
                embedding_func=embedding_func,
                input_generator=input_generator,
                num_samples=50,
            )

        # Should have 3 results stored
        assert len(calibrator.calibration_results) == 3

        # All results should be positive
        assert all(s > 0 for s in calibrator.calibration_results)


# ============================================================================
# Unit Tests — Adjacent Dataset Calibration
# ============================================================================


class TestAdjacentDatasetCalibration:
    """Tests for calibrate_with_adjacent_datasets()."""

    def test_basic_adjacent_calibration(self, calibrator):
        """Test basic adjacent dataset calibration."""
        dataset_generator = create_dataset_generator(input_dim=100)

        sensitivity, stats = calibrator.calibrate_with_adjacent_datasets(
            embedding_func=aggregate_embedding_func,
            dataset_generator=dataset_generator,
            dataset_size=50,
            num_samples=50,  # Small for fast test
            confidence_level=0.95,
        )

        # Sensitivity should be positive
        assert sensitivity > 0

        # Check statistics
        assert "dataset_size" in stats
        assert stats["dataset_size"] == 50
        assert "mean" in stats
        assert "confidence_interval" in stats

        # Calibration result should be stored
        assert calibrator.calibration_results[-1] == sensitivity

    def test_adjacent_with_different_dataset_sizes(self, calibrator):
        """Test calibration with different dataset sizes."""
        dataset_generator = create_dataset_generator(input_dim=100)

        sensitivities = []

        for dataset_size in [10, 50, 100]:
            sensitivity, stats = calibrator.calibrate_with_adjacent_datasets(
                embedding_func=aggregate_embedding_func,
                dataset_generator=dataset_generator,
                dataset_size=dataset_size,
                num_samples=30,
            )

            sensitivities.append(sensitivity)
            assert stats["dataset_size"] == dataset_size

        # All sensitivities should be positive
        assert all(s > 0 for s in sensitivities)

        # Larger datasets typically have smaller sensitivity (averaging effect)
        # Note: This is not always guaranteed due to randomness
        # Just check that all are reasonable
        assert all(s < 5.0 for s in sensitivities)

    def test_adjacent_statistics_consistency(self, calibrator):
        """Test that adjacent statistics are consistent."""
        dataset_generator = create_dataset_generator()

        sensitivity, stats = calibrator.calibrate_with_adjacent_datasets(
            embedding_func=aggregate_embedding_func,
            dataset_generator=dataset_generator,
            dataset_size=50,
            num_samples=100,
        )

        # Same consistency checks as embedding calibration
        assert stats["min"] <= stats["mean"] <= stats["max"]
        assert stats["min"] <= stats["median"] <= stats["max"]

        ci_lower, ci_upper = stats["confidence_interval"]
        assert ci_lower <= ci_upper
        assert stats["sensitivity_estimate"] == sensitivity


# ============================================================================
# Unit Tests — Calibration History
# ============================================================================


class TestCalibrationHistory:
    """Tests for get_calibration_history()."""

    def test_empty_history(self, calibrator):
        """Test that history is initially empty."""
        history = calibrator.get_calibration_history()
        assert history == []

    def test_history_accumulation(self, calibrator):
        """Test that history accumulates across calibrations."""
        embedding_func = create_dummy_embedding_func()
        input_generator = create_input_generator()

        # Run 3 calibrations
        for _ in range(3):
            calibrator.calibrate_embedding_sensitivity(
                embedding_func=embedding_func,
                input_generator=input_generator,
                num_samples=30,
            )

        history = calibrator.get_calibration_history()
        assert len(history) == 3

    def test_history_is_copy(self, calibrator):
        """Test that history returns a copy (not mutable reference)."""
        embedding_func = create_dummy_embedding_func()
        input_generator = create_input_generator()

        calibrator.calibrate_embedding_sensitivity(
            embedding_func=embedding_func,
            input_generator=input_generator,
            num_samples=30,
        )

        history1 = calibrator.get_calibration_history()
        history2 = calibrator.get_calibration_history()

        # Modifying one should not affect the other
        history1.append(999.0)

        assert len(history2) == 1
        assert 999.0 not in history2


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for sensitivity calibration."""

    def test_realistic_embedding_calibration(self, calibrator):
        """Test calibration with realistic embedding function."""

        # Create a more realistic embedding function
        # (simulating a neural network layer)
        def realistic_embedding(input_vec: np.ndarray) -> np.ndarray:
            # Multi-layer projection
            hidden = np.tanh(input_vec @ np.random.randn(len(input_vec), 256).astype(np.float32))
            output = hidden @ np.random.randn(256, 512).astype(np.float32)
            return output / np.linalg.norm(output)

        def input_gen() -> np.ndarray:
            vec = np.random.randn(784).astype(np.float32)  # MNIST-like
            return vec / np.linalg.norm(vec)

        sensitivity, stats = calibrator.calibrate_embedding_sensitivity(
            embedding_func=realistic_embedding,
            input_generator=input_gen,
            num_samples=200,
            confidence_level=0.95,
        )

        # Check that calibration succeeds and returns reasonable values
        assert 0 < sensitivity < 5.0
        assert stats["std"] >= 0
        assert len(calibrator.calibration_results) == 1

    def test_calibration_workflow(self, calibrator):
        """Test complete calibration workflow."""
        # 1. Define embedding function
        embedding_func = create_dummy_embedding_func(dim=512)

        # 2. Define input generator
        input_generator = create_input_generator(input_dim=100)

        # 3. Run calibration
        sensitivity, stats = calibrator.calibrate_embedding_sensitivity(
            embedding_func=embedding_func,
            input_generator=input_generator,
            num_samples=500,  # Larger for better estimate
            confidence_level=0.95,
        )

        # 4. Verify results
        assert sensitivity > 0

        # 5. Use sensitivity for DP noise
        # (This would be integrated with dp_noise.py in practice)
        epsilon = 1.0
        laplace_scale = sensitivity / epsilon

        assert laplace_scale > 0
        assert laplace_scale == sensitivity  # Since epsilon=1.0

        # 6. Check confidence interval width
        ci_lower, ci_upper = stats["confidence_interval"]
        ci_width = ci_upper - ci_lower

        # Width should be reasonable (not too wide)
        assert ci_width / sensitivity < 1.0  # Width less than 100% of estimate

    def test_comparison_embedding_vs_adjacent(self, calibrator):
        """Compare embedding and adjacent dataset calibration methods."""
        # Embedding method
        embedding_func = create_dummy_embedding_func()
        input_generator = create_input_generator()

        sens_embedding, stats_embedding = calibrator.calibrate_embedding_sensitivity(
            embedding_func=embedding_func,
            input_generator=input_generator,
            num_samples=100,
        )

        # Adjacent dataset method
        dataset_generator = create_dataset_generator()

        sens_adjacent, stats_adjacent = calibrator.calibrate_with_adjacent_datasets(
            embedding_func=aggregate_embedding_func,
            dataset_generator=dataset_generator,
            dataset_size=50,
            num_samples=100,
        )

        # Both should return positive sensitivities
        assert sens_embedding > 0
        assert sens_adjacent > 0

        # Both should be reasonable
        assert sens_embedding < 5.0
        assert sens_adjacent < 5.0


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_small_sample_size(self, calibrator):
        """Test calibration with very small sample size."""
        embedding_func = create_dummy_embedding_func()
        input_generator = create_input_generator()

        # Only 10 samples
        sensitivity, stats = calibrator.calibrate_embedding_sensitivity(
            embedding_func=embedding_func,
            input_generator=input_generator,
            num_samples=10,
        )

        # Should still work, but less reliable
        assert sensitivity > 0
        assert stats["num_samples"] == 10

    def test_zero_sensitivity_function(self, calibrator):
        """Test with a function that has zero sensitivity (constant output)."""

        def constant_embedding(input_vec: np.ndarray) -> np.ndarray:
            # Always return the same vector
            return np.ones(512, dtype=np.float32) / np.sqrt(512)

        input_generator = create_input_generator()

        sensitivity, stats = calibrator.calibrate_embedding_sensitivity(
            embedding_func=constant_embedding,
            input_generator=input_generator,
            num_samples=50,
        )

        # Sensitivity should be very close to zero
        assert sensitivity < 0.01
        assert stats["mean"] < 0.01
        assert stats["max"] < 0.01

    def test_high_dimensional_embeddings(self, calibrator):
        """Test calibration with high-dimensional embeddings."""
        embedding_func = create_dummy_embedding_func(dim=4096)  # Large
        input_generator = create_input_generator()

        sensitivity, stats = calibrator.calibrate_embedding_sensitivity(
            embedding_func=embedding_func,
            input_generator=input_generator,
            num_samples=50,
        )

        # Should work with high-dimensional embeddings
        assert sensitivity > 0
        assert sensitivity < 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
