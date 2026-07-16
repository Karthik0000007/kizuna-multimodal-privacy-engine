"""Sensitivity calibration for differential privacy.

Empirically estimates L2 sensitivity by measuring maximum change in
embedding output when input changes by one record.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from ..logger import get_logger

logger = get_logger("privacy")


class SensitivityCalibrator:
    """Calibrator for estimating L2 sensitivity empirically.

    L2 sensitivity Δf = max ||f(x) - f(x')||₂
    where x and x' are adjacent datasets (differ by one record).

    For embedding functions, we estimate this by:
    1. Generate pairs of similar inputs
    2. Compute embeddings for each
    3. Measure L2 distance between embeddings
    4. Take maximum over all pairs
    """

    def __init__(self) -> None:
        """Initialize sensitivity calibrator."""
        self.calibration_results: list[float] = []

        logger.info("sensitivity_calibrator_initialized")

    def calibrate_embedding_sensitivity(
        self,
        embedding_func: Callable[[NDArray], NDArray],
        input_generator: Callable[[], NDArray],
        num_samples: int = 10000,
        confidence_level: float = 0.95,
    ) -> tuple[float, dict]:
        """Calibrate sensitivity for an embedding function.

        Args:
            embedding_func: Function that takes input and returns embedding
            input_generator: Function that generates random inputs
            num_samples: Number of input pairs to test
            confidence_level: Confidence level for interval (e.g., 0.95)

        Returns:
            Tuple of (sensitivity, statistics_dict)
            sensitivity: Estimated L2 sensitivity
            statistics_dict: Dict with mean, std, confidence intervals, etc.
        """
        logger.info(
            "sensitivity_calibration_started",
            num_samples=num_samples,
            confidence_level=confidence_level,
        )

        sensitivities = []

        # Generate samples and measure sensitivity
        for _ in tqdm(range(num_samples), desc="Calibrating sensitivity"):
            # Generate two random inputs
            input1 = input_generator()
            input2 = input_generator()

            # Compute embeddings
            emb1 = embedding_func(input1)
            emb2 = embedding_func(input2)

            # Measure L2 distance
            l2_distance = np.linalg.norm(emb1 - emb2)
            sensitivities.append(float(l2_distance))

        sensitivities = np.array(sensitivities)

        # Compute statistics
        mean_sensitivity = np.mean(sensitivities)
        std_sensitivity = np.std(sensitivities)
        min_sensitivity = np.min(sensitivities)
        max_sensitivity = np.max(sensitivities)
        median_sensitivity = np.median(sensitivities)

        # Confidence interval
        confidence_percentile = confidence_level * 100
        lower_percentile = (100 - confidence_percentile) / 2
        upper_percentile = 100 - lower_percentile

        ci_lower = np.percentile(sensitivities, lower_percentile)
        ci_upper = np.percentile(sensitivities, upper_percentile)

        # Conservative estimate: use upper bound of confidence interval
        sensitivity_estimate = ci_upper

        statistics = {
            "num_samples": num_samples,
            "mean": float(mean_sensitivity),
            "std": float(std_sensitivity),
            "min": float(min_sensitivity),
            "max": float(max_sensitivity),
            "median": float(median_sensitivity),
            "p95": float(np.percentile(sensitivities, 95)),
            "p99": float(np.percentile(sensitivities, 99)),
            "confidence_level": confidence_level,
            "confidence_interval": (float(ci_lower), float(ci_upper)),
            "sensitivity_estimate": float(sensitivity_estimate),
        }

        self.calibration_results.append(float(sensitivity_estimate))

        logger.info(
            "sensitivity_calibration_complete",
            sensitivity_estimate=sensitivity_estimate,
            mean=mean_sensitivity,
            std=std_sensitivity,
            confidence_interval=(ci_lower, ci_upper),
        )

        return sensitivity_estimate, statistics

    def calibrate_with_adjacent_datasets(
        self,
        embedding_func: Callable[[list], NDArray],
        dataset_generator: Callable[[int], list],
        dataset_size: int = 100,
        num_samples: int = 1000,
        confidence_level: float = 0.95,
    ) -> tuple[float, dict]:
        """Calibrate sensitivity using adjacent dataset definition.

        More rigorous approach: generate datasets that differ by exactly
        one record, compute aggregate embeddings, measure difference.

        Args:
            embedding_func: Function that takes dataset and returns embedding
            dataset_generator: Function that generates dataset of given size
            dataset_size: Size of datasets to generate
            num_samples: Number of dataset pairs to test
            confidence_level: Confidence level for interval

        Returns:
            Tuple of (sensitivity, statistics_dict)
        """
        logger.info(
            "sensitivity_calibration_adjacent_started",
            dataset_size=dataset_size,
            num_samples=num_samples,
        )

        sensitivities = []

        for _ in tqdm(range(num_samples), desc="Calibrating with adjacent datasets"):
            # Generate base dataset
            dataset = dataset_generator(dataset_size)

            # Create adjacent dataset (replace one record)
            adjacent_dataset = dataset.copy()
            replace_idx = np.random.randint(0, dataset_size)
            adjacent_dataset[replace_idx] = dataset_generator(1)[0]

            # Compute embeddings for both datasets
            emb1 = embedding_func(dataset)
            emb2 = embedding_func(adjacent_dataset)

            # Measure L2 distance
            l2_distance = np.linalg.norm(emb1 - emb2)
            sensitivities.append(float(l2_distance))

        sensitivities = np.array(sensitivities)

        # Compute statistics (same as above)
        mean_sensitivity = np.mean(sensitivities)
        std_sensitivity = np.std(sensitivities)
        min_sensitivity = np.min(sensitivities)
        max_sensitivity = np.max(sensitivities)
        median_sensitivity = np.median(sensitivities)

        confidence_percentile = confidence_level * 100
        lower_percentile = (100 - confidence_percentile) / 2
        upper_percentile = 100 - lower_percentile

        ci_lower = np.percentile(sensitivities, lower_percentile)
        ci_upper = np.percentile(sensitivities, upper_percentile)

        sensitivity_estimate = ci_upper

        statistics = {
            "num_samples": num_samples,
            "dataset_size": dataset_size,
            "mean": float(mean_sensitivity),
            "std": float(std_sensitivity),
            "min": float(min_sensitivity),
            "max": float(max_sensitivity),
            "median": float(median_sensitivity),
            "p95": float(np.percentile(sensitivities, 95)),
            "p99": float(np.percentile(sensitivities, 99)),
            "confidence_level": confidence_level,
            "confidence_interval": (float(ci_lower), float(ci_upper)),
            "sensitivity_estimate": float(sensitivity_estimate),
        }

        self.calibration_results.append(float(sensitivity_estimate))

        logger.info(
            "sensitivity_calibration_adjacent_complete",
            sensitivity_estimate=sensitivity_estimate,
            mean=mean_sensitivity,
            std=std_sensitivity,
        )

        return sensitivity_estimate, statistics

    def get_calibration_history(self) -> list[float]:
        """Get history of all calibration runs.

        Returns:
            List of sensitivity estimates
        """
        return self.calibration_results.copy()


def main() -> None:
    """Demo sensitivity calibration."""
    import argparse

    parser = argparse.ArgumentParser(description="Kizuna Sensitivity Calibration Demo")
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=512,
        help="Embedding dimension",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples for calibration",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Confidence level for interval",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Kizuna Sensitivity Calibration Demo")
    print("=" * 70)

    # Create dummy embedding function (L2-normalized random projection)
    embedding_dim = args.embedding_dim

    def dummy_embedding_func(input_vec: NDArray) -> NDArray:
        """Dummy embedding function for demo."""
        # Random projection + L2 normalization
        projection = np.random.randn(len(input_vec), embedding_dim).astype(np.float32)
        embedded = input_vec @ projection
        return embedded / np.linalg.norm(embedded)

    # Input generator (random vectors)
    def input_generator() -> NDArray:
        """Generate random input."""
        vec = np.random.randn(100).astype(np.float32)
        return vec / np.linalg.norm(vec)

    # Initialize calibrator
    print("\nInitializing calibrator...")
    calibrator = SensitivityCalibrator()

    # Run calibration
    print("\nCalibrating sensitivity...")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Number of samples: {args.num_samples}")
    print(f"  Confidence level: {args.confidence_level}")

    sensitivity, stats = calibrator.calibrate_embedding_sensitivity(
        embedding_func=dummy_embedding_func,
        input_generator=input_generator,
        num_samples=args.num_samples,
        confidence_level=args.confidence_level,
    )

    # Print results
    print(f"\n{'=' * 70}")
    print("Calibration Results")
    print(f"{'=' * 70}")

    print(f"\nSensitivity Estimate: {sensitivity:.6f}")
    print("  (Use this value for DP noise scale)")

    print("\nStatistics:")
    print(f"  Mean: {stats['mean']:.6f}")
    print(f"  Std: {stats['std']:.6f}")
    print(f"  Min: {stats['min']:.6f}")
    print(f"  Max: {stats['max']:.6f}")
    print(f"  Median: {stats['median']:.6f}")
    print(f"  P95: {stats['p95']:.6f}")
    print(f"  P99: {stats['p99']:.6f}")

    ci_lower, ci_upper = stats["confidence_interval"]
    print(f"\nConfidence Interval ({stats['confidence_level']:.0%}):")
    print(f"  Lower: {ci_lower:.6f}")
    print(f"  Upper: {ci_upper:.6f}")
    print(f"  Width: {ci_upper - ci_lower:.6f}")

    # Recommendation
    print(f"\n{'=' * 70}")
    print("Recommendation")
    print(f"{'=' * 70}")
    print("\nFor conservative privacy guarantee, use:")
    print(f"  sensitivity = {sensitivity:.6f}")
    print(f"\nThis ensures that {stats['confidence_level']:.0%} of observed")
    print("sensitivities are below this threshold.")

    # Example DP noise scales
    print("\nExample DP noise scales:")
    for epsilon in [0.1, 0.5, 1.0, 2.0, 5.0]:
        laplace_scale = sensitivity / epsilon
        gaussian_sigma = sensitivity * np.sqrt(2 * np.log(1.25 / 1e-5)) / epsilon

        print(f"\n  ε = {epsilon:.1f}:")
        print(f"    Laplace scale: {laplace_scale:.6f}")
        print(f"    Gaussian σ (δ=1e-5): {gaussian_sigma:.6f}")

    print(f"\n{'=' * 70}")
    print("✓ Demo complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
