"""Differential privacy noise mechanisms.

Implements Laplace and Gaussian mechanisms for adding calibrated noise
to embedding vectors to achieve (ε)-DP or (ε,δ)-DP guarantees.
"""

import math
from enum import Enum
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..logger import get_logger

logger = get_logger("privacy")


class DPMechanism(str, Enum):
    """Differential privacy mechanism types."""

    LAPLACE = "laplace"
    GAUSSIAN = "gaussian"


class LaplaceMechanism:
    """Laplace mechanism for (ε)-differential privacy.

    Adds noise drawn from Laplace distribution with scale parameter b = Δf/ε,
    where Δf is the sensitivity and ε is the privacy budget.

    Provides (ε)-differential privacy guarantee.
    """

    def __init__(self, epsilon: float, sensitivity: float) -> None:
        """Initialize Laplace mechanism.

        Args:
            epsilon: Privacy budget (lower = more private, more noise)
            sensitivity: L2 sensitivity of the query function

        Raises:
            ValueError: If epsilon or sensitivity are not positive
        """
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if sensitivity <= 0:
            raise ValueError(f"sensitivity must be positive, got {sensitivity}")

        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self.scale = sensitivity / epsilon

        logger.info(
            "laplace_mechanism_initialized",
            epsilon=epsilon,
            sensitivity=sensitivity,
            scale=self.scale,
        )

    def add_noise(self, vector: NDArray[np.float32]) -> NDArray[np.float32]:
        """Add Laplace noise to a vector.

        Args:
            vector: Input vector (D,)

        Returns:
            Noised vector (D,), same shape and dtype as input

        Raises:
            ValueError: If vector is not 1D
        """
        if vector.ndim != 1:
            raise ValueError(f"Expected 1D vector, got shape {vector.shape}")

        # Generate Laplace noise: scale parameter = b
        # Laplace distribution: f(x) = (1/2b) * exp(-|x|/b)
        noise = np.random.laplace(loc=0.0, scale=self.scale, size=vector.shape)

        # Add noise
        noised_vector = vector + noise.astype(vector.dtype)

        # Calculate noise magnitude for logging
        noise_magnitude = np.linalg.norm(noise)
        signal_magnitude = np.linalg.norm(vector)
        snr = signal_magnitude / noise_magnitude if noise_magnitude > 0 else float("inf")

        logger.debug(
            "laplace_noise_added",
            vector_dim=len(vector),
            noise_magnitude=float(noise_magnitude),
            signal_magnitude=float(signal_magnitude),
            snr=float(snr),
            epsilon=self.epsilon,
        )

        return noised_vector

    def add_noise_batch(
        self,
        vectors: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Add Laplace noise to a batch of vectors.

        Args:
            vectors: Input vectors (B, D)

        Returns:
            Noised vectors (B, D), same shape and dtype as input

        Raises:
            ValueError: If vectors is not 2D
        """
        if vectors.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {vectors.shape}")

        # Generate Laplace noise for entire batch
        noise = np.random.laplace(loc=0.0, scale=self.scale, size=vectors.shape)

        # Add noise
        noised_vectors = vectors + noise.astype(vectors.dtype)

        logger.debug(
            "laplace_noise_added_batch",
            batch_size=vectors.shape[0],
            vector_dim=vectors.shape[1],
            epsilon=self.epsilon,
        )

        return noised_vectors

    def get_privacy_guarantee(self) -> dict:
        """Get privacy guarantee parameters.

        Returns:
            Dictionary with privacy parameters
        """
        return {
            "mechanism": "laplace",
            "epsilon": self.epsilon,
            "delta": 0.0,  # Pure ε-DP
            "sensitivity": self.sensitivity,
        }


class GaussianMechanism:
    """Gaussian mechanism for (ε,δ)-differential privacy.

    Adds noise drawn from Gaussian distribution with standard deviation
    σ = sensitivity * sqrt(2 * ln(1.25/δ)) / ε.

    Provides (ε,δ)-differential privacy guarantee.
    """

    def __init__(
        self,
        epsilon: float,
        delta: float,
        sensitivity: float,
    ) -> None:
        """Initialize Gaussian mechanism.

        Args:
            epsilon: Privacy budget (lower = more private, more noise)
            delta: Probability of privacy breach (typically 1e-5)
            sensitivity: L2 sensitivity of the query function

        Raises:
            ValueError: If epsilon, delta, or sensitivity are invalid
        """
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if delta <= 0 or delta >= 1:
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        if sensitivity <= 0:
            raise ValueError(f"sensitivity must be positive, got {sensitivity}")

        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity

        # Compute sigma using the composition theorem
        # σ = Δf * sqrt(2 * ln(1.25/δ)) / ε
        self.sigma = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon

        logger.info(
            "gaussian_mechanism_initialized",
            epsilon=epsilon,
            delta=delta,
            sensitivity=sensitivity,
            sigma=self.sigma,
        )

    def add_noise(self, vector: NDArray[np.float32]) -> NDArray[np.float32]:
        """Add Gaussian noise to a vector.

        Args:
            vector: Input vector (D,)

        Returns:
            Noised vector (D,), same shape and dtype as input

        Raises:
            ValueError: If vector is not 1D
        """
        if vector.ndim != 1:
            raise ValueError(f"Expected 1D vector, got shape {vector.shape}")

        # Generate Gaussian noise: standard deviation = sigma
        # Gaussian distribution: f(x) = (1/sqrt(2πσ²)) * exp(-x²/2σ²)
        noise = np.random.normal(loc=0.0, scale=self.sigma, size=vector.shape)

        # Add noise
        noised_vector = vector + noise.astype(vector.dtype)

        # Calculate noise magnitude for logging
        noise_magnitude = np.linalg.norm(noise)
        signal_magnitude = np.linalg.norm(vector)
        snr = signal_magnitude / noise_magnitude if noise_magnitude > 0 else float("inf")

        logger.debug(
            "gaussian_noise_added",
            vector_dim=len(vector),
            noise_magnitude=float(noise_magnitude),
            signal_magnitude=float(signal_magnitude),
            snr=float(snr),
            epsilon=self.epsilon,
            delta=self.delta,
        )

        return noised_vector

    def add_noise_batch(
        self,
        vectors: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Add Gaussian noise to a batch of vectors.

        Args:
            vectors: Input vectors (B, D)

        Returns:
            Noised vectors (B, D), same shape and dtype as input

        Raises:
            ValueError: If vectors is not 2D
        """
        if vectors.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {vectors.shape}")

        # Generate Gaussian noise for entire batch
        noise = np.random.normal(loc=0.0, scale=self.sigma, size=vectors.shape)

        # Add noise
        noised_vectors = vectors + noise.astype(vectors.dtype)

        logger.debug(
            "gaussian_noise_added_batch",
            batch_size=vectors.shape[0],
            vector_dim=vectors.shape[1],
            epsilon=self.epsilon,
            delta=self.delta,
        )

        return noised_vectors

    def get_privacy_guarantee(self) -> dict:
        """Get privacy guarantee parameters.

        Returns:
            Dictionary with privacy parameters
        """
        return {
            "mechanism": "gaussian",
            "epsilon": self.epsilon,
            "delta": self.delta,
            "sensitivity": self.sensitivity,
        }


class DPNoiseAdder:
    """Factory class for differential privacy noise mechanisms.

    Provides a unified interface for adding DP noise using either
    Laplace or Gaussian mechanism.
    """

    def __init__(
        self,
        mechanism: str | DPMechanism,
        epsilon: float,
        sensitivity: float,
        delta: Optional[float] = None,
    ) -> None:
        """Initialize DP noise adder.

        Args:
            mechanism: "laplace" or "gaussian"
            epsilon: Privacy budget
            sensitivity: L2 sensitivity
            delta: Probability of privacy breach (required for Gaussian)

        Raises:
            ValueError: If mechanism is invalid or delta missing for Gaussian
        """
        if isinstance(mechanism, str):
            mechanism = DPMechanism(mechanism.lower())

        self.mechanism_type = mechanism

        if mechanism == DPMechanism.LAPLACE:
            self.mechanism = LaplaceMechanism(epsilon=epsilon, sensitivity=sensitivity)
        elif mechanism == DPMechanism.GAUSSIAN:
            if delta is None:
                raise ValueError("delta is required for Gaussian mechanism")
            self.mechanism = GaussianMechanism(
                epsilon=epsilon,
                delta=delta,
                sensitivity=sensitivity,
            )
        else:
            raise ValueError(f"Unknown mechanism: {mechanism}")

        logger.info(
            "dp_noise_adder_initialized",
            mechanism=mechanism.value,
            epsilon=epsilon,
            sensitivity=sensitivity,
            delta=delta,
        )

    def add_noise(self, vector: NDArray[np.float32]) -> NDArray[np.float32]:
        """Add DP noise to a vector.

        Args:
            vector: Input vector (D,)

        Returns:
            Noised vector (D,)
        """
        return self.mechanism.add_noise(vector)

    def add_noise_batch(
        self,
        vectors: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Add DP noise to a batch of vectors.

        Args:
            vectors: Input vectors (B, D)

        Returns:
            Noised vectors (B, D)
        """
        return self.mechanism.add_noise_batch(vectors)

    def get_privacy_guarantee(self) -> dict:
        """Get privacy guarantee parameters.

        Returns:
            Dictionary with privacy parameters
        """
        return self.mechanism.get_privacy_guarantee()


def main() -> None:
    """Demo DP noise mechanisms."""
    import argparse

    parser = argparse.ArgumentParser(description="Kizuna DP Noise Demo")
    parser.add_argument(
        "--mechanism",
        type=str,
        default="laplace",
        choices=["laplace", "gaussian"],
        help="DP mechanism to use",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0,
        help="Privacy budget (lower = more private)",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        help="Delta for Gaussian mechanism",
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=2.0,
        help="L2 sensitivity",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=512,
        help="Embedding dimension",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of noise samples for visualization",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Kizuna Differential Privacy Noise Demo")
    print("=" * 70)

    # Create DP noise adder
    print(f"\nInitializing {args.mechanism.upper()} mechanism...")
    print(f"  Epsilon: {args.epsilon}")
    print(f"  Delta: {args.delta if args.mechanism == 'gaussian' else 'N/A'}")
    print(f"  Sensitivity: {args.sensitivity}")

    dp_adder = DPNoiseAdder(
        mechanism=args.mechanism,
        epsilon=args.epsilon,
        sensitivity=args.sensitivity,
        delta=args.delta if args.mechanism == "gaussian" else None,
    )

    guarantee = dp_adder.get_privacy_guarantee()
    print(f"\nPrivacy Guarantee:")
    print(f"  Mechanism: {guarantee['mechanism']}")
    print(f"  ε: {guarantee['epsilon']}")
    print(f"  δ: {guarantee['delta']}")
    print(f"  Sensitivity: {guarantee['sensitivity']}")

    # Create a test embedding (unit vector)
    print(f"\nGenerating test embedding (dimension: {args.dimension})...")
    original = np.random.randn(args.dimension).astype(np.float32)
    original = original / np.linalg.norm(original)  # L2 normalize

    print(f"  Original norm: {np.linalg.norm(original):.6f}")

    # Add noise
    print(f"\nAdding DP noise...")
    noised = dp_adder.add_noise(original)

    print(f"  Noised norm: {np.linalg.norm(noised):.6f}")

    # Calculate similarity
    cosine_sim = np.dot(original, noised) / (np.linalg.norm(original) * np.linalg.norm(noised))
    l2_distance = np.linalg.norm(original - noised)

    print(f"\nSimilarity Metrics:")
    print(f"  Cosine similarity: {cosine_sim:.6f}")
    print(f"  L2 distance: {l2_distance:.6f}")

    # Analyze noise distribution
    print(f"\nAnalyzing noise distribution ({args.num_samples} samples)...")

    noise_samples = []
    for _ in range(args.num_samples):
        noised_sample = dp_adder.add_noise(original)
        noise = noised_sample - original
        noise_samples.append(np.linalg.norm(noise))

    noise_samples = np.array(noise_samples)

    print(f"  Noise magnitude statistics:")
    print(f"    Mean: {np.mean(noise_samples):.6f}")
    print(f"    Std: {np.std(noise_samples):.6f}")
    print(f"    Min: {np.min(noise_samples):.6f}")
    print(f"    Max: {np.max(noise_samples):.6f}")
    print(f"    P50 (median): {np.percentile(noise_samples, 50):.6f}")
    print(f"    P95: {np.percentile(noise_samples, 95):.6f}")

    # Test batch processing
    print(f"\nTesting batch processing (10 vectors)...")
    batch = np.random.randn(10, args.dimension).astype(np.float32)
    batch = batch / np.linalg.norm(batch, axis=1, keepdims=True)

    noised_batch = dp_adder.add_noise_batch(batch)

    print(f"  Original batch shape: {batch.shape}")
    print(f"  Noised batch shape: {noised_batch.shape}")

    # Calculate average cosine similarity for batch
    cosine_sims = []
    for orig, nois in zip(batch, noised_batch):
        sim = np.dot(orig, nois) / (np.linalg.norm(orig) * np.linalg.norm(nois))
        cosine_sims.append(sim)

    print(f"  Average cosine similarity: {np.mean(cosine_sims):.6f}")

    print(f"\n{'=' * 70}")
    print("✓ Demo complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
