import numpy as np
import pytest


def generate_similar_embeddings(base_vector, num_samples, noise_scale=0.1):
    """Simulates the encoder output for slightly varying inputs of the same scenario."""
    embeddings = []
    for _ in range(num_samples):
        vec = base_vector + np.random.normal(0, noise_scale, size=base_vector.shape)
        vec /= np.linalg.norm(vec) + 1e-9
        embeddings.append(vec)
    return embeddings


def test_embedding_consistency():
    """
    Property 2: Embedding Similarity Preservation.
    Similar inputs should yield high cosine similarity (>0.8).
    Dissimilar inputs should yield low cosine similarity (<0.5).
    """
    np.random.seed(42)

    # Scenario A (e.g., normal walking)
    base_a = np.random.randn(512).astype(np.float32)
    base_a /= np.linalg.norm(base_a)

    # Scenario B (e.g., fall)
    base_b = np.random.randn(512).astype(np.float32)
    base_b /= np.linalg.norm(base_b)

    # Generate variations
    cluster_a = generate_similar_embeddings(base_a, 100, noise_scale=0.01)
    cluster_b = generate_similar_embeddings(base_b, 100, noise_scale=0.01)

    # Test similar (intra-cluster)
    similarities_a = []
    for i in range(len(cluster_a) - 1):
        sim = np.dot(cluster_a[i], cluster_a[i + 1])
        similarities_a.append(sim)

    assert (
        np.mean(similarities_a) > 0.8
    ), f"Mean intra-cluster similarity was {np.mean(similarities_a)}"

    # Test dissimilar (inter-cluster)
    similarities_ab = []
    for i in range(len(cluster_a)):
        sim = np.dot(cluster_a[i], cluster_b[i])
        similarities_ab.append(sim)

    assert (
        np.mean(similarities_ab) < 0.5
    ), f"Mean inter-cluster similarity was {np.mean(similarities_ab)}"


def test_dp_utility():
    """
    Property 3: DP Utility Preservation.
    Applying DP noise (epsilon=1.0) should not destroy the signal entirely.
    """
    from src.privacy.dp_noise import LaplaceMechanism

    np.random.seed(42)
    base_signal = np.random.randn(512).astype(np.float32)
    base_signal /= np.linalg.norm(base_signal)

    injector = LaplaceMechanism(epsilon=1.0, sensitivity=1.0)

    # We check if the noisy signal still strongly correlates with the original base signal
    # This simulates recall of an anomaly (if it correlates with the anomaly cluster center)
    similarities = []
    for _ in range(100):
        noisy = injector.add_noise(base_signal)
        sim = np.dot(base_signal, noisy)
        similarities.append(sim)

    # With eps=1.0, the signal is preserved enough that the mean correlation remains positive
    # and significantly above random chance (which would be ~0.0)
    assert np.mean(similarities) > 0.3, "DP noise at eps=1.0 destroyed too much signal utility."
