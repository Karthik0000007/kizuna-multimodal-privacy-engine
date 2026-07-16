import time
from typing import List

import numpy as np

from src.database.base import VectorStore


class AnomalyEnroller:
    """
    Enrolls new anomaly types using a few-shot paradigm.
    Computes a prototype and decision boundary, then stores it in the vector DB.
    """

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def enroll(self, examples: List[np.ndarray], label: str) -> str:
        """
        Accepts N example embeddings for a new anomaly type.
        Computes prototype (mean embedding) and decision boundary (mean distance + 2σ of examples).
        Stores as a new cluster centroid in the vector database.

        Args:
            examples: List of embedding vectors for the anomaly class.
            label: The event_type / label for this anomaly.

        Returns:
            The point UUID of the inserted centroid.
        """
        if not examples:
            raise ValueError("At least one example is required for enrollment.")

        # Convert list of vectors into a 2D array
        examples_array = np.vstack(examples)

        # Compute prototype (mean embedding)
        prototype = np.mean(examples_array, axis=0)

        # Normalize the prototype to maintain cosine similarity metrics if required
        norm = np.linalg.norm(prototype)
        if norm > 0:
            prototype = prototype / norm

        # Compute decision boundary based on distance of examples from the prototype
        if len(examples) > 1:
            # Cosine distance from each example to the prototype
            # Since vectors are normalized, dot product = cosine similarity
            # Distance = 1 - similarity
            similarities = np.dot(examples_array, prototype)
            distances = 1.0 - similarities

            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            decision_boundary = mean_dist + (2 * std_dist)
        else:
            # Fallback decision boundary for 1-shot learning
            decision_boundary = 0.2  # arbitrary conservative threshold

        # Prepare metadata for storage
        metadata = {
            "timestamp": time.time(),
            "source_node_id": "enrollment_system",
            "modalities_fused": ["video", "audio", "sensor"],  # default assumption
            "event_type": label,
            "dp_epsilon": 0.0,  # Not applicable for synthetic/enrolled exemplars
            "is_centroid": True,
            "is_exemplar": True,
            "decision_boundary": float(decision_boundary),
        }

        # Store in vector DB
        point_id = self.vector_store.insert(prototype, metadata)
        return point_id


def pre_enroll_japan_scenarios(vector_store: VectorStore):
    """
    Pre-enrolls mock exemplars for Japan-specific scenarios so the AnomalyClassifier
    can detect them via few-shot matching.
    """
    from src.anomaly.scenarios import JapanScenario

    enroller = AnomalyEnroller(vector_store)

    # Generate mock embeddings for each scenario
    # In a real system, these would be extracted from the UP-Fall dataset, ESC-50, etc.
    np.random.seed(42)

    scenarios = {
        JapanScenario.FALL_RISK.value: np.random.randn(5, 512).astype(np.float32),
        JapanScenario.WANDERING.value: np.random.randn(5, 512).astype(np.float32),
        JapanScenario.CONGESTION.value: np.random.randn(5, 512).astype(np.float32),
        JapanScenario.UNUSUAL_SOUND.value: np.random.randn(5, 512).astype(np.float32),
    }

    for event_type, examples in scenarios.items():
        # Normalize mock examples
        examples = [ex / (np.linalg.norm(ex) + 1e-9) for ex in examples]
        enroller.enroll(examples, label=event_type)
