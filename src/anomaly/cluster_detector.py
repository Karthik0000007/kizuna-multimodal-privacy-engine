from typing import List

import numpy as np

from src.anomaly.models import AnomalyResult
from src.database.base import VectorStore


class ClusterAnomalyDetector:
    """
    Detects anomalies by checking distance against predefined known clusters (centroids).
    """

    def __init__(self, vector_store: VectorStore, cluster_threshold: float = 0.7):
        """
        Args:
            vector_store: Vector DB where centroids are stored.
            cluster_threshold: Distance threshold for out-of-distribution.
        """
        self.vector_store = vector_store
        self.cluster_threshold = cluster_threshold
        self.name = "cluster"

    def detect(self, embedding: np.ndarray) -> AnomalyResult:
        # Search against points that are cluster centroids
        results = self.vector_store.search(
            query=embedding,
            top_k=1,
            filters={"is_centroid": True},  # Assuming centroids are tagged with this
        )

        if not results:
            # No clusters defined yet
            return AnomalyResult(
                is_anomaly=False, score=0.0, nearest_neighbors=[], detector_name=self.name
            )

        nearest_cluster = results[0]
        # Calculate distance proxy (1 - cosine similarity)
        distance = 1.0 - nearest_cluster.score

        is_anomaly = distance > self.cluster_threshold

        return AnomalyResult(
            is_anomaly=is_anomaly,
            score=distance,
            nearest_neighbors=[nearest_cluster.id],
            detector_name=self.name,
        )
