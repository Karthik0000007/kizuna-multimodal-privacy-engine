from typing import List

import numpy as np

from src.anomaly.models import AnomalyResult
from src.database.base import VectorStore


class KNNAnomalyDetector:
    """
    Detects anomalies by querying the vector DB for K nearest neighbors.
    If the average distance (or similarity) is beyond a threshold, it is flagged as an anomaly.
    """

    def __init__(self, vector_store: VectorStore, k: int = 10, threshold: float = 0.8):
        """
        Args:
            vector_store: Interface to the vector database.
            k: Number of nearest neighbors to retrieve.
            threshold: Similarity threshold. If average similarity is < threshold, it's an anomaly.
        """
        self.vector_store = vector_store
        self.k = k
        self.threshold = threshold
        self.name = "knn"

    def detect(self, embedding: np.ndarray) -> AnomalyResult:
        results = self.vector_store.search(embedding, top_k=self.k)

        if not results:
            # If no historical data exists, it's anomalous (or assume normal, but let's say normal for cold start)
            return AnomalyResult(
                is_anomaly=False, score=0.0, nearest_neighbors=[], detector_name=self.name
            )

        avg_similarity = sum(r.score for r in results) / len(results)

        # Lower similarity means higher anomaly score
        anomaly_score = 1.0 - avg_similarity

        is_anomaly = avg_similarity < self.threshold

        neighbor_ids = [r.id for r in results]

        return AnomalyResult(
            is_anomaly=is_anomaly,
            score=anomaly_score,
            nearest_neighbors=neighbor_ids,
            detector_name=self.name,
        )
