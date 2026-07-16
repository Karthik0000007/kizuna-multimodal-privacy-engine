import time

import numpy as np

from src.anomaly.classifier import AnomalyClassifier
from src.anomaly.cluster_detector import ClusterAnomalyDetector
from src.anomaly.density_detector import DensityAnomalyDetector
from src.anomaly.knn_detector import KNNAnomalyDetector
from src.anomaly.models import PrivacyEvent
from src.database.base import VectorStore


class AnomalyOrchestrator:
    """
    Combines KNN, density, and cluster detectors.
    Implements ensemble voting and triggers the classifier if an anomaly is confirmed.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        knn_threshold: float = 0.8,
        density_threshold: float = 1.5,
        cluster_threshold: float = 0.7,
    ):
        self.knn_detector = KNNAnomalyDetector(vector_store, threshold=knn_threshold)
        self.density_detector = DensityAnomalyDetector(
            vector_store, lof_threshold=density_threshold
        )
        self.cluster_detector = ClusterAnomalyDetector(
            vector_store, cluster_threshold=cluster_threshold
        )
        self.classifier = AnomalyClassifier(vector_store)

    def process(self, embedding: np.ndarray, source_node_id: str) -> PrivacyEvent | None:
        """
        Process an embedding through the ensemble.
        Returns a PrivacyEvent if an anomaly is confirmed, else None.
        Raises ValueError if embedding contains NaN or Inf.
        """
        if not np.isfinite(embedding).all():
            raise ValueError("Embedding contains NaN or Inf values")

        # Run individual detectors
        knn_res = self.knn_detector.detect(embedding)
        density_res = self.density_detector.detect(embedding)
        cluster_res = self.cluster_detector.detect(embedding)

        results = [knn_res, density_res, cluster_res]

        # Ensemble voting: ≥ 2 of 3 detectors agree
        votes = sum(1 for r in results if r.is_anomaly)
        is_confirmed = votes >= 2

        if not is_confirmed:
            return None

        # Compute unified confidence score as weighted average
        # In a real system, weights could be learned. For now, simple average of normalized scores.
        # Normalize scores to [0, 1] roughly.
        knn_conf = min(knn_res.score / 1.0, 1.0)
        density_conf = min(density_res.score / 2.0, 1.0)
        cluster_conf = min(cluster_res.score / 1.0, 1.0)

        confidence = (knn_conf + density_conf + cluster_conf) / 3.0

        # Classify the anomaly
        classes = self.classifier.classify(embedding)
        top_type = classes[0][0] if classes else "unknown"
        top_k_types = [c[0] for c in classes]

        event = PrivacyEvent(
            timestamp=time.time(),
            source_node_id=source_node_id,
            is_anomaly=True,
            confidence=float(confidence),
            anomaly_type=top_type,
            top_k_types=top_k_types,
            detector_results=results,
        )

        return event
