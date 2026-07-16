import numpy as np
import pytest

from src.anomaly.classifier import AnomalyClassifier
from src.anomaly.cluster_detector import ClusterAnomalyDetector
from src.anomaly.density_detector import DensityAnomalyDetector
from src.anomaly.detector import AnomalyOrchestrator
from src.anomaly.knn_detector import KNNAnomalyDetector
from src.anomaly.models import AnomalyResult, PrivacyEvent
from src.database.base import SearchResult, VectorStore


class MockVectorStore(VectorStore):
    def __init__(self, mock_results=None):
        self.mock_results = mock_results or []

    def insert(self, vector, metadata):
        pass

    def search(self, query, top_k=10, filters=None):
        return self.mock_results

    def create_collection(self, dimension, distance="Cosine"):
        pass

    def delete_collection(self):
        pass

    def get_collection_info(self):
        return {}


def test_knn_detector():
    # Setup mock returning low similarity (high distance -> anomaly)
    mock_store_anom = MockVectorStore(
        [SearchResult(id="1", score=0.5, payload={}), SearchResult(id="2", score=0.6, payload={})]
    )
    knn = KNNAnomalyDetector(mock_store_anom, threshold=0.8)
    res = knn.detect(np.array([0.1]))
    assert res.is_anomaly is True
    assert res.score > 0

    # Setup mock returning high similarity (normal)
    mock_store_norm = MockVectorStore(
        [SearchResult(id="1", score=0.9, payload={}), SearchResult(id="2", score=0.95, payload={})]
    )
    knn_norm = KNNAnomalyDetector(mock_store_norm, threshold=0.8)
    res_norm = knn_norm.detect(np.array([0.1]))
    assert res_norm.is_anomaly is False


def test_density_detector():
    # High distance (low similarity proxy)
    mock_store_anom = MockVectorStore(
        [SearchResult(id="1", score=0.1, payload={}) for _ in range(5)]
    )
    density = DensityAnomalyDetector(mock_store_anom, lof_threshold=0.5)
    res = density.detect(np.array([0.1]))
    assert res.is_anomaly is True

    # Normal distance
    mock_store_norm = MockVectorStore(
        [SearchResult(id="1", score=0.9, payload={}) for _ in range(5)]
    )
    density_norm = DensityAnomalyDetector(mock_store_norm, lof_threshold=0.5)
    res_norm = density_norm.detect(np.array([0.1]))
    assert res_norm.is_anomaly is False


def test_cluster_detector():
    # Low similarity to cluster centroid -> OOD
    mock_store_anom = MockVectorStore(
        [SearchResult(id="c1", score=0.1, payload={"is_centroid": True})]
    )
    cluster = ClusterAnomalyDetector(mock_store_anom, cluster_threshold=0.7)
    res = cluster.detect(np.array([0.1]))
    assert res.is_anomaly is True


def test_classifier():
    mock_store = MockVectorStore(
        [
            SearchResult(
                id="1", score=0.9, payload={"event_type": "fall_risk", "is_exemplar": True}
            ),
            SearchResult(
                id="2", score=0.8, payload={"event_type": "fall_risk", "is_exemplar": True}
            ),
            SearchResult(
                id="3", score=0.7, payload={"event_type": "wandering", "is_exemplar": True}
            ),
        ]
    )
    classifier = AnomalyClassifier(mock_store)
    classes = classifier.classify(np.array([0.1]))
    assert classes[0][0] == "fall_risk"
    assert classes[1][0] == "wandering"


def test_orchestrator():
    # We want 2 out of 3 detectors to vote True
    # KNN will vote True (avg sim < 0.8), Density will vote True (avg sim < 0.5), Cluster will vote False

    # Actually, we should mock the individual detectors or the vector store.
    # It's easier to mock the vector store with a single search response that triggers multiple things.
    # KNN threshold 0.8 (sim < 0.8 is anomaly)
    # Density threshold 1.5 (dist > 1.5 is anomaly. meaning sim < -0.5, wait, sim is [0,1], distance=1-sim, so dist max is 1. If lof_threshold=0.5, dist > 0.5 -> sim < 0.5 is anomaly)
    # Let's set the orchestrator with specific thresholds

    mock_store = MockVectorStore(
        [
            SearchResult(
                id="1", score=0.4, payload={"event_type": "congestion_alert", "is_exemplar": True}
            ),
            SearchResult(id="2", score=0.4, payload={}),
            SearchResult(id="3", score=0.4, payload={}),
            SearchResult(id="4", score=0.4, payload={}),
            SearchResult(id="5", score=0.4, payload={}),
        ]
    )

    orch = AnomalyOrchestrator(
        mock_store,
        knn_threshold=0.8,
        density_threshold=0.5,
        cluster_threshold=0.9,  # requires sim > 0.1 to be normal. distance < 0.9. Here distance is 1-0.4 = 0.6. So normal!
    )

    event = orch.process(np.array([0.1]), "node-1")

    assert event is not None
    assert event.is_anomaly is True
    assert event.anomaly_type == "congestion_alert"
    assert event.source_node_id == "node-1"

    # 2 out of 3 voted anomaly.
    votes = sum(1 for r in event.detector_results if r.is_anomaly)
    assert votes >= 2
