import os
import shutil

import numpy as np
import pytest

from src.anomaly.cross_domain import CrossDomainEvaluator
from src.anomaly.detector import AnomalyOrchestrator
from src.anomaly.enrollment import AnomalyEnroller
from src.database.faiss_store import FAISSStore


@pytest.fixture
def faiss_store():
    store_dir = "tests/temp_cross_domain"
    os.makedirs(store_dir, exist_ok=True)
    store = FAISSStore(
        index_path=f"{store_dir}/index.bin", metadata_path=f"{store_dir}/metadata.json"
    )
    store.create_collection(dimension=64)
    yield store
    store.delete_collection()
    if os.path.exists(store_dir):
        shutil.rmtree(store_dir)


def test_anomaly_enroller(faiss_store):
    enroller = AnomalyEnroller(faiss_store)

    # Generate 5 examples of an anomaly
    examples = [np.random.rand(64) for _ in range(5)]

    point_id = enroller.enroll(examples, label="test_anomaly")
    assert point_id is not None

    # The prototype should be inserted as a centroid
    # Let's search for it
    query = np.mean(examples, axis=0)
    query = query / np.linalg.norm(query)

    results = faiss_store.search(query, top_k=1, filters={"is_centroid": True})
    assert len(results) == 1
    assert results[0].payload["event_type"] == "test_anomaly"
    assert "decision_boundary" in results[0].payload
    assert results[0].payload["is_exemplar"] is True


def test_cross_domain_evaluator(faiss_store):
    orchestrator = AnomalyOrchestrator(faiss_store)
    evaluator = CrossDomainEvaluator(faiss_store, orchestrator)

    # Generate some dummy test data
    test_embeddings = [np.random.rand(64) for _ in range(10)]
    test_labels = [0] * 5 + [1] * 5

    # Test zero-shot
    zs_results = evaluator.evaluate_zero_shot(test_embeddings, test_labels)
    assert "precision" in zs_results
    assert "recall" in zs_results
    assert "f1" in zs_results

    # Test few-shot
    enrollment_embeddings = [np.random.rand(64) for _ in range(3)]
    fs_results = evaluator.evaluate_few_shot(
        enrollment_embeddings, "dummy_anomaly", test_embeddings, test_labels
    )

    assert "precision" in fs_results
    assert "recall" in fs_results
    assert "f1" in fs_results
