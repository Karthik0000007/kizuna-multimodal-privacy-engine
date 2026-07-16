import json
import os

import numpy as np

from src.anomaly.cross_domain import CrossDomainEvaluator
from src.anomaly.detector import AnomalyOrchestrator
from src.anomaly.enrollment import AnomalyEnroller
from src.database.faiss_store import FAISSStore


def generate_synthetic_data(
    num_samples: int, dimension: int, center: np.ndarray, spread: float = 0.1
) -> list[np.ndarray]:
    data = []
    for _ in range(num_samples):
        vec = center + np.random.normal(0, spread, dimension)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        data.append(vec)
    return data


def main():
    print("Running cross-domain benchmarking...")

    # 1. Setup in-memory vector store
    store = FAISSStore(
        index_path="data/test_faiss_index.bin", metadata_path="data/test_faiss_metadata.json"
    )
    store.create_collection(dimension=512)

    # 2. Setup orchestrator
    # For testing, we use thresholds that are permissive for testing
    orchestrator = AnomalyOrchestrator(
        store, knn_threshold=0.8, density_threshold=0.5, cluster_threshold=0.5
    )

    evaluator = CrossDomainEvaluator(store, orchestrator)
    enroller = AnomalyEnroller(store)

    # Base centers for domains
    domain_a_center = np.random.rand(512)
    domain_a_center /= np.linalg.norm(domain_a_center)

    domain_b_center = np.random.rand(512)
    domain_b_center /= np.linalg.norm(domain_b_center)

    # We first simulate some historical data in Domain A (normal points)
    domain_a_normal = generate_synthetic_data(100, 512, domain_a_center, spread=0.1)
    for vec in domain_a_normal:
        store.insert(
            vec,
            {
                "timestamp": 0.0,
                "source_node_id": "domain_a_node",
                "modalities_fused": ["video"],
                "event_type": "normal",
                "dp_epsilon": 0.0,
            },
        )

    # Domain A anomalies (e.g., elderly falls)
    domain_a_anom_center = domain_a_center + np.random.normal(0, 0.5, 512)
    domain_a_anom_center /= np.linalg.norm(domain_a_anom_center)
    domain_a_anomalies = generate_synthetic_data(10, 512, domain_a_anom_center, spread=0.05)
    enroller.enroll(domain_a_anomalies, "elderly_fall")

    # 3. Simulate Domain B (e.g., crowd incidents)
    domain_b_normal = generate_synthetic_data(200, 512, domain_b_center, spread=0.1)
    domain_b_anom_center = domain_b_center + np.random.normal(0, 0.5, 512)
    domain_b_anom_center /= np.linalg.norm(domain_b_anom_center)
    domain_b_anomalies = generate_synthetic_data(50, 512, domain_b_anom_center, spread=0.05)

    test_embeddings = domain_b_normal + domain_b_anomalies
    test_labels = [0] * len(domain_b_normal) + [1] * len(domain_b_anomalies)

    # Shuffle
    combined = list(zip(test_embeddings, test_labels, strict=False))
    np.random.shuffle(combined)
    test_embeddings, test_labels = zip(*combined, strict=False)

    # A. Zero-shot transfer (Evaluating on Domain B without any Domain B knowledge)
    zero_shot_results = evaluator.evaluate_zero_shot(test_embeddings, test_labels)

    # B. Few-shot transfer (Enroll 10 examples from Domain B)
    domain_b_few_shot = generate_synthetic_data(10, 512, domain_b_anom_center, spread=0.05)
    few_shot_results = evaluator.evaluate_few_shot(
        domain_b_few_shot, "crowd_incident", test_embeddings, test_labels
    )

    # Clean up
    store.delete_collection()

    results = {
        "domain_transfer": "elderly_falls -> crowd_incidents",
        "zero_shot": zero_shot_results,
        "few_shot_10": few_shot_results,
    }

    os.makedirs("benchmarks", exist_ok=True)
    with open("benchmarks/cross_domain_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Benchmarking complete. Results saved to benchmarks/cross_domain_results.json")
    print("Zero-shot F1:", zero_shot_results["f1"])
    print("Few-shot 10 F1:", few_shot_results["f1"])


if __name__ == "__main__":
    main()
