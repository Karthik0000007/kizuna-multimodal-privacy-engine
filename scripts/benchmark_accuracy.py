import json
import os

import numpy as np

from src.anomaly.cross_domain import CrossDomainEvaluator
from src.anomaly.detector import AnomalyOrchestrator
from src.database.faiss_store import FAISSStore


def generate_labeled_data(num_samples=100) -> tuple[list[np.ndarray], list[int]]:
    # Create base cluster
    center = np.random.rand(256)
    center /= np.linalg.norm(center)

    embeddings = []
    labels = []

    for _ in range(num_samples):
        # 80% normal, 20% anomaly
        is_anomaly = np.random.rand() < 0.2
        if is_anomaly:
            # Shift away from center
            vec = center + np.random.normal(0.5, 0.2, 256)
            labels.append(1)
        else:
            vec = center + np.random.normal(0.0, 0.1, 256)
            labels.append(0)

        vec /= np.linalg.norm(vec)
        embeddings.append(vec)

    return embeddings, labels


def inject_dp_noise(embeddings: list[np.ndarray], epsilon: float) -> list[np.ndarray]:
    if epsilon == 0 or epsilon > 100:
        return embeddings  # no noise for infinity/high eps

    noisy = []
    # Simplified sensitivity / epsilon calculation
    scale = 1.0 / epsilon
    for vec in embeddings:
        noise = np.random.laplace(0, scale, size=vec.shape)
        noisy_vec = vec + noise
        norm = np.linalg.norm(noisy_vec)
        if norm > 0:
            noisy_vec /= norm
        noisy.append(noisy_vec)
    return noisy


def benchmark_accuracy():
    print("Starting accuracy vs privacy benchmark...")

    epsilons = [100.0, 10.0, 1.0, 0.1, 0.01]
    results = {}

    # Base dataset
    base_embeddings, test_labels = generate_labeled_data(200)

    for eps in epsilons:
        # Create fresh DB
        store = FAISSStore(f"data/bench_acc_{eps}.bin", f"data/bench_acc_{eps}.json")
        store.create_collection(dimension=256)

        # Inject normal baseline data for the orchestrator to know what's normal
        baseline_embeddings, _ = generate_labeled_data(100)
        for vec in baseline_embeddings:
            store.insert(
                vec,
                {"timestamp": 0, "dp_epsilon": eps, "event_type": "normal", "modalities_fused": []},
            )

        orchestrator = AnomalyOrchestrator(
            store, knn_threshold=0.8, density_threshold=0.5, cluster_threshold=0.5
        )
        evaluator = CrossDomainEvaluator(store, orchestrator)

        noisy_embeddings = inject_dp_noise(base_embeddings, eps)

        res = evaluator.evaluate_zero_shot(noisy_embeddings, test_labels)

        results[f"eps_{eps}"] = res
        store.delete_collection()

    os.makedirs("benchmarks", exist_ok=True)
    with open("benchmarks/accuracy_report.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Accuracy benchmark complete. Saved to benchmarks/accuracy_report.json")
    for k, v in results.items():
        print(
            f"[{k}] F1: {v['f1']:.3f} | Recall: {v['recall']:.3f} | Precision: {v['precision']:.3f}"
        )

    try:
        import mlflow

        mlflow.set_experiment("benchmarks")
        with mlflow.start_run(run_name="accuracy_benchmark"):
            for eps_key, metrics in results.items():
                mlflow.log_metric(f"{eps_key}_f1", metrics["f1"])
                mlflow.log_metric(f"{eps_key}_recall", metrics["recall"])
                mlflow.log_metric(f"{eps_key}_precision", metrics["precision"])
    except ImportError:
        print("MLflow not installed; skipping MLflow logging.")


if __name__ == "__main__":
    benchmark_accuracy()
