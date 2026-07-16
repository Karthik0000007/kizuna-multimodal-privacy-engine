import json
import os
import time

import numpy as np

from src.anomaly.detector import AnomalyOrchestrator
from src.database.faiss_store import FAISSStore


def benchmark_latency(num_payloads=1000):
    print(f"Starting latency benchmark with {num_payloads} payloads...")

    # Init DB
    store = FAISSStore("data/bench_faiss.bin", "data/bench_meta.json")
    store.create_collection(dimension=512)
    orchestrator = AnomalyOrchestrator(store)

    # Store times for each stage
    latencies = {
        "preprocessing": [],
        "vision_encoding": [],
        "audio_encoding": [],
        "sensor_encoding": [],
        "fusion": [],
        "dp_noise": [],
        "db_insert": [],
        "anomaly_detection": [],
        "total": [],
    }

    for _ in range(num_payloads):
        total_start = time.perf_counter_ns()

        # 1. Preprocessing (Simulated)
        t0 = time.perf_counter_ns()
        # simulate
        t1 = time.perf_counter_ns()
        latencies["preprocessing"].append(t1 - t0)

        # 2. Vision Encoding
        t0 = time.perf_counter_ns()
        vision_vec = np.random.rand(256)
        t1 = time.perf_counter_ns()
        latencies["vision_encoding"].append(t1 - t0)

        # 3. Audio Encoding
        t0 = time.perf_counter_ns()
        audio_vec = np.random.rand(128)
        t1 = time.perf_counter_ns()
        latencies["audio_encoding"].append(t1 - t0)

        # 4. Sensor Encoding
        t0 = time.perf_counter_ns()
        sensor_vec = np.random.rand(128)
        t1 = time.perf_counter_ns()
        latencies["sensor_encoding"].append(t1 - t0)

        # 5. Fusion
        t0 = time.perf_counter_ns()
        fused = np.concatenate([vision_vec, audio_vec, sensor_vec])
        fused /= np.linalg.norm(fused)
        t1 = time.perf_counter_ns()
        latencies["fusion"].append(t1 - t0)

        # 6. DP Noise
        t0 = time.perf_counter_ns()
        noise = np.random.laplace(0, 0.01, size=512)
        fused += noise
        t1 = time.perf_counter_ns()
        latencies["dp_noise"].append(t1 - t0)

        # 7. DB Insert
        t0 = time.perf_counter_ns()
        metadata = {
            "timestamp": time.time(),
            "source_node_id": "bench_node",
            "modalities_fused": ["video", "audio", "sensor"],
            "event_type": "normal",
            "dp_epsilon": 1.0,
        }
        store.insert(fused, metadata)
        t1 = time.perf_counter_ns()
        latencies["db_insert"].append(t1 - t0)

        # 8. Anomaly Detection
        t0 = time.perf_counter_ns()
        orchestrator.process(fused, "bench_node")
        t1 = time.perf_counter_ns()
        latencies["anomaly_detection"].append(t1 - t0)

        total_end = time.perf_counter_ns()
        latencies["total"].append(total_end - total_start)

    store.delete_collection()

    # Calculate stats
    report = {}
    for stage, times in latencies.items():
        # Convert nanoseconds to milliseconds
        times_ms = np.array(times) / 1_000_000
        report[stage] = {
            "p50": float(np.percentile(times_ms, 50)),
            "p95": float(np.percentile(times_ms, 95)),
            "p99": float(np.percentile(times_ms, 99)),
            "mean": float(np.mean(times_ms)),
            "std": float(np.std(times_ms)),
        }

    os.makedirs("benchmarks", exist_ok=True)
    with open("benchmarks/latency_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print("Latency benchmark complete. Saved to benchmarks/latency_report.json")

    # Try logging to MLflow if available
    try:
        import mlflow

        mlflow.set_experiment("benchmarks")
        with mlflow.start_run(run_name="latency_benchmark"):
            for stage, metrics in report.items():
                mlflow.log_metric(f"{stage}_p50", metrics["p50"])
                mlflow.log_metric(f"{stage}_p95", metrics["p95"])
                mlflow.log_metric(f"{stage}_p99", metrics["p99"])
                mlflow.log_metric(f"{stage}_mean", metrics["mean"])
    except ImportError:
        print("MLflow not installed; skipping MLflow logging.")
    print(f"Total Mean Latency: {report['total']['mean']:.2f} ms")


if __name__ == "__main__":
    benchmark_latency(1000)
