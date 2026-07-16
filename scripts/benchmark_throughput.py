import json
import math
import os
import time

import numpy as np

from src.anomaly.detector import AnomalyOrchestrator
from src.database.faiss_store import FAISSStore


def benchmark_throughput():
    print("Starting throughput benchmark...")

    # Init DB
    store = FAISSStore("data/bench_tp_faiss.bin", "data/bench_tp_meta.json")
    store.create_collection(dimension=512)
    orchestrator = AnomalyOrchestrator(store)

    duration = 10.0  # seconds per config
    configs = ["video_only", "audio_only", "all_modalities"]

    report = {}

    for config in configs:
        count = 0
        start_time = time.time()

        while time.time() - start_time < duration:
            # Generate dummy payload depending on config
            vec = np.random.rand(512)
            vec /= np.linalg.norm(vec)

            metadata = {
                "timestamp": time.time(),
                "source_node_id": "bench_node",
                "modalities_fused": [config],
                "event_type": "normal",
                "dp_epsilon": 1.0,
            }

            # Simulated Processing
            store.insert(vec, metadata)
            orchestrator.process(vec, "bench_node")

            count += 1

        throughput = count / duration

        # Approximate 95% CI assuming Poisson arrival
        # Standard error of count is sqrt(count)
        # CI = 1.96 * sqrt(count) / duration
        ci = 1.96 * math.sqrt(count) / duration

        report[config] = {
            "payloads_processed": count,
            "duration_sec": duration,
            "throughput_hz": throughput,
            "ci_95": ci,
        }

    store.delete_collection()

    os.makedirs("benchmarks", exist_ok=True)
    with open("benchmarks/throughput_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print("Throughput benchmark complete. Saved to benchmarks/throughput_report.json")
    for k, v in report.items():
        print(f"[{k}] {v['throughput_hz']:.2f} ± {v['ci_95']:.2f} payloads/sec")


if __name__ == "__main__":
    benchmark_throughput()
