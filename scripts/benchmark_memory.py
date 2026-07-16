import json
import os
import time
import tracemalloc

import numpy as np
import psutil

from src.anomaly.detector import AnomalyOrchestrator
from src.database.faiss_store import FAISSStore


def benchmark_memory(num_payloads=10000):
    print(f"Starting memory benchmark with {num_payloads} payloads...")

    process = psutil.Process(os.getpid())
    start_rss = process.memory_info().rss / (1024 * 1024)  # MB

    tracemalloc.start()

    # Init DB
    store = FAISSStore("data/bench_mem_faiss.bin", "data/bench_mem_meta.json")
    store.create_collection(dimension=512)
    orchestrator = AnomalyOrchestrator(store)

    rss_over_time = []

    for i in range(num_payloads):
        # Generate payload
        vec = np.random.rand(512)
        vec /= np.linalg.norm(vec)

        # Process
        metadata = {
            "timestamp": time.time(),
            "source_node_id": "bench_node",
            "modalities_fused": ["video"],
            "event_type": "normal",
            "dp_epsilon": 1.0,
        }
        store.insert(vec, metadata)
        orchestrator.process(vec, "bench_node")

        if i % 1000 == 0:
            current_rss = process.memory_info().rss / (1024 * 1024)
            rss_over_time.append(current_rss)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    end_rss = process.memory_info().rss / (1024 * 1024)
    peak_rss = max(rss_over_time + [end_rss])

    store.delete_collection()

    report = {
        "start_rss_mb": start_rss,
        "end_rss_mb": end_rss,
        "peak_rss_mb": peak_rss,
        "tracemalloc_current_mb": current / (1024 * 1024),
        "tracemalloc_peak_mb": peak / (1024 * 1024),
        "rss_trend": rss_over_time,
        "under_edge_limit_1_5gb": peak_rss < 1500,
    }

    os.makedirs("benchmarks", exist_ok=True)
    with open("benchmarks/memory_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print("Memory benchmark complete. Saved to benchmarks/memory_report.json")
    print(f"Peak RSS: {peak_rss:.2f} MB")
    if report["under_edge_limit_1_5gb"]:
        print("PASS: Stays below 1.5GB edge constraint")
    else:
        print("FAIL: Exceeded 1.5GB edge constraint")


if __name__ == "__main__":
    benchmark_memory(10000)
