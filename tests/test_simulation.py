import json
import threading
import time
import urllib.request

import numpy as np
import pytest

from src.central_node import main as central_main


@pytest.fixture
def central_server():
    # Setup mock FAISS for central node instead of Qdrant
    import os

    os.environ["QDRANT_HOST"] = "invalid_host"  # Force FAISS fallback
    os.environ["BIND_PORT"] = "8081"

    server_thread = threading.Thread(target=central_main, daemon=True)
    server_thread.start()

    # Wait for server to start
    url = "http://127.0.0.1:8081"
    for _ in range(30):
        try:
            req = urllib.request.Request(f"{url}/health")
            with urllib.request.urlopen(req, timeout=1) as response:
                if response.status == 200:
                    break
        except Exception:
            time.sleep(0.5)

    yield url


def test_edge_to_central_transmission(central_server):
    # Simulate what edge_node does
    embedding = np.random.rand(512)
    embedding /= np.linalg.norm(embedding)

    payload = {
        "vector": embedding.tolist(),
        "metadata": {
            "timestamp": time.time(),
            "source_node_id": "test-edge",
            "modalities_fused": ["video"],
            "event_type": "normal",
            "dp_epsilon": 1.0,
        },
    }

    req = urllib.request.Request(
        f"{central_server}/ingest",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req, timeout=5) as response:
        assert response.status == 200
        data = json.loads(response.read().decode())
        assert data["status"] == "processed"


def test_central_health(central_server):
    req = urllib.request.Request(f"{central_server}/health")
    with urllib.request.urlopen(req, timeout=5) as response:
        assert response.status == 200
        data = json.loads(response.read().decode())
        assert data["status"] == "ok"
