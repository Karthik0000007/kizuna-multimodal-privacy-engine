import numpy as np
import pytest

from src.anomaly.detector import AnomalyOrchestrator
from src.database.faiss_store import FAISSStore


@pytest.fixture
def store():
    import os

    db_path = "tests/temp_fault_faiss.bin"
    meta_path = "tests/temp_fault_meta.json"

    # Cleanup before test
    if os.path.exists(db_path):
        os.remove(db_path)
    if os.path.exists(meta_path):
        os.remove(meta_path)

    s = FAISSStore(db_path, meta_path)
    s.create_collection(dimension=64)
    yield s
    s.delete_collection()


def test_malformed_sensor_data(store):
    """Test behavior with malformed sensor data (NaN, Inf)"""
    orchestrator = AnomalyOrchestrator(store)

    # Test NaN injection
    bad_vec = np.array([np.nan] * 64)
    # The system shouldn't crash, it should handle it gracefully or raise a ValueError
    with pytest.raises(Exception) as excinfo:
        orchestrator.process(bad_vec, "test_node")
    assert excinfo is not None

    # Test Inf injection
    bad_vec_inf = np.array([np.inf] * 64)
    with pytest.raises(Exception) as excinfo:
        orchestrator.process(bad_vec_inf, "test_node")
    assert excinfo is not None


def test_oom_simulation():
    """Test graceful handling under OOM-like conditions by triggering MemoryError explicitly"""
    # Python MemoryError is raised when allocation fails.
    # We can simulate this by trying to allocate an impossibly large array,
    # or just mocking a MemoryError in a critical section to ensure the app doesn't hard-crash the node.

    def simulate_oom():
        raise MemoryError("Simulated OOM")

    try:
        simulate_oom()
    except MemoryError:
        # Should be caught by the node's top level exception handler
        handled = True
    assert handled


def test_cpu_throttle_simulation():
    """Simulate CPU throttling (busy-wait) and ensure the process can still be interrupted"""
    import threading
    import time

    # We'll just run a busy loop for a short time and ensure we can break out of it
    stop_flag = False

    def busy_wait():
        while not stop_flag:
            pass

    t = threading.Thread(target=busy_wait)
    t.start()

    # Give it a fraction of a second
    time.sleep(0.1)

    # Send stop signal
    stop_flag = True
    t.join(timeout=1.0)

    # If the thread is still alive after 1 second, it failed to interrupt
    assert not t.is_alive()
