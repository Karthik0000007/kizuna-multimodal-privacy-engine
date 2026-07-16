import time

import numpy as np
import pytest

from src.dashboard.api import CacheEntry, DashboardAPI
from src.dashboard.live_state import LiveEvent, LiveStateManager, get_live_state
from src.dashboard.projection import ProjectionService
from src.database.base import SearchResult, VectorStore


class MockVectorStore(VectorStore):
    """Mock vector store for dashboard API testing."""

    def __init__(self, mock_results=None):
        self.mock_results = mock_results or []
        self._inserted = []

    def insert(self, vector, metadata):
        self._inserted.append((vector, metadata))
        return f"mock-{len(self._inserted)}"

    def search(self, query, top_k=10, filters=None):
        return self.mock_results[:top_k]

    def create_collection(self, dimension, distance="Cosine"):
        pass

    def delete_collection(self):
        pass

    def get_collection_info(self):
        return {"vectors_count": len(self.mock_results), "status": "green"}


# ======================================================================
# DashboardAPI Tests
# ======================================================================


class TestDashboardAPI:
    def setup_method(self):
        self.mock_results = [
            SearchResult(
                id=f"pt-{i}",
                score=0.9 - i * 0.01,
                payload={
                    "timestamp": time.time() - i,
                    "source_node_id": f"edge-node-{(i % 3) + 1}",
                    "event_type": "normal" if i % 5 != 0 else "fall_risk",
                    "modalities_fused": ["video", "audio"],
                    "dp_epsilon": 1.0,
                },
            )
            for i in range(50)
        ]
        self.store = MockVectorStore(self.mock_results)
        self.api = DashboardAPI(self.store, live_ttl=1.0, historical_ttl=60.0)

    def test_get_recent_events(self):
        events = self.api.get_recent_events(limit=10)
        assert len(events) == 10
        assert "id" in events[0]
        assert "event_type" in events[0]
        assert "source_node_id" in events[0]

    def test_get_recent_events_cached(self):
        events1 = self.api.get_recent_events()
        events2 = self.api.get_recent_events()
        # Should return same cached data
        assert events1 == events2

    def test_get_node_status(self):
        status = self.api.get_node_status()
        assert isinstance(status, dict)
        # Should have up to 3 nodes
        assert len(status) <= 3
        for node_id, info in status.items():
            assert "payload_count" in info
            assert "anomaly_count" in info

    def test_get_anomaly_history(self):
        anomalies = self.api.get_anomaly_history()
        # Every 5th entry is an anomaly
        assert len(anomalies) > 0
        for a in anomalies:
            assert a["event_type"] != "normal"

    def test_get_system_health(self):
        health = self.api.get_system_health()
        assert "collection_info" in health
        assert "active_nodes" in health
        assert "total_payloads" in health

    def test_get_privacy_budget(self):
        budget = self.api.get_privacy_budget_status()
        assert "epsilon_spent" in budget
        assert "remaining" in budget
        assert budget["remaining"] >= 0

    def test_rate_limiting(self):
        api = DashboardAPI(self.store, max_requests_per_sec=2)
        # Exhaust rate limit
        for _ in range(5):
            api.get_recent_events()
        # Should still work (returns cached or empty, doesn't crash)
        result = api.get_recent_events()
        assert isinstance(result, list)


# ======================================================================
# LiveStateManager Tests
# ======================================================================


class TestLiveStateManager:
    def test_push_and_retrieve_events(self):
        mgr = LiveStateManager(max_buffer_size=10)
        for i in range(5):
            mgr.push_event(
                LiveEvent(timestamp=time.time(), source_node_id=f"node-{i}", event_type="normal")
            )

        events = mgr.get_recent_events(limit=5)
        assert len(events) == 5

    def test_buffer_overflow(self):
        mgr = LiveStateManager(max_buffer_size=5)
        for i in range(10):
            mgr.push_event(
                LiveEvent(timestamp=time.time(), source_node_id="node-1", event_type="normal")
            )

        events = mgr.get_recent_events(limit=100)
        assert len(events) == 5  # capped at max_buffer_size

    def test_active_alerts(self):
        mgr = LiveStateManager()
        mgr.push_event(
            LiveEvent(
                timestamp=time.time(),
                source_node_id="node-1",
                event_type="fall_risk",
                confidence=0.95,
            )
        )
        mgr.push_event(
            LiveEvent(timestamp=time.time(), source_node_id="node-2", event_type="normal")
        )

        alerts = mgr.get_active_alerts()
        assert len(alerts) == 1
        assert alerts[0]["event_type"] == "fall_risk"

    def test_should_refresh(self):
        mgr = LiveStateManager()
        old_time = time.time() - 10
        assert not mgr.should_refresh(old_time)  # no events yet, _last_update is 0

        mgr.push_event(
            LiveEvent(timestamp=time.time(), source_node_id="node-1", event_type="normal")
        )
        assert mgr.should_refresh(old_time)

    def test_latency_sparkline(self):
        mgr = LiveStateManager()
        base = time.time()
        for i in range(10):
            mgr.push_event(
                LiveEvent(timestamp=base + i * 0.05, source_node_id="node-1", event_type="normal")
            )

        sparkline = mgr.get_latency_sparkline("node-1")
        assert len(sparkline) == 9  # 10 events produce 9 deltas


# ======================================================================
# ProjectionService Tests
# ======================================================================


class TestProjectionService:
    def test_empty_store(self):
        store = MockVectorStore([])
        proj = ProjectionService(store)
        result = proj.compute_projection()
        assert result["num_points"] == 0

    def test_cache_invalidation(self):
        store = MockVectorStore(
            [
                SearchResult(id=f"p-{i}", score=0.5, payload={"event_type": "normal"})
                for i in range(10)
            ]
        )
        proj = ProjectionService(store, cache_ttl=300)
        proj.compute_projection()
        proj.invalidate_cache()
        assert proj._cached_projection is None

    def test_incremental_point(self):
        store = MockVectorStore(
            [
                SearchResult(id=f"p-{i}", score=0.5, payload={"event_type": "normal"})
                for i in range(10)
            ]
        )
        proj = ProjectionService(store, cache_ttl=300)
        result = proj.compute_projection()

        initial_points = result["num_points"]
        if initial_points > 0:
            added = proj.add_incremental_point("new-1", "fall_risk", "node-1")
            assert added is True
            assert proj._cached_projection["num_points"] == initial_points + 1
