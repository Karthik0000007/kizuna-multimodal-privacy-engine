import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.database.base import VectorStore


@dataclass
class CacheEntry:
    """A cached value with a TTL."""

    data: Any
    expires_at: float


class DashboardAPI:
    """
    Thin data access layer for the Streamlit dashboard.
    Queries the vector store, reads telemetry, and computes dashboard metrics.
    Implements TTL-based caching and simple rate limiting.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        live_ttl: float = 1.0,
        historical_ttl: float = 60.0,
        max_requests_per_sec: float = 10.0,
    ):
        """
        Args:
            vector_store: The backing vector database.
            live_ttl: Cache TTL in seconds for live data (default 1s).
            historical_ttl: Cache TTL in seconds for historical data (default 60s).
            max_requests_per_sec: Rate limit ceiling.
        """
        self.vector_store = vector_store
        self.live_ttl = live_ttl
        self.historical_ttl = historical_ttl
        self.max_requests_per_sec = max_requests_per_sec

        self._cache: dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self._request_timestamps: list[float] = []

    # ------------------------------------------------------------------
    # Rate Limiting
    # ------------------------------------------------------------------

    def _check_rate_limit(self) -> bool:
        """Returns True if the request is allowed, False if rate-limited."""
        now = time.time()
        window = 1.0  # 1-second sliding window
        with self._lock:
            self._request_timestamps = [t for t in self._request_timestamps if now - t < window]
            if len(self._request_timestamps) >= self.max_requests_per_sec:
                return False
            self._request_timestamps.append(now)
            return True

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def _get_cached(self, key: str) -> Any | None:
        entry = self._cache.get(key)
        if entry and time.time() < entry.expires_at:
            return entry.data
        return None

    def _set_cached(self, key: str, data: Any, ttl: float):
        self._cache[key] = CacheEntry(data=data, expires_at=time.time() + ttl)

    # ------------------------------------------------------------------
    # Live Data Endpoints
    # ------------------------------------------------------------------

    def get_recent_events(self, limit: int = 50) -> list[dict[str, Any]]:
        """Fetch the most recent events from the vector store (live TTL)."""
        if not self._check_rate_limit():
            cached = self._get_cached("recent_events")
            return cached if cached else []

        cached = self._get_cached("recent_events")
        if cached is not None:
            return cached

        # Query vector store for recent points (use a random query vector to get latest)
        query = np.zeros(512, dtype=np.float32)
        results = self.vector_store.search(query, top_k=limit)

        events = []
        for r in results:
            events.append(
                {
                    "id": r.id,
                    "score": r.score,
                    "timestamp": r.payload.get("timestamp", 0),
                    "source_node_id": r.payload.get("source_node_id", "unknown"),
                    "event_type": r.payload.get("event_type", "normal"),
                    "modalities": r.payload.get("modalities_fused", []),
                    "dp_epsilon": r.payload.get("dp_epsilon", 0.0),
                }
            )

        self._set_cached("recent_events", events, self.live_ttl)
        return events

    def get_node_status(self) -> dict[str, dict[str, Any]]:
        """Compute per-node status from recent events (live TTL)."""
        cached = self._get_cached("node_status")
        if cached is not None:
            return cached

        events = self.get_recent_events(limit=200)
        nodes: dict[str, dict[str, Any]] = {}

        for evt in events:
            node_id = evt["source_node_id"]
            if node_id not in nodes:
                nodes[node_id] = {
                    "payload_count": 0,
                    "last_seen": 0,
                    "anomaly_count": 0,
                    "latencies": [],
                }
            nodes[node_id]["payload_count"] += 1
            nodes[node_id]["last_seen"] = max(nodes[node_id]["last_seen"], evt["timestamp"])
            if evt["event_type"] != "normal":
                nodes[node_id]["anomaly_count"] += 1

        self._set_cached("node_status", nodes, self.live_ttl)
        return nodes

    # ------------------------------------------------------------------
    # Historical Data Endpoints
    # ------------------------------------------------------------------

    def get_anomaly_history(
        self,
        event_type: str | None = None,
        source_node: str | None = None,
        min_confidence: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Fetch anomaly history with filters (historical TTL)."""
        cache_key = f"anomaly_history:{event_type}:{source_node}:{min_confidence}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        if not self._check_rate_limit():
            return cached if cached else []

        # Build filters dict
        filters = {}
        if event_type:
            filters["event_type"] = event_type
        if source_node:
            filters["source_node_id"] = source_node

        query = np.zeros(512, dtype=np.float32)
        results = self.vector_store.search(query, top_k=500, filters=filters if filters else None)

        anomalies = []
        for r in results:
            if r.payload.get("event_type", "normal") != "normal":
                anomalies.append(
                    {
                        "id": r.id,
                        "timestamp": r.payload.get("timestamp", 0),
                        "source_node_id": r.payload.get("source_node_id", "unknown"),
                        "event_type": r.payload.get("event_type", "unknown"),
                        "score": r.score,
                        "dp_epsilon": r.payload.get("dp_epsilon", 0.0),
                    }
                )

        self._set_cached(cache_key, anomalies, self.historical_ttl)
        return anomalies

    def get_system_health(self) -> dict[str, Any]:
        """Return system-level health metrics (live TTL)."""
        cached = self._get_cached("system_health")
        if cached is not None:
            return cached

        try:
            collection_info = self.vector_store.get_collection_info()
        except Exception:
            collection_info = {}

        node_status = self.get_node_status()

        health = {
            "collection_info": collection_info,
            "active_nodes": len(node_status),
            "total_payloads": sum(n["payload_count"] for n in node_status.values()),
            "total_anomalies": sum(n["anomaly_count"] for n in node_status.values()),
            "nodes": node_status,
        }

        self._set_cached("system_health", health, self.live_ttl)
        return health

    def get_privacy_budget_status(self) -> dict[str, float]:
        """Return privacy budget tracking (live TTL)."""
        cached = self._get_cached("privacy_budget")
        if cached is not None:
            return cached

        events = self.get_recent_events(limit=500)
        total_epsilon_spent = sum(evt.get("dp_epsilon", 0.0) for evt in events)

        budget = {
            "epsilon_spent": total_epsilon_spent,
            "epsilon_budget": 10.0,  # from config default
            "remaining": max(0, 10.0 - total_epsilon_spent),
            "utilization_pct": min(100.0, (total_epsilon_spent / 10.0) * 100),
        }

        self._set_cached("privacy_budget", budget, self.live_ttl)
        return budget
