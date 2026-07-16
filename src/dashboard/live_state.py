import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class LiveEvent:
    """A single live event pushed to the dashboard."""

    timestamp: float
    source_node_id: str
    event_type: str
    confidence: float = 0.0
    anomaly_type: Optional[str] = None


class LiveStateManager:
    """
    Manages real-time state for the Streamlit dashboard.
    Uses an in-memory event buffer compatible with Streamlit's st.session_state
    and st.rerun() for live data refresh without WebSockets.
    """

    def __init__(self, max_buffer_size: int = 200):
        self.max_buffer_size = max_buffer_size
        self._events: List[LiveEvent] = []
        self._alerts: List[LiveEvent] = []
        self._lock = threading.Lock()
        self._subscribers: List[Callable[[LiveEvent], None]] = []
        self._last_update: float = 0.0

    def push_event(self, event: LiveEvent):
        """Push a new event into the live buffer."""
        with self._lock:
            self._events.append(event)
            if len(self._events) > self.max_buffer_size:
                self._events = self._events[-self.max_buffer_size :]

            if event.event_type != "normal":
                self._alerts.append(event)
                if len(self._alerts) > 50:
                    self._alerts = self._alerts[-50:]

            self._last_update = time.time()

        # Notify subscribers
        for callback in self._subscribers:
            try:
                callback(event)
            except Exception:
                pass

    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return the most recent events as dicts for Streamlit rendering."""
        with self._lock:
            events = self._events[-limit:]

        return [
            {
                "timestamp": e.timestamp,
                "source_node_id": e.source_node_id,
                "event_type": e.event_type,
                "confidence": e.confidence,
                "anomaly_type": e.anomaly_type,
            }
            for e in reversed(events)
        ]

    def get_active_alerts(self, max_age_sec: float = 60.0) -> List[Dict[str, Any]]:
        """Return active alerts (anomalies) within the last max_age_sec seconds."""
        cutoff = time.time() - max_age_sec
        with self._lock:
            active = [a for a in self._alerts if a.timestamp > cutoff]

        return [
            {
                "timestamp": a.timestamp,
                "source_node_id": a.source_node_id,
                "event_type": a.event_type,
                "confidence": a.confidence,
                "anomaly_type": a.anomaly_type,
            }
            for a in reversed(active)
        ]

    def get_latency_sparkline(self, node_id: str, window: int = 100) -> List[float]:
        """
        Generate a latency sparkline for a given node.
        In production this would track actual processing times;
        here we derive it from event inter-arrival times as a proxy.
        """
        with self._lock:
            node_events = [e for e in self._events if e.source_node_id == node_id]

        if len(node_events) < 2:
            return []

        node_events = node_events[-window:]
        deltas = []
        for i in range(1, len(node_events)):
            delta_ms = (node_events[i].timestamp - node_events[i - 1].timestamp) * 1000
            deltas.append(delta_ms)

        return deltas

    def get_last_update_time(self) -> float:
        return self._last_update

    def subscribe(self, callback: Callable[[LiveEvent], None]):
        """Register a callback for new events."""
        self._subscribers.append(callback)

    def should_refresh(self, last_seen: float) -> bool:
        """Check if there's new data since `last_seen` timestamp."""
        return self._last_update > last_seen


# Singleton instance for use across Streamlit pages
_global_state: Optional[LiveStateManager] = None


def get_live_state() -> LiveStateManager:
    """Get or create the global LiveStateManager singleton."""
    global _global_state
    if _global_state is None:
        _global_state = LiveStateManager()
    return _global_state
