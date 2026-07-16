"""Unit tests for privacy budget tracker.

Tests verify budget tracking, composition, persistence, and alerting.
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.privacy.budget import PrivacyBudgetTracker, PrivacyQuery

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_budget_file():
    """Create a temporary file for budget persistence."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def tracker():
    """Create a standard budget tracker without persistence."""
    return PrivacyBudgetTracker(
        total_budget=10.0,
        alert_threshold=0.8,
        composition="sequential",
    )


@pytest.fixture
def tracker_with_persistence(temp_budget_file):
    """Create a budget tracker with persistence."""
    return PrivacyBudgetTracker(
        total_budget=10.0,
        alert_threshold=0.8,
        composition="sequential",
        persistence_path=temp_budget_file,
    )


# ============================================================================
# Unit Tests — Initialization
# ============================================================================


class TestInitialization:
    """Tests for tracker initialization."""

    def test_valid_initialization(self):
        """Test valid initialization."""
        tracker = PrivacyBudgetTracker(
            total_budget=10.0,
            alert_threshold=0.8,
            composition="sequential",
        )

        assert tracker.total_budget == 10.0
        assert tracker.alert_threshold == 0.8
        assert tracker.composition == "sequential"
        assert tracker.epsilon_spent == 0.0
        assert tracker.max_delta == 0.0
        assert not tracker.alert_triggered
        assert not tracker.budget_exhausted

    def test_invalid_total_budget(self):
        """Test initialization with invalid total budget."""
        with pytest.raises(ValueError, match="total_budget must be positive"):
            PrivacyBudgetTracker(
                total_budget=0.0,
                alert_threshold=0.8,
            )

        with pytest.raises(ValueError, match="total_budget must be positive"):
            PrivacyBudgetTracker(
                total_budget=-1.0,
                alert_threshold=0.8,
            )

    def test_invalid_alert_threshold(self):
        """Test initialization with invalid alert threshold."""
        with pytest.raises(ValueError, match="alert_threshold must be in"):
            PrivacyBudgetTracker(
                total_budget=10.0,
                alert_threshold=0.0,
            )

        with pytest.raises(ValueError, match="alert_threshold must be in"):
            PrivacyBudgetTracker(
                total_budget=10.0,
                alert_threshold=1.0,
            )

    def test_invalid_composition(self):
        """Test initialization with invalid composition."""
        with pytest.raises(ValueError, match="composition must be"):
            PrivacyBudgetTracker(
                total_budget=10.0,
                composition="invalid",
            )


# ============================================================================
# Unit Tests — Query Addition and Budget Tracking
# ============================================================================


class TestQueryAddition:
    """Tests for adding queries and budget tracking."""

    def test_add_single_query(self, tracker):
        """Test adding a single query."""
        accepted = tracker.add_query(
            epsilon=1.0,
            delta=1e-5,
            mechanism="laplace",
            query_type="embedding",
        )

        assert accepted
        assert tracker.epsilon_spent == 1.0
        assert tracker.max_delta == 1e-5
        assert len(tracker.queries) == 1

    def test_add_multiple_queries(self, tracker):
        """Test adding multiple queries."""
        for i in range(5):
            accepted = tracker.add_query(
                epsilon=1.0,
                delta=1e-5,
                mechanism="laplace",
                query_type="embedding",
                metadata={"query_id": i},
            )
            assert accepted

        assert tracker.epsilon_spent == 5.0
        assert len(tracker.queries) == 5

    def test_query_metadata(self, tracker):
        """Test that query metadata is stored."""
        metadata = {"node_id": "edge-1", "timestamp": "2026-07-14"}

        tracker.add_query(
            epsilon=1.0,
            delta=1e-5,
            mechanism="laplace",
            query_type="embedding",
            metadata=metadata,
        )

        query = tracker.queries[0]
        assert query.metadata == metadata

    def test_invalid_epsilon(self, tracker):
        """Test that negative epsilon raises error."""
        with pytest.raises(ValueError, match="epsilon must be non-negative"):
            tracker.add_query(
                epsilon=-1.0,
                delta=1e-5,
                mechanism="laplace",
                query_type="embedding",
            )

    def test_invalid_delta(self, tracker):
        """Test that negative delta raises error."""
        with pytest.raises(ValueError, match="delta must be non-negative"):
            tracker.add_query(
                epsilon=1.0,
                delta=-1.0,
                mechanism="laplace",
                query_type="embedding",
            )


# ============================================================================
# Unit Tests — Sequential Composition
# ============================================================================


class TestSequentialComposition:
    """Tests for sequential composition."""

    def test_sequential_accumulation(self, tracker):
        """Test that epsilon accumulates additively."""
        epsilons = [1.0, 2.0, 1.5, 0.5]

        for eps in epsilons:
            tracker.add_query(
                epsilon=eps,
                delta=1e-5,
                mechanism="laplace",
                query_type="embedding",
            )

        expected_total = sum(epsilons)
        assert tracker.epsilon_spent == expected_total

    def test_sequential_with_varying_epsilon(self, tracker):
        """Test sequential composition with varying epsilon values."""
        # Add queries with different epsilon values
        tracker.add_query(epsilon=0.1, delta=1e-5, mechanism="laplace", query_type="embedding")
        assert tracker.epsilon_spent == 0.1

        tracker.add_query(epsilon=0.5, delta=1e-5, mechanism="laplace", query_type="embedding")
        assert tracker.epsilon_spent == 0.6

        tracker.add_query(epsilon=2.0, delta=1e-5, mechanism="laplace", query_type="embedding")
        assert tracker.epsilon_spent == 2.6


# ============================================================================
# Unit Tests — Parallel Composition
# ============================================================================


class TestParallelComposition:
    """Tests for parallel composition."""

    def test_parallel_maximum(self):
        """Test that epsilon takes maximum (not sum)."""
        tracker = PrivacyBudgetTracker(
            total_budget=10.0,
            composition="parallel",
        )

        # Add queries with different epsilon values
        tracker.add_query(epsilon=1.0, delta=1e-5, mechanism="laplace", query_type="embedding")
        assert tracker.epsilon_spent == 1.0

        tracker.add_query(epsilon=0.5, delta=1e-5, mechanism="laplace", query_type="embedding")
        assert tracker.epsilon_spent == 1.0  # max(1.0, 0.5)

        tracker.add_query(epsilon=2.0, delta=1e-5, mechanism="laplace", query_type="embedding")
        assert tracker.epsilon_spent == 2.0  # max(1.0, 0.5, 2.0)

    def test_parallel_independent_datasets(self):
        """Test parallel composition for independent datasets."""
        tracker = PrivacyBudgetTracker(
            total_budget=10.0,
            composition="parallel",
        )

        # Multiple queries on independent datasets
        for _ in range(10):
            tracker.add_query(epsilon=1.0, delta=1e-5, mechanism="laplace", query_type="embedding")

        # With parallel composition, total epsilon = max(εᵢ) = 1.0
        assert tracker.epsilon_spent == 1.0
        assert len(tracker.queries) == 10


# ============================================================================
# Unit Tests — Budget Ceiling Enforcement
# ============================================================================


class TestBudgetCeilingEnforcement:
    """Tests for budget ceiling enforcement."""

    def test_budget_ceiling_exact(self, tracker):
        """Test that queries at exactly the ceiling are accepted, then budget exhausted."""
        # Use up entire budget
        for _i in range(10):
            accepted = tracker.add_query(
                epsilon=1.0,
                delta=1e-5,
                mechanism="laplace",
                query_type="embedding",
            )
            assert accepted

        assert tracker.epsilon_spent == 10.0

        # Try to add one more - this should be rejected and trigger exhaustion
        accepted = tracker.add_query(
            epsilon=0.1,
            delta=1e-5,
            mechanism="laplace",
            query_type="embedding",
        )

        assert not accepted
        assert tracker.budget_exhausted

    def test_budget_ceiling_exceeded(self, tracker):
        """Test that queries exceeding ceiling are rejected."""
        # Use up 9.0 of 10.0 budget
        for _ in range(9):
            tracker.add_query(
                epsilon=1.0,
                delta=1e-5,
                mechanism="laplace",
                query_type="embedding",
            )

        assert tracker.epsilon_spent == 9.0
        assert not tracker.budget_exhausted

        # Try to add query that would exceed budget
        accepted = tracker.add_query(
            epsilon=2.0,  # 9.0 + 2.0 = 11.0 > 10.0
            delta=1e-5,
            mechanism="laplace",
            query_type="embedding",
        )

        assert not accepted
        assert tracker.epsilon_spent == 9.0  # Should not change
        assert tracker.budget_exhausted

    def test_subsequent_queries_after_exhaustion(self, tracker):
        """Test that all queries are rejected after exhaustion."""
        # Exhaust budget
        for _ in range(10):
            tracker.add_query(epsilon=1.0, delta=1e-5, mechanism="laplace", query_type="embedding")

        # Try to add one more to trigger exhaustion
        accepted = tracker.add_query(
            epsilon=0.1, delta=1e-5, mechanism="laplace", query_type="embedding"
        )
        assert not accepted
        assert tracker.budget_exhausted

        # Try to add more queries
        for _ in range(5):
            accepted = tracker.add_query(
                epsilon=0.1,
                delta=1e-5,
                mechanism="laplace",
                query_type="embedding",
            )
            assert not accepted

        # Budget should not change
        assert tracker.epsilon_spent == 10.0
        assert len(tracker.queries) == 10  # Only first 10 queries


# ============================================================================
# Unit Tests — Alert Threshold
# ============================================================================


class TestAlertThreshold:
    """Tests for alert threshold."""

    def test_alert_at_80_percent(self, tracker):
        """Test that alert triggers at 80% threshold."""
        # total_budget=10.0, alert_threshold=0.8
        # Alert should trigger when epsilon_spent >= 8.0

        # Use 7.9 epsilon (79% - no alert)
        for _ in range(7):
            tracker.add_query(epsilon=1.0, delta=1e-5, mechanism="laplace", query_type="embedding")
        tracker.add_query(epsilon=0.9, delta=1e-5, mechanism="laplace", query_type="embedding")

        assert not tracker.alert_triggered

        # Add one more query to reach 8.0 (80%)
        tracker.add_query(epsilon=0.1, delta=1e-5, mechanism="laplace", query_type="embedding")

        assert tracker.alert_triggered
        assert tracker.epsilon_spent == 8.0

    def test_alert_triggered_once(self, tracker):
        """Test that alert is triggered only once."""
        # Reach 80% threshold
        for _ in range(8):
            tracker.add_query(epsilon=1.0, delta=1e-5, mechanism="laplace", query_type="embedding")

        assert tracker.alert_triggered

        # Add more queries - alert should remain True (not re-triggered)
        tracker.add_query(epsilon=0.5, delta=1e-5, mechanism="laplace", query_type="embedding")

        assert tracker.alert_triggered
        assert tracker.epsilon_spent == 8.5

    def test_custom_alert_threshold(self):
        """Test custom alert threshold."""
        tracker = PrivacyBudgetTracker(
            total_budget=10.0,
            alert_threshold=0.5,  # Alert at 50%
        )

        # Use 4.9 epsilon (49% - no alert)
        for _ in range(4):
            tracker.add_query(epsilon=1.0, delta=1e-5, mechanism="laplace", query_type="embedding")
        tracker.add_query(epsilon=0.9, delta=1e-5, mechanism="laplace", query_type="embedding")

        assert not tracker.alert_triggered

        # Add 0.1 to reach 5.0 (50%)
        tracker.add_query(epsilon=0.1, delta=1e-5, mechanism="laplace", query_type="embedding")

        assert tracker.alert_triggered


# ============================================================================
# Unit Tests — Budget Queries
# ============================================================================


class TestBudgetQueries:
    """Tests for budget query methods."""

    def test_get_remaining_budget(self, tracker):
        """Test get_remaining_budget() method."""
        assert tracker.get_remaining_budget() == 10.0

        tracker.add_query(epsilon=3.0, delta=1e-5, mechanism="laplace", query_type="embedding")
        assert tracker.get_remaining_budget() == 7.0

        tracker.add_query(epsilon=2.5, delta=1e-5, mechanism="laplace", query_type="embedding")
        assert tracker.get_remaining_budget() == 4.5

    def test_get_utilization(self, tracker):
        """Test get_utilization() method."""
        assert tracker.get_utilization() == 0.0

        tracker.add_query(epsilon=5.0, delta=1e-5, mechanism="laplace", query_type="embedding")
        assert tracker.get_utilization() == 0.5  # 5.0 / 10.0

        tracker.add_query(epsilon=3.0, delta=1e-5, mechanism="laplace", query_type="embedding")
        assert tracker.get_utilization() == 0.8  # 8.0 / 10.0

    def test_get_status(self, tracker):
        """Test get_status() method."""
        tracker.add_query(epsilon=6.0, delta=1e-5, mechanism="laplace", query_type="embedding")

        status = tracker.get_status()

        assert status["total_budget"] == 10.0
        assert status["epsilon_spent"] == 6.0
        assert status["remaining_budget"] == 4.0
        assert status["utilization"] == 0.6
        assert status["num_queries"] == 1
        assert not status["alert_triggered"]
        assert not status["budget_exhausted"]
        assert status["composition"] == "sequential"

    def test_get_query_history(self, tracker):
        """Test get_query_history() method."""
        # Add multiple queries
        for i in range(3):
            tracker.add_query(
                epsilon=1.0,
                delta=1e-5,
                mechanism="laplace",
                query_type="embedding",
                metadata={"id": i},
            )

        history = tracker.get_query_history()

        assert len(history) == 3
        assert all(isinstance(q, PrivacyQuery) for q in history)
        assert history[0].metadata["id"] == 0
        assert history[2].metadata["id"] == 2

    def test_max_delta_tracking(self, tracker):
        """Test that max delta is tracked correctly."""
        tracker.add_query(epsilon=1.0, delta=1e-5, mechanism="gaussian", query_type="embedding")
        assert tracker.max_delta == 1e-5

        tracker.add_query(epsilon=1.0, delta=1e-6, mechanism="gaussian", query_type="embedding")
        assert tracker.max_delta == 1e-5  # Should remain at maximum

        tracker.add_query(epsilon=1.0, delta=1e-4, mechanism="gaussian", query_type="embedding")
        assert tracker.max_delta == 1e-4  # Should update to new maximum


# ============================================================================
# Unit Tests — Budget Reset
# ============================================================================


class TestBudgetReset:
    """Tests for budget reset functionality."""

    def test_reset_clears_all_state(self, tracker):
        """Test that reset clears all state."""
        # Add some queries
        for _ in range(5):
            tracker.add_query(epsilon=1.0, delta=1e-5, mechanism="laplace", query_type="embedding")

        assert tracker.epsilon_spent == 5.0
        assert len(tracker.queries) == 5

        # Reset
        tracker.reset()

        assert tracker.epsilon_spent == 0.0
        assert tracker.max_delta == 0.0
        assert len(tracker.queries) == 0
        assert not tracker.alert_triggered
        assert not tracker.budget_exhausted

    def test_reset_allows_new_queries(self, tracker):
        """Test that queries can be added after reset."""
        # Exhaust budget
        for _ in range(10):
            tracker.add_query(epsilon=1.0, delta=1e-5, mechanism="laplace", query_type="embedding")

        # Try to exceed - triggers exhaustion
        accepted = tracker.add_query(
            epsilon=0.1, delta=1e-5, mechanism="laplace", query_type="embedding"
        )
        assert not accepted
        assert tracker.budget_exhausted

        # Reset
        tracker.reset()

        # Should be able to add queries again
        accepted = tracker.add_query(
            epsilon=1.0, delta=1e-5, mechanism="laplace", query_type="embedding"
        )
        assert accepted
        assert tracker.epsilon_spent == 1.0


# ============================================================================
# Unit Tests — Persistence
# ============================================================================


class TestPersistence:
    """Tests for budget state persistence."""

    def test_state_saved_to_file(self, tracker_with_persistence, temp_budget_file):
        """Test that state is saved to file."""
        # Add query
        tracker_with_persistence.add_query(
            epsilon=3.0,
            delta=1e-5,
            mechanism="laplace",
            query_type="embedding",
        )

        # Check file exists and contains data
        assert temp_budget_file.exists()

        with open(temp_budget_file) as f:
            state = json.load(f)

        assert state["epsilon_spent"] == 3.0
        assert len(state["queries"]) == 1

    def test_state_loaded_from_file(self, temp_budget_file):
        """Test that state is loaded from file."""
        # Create tracker and add query
        tracker1 = PrivacyBudgetTracker(
            total_budget=10.0,
            persistence_path=temp_budget_file,
        )

        tracker1.add_query(epsilon=5.0, delta=1e-5, mechanism="laplace", query_type="embedding")
        tracker1.add_query(epsilon=2.0, delta=1e-5, mechanism="laplace", query_type="embedding")

        assert tracker1.epsilon_spent == 7.0
        assert len(tracker1.queries) == 2

        # Create new tracker from same file (simulates restart)
        tracker2 = PrivacyBudgetTracker(
            total_budget=10.0,
            persistence_path=temp_budget_file,
        )

        # State should be recovered
        assert tracker2.epsilon_spent == 7.0
        assert len(tracker2.queries) == 2
        assert tracker2.get_remaining_budget() == 3.0

    def test_persistence_across_multiple_sessions(self, temp_budget_file):
        """Test persistence across multiple sessions."""
        # Session 1
        tracker1 = PrivacyBudgetTracker(total_budget=10.0, persistence_path=temp_budget_file)
        tracker1.add_query(epsilon=2.0, delta=1e-5, mechanism="laplace", query_type="embedding")
        del tracker1

        # Session 2
        tracker2 = PrivacyBudgetTracker(total_budget=10.0, persistence_path=temp_budget_file)
        assert tracker2.epsilon_spent == 2.0
        tracker2.add_query(epsilon=3.0, delta=1e-5, mechanism="laplace", query_type="embedding")
        del tracker2

        # Session 3
        tracker3 = PrivacyBudgetTracker(total_budget=10.0, persistence_path=temp_budget_file)
        assert tracker3.epsilon_spent == 5.0
        assert len(tracker3.queries) == 2

    def test_reset_clears_persistent_state(self, tracker_with_persistence, temp_budget_file):
        """Test that reset clears persistent state."""
        # Add queries
        tracker_with_persistence.add_query(
            epsilon=5.0, delta=1e-5, mechanism="laplace", query_type="embedding"
        )

        # Verify state is saved
        with open(temp_budget_file) as f:
            state = json.load(f)
        assert state["epsilon_spent"] == 5.0

        # Reset
        tracker_with_persistence.reset()

        # Verify state is cleared in file
        with open(temp_budget_file) as f:
            state = json.load(f)
        assert state["epsilon_spent"] == 0.0
        assert len(state["queries"]) == 0


# ============================================================================
# Unit Tests — Thread Safety
# ============================================================================


class TestThreadSafety:
    """Tests for thread safety (basic checks)."""

    def test_concurrent_query_addition(self, tracker):
        """Test that concurrent queries are handled safely."""
        import threading

        num_threads = 10
        queries_per_thread = 10

        def add_queries():
            for _ in range(queries_per_thread):
                tracker.add_query(
                    epsilon=0.01,
                    delta=1e-5,
                    mechanism="laplace",
                    query_type="embedding",
                )

        threads = [threading.Thread(target=add_queries) for _ in range(num_threads)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All queries should be recorded
        expected_epsilon = num_threads * queries_per_thread * 0.01
        assert abs(tracker.epsilon_spent - expected_epsilon) < 0.001
        assert len(tracker.queries) == num_threads * queries_per_thread


# ============================================================================
# Unit Tests — PrivacyQuery Dataclass
# ============================================================================


class TestPrivacyQuery:
    """Tests for PrivacyQuery dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        query = PrivacyQuery(
            timestamp="2026-07-14T12:00:00",
            epsilon=1.0,
            delta=1e-5,
            mechanism="laplace",
            query_type="embedding",
            metadata={"node_id": "edge-1"},
        )

        data = query.to_dict()

        assert data["timestamp"] == "2026-07-14T12:00:00"
        assert data["epsilon"] == 1.0
        assert data["delta"] == 1e-5
        assert data["mechanism"] == "laplace"
        assert data["query_type"] == "embedding"
        assert data["metadata"] == {"node_id": "edge-1"}

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "timestamp": "2026-07-14T12:00:00",
            "epsilon": 1.0,
            "delta": 1e-5,
            "mechanism": "laplace",
            "query_type": "embedding",
            "metadata": {"node_id": "edge-1"},
        }

        query = PrivacyQuery.from_dict(data)

        assert query.timestamp == "2026-07-14T12:00:00"
        assert query.epsilon == 1.0
        assert query.delta == 1e-5
        assert query.mechanism == "laplace"
        assert query.query_type == "embedding"
        assert query.metadata == {"node_id": "edge-1"}

    def test_round_trip(self):
        """Test to_dict -> from_dict round trip."""
        original = PrivacyQuery(
            timestamp="2026-07-14T12:00:00",
            epsilon=2.5,
            delta=1e-6,
            mechanism="gaussian",
            query_type="search",
            metadata={"key": "value"},
        )

        # Convert to dict and back
        reconstructed = PrivacyQuery.from_dict(original.to_dict())

        assert reconstructed.timestamp == original.timestamp
        assert reconstructed.epsilon == original.epsilon
        assert reconstructed.delta == original.delta
        assert reconstructed.mechanism == original.mechanism
        assert reconstructed.query_type == original.query_type
        assert reconstructed.metadata == original.metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
