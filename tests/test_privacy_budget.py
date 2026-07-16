"""Unit tests for privacy budget tracker."""

import json
import tempfile
from pathlib import Path

import pytest

from src.privacy import PrivacyBudgetTracker, PrivacyQuery


class TestPrivacyQuery:
    """Test suite for PrivacyQuery dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        query = PrivacyQuery(
            timestamp="2026-07-14T10:00:00",
            epsilon=1.0,
            delta=1e-5,
            mechanism="laplace",
            query_type="embedding",
            metadata={"query_id": 1},
        )

        d = query.to_dict()

        assert d["timestamp"] == "2026-07-14T10:00:00"
        assert d["epsilon"] == 1.0
        assert d["delta"] == 1e-5
        assert d["mechanism"] == "laplace"
        assert d["query_type"] == "embedding"
        assert d["metadata"] == {"query_id": 1}

    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {
            "timestamp": "2026-07-14T10:00:00",
            "epsilon": 1.0,
            "delta": 1e-5,
            "mechanism": "laplace",
            "query_type": "embedding",
            "metadata": {"query_id": 1},
        }

        query = PrivacyQuery.from_dict(d)

        assert query.timestamp == "2026-07-14T10:00:00"
        assert query.epsilon == 1.0
        assert query.delta == 1e-5
        assert query.mechanism == "laplace"
        assert query.query_type == "embedding"
        assert query.metadata == {"query_id": 1}


class TestPrivacyBudgetTracker:
    """Test suite for PrivacyBudgetTracker."""

    def test_init_success(self):
        """Test successful initialization."""
        tracker = PrivacyBudgetTracker(
            total_budget=10.0,
            alert_threshold=0.8,
            composition="sequential",
        )

        assert tracker.total_budget == 10.0
        assert tracker.alert_threshold == 0.8
        assert tracker.composition == "sequential"
        assert tracker.epsilon_spent == 0.0
        assert len(tracker.queries) == 0

    def test_init_invalid_total_budget(self):
        """Test initialization fails with invalid total budget."""
        with pytest.raises(ValueError, match="total_budget must be positive"):
            PrivacyBudgetTracker(total_budget=0.0)

        with pytest.raises(ValueError, match="total_budget must be positive"):
            PrivacyBudgetTracker(total_budget=-1.0)

    def test_init_invalid_alert_threshold(self):
        """Test initialization fails with invalid alert threshold."""
        with pytest.raises(ValueError, match="alert_threshold must be in"):
            PrivacyBudgetTracker(total_budget=10.0, alert_threshold=0.0)

        with pytest.raises(ValueError, match="alert_threshold must be in"):
            PrivacyBudgetTracker(total_budget=10.0, alert_threshold=1.0)

    def test_init_invalid_composition(self):
        """Test initialization fails with invalid composition."""
        with pytest.raises(ValueError, match="composition must be"):
            PrivacyBudgetTracker(total_budget=10.0, composition="invalid")

    def test_add_query_success(self):
        """Test adding query successfully."""
        tracker = PrivacyBudgetTracker(total_budget=10.0)

        result = tracker.add_query(
            epsilon=1.0,
            delta=1e-5,
            mechanism="laplace",
            query_type="embedding",
        )

        assert result is True
        assert tracker.epsilon_spent == 1.0
        assert len(tracker.queries) == 1

    def test_add_query_negative_epsilon(self):
        """Test adding query fails with negative epsilon."""
        tracker = PrivacyBudgetTracker(total_budget=10.0)

        with pytest.raises(ValueError, match="epsilon must be non-negative"):
            tracker.add_query(
                epsilon=-1.0,
                delta=1e-5,
                mechanism="laplace",
                query_type="embedding",
            )

    def test_add_query_negative_delta(self):
        """Test adding query fails with negative delta."""
        tracker = PrivacyBudgetTracker(total_budget=10.0)

        with pytest.raises(ValueError, match="delta must be non-negative"):
            tracker.add_query(
                epsilon=1.0,
                delta=-1e-5,
                mechanism="laplace",
                query_type="embedding",
            )

    def test_add_query_sequential_composition(self):
        """Test sequential composition sums epsilon."""
        tracker = PrivacyBudgetTracker(
            total_budget=10.0,
            composition="sequential",
        )

        tracker.add_query(epsilon=1.0, delta=1e-5, mechanism="laplace", query_type="test")
        tracker.add_query(epsilon=2.0, delta=1e-5, mechanism="laplace", query_type="test")
        tracker.add_query(epsilon=3.0, delta=1e-5, mechanism="laplace", query_type="test")

        # Sequential: total = sum
        assert tracker.epsilon_spent == 6.0

    def test_add_query_parallel_composition(self):
        """Test parallel composition takes maximum epsilon."""
        tracker = PrivacyBudgetTracker(
            total_budget=10.0,
            composition="parallel",
        )

        tracker.add_query(epsilon=1.0, delta=1e-5, mechanism="laplace", query_type="test")
        tracker.add_query(epsilon=3.0, delta=1e-5, mechanism="laplace", query_type="test")
        tracker.add_query(epsilon=2.0, delta=1e-5, mechanism="laplace", query_type="test")

        # Parallel: total = max
        assert tracker.epsilon_spent == 3.0

    def test_add_query_exceeds_budget(self):
        """Test query rejection when exceeding budget."""
        tracker = PrivacyBudgetTracker(total_budget=10.0)

        # Use up 9.0
        tracker.add_query(epsilon=9.0, delta=1e-5, mechanism="laplace", query_type="test")

        # Try to add 2.0 (would exceed 10.0)
        result = tracker.add_query(epsilon=2.0, delta=1e-5, mechanism="laplace", query_type="test")

        assert result is False
        assert tracker.epsilon_spent == 9.0  # Unchanged
        assert len(tracker.queries) == 1  # Only first query recorded
        assert tracker.budget_exhausted is True

    def test_add_query_after_budget_exhausted(self):
        """Test query rejection after budget exhausted."""
        tracker = PrivacyBudgetTracker(total_budget=10.0)

        # Exhaust budget
        tracker.add_query(epsilon=11.0, delta=1e-5, mechanism="laplace", query_type="test")

        # Try to add another (should be rejected immediately)
        result = tracker.add_query(epsilon=0.1, delta=1e-5, mechanism="laplace", query_type="test")

        assert result is False
        assert len(tracker.queries) == 0  # No queries accepted

    def test_alert_threshold(self):
        """Test alert triggers at threshold."""
        tracker = PrivacyBudgetTracker(
            total_budget=10.0,
            alert_threshold=0.8,  # 80% = 8.0
        )

        assert tracker.alert_triggered is False

        # Use 7.0 (70% - below threshold)
        tracker.add_query(epsilon=7.0, delta=1e-5, mechanism="laplace", query_type="test")
        assert tracker.alert_triggered is False

        # Use 1.5 more (85% - above threshold)
        tracker.add_query(epsilon=1.5, delta=1e-5, mechanism="laplace", query_type="test")
        assert tracker.alert_triggered is True

    def test_get_remaining_budget(self):
        """Test get_remaining_budget calculation."""
        tracker = PrivacyBudgetTracker(total_budget=10.0)

        assert tracker.get_remaining_budget() == 10.0

        tracker.add_query(epsilon=3.0, delta=1e-5, mechanism="laplace", query_type="test")
        assert tracker.get_remaining_budget() == 7.0

        tracker.add_query(epsilon=2.0, delta=1e-5, mechanism="laplace", query_type="test")
        assert tracker.get_remaining_budget() == 5.0

    def test_get_utilization(self):
        """Test get_utilization calculation."""
        tracker = PrivacyBudgetTracker(total_budget=10.0)

        assert tracker.get_utilization() == 0.0

        tracker.add_query(epsilon=5.0, delta=1e-5, mechanism="laplace", query_type="test")
        assert tracker.get_utilization() == 0.5

        tracker.add_query(epsilon=3.0, delta=1e-5, mechanism="laplace", query_type="test")
        assert tracker.get_utilization() == 0.8

    def test_get_status(self):
        """Test get_status returns complete status."""
        tracker = PrivacyBudgetTracker(total_budget=10.0)

        tracker.add_query(epsilon=3.0, delta=1e-5, mechanism="laplace", query_type="test")

        status = tracker.get_status()

        assert status["total_budget"] == 10.0
        assert status["epsilon_spent"] == 3.0
        assert status["remaining_budget"] == 7.0
        assert status["utilization"] == 0.3
        assert status["num_queries"] == 1
        assert "alert_triggered" in status
        assert "budget_exhausted" in status

    def test_get_query_history(self):
        """Test get_query_history returns copy."""
        tracker = PrivacyBudgetTracker(total_budget=10.0)

        tracker.add_query(epsilon=1.0, delta=1e-5, mechanism="laplace", query_type="test1")
        tracker.add_query(epsilon=2.0, delta=1e-5, mechanism="laplace", query_type="test2")

        history = tracker.get_query_history()

        assert len(history) == 2
        assert history[0].epsilon == 1.0
        assert history[1].epsilon == 2.0

        # Verify it's a copy (modifying doesn't affect tracker)
        history.clear()
        assert len(tracker.queries) == 2

    def test_reset(self):
        """Test reset clears all state."""
        tracker = PrivacyBudgetTracker(total_budget=10.0)

        tracker.add_query(epsilon=5.0, delta=1e-5, mechanism="laplace", query_type="test")
        tracker.add_query(epsilon=3.0, delta=1e-5, mechanism="laplace", query_type="test")

        assert tracker.epsilon_spent == 8.0
        assert len(tracker.queries) == 2

        tracker.reset()

        assert tracker.epsilon_spent == 0.0
        assert len(tracker.queries) == 0
        assert tracker.alert_triggered is False
        assert tracker.budget_exhausted is False

    def test_persistence(self):
        """Test persistence to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence_path = Path(tmpdir) / "budget.json"

            # Create tracker and add queries
            tracker1 = PrivacyBudgetTracker(
                total_budget=10.0,
                persistence_path=persistence_path,
            )

            tracker1.add_query(epsilon=3.0, delta=1e-5, mechanism="laplace", query_type="test1")
            tracker1.add_query(epsilon=2.0, delta=1e-5, mechanism="laplace", query_type="test2")

            # Verify file was created
            assert persistence_path.exists()

            # Create new tracker from same file
            tracker2 = PrivacyBudgetTracker(
                total_budget=10.0,
                persistence_path=persistence_path,
            )

            # State should be loaded
            assert tracker2.epsilon_spent == 5.0
            assert len(tracker2.queries) == 2
            assert tracker2.queries[0].epsilon == 3.0
            assert tracker2.queries[1].epsilon == 2.0

    def test_thread_safety(self):
        """Test thread-safe concurrent query additions."""
        import threading

        tracker = PrivacyBudgetTracker(total_budget=100.0)

        num_threads = 10
        queries_per_thread = 10
        epsilon_per_query = 0.5

        def add_queries():
            for _ in range(queries_per_thread):
                tracker.add_query(
                    epsilon=epsilon_per_query,
                    delta=1e-5,
                    mechanism="laplace",
                    query_type="test",
                )

        threads = [threading.Thread(target=add_queries) for _ in range(num_threads)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # All queries should be recorded
        expected_epsilon = num_threads * queries_per_thread * epsilon_per_query
        assert tracker.epsilon_spent == expected_epsilon
        assert len(tracker.queries) == num_threads * queries_per_thread


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
