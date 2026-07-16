"""Privacy budget tracking and composition.

Tracks cumulative privacy budget (epsilon) spend and enforces budget ceilings.
Supports sequential and parallel composition.
"""

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..logger import get_logger

logger = get_logger("privacy")


@dataclass
class PrivacyQuery:
    """Record of a single privacy query."""

    timestamp: str
    epsilon: float
    delta: float
    mechanism: str
    query_type: str  # e.g., "embedding", "search", "aggregate"
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "epsilon": self.epsilon,
            "delta": self.delta,
            "mechanism": self.mechanism,
            "query_type": self.query_type,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PrivacyQuery":
        """Create from dictionary."""
        return cls(
            timestamp=data["timestamp"],
            epsilon=data["epsilon"],
            delta=data["delta"],
            mechanism=data["mechanism"],
            query_type=data["query_type"],
            metadata=data.get("metadata", {}),
        )


class PrivacyBudgetTracker:
    """Privacy budget tracker with sequential and parallel composition.

    Tracks cumulative epsilon spend across all queries and enforces
    a total budget ceiling. Supports persistence to disk for recovery
    after restarts.

    Thread-safe for concurrent query logging.
    """

    def __init__(
        self,
        total_budget: float,
        alert_threshold: float = 0.8,
        composition: str = "sequential",
        persistence_path: str | Path | None = None,
    ) -> None:
        """Initialize privacy budget tracker.

        Args:
            total_budget: Total epsilon budget ceiling
            alert_threshold: Fraction at which to alert (0-1)
            composition: "sequential" or "parallel"
            persistence_path: Path to save budget state (JSON)

        Raises:
            ValueError: If parameters are invalid
        """
        if total_budget <= 0:
            raise ValueError(f"total_budget must be positive, got {total_budget}")
        if not 0 < alert_threshold < 1:
            raise ValueError(f"alert_threshold must be in (0, 1), got {alert_threshold}")
        if composition not in ["sequential", "parallel"]:
            raise ValueError(f"composition must be 'sequential' or 'parallel', got {composition}")

        self.total_budget = total_budget
        self.alert_threshold = alert_threshold
        self.composition = composition
        self.persistence_path = Path(persistence_path) if persistence_path else None

        # Query history
        self.queries: list[PrivacyQuery] = []

        # Current cumulative spend
        self.epsilon_spent = 0.0
        self.max_delta = 0.0  # Track maximum delta

        # Alerts
        self.alert_triggered = False
        self.budget_exhausted = False

        # Thread safety
        self._lock = threading.Lock()

        # Load persisted state if available
        if self.persistence_path and self.persistence_path.exists():
            self._load_state()

        logger.info(
            "privacy_budget_tracker_initialized",
            total_budget=total_budget,
            alert_threshold=alert_threshold,
            composition=composition,
            persistence_path=str(persistence_path) if persistence_path else None,
        )

    def add_query(
        self,
        epsilon: float,
        delta: float,
        mechanism: str,
        query_type: str,
        metadata: dict | None = None,
    ) -> bool:
        """Record a privacy query and update budget.

        Args:
            epsilon: Epsilon spent for this query
            delta: Delta for this query
            mechanism: DP mechanism used ("laplace", "gaussian")
            query_type: Type of query
            metadata: Optional metadata about the query

        Returns:
            True if query was accepted (budget available), False if rejected

        Raises:
            ValueError: If epsilon or delta are negative
        """
        if epsilon < 0:
            raise ValueError(f"epsilon must be non-negative, got {epsilon}")
        if delta < 0:
            raise ValueError(f"delta must be non-negative, got {delta}")

        with self._lock:
            # Check if budget is exhausted
            if self.budget_exhausted:
                logger.warning(
                    "privacy_query_rejected_budget_exhausted",
                    epsilon_requested=epsilon,
                    epsilon_spent=self.epsilon_spent,
                    total_budget=self.total_budget,
                )
                return False

            # Create query record
            query = PrivacyQuery(
                timestamp=datetime.utcnow().isoformat(),
                epsilon=epsilon,
                delta=delta,
                mechanism=mechanism,
                query_type=query_type,
                metadata=metadata or {},
            )

            # Compute new cumulative spend based on composition
            new_epsilon_spent = self._compute_composition(epsilon)

            # Check if new spend would exceed budget
            if new_epsilon_spent > self.total_budget:
                logger.warning(
                    "privacy_query_rejected_exceeds_budget",
                    epsilon_requested=epsilon,
                    epsilon_spent=self.epsilon_spent,
                    new_epsilon_spent=new_epsilon_spent,
                    total_budget=self.total_budget,
                )
                self.budget_exhausted = True
                return False

            # Accept query
            self.queries.append(query)
            self.epsilon_spent = new_epsilon_spent
            self.max_delta = max(self.max_delta, delta)

            # Check alert threshold
            if not self.alert_triggered:
                utilization = self.epsilon_spent / self.total_budget
                if utilization >= self.alert_threshold:
                    self.alert_triggered = True
                    logger.warning(
                        "privacy_budget_alert",
                        epsilon_spent=self.epsilon_spent,
                        total_budget=self.total_budget,
                        utilization=utilization,
                        alert_threshold=self.alert_threshold,
                    )

            logger.info(
                "privacy_query_accepted",
                epsilon=epsilon,
                delta=delta,
                mechanism=mechanism,
                query_type=query_type,
                epsilon_spent=self.epsilon_spent,
                budget_remaining=self.total_budget - self.epsilon_spent,
                utilization=self.epsilon_spent / self.total_budget,
            )

            # Persist state
            if self.persistence_path:
                self._save_state()

            return True

    def _compute_composition(self, epsilon: float) -> float:
        """Compute cumulative epsilon based on composition strategy.

        Args:
            epsilon: Epsilon for new query

        Returns:
            New cumulative epsilon
        """
        if self.composition == "sequential":
            # Sequential composition: total ε = Σ εᵢ
            return self.epsilon_spent + epsilon
        elif self.composition == "parallel":
            # Parallel composition: total ε = max(εᵢ)
            # (assumes queries are independent/disjoint)
            return max(self.epsilon_spent, epsilon)
        else:
            raise ValueError(f"Unknown composition: {self.composition}")

    def get_remaining_budget(self) -> float:
        """Get remaining privacy budget.

        Returns:
            Remaining epsilon budget
        """
        with self._lock:
            return self.total_budget - self.epsilon_spent

    def get_utilization(self) -> float:
        """Get budget utilization as fraction.

        Returns:
            Utilization (0-1), where 1.0 = fully exhausted
        """
        with self._lock:
            return self.epsilon_spent / self.total_budget

    def get_status(self) -> dict:
        """Get current budget status.

        Returns:
            Dictionary with status information
        """
        with self._lock:
            return {
                "total_budget": self.total_budget,
                "epsilon_spent": self.epsilon_spent,
                "max_delta": self.max_delta,
                "remaining_budget": self.total_budget - self.epsilon_spent,
                "utilization": self.epsilon_spent / self.total_budget,
                "num_queries": len(self.queries),
                "alert_triggered": self.alert_triggered,
                "budget_exhausted": self.budget_exhausted,
                "composition": self.composition,
            }

    def get_query_history(self) -> list[PrivacyQuery]:
        """Get query history.

        Returns:
            List of privacy queries
        """
        with self._lock:
            return self.queries.copy()

    def reset(self) -> None:
        """Reset budget tracker (clears all queries).

        WARNING: This clears all privacy accounting. Use with caution.
        """
        with self._lock:
            self.queries.clear()
            self.epsilon_spent = 0.0
            self.max_delta = 0.0
            self.alert_triggered = False
            self.budget_exhausted = False

            logger.warning("privacy_budget_reset")

            if self.persistence_path:
                self._save_state()

    def _save_state(self) -> None:
        """Save budget state to disk (not thread-safe, call with lock held)."""
        if not self.persistence_path:
            return

        try:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)

            state = {
                "total_budget": self.total_budget,
                "epsilon_spent": self.epsilon_spent,
                "max_delta": self.max_delta,
                "alert_triggered": self.alert_triggered,
                "budget_exhausted": self.budget_exhausted,
                "composition": self.composition,
                "queries": [q.to_dict() for q in self.queries],
            }

            with open(self.persistence_path, "w") as f:
                json.dump(state, f, indent=2)

            logger.debug("privacy_budget_state_saved", path=str(self.persistence_path))

        except Exception as e:
            logger.error("privacy_budget_save_failed", error=str(e))

    def _load_state(self) -> None:
        """Load budget state from disk (not thread-safe, call during init)."""
        if not self.persistence_path or not self.persistence_path.exists():
            return

        try:
            with open(self.persistence_path) as f:
                state = json.load(f)

            self.epsilon_spent = state["epsilon_spent"]
            self.max_delta = state["max_delta"]
            self.alert_triggered = state["alert_triggered"]
            self.budget_exhausted = state["budget_exhausted"]
            self.queries = [PrivacyQuery.from_dict(q) for q in state["queries"]]

            logger.info(
                "privacy_budget_state_loaded",
                path=str(self.persistence_path),
                epsilon_spent=self.epsilon_spent,
                num_queries=len(self.queries),
            )

        except Exception as e:
            logger.error("privacy_budget_load_failed", error=str(e))


def main() -> None:
    """Demo privacy budget tracker."""
    import argparse
    import tempfile

    parser = argparse.ArgumentParser(description="Kizuna Privacy Budget Tracker Demo")
    parser.add_argument(
        "--total-budget",
        type=float,
        default=10.0,
        help="Total epsilon budget",
    )
    parser.add_argument(
        "--composition",
        type=str,
        default="sequential",
        choices=["sequential", "parallel"],
        help="Composition strategy",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=20,
        help="Number of test queries",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Kizuna Privacy Budget Tracker Demo")
    print("=" * 70)

    # Create temporary persistence file
    temp_file = Path(tempfile.mktemp(suffix=".json"))

    # Initialize tracker
    print("\nInitializing tracker...")
    print(f"  Total budget: ε = {args.total_budget}")
    print(f"  Composition: {args.composition}")
    print(f"  Persistence: {temp_file}")

    tracker = PrivacyBudgetTracker(
        total_budget=args.total_budget,
        alert_threshold=0.8,
        composition=args.composition,
        persistence_path=temp_file,
    )

    print("\n✓ Tracker initialized")

    # Simulate queries
    print(f"\nSimulating {args.num_queries} queries...")

    epsilon_per_query = args.total_budget / (args.num_queries * 1.2)  # Will exceed budget

    accepted_count = 0
    rejected_count = 0

    for i in range(args.num_queries):
        accepted = tracker.add_query(
            epsilon=epsilon_per_query,
            delta=1e-5,
            mechanism="laplace",
            query_type="embedding",
            metadata={"query_id": i},
        )

        if accepted:
            accepted_count += 1
        else:
            rejected_count += 1

        # Print status every 5 queries
        if (i + 1) % 5 == 0 or not accepted:
            status = tracker.get_status()
            print(f"\n  After query {i + 1}:")
            print(f"    ε spent: {status['epsilon_spent']:.4f} / {status['total_budget']:.4f}")
            print(f"    Utilization: {status['utilization']:.1%}")
            print(f"    Remaining: {status['remaining_budget']:.4f}")

            if not accepted:
                print("    ⚠ Query rejected (budget exhausted)")
                break

    # Final status
    print(f"\n{'=' * 70}")
    print("Final Status")
    print(f"{'=' * 70}")

    status = tracker.get_status()
    print(f"  Queries accepted: {accepted_count}")
    print(f"  Queries rejected: {rejected_count}")
    print(f"  Total queries: {status['num_queries']}")
    print(f"  ε spent: {status['epsilon_spent']:.4f}")
    print(f"  ε remaining: {status['remaining_budget']:.4f}")
    print(f"  Utilization: {status['utilization']:.1%}")
    print(f"  Alert triggered: {status['alert_triggered']}")
    print(f"  Budget exhausted: {status['budget_exhausted']}")

    # Test persistence
    print("\nTesting persistence...")
    print(f"  Saving state to: {temp_file}")

    # Create new tracker from same file
    tracker2 = PrivacyBudgetTracker(
        total_budget=args.total_budget,
        alert_threshold=0.8,
        composition=args.composition,
        persistence_path=temp_file,
    )

    status2 = tracker2.get_status()

    if status2["epsilon_spent"] == status["epsilon_spent"]:
        print("  ✓ State loaded successfully")
        print(f"    Recovered ε: {status2['epsilon_spent']:.4f}")
        print(f"    Recovered queries: {status2['num_queries']}")
    else:
        print("  ✗ State mismatch")

    # Cleanup
    temp_file.unlink()

    print(f"\n{'=' * 70}")
    print("✓ Demo complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
