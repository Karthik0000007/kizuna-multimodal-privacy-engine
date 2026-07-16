import numpy as np

from src.anomaly.detector import AnomalyOrchestrator
from src.anomaly.enrollment import AnomalyEnroller
from src.database.base import VectorStore


class CrossDomainEvaluator:
    """
    Evaluates cross-domain transfer learning accuracy for anomaly detection.
    """

    def __init__(self, vector_store: VectorStore, orchestrator: AnomalyOrchestrator):
        self.vector_store = vector_store
        self.orchestrator = orchestrator
        self.enroller = AnomalyEnroller(vector_store)

    def evaluate_zero_shot(
        self, test_embeddings: list[np.ndarray], test_labels: list[int]
    ) -> dict[str, float]:
        """
        Evaluate zero-shot transfer accuracy (no retraining or enrollment).

        Args:
            test_embeddings: Embeddings from the new domain (Domain B).
            test_labels: Binary labels (1 for anomaly, 0 for normal).

        Returns:
            Dictionary with metrics: precision, recall, f1, accuracy.
        """
        return self._evaluate(test_embeddings, test_labels)

    def evaluate_few_shot(
        self,
        enrollment_embeddings: list[np.ndarray],
        enrollment_label: str,
        test_embeddings: list[np.ndarray],
        test_labels: list[int],
    ) -> dict[str, float]:
        """
        Evaluate few-shot transfer accuracy by enrolling examples from Domain B,
        then testing on Domain B.

        Args:
            enrollment_embeddings: Few-shot examples (e.g., 5, 10, or 20) of anomalies.
            enrollment_label: Label for the new anomaly type.
            test_embeddings: Test embeddings from Domain B.
            test_labels: Binary labels.

        Returns:
            Dictionary with metrics.
        """
        # Enroll the new anomalies
        self.enroller.enroll(enrollment_embeddings, enrollment_label)

        # Now evaluate
        return self._evaluate(test_embeddings, test_labels)

    def _evaluate(
        self, test_embeddings: list[np.ndarray], test_labels: list[int]
    ) -> dict[str, float]:
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0

        for embedding, label in zip(test_embeddings, test_labels, strict=False):
            # Process using orchestrator
            event = self.orchestrator.process(embedding, source_node_id="evaluator")

            pred_is_anomaly = event is not None and event.is_anomaly
            true_is_anomaly = label == 1

            if pred_is_anomaly and true_is_anomaly:
                true_positives += 1
            elif pred_is_anomaly and not true_is_anomaly:
                false_positives += 1
            elif not pred_is_anomaly and not true_is_anomaly:
                true_negatives += 1
            elif not pred_is_anomaly and true_is_anomaly:
                false_negatives += 1

        precision = 0.0
        if (true_positives + false_positives) > 0:
            precision = true_positives / (true_positives + false_positives)

        recall = 0.0
        if (true_positives + false_negatives) > 0:
            recall = true_positives / (true_positives + false_negatives)

        f1 = 0.0
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)

        accuracy = (true_positives + true_negatives) / len(test_labels) if test_labels else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives,
        }
