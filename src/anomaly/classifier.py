from typing import List, Tuple

import numpy as np

from src.database.base import VectorStore


class AnomalyClassifier:
    """
    Classifies the type of an anomaly using few-shot matching against labeled exemplars.
    """

    def __init__(self, vector_store: VectorStore, top_k: int = 3):
        """
        Args:
            vector_store: Vector DB where labeled exemplars are stored.
            top_k: Number of predictions to return.
        """
        self.vector_store = vector_store
        self.top_k = top_k
        self.supported_types = [
            "fall_risk",
            "wandering",
            "congestion_alert",
            "unusual_sound",
            "environmental_anomaly",
        ]

    def classify(self, embedding: np.ndarray) -> List[Tuple[str, float]]:
        """
        Returns a list of tuples (anomaly_type, confidence_score) sorted by confidence.
        """
        # We search for the nearest neighbors that have a defined event_type
        # In a real system we'd filter for exemplars (e.g., is_exemplar: True)
        results = self.vector_store.search(query=embedding, top_k=10, filters={"is_exemplar": True})

        if not results:
            return [("unknown", 1.0)]

        type_scores = {}
        for r in results:
            a_type = r.payload.get("event_type", "unknown")
            if a_type not in type_scores:
                type_scores[a_type] = []
            type_scores[a_type].append(r.score)

        # Average the scores for each type found
        aggregated = []
        for a_type, scores in type_scores.items():
            avg_score = sum(scores) / len(scores)
            aggregated.append((a_type, avg_score))

        # Sort by highest similarity/confidence
        aggregated.sort(key=lambda x: x[1], reverse=True)

        return aggregated[: self.top_k]
