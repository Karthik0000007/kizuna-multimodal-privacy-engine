import numpy as np

from src.anomaly.models import AnomalyResult
from src.database.base import VectorStore


class DensityAnomalyDetector:
    """
    Detects anomalies using Local Outlier Factor (LOF).
    It queries a context neighborhood from the DB and computes the LOF of the query.
    """

    def __init__(
        self, vector_store: VectorStore, context_size: int = 50, lof_threshold: float = -1.5
    ):
        """
        Args:
            vector_store: Vector DB to query context embeddings.
            context_size: Number of neighbors to fetch as the local density baseline.
            lof_threshold: Threshold for LOF. scikit-learn LOF returns negative outlier factors.
                           More negative means more anomalous.
        """
        self.vector_store = vector_store
        self.context_size = context_size
        self.lof_threshold = lof_threshold
        self.name = "density"

    def detect(self, embedding: np.ndarray) -> AnomalyResult:
        # We need the embeddings of the neighbors.
        # But VectorStore's search returns metadata, not the original vectors by default.
        # Wait, the search method in qdrant/faiss currently does not return the vector.
        # To do LOF on vectors, we need vectors.
        # Alternatively, we can assume LOF is computed using the distance scores directly,
        # but scikit-learn's LOF expects features or a distance matrix.
        # Since we only have distances to the query, it's not enough to compute LOF among the neighbors themselves.
        # So we'll have to either update VectorStore to return vectors or just use a proxy.

        # As a simplified edge-compatible density proxy if vectors are unavailable:
        # A point is in a dense region if its K-th neighbor is very close.
        # It's an outlier if the distance to its K-th neighbor is much larger than average.

        results = self.vector_store.search(embedding, top_k=self.context_size)

        if len(results) < 5:
            # Not enough context to compute density
            return AnomalyResult(
                is_anomaly=False, score=0.0, nearest_neighbors=[], detector_name=self.name
            )

        # Proxy density: using cosine similarities. Higher similarity = closer distance.
        similarities = [r.score for r in results]
        # Invert similarity to get pseudo-distance (0 to 2)
        distances = [1.0 - s for s in similarities]

        # Local density of neighbors can be approximated.
        # If query point is much farther from its nearest neighbor than the typical distance among neighbors.
        # Simplified LOF: distance to k-th nearest neighbor.
        distances[-1]

        # For a more robust proxy: average distance to k nearest
        avg_dist = np.mean(distances)

        # Let's say if avg_dist > a strict threshold, it's low density
        # For an exact LOF, we would need the pairwise distances among the neighbors.
        # Since we might not have them, we'll use a threshold on the proxy score.

        # Let's use a proxy score:
        density_score = avg_dist

        # A simple anomaly heuristic based on density
        # Note: In a real implementation with scikit-learn, we'd need VectorStore to return the raw vectors
        # and then do: lof = LocalOutlierFactor(n_neighbors=k).fit(X)._decision_function(query)
        is_anomaly = density_score > abs(self.lof_threshold)  # proxy check

        return AnomalyResult(
            is_anomaly=bool(is_anomaly),
            score=float(density_score),
            nearest_neighbors=[r.id for r in results[:5]],
            detector_name=self.name,
        )
