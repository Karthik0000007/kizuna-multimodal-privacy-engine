import threading
import time
from typing import Any

import numpy as np
from sklearn.manifold import TSNE

from src.database.base import VectorStore


class ProjectionService:
    """
    Computes and caches 2D projections (t-SNE) of stored embeddings
    for the Vector Explorer dashboard page.
    """

    def __init__(self, vector_store: VectorStore, cache_ttl: float = 300.0, perplexity: int = 30):
        """
        Args:
            vector_store: The vector database to pull embeddings from.
            cache_ttl: How long (seconds) a cached projection is valid (default 5 min).
            perplexity: t-SNE perplexity parameter.
        """
        self.vector_store = vector_store
        self.cache_ttl = cache_ttl
        self.perplexity = perplexity

        self._cached_projection: dict[str, Any] | None = None
        self._cache_time: float = 0.0
        self._lock = threading.Lock()

    def compute_projection(self, max_points: int = 1000, force: bool = False) -> dict[str, Any]:
        """
        Compute or return cached 2D t-SNE projection.

        Returns a dict with:
            - coords: np.ndarray of shape (N, 2)
            - metadata: list of dicts with point metadata
            - computed_at: timestamp
            - num_points: int
        """
        with self._lock:
            if (
                not force
                and self._cached_projection
                and (time.time() - self._cache_time < self.cache_ttl)
            ):
                return self._cached_projection

        # Fetch embeddings from the store
        # We use a zero-vector query to get a broad sample
        query = np.zeros(512, dtype=np.float32)
        results = self.vector_store.search(query, top_k=max_points)

        if len(results) < 5:
            # Not enough data for t-SNE
            return {
                "coords": np.array([]),
                "metadata": [],
                "computed_at": time.time(),
                "num_points": 0,
            }

        # We need the actual vectors for t-SNE. Since VectorStore.search
        # doesn't return vectors, we'll generate proxy coordinates from
        # the similarity scores relative to multiple reference queries.
        # This is a practical workaround for dashboard visualization.
        #
        # For a production system, we'd extend VectorStore to return vectors,
        # or maintain a shadow copy. Here we use score-based proxy features.

        n_probes = min(20, len(results))
        probe_vectors = [np.random.randn(512).astype(np.float32) for _ in range(n_probes)]

        feature_matrix = np.zeros((len(results), n_probes), dtype=np.float32)
        for j, probe in enumerate(probe_vectors):
            probe_results = self.vector_store.search(probe, top_k=max_points)
            # Build a lookup of id -> score
            score_map = {r.id: r.score for r in probe_results}
            for i, r in enumerate(results):
                feature_matrix[i, j] = score_map.get(r.id, 0.0)

        # Add slight noise to avoid zero-variance NaNs, particularly with mock test data
        feature_matrix += np.random.normal(0, 1e-4, feature_matrix.shape).astype(np.float32)

        # Run t-SNE on the proxy feature matrix
        effective_perplexity = min(self.perplexity, max(1, len(results) // 2 - 1))
        tsne_method = "exact" if len(results) < 50 else "barnes_hut"
        tsne = TSNE(
            n_components=2,
            perplexity=effective_perplexity,
            random_state=42,
            max_iter=300,
            method=tsne_method,
        )
        coords = tsne.fit_transform(feature_matrix)

        metadata = []
        for r in results:
            metadata.append(
                {
                    "id": r.id,
                    "event_type": r.payload.get("event_type", "normal"),
                    "source_node_id": r.payload.get("source_node_id", "unknown"),
                    "modalities": r.payload.get("modalities_fused", []),
                    "timestamp": r.payload.get("timestamp", 0),
                    "is_centroid": r.payload.get("is_centroid", False),
                }
            )

        projection = {
            "coords": coords,
            "metadata": metadata,
            "computed_at": time.time(),
            "num_points": len(results),
        }

        with self._lock:
            self._cached_projection = projection
            self._cache_time = time.time()

        return projection

    def add_incremental_point(self, point_id: str, event_type: str, source_node: str) -> bool:
        """
        Add a new point to the existing cached projection without full recomputation.
        Places the new point near similar existing points using a simple heuristic.
        """
        with self._lock:
            if not self._cached_projection or self._cached_projection["num_points"] == 0:
                return False

            coords = self._cached_projection["coords"]
            meta = self._cached_projection["metadata"]

            # Find points of the same event_type and average their coordinates
            same_type_coords = []
            for i, m in enumerate(meta):
                if m["event_type"] == event_type:
                    same_type_coords.append(coords[i])

            if same_type_coords:
                center = np.mean(same_type_coords, axis=0)
                # Add jitter
                new_coord = center + np.random.normal(0, 0.3, 2)
            else:
                # Place randomly near the centroid of all points
                center = np.mean(coords, axis=0)
                new_coord = center + np.random.normal(0, 1.0, 2)

            new_coords = np.vstack([coords, new_coord.reshape(1, 2)])
            meta.append(
                {
                    "id": point_id,
                    "event_type": event_type,
                    "source_node_id": source_node,
                    "modalities": [],
                    "timestamp": time.time(),
                    "is_centroid": False,
                }
            )

            self._cached_projection["coords"] = new_coords
            self._cached_projection["metadata"] = meta
            self._cached_projection["num_points"] += 1

        return True

    def invalidate_cache(self):
        """Force cache invalidation."""
        with self._lock:
            self._cached_projection = None
            self._cache_time = 0.0
