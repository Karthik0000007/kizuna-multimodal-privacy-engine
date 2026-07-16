import json
import os
import uuid
from typing import Any

import faiss
import numpy as np

from src.database.base import SearchResult, VectorStore
from src.database.metadata import MetadataFilter, PayloadMetadata


class FAISSStore(VectorStore):
    def __init__(
        self,
        index_path: str = "data/faiss_index.bin",
        metadata_path: str = "data/faiss_metadata.json",
        max_size: int = 1000000,
    ):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.max_size = max_size
        self.dimension = None
        self.index = None
        self.metadata_store: dict[str, dict[str, Any]] = {}
        self.id_to_index: dict[str, int] = {}
        self.index_to_id: dict[int, str] = {}
        self.current_count = 0

        self._load()

    def create_collection(self, dimension: int, distance: str = "Cosine") -> None:
        if self.index is None:
            self.dimension = dimension
            # For cosine similarity, vectors should be L2 normalized before adding
            # Inner product of normalized vectors equals cosine similarity
            self.index = faiss.IndexFlatIP(dimension)
            self._save()

    def delete_collection(self) -> None:
        self.index = None
        self.dimension = None
        self.metadata_store = {}
        self.id_to_index = {}
        self.index_to_id = {}
        self.current_count = 0
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.metadata_path):
            os.remove(self.metadata_path)

    def get_collection_info(self) -> dict[str, Any]:
        return {"dimension": self.dimension, "count": self.current_count, "max_size": self.max_size}

    def insert(self, vector: np.ndarray, metadata: dict[str, Any]) -> str:
        if self.index is None:
            raise ValueError("Collection not created. Call create_collection first.")

        if self.current_count >= self.max_size:
            # Simple approach: clear everything (in a real app, maybe implement a ring buffer or delete oldest)
            self.delete_collection()
            self.create_collection(self.dimension)

        point_id = str(uuid.uuid4())
        valid_metadata = PayloadMetadata(**metadata).model_dump()

        # Ensure vector is L2 normalized for cosine similarity
        norm_vector = vector / np.linalg.norm(vector)
        norm_vector = norm_vector.astype("float32").reshape(1, -1)

        self.index.add(norm_vector)

        idx = self.current_count
        self.metadata_store[point_id] = valid_metadata
        self.id_to_index[point_id] = idx
        self.index_to_id[idx] = point_id

        self.current_count += 1

        # Periodically save
        if self.current_count % 100 == 0:
            self._save()

        return point_id

    def search(
        self, query: np.ndarray, top_k: int = 10, filters: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        if self.index is None or self.current_count == 0:
            return []

        # We need to fetch more results if we have filters, then filter them post-search
        # A proper implementation would pre-filter, but FAISS flat indexes don't support it natively easily
        search_k = top_k * 10 if filters else top_k
        search_k = min(search_k, self.current_count)

        norm_query = query / np.linalg.norm(query)
        norm_query = norm_query.astype("float32").reshape(1, -1)

        distances, indices = self.index.search(norm_query, search_k)

        results = []
        valid_filters = MetadataFilter(**filters) if filters else None

        for i in range(len(indices[0])):
            idx = int(indices[0][i])
            if idx == -1:
                continue

            score = float(distances[0][i])
            point_id = self.index_to_id.get(idx)
            if not point_id:
                continue

            payload = self.metadata_store.get(point_id, {})

            # Apply filters manually
            if valid_filters:
                if (
                    valid_filters.start_time is not None
                    and payload.get("timestamp", 0) < valid_filters.start_time
                ):
                    continue
                if (
                    valid_filters.end_time is not None
                    and payload.get("timestamp", float("inf")) > valid_filters.end_time
                ):
                    continue
                if (
                    valid_filters.source_node_id
                    and payload.get("source_node_id") != valid_filters.source_node_id
                ):
                    continue
                if (
                    valid_filters.modalities_contains
                    and valid_filters.modalities_contains not in payload.get("modalities_fused", [])
                ):
                    continue
                if (
                    valid_filters.event_type
                    and payload.get("event_type") != valid_filters.event_type
                ):
                    continue
                if (
                    valid_filters.is_centroid is not None
                    and payload.get("is_centroid") != valid_filters.is_centroid
                ):
                    continue
                if (
                    valid_filters.is_exemplar is not None
                    and payload.get("is_exemplar") != valid_filters.is_exemplar
                ):
                    continue

            results.append(SearchResult(id=point_id, score=score, payload=payload))

            if len(results) >= top_k:
                break

        return results

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)

        with open(self.metadata_path, "w") as f:
            json.update = {
                "dimension": self.dimension,
                "current_count": self.current_count,
                "metadata_store": self.metadata_store,
                "id_to_index": self.id_to_index,
                "index_to_id": {str(k): v for k, v in self.index_to_id.items()},
            }
            json.dump(json.update, f)

    def _load(self) -> None:
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path) as f:
                data = json.load(f)
                self.dimension = data.get("dimension")
                self.current_count = data.get("current_count", 0)
                self.metadata_store = data.get("metadata_store", {})
                self.id_to_index = data.get("id_to_index", {})

                # Convert string keys back to int
                idx_to_id = data.get("index_to_id", {})
                self.index_to_id = {int(k): v for k, v in idx_to_id.items()}
