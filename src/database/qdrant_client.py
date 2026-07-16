import uuid
from typing import Any, Dict, List, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from src.database.base import SearchResult, VectorStore
from src.database.metadata import MetadataFilter, PayloadMetadata


class QdrantStore(VectorStore):
    def __init__(
        self, host: str = "localhost", port: int = 6333, collection_name: str = "kizuna_embeddings"
    ):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name

    def create_collection(self, dimension: int, distance: str = "Cosine") -> None:
        try:
            self.client.get_collection(self.collection_name)
        except UnexpectedResponse:
            dist_metric = qmodels.Distance.COSINE
            if distance.lower() == "euclid":
                dist_metric = qmodels.Distance.EUCLID
            elif distance.lower() == "dot":
                dist_metric = qmodels.Distance.DOT

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(size=dimension, distance=dist_metric),
                hnsw_config=qmodels.HnswConfigDiff(m=16, ef_construct=100),
            )

    def delete_collection(self) -> None:
        self.client.delete_collection(self.collection_name)

    def get_collection_info(self) -> Dict[str, Any]:
        info = self.client.get_collection(self.collection_name)
        return info.model_dump()

    def insert(self, vector: np.ndarray, metadata: Dict[str, Any]) -> str:
        point_id = str(uuid.uuid4())

        # Validate metadata with Pydantic
        valid_metadata = PayloadMetadata(**metadata).model_dump()

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                qmodels.PointStruct(id=point_id, vector=vector.tolist(), payload=valid_metadata)
            ],
        )
        return point_id

    def search(
        self, query: np.ndarray, top_k: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        query_filter = None
        if filters:
            conditions = []
            valid_filters = MetadataFilter(**filters)

            if valid_filters.start_time is not None or valid_filters.end_time is not None:
                range_cond = qmodels.Range()
                if valid_filters.start_time is not None:
                    range_cond.gte = valid_filters.start_time
                if valid_filters.end_time is not None:
                    range_cond.lte = valid_filters.end_time
                conditions.append(qmodels.FieldCondition(key="timestamp", range=range_cond))

            if valid_filters.source_node_id:
                conditions.append(
                    qmodels.FieldCondition(
                        key="source_node_id",
                        match=qmodels.MatchValue(value=valid_filters.source_node_id),
                    )
                )

            if valid_filters.modalities_contains:
                conditions.append(
                    qmodels.FieldCondition(
                        key="modalities_fused",
                        match=qmodels.MatchValue(value=valid_filters.modalities_contains),
                    )
                )

            if valid_filters.event_type:
                conditions.append(
                    qmodels.FieldCondition(
                        key="event_type", match=qmodels.MatchValue(value=valid_filters.event_type)
                    )
                )

            if valid_filters.is_centroid is not None:
                conditions.append(
                    qmodels.FieldCondition(
                        key="is_centroid", match=qmodels.MatchValue(value=valid_filters.is_centroid)
                    )
                )

            if valid_filters.is_exemplar is not None:
                conditions.append(
                    qmodels.FieldCondition(
                        key="is_exemplar", match=qmodels.MatchValue(value=valid_filters.is_exemplar)
                    )
                )

            if conditions:
                query_filter = qmodels.Filter(must=conditions)

        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query.tolist(),
            limit=top_k,
            query_filter=query_filter,
        )

        return [
            SearchResult(id=str(hit.id), score=hit.score, payload=hit.payload or {}) for hit in hits
        ]
