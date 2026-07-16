from pydantic import BaseModel, Field


class PayloadMetadata(BaseModel):
    """Schema for Qdrant points metadata."""

    timestamp: float = Field(..., description="Timestamp of the event")
    source_node_id: str = Field(..., description="ID of the source edge node")
    modalities_fused: list[str] = Field(..., description="List of modalities used for embedding")
    event_type: str | None = Field(default=None, description="Type of event if anomalous")
    dp_epsilon: float = Field(..., description="Privacy budget used for this embedding")
    is_centroid: bool | None = Field(
        default=False, description="Whether this point is a cluster centroid"
    )
    is_exemplar: bool | None = Field(
        default=False, description="Whether this point is a labeled exemplar"
    )
    decision_boundary: float | None = Field(
        default=None, description="Decision boundary for centroids"
    )


class MetadataFilter(BaseModel):
    """Filter schema for vector database queries."""

    start_time: float | None = None
    end_time: float | None = None
    source_node_id: str | None = None
    modalities_contains: str | None = None
    event_type: str | None = None
    is_centroid: bool | None = None
    is_exemplar: bool | None = None
