from typing import List, Optional

from pydantic import BaseModel, Field


class PayloadMetadata(BaseModel):
    """Schema for Qdrant points metadata."""

    timestamp: float = Field(..., description="Timestamp of the event")
    source_node_id: str = Field(..., description="ID of the source edge node")
    modalities_fused: List[str] = Field(..., description="List of modalities used for embedding")
    event_type: Optional[str] = Field(default=None, description="Type of event if anomalous")
    dp_epsilon: float = Field(..., description="Privacy budget used for this embedding")
    is_centroid: Optional[bool] = Field(
        default=False, description="Whether this point is a cluster centroid"
    )
    is_exemplar: Optional[bool] = Field(
        default=False, description="Whether this point is a labeled exemplar"
    )
    decision_boundary: Optional[float] = Field(
        default=None, description="Decision boundary for centroids"
    )


class MetadataFilter(BaseModel):
    """Filter schema for vector database queries."""

    start_time: Optional[float] = None
    end_time: Optional[float] = None
    source_node_id: Optional[str] = None
    modalities_contains: Optional[str] = None
    event_type: Optional[str] = None
    is_centroid: Optional[bool] = None
    is_exemplar: Optional[bool] = None
