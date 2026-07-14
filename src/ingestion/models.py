"""Data models for ingestion pipeline."""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class SensorPayload:
    """Raw temporal slice of incoming multimodal data.

    This structure represents a single time-aligned snapshot of all available
    sensor modalities. It is destroyed immediately after embedding generation
    to ensure privacy compliance.
    """

    timestamp: float
    camera_id: str

    # Video modality (optional)
    video_frame: Optional[NDArray[np.uint8]] = None  # Shape: (H, W, 3)

    # Audio modality (optional)
    audio_chunk: Optional[NDArray[np.float32]] = None  # Shape: (num_samples,)

    # Environmental sensor modality (optional)
    env_data: Optional[Dict[str, float]] = None  # e.g., {"temperature": 22.1, "motion": 1}

    def __post_init__(self) -> None:
        """Validate that at least one modality is present."""
        if self.video_frame is None and self.audio_chunk is None and self.env_data is None:
            raise ValueError("SensorPayload must contain at least one modality")

    def get_modalities(self) -> List[str]:
        """Get list of available modalities.

        Returns:
            List of modality names present in this payload
        """
        modalities = []
        if self.video_frame is not None:
            modalities.append("video")
        if self.audio_chunk is not None:
            modalities.append("audio")
        if self.env_data is not None:
            modalities.append("environmental")
        return modalities

    def is_complete(self) -> bool:
        """Check if all three modalities are present.

        Returns:
            True if video, audio, and environmental data are all present
        """
        return (
            self.video_frame is not None
            and self.audio_chunk is not None
            and self.env_data is not None
        )

    def get_size_bytes(self) -> int:
        """Estimate memory size of payload in bytes.

        Returns:
            Approximate size in bytes
        """
        size = 0

        if self.video_frame is not None:
            size += self.video_frame.nbytes

        if self.audio_chunk is not None:
            size += self.audio_chunk.nbytes

        if self.env_data is not None:
            # Rough estimate: 8 bytes per float value
            size += len(self.env_data) * 8

        # Add overhead for metadata
        size += 100  # timestamp, camera_id, etc.

        return size
