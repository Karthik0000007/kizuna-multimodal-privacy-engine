"""SensorPayload assembler with backpressure handling.

Converts temporally aligned multimodal data into SensorPayload objects
while handling backpressure and queue management.
"""

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Dict, Generator, Optional

from ..logger import get_ingestion_logger
from .models import SensorPayload
from .temporal_align import AlignedData, TemporalAligner

logger = get_ingestion_logger()


class BackpressurePolicy(Enum):
    """Backpressure handling policies."""

    NEWEST_WINS = "newest_wins"  # Drop oldest, keep newest
    OLDEST_WINS = "oldest_wins"  # Drop newest, keep oldest
    DROP_RANDOM = "drop_random"  # Drop random item


@dataclass
class AssemblerStats:
    """Statistics for the assembler."""

    payloads_created: int = 0
    payloads_dropped: int = 0
    video_only_payloads: int = 0
    audio_only_payloads: int = 0
    env_only_payloads: int = 0
    complete_payloads: int = 0
    partial_payloads: int = 0
    average_jitter_ms: float = 0.0
    total_jitter_ms: float = 0.0


class PayloadAssembler:
    """Assemble SensorPayload objects from aligned multimodal data.

    Features:
    - Converts AlignedData to SensorPayload
    - Validates that at least one modality is present
    - Handles backpressure with configurable policies
    - Tracks statistics and dropped payloads
    """

    def __init__(
        self,
        camera_id: str = "camera_0",
        max_queue_size: int = 1000,
        backpressure_policy: BackpressurePolicy = BackpressurePolicy.NEWEST_WINS,
        allow_partial_payloads: bool = True,
    ) -> None:
        """Initialize payload assembler.

        Args:
            camera_id: Camera/source identifier
            max_queue_size: Maximum number of payloads to queue
            backpressure_policy: Policy for handling queue overflow
            allow_partial_payloads: If True, allow payloads with missing modalities
        """
        if max_queue_size < 1:
            raise ValueError(f"Max queue size must be at least 1, got {max_queue_size}")

        self.camera_id = camera_id
        self.max_queue_size = max_queue_size
        self.backpressure_policy = backpressure_policy
        self.allow_partial_payloads = allow_partial_payloads

        self.payload_queue: Deque[SensorPayload] = deque(maxlen=max_queue_size)
        self.stats = AssemblerStats()

        logger.info(
            "assembler_initialized",
            camera_id=camera_id,
            max_queue_size=max_queue_size,
            backpressure_policy=backpressure_policy.value,
            allow_partial_payloads=allow_partial_payloads,
        )

    def assemble(self, aligned_data: AlignedData) -> Optional[SensorPayload]:
        """Assemble a SensorPayload from aligned multimodal data.

        Args:
            aligned_data: Temporally aligned multimodal data

        Returns:
            SensorPayload if successful, None if validation fails

        Raises:
            ValueError: If no modalities are present and allow_partial_payloads is False
        """
        # Check if at least one modality is present
        has_video = aligned_data.video is not None
        has_audio = aligned_data.audio is not None
        has_env = aligned_data.environmental is not None

        if not (has_video or has_audio or has_env):
            logger.warning("aligned_data_empty", timestamp=aligned_data.timestamp)
            return None

        # If partial payloads not allowed, require all modalities
        if not self.allow_partial_payloads and not (has_video and has_audio and has_env):
            logger.warning(
                "partial_payload_rejected",
                timestamp=aligned_data.timestamp,
                has_video=has_video,
                has_audio=has_audio,
                has_env=has_env,
            )
            return None

        # Extract data from aligned modalities
        video_frame = aligned_data.video.frame if aligned_data.video else None

        audio_chunk = aligned_data.audio.audio if aligned_data.audio else None

        env_data = None
        if aligned_data.environmental:
            env_data = {
                "temperature": aligned_data.environmental.temperature,
                "humidity": aligned_data.environmental.humidity,
                "motion": aligned_data.environmental.motion,
                "light": aligned_data.environmental.light,
                "air_quality": aligned_data.environmental.air_quality,
            }

        # Create SensorPayload
        try:
            payload = SensorPayload(
                timestamp=aligned_data.timestamp,
                camera_id=self.camera_id,
                video_frame=video_frame,
                audio_chunk=audio_chunk,
                env_data=env_data,
            )
        except ValueError as e:
            logger.error("payload_creation_failed", error=str(e))
            return None

        # Update statistics
        self.stats.payloads_created += 1
        self.stats.total_jitter_ms += aligned_data.jitter * 1000

        if payload.is_complete():
            self.stats.complete_payloads += 1
        else:
            self.stats.partial_payloads += 1

            # Track single-modality payloads
            modalities = payload.get_modalities()
            if len(modalities) == 1:
                if "video" in modalities:
                    self.stats.video_only_payloads += 1
                elif "audio" in modalities:
                    self.stats.audio_only_payloads += 1
                elif "environmental" in modalities:
                    self.stats.env_only_payloads += 1

        # Update average jitter
        if self.stats.payloads_created > 0:
            self.stats.average_jitter_ms = self.stats.total_jitter_ms / self.stats.payloads_created

        logger.debug(
            "payload_assembled",
            timestamp=payload.timestamp,
            modalities=payload.get_modalities(),
            jitter_ms=aligned_data.jitter * 1000,
            size_bytes=payload.get_size_bytes(),
        )

        return payload

    def enqueue(self, payload: SensorPayload) -> bool:
        """Add payload to queue with backpressure handling.

        Args:
            payload: SensorPayload to enqueue

        Returns:
            True if enqueued successfully, False if dropped
        """
        # Check if queue is full
        if len(self.payload_queue) >= self.max_queue_size:
            # Apply backpressure policy
            dropped = self._apply_backpressure(payload)

            if dropped:
                self.stats.payloads_dropped += 1
                logger.warning(
                    "payload_dropped_backpressure",
                    policy=self.backpressure_policy.value,
                    queue_size=len(self.payload_queue),
                )
                return False

        # Add to queue
        self.payload_queue.append(payload)
        return True

    def dequeue(self) -> Optional[SensorPayload]:
        """Remove and return the oldest payload from queue.

        Returns:
            SensorPayload if queue is not empty, None otherwise
        """
        if self.payload_queue:
            return self.payload_queue.popleft()
        return None

    def stream(
        self, aligner: TemporalAligner, duration_seconds: Optional[float] = None
    ) -> Generator[SensorPayload, None, None]:
        """Stream assembled payloads from temporal aligner.

        Args:
            aligner: TemporalAligner instance
            duration_seconds: Duration to stream (None = infinite)

        Yields:
            SensorPayload objects
        """
        start_time = time.time()

        logger.info("payload_stream_started", duration_seconds=duration_seconds)

        while True:
            # Check duration limit
            if duration_seconds and (time.time() - start_time) >= duration_seconds:
                break

            # Try to get aligned data
            aligned = aligner.try_align()

            if aligned:
                # Assemble payload
                payload = self.assemble(aligned)

                if payload:
                    # Try to enqueue
                    if self.enqueue(payload):
                        # Dequeue and yield
                        queued_payload = self.dequeue()
                        if queued_payload:
                            yield queued_payload

            # Cleanup stale data
            aligner.cleanup_stale_data()

            # Small sleep to avoid busy waiting
            time.sleep(0.001)

        logger.info(
            "payload_stream_completed",
            payloads_created=self.stats.payloads_created,
            payloads_dropped=self.stats.payloads_dropped,
        )

    def get_queue_size(self) -> int:
        """Get current queue size.

        Returns:
            Number of payloads in queue
        """
        return len(self.payload_queue)

    def get_queue_utilization(self) -> float:
        """Get queue utilization as a percentage.

        Returns:
            Queue utilization (0.0-1.0)
        """
        return len(self.payload_queue) / self.max_queue_size

    def get_stats(self) -> Dict:
        """Get assembler statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "payloads_created": self.stats.payloads_created,
            "payloads_dropped": self.stats.payloads_dropped,
            "video_only_payloads": self.stats.video_only_payloads,
            "audio_only_payloads": self.stats.audio_only_payloads,
            "env_only_payloads": self.stats.env_only_payloads,
            "complete_payloads": self.stats.complete_payloads,
            "partial_payloads": self.stats.partial_payloads,
            "average_jitter_ms": self.stats.average_jitter_ms,
            "queue_size": len(self.payload_queue),
            "queue_utilization": self.get_queue_utilization(),
        }

    def reset(self) -> None:
        """Reset assembler state."""
        self.payload_queue.clear()
        self.stats = AssemblerStats()
        logger.info("assembler_reset")

    # Private methods

    def _apply_backpressure(self, new_payload: SensorPayload) -> bool:
        """Apply backpressure policy when queue is full.

        Args:
            new_payload: New payload trying to be added

        Returns:
            True if new payload should be dropped, False if old payload was dropped
        """
        if self.backpressure_policy == BackpressurePolicy.NEWEST_WINS:
            # Drop oldest payload, keep newest
            if self.payload_queue:
                self.payload_queue.popleft()
            return False  # New payload not dropped

        elif self.backpressure_policy == BackpressurePolicy.OLDEST_WINS:
            # Drop newest payload (don't add it)
            return True  # New payload dropped

        elif self.backpressure_policy == BackpressurePolicy.DROP_RANDOM:
            # Drop random payload
            import random

            if self.payload_queue and random.random() < 0.5:
                # Drop old payload
                self.payload_queue.popleft()
                return False
            else:
                # Drop new payload
                return True

        return True  # Default: drop new payload


def main() -> None:
    """Demo payload assembler."""
    from .audio_simulator import AudioScenario, AudioSimulator
    from .env_simulator import EnvironmentalScenario, EnvironmentalSimulator
    from .temporal_align import TemporalAligner
    from .video_simulator import VideoScenario, VideoSimulator

    print("Testing payload assembler with synthetic data streams...")

    # Create simulators
    video_sim = VideoSimulator(fps=10, scenario=VideoScenario.PERSON_WALKING)
    audio_sim = AudioSimulator(
        sample_rate=16000, chunk_duration=1.0, scenario=AudioScenario.NORMAL_AMBIENT
    )
    env_sim = EnvironmentalSimulator(polling_rate=1.0, scenario=EnvironmentalScenario.OCCUPIED_ROOM)

    # Create aligner
    aligner = TemporalAligner(window_size=1.0, jitter_tolerance=0.1, require_all_modalities=False)

    # Create assembler
    assembler = PayloadAssembler(
        camera_id="demo_camera",
        max_queue_size=50,
        backpressure_policy=BackpressurePolicy.NEWEST_WINS,
        allow_partial_payloads=True,
    )

    # Generate data for 5 seconds
    video_gen = video_sim.generate(duration_seconds=5)
    audio_gen = audio_sim.generate(duration_seconds=5)
    env_gen = env_sim.generate(duration_seconds=5)

    print(f"{'Time':<10} {'Modalities':<30} {'Size(KB)':<10} {'Jitter(ms)':<12}")
    print("-" * 70)

    payload_count = 0

    # Feed data to aligner
    for _ in range(50):
        try:
            aligner.add_video(next(video_gen))
        except StopIteration:
            pass

        try:
            aligner.add_audio(next(audio_gen))
        except StopIteration:
            pass

        try:
            aligner.add_environmental(next(env_gen))
        except StopIteration:
            pass

        # Try to align and assemble
        aligned = aligner.try_align()
        if aligned:
            payload = assembler.assemble(aligned)
            if payload:
                payload_count += 1
                modalities_str = ", ".join(payload.get_modalities())
                size_kb = payload.get_size_bytes() / 1024

                print(
                    f"{payload.timestamp:>8.2f}s "
                    f"{modalities_str:<30} "
                    f"{size_kb:>8.2f}   "
                    f"{aligned.jitter * 1000:>10.1f}"
                )

        aligner.cleanup_stale_data()
        time.sleep(0.1)

    print(f"\nAssembler Statistics:")
    stats = assembler.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
