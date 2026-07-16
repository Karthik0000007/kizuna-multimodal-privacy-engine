"""Temporal alignment manager for multimodal data streams.

Synchronizes video frames, audio chunks, and sensor readings to a common timestamp
using sliding window buffering and jitter tolerance.
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from ..logger import get_ingestion_logger
from .audio_simulator import AudioChunk
from .env_simulator import SensorReading
from .video_simulator import VideoFrame

logger = get_ingestion_logger()


@dataclass
class AlignedData:
    """Container for temporally aligned multimodal data."""

    timestamp: float  # Common reference timestamp
    video: Optional[VideoFrame]
    audio: Optional[AudioChunk]
    environmental: Optional[SensorReading]
    jitter: float  # Maximum time difference between modalities


class TemporalAligner:
    """Temporal alignment manager for multimodal streams.

    Uses sliding window buffering to align video, audio, and environmental
    sensor data to a common timestamp within a specified jitter tolerance.

    Algorithm:
    1. Buffer incoming data from all modalities
    2. Find closest matches across modalities within jitter tolerance
    3. Emit aligned payloads when matches are found
    4. Drop stale data that exceeds window size
    """

    def __init__(
        self,
        window_size: float = 1.0,
        jitter_tolerance: float = 0.05,
        buffer_size: int = 100,
        require_all_modalities: bool = False,
    ) -> None:
        """Initialize temporal aligner.

        Args:
            window_size: Time window size in seconds
            jitter_tolerance: Maximum allowed time difference in seconds (±50ms default)
            buffer_size: Maximum number of items to buffer per modality
            require_all_modalities: If True, only emit when all modalities are present
        """
        if window_size <= 0:
            raise ValueError(f"Window size must be positive, got {window_size}")
        if jitter_tolerance < 0:
            raise ValueError(f"Jitter tolerance must be non-negative, got {jitter_tolerance}")
        if buffer_size < 1:
            raise ValueError(f"Buffer size must be at least 1, got {buffer_size}")

        self.window_size = window_size
        self.jitter_tolerance = jitter_tolerance
        self.buffer_size = buffer_size
        self.require_all_modalities = require_all_modalities

        # Buffers for each modality (deque for efficient append/popleft)
        self.video_buffer: Deque[VideoFrame] = deque(maxlen=buffer_size)
        self.audio_buffer: Deque[AudioChunk] = deque(maxlen=buffer_size)
        self.env_buffer: Deque[SensorReading] = deque(maxlen=buffer_size)

        # Statistics
        self.stats = {
            "video_received": 0,
            "audio_received": 0,
            "env_received": 0,
            "aligned_emitted": 0,
            "video_dropped": 0,
            "audio_dropped": 0,
            "env_dropped": 0,
        }

        logger.info(
            "temporal_aligner_initialized",
            window_size=window_size,
            jitter_tolerance=jitter_tolerance,
            buffer_size=buffer_size,
            require_all_modalities=require_all_modalities,
        )

    def add_video(self, frame: VideoFrame) -> None:
        """Add video frame to buffer.

        Args:
            frame: VideoFrame to add
        """
        self.video_buffer.append(frame)
        self.stats["video_received"] += 1

    def add_audio(self, chunk: AudioChunk) -> None:
        """Add audio chunk to buffer.

        Args:
            chunk: AudioChunk to add
        """
        self.audio_buffer.append(chunk)
        self.stats["audio_received"] += 1

    def add_environmental(self, reading: SensorReading) -> None:
        """Add environmental sensor reading to buffer.

        Args:
            reading: SensorReading to add
        """
        self.env_buffer.append(reading)
        self.stats["env_received"] += 1

    def try_align(self) -> Optional[AlignedData]:
        """Attempt to align data from all modalities.

        Returns:
            AlignedData if alignment is successful, None otherwise
        """
        # Get available items
        available_modalities = self._get_available_modalities()

        if not available_modalities:
            return None

        # If require_all_modalities, check all are present
        if self.require_all_modalities and len(available_modalities) < 3:
            return None

        # Find reference timestamp (use video if available, else audio, else env)
        ref_timestamp, ref_modality = self._get_reference_timestamp()

        if ref_timestamp is None:
            return None

        # Find closest matches for other modalities
        video_match, video_jitter = self._find_closest_video(ref_timestamp)
        audio_match, audio_jitter = self._find_closest_audio(ref_timestamp)
        env_match, env_jitter = self._find_closest_env(ref_timestamp)

        # Check if matches are within jitter tolerance
        max_jitter = 0.0

        if video_match and video_jitter > self.jitter_tolerance:
            video_match = None
        else:
            max_jitter = max(max_jitter, video_jitter)

        if audio_match and audio_jitter > self.jitter_tolerance:
            audio_match = None
        else:
            max_jitter = max(max_jitter, audio_jitter)

        if env_match and env_jitter > self.jitter_tolerance:
            env_match = None
        else:
            max_jitter = max(max_jitter, env_jitter)

        # If require_all_modalities, check all matched
        if self.require_all_modalities:
            if not (video_match and audio_match and env_match):
                return None

        # If at least one modality matched, emit aligned data
        if video_match or audio_match or env_match:
            aligned = AlignedData(
                timestamp=ref_timestamp,
                video=video_match,
                audio=audio_match,
                environmental=env_match,
                jitter=max_jitter,
            )

            # Remove matched items from buffers
            if video_match:
                self._remove_video(video_match)
            if audio_match:
                self._remove_audio(audio_match)
            if env_match:
                self._remove_env(env_match)

            self.stats["aligned_emitted"] += 1

            logger.debug(
                "data_aligned",
                timestamp=ref_timestamp,
                ref_modality=ref_modality,
                max_jitter=max_jitter,
                video_present=video_match is not None,
                audio_present=audio_match is not None,
                env_present=env_match is not None,
            )

            return aligned

        return None

    def cleanup_stale_data(self, current_time: Optional[float] = None) -> None:
        """Remove stale data that exceeds window size.

        Args:
            current_time: Current time reference (uses time.time() if None)
        """
        if current_time is None:
            current_time = time.time()

        cutoff_time = current_time - self.window_size

        # Remove stale video frames
        while self.video_buffer and self.video_buffer[0].timestamp < cutoff_time:
            self.video_buffer.popleft()
            self.stats["video_dropped"] += 1

        # Remove stale audio chunks
        while self.audio_buffer and self.audio_buffer[0].timestamp < cutoff_time:
            self.audio_buffer.popleft()
            self.stats["audio_dropped"] += 1

        # Remove stale environmental readings
        while self.env_buffer and self.env_buffer[0].timestamp < cutoff_time:
            self.env_buffer.popleft()
            self.stats["env_dropped"] += 1

    def get_buffer_status(self) -> Dict[str, int]:
        """Get current buffer sizes.

        Returns:
            Dictionary with buffer sizes for each modality
        """
        return {
            "video": len(self.video_buffer),
            "audio": len(self.audio_buffer),
            "environmental": len(self.env_buffer),
        }

    def get_stats(self) -> Dict[str, int]:
        """Get alignment statistics.

        Returns:
            Dictionary with statistics
        """
        return self.stats.copy()

    def reset(self) -> None:
        """Reset aligner state."""
        self.video_buffer.clear()
        self.audio_buffer.clear()
        self.env_buffer.clear()

        self.stats = {
            "video_received": 0,
            "audio_received": 0,
            "env_received": 0,
            "aligned_emitted": 0,
            "video_dropped": 0,
            "audio_dropped": 0,
            "env_dropped": 0,
        }

        logger.info("temporal_aligner_reset")

    # Private helper methods

    def _get_available_modalities(self) -> List[str]:
        """Get list of modalities with buffered data."""
        modalities = []
        if self.video_buffer:
            modalities.append("video")
        if self.audio_buffer:
            modalities.append("audio")
        if self.env_buffer:
            modalities.append("environmental")
        return modalities

    def _get_reference_timestamp(self) -> Tuple[Optional[float], Optional[str]]:
        """Get reference timestamp for alignment.

        Returns:
            Tuple of (timestamp, modality_name)
        """
        # Prefer video as reference (most visual context)
        if self.video_buffer:
            return self.video_buffer[0].timestamp, "video"

        # Fallback to audio
        if self.audio_buffer:
            return self.audio_buffer[0].timestamp, "audio"

        # Fallback to environmental
        if self.env_buffer:
            return self.env_buffer[0].timestamp, "environmental"

        return None, None

    def _find_closest_video(self, ref_timestamp: float) -> Tuple[Optional[VideoFrame], float]:
        """Find closest video frame to reference timestamp.

        Args:
            ref_timestamp: Reference timestamp

        Returns:
            Tuple of (VideoFrame, jitter) or (None, inf)
        """
        if not self.video_buffer:
            return None, float("inf")

        closest = min(self.video_buffer, key=lambda v: abs(v.timestamp - ref_timestamp))
        jitter = abs(closest.timestamp - ref_timestamp)
        return closest, jitter

    def _find_closest_audio(self, ref_timestamp: float) -> Tuple[Optional[AudioChunk], float]:
        """Find closest audio chunk to reference timestamp.

        Args:
            ref_timestamp: Reference timestamp

        Returns:
            Tuple of (AudioChunk, jitter) or (None, inf)
        """
        if not self.audio_buffer:
            return None, float("inf")

        closest = min(self.audio_buffer, key=lambda a: abs(a.timestamp - ref_timestamp))
        jitter = abs(closest.timestamp - ref_timestamp)
        return closest, jitter

    def _find_closest_env(self, ref_timestamp: float) -> Tuple[Optional[SensorReading], float]:
        """Find closest environmental reading to reference timestamp.

        Args:
            ref_timestamp: Reference timestamp

        Returns:
            Tuple of (SensorReading, jitter) or (None, inf)
        """
        if not self.env_buffer:
            return None, float("inf")

        closest = min(self.env_buffer, key=lambda e: abs(e.timestamp - ref_timestamp))
        jitter = abs(closest.timestamp - ref_timestamp)
        return closest, jitter

    def _remove_video(self, frame: VideoFrame) -> None:
        """Remove video frame from buffer."""
        try:
            self.video_buffer.remove(frame)
        except ValueError:
            pass  # Already removed

    def _remove_audio(self, chunk: AudioChunk) -> None:
        """Remove audio chunk from buffer."""
        try:
            self.audio_buffer.remove(chunk)
        except ValueError:
            pass  # Already removed

    def _remove_env(self, reading: SensorReading) -> None:
        """Remove environmental reading from buffer."""
        try:
            self.env_buffer.remove(reading)
        except ValueError:
            pass  # Already removed


def main() -> None:
    """Demo temporal aligner."""
    from .audio_simulator import AudioScenario, AudioSimulator
    from .env_simulator import EnvironmentalScenario, EnvironmentalSimulator
    from .video_simulator import VideoScenario, VideoSimulator

    print("Testing temporal aligner with synthetic data streams...")

    # Create simulators
    video_sim = VideoSimulator(fps=10, scenario=VideoScenario.PERSON_WALKING)
    audio_sim = AudioSimulator(sample_rate=16000, scenario=AudioScenario.NORMAL_AMBIENT)
    env_sim = EnvironmentalSimulator(polling_rate=1.0, scenario=EnvironmentalScenario.NORMAL_INDOOR)

    # Create aligner
    aligner = TemporalAligner(window_size=1.0, jitter_tolerance=0.1, require_all_modalities=False)

    # Generate data for 5 seconds
    video_gen = video_sim.generate(duration_seconds=5)
    audio_gen = audio_sim.generate(duration_seconds=5)
    env_gen = env_sim.generate(duration_seconds=5)

    aligned_count = 0

    print(f"{'Time':<10} {'Video':<8} {'Audio':<8} {'Env':<8} {'Jitter(ms)':<12}")
    print("-" * 60)

    # Feed data to aligner in interleaved manner
    for _ in range(50):  # 5 seconds at ~10 Hz
        try:
            # Add video frame
            frame = next(video_gen)
            aligner.add_video(frame)
        except StopIteration:
            pass

        try:
            # Add audio chunk
            chunk = next(audio_gen)
            aligner.add_audio(chunk)
        except StopIteration:
            pass

        try:
            # Add environmental reading
            reading = next(env_gen)
            aligner.add_environmental(reading)
        except StopIteration:
            pass

        # Try to align
        aligned = aligner.try_align()
        if aligned:
            aligned_count += 1
            print(
                f"{aligned.timestamp:>8.2f}s "
                f"{'✓' if aligned.video else '✗':^8} "
                f"{'✓' if aligned.audio else '✗':^8} "
                f"{'✓' if aligned.environmental else '✗':^8} "
                f"{aligned.jitter * 1000:>10.1f}"
            )

        # Cleanup stale data
        aligner.cleanup_stale_data()

        time.sleep(0.1)  # Simulate real-time processing

    print(f"\nAlignment Statistics:")
    stats = aligner.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    buffer_status = aligner.get_buffer_status()
    print(f"\nFinal Buffer Status:")
    for modality, count in buffer_status.items():
        print(f"  {modality}: {count} items")


if __name__ == "__main__":
    main()
