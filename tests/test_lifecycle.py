"""Unit tests for payload lifecycle manager."""

import time
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from src.engine.engine import UnifiedEmbedding
from src.ingestion.models import SensorPayload
from src.privacy.lifecycle import LifecycleException, LifecycleResult, PayloadLifecycle
from src.privacy.memory_wiper import WipeResult


@pytest.fixture
def mock_embedding_engine():
    """Create a mock embedding engine."""
    engine = Mock()

    # Mock embed() to return a UnifiedEmbedding
    def mock_embed(payload):
        embedding = np.random.randn(512).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        return UnifiedEmbedding(
            embedding=embedding,
            timestamp=payload.timestamp,
            camera_id=payload.camera_id,
            modalities_used=payload.get_modalities(),
            latencies_ms={
                "vision_encoding": 30.0,
                "audio_encoding": 25.0,
                "sensor_encoding": 5.0,
                "fusion": 10.0,
                "total": 70.0,
            },
        )

    engine.embed = Mock(side_effect=mock_embed)
    return engine


@pytest.fixture
def mock_dp_noise_adder():
    """Create a mock DP noise adder."""
    adder = Mock()

    def mock_add_noise(vector):
        # Add small noise
        noise = np.random.randn(*vector.shape).astype(vector.dtype) * 0.01
        return vector + noise

    adder.add_noise = Mock(side_effect=mock_add_noise)
    adder.get_privacy_guarantee = Mock(
        return_value={
            "mechanism": "laplace",
            "epsilon": 1.0,
            "delta": 0.0,
            "sensitivity": 2.0,
        }
    )

    return adder


@pytest.fixture
def sample_payload():
    """Create a sample sensor payload."""
    return SensorPayload(
        timestamp=time.time(),
        camera_id="camera_001",
        video_frame=np.random.randint(0, 256, (320, 320, 3), dtype=np.uint8),
        audio_chunk=np.random.randn(16000).astype(np.float32),
        env_data={
            "temperature": 22.5,
            "humidity": 55.0,
            "motion": 1.0,
            "light": 500.0,
            "air_quality": 120.0,
        },
    )


class TestPayloadLifecycle:
    """Tests for PayloadLifecycle class."""

    def test_initialization(self, mock_embedding_engine, mock_dp_noise_adder):
        """Test lifecycle manager initializes correctly."""
        lifecycle = PayloadLifecycle(
            embedding_engine=mock_embedding_engine,
            dp_noise_adder=mock_dp_noise_adder,
            use_native_wiper=False,  # Use Python fallback for testing
            verify_wipe=True,
            num_wipe_passes=1,
        )

        assert lifecycle.embedding_engine is mock_embedding_engine
        assert lifecycle.dp_noise_adder is mock_dp_noise_adder
        assert lifecycle.verify_wipe is True

    def test_process_complete_payload(
        self, mock_embedding_engine, mock_dp_noise_adder, sample_payload
    ):
        """Test processing a complete payload with all modalities."""
        lifecycle = PayloadLifecycle(
            embedding_engine=mock_embedding_engine,
            dp_noise_adder=mock_dp_noise_adder,
            use_native_wiper=False,
            verify_wipe=True,
        )

        # Store original values to verify they're wiped
        video_data = sample_payload.video_frame.copy()
        audio_data = sample_payload.audio_chunk.copy()

        result = lifecycle.process(sample_payload)

        # Verify result structure
        assert isinstance(result, LifecycleResult)
        assert isinstance(result.embedding, UnifiedEmbedding)
        assert result.raw_data_destroyed is True
        assert len(result.wipe_results) == 2  # video + audio
        assert result.processing_time_ms > 0

        # Verify stage latencies
        assert "embedding" in result.stage_latencies
        assert "dp_noise" in result.stage_latencies
        assert "wipe" in result.stage_latencies
        assert "total" in result.stage_latencies

        # Verify embedding engine was called
        mock_embedding_engine.embed.assert_called_once()

        # Verify DP noise was added
        mock_dp_noise_adder.add_noise.assert_called_once()

        # Verify raw data is destroyed
        assert sample_payload.video_frame is None
        assert sample_payload.audio_chunk is None
        assert sample_payload.env_data is None

    def test_process_partial_payload(self, mock_embedding_engine, mock_dp_noise_adder):
        """Test processing payload with only some modalities."""
        lifecycle = PayloadLifecycle(
            embedding_engine=mock_embedding_engine,
            dp_noise_adder=mock_dp_noise_adder,
            use_native_wiper=False,
        )

        # Payload with only video
        payload = SensorPayload(
            timestamp=time.time(),
            camera_id="camera_002",
            video_frame=np.random.randint(0, 256, (320, 320, 3), dtype=np.uint8),
        )

        result = lifecycle.process(payload)

        assert result.raw_data_destroyed is True
        assert len(result.wipe_results) == 1  # Only video
        assert payload.video_frame is None

    def test_wipe_performed_even_on_error(
        self, mock_embedding_engine, mock_dp_noise_adder, sample_payload
    ):
        """Test that wipe is performed even if embedding or DP fails."""
        lifecycle = PayloadLifecycle(
            embedding_engine=mock_embedding_engine,
            dp_noise_adder=mock_dp_noise_adder,
            use_native_wiper=False,
        )

        # Make embedding fail
        mock_embedding_engine.embed.side_effect = RuntimeError("Embedding failed")

        # Process should raise exception
        with pytest.raises(LifecycleException, match="Embedding extraction failed"):
            lifecycle.process(sample_payload)

        # But raw data should still be wiped
        assert sample_payload.video_frame is None
        assert sample_payload.audio_chunk is None
        assert sample_payload.env_data is None

    def test_dp_noise_failure_still_wipes(
        self, mock_embedding_engine, mock_dp_noise_adder, sample_payload
    ):
        """Test that wipe happens even if DP noise application fails."""
        lifecycle = PayloadLifecycle(
            embedding_engine=mock_embedding_engine,
            dp_noise_adder=mock_dp_noise_adder,
            use_native_wiper=False,
        )

        # Make DP noise fail
        mock_dp_noise_adder.add_noise.side_effect = RuntimeError("DP failed")

        with pytest.raises(LifecycleException, match="DP noise application failed"):
            lifecycle.process(sample_payload)

        # Raw data should still be wiped
        assert sample_payload.video_frame is None
        assert sample_payload.audio_chunk is None

    def test_wipe_failure_raises_critical_error(
        self, mock_embedding_engine, mock_dp_noise_adder, sample_payload
    ):
        """Test that wipe failure raises critical security exception."""
        lifecycle = PayloadLifecycle(
            embedding_engine=mock_embedding_engine,
            dp_noise_adder=mock_dp_noise_adder,
            use_native_wiper=False,
        )

        # Mock wiper to fail
        with patch.object(lifecycle.wiper, "wipe", side_effect=RuntimeError("Wipe failed")):
            with pytest.raises(LifecycleException, match="CRITICAL: Memory wipe failed"):
                lifecycle.process(sample_payload)

    def test_arrays_are_actually_wiped(
        self, mock_embedding_engine, mock_dp_noise_adder, sample_payload
    ):
        """Test that array contents are actually zeroed."""
        lifecycle = PayloadLifecycle(
            embedding_engine=mock_embedding_engine,
            dp_noise_adder=mock_dp_noise_adder,
            use_native_wiper=False,
        )

        # Keep references to the original arrays
        video_ref = sample_payload.video_frame
        audio_ref = sample_payload.audio_chunk

        # Verify arrays have non-zero data
        assert not np.all(video_ref == 0)
        assert not np.all(audio_ref == 0)

        result = lifecycle.process(sample_payload)

        # Verify arrays are now all zeros
        assert np.all(video_ref == 0)
        assert np.all(audio_ref == 0)

        # Verify wipe was successful
        assert all(wr.success for wr in result.wipe_results)
        assert all(wr.verification_passed for wr in result.wipe_results)

    def test_get_stats(self, mock_embedding_engine, mock_dp_noise_adder):
        """Test get_stats returns correct information."""
        lifecycle = PayloadLifecycle(
            embedding_engine=mock_embedding_engine,
            dp_noise_adder=mock_dp_noise_adder,
            use_native_wiper=False,
        )

        stats = lifecycle.get_stats()

        assert "wiper_type" in stats
        assert "using_native" in stats
        assert "verify_wipe" in stats
        assert "dp_mechanism" in stats
        assert "dp_epsilon" in stats

        assert stats["wiper_type"] == "SecureWiper"
        assert stats["dp_mechanism"] == "laplace"
        assert stats["dp_epsilon"] == 1.0


class TestLifecyclePerformance:
    """Performance tests for lifecycle manager."""

    def test_wipe_latency_target(self, mock_embedding_engine, mock_dp_noise_adder, sample_payload):
        """Test that wipe stage meets latency target."""
        lifecycle = PayloadLifecycle(
            embedding_engine=mock_embedding_engine,
            dp_noise_adder=mock_dp_noise_adder,
            use_native_wiper=False,
        )

        result = lifecycle.process(sample_payload)

        wipe_latency = result.stage_latencies["wipe"]

        # Target: < 5ms for wipe stage
        assert wipe_latency < 5.0, f"Wipe latency {wipe_latency:.2f}ms exceeds 5ms target"

    def test_total_overhead_target(
        self, mock_embedding_engine, mock_dp_noise_adder, sample_payload
    ):
        """Test that lifecycle overhead (excluding embedding) is acceptable."""
        lifecycle = PayloadLifecycle(
            embedding_engine=mock_embedding_engine,
            dp_noise_adder=mock_dp_noise_adder,
            use_native_wiper=False,
        )

        result = lifecycle.process(sample_payload)

        # Overhead = DP noise + wipe
        overhead = result.stage_latencies["dp_noise"] + result.stage_latencies["wipe"]

        # Target: < 10ms overhead
        assert overhead < 10.0, f"Lifecycle overhead {overhead:.2f}ms exceeds 10ms target"

    def test_multiple_payloads_no_memory_leak(self, mock_embedding_engine, mock_dp_noise_adder):
        """Test processing multiple payloads doesn't cause memory leaks."""
        lifecycle = PayloadLifecycle(
            embedding_engine=mock_embedding_engine,
            dp_noise_adder=mock_dp_noise_adder,
            use_native_wiper=False,
        )

        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Process 100 payloads
        for _ in range(100):
            payload = SensorPayload(
                timestamp=time.time(),
                camera_id="camera_test",
                video_frame=np.random.randint(0, 256, (320, 320, 3), dtype=np.uint8),
                audio_chunk=np.random.randn(16000).astype(np.float32),
                env_data={"temperature": 22.0},
            )

            result = lifecycle.process(payload)
            assert result.raw_data_destroyed

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory

        # Memory growth should be < 10 MB
        assert memory_growth < 10.0, f"Memory grew by {memory_growth:.2f} MB (possible leak)"


class TestLifecycleAuditLog:
    """Test audit logging for lifecycle events."""

    def test_lifecycle_events_logged(
        self, mock_embedding_engine, mock_dp_noise_adder, sample_payload, caplog
    ):
        """Test that lifecycle events are logged."""
        import logging

        caplog.set_level(logging.INFO)

        lifecycle = PayloadLifecycle(
            embedding_engine=mock_embedding_engine,
            dp_noise_adder=mock_dp_noise_adder,
            use_native_wiper=False,
        )

        lifecycle.process(sample_payload)

        # Check for key log messages
        log_messages = [record.message for record in caplog.records]

        # Should have lifecycle started and complete messages
        assert any("lifecycle_started" in msg or "started" in msg.lower() for msg in log_messages)
        assert any("lifecycle_complete" in msg or "complete" in msg.lower() for msg in log_messages)

    def test_wipe_results_logged(
        self, mock_embedding_engine, mock_dp_noise_adder, sample_payload, caplog
    ):
        """Test that wipe results are logged."""
        import logging

        caplog.set_level(logging.INFO)

        lifecycle = PayloadLifecycle(
            embedding_engine=mock_embedding_engine,
            dp_noise_adder=mock_dp_noise_adder,
            use_native_wiper=False,
        )

        lifecycle.process(sample_payload)

        # Should have wipe complete log
        log_messages = [record.message for record in caplog.records]
        assert any("wipe" in msg.lower() for msg in log_messages)


class TestLifecycleEdgeCases:
    """Test edge cases and error conditions."""

    def test_payload_with_numpy_array_in_env_data(self, mock_embedding_engine, mock_dp_noise_adder):
        """Test payload with NumPy arrays in env_data."""
        lifecycle = PayloadLifecycle(
            embedding_engine=mock_embedding_engine,
            dp_noise_adder=mock_dp_noise_adder,
            use_native_wiper=False,
        )

        payload = SensorPayload(
            timestamp=time.time(),
            camera_id="camera_003",
            video_frame=np.random.randint(0, 256, (320, 320, 3), dtype=np.uint8),
            env_data={
                "temperature": 22.0,
                "sensor_array": np.random.randn(100).astype(np.float32),  # NumPy array in env_data
            },
        )

        result = lifecycle.process(payload)

        # Should wipe video + sensor_array
        assert len(result.wipe_results) >= 2
        assert payload.video_frame is None
        assert payload.env_data is None

    def test_empty_payload_fields(self, mock_embedding_engine, mock_dp_noise_adder):
        """Test payload with empty arrays."""
        lifecycle = PayloadLifecycle(
            embedding_engine=mock_embedding_engine,
            dp_noise_adder=mock_dp_noise_adder,
            use_native_wiper=False,
        )

        payload = SensorPayload(
            timestamp=time.time(),
            camera_id="camera_004",
            video_frame=np.array([], dtype=np.uint8).reshape(0, 0, 3),  # Empty video
        )

        result = lifecycle.process(payload)

        assert result.raw_data_destroyed is True
        assert payload.video_frame is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
