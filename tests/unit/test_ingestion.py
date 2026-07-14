"""Unit tests for ingestion pipeline."""

import numpy as np
import pytest

from src.ingestion import (
    AlignedData,
    AudioChunk,
    AudioScenario,
    AudioSimulator,
    BackpressurePolicy,
    EnvironmentalScenario,
    EnvironmentalSimulator,
    PayloadAssembler,
    SensorPayload,
    SensorReading,
    TemporalAligner,
    VideoFrame,
    VideoScenario,
    VideoSimulator,
)


class TestVideoSimulator:
    """Tests for VideoSimulator."""

    def test_init_valid_params(self) -> None:
        """Test initialization with valid parameters."""
        sim = VideoSimulator(fps=15, resolution=(320, 320))
        assert sim.fps == 15
        assert sim.resolution == (320, 320)
        assert sim.frame_number == 0

    def test_init_invalid_fps(self) -> None:
        """Test initialization with invalid FPS."""
        with pytest.raises(ValueError, match="FPS must be between"):
            VideoSimulator(fps=100)

    def test_init_invalid_resolution(self) -> None:
        """Test initialization with invalid resolution."""
        with pytest.raises(ValueError, match="Invalid resolution"):
            VideoSimulator(fps=15, resolution=(320,))

    def test_generate_empty_room(self) -> None:
        """Test empty room scenario generation."""
        sim = VideoSimulator(fps=10, scenario=VideoScenario.EMPTY_ROOM)
        gen = sim.generate(duration_seconds=0.5)

        frame_obj = next(gen)
        assert isinstance(frame_obj, VideoFrame)
        assert frame_obj.frame.shape == (320, 320, 3)
        assert frame_obj.frame.dtype == np.uint8
        assert frame_obj.scenario == VideoScenario.EMPTY_ROOM

    def test_generate_person_walking(self) -> None:
        """Test person walking scenario generation."""
        sim = VideoSimulator(fps=10, scenario=VideoScenario.PERSON_WALKING)
        gen = sim.generate(duration_seconds=0.5)

        frame_count = 0
        for frame_obj in gen:
            assert isinstance(frame_obj.frame, np.ndarray)
            assert frame_obj.frame.shape == (320, 320, 3)
            frame_count += 1

        assert frame_count == 5  # 0.5s at 10 FPS

    def test_generate_duration(self) -> None:
        """Test generation with specified duration."""
        sim = VideoSimulator(fps=5)
        gen = sim.generate(duration_seconds=1.0)

        frames = list(gen)
        assert len(frames) == 5  # 1s at 5 FPS

    def test_frame_incrementing(self) -> None:
        """Test frame numbers increment correctly."""
        sim = VideoSimulator(fps=10)
        gen = sim.generate(duration_seconds=0.3)

        frame_numbers = [f.frame_number for f in gen]
        assert frame_numbers == [0, 1, 2]

    def test_reset(self) -> None:
        """Test simulator reset."""
        sim = VideoSimulator(fps=10)
        gen = sim.generate(duration_seconds=0.2)
        list(gen)  # Consume generator

        assert sim.frame_number > 0

        sim.reset()
        assert sim.frame_number == 0


class TestAudioSimulator:
    """Tests for AudioSimulator."""

    def test_init_valid_params(self) -> None:
        """Test initialization with valid parameters."""
        sim = AudioSimulator(sample_rate=16000, chunk_duration=1.0)
        assert sim.sample_rate == 16000
        assert sim.chunk_duration == 1.0
        assert sim.samples_per_chunk == 16000

    def test_init_invalid_sample_rate(self) -> None:
        """Test initialization with invalid sample rate."""
        with pytest.raises(ValueError, match="Sample rate must be between"):
            AudioSimulator(sample_rate=4000)

    def test_init_invalid_duration(self) -> None:
        """Test initialization with invalid chunk duration."""
        with pytest.raises(ValueError, match="Chunk duration must be between"):
            AudioSimulator(chunk_duration=15.0)

    def test_generate_silence(self) -> None:
        """Test silence scenario generation."""
        sim = AudioSimulator(scenario=AudioScenario.SILENCE, enable_noise=False)
        gen = sim.generate(duration_seconds=1.0)

        chunk_obj = next(gen)
        assert isinstance(chunk_obj, AudioChunk)
        assert len(chunk_obj.audio) == sim.samples_per_chunk
        assert chunk_obj.audio.dtype == np.float32
        assert np.abs(chunk_obj.audio).max() < 0.1  # Nearly silent

    def test_generate_normal_ambient(self) -> None:
        """Test normal ambient scenario generation."""
        sim = AudioSimulator(scenario=AudioScenario.NORMAL_AMBIENT)
        gen = sim.generate(duration_seconds=1.0)

        chunk_obj = next(gen)
        assert isinstance(chunk_obj.audio, np.ndarray)
        assert len(chunk_obj.audio) == sim.samples_per_chunk
        assert -1.0 <= chunk_obj.audio.max() <= 1.0  # Normalized

    def test_generate_duration(self) -> None:
        """Test generation with specified duration."""
        sim = AudioSimulator(chunk_duration=1.0)
        gen = sim.generate(duration_seconds=2.0)

        chunks = list(gen)
        assert len(chunks) == 2  # 2s with 1s chunks

    def test_chunk_incrementing(self) -> None:
        """Test chunk numbers increment correctly."""
        sim = AudioSimulator(chunk_duration=1.0)
        gen = sim.generate(duration_seconds=3.0)

        chunk_numbers = [c.chunk_number for c in gen]
        assert chunk_numbers == [0, 1, 2]

    def test_reset(self) -> None:
        """Test simulator reset."""
        sim = AudioSimulator()
        gen = sim.generate(duration_seconds=1.0)
        list(gen)

        assert sim.chunk_number > 0

        sim.reset()
        assert sim.chunk_number == 0


class TestEnvironmentalSimulator:
    """Tests for EnvironmentalSimulator."""

    def test_init_valid_params(self) -> None:
        """Test initialization with valid parameters."""
        sim = EnvironmentalSimulator(polling_rate=1.0)
        assert sim.polling_rate == 1.0
        assert sim.reading_number == 0

    def test_init_invalid_polling_rate(self) -> None:
        """Test initialization with invalid polling rate."""
        with pytest.raises(ValueError, match="Polling rate must be between"):
            EnvironmentalSimulator(polling_rate=100.0)

    def test_generate_normal_indoor(self) -> None:
        """Test normal indoor scenario generation."""
        sim = EnvironmentalSimulator(scenario=EnvironmentalScenario.NORMAL_INDOOR)
        gen = sim.generate(duration_seconds=1.0)

        reading_obj = next(gen)
        assert isinstance(reading_obj, SensorReading)
        assert 15.0 <= reading_obj.temperature <= 40.0
        assert 20.0 <= reading_obj.humidity <= 90.0
        assert reading_obj.motion in [0, 1]
        assert reading_obj.light >= 0
        assert 0.0 <= reading_obj.air_quality <= 500.0

    def test_value_ranges(self) -> None:
        """Test sensor values are within valid ranges."""
        sim = EnvironmentalSimulator()
        gen = sim.generate(duration_seconds=3.0)

        for reading in gen:
            assert 15.0 <= reading.temperature <= 40.0
            assert 20.0 <= reading.humidity <= 90.0
            assert reading.motion in [0, 1]
            assert reading.light >= 0
            assert 0.0 <= reading.air_quality <= 500.0

    def test_generate_duration(self) -> None:
        """Test generation with specified duration."""
        sim = EnvironmentalSimulator(polling_rate=1.0)
        gen = sim.generate(duration_seconds=3.0)

        readings = list(gen)
        assert len(readings) == 3  # 3s at 1s polling

    def test_reset(self) -> None:
        """Test simulator reset."""
        sim = EnvironmentalSimulator()
        gen = sim.generate(duration_seconds=2.0)
        list(gen)

        assert sim.reading_number > 0

        sim.reset()
        assert sim.reading_number == 0


class TestSensorPayload:
    """Tests for SensorPayload."""

    def test_valid_payload_all_modalities(self) -> None:
        """Test payload with all modalities."""
        payload = SensorPayload(
            timestamp=1.0,
            camera_id="test",
            video_frame=np.zeros((224, 224, 3), dtype=np.uint8),
            audio_chunk=np.zeros(16000, dtype=np.float32),
            env_data={"temperature": 22.0, "humidity": 50.0},
        )

        assert payload.is_complete()
        assert set(payload.get_modalities()) == {"video", "audio", "environmental"}

    def test_valid_payload_partial(self) -> None:
        """Test payload with partial modalities."""
        payload = SensorPayload(
            timestamp=1.0,
            camera_id="test",
            video_frame=np.zeros((224, 224, 3), dtype=np.uint8),
        )

        assert not payload.is_complete()
        assert payload.get_modalities() == ["video"]

    def test_invalid_payload_no_modalities(self) -> None:
        """Test payload with no modalities raises error."""
        with pytest.raises(ValueError, match="must contain at least one modality"):
            SensorPayload(timestamp=1.0, camera_id="test")

    def test_get_size_bytes(self) -> None:
        """Test payload size estimation."""
        payload = SensorPayload(
            timestamp=1.0,
            camera_id="test",
            video_frame=np.zeros((224, 224, 3), dtype=np.uint8),
            audio_chunk=np.zeros(16000, dtype=np.float32),
        )

        size = payload.get_size_bytes()
        assert size > 0
        assert size == payload.video_frame.nbytes + payload.audio_chunk.nbytes + 100


class TestTemporalAligner:
    """Tests for TemporalAligner."""

    def test_init_valid_params(self) -> None:
        """Test initialization with valid parameters."""
        aligner = TemporalAligner(window_size=1.0, jitter_tolerance=0.05)
        assert aligner.window_size == 1.0
        assert aligner.jitter_tolerance == 0.05

    def test_init_invalid_window_size(self) -> None:
        """Test initialization with invalid window size."""
        with pytest.raises(ValueError, match="Window size must be positive"):
            TemporalAligner(window_size=-1.0)

    def test_init_invalid_jitter(self) -> None:
        """Test initialization with invalid jitter tolerance."""
        with pytest.raises(ValueError, match="Jitter tolerance must be non-negative"):
            TemporalAligner(jitter_tolerance=-0.1)

    def test_add_and_buffer(self) -> None:
        """Test adding data to buffers."""
        aligner = TemporalAligner()

        video = VideoFrame(
            frame=np.zeros((224, 224, 3), dtype=np.uint8),
            timestamp=1.0,
            frame_number=0,
            scenario=VideoScenario.EMPTY_ROOM,
            metadata={},
        )
        aligner.add_video(video)

        status = aligner.get_buffer_status()
        assert status["video"] == 1
        assert status["audio"] == 0
        assert status["environmental"] == 0

    def test_try_align_single_modality(self) -> None:
        """Test alignment with single modality."""
        aligner = TemporalAligner(require_all_modalities=False)

        video = VideoFrame(
            frame=np.zeros((224, 224, 3), dtype=np.uint8),
            timestamp=1.0,
            frame_number=0,
            scenario=VideoScenario.EMPTY_ROOM,
            metadata={},
        )
        aligner.add_video(video)

        aligned = aligner.try_align()
        assert aligned is not None
        assert aligned.video is not None
        assert aligned.audio is None
        assert aligned.environmental is None

    def test_try_align_all_modalities(self) -> None:
        """Test alignment with all modalities."""
        aligner = TemporalAligner(require_all_modalities=False, jitter_tolerance=0.1)

        timestamp = 1.0

        video = VideoFrame(
            frame=np.zeros((224, 224, 3), dtype=np.uint8),
            timestamp=timestamp,
            frame_number=0,
            scenario=VideoScenario.EMPTY_ROOM,
            metadata={},
        )
        audio = AudioChunk(
            audio=np.zeros(16000, dtype=np.float32),
            timestamp=timestamp + 0.01,  # Small jitter
            chunk_number=0,
            sample_rate=16000,
            scenario=AudioScenario.SILENCE,
            metadata={},
        )
        env = SensorReading(
            temperature=22.0,
            humidity=50.0,
            motion=0,
            light=400.0,
            air_quality=50.0,
            timestamp=timestamp - 0.01,  # Small jitter
            reading_number=0,
            scenario=EnvironmentalScenario.NORMAL_INDOOR,
            metadata={},
        )

        aligner.add_video(video)
        aligner.add_audio(audio)
        aligner.add_environmental(env)

        aligned = aligner.try_align()
        assert aligned is not None
        assert aligned.video is not None
        assert aligned.audio is not None
        assert aligned.environmental is not None
        assert aligned.jitter <= 0.1

    def test_cleanup_stale_data(self) -> None:
        """Test cleanup of stale data."""
        aligner = TemporalAligner(window_size=0.5)

        old_video = VideoFrame(
            frame=np.zeros((224, 224, 3), dtype=np.uint8),
            timestamp=0.0,
            frame_number=0,
            scenario=VideoScenario.EMPTY_ROOM,
            metadata={},
        )
        aligner.add_video(old_video)

        # Cleanup with current time >> old timestamp
        aligner.cleanup_stale_data(current_time=2.0)

        status = aligner.get_buffer_status()
        assert status["video"] == 0  # Old frame removed

    def test_reset(self) -> None:
        """Test aligner reset."""
        aligner = TemporalAligner()

        video = VideoFrame(
            frame=np.zeros((224, 224, 3), dtype=np.uint8),
            timestamp=1.0,
            frame_number=0,
            scenario=VideoScenario.EMPTY_ROOM,
            metadata={},
        )
        aligner.add_video(video)

        aligner.reset()

        status = aligner.get_buffer_status()
        assert all(v == 0 for v in status.values())


class TestPayloadAssembler:
    """Tests for PayloadAssembler."""

    def test_init_valid_params(self) -> None:
        """Test initialization with valid parameters."""
        assembler = PayloadAssembler(camera_id="test", max_queue_size=100)
        assert assembler.camera_id == "test"
        assert assembler.max_queue_size == 100

    def test_init_invalid_queue_size(self) -> None:
        """Test initialization with invalid queue size."""
        with pytest.raises(ValueError, match="Max queue size must be at least 1"):
            PayloadAssembler(max_queue_size=0)

    def test_assemble_complete_payload(self) -> None:
        """Test assembling complete payload."""
        assembler = PayloadAssembler()

        aligned = AlignedData(
            timestamp=1.0,
            video=VideoFrame(
                frame=np.zeros((224, 224, 3), dtype=np.uint8),
                timestamp=1.0,
                frame_number=0,
                scenario=VideoScenario.EMPTY_ROOM,
                metadata={},
            ),
            audio=AudioChunk(
                audio=np.zeros(16000, dtype=np.float32),
                timestamp=1.0,
                chunk_number=0,
                sample_rate=16000,
                scenario=AudioScenario.SILENCE,
                metadata={},
            ),
            environmental=SensorReading(
                temperature=22.0,
                humidity=50.0,
                motion=0,
                light=400.0,
                air_quality=50.0,
                timestamp=1.0,
                reading_number=0,
                scenario=EnvironmentalScenario.NORMAL_INDOOR,
                metadata={},
            ),
            jitter=0.01,
        )

        payload = assembler.assemble(aligned)
        assert payload is not None
        assert payload.is_complete()
        assert assembler.stats.complete_payloads == 1

    def test_assemble_partial_payload(self) -> None:
        """Test assembling partial payload."""
        assembler = PayloadAssembler(allow_partial_payloads=True)

        aligned = AlignedData(
            timestamp=1.0,
            video=VideoFrame(
                frame=np.zeros((224, 224, 3), dtype=np.uint8),
                timestamp=1.0,
                frame_number=0,
                scenario=VideoScenario.EMPTY_ROOM,
                metadata={},
            ),
            audio=None,
            environmental=None,
            jitter=0.0,
        )

        payload = assembler.assemble(aligned)
        assert payload is not None
        assert not payload.is_complete()
        assert assembler.stats.partial_payloads == 1
        assert assembler.stats.video_only_payloads == 1

    def test_backpressure_newest_wins(self) -> None:
        """Test backpressure with newest wins policy."""
        assembler = PayloadAssembler(
            max_queue_size=2, backpressure_policy=BackpressurePolicy.NEWEST_WINS
        )

        # Create 3 payloads (queue size is 2)
        for i in range(3):
            payload = SensorPayload(
                timestamp=float(i),
                camera_id="test",
                video_frame=np.zeros((224, 224, 3), dtype=np.uint8),
            )
            assembler.enqueue(payload)

        # Should have dropped oldest
        assert assembler.get_queue_size() == 2
        assert assembler.stats.payloads_dropped >= 1

    def test_get_stats(self) -> None:
        """Test statistics tracking."""
        assembler = PayloadAssembler()

        aligned = AlignedData(
            timestamp=1.0,
            video=VideoFrame(
                frame=np.zeros((224, 224, 3), dtype=np.uint8),
                timestamp=1.0,
                frame_number=0,
                scenario=VideoScenario.EMPTY_ROOM,
                metadata={},
            ),
            audio=None,
            environmental=None,
            jitter=0.05,
        )

        assembler.assemble(aligned)

        stats = assembler.get_stats()
        assert stats["payloads_created"] == 1
        assert stats["partial_payloads"] == 1
        assert stats["average_jitter_ms"] == 50.0  # 0.05s = 50ms

    def test_reset(self) -> None:
        """Test assembler reset."""
        assembler = PayloadAssembler()

        payload = SensorPayload(
            timestamp=1.0,
            camera_id="test",
            video_frame=np.zeros((224, 224, 3), dtype=np.uint8),
        )
        assembler.enqueue(payload)

        assembler.reset()

        assert assembler.get_queue_size() == 0
        assert assembler.stats.payloads_created == 0
