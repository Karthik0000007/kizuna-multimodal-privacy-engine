"""Audio embedding encoder using ONNX Runtime.

Extracts dense embeddings from audio chunks using a pre-trained audio model.
"""

import time
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from ..logger import get_engine_logger

logger = get_engine_logger()


class AudioEncoder:
    """Audio embedding extractor using ONNX Runtime.

    Loads a pre-trained audio model (ONNX format) and extracts embeddings
    from raw audio waveforms. Supports both FP32 and INT8 quantized models.
    """

    def __init__(
        self,
        model_path: str | Path,
        execution_provider: str = "CPUExecutionProvider",
        intra_op_num_threads: int = 2,
        inter_op_num_threads: int = 1,
        target_sample_rate: int = 16000,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        chunk_duration: float = 1.0,
    ) -> None:
        """Initialize audio encoder.

        Args:
            model_path: Path to ONNX model file
            execution_provider: ONNX Runtime execution provider
            intra_op_num_threads: Number of intra-op threads
            inter_op_num_threads: Number of inter-op threads
            target_sample_rate: Target audio sample rate in Hz
            n_mels: Number of mel filterbanks
            n_fft: FFT window size
            hop_length: Hop length for STFT
            chunk_duration: Expected audio chunk duration in seconds

        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model cannot be loaded
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model_path = model_path
        self.target_sample_rate = target_sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.chunk_duration = chunk_duration
        self.expected_samples = int(target_sample_rate * chunk_duration)

        # ONNX Runtime session options
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = intra_op_num_threads
        sess_options.inter_op_num_threads = inter_op_num_threads
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Load ONNX model
        try:
            self.session = ort.InferenceSession(
                str(model_path),
                sess_options=sess_options,
                providers=[execution_provider],
            )

            # Get input/output names
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

            # Get output shape (embedding dimension)
            output_shape = self.session.get_outputs()[0].shape
            self.embedding_dim = output_shape[-1]  # Last dimension

            logger.info(
                "audio_encoder_initialized",
                model_path=str(model_path),
                execution_provider=execution_provider,
                embedding_dim=self.embedding_dim,
                target_sample_rate=target_sample_rate,
                n_mels=n_mels,
            )

        except Exception as e:
            logger.error("audio_encoder_init_failed", error=str(e))
            raise RuntimeError(f"Failed to load ONNX model: {e}") from e

    def encode(self, audio_chunk: NDArray[np.float32]) -> NDArray[np.float32]:
        """Extract embedding from audio chunk.

        Args:
            audio_chunk: Audio waveform as numpy array (N,), dtype=float32, range [-1, 1]
                        Expected length: sample_rate * chunk_duration

        Returns:
            Embedding vector (D,), dtype=float32, L2-normalized

        Raises:
            ValueError: If audio_chunk shape or dtype is invalid
        """
        # Validate input
        if audio_chunk.ndim != 1:
            raise ValueError(f"Expected audio_chunk shape (N,), got {audio_chunk.shape}")
        if audio_chunk.dtype != np.float32:
            raise ValueError(f"Expected audio_chunk dtype float32, got {audio_chunk.dtype}")

        start_time = time.perf_counter()

        # Preprocess
        processed = self._preprocess(audio_chunk)

        # Run inference
        try:
            embedding = self.session.run(
                [self.output_name],
                {self.input_name: processed},
            )[0]

            # Squeeze batch dimension
            embedding = embedding.squeeze(0)

            # Ensure float32
            embedding = embedding.astype(np.float32)

            # L2 normalize
            embedding = self._normalize_l2(embedding)

        except Exception as e:
            logger.error("audio_encoding_failed", error=str(e))
            raise RuntimeError(f"Audio encoding failed: {e}") from e

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        logger.debug(
            "audio_encoded",
            latency_ms=elapsed_ms,
            embedding_dim=len(embedding),
            embedding_norm=float(np.linalg.norm(embedding)),
            audio_duration=len(audio_chunk) / self.target_sample_rate,
        )

        return embedding

    def encode_batch(self, audio_chunks: NDArray[np.float32]) -> NDArray[np.float32]:
        """Extract embeddings from batch of audio chunks.

        Args:
            audio_chunks: Batch of audio waveforms (B, N), dtype=float32, range [-1, 1]

        Returns:
            Embedding matrix (B, D), dtype=float32, L2-normalized per row

        Raises:
            ValueError: If audio_chunks shape or dtype is invalid
        """
        # Validate input
        if audio_chunks.ndim != 2:
            raise ValueError(f"Expected audio_chunks shape (B, N), got {audio_chunks.shape}")
        if audio_chunks.dtype != np.float32:
            raise ValueError(f"Expected audio_chunks dtype float32, got {audio_chunks.dtype}")

        start_time = time.perf_counter()

        batch_size = audio_chunks.shape[0]

        # Preprocess batch
        processed = np.stack([self._preprocess(chunk) for chunk in audio_chunks])
        processed = processed.squeeze(1)  # Remove extra batch dim from each chunk

        # Run inference
        try:
            embeddings = self.session.run(
                [self.output_name],
                {self.input_name: processed},
            )[0]

            # Ensure float32
            embeddings = embeddings.astype(np.float32)

            # L2 normalize each row
            embeddings = np.apply_along_axis(self._normalize_l2, 1, embeddings)

        except Exception as e:
            logger.error("audio_batch_encoding_failed", error=str(e))
            raise RuntimeError(f"Audio batch encoding failed: {e}") from e

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        logger.debug(
            "audio_batch_encoded",
            batch_size=batch_size,
            latency_ms=elapsed_ms,
            latency_per_chunk_ms=elapsed_ms / batch_size,
        )

        return embeddings

    def warm_up(self, num_iterations: int = 10) -> float:
        """Warm up the model with dummy inference runs.

        This triggers JIT compilation and optimizations for faster
        subsequent inference.

        Args:
            num_iterations: Number of warm-up iterations

        Returns:
            Average warm-up latency in milliseconds
        """
        logger.info("audio_encoder_warmup_started", num_iterations=num_iterations)

        # Create dummy input
        dummy_audio = np.random.randn(self.expected_samples).astype(np.float32)

        latencies = []
        for _i in range(num_iterations):
            start = time.perf_counter()
            self.encode(dummy_audio)
            latencies.append((time.perf_counter() - start) * 1000)

        avg_latency = np.mean(latencies)

        logger.info("audio_encoder_warmup_complete", avg_latency_ms=avg_latency)

        return avg_latency

    def get_embedding_dim(self) -> int:
        """Get embedding dimension.

        Returns:
            Embedding dimension
        """
        return self.embedding_dim

    def _preprocess(self, audio_chunk: NDArray[np.float32]) -> NDArray[np.float32]:
        """Preprocess audio chunk for model input.

        Args:
            audio_chunk: Audio waveform (N,), float32, [-1, 1]

        Returns:
            Preprocessed tensor (1, n_mels, T), float32, mel-spectrogram
        """
        # Resample if needed (for now, assume already at target rate)
        # In production, use librosa.resample if source rate differs

        # Pad or trim to expected length
        if len(audio_chunk) < self.expected_samples:
            # Pad with zeros
            audio_chunk = np.pad(
                audio_chunk,
                (0, self.expected_samples - len(audio_chunk)),
                mode="constant",
                constant_values=0.0,
            )
        elif len(audio_chunk) > self.expected_samples:
            # Trim to expected length
            audio_chunk = audio_chunk[: self.expected_samples]

        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_chunk,
            sr=self.target_sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0,  # Power spectrogram
        )

        # Convert to log scale (dB)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize to zero mean, unit variance
        log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + 1e-8)

        # Add batch dimension: (1, n_mels, T)
        preprocessed = np.expand_dims(log_mel_spec, axis=0).astype(np.float32)

        return preprocessed

    @staticmethod
    def _normalize_l2(vector: NDArray[np.float32]) -> NDArray[np.float32]:
        """L2 normalize vector.

        Args:
            vector: Input vector

        Returns:
            L2-normalized vector
        """
        norm = np.linalg.norm(vector)
        if norm < 1e-8:
            return vector
        return vector / norm


def main() -> None:
    """Demo audio encoder."""
    import argparse

    parser = argparse.ArgumentParser(description="Kizuna Audio Encoder Demo")
    parser.add_argument(
        "--model",
        type=str,
        default="models/audio/model_int8.onnx",
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--audio",
        type=str,
        help="Path to test audio file (optional)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for testing",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warm-up iterations",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Audio chunk duration in seconds",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Kizuna Audio Encoder Demo")
    print("=" * 70)

    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"\n✗ Model not found: {model_path}")
        print("  Run: python scripts/export_audio_onnx.py")
        print("  Then: python scripts/quantize_models.py --model audio")
        return

    # Initialize encoder
    print(f"\nLoading model: {model_path}")
    encoder = AudioEncoder(model_path=model_path, chunk_duration=args.duration)
    print("✓ Model loaded")
    print(f"  Embedding dimension: {encoder.get_embedding_dim()}")
    print(f"  Target sample rate: {encoder.target_sample_rate} Hz")
    print(f"  Expected samples: {encoder.expected_samples}")

    # Warm up
    print(f"\nWarming up ({args.warmup} iterations)...")
    avg_warmup_latency = encoder.warm_up(num_iterations=args.warmup)
    print("✓ Warm-up complete")
    print(f"  Average latency: {avg_warmup_latency:.2f}ms")

    # Test with audio file or random data
    if args.audio and Path(args.audio).exists():
        print(f"\nLoading test audio: {args.audio}")
        audio_chunk, sr = librosa.load(
            args.audio, sr=encoder.target_sample_rate, duration=args.duration, mono=True
        )
        print(f"  Audio shape: {audio_chunk.shape}")
        print(f"  Sample rate: {sr} Hz")
        print(f"  Duration: {len(audio_chunk) / sr:.2f}s")
    else:
        print("\nGenerating random test audio...")
        audio_chunk = np.random.randn(encoder.expected_samples).astype(np.float32)
        audio_chunk = audio_chunk / (np.abs(audio_chunk).max() + 1e-8)  # Normalize to [-1, 1]

    # Single inference
    print("\nRunning single inference...")
    start = time.perf_counter()
    embedding = encoder.encode(audio_chunk)
    latency = (time.perf_counter() - start) * 1000

    print("✓ Inference complete")
    print(f"  Latency: {latency:.2f}ms")
    print(f"  Embedding shape: {embedding.shape}")
    print(f"  Embedding norm: {np.linalg.norm(embedding):.6f}")
    print(f"  Embedding range: [{embedding.min():.6f}, {embedding.max():.6f}]")

    # Batch inference
    if args.batch_size > 1:
        print(f"\nRunning batch inference (batch_size={args.batch_size})...")
        audio_chunks = np.stack([audio_chunk] * args.batch_size)

        start = time.perf_counter()
        embeddings = encoder.encode_batch(audio_chunks)
        batch_latency = (time.perf_counter() - start) * 1000

        print("✓ Batch inference complete")
        print(f"  Total latency: {batch_latency:.2f}ms")
        print(f"  Per-chunk latency: {batch_latency / args.batch_size:.2f}ms")
        print(f"  Embeddings shape: {embeddings.shape}")

    # Show mel-spectrogram info
    print("\nMel-spectrogram configuration:")
    print(f"  n_mels: {encoder.n_mels}")
    print(f"  n_fft: {encoder.n_fft}")
    print(f"  hop_length: {encoder.hop_length}")

    mel_spec = librosa.feature.melspectrogram(
        y=audio_chunk,
        sr=encoder.target_sample_rate,
        n_fft=encoder.n_fft,
        hop_length=encoder.hop_length,
        n_mels=encoder.n_mels,
        power=2.0,
    )
    print(f"  Mel-spec shape: {mel_spec.shape}")
    print(f"  Time frames: {mel_spec.shape[1]}")


if __name__ == "__main__":
    main()
