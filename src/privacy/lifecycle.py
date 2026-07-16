"""Payload lifecycle manager for guaranteed raw data destruction.

Orchestrates the complete processing flow from raw sensor data through embedding
generation and DP noise application to secure memory destruction. Ensures raw
data never exists in memory after processing completes.

This is the critical component for APPI compliance - it guarantees that personal
information (raw sensor data) is destroyed immediately after it's no longer needed.
"""

import time
from dataclasses import dataclass

import numpy as np

from ..engine.engine import EmbeddingEngine, UnifiedEmbedding
from ..ingestion.models import SensorPayload
from ..logger import get_logger
from .dp_noise import DPNoiseAdder
from .memory_wiper import SecureWiper, WipeResult
from .memory_wiper_native import NativeSecureWiper

logger = get_logger("privacy")


@dataclass
class LifecycleResult:
    """Result of payload lifecycle processing.

    Attributes:
        embedding: Final privacy-preserved embedding
        raw_data_destroyed: Whether raw data was successfully destroyed
        wipe_results: List of wipe results for each array
        processing_time_ms: Total processing time in milliseconds
        stage_latencies: Latency breakdown by stage
    """

    embedding: UnifiedEmbedding
    raw_data_destroyed: bool
    wipe_results: list[WipeResult]
    processing_time_ms: float
    stage_latencies: dict[str, float]


class LifecycleException(Exception):
    """Exception raised when lifecycle processing fails."""

    pass


class PayloadLifecycle:
    """Payload lifecycle manager.

    Orchestrates the complete processing pipeline:
    1. Receive SensorPayload (raw data)
    2. Extract embeddings via EmbeddingEngine
    3. Apply differential privacy noise
    4. Securely wipe all raw data arrays
    5. Set all raw data fields to None
    6. Return only the privacy-preserved embedding

    Enforces the invariant that raw data never exists in memory after step 4.
    All operations are exception-safe - raw data is wiped even if an error occurs.
    """

    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        dp_noise_adder: DPNoiseAdder,
        use_native_wiper: bool = True,
        verify_wipe: bool = True,
        num_wipe_passes: int = 1,
    ) -> None:
        """Initialize payload lifecycle manager.

        Args:
            embedding_engine: Engine for extracting embeddings
            dp_noise_adder: Mechanism for adding DP noise
            use_native_wiper: Whether to use native C++ wiper (auto-fallback if unavailable)
            verify_wipe: Whether to verify wipe success
            num_wipe_passes: Number of overwrite passes (1-10)

        Raises:
            ValueError: If num_wipe_passes is out of range
        """
        self.embedding_engine = embedding_engine
        self.dp_noise_adder = dp_noise_adder

        # Initialize wiper
        if use_native_wiper:
            self.wiper: SecureWiper | NativeSecureWiper = NativeSecureWiper(
                verify=verify_wipe,
                num_passes=num_wipe_passes,
            )
        else:
            self.wiper = SecureWiper(
                verify=verify_wipe,
                num_passes=num_wipe_passes,
            )

        self.verify_wipe = verify_wipe

        logger.info(
            "payload_lifecycle_initialized",
            wiper_type=type(self.wiper).__name__,
            verify_wipe=verify_wipe,
            num_wipe_passes=num_wipe_passes,
        )

    def process(self, payload: SensorPayload) -> LifecycleResult:
        """Process a sensor payload through the complete lifecycle.

        This is the main entry point. It guarantees that raw data is destroyed
        even if an exception occurs during processing.

        Args:
            payload: Raw sensor payload to process

        Returns:
            LifecycleResult containing the final embedding and processing metadata

        Raises:
            LifecycleException: If processing fails critically
        """
        start_time = time.perf_counter()
        stage_latencies = {}

        logger.info(
            "lifecycle_started",
            timestamp=payload.timestamp,
            camera_id=payload.camera_id,
            modalities=payload.get_modalities(),
        )

        unified_embedding = None
        noised_embedding = None
        wipe_results = []
        exception_occurred = None

        try:
            # Stage 1: Extract embeddings
            stage_start = time.perf_counter()
            try:
                unified_embedding = self.embedding_engine.embed(payload)
                stage_latencies["embedding"] = (time.perf_counter() - stage_start) * 1000

                logger.debug(
                    "lifecycle_embedding_complete",
                    latency_ms=stage_latencies["embedding"],
                )
            except Exception as e:
                logger.error("lifecycle_embedding_failed", error=str(e))
                exception_occurred = e
                raise LifecycleException(f"Embedding extraction failed: {e}") from e

            # Stage 2: Apply DP noise
            stage_start = time.perf_counter()
            try:
                noised_vector = self.dp_noise_adder.add_noise(unified_embedding.embedding)

                # Create noised embedding with updated vector
                noised_embedding = UnifiedEmbedding(
                    embedding=noised_vector,
                    timestamp=unified_embedding.timestamp,
                    camera_id=unified_embedding.camera_id,
                    modalities_used=unified_embedding.modalities_used,
                    latencies_ms=unified_embedding.latencies_ms.copy(),
                )

                stage_latencies["dp_noise"] = (time.perf_counter() - stage_start) * 1000

                logger.debug(
                    "lifecycle_dp_noise_complete",
                    latency_ms=stage_latencies["dp_noise"],
                )
            except Exception as e:
                logger.error("lifecycle_dp_noise_failed", error=str(e))
                exception_occurred = e
                raise LifecycleException(f"DP noise application failed: {e}") from e

        finally:
            # Stage 3: CRITICAL - Wipe raw data (ALWAYS executed, even on error)
            stage_start = time.perf_counter()
            try:
                wipe_results = self._wipe_payload(payload)
                stage_latencies["wipe"] = (time.perf_counter() - stage_start) * 1000

                logger.info(
                    "lifecycle_wipe_complete",
                    num_arrays_wiped=len(wipe_results),
                    total_bytes_wiped=sum(r.size_bytes for r in wipe_results),
                    latency_ms=stage_latencies["wipe"],
                )
            except Exception as wipe_error:
                # Wipe failure is CRITICAL - log as error
                logger.error(
                    "lifecycle_wipe_failed_critical",
                    error=str(wipe_error),
                    original_exception=str(exception_occurred) if exception_occurred else None,
                )

                # If wipe fails, this is a security violation
                # Re-raise as LifecycleException even if there was a previous exception
                raise LifecycleException(
                    f"CRITICAL: Memory wipe failed: {wipe_error}. "
                    f"Raw data may still exist in memory!"
                ) from wipe_error

            # Stage 4: Nullify references
            self._nullify_payload_fields(payload)

            logger.debug("lifecycle_fields_nullified")

        # If we had an exception during processing (before wipe), re-raise it now
        if exception_occurred:
            raise exception_occurred

        # Calculate total time
        total_time_ms = (time.perf_counter() - start_time) * 1000
        stage_latencies["total"] = total_time_ms

        logger.info(
            "lifecycle_complete",
            timestamp=payload.timestamp,
            camera_id=payload.camera_id,
            total_time_ms=total_time_ms,
        )

        return LifecycleResult(
            embedding=noised_embedding,
            raw_data_destroyed=True,
            wipe_results=wipe_results,
            processing_time_ms=total_time_ms,
            stage_latencies=stage_latencies,
        )

    def _wipe_payload(self, payload: SensorPayload) -> list[WipeResult]:
        """Wipe all raw data arrays in the payload.

        Args:
            payload: Payload containing arrays to wipe

        Returns:
            List of WipeResult for each array

        Raises:
            Exception: If wipe fails
        """
        arrays_to_wipe = []

        # Collect all NumPy arrays
        if payload.video_frame is not None:
            arrays_to_wipe.append(("video_frame", payload.video_frame))

        if payload.audio_chunk is not None:
            arrays_to_wipe.append(("audio_chunk", payload.audio_chunk))

        # Note: env_data is typically a dict of floats, not arrays
        # But we should handle it if it contains NumPy arrays
        if payload.env_data is not None:
            for key, value in payload.env_data.items():
                if isinstance(value, np.ndarray):
                    arrays_to_wipe.append((f"env_data.{key}", value))

        logger.debug(
            "wiping_payload_arrays",
            num_arrays=len(arrays_to_wipe),
        )

        # Wipe all arrays
        results = []
        for name, array in arrays_to_wipe:
            logger.debug("wiping_array", name=name, shape=array.shape, dtype=array.dtype)
            result = self.wiper.wipe(array)
            results.append(result)

        return results

    def _nullify_payload_fields(self, payload: SensorPayload) -> None:
        """Set all raw data fields to None.

        This ensures references to the raw data are removed, allowing
        Python's garbage collector to reclaim the memory.

        Args:
            payload: Payload to nullify
        """
        payload.video_frame = None
        payload.audio_chunk = None
        payload.env_data = None

    def get_stats(self) -> dict:
        """Get statistics about the lifecycle manager.

        Returns:
            Dictionary with configuration and status info
        """
        return {
            "wiper_type": type(self.wiper).__name__,
            "using_native": (
                self.wiper.using_native if isinstance(self.wiper, NativeSecureWiper) else False
            ),
            "verify_wipe": self.verify_wipe,
            "dp_mechanism": self.dp_noise_adder.get_privacy_guarantee()["mechanism"],
            "dp_epsilon": self.dp_noise_adder.get_privacy_guarantee()["epsilon"],
        }


def main() -> None:
    """Demo payload lifecycle manager."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Kizuna Payload Lifecycle Demo")
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Base directory containing all models",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0,
        help="Privacy budget (epsilon)",
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=2.0,
        help="L2 sensitivity",
    )
    parser.add_argument(
        "--num-payloads",
        type=int,
        default=10,
        help="Number of payloads to process",
    )
    parser.add_argument(
        "--use-native",
        action="store_true",
        help="Use native C++ wiper",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Kizuna Payload Lifecycle Demo")
    print("=" * 70)

    # Check if models exist
    models_dir = Path(args.models_dir)

    vision_model = models_dir / "vision" / "model_int8.onnx"
    audio_model = models_dir / "audio" / "model_int8.onnx"
    sensor_model = models_dir / "sensor" / "model_int8.onnx"
    fusion_512 = models_dir / "fusion" / "projection_512.onnx"
    fusion_1024 = models_dir / "fusion" / "projection_1024.onnx"
    fusion_1536 = models_dir / "fusion" / "projection_1536.onnx"

    missing_models = []
    for model_path in [
        vision_model,
        audio_model,
        sensor_model,
        fusion_512,
        fusion_1024,
        fusion_1536,
    ]:
        if not model_path.exists():
            missing_models.append(model_path)

    if missing_models:
        print("\n✗ Models not found:")
        for path in missing_models:
            print(f"  {path}")
        print("\nPlease run model export scripts first.")
        return

    # Initialize embedding engine
    print("\nInitializing embedding engine...")
    embedding_engine = EmbeddingEngine(
        vision_model_path=vision_model,
        audio_model_path=audio_model,
        sensor_model_path=sensor_model,
        fusion_model_512_path=fusion_512,
        fusion_model_1024_path=fusion_1024,
        fusion_model_1536_path=fusion_1536,
    )
    print("✓ Embedding engine initialized")

    # Initialize DP noise adder
    print("\nInitializing DP noise adder...")
    print(f"  Epsilon: {args.epsilon}")
    print(f"  Sensitivity: {args.sensitivity}")

    dp_adder = DPNoiseAdder(
        mechanism="laplace",
        epsilon=args.epsilon,
        sensitivity=args.sensitivity,
    )
    print("✓ DP noise adder initialized")

    # Initialize lifecycle manager
    print("\nInitializing lifecycle manager...")
    print(f"  Native wiper: {args.use_native}")

    lifecycle = PayloadLifecycle(
        embedding_engine=embedding_engine,
        dp_noise_adder=dp_adder,
        use_native_wiper=args.use_native,
        verify_wipe=True,
        num_wipe_passes=1,
    )

    stats = lifecycle.get_stats()
    print("✓ Lifecycle manager initialized")
    print(f"  Wiper: {stats['wiper_type']}")
    print(f"  Using native: {stats['using_native']}")
    print(f"  DP mechanism: {stats['dp_mechanism']}")

    # Process test payloads
    print(f"\n{'=' * 70}")
    print(f"Processing {args.num_payloads} Test Payloads")
    print(f"{'=' * 70}")

    total_latencies = []
    wipe_latencies = []

    for i in range(args.num_payloads):
        # Create sensor payload with raw data
        payload = SensorPayload(
            timestamp=time.time(),
            camera_id=f"camera_{i % 3}",
            video_frame=np.random.randint(0, 256, (320, 320, 3), dtype=np.uint8),
            audio_chunk=np.random.randn(16000).astype(np.float32),
            env_data={
                "temperature": 20.0 + i,
                "humidity": 50.0 + i * 2,
                "motion": float(i % 2),
                "light": 500.0 + i * 10,
                "air_quality": 100.0 + i * 5,
            },
        )

        # Verify payload has data before processing
        assert payload.video_frame is not None
        assert payload.audio_chunk is not None
        assert payload.env_data is not None

        # Process through lifecycle
        result = lifecycle.process(payload)

        # Verify raw data is destroyed
        assert payload.video_frame is None, "video_frame should be None after lifecycle"
        assert payload.audio_chunk is None, "audio_chunk should be None after lifecycle"
        assert payload.env_data is None, "env_data should be None after lifecycle"
        assert result.raw_data_destroyed

        total_latencies.append(result.processing_time_ms)
        wipe_latencies.append(result.stage_latencies["wipe"])

        if i == 0:
            # Print detailed info for first payload
            print(f"\nPayload {i + 1}:")
            print("  Stage latencies:")
            for stage, latency in result.stage_latencies.items():
                print(f"    {stage}: {latency:.2f}ms")
            print("  Wipe results:")
            for wipe_result in result.wipe_results:
                print(
                    f"    {wipe_result.array_shape} {wipe_result.array_dtype}: "
                    f"{wipe_result.size_bytes / 1024:.2f} KB in {wipe_result.duration_ms:.3f}ms"
                )
            print(f"  Final embedding shape: {result.embedding.embedding.shape}")
            print(f"  Final embedding norm: {np.linalg.norm(result.embedding.embedding):.6f}")

    # Summary statistics
    print(f"\n{'=' * 70}")
    print(f"Processing Summary ({args.num_payloads} payloads)")
    print(f"{'=' * 70}")
    print("Total Processing:")
    print(f"  Mean: {np.mean(total_latencies):.2f}ms")
    print(f"  Median: {np.percentile(total_latencies, 50):.2f}ms")
    print(f"  P95: {np.percentile(total_latencies, 95):.2f}ms")
    print(f"  P99: {np.percentile(total_latencies, 99):.2f}ms")

    print("\nWipe Operations:")
    print(f"  Mean: {np.mean(wipe_latencies):.2f}ms")
    print(f"  Median: {np.percentile(wipe_latencies, 50):.2f}ms")
    print(f"  P95: {np.percentile(wipe_latencies, 95):.2f}ms")
    print(f"  P99: {np.percentile(wipe_latencies, 99):.2f}ms")

    # Check wipe performance target
    target_wipe = 5.0  # ms
    mean_wipe = np.mean(wipe_latencies)

    print("\nWipe Performance Target:")
    print(f"  Target: < {target_wipe}ms")
    print(f"  Actual: {mean_wipe:.2f}ms")

    if mean_wipe < target_wipe:
        print("  ✓ Target met!")
    else:
        print(f"  ✗ Target missed by {mean_wipe - target_wipe:.2f}ms")

    print("\n✓ All raw data successfully destroyed")
    print("✓ Demo complete")


if __name__ == "__main__":
    main()
