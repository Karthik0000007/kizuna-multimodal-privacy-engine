"""Quantize ONNX models to INT8 for edge deployment.

This script quantizes FP32 ONNX models to INT8 using dynamic or static
quantization, and validates the quantized models.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantType,
    quantize_dynamic,
    quantize_static,
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logger import get_logger

logger = get_logger("model_quantization")


class CalibrationDataReaderBase(CalibrationDataReader):
    """Base calibration data reader.

    Generates random calibration data for static quantization.
    In production, use real representative data from your dataset.
    """

    def __init__(
        self,
        num_samples: int = 500,
        input_name: str = "input",
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        data_type: str = "vision",
    ) -> None:
        """Initialize calibration data reader.

        Args:
            num_samples: Number of calibration samples
            input_name: Model input name
            input_shape: Input tensor shape
            data_type: Type of data ("vision" or "audio")
        """
        self.num_samples = num_samples
        self.input_name = input_name
        self.input_shape = input_shape
        self.data_type = data_type
        self.current_index = 0

        # Generate random calibration data
        logger.info("generating_calibration_data", num_samples=num_samples, data_type=data_type)
        print(f"Generating {num_samples} {data_type} calibration samples...")

        self.calibration_data = []
        for i in range(num_samples):
            # Generate random data
            if data_type == "vision":
                # Random image (normalized to [0, 1])
                data = np.random.rand(*input_shape).astype(np.float32)
            elif data_type == "audio":
                # Random mel-spectrogram (normalized to ~N(0, 1))
                data = np.random.randn(*input_shape).astype(np.float32)
            else:
                # Generic random data
                data = np.random.randn(*input_shape).astype(np.float32)

            self.calibration_data.append({input_name: data})

            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{num_samples} samples")

        logger.info("calibration_data_generated")

    def get_next(self) -> dict:
        """Get next calibration sample.

        Returns:
            Dictionary with input tensors
        """
        if self.current_index < len(self.calibration_data):
            sample = self.calibration_data[self.current_index]
            self.current_index += 1
            return sample
        return None


def quantize_model(
    input_path: Path,
    output_path: Path,
    model_type: str = "vision",
    quantization_mode: str = "dynamic",
    calibration_samples: int = 500,
) -> None:
    """Quantize model to INT8.

    Args:
        input_path: Input FP32 ONNX model path
        output_path: Output INT8 ONNX model path
        model_type: Type of model ("vision", "audio", "sensor")
        quantization_mode: "dynamic" or "static"
        calibration_samples: Number of calibration samples for static quantization

    Raises:
        RuntimeError: If quantization fails
    """
    logger.info(
        "quantizing_model",
        input_path=str(input_path),
        model_type=model_type,
        mode=quantization_mode,
    )
    print(f"\nQuantizing {model_type} model: {input_path}")
    print(f"Mode: {quantization_mode}")

    # Get model-specific input configuration
    if model_type == "vision":
        input_name = "pixel_values"
        input_shape = (1, 3, 224, 224)
        data_type = "vision"
    elif model_type == "audio":
        input_name = "mel_spectrogram"
        input_shape = (1, 128, 31)  # (1, n_mels, time_frames) for 1-second audio
        data_type = "audio"
    elif model_type == "sensor":
        input_name = "sensor_values"
        input_shape = (1, 5)  # 5 sensor features
        data_type = "sensor"
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    try:
        if quantization_mode == "dynamic":
            # Dynamic quantization (simpler, no calibration needed)
            print("Using dynamic quantization (weights only)...")

            quantize_dynamic(
                model_input=str(input_path),
                model_output=str(output_path),
                weight_type=QuantType.QInt8,
                per_channel=True,
                reduce_range=False,
                optimize_model=True,
            )

        elif quantization_mode == "static":
            # Static quantization (better accuracy, requires calibration)
            print("Using static quantization (weights + activations)...")

            # Create calibration data reader
            calibration_reader = CalibrationDataReaderBase(
                num_samples=calibration_samples,
                input_name=input_name,
                input_shape=input_shape,
                data_type=data_type,
            )

            quantize_static(
                model_input=str(input_path),
                model_output=str(output_path),
                calibration_data_reader=calibration_reader,
                quant_format=QuantType.QInt8,
                per_channel=True,
                reduce_range=False,
                optimize_model=True,
            )

        else:
            raise ValueError(f"Invalid quantization mode: {quantization_mode}")

        logger.info("quantization_complete", output_path=str(output_path))
        print(f"✓ Quantization complete: {output_path}")

    except Exception as e:
        logger.error("quantization_failed", error=str(e))
        raise RuntimeError(f"Quantization failed: {e}")


def compare_models(
    fp32_path: Path,
    int8_path: Path,
    model_type: str = "vision",
    num_samples: int = 100,
) -> dict:
    """Compare FP32 and INT8 model outputs.

    Args:
        fp32_path: FP32 model path
        int8_path: INT8 model path
        model_type: Type of model ("vision", "audio", "sensor")
        num_samples: Number of test samples

    Returns:
        Dictionary with comparison metrics
    """
    logger.info("comparing_models", fp32_path=str(fp32_path), int8_path=str(int8_path))
    print(f"\nComparing FP32 vs INT8 models...")

    # Load models
    fp32_session = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    int8_session = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])

    # Get input name and shape based on model type
    input_name = fp32_session.get_inputs()[0].name

    if model_type == "vision":
        input_shape = (1, 3, 224, 224)
    elif model_type == "audio":
        input_shape = (1, 128, 31)
    elif model_type == "sensor":
        input_shape = (1, 5)
    else:
        input_shape = tuple(fp32_session.get_inputs()[0].shape)

    # Compare outputs
    cosine_similarities = []
    max_diffs = []
    mean_diffs = []

    print(f"Testing on {num_samples} samples...")

    for i in range(num_samples):
        # Random input
        test_input = np.random.randn(*input_shape).astype(np.float32)

        # FP32 inference
        fp32_output = fp32_session.run(None, {input_name: test_input})[0]

        # INT8 inference
        int8_output = int8_session.run(None, {input_name: test_input})[0]

        # Compute metrics
        # Cosine similarity
        cos_sim = np.dot(fp32_output.flatten(), int8_output.flatten()) / (
            np.linalg.norm(fp32_output.flatten()) * np.linalg.norm(int8_output.flatten())
        )
        cosine_similarities.append(cos_sim)

        # Absolute differences
        max_diff = np.abs(fp32_output - int8_output).max()
        mean_diff = np.abs(fp32_output - int8_output).mean()
        max_diffs.append(max_diff)
        mean_diffs.append(mean_diff)

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{num_samples} samples")

    # Aggregate metrics
    metrics = {
        "cosine_similarity_mean": float(np.mean(cosine_similarities)),
        "cosine_similarity_min": float(np.min(cosine_similarities)),
        "max_diff_mean": float(np.mean(max_diffs)),
        "max_diff_max": float(np.max(max_diffs)),
        "mean_diff_mean": float(np.mean(mean_diffs)),
    }

    # Display results
    print("\nComparison Results:")
    print(f"  Cosine Similarity (mean): {metrics['cosine_similarity_mean']:.6f}")
    print(f"  Cosine Similarity (min):  {metrics['cosine_similarity_min']:.6f}")
    print(f"  Max Absolute Diff (mean): {metrics['max_diff_mean']:.6f}")
    print(f"  Max Absolute Diff (max):  {metrics['max_diff_max']:.6f}")
    print(f"  Mean Absolute Diff:       {metrics['mean_diff_mean']:.6f}")

    # Quality check
    if metrics["cosine_similarity_mean"] > 0.95:
        print("✓ Quality check passed (cosine similarity > 0.95)")
        logger.info("quality_check_passed", metrics=metrics)
    else:
        print("✗ Quality check failed (cosine similarity <= 0.95)")
        logger.warning("quality_check_failed", metrics=metrics)

    return metrics


def benchmark_latency(
    fp32_path: Path,
    int8_path: Path,
    model_type: str = "vision",
    num_runs: int = 100,
) -> dict:
    """Benchmark model inference latency.

    Args:
        fp32_path: FP32 model path
        int8_path: INT8 model path
        model_type: Type of model ("vision", "audio", "sensor")
        num_runs: Number of inference runs

    Returns:
        Dictionary with latency metrics
    """
    logger.info("benchmarking_latency", num_runs=num_runs)
    print(f"\nBenchmarking inference latency ({num_runs} runs)...")

    # Load models
    fp32_session = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    int8_session = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])

    # Get input name
    input_name = fp32_session.get_inputs()[0].name

    # Generate test input based on model type
    if model_type == "vision":
        test_input = np.random.rand(1, 3, 224, 224).astype(np.float32)
    elif model_type == "audio":
        test_input = np.random.randn(1, 128, 31).astype(np.float32)
    elif model_type == "sensor":
        test_input = np.random.randn(1, 5).astype(np.float32)
    else:
        input_shape = tuple(fp32_session.get_inputs()[0].shape)
        test_input = np.random.randn(*input_shape).astype(np.float32)

    # Warm-up
    for _ in range(10):
        fp32_session.run(None, {input_name: test_input})
        int8_session.run(None, {input_name: test_input})

    # Benchmark FP32
    fp32_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        fp32_session.run(None, {input_name: test_input})
        fp32_times.append((time.perf_counter() - start) * 1000)  # ms

    # Benchmark INT8
    int8_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        int8_session.run(None, {input_name: test_input})
        int8_times.append((time.perf_counter() - start) * 1000)  # ms

    # Compute metrics
    metrics = {
        "fp32_mean_ms": float(np.mean(fp32_times)),
        "fp32_p50_ms": float(np.percentile(fp32_times, 50)),
        "fp32_p95_ms": float(np.percentile(fp32_times, 95)),
        "fp32_p99_ms": float(np.percentile(fp32_times, 99)),
        "int8_mean_ms": float(np.mean(int8_times)),
        "int8_p50_ms": float(np.percentile(int8_times, 50)),
        "int8_p95_ms": float(np.percentile(int8_times, 95)),
        "int8_p99_ms": float(np.percentile(int8_times, 99)),
        "speedup": float(np.mean(fp32_times) / np.mean(int8_times)),
    }

    # Display results
    print("\nLatency Benchmarks:")
    print(
        f"  FP32 - Mean: {metrics['fp32_mean_ms']:.2f}ms, "
        f"P50: {metrics['fp32_p50_ms']:.2f}ms, "
        f"P95: {metrics['fp32_p95_ms']:.2f}ms"
    )
    print(
        f"  INT8 - Mean: {metrics['int8_mean_ms']:.2f}ms, "
        f"P50: {metrics['int8_p50_ms']:.2f}ms, "
        f"P95: {metrics['int8_p95_ms']:.2f}ms"
    )
    print(f"  Speedup: {metrics['speedup']:.2f}×")

    logger.info("latency_benchmark_complete", metrics=metrics)

    return metrics


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Quantize ONNX models to INT8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["vision", "audio", "sensor", "all"],
        default="vision",
        help="Model to quantize",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Models directory",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["dynamic", "static"],
        default="dynamic",
        help="Quantization mode (default: dynamic)",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=500,
        help="Number of calibration samples for static quantization",
    )
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Skip FP32 vs INT8 comparison",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip latency benchmarking",
    )

    args = parser.parse_args()

    models_dir = Path(args.models_dir)

    print("=" * 70)
    print("Kizuna Model Quantization")
    print("=" * 70)
    print(f"\nModels directory: {models_dir}")
    print(f"Quantization mode: {args.mode}")

    # Determine which models to quantize
    if args.model == "all":
        models_to_quantize = ["vision", "audio", "sensor"]
    else:
        models_to_quantize = [args.model]

    # Quantize each model
    for model_name in models_to_quantize:
        model_dir = models_dir / model_name
        fp32_path = model_dir / "model.onnx"
        int8_path = model_dir / "model_int8.onnx"

        if not fp32_path.exists():
            print(f"\n✗ FP32 model not found: {fp32_path}")
            print(f"  Run: python scripts/export_{model_name}_onnx.py")
            continue

        print(f"\n{'=' * 70}")
        print(f"Quantizing {model_name.upper()} Model")
        print(f"{'=' * 70}")

        # Quantize
        quantize_model(
            fp32_path,
            int8_path,
            model_type=model_name,
            quantization_mode=args.mode,
            calibration_samples=args.calibration_samples,
        )

        # Compare models
        if not args.skip_comparison:
            compare_models(fp32_path, int8_path, model_type=model_name, num_samples=100)

        # Benchmark latency
        if not args.skip_benchmark:
            benchmark_latency(fp32_path, int8_path, model_type=model_name, num_runs=100)

        # Model sizes
        fp32_size = fp32_path.stat().st_size / 1024 / 1024
        int8_size = int8_path.stat().st_size / 1024 / 1024
        size_reduction = (1 - int8_size / fp32_size) * 100

        print(f"\nModel Sizes:")
        print(f"  FP32: {fp32_size:.2f} MB")
        print(f"  INT8: {int8_size:.2f} MB")
        print(f"  Reduction: {size_reduction:.1f}%")

    print("\n" + "=" * 70)
    print("Quantization Complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Test quantized vision model:")
    print("     python -m src.engine.vision_encoder --model models/vision/model_int8.onnx")
    print("  2. Test quantized audio model:")
    print("     python -m src.engine.audio_encoder --model models/audio/model_int8.onnx")
    print("  3. Integrate into embedding pipeline")


if __name__ == "__main__":
    main()
