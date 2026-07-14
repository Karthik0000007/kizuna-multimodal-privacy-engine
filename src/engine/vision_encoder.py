"""Vision embedding encoder using ONNX Runtime.

Extracts dense embeddings from video frames using a pre-trained vision model.
"""

import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from ..logger import get_engine_logger

logger = get_engine_logger()


class VisionEncoder:
    """Vision embedding extractor using ONNX Runtime.
    
    Loads a pre-trained vision model (ONNX format) and extracts embeddings
    from RGB images. Supports both FP32 and INT8 quantized models.
    """

    def __init__(
        self,
        model_path: str | Path,
        execution_provider: str = "CPUExecutionProvider",
        intra_op_num_threads: int = 2,
        inter_op_num_threads: int = 1,
        input_size: Tuple[int, int] = (224, 224),
        normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        channel_order: str = "CHW",
    ) -> None:
        """Initialize vision encoder.
        
        Args:
            model_path: Path to ONNX model file
            execution_provider: ONNX Runtime execution provider
            intra_op_num_threads: Number of intra-op threads
            inter_op_num_threads: Number of inter-op threads
            input_size: Input image size (width, height)
            normalize_mean: ImageNet normalization mean (RGB)
            normalize_std: ImageNet normalization std (RGB)
            channel_order: Channel order ("CHW" or "HWC")
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model cannot be loaded
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model_path = model_path
        self.input_size = input_size
        self.normalize_mean = np.array(normalize_mean, dtype=np.float32).reshape(1, 1, 3)
        self.normalize_std = np.array(normalize_std, dtype=np.float32).reshape(1, 1, 3)
        self.channel_order = channel_order
        
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
                "vision_encoder_initialized",
                model_path=str(model_path),
                execution_provider=execution_provider,
                embedding_dim=self.embedding_dim,
                input_size=input_size,
            )
            
        except Exception as e:
            logger.error("vision_encoder_init_failed", error=str(e))
            raise RuntimeError(f"Failed to load ONNX model: {e}")
    
    def encode(self, frame: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Extract embedding from video frame.
        
        Args:
            frame: RGB image as numpy array (H, W, 3), dtype=uint8, range [0, 255]
            
        Returns:
            Embedding vector (D,), dtype=float32, L2-normalized
            
        Raises:
            ValueError: If frame shape or dtype is invalid
        """
        # Validate input
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected frame shape (H, W, 3), got {frame.shape}")
        if frame.dtype != np.uint8:
            raise ValueError(f"Expected frame dtype uint8, got {frame.dtype}")
        
        start_time = time.perf_counter()
        
        # Preprocess
        processed = self._preprocess(frame)
        
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
            logger.error("vision_encoding_failed", error=str(e))
            raise RuntimeError(f"Vision encoding failed: {e}")
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        logger.debug(
            "vision_encoded",
            latency_ms=elapsed_ms,
            embedding_dim=len(embedding),
            embedding_norm=float(np.linalg.norm(embedding)),
        )
        
        return embedding
    
    def encode_batch(self, frames: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Extract embeddings from batch of frames.
        
        Args:
            frames: Batch of RGB images (B, H, W, 3), dtype=uint8, range [0, 255]
            
        Returns:
            Embedding matrix (B, D), dtype=float32, L2-normalized per row
            
        Raises:
            ValueError: If frames shape or dtype is invalid
        """
        # Validate input
        if frames.ndim != 4 or frames.shape[3] != 3:
            raise ValueError(f"Expected frames shape (B, H, W, 3), got {frames.shape}")
        if frames.dtype != np.uint8:
            raise ValueError(f"Expected frames dtype uint8, got {frames.dtype}")
        
        start_time = time.perf_counter()
        
        batch_size = frames.shape[0]
        
        # Preprocess batch
        processed = np.stack([self._preprocess(frame) for frame in frames])
        processed = processed.squeeze(1)  # Remove extra batch dim from each frame
        
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
            logger.error("vision_batch_encoding_failed", error=str(e))
            raise RuntimeError(f"Vision batch encoding failed: {e}")
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        logger.debug(
            "vision_batch_encoded",
            batch_size=batch_size,
            latency_ms=elapsed_ms,
            latency_per_frame_ms=elapsed_ms / batch_size,
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
        logger.info("vision_encoder_warmup_started", num_iterations=num_iterations)
        
        # Create dummy input
        dummy_frame = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        
        latencies = []
        for i in range(num_iterations):
            start = time.perf_counter()
            self.encode(dummy_frame)
            latencies.append((time.perf_counter() - start) * 1000)
        
        avg_latency = np.mean(latencies)
        
        logger.info("vision_encoder_warmup_complete", avg_latency_ms=avg_latency)
        
        return avg_latency
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension.
        
        Returns:
            Embedding dimension
        """
        return self.embedding_dim
    
    def _preprocess(self, frame: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Preprocess frame for model input.
        
        Args:
            frame: RGB image (H, W, 3), uint8, [0, 255]
            
        Returns:
            Preprocessed tensor (1, C, H, W) or (1, H, W, C), float32, normalized
        """
        # Resize
        resized = cv2.resize(frame, self.input_size)
        
        # Convert to float [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # Normalize with ImageNet mean/std
        normalized = (normalized - self.normalize_mean) / self.normalize_std
        
        # Convert to correct channel order
        if self.channel_order == "CHW":
            # HWC -> CHW
            normalized = np.transpose(normalized, (2, 0, 1))
        elif self.channel_order == "HWC":
            # Already HWC
            pass
        else:
            raise ValueError(f"Invalid channel order: {self.channel_order}")
        
        # Add batch dimension
        preprocessed = np.expand_dims(normalized, axis=0)
        
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
    """Demo vision encoder."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Kizuna Vision Encoder Demo")
    parser.add_argument(
        "--model",
        type=str,
        default="models/vision/model_int8.onnx",
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to test image (optional)",
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
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Kizuna Vision Encoder Demo")
    print("=" * 70)
    
    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"\n✗ Model not found: {model_path}")
        print("  Run: python scripts/export_vision_onnx.py")
        print("  Then: python scripts/quantize_models.py --model vision")
        return
    
    # Initialize encoder
    print(f"\nLoading model: {model_path}")
    encoder = VisionEncoder(model_path=model_path)
    print(f"✓ Model loaded")
    print(f"  Embedding dimension: {encoder.get_embedding_dim()}")
    
    # Warm up
    print(f"\nWarming up ({args.warmup} iterations)...")
    avg_warmup_latency = encoder.warm_up(num_iterations=args.warmup)
    print(f"✓ Warm-up complete")
    print(f"  Average latency: {avg_warmup_latency:.2f}ms")
    
    # Test with image or random data
    if args.image and Path(args.image).exists():
        print(f"\nLoading test image: {args.image}")
        frame = cv2.imread(args.image)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print(f"  Image shape: {frame.shape}")
    else:
        print(f"\nGenerating random test image...")
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    
    # Single inference
    print(f"\nRunning single inference...")
    start = time.perf_counter()
    embedding = encoder.encode(frame)
    latency = (time.perf_counter() - start) * 1000
    
    print(f"✓ Inference complete")
    print(f"  Latency: {latency:.2f}ms")
    print(f"  Embedding shape: {embedding.shape}")
    print(f"  Embedding norm: {np.linalg.norm(embedding):.6f}")
    print(f"  Embedding range: [{embedding.min():.6f}, {embedding.max():.6f}]")
    
    # Batch inference
    if args.batch_size > 1:
        print(f"\nRunning batch inference (batch_size={args.batch_size})...")
        frames = np.stack([frame] * args.batch_size)
        
        start = time.perf_counter()
        embeddings = encoder.encode_batch(frames)
        batch_latency = (time.perf_counter() - start) * 1000
        
        print(f"✓ Batch inference complete")
        print(f"  Total latency: {batch_latency:.2f}ms")
        print(f"  Per-frame latency: {batch_latency / args.batch_size:.2f}ms")
        print(f"  Embeddings shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
