"""Export audio model to ONNX format.

This script exports a pre-trained audio model (AudioCLIP or similar) to ONNX
format for efficient edge inference.
"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn


class SimpleAudioEncoder(nn.Module):
    """Simple audio encoder for demonstration purposes.

    This is a lightweight model that processes mel-spectrograms and produces
    512D embeddings. In production, replace with AudioCLIP or similar.

    Architecture:
    - Input: (B, n_mels, T) mel-spectrogram
    - 3x Conv2D layers with batch norm and max pooling
    - Global average pooling
    - 2x FC layers
    - Output: (B, 512) L2-normalized embedding
    """

    def __init__(
        self,
        n_mels: int = 128,
        embedding_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.n_mels = n_mels
        self.embedding_dim = embedding_dim

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout)

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, embedding_dim)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input mel-spectrogram (B, n_mels, T)

        Returns:
            L2-normalized embedding (B, embedding_dim)
        """
        # Add channel dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, n_mels, T)

        # Convolutional layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)

        # Global average pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # Flatten

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        # L2 normalize
        x = nn.functional.normalize(x, p=2, dim=1)

        return x


def export_audio_model(
    output_path: Path,
    n_mels: int = 128,
    embedding_dim: int = 512,
    sample_duration: float = 1.0,
    sample_rate: int = 16000,
    hop_length: int = 512,
) -> None:
    """Export audio model to ONNX format.

    Args:
        output_path: Path to save ONNX model
        n_mels: Number of mel filterbanks
        embedding_dim: Output embedding dimension
        sample_duration: Audio chunk duration in seconds
        sample_rate: Audio sample rate
        hop_length: STFT hop length
    """
    print(f"Creating audio encoder model...")
    print(f"  n_mels: {n_mels}")
    print(f"  embedding_dim: {embedding_dim}")
    print(f"  sample_duration: {sample_duration}s")
    print(f"  sample_rate: {sample_rate} Hz")

    # Create model
    model = SimpleAudioEncoder(
        n_mels=n_mels,
        embedding_dim=embedding_dim,
        dropout=0.0,  # Disable dropout for inference
    )
    model.eval()

    # Calculate expected time frames in mel-spectrogram
    n_samples = int(sample_rate * sample_duration)
    n_frames = 1 + (n_samples - 2048) // hop_length  # Assuming n_fft=2048

    print(f"  Expected samples: {n_samples}")
    print(f"  Expected time frames: {n_frames}")

    # Create dummy input (batch_size=1, n_mels, time_frames)
    dummy_input = torch.randn(1, n_mels, n_frames)

    # Test forward pass
    with torch.no_grad():
        output = model(dummy_input)
    print(f"\n✓ Model forward pass successful")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output norm: {torch.norm(output, p=2, dim=1).item():.6f}")

    # Export to ONNX
    print(f"\nExporting to ONNX: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["mel_spectrogram"],
        output_names=["embedding"],
        dynamic_axes={
            "mel_spectrogram": {0: "batch_size", 2: "time_frames"},
            "embedding": {0: "batch_size"},
        },
        opset_version=14,
        do_constant_folding=True,
        export_params=True,
    )

    print(f"✓ ONNX export complete")

    # Validate ONNX model
    print(f"\nValidating ONNX model...")
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print(f"✓ ONNX model is valid")

    # Test ONNX Runtime inference
    print(f"\nTesting ONNX Runtime inference...")
    sess = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])

    dummy_input_np = dummy_input.numpy()
    onnx_output = sess.run(None, {"mel_spectrogram": dummy_input_np})[0]

    print(f"✓ ONNX Runtime inference successful")
    print(f"  Output shape: {onnx_output.shape}")
    print(f"  Output norm: {np.linalg.norm(onnx_output):.6f}")

    # Compare PyTorch vs ONNX outputs
    pytorch_output = output.numpy()
    max_diff = np.abs(pytorch_output - onnx_output).max()
    mean_diff = np.abs(pytorch_output - onnx_output).mean()
    cosine_sim = np.dot(pytorch_output.flatten(), onnx_output.flatten()) / (
        np.linalg.norm(pytorch_output) * np.linalg.norm(onnx_output)
    )

    print(f"\nPyTorch vs ONNX comparison:")
    print(f"  Max absolute error: {max_diff:.6e}")
    print(f"  Mean absolute error: {mean_diff:.6e}")
    print(f"  Cosine similarity: {cosine_sim:.6f}")

    if max_diff < 1e-4 and cosine_sim > 0.9999:
        print(f"\n✓ Export successful - outputs match within tolerance")
    else:
        print(f"\n⚠ Warning: Outputs differ more than expected")

    # Print model info
    model_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nModel information:")
    print(f"  File size: {model_size_mb:.2f} MB")
    print(f"  Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export audio model to ONNX")
    parser.add_argument(
        "--output",
        type=str,
        default="models/audio/model.onnx",
        help="Output path for ONNX model",
    )
    parser.add_argument(
        "--n-mels",
        type=int,
        default=128,
        help="Number of mel filterbanks",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=512,
        help="Output embedding dimension",
    )
    parser.add_argument(
        "--sample-duration",
        type=float,
        default=1.0,
        help="Audio chunk duration in seconds",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Audio sample rate in Hz",
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=512,
        help="STFT hop length",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Kizuna Audio Model ONNX Export")
    print("=" * 70)

    output_path = Path(args.output)

    export_audio_model(
        output_path=output_path,
        n_mels=args.n_mels,
        embedding_dim=args.embedding_dim,
        sample_duration=args.sample_duration,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
    )

    print(f"\n" + "=" * 70)
    print(f"Export complete!")
    print(f"=" * 70)
    print(f"\nNext steps:")
    print(f"1. Optionally run ONNX Simplifier:")
    print(f"   pip install onnx-simplifier")
    print(f"   onnxsim {output_path} {output_path}")
    print(f"\n2. Quantize to INT8:")
    print(f"   python scripts/quantize_models.py --model audio")
    print(f"\n3. Test the encoder:")
    print(f"   python -m src.engine.audio_encoder --model {output_path}")


if __name__ == "__main__":
    main()
