"""Export vision model to ONNX format.

This script downloads a pre-trained vision model (MobileCLIP or ResNet-18 fallback),
exports it to ONNX format, and validates the export.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxsim
import torch
import torch.nn as nn
from torch import Tensor

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logger import get_logger

logger = get_logger("model_export")


class SimpleCLIPVision(nn.Module):
    """Simplified CLIP-style vision encoder for demo purposes.
    
    This is a lightweight demonstration model. In production, use actual
    pre-trained models from HuggingFace (MobileCLIP, CLIP-ViT, etc.).
    """

    def __init__(self, embed_dim: int = 512) -> None:
        """Initialize simple vision encoder.
        
        Args:
            embed_dim: Output embedding dimension
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Simple CNN backbone
        self.backbone = nn.Sequential(
            # Conv block 1: 224x224x3 -> 112x112x32
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Conv block 2: 112x112x32 -> 56x56x64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Conv block 3: 56x56x64 -> 28x28x128
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Conv block 4: 28x28x128 -> 14x14x256
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Conv block 5: 14x14x256 -> 7x7x512
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Global average pooling: 7x7x512 -> 1x1x512
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        # Projection head to embed_dim
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, pixel_values: Tensor) -> Tensor:
        """Forward pass.
        
        Args:
            pixel_values: Input images (B, 3, 224, 224)
            
        Returns:
            Embeddings (B, embed_dim)
        """
        features = self.backbone(pixel_values)
        embeddings = self.projection(features)
        
        # L2 normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings


def download_mobileclip() -> nn.Module:
    """Download MobileCLIP model from HuggingFace.
    
    Returns:
        PyTorch model
        
    Raises:
        ImportError: If transformers not installed
        RuntimeError: If model cannot be downloaded
    """
    try:
        from transformers import CLIPVisionModel
        
        logger.info("downloading_mobileclip")
        print("Downloading MobileCLIP-S2 from HuggingFace...")
        
        model = CLIPVisionModel.from_pretrained("apple/mobileclip-s2-224")
        
        logger.info("mobileclip_downloaded")
        return model
        
    except ImportError:
        logger.error("transformers_not_installed")
        raise ImportError(
            "transformers library not installed. "
            "Install with: pip install transformers"
        )
    except Exception as e:
        logger.error("mobileclip_download_failed", error=str(e))
        raise RuntimeError(f"Failed to download MobileCLIP: {e}")


def download_resnet18() -> nn.Module:
    """Download ResNet-18 as fallback model.
    
    Returns:
        PyTorch model
    """
    try:
        import torchvision.models as models
        
        logger.info("downloading_resnet18")
        print("Downloading ResNet-18 as fallback...")
        
        # Load pre-trained ResNet-18
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Remove classification head, keep feature extractor
        # ResNet-18 outputs 512-dim features from avgpool layer
        model = nn.Sequential(*list(resnet.children())[:-1])  # Remove FC layer
        
        # Add projection to ensure 512-dim output with normalization
        class ResNetWrapper(nn.Module):
            def __init__(self, backbone: nn.Module) -> None:
                super().__init__()
                self.backbone = backbone
                self.projection = nn.Sequential(
                    nn.Flatten(),
                    nn.LayerNorm(512),
                )
            
            def forward(self, x: Tensor) -> Tensor:
                features = self.backbone(x)
                embeddings = self.projection(features)
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                return embeddings
        
        wrapped_model = ResNetWrapper(model)
        
        logger.info("resnet18_downloaded")
        return wrapped_model
        
    except Exception as e:
        logger.error("resnet18_download_failed", error=str(e))
        raise RuntimeError(f"Failed to download ResNet-18: {e}")


def create_simple_model(embed_dim: int = 512) -> nn.Module:
    """Create simple demonstration model.
    
    Args:
        embed_dim: Output embedding dimension
        
    Returns:
        PyTorch model
    """
    logger.info("creating_simple_model", embed_dim=embed_dim)
    print(f"Creating simple demonstration model ({embed_dim}-dim)...")
    
    model = SimpleCLIPVision(embed_dim=embed_dim)
    
    logger.info("simple_model_created")
    return model


def export_to_onnx(
    model: nn.Module,
    output_path: Path,
    input_size: tuple = (1, 3, 224, 224),
    opset_version: int = 14,
) -> None:
    """Export PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model to export
        output_path: Output ONNX file path
        input_size: Input tensor size (batch, channels, height, width)
        opset_version: ONNX opset version
        
    Raises:
        RuntimeError: If export fails
    """
    logger.info("exporting_to_onnx", output_path=str(output_path))
    print(f"\nExporting to ONNX: {output_path}")
    
    # Set model to eval mode
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(*input_size)
    
    # Export to ONNX
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            input_names=["pixel_values"],
            output_names=["embeddings"],
            dynamic_axes={
                "pixel_values": {0: "batch_size"},
                "embeddings": {0: "batch_size"},
            },
            opset_version=opset_version,
            do_constant_folding=True,
        )
        
        logger.info("onnx_export_complete")
        print(f"✓ ONNX export complete: {output_path}")
        
    except Exception as e:
        logger.error("onnx_export_failed", error=str(e))
        raise RuntimeError(f"ONNX export failed: {e}")


def simplify_onnx(input_path: Path, output_path: Path) -> None:
    """Simplify ONNX model using onnx-simplifier.
    
    Args:
        input_path: Input ONNX file
        output_path: Output simplified ONNX file
        
    Raises:
        RuntimeError: If simplification fails
    """
    logger.info("simplifying_onnx", input_path=str(input_path))
    print(f"\nSimplifying ONNX model...")
    
    try:
        # Load ONNX model
        model = onnx.load(str(input_path))
        
        # Simplify
        model_simplified, check = onnxsim.simplify(model)
        
        if not check:
            logger.warning("onnx_simplification_check_failed")
            print("⚠ Simplification check failed, using original model")
            model_simplified = model
        
        # Save simplified model
        onnx.save(model_simplified, str(output_path))
        
        logger.info("onnx_simplification_complete")
        print(f"✓ ONNX simplification complete: {output_path}")
        
    except Exception as e:
        logger.error("onnx_simplification_failed", error=str(e))
        print(f"⚠ Simplification failed: {e}")
        print(f"  Using original model at: {input_path}")


def validate_onnx_export(
    pytorch_model: nn.Module,
    onnx_path: Path,
    tolerance: float = 1e-4,
) -> bool:
    """Validate ONNX export matches PyTorch output.
    
    Args:
        pytorch_model: Original PyTorch model
        onnx_path: Path to ONNX model
        tolerance: Maximum allowed absolute error
        
    Returns:
        True if validation passes
    """
    logger.info("validating_onnx_export", onnx_path=str(onnx_path))
    print(f"\nValidating ONNX export...")
    
    try:
        import onnxruntime as ort
        
        # Create test input
        test_input = torch.randn(1, 3, 224, 224)
        
        # PyTorch inference
        pytorch_model.eval()
        with torch.no_grad():
            pytorch_output = pytorch_model(test_input).numpy()
        
        # ONNX inference
        ort_session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        onnx_output = ort_session.run(
            None,
            {"pixel_values": test_input.numpy()},
        )[0]
        
        # Compare outputs
        max_diff = np.abs(pytorch_output - onnx_output).max()
        mean_diff = np.abs(pytorch_output - onnx_output).mean()
        
        print(f"  Output shape: {onnx_output.shape}")
        print(f"  Max absolute difference: {max_diff:.6f}")
        print(f"  Mean absolute difference: {mean_diff:.6f}")
        
        if max_diff < tolerance:
            logger.info("onnx_validation_passed", max_diff=max_diff)
            print(f"✓ Validation passed (max diff: {max_diff:.6f} < {tolerance})")
            return True
        else:
            logger.warning("onnx_validation_failed", max_diff=max_diff, tolerance=tolerance)
            print(f"✗ Validation failed (max diff: {max_diff:.6f} >= {tolerance})")
            return False
            
    except Exception as e:
        logger.error("onnx_validation_error", error=str(e))
        print(f"✗ Validation error: {e}")
        return False


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Export vision model to ONNX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["mobileclip", "resnet18", "simple"],
        default="simple",
        help="Model to export (default: simple demo model)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/vision",
        help="Output directory for ONNX models",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=512,
        help="Embedding dimension (default: 512)",
    )
    parser.add_argument(
        "--skip-simplify",
        action="store_true",
        help="Skip ONNX simplification step",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation step",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Kizuna Vision Model ONNX Export")
    print("=" * 70)
    print(f"\nModel: {args.model}")
    print(f"Output directory: {output_dir}")
    print(f"Embedding dimension: {args.embed_dim}")
    
    # Download/create model
    if args.model == "mobileclip":
        try:
            model = download_mobileclip()
        except Exception as e:
            print(f"\n✗ MobileCLIP download failed: {e}")
            print("  Falling back to simple model...")
            model = create_simple_model(args.embed_dim)
    elif args.model == "resnet18":
        model = download_resnet18()
    else:  # simple
        model = create_simple_model(args.embed_dim)
    
    # Export to ONNX
    onnx_path = output_dir / "model.onnx"
    export_to_onnx(model, onnx_path)
    
    # Simplify ONNX (optional)
    if not args.skip_simplify:
        simplified_path = output_dir / "model_simplified.onnx"
        simplify_onnx(onnx_path, simplified_path)
        onnx_path = simplified_path  # Use simplified version for validation
    
    # Validate export (optional)
    if not args.skip_validation:
        validation_passed = validate_onnx_export(model, onnx_path)
        if not validation_passed:
            print("\n⚠ Warning: Validation failed. Check model outputs.")
    
    print("\n" + "=" * 70)
    print("ONNX Export Complete!")
    print("=" * 70)
    print(f"\nExported model: {onnx_path}")
    print(f"Model size: {onnx_path.stat().st_size / 1024 / 1024:.2f} MB")
    print("\nNext steps:")
    print("  1. Run: python scripts/quantize_models.py --model vision")
    print("  2. Test: python -c \"import onnxruntime; print('ONNX Runtime OK')\"")
    

if __name__ == "__main__":
    main()
