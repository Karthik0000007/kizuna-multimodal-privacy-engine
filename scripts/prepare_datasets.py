"""Dataset preparation script for Kizuna Privacy Engine.

Downloads and prepares public datasets:
- UP-Fall Detection Dataset
- ESC-50 Environmental Sound Classification
- UrbanSound8K (optional)
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logger import get_logger

logger = get_logger("dataset_prep")


def download_up_fall(data_dir: Path) -> None:
    """Download UP-Fall Detection Dataset.

    Args:
        data_dir: Data directory path
    """
    up_fall_dir = data_dir / "raw" / "up-fall"
    up_fall_dir.mkdir(parents=True, exist_ok=True)

    logger.info("preparing_up_fall_dataset", path=str(up_fall_dir))

    print("\n" + "=" * 70)
    print("UP-Fall Detection Dataset")
    print("=" * 70)
    print("\nDataset Information:")
    print("  - Source: University of Pretoria")
    print("  - URL: http://www.up.ac.za/upfall")
    print("  - License: Research use only")
    print("  - Size: ~2 GB")
    print("  - Contains: Accelerometer + camera data for fall detection")

    print("\nManual Download Required:")
    print("  1. Visit: http://www.up.ac.za/upfall")
    print("  2. Download the dataset")
    print("  3. Extract to:", str(up_fall_dir))
    print("  4. Re-run this script with --prepare-only flag")

    logger.info("up_fall_manual_download_required")


def download_esc50(data_dir: Path) -> None:
    """Download ESC-50 Environmental Sound Classification Dataset.

    Args:
        data_dir: Data directory path
    """
    esc50_dir = data_dir / "raw" / "esc-50"
    esc50_dir.mkdir(parents=True, exist_ok=True)

    logger.info("preparing_esc50_dataset", path=str(esc50_dir))

    print("\n" + "=" * 70)
    print("ESC-50 Environmental Sound Classification")
    print("=" * 70)
    print("\nDataset Information:")
    print("  - Source: Karol J. Piczak")
    print("  - URL: https://github.com/karolpiczak/ESC-50")
    print("  - License: Creative Commons Attribution Non-Commercial")
    print("  - Size: ~600 MB")
    print("  - Contains: 2000 environmental audio recordings (5s each)")

    print("\nManual Download Required:")
    print("  1. Visit: https://github.com/karolpiczak/ESC-50")
    print("  2. Clone or download the repository")
    print("  3. Copy audio files to:", str(esc50_dir))
    print("  4. Re-run this script with --prepare-only flag")

    logger.info("esc50_manual_download_required")


def download_urbansound8k(data_dir: Path) -> None:
    """Download UrbanSound8K Dataset (optional).

    Args:
        data_dir: Data directory path
    """
    urban_dir = data_dir / "raw" / "urbansound8k"
    urban_dir.mkdir(parents=True, exist_ok=True)

    logger.info("preparing_urbansound8k_dataset", path=str(urban_dir))

    print("\n" + "=" * 70)
    print("UrbanSound8K (Optional)")
    print("=" * 70)
    print("\nDataset Information:")
    print("  - Source: NYU")
    print("  - URL: https://urbansounddataset.weebly.com/urbansound8k.html")
    print("  - License: Creative Commons Attribution Non-Commercial")
    print("  - Size: ~6 GB")
    print("  - Contains: 8732 urban sound excerpts")

    print("\nManual Download Required:")
    print("  1. Visit: https://urbansounddataset.weebly.com/urbansound8k.html")
    print("  2. Download the dataset (requires registration)")
    print("  3. Extract to:", str(urban_dir))
    print("  4. Re-run this script with --prepare-only flag")

    logger.info("urbansound8k_manual_download_required")


def prepare_datasets(data_dir: Path, datasets: list[str]) -> None:
    """Prepare downloaded datasets for use with Kizuna.

    Args:
        data_dir: Data directory path
        datasets: List of datasets to prepare
    """
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    logger.info("preparing_datasets", datasets=datasets)

    print("\n" + "=" * 70)
    print("Dataset Preparation")
    print("=" * 70)

    if "up-fall" in datasets:
        up_fall_dir = data_dir / "raw" / "up-fall"
        if up_fall_dir.exists() and any(up_fall_dir.iterdir()):
            print(f"\n✓ UP-Fall dataset found at {up_fall_dir}")
            print("  Processing...")
            # TODO: Add actual processing logic once dataset is available
            print("  Note: Processing logic will be implemented when dataset is available")
        else:
            print(f"\n✗ UP-Fall dataset not found at {up_fall_dir}")
            print("  Please download manually (see instructions above)")

    if "esc-50" in datasets:
        esc50_dir = data_dir / "raw" / "esc-50"
        if esc50_dir.exists() and any(esc50_dir.iterdir()):
            print(f"\n✓ ESC-50 dataset found at {esc50_dir}")
            print("  Processing...")
            # TODO: Add actual processing logic once dataset is available
            print("  Note: Processing logic will be implemented when dataset is available")
        else:
            print(f"\n✗ ESC-50 dataset not found at {esc50_dir}")
            print("  Please download manually (see instructions above)")

    if "urbansound8k" in datasets:
        urban_dir = data_dir / "raw" / "urbansound8k"
        if urban_dir.exists() and any(urban_dir.iterdir()):
            print(f"\n✓ UrbanSound8K dataset found at {urban_dir}")
            print("  Processing...")
            # TODO: Add actual processing logic once dataset is available
            print("  Note: Processing logic will be implemented when dataset is available")
        else:
            print(f"\n✗ UrbanSound8K dataset not found at {urban_dir}")
            print("  Please download manually (see instructions above)")

    # Create metadata file
    metadata_file = processed_dir / "datasets_metadata.txt"
    with open(metadata_file, "w") as f:
        f.write("Kizuna Privacy Engine - Datasets Metadata\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Datasets prepared: {', '.join(datasets)}\n")
        f.write(f"Raw data: {data_dir / 'raw'}\n")
        f.write(f"Processed data: {processed_dir}\n")

    print(f"\n✓ Metadata written to {metadata_file}")
    logger.info("datasets_prepared", datasets=datasets)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare datasets for Kizuna Privacy Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["up-fall", "esc-50", "urbansound8k", "all"],
        default="all",
        help="Dataset to download/prepare",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory path",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only show download instructions",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only prepare already downloaded datasets",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    datasets = []
    if args.dataset == "all":
        datasets = ["up-fall", "esc-50", "urbansound8k"]
    else:
        datasets = [args.dataset]

    print("\n" + "=" * 70)
    print("Kizuna Privacy Engine - Dataset Preparation")
    print("=" * 70)
    print(f"\nData directory: {data_dir.absolute()}")
    print(f"Datasets: {', '.join(datasets)}")

    if not args.prepare_only:
        # Show download instructions
        if "up-fall" in datasets:
            download_up_fall(data_dir)

        if "esc-50" in datasets:
            download_esc50(data_dir)

        if "urbansound8k" in datasets:
            download_urbansound8k(data_dir)

    if not args.download_only:
        # Prepare datasets
        prepare_datasets(data_dir, datasets)

    print("\n" + "=" * 70)
    print("Dataset preparation workflow complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Download datasets manually using instructions above")
    print("  2. Run: python scripts/prepare_datasets.py --prepare-only")
    print("  3. Start using Kizuna with real datasets!")


if __name__ == "__main__":
    main()
