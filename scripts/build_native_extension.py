#!/usr/bin/env python3
"""Build script for native C++ extension.

Builds the kizuna_native extension using CMake and pybind11.
Requires: cmake, C++ compiler, pybind11
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def check_cmake():
    """Check if CMake is installed."""
    try:
        result = subprocess.run(
            ["cmake", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        version = result.stdout.split("\n")[0]
        print(f"✓ CMake found: {version}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ CMake not found")
        print("  Install CMake: https://cmake.org/download/")
        return False


def check_compiler():
    """Check if a C++ compiler is available."""
    system = platform.system()

    if system == "Windows":
        # Check for MSVC
        try:
            result = subprocess.run(
                ["cl"],
                capture_output=True,
                text=True,
            )
            print("✓ MSVC compiler found")
            return True
        except FileNotFoundError:
            print("✗ MSVC compiler not found")
            print("  Install Visual Studio with C++ build tools")
            return False
    else:
        # Check for g++ or clang++
        for compiler in ["g++", "clang++"]:
            try:
                result = subprocess.run(
                    [compiler, "--version"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                version = result.stdout.split("\n")[0]
                print(f"✓ {compiler} found: {version}")
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue

        print("✗ C++ compiler not found")
        print("  Install g++ or clang++")
        return False


def check_pybind11():
    """Check if pybind11 is installed."""
    try:
        import pybind11

        print(f"✓ pybind11 found: {pybind11.__version__}")
        return True
    except ImportError:
        print("✗ pybind11 not found")
        print("  Install: pip install pybind11")
        return False


def clean_build_dir(build_dir: Path):
    """Clean the build directory."""
    if build_dir.exists():
        print(f"Cleaning build directory: {build_dir}")
        shutil.rmtree(build_dir)

    build_dir.mkdir(parents=True, exist_ok=True)


def run_cmake_configure(src_dir: Path, build_dir: Path, build_type: str) -> bool:
    """Run CMake configuration."""
    print(f"\nConfiguring with CMake...")
    print(f"  Source: {src_dir}")
    print(f"  Build: {build_dir}")
    print(f"  Type: {build_type}")

    cmd = [
        "cmake",
        str(src_dir),
        f"-DCMAKE_BUILD_TYPE={build_type}",
    ]

    # Add generator for Windows
    if platform.system() == "Windows":
        # Use Ninja if available, otherwise default generator
        try:
            subprocess.run(["ninja", "--version"], capture_output=True, check=True)
            cmd.extend(["-G", "Ninja"])
            print("  Using Ninja generator")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("  Using default Visual Studio generator")

    try:
        result = subprocess.run(
            cmd,
            cwd=build_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        print("✓ CMake configuration successful")
        return True
    except subprocess.CalledProcessError as e:
        print("✗ CMake configuration failed")
        print(f"\nStdout:\n{e.stdout}")
        print(f"\nStderr:\n{e.stderr}")
        return False


def run_cmake_build(build_dir: Path, build_type: str, verbose: bool = False) -> bool:
    """Run CMake build."""
    print(f"\nBuilding...")

    cmd = [
        "cmake",
        "--build",
        str(build_dir),
        "--config",
        build_type,
    ]

    if verbose:
        cmd.append("--verbose")

    # Use multiple cores
    import multiprocessing

    num_cores = multiprocessing.cpu_count()
    cmd.extend(["--parallel", str(num_cores)])

    print(f"  Using {num_cores} parallel jobs")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        if verbose:
            print(result.stdout)
        print("✓ Build successful")
        return True
    except subprocess.CalledProcessError as e:
        print("✗ Build failed")
        print(f"\nStdout:\n{e.stdout}")
        print(f"\nStderr:\n{e.stderr}")
        return False


def install_extension(build_dir: Path, install_dir: Path) -> bool:
    """Install the built extension to the install directory."""
    print(f"\nInstalling extension...")
    print(f"  From: {build_dir}")
    print(f"  To: {install_dir}")

    # Find the built extension
    if platform.system() == "Windows":
        patterns = ["*.pyd", "*/Release/*.pyd", "*/Debug/*.pyd"]
    else:
        patterns = ["*.so"]

    extension_file = None
    for pattern in patterns:
        matches = list(build_dir.glob(pattern))
        if matches:
            extension_file = matches[0]
            break

    if extension_file is None:
        print("✗ Could not find built extension file")
        return False

    print(f"  Found: {extension_file}")

    # Copy to install directory
    install_dir.mkdir(parents=True, exist_ok=True)
    dest = install_dir / extension_file.name

    shutil.copy2(extension_file, dest)
    print(f"✓ Installed to: {dest}")

    return True


def test_extension() -> bool:
    """Test that the extension can be imported."""
    print(f"\nTesting extension...")

    try:
        import kizuna_native

        print(f"✓ Extension imported successfully")
        print(f"  Implementation: {kizuna_native.get_implementation_info()}")

        # Quick functionality test
        import numpy as np

        arr = np.ones(100, dtype=np.float32)
        result = kizuna_native.secure_wipe(arr)

        if result == 0 and np.all(arr == 0):
            print("✓ Basic functionality test passed")
            return True
        else:
            print("✗ Basic functionality test failed")
            return False

    except ImportError as e:
        print(f"✗ Failed to import extension: {e}")
        return False
    except Exception as e:
        print(f"✗ Extension test failed: {e}")
        return False


def main():
    """Main build script."""
    parser = argparse.ArgumentParser(description="Build kizuna_native C++ extension")
    parser.add_argument(
        "--build-type",
        type=str,
        default="Release",
        choices=["Debug", "Release"],
        help="Build type (Debug or Release)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean build directory before building",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose build output",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip extension import test",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Kizuna Native Extension Build Script")
    print("=" * 70)

    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Paths
    src_dir = project_root / "src" / "privacy" / "native"
    build_dir = project_root / "build" / "native"
    install_dir = project_root / "src" / "privacy"

    # Check prerequisites
    print("\nChecking prerequisites...")

    all_ok = True
    all_ok &= check_cmake()
    all_ok &= check_compiler()
    all_ok &= check_pybind11()

    if not all_ok:
        print("\n✗ Prerequisites not met")
        print("Please install missing dependencies and try again")
        return 1

    print("\n✓ All prerequisites met")

    # Clean if requested
    if args.clean:
        clean_build_dir(build_dir)
    else:
        build_dir.mkdir(parents=True, exist_ok=True)

    # Configure
    if not run_cmake_configure(src_dir, build_dir, args.build_type):
        return 1

    # Build
    if not run_cmake_build(build_dir, args.build_type, args.verbose):
        return 1

    # Install
    if not install_extension(build_dir, install_dir):
        return 1

    # Test
    if not args.skip_test:
        if not test_extension():
            print("\n⚠ Extension test failed, but build completed")
            print("  You may need to add the install directory to PYTHONPATH")
            return 1

    print("\n" + "=" * 70)
    print("✓ Build complete!")
    print("=" * 70)
    print(f"\nExtension installed to: {install_dir}")
    print("\nYou can now use the native extension:")
    print("  from src.privacy.memory_wiper_native import NativeSecureWiper")

    return 0


if __name__ == "__main__":
    sys.exit(main())
