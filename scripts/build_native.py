"""Build script for native C++ extensions.

Compiles the secure memory wiper extension with proper error handling
and fallback behavior if build fails.
"""

import subprocess
import sys
from pathlib import Path


def build_native_extension() -> bool:
    """Build the native C++ extension.

    Returns:
        True if build succeeded, False otherwise
    """
    print("=" * 70)
    print("Building Kizuna Native Extension")
    print("=" * 70)

    # Check if pybind11 is available
    try:
        import pybind11

        print(f"✓ pybind11 found: {pybind11.__version__}")
    except ImportError:
        print("✗ pybind11 not found")
        print("\nInstall pybind11:")
        print("  pip install pybind11")
        return False

    # Check if C++ compiler is available
    print("\nChecking for C++ compiler...")

    if sys.platform == "win32":
        # Windows: Check for MSVC
        try:
            result = subprocess.run(
                ["cl"],
                capture_output=True,
                text=True,
            )
            print("✓ MSVC compiler found")
        except FileNotFoundError:
            print("✗ MSVC compiler not found")
            print("\nInstall Visual Studio Build Tools:")
            print("  https://visualstudio.microsoft.com/downloads/")
            return False
    else:
        # Linux/macOS: Check for g++ or clang
        try:
            result = subprocess.run(
                ["g++", "--version"],
                capture_output=True,
                text=True,
            )
            print(f"✓ g++ compiler found")
        except FileNotFoundError:
            try:
                result = subprocess.run(
                    ["clang++", "--version"],
                    capture_output=True,
                    text=True,
                )
                print(f"✓ clang++ compiler found")
            except FileNotFoundError:
                print("✗ No C++ compiler found (g++ or clang++)")
                print("\nInstall a C++ compiler:")
                print("  Ubuntu/Debian: sudo apt-get install build-essential")
                print("  macOS: xcode-select --install")
                return False

    # Build extension
    print("\nBuilding extension...")

    try:
        result = subprocess.run(
            [sys.executable, "setup_native.py", "build_ext", "--inplace"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

        if result.returncode == 0:
            print("✓ Build succeeded")
            print("\nBuild output:")
            print(result.stdout)
            return True
        else:
            print("✗ Build failed")
            print("\nError output:")
            print(result.stderr)
            return False

    except Exception as e:
        print(f"✗ Build failed with exception: {e}")
        return False


def main() -> None:
    """Main entry point."""
    success = build_native_extension()

    print("\n" + "=" * 70)

    if success:
        print("✓ Native extension built successfully")
        print("\nThe SecureWiper will now use the native implementation")
        print("for improved performance and security.")
    else:
        print("✗ Native extension build failed")
        print("\nThe SecureWiper will fall back to Python implementation.")
        print("This is not critical, but the native implementation provides")
        print("better performance and stronger security guarantees.")

    print("=" * 70)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
