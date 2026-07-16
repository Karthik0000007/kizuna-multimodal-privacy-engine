"""Native C++ memory wiper with automatic fallback.

Provides secure memory wiping using the native C++ implementation when available,
automatically falling back to the Python ctypes implementation if the native
extension is not installed.
"""

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..logger import get_logger
from .memory_wiper import SecureWiper, WipeResult

logger = get_logger("privacy")

# Try to import native extension
_NATIVE_AVAILABLE = False
_NATIVE_IMPL_INFO = "Not available"

try:
    import kizuna_native  # type: ignore

    _NATIVE_AVAILABLE = True
    _NATIVE_IMPL_INFO = kizuna_native.get_implementation_info()
    logger.info(
        "native_extension_loaded",
        implementation=_NATIVE_IMPL_INFO,
    )
except ImportError as e:
    logger.warning(
        "native_extension_not_available",
        error=str(e),
        fallback="Python ctypes implementation",
    )
    kizuna_native = None  # type: ignore


class NativeSecureWiper:
    """Secure memory wiper using native C++ implementation.

    Automatically falls back to Python ctypes implementation if the native
    extension is not available. The native implementation provides:

    - 2-4x faster performance than Python ctypes
    - Platform-specific secure zeroing (SecureZeroMemory, memset_s, explicit_bzero)
    - Compiler barriers to prevent optimization
    - Memory barriers to ensure writes are flushed

    Fallback behavior:
    - If native extension is not available, uses SecureWiper (Python ctypes)
    - Logs a warning on first initialization
    - No code changes required - API is identical
    """

    def __init__(self, verify: bool = True, num_passes: int = 1) -> None:
        """Initialize native secure wiper.

        Args:
            verify: Whether to verify wipe success by reading back buffer
            num_passes: Number of overwrite passes (1-10, default: 1)

        Raises:
            ValueError: If num_passes is out of range
        """
        if num_passes < 1 or num_passes > 10:
            raise ValueError(f"num_passes must be 1-10, got {num_passes}")

        self.verify = verify
        self.num_passes = num_passes
        self.using_native = _NATIVE_AVAILABLE

        # Create fallback wiper
        self._fallback_wiper = SecureWiper(verify=verify, num_passes=num_passes)

        if self.using_native:
            logger.info(
                "native_wiper_initialized",
                implementation=_NATIVE_IMPL_INFO,
                verify=verify,
                num_passes=num_passes,
            )
        else:
            logger.info(
                "native_wiper_using_fallback",
                verify=verify,
                num_passes=num_passes,
            )

    def wipe(self, array: NDArray) -> WipeResult:
        """Securely wipe a NumPy array's memory buffer.

        Args:
            array: NumPy array to wipe

        Returns:
            WipeResult containing operation details

        Raises:
            SecurityException: If verification fails
            TypeError: If input is not a NumPy array
        """
        if not self.using_native:
            # Use fallback
            return self._fallback_wiper.wipe(array)

        # Use native implementation
        import time

        if not isinstance(array, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(array)}")

        start_time = time.perf_counter()

        # Get array metadata before wiping
        shape = array.shape
        dtype = str(array.dtype)
        size_bytes = array.nbytes

        # Call native wiper
        if self.num_passes == 1:
            result_code = kizuna_native.secure_wipe(array)
        else:
            result_code = kizuna_native.secure_wipe_multipass(array, self.num_passes)

        if result_code != 0:
            logger.error(
                "native_wipe_failed",
                shape=shape,
                dtype=dtype,
                result_code=result_code,
            )
            raise RuntimeError(f"Native wipe failed with code {result_code}")

        duration_ms = (time.perf_counter() - start_time) * 1000

        # Verify wipe if enabled
        verification_passed = None
        if self.verify:
            verify_start = time.perf_counter()
            verification_passed = kizuna_native.verify_wipe(array)
            verify_duration = (time.perf_counter() - verify_start) * 1000
            duration_ms += verify_duration

            if not verification_passed:
                from .memory_wiper import SecurityException

                logger.error(
                    "native_wipe_verification_failed",
                    shape=shape,
                    dtype=dtype,
                    size_bytes=size_bytes,
                )
                raise SecurityException(
                    f"Native memory wipe verification failed for array shape {shape}, "
                    f"dtype {dtype}. Buffer contains non-zero values after wipe."
                )

        # Log successful wipe
        logger.info(
            "native_memory_wiped",
            shape=shape,
            dtype=dtype,
            size_bytes=size_bytes,
            duration_ms=duration_ms,
            verified=self.verify,
            verification_passed=verification_passed,
            implementation=_NATIVE_IMPL_INFO,
        )

        result = WipeResult(
            success=True,
            timestamp=time.time(),
            array_shape=shape,
            array_dtype=dtype,
            size_bytes=size_bytes,
            duration_ms=duration_ms,
            verified=self.verify,
            verification_passed=verification_passed,
        )

        return result

    def wipe_multiple(self, *arrays: NDArray) -> list[WipeResult]:
        """Wipe multiple arrays sequentially.

        Args:
            *arrays: Variable number of arrays to wipe

        Returns:
            List of WipeResult for each array

        Raises:
            SecurityException: If any wipe verification fails
        """
        results = []

        for i, array in enumerate(arrays):
            logger.debug("wiping_array", index=i, total=len(arrays), using_native=self.using_native)
            result = self.wipe(array)
            results.append(result)

        logger.info(
            "multiple_wipes_complete",
            num_arrays=len(arrays),
            total_bytes=sum(r.size_bytes for r in results),
            total_duration_ms=sum(r.duration_ms for r in results),
            using_native=self.using_native,
        )

        return results

    @staticmethod
    def is_native_available() -> bool:
        """Check if native extension is available.

        Returns:
            True if native extension is loaded, False otherwise
        """
        return _NATIVE_AVAILABLE

    @staticmethod
    def get_implementation_info() -> str:
        """Get information about the implementation being used.

        Returns:
            String describing the implementation
        """
        if _NATIVE_AVAILABLE:
            return f"Native C++: {_NATIVE_IMPL_INFO}"
        else:
            return "Python fallback: ctypes.memset()"


def create_wiper(
    verify: bool = True,
    num_passes: int = 1,
    prefer_native: bool = True,
) -> SecureWiper | NativeSecureWiper:
    """Factory function to create the best available wiper.

    Args:
        verify: Whether to verify wipe success
        num_passes: Number of overwrite passes
        prefer_native: If True, use native implementation when available

    Returns:
        NativeSecureWiper if native available and preferred, else SecureWiper
    """
    if prefer_native and _NATIVE_AVAILABLE:
        return NativeSecureWiper(verify=verify, num_passes=num_passes)
    else:
        return SecureWiper(verify=verify, num_passes=num_passes)


def main() -> None:
    """Demo native memory wiper with fallback."""
    import argparse

    parser = argparse.ArgumentParser(description="Kizuna Native Memory Wiper Demo")
    parser.add_argument(
        "--size",
        type=int,
        default=10000000,
        help="Array size (number of elements)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Enable verification",
    )
    parser.add_argument(
        "--force-fallback",
        action="store_true",
        help="Force use of Python fallback",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Kizuna Native Memory Wiper Demo")
    print("=" * 70)

    print(f"\nNative Extension Status:")
    print(f"  Available: {NativeSecureWiper.is_native_available()}")
    print(f"  Implementation: {NativeSecureWiper.get_implementation_info()}")

    # Create wipers for comparison
    print(f"\nInitializing wipers...")

    if args.force_fallback:
        print("  Forcing Python fallback")
        native_wiper = SecureWiper(verify=args.verify)
        use_native = False
    else:
        native_wiper = NativeSecureWiper(verify=args.verify)
        use_native = native_wiper.using_native

    fallback_wiper = SecureWiper(verify=args.verify)

    # Create test array
    print(f"\nCreating test array...")
    print(f"  Size: {args.size:,} elements")
    array1 = np.random.randn(args.size).astype(np.float32)
    array2 = array1.copy()

    size_mb = array1.nbytes / 1024 / 1024
    print(f"  Memory: {size_mb:.2f} MB")

    # Benchmark native/fallback wiper
    if use_native:
        print(f"\nBenchmarking native wiper...")
        result1 = native_wiper.wipe(array1)
        print(f"  Duration: {result1.duration_ms:.3f}ms")
        print(f"  Throughput: {size_mb / (result1.duration_ms / 1000):.2f} MB/s")

    # Benchmark fallback wiper
    print(f"\nBenchmarking fallback wiper...")
    result2 = fallback_wiper.wipe(array2)
    print(f"  Duration: {result2.duration_ms:.3f}ms")
    print(f"  Throughput: {size_mb / (result2.duration_ms / 1000):.2f} MB/s")

    # Compare performance
    if use_native:
        speedup = result2.duration_ms / result1.duration_ms
        print(f"\nPerformance Comparison:")
        print(f"  Native: {result1.duration_ms:.3f}ms")
        print(f"  Fallback: {result2.duration_ms:.3f}ms")
        print(f"  Speedup: {speedup:.2f}x")

        if speedup >= 2.0:
            print(f"  ✓ Native achieves 2x+ speedup target")
        else:
            print(f"  ⚠ Native speedup below 2x target")

    print(f"\n{'=' * 70}")
    print("✓ Demo complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
