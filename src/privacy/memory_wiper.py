"""Secure memory wiper for raw data destruction.

Implements secure overwriting of NumPy array buffers to prevent data recovery
through memory forensics or system dumps. Critical for APPI compliance.
"""

import ctypes
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..logger import get_logger

logger = get_logger("privacy")


@dataclass
class WipeResult:
    """Result of a memory wipe operation.

    Attributes:
        success: Whether the wipe completed successfully
        timestamp: Unix timestamp when wipe completed
        array_shape: Shape of the wiped array
        array_dtype: Data type of the wiped array
        size_bytes: Number of bytes wiped
        duration_ms: Time taken to wipe in milliseconds
        verified: Whether wipe was verified
        verification_passed: Whether verification succeeded (None if not verified)
    """

    success: bool
    timestamp: float
    array_shape: tuple
    array_dtype: str
    size_bytes: int
    duration_ms: float
    verified: bool
    verification_passed: bool | None = None


class SecurityException(Exception):
    """Exception raised when security-critical operations fail."""

    pass


class SecureWiper:
    """Secure memory wiper using ctypes.memset().

    Overwrites NumPy array buffers with zeros using low-level memory operations
    that cannot be optimized away by the compiler. Optionally verifies the wipe
    by reading back the buffer contents.

    This is the Python fallback implementation. For maximum security and performance,
    use the native C++ implementation (memory_wiper_native).
    """

    def __init__(self, verify: bool = True, num_passes: int = 1) -> None:
        """Initialize secure wiper.

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

        logger.info(
            "secure_wiper_initialized",
            implementation="python_ctypes",
            verify=verify,
            num_passes=num_passes,
        )

    def wipe(self, array: NDArray) -> WipeResult:
        """Securely wipe a NumPy array's memory buffer.

        Overwrites the entire memory buffer with zeros using ctypes.memset(),
        which operates at a low enough level to prevent compiler optimization.
        Optionally verifies the wipe succeeded.

        Args:
            array: NumPy array to wipe

        Returns:
            WipeResult containing operation details

        Raises:
            SecurityException: If verification fails
            TypeError: If input is not a NumPy array
        """
        if not isinstance(array, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(array)}")

        start_time = time.perf_counter()

        # Get array metadata before wiping
        shape = array.shape
        dtype = str(array.dtype)
        size_bytes = array.nbytes

        # Get pointer to the array's data buffer
        array_ptr = array.ctypes.data_as(ctypes.c_void_p)

        # Perform multiple overwrite passes
        for pass_num in range(self.num_passes):
            # Overwrite buffer with zeros using memset
            # ctypes.memset(dst, value, count)
            ctypes.memset(array_ptr, 0, size_bytes)

            logger.debug(
                "memory_overwrite_pass",
                pass_number=pass_num + 1,
                total_passes=self.num_passes,
                size_bytes=size_bytes,
            )

        duration_ms = (time.perf_counter() - start_time) * 1000

        # Verify wipe if enabled
        verification_passed = None
        if self.verify:
            verification_passed = self._verify_wipe(array)

            if not verification_passed:
                logger.error(
                    "wipe_verification_failed",
                    shape=shape,
                    dtype=dtype,
                    size_bytes=size_bytes,
                )
                raise SecurityException(
                    f"Memory wipe verification failed for array shape {shape}, "
                    f"dtype {dtype}. Buffer contains non-zero values after wipe."
                )

        # Log successful wipe
        logger.info(
            "memory_wiped",
            shape=shape,
            dtype=dtype,
            size_bytes=size_bytes,
            duration_ms=duration_ms,
            verified=self.verify,
            verification_passed=verification_passed,
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

    def _verify_wipe(self, array: NDArray) -> bool:
        """Verify that array buffer contains only zeros.

        Args:
            array: Array to verify

        Returns:
            True if all bytes are zero, False otherwise
        """
        # Check if all values are zero
        # Use np.all() which is optimized C code
        all_zeros = np.all(array == 0)

        if not all_zeros:
            # Find first non-zero value for logging
            nonzero_indices = np.nonzero(array)
            if len(nonzero_indices[0]) > 0:
                first_nonzero_idx = tuple(idx[0] for idx in nonzero_indices)
                first_nonzero_val = array[first_nonzero_idx]

                logger.warning(
                    "verification_found_nonzero",
                    position=first_nonzero_idx,
                    value=first_nonzero_val,
                )

        return all_zeros

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
            logger.debug("wiping_array", index=i, total=len(arrays))
            result = self.wipe(array)
            results.append(result)

        logger.info(
            "multiple_wipes_complete",
            num_arrays=len(arrays),
            total_bytes=sum(r.size_bytes for r in results),
            total_duration_ms=sum(r.duration_ms for r in results),
        )

        return results


class SamplingVerifier:
    """Memory wipe verifier using statistical sampling.

    For large arrays, checking every byte can be slow. This verifier checks
    a random sample of bytes for better performance, escalating to full
    verification if any non-zero bytes are found.
    """

    def __init__(self, sample_rate: float = 0.1) -> None:
        """Initialize sampling verifier.

        Args:
            sample_rate: Fraction of bytes to check (0.0-1.0, default: 0.1 = 10%)

        Raises:
            ValueError: If sample_rate is out of range
        """
        if sample_rate <= 0 or sample_rate > 1.0:
            raise ValueError(f"sample_rate must be in (0, 1], got {sample_rate}")

        self.sample_rate = sample_rate

        logger.info(
            "sampling_verifier_initialized",
            sample_rate=sample_rate,
        )

    def verify(self, array: NDArray) -> tuple[bool, dict]:
        """Verify wipe using statistical sampling.

        Args:
            array: Array to verify

        Returns:
            Tuple of (passed, stats_dict)
            - passed: True if verification passed
            - stats_dict: Statistics about the verification
        """
        start_time = time.perf_counter()

        # Flatten array for sampling
        flat = array.ravel()
        total_elements = len(flat)

        # Calculate sample size
        sample_size = max(1, int(total_elements * self.sample_rate))

        # Random sample of indices
        sample_indices = np.random.choice(total_elements, size=sample_size, replace=False)

        # Check sampled values
        sampled_values = flat[sample_indices]
        sampled_all_zero = np.all(sampled_values == 0)

        duration_ms = (time.perf_counter() - start_time) * 1000

        stats = {
            "total_elements": total_elements,
            "sample_size": sample_size,
            "sample_rate_actual": sample_size / total_elements,
            "sampled_all_zero": sampled_all_zero,
            "duration_ms": duration_ms,
            "escalated_to_full": False,
        }

        if not sampled_all_zero:
            # Escalate to full verification
            logger.warning(
                "sampling_verification_failed_escalating",
                sample_size=sample_size,
                total_elements=total_elements,
            )

            escalate_start = time.perf_counter()
            full_all_zero = np.all(flat == 0)
            escalate_duration_ms = (time.perf_counter() - escalate_start) * 1000

            stats["escalated_to_full"] = True
            stats["full_verification_all_zero"] = full_all_zero
            stats["escalate_duration_ms"] = escalate_duration_ms
            stats["duration_ms"] += escalate_duration_ms

            return full_all_zero, stats

        return True, stats


def main() -> None:
    """Demo secure memory wiper."""
    import argparse

    parser = argparse.ArgumentParser(description="Kizuna Secure Memory Wiper Demo")
    parser.add_argument(
        "--size",
        type=int,
        default=1000000,
        help="Array size (number of elements)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="Array data type",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Enable verification",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=1,
        help="Number of overwrite passes",
    )
    parser.add_argument(
        "--num-arrays",
        type=int,
        default=3,
        help="Number of arrays to wipe",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Kizuna Secure Memory Wiper Demo")
    print("=" * 70)

    # Initialize wiper
    print("\nInitializing SecureWiper...")
    print(f"  Verification: {args.verify}")
    print(f"  Overwrite passes: {args.passes}")

    wiper = SecureWiper(verify=args.verify, num_passes=args.passes)

    # Create test arrays with sensitive data
    print(f"\nCreating {args.num_arrays} test arrays...")
    print(f"  Size: {args.size} elements")
    print(f"  Dtype: {args.dtype}")

    arrays = []
    for i in range(args.num_arrays):
        # Fill with random data to simulate sensitive information
        arr = np.random.randn(args.size).astype(args.dtype)
        arrays.append(arr)
        print(
            f"  Array {i}: mean={arr.mean():.4f}, std={arr.std():.4f}, "
            f"size={arr.nbytes / 1024:.2f} KB"
        )

    # Wipe arrays
    print(f"\nWiping {len(arrays)} arrays...")
    results = wiper.wipe_multiple(*arrays)

    print("\nWipe Results:")
    for i, result in enumerate(results):
        print(f"  Array {i}:")
        print(f"    Success: {result.success}")
        print(f"    Duration: {result.duration_ms:.3f}ms")
        print(f"    Size: {result.size_bytes / 1024:.2f} KB")
        print(f"    Verified: {result.verified}")
        if result.verified:
            print(f"    Verification: {'PASSED' if result.verification_passed else 'FAILED'}")

    # Verify all arrays are now zero
    print("\nPost-wipe verification:")
    for i, arr in enumerate(arrays):
        all_zero = np.all(arr == 0)
        mean = arr.mean()
        std = arr.std()
        print(f"  Array {i}: all_zero={all_zero}, mean={mean:.10f}, std={std:.10f}")

    # Performance summary
    total_bytes = sum(r.size_bytes for r in results)
    total_duration = sum(r.duration_ms for r in results)
    throughput_mbps = (total_bytes / 1024 / 1024) / (total_duration / 1000)

    print("\nPerformance Summary:")
    print(f"  Total bytes wiped: {total_bytes / 1024 / 1024:.2f} MB")
    print(f"  Total duration: {total_duration:.2f}ms")
    print(f"  Throughput: {throughput_mbps:.2f} MB/s")
    print(f"  Average latency: {total_duration / len(results):.2f}ms per array")

    # Test sampling verifier
    print(f"\n{'=' * 70}")
    print("Testing Sampling Verifier")
    print(f"{'=' * 70}")

    sampling_verifier = SamplingVerifier(sample_rate=0.1)

    # Create and wipe a large array
    large_array = np.random.randn(10000000).astype(np.float32)
    print(f"\nCreated large array: {large_array.nbytes / 1024 / 1024:.2f} MB")

    wipe_result = wiper.wipe(large_array)
    print(f"Wiped in {wipe_result.duration_ms:.2f}ms")

    # Verify with sampling
    passed, stats = sampling_verifier.verify(large_array)

    print("\nSampling Verification Results:")
    print(f"  Passed: {passed}")
    print(f"  Total elements: {stats['total_elements']:,}")
    print(f"  Sample size: {stats['sample_size']:,}")
    print(f"  Sample rate: {stats['sample_rate_actual'] * 100:.1f}%")
    print(f"  Duration: {stats['duration_ms']:.2f}ms")
    print(f"  Escalated to full: {stats['escalated_to_full']}")

    print(f"\n{'=' * 70}")
    print("✓ Demo complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
