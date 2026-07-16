/**
 * Native C++ secure memory wiper implementation.
 * 
 * Provides maximum security and performance for memory wiping operations
 * by using compiler-proof memory zeroing techniques and memory barriers.
 * 
 * Exposed to Python via pybind11 as kizuna_native.secure_wipe().
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#ifdef _WIN32
    #include <windows.h>
    #define HAVE_SECUREZEROMEMORY
#elif defined(__APPLE__)
    #include <string.h>
    #define HAVE_MEMSET_S
#elif defined(__linux__)
    #include <string.h>
    #if defined(__GLIBC__) && __GLIBC__ >= 2 && __GLIBC_MINOR__ >= 25
        #define HAVE_EXPLICIT_BZERO
    #endif
#endif

// Compiler barrier to prevent reordering of memory operations
#if defined(__GNUC__) || defined(__clang__)
    #define COMPILER_BARRIER() __asm__ __volatile__("" ::: "memory")
#elif defined(_MSC_VER)
    #include <intrin.h>
    #define COMPILER_BARRIER() _ReadWriteBarrier()
#else
    #define COMPILER_BARRIER()
#endif

namespace kizuna {

/**
 * Volatile memory write that cannot be optimized away.
 * 
 * This function writes zeros to memory using volatile pointer operations,
 * which prevents the compiler from optimizing away the writes.
 * 
 * @param ptr Pointer to memory region to zero
 * @param len Length in bytes
 */
static void volatile_memset_zero(void* ptr, size_t len) {
    volatile uint8_t* volatile_ptr = static_cast<volatile uint8_t*>(ptr);
    
    for (size_t i = 0; i < len; ++i) {
        volatile_ptr[i] = 0;
    }
    
    // Compiler barrier to ensure writes are not reordered
    COMPILER_BARRIER();
}

/**
 * Securely wipe memory using platform-specific secure zeroing.
 * 
 * Uses the most secure method available on the platform:
 * - Windows: SecureZeroMemory()
 * - macOS: memset_s()
 * - Linux (glibc >= 2.25): explicit_bzero()
 * - Fallback: volatile pointer writes with compiler barrier
 * 
 * @param ptr Pointer to memory region to wipe
 * @param len Length in bytes
 * @return 0 on success, -1 on error
 */
int secure_wipe(void* ptr, size_t len) {
    if (ptr == nullptr) {
        return -1;
    }
    
    if (len == 0) {
        return 0;  // Nothing to wipe
    }
    
#ifdef HAVE_SECUREZEROMEMORY
    // Windows: SecureZeroMemory is guaranteed not to be optimized away
    SecureZeroMemory(ptr, len);
    
#elif defined(HAVE_MEMSET_S)
    // macOS/BSD: memset_s is the C11 secure version
    if (memset_s(ptr, len, 0, len) != 0) {
        // Fall back to volatile method on error
        volatile_memset_zero(ptr, len);
    }
    
#elif defined(HAVE_EXPLICIT_BZERO)
    // Linux (glibc >= 2.25): explicit_bzero is guaranteed secure
    explicit_bzero(ptr, len);
    
#else
    // Fallback: Use volatile pointer writes
    volatile_memset_zero(ptr, len);
    
#endif
    
    // Memory barrier to ensure all writes are flushed to RAM
    COMPILER_BARRIER();
    
    return 0;
}

/**
 * Securely wipe memory with multiple passes.
 * 
 * Performs multiple overwrite passes for additional security.
 * Some security standards (e.g., DoD 5220.22-M) require multiple passes.
 * 
 * @param ptr Pointer to memory region to wipe
 * @param len Length in bytes
 * @param num_passes Number of overwrite passes (1-10)
 * @return 0 on success, -1 on error
 */
int secure_wipe_multipass(void* ptr, size_t len, int num_passes) {
    if (ptr == nullptr) {
        return -1;
    }
    
    if (num_passes < 1 || num_passes > 10) {
        return -1;  // Invalid number of passes
    }
    
    for (int pass = 0; pass < num_passes; ++pass) {
        if (secure_wipe(ptr, len) != 0) {
            return -1;
        }
    }
    
    return 0;
}

/**
 * Verify that memory region contains only zeros.
 * 
 * Checks every byte in the memory region to ensure it's zero.
 * Used for wipe verification.
 * 
 * @param ptr Pointer to memory region to verify
 * @param len Length in bytes
 * @return true if all zeros, false otherwise
 */
bool verify_wipe(const void* ptr, size_t len) {
    if (ptr == nullptr) {
        return false;
    }
    
    const uint8_t* byte_ptr = static_cast<const uint8_t*>(ptr);
    
    for (size_t i = 0; i < len; ++i) {
        if (byte_ptr[i] != 0) {
            return false;
        }
    }
    
    return true;
}

}  // namespace kizuna


// Python bindings using pybind11
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

PYBIND11_MODULE(kizuna_native, m) {
    m.doc() = "Kizuna native C++ extensions for secure memory operations";
    
    m.def("secure_wipe",
          [](py::array array) -> int {
              // Get buffer info from NumPy array
              py::buffer_info buf = array.request();
              
              // Wipe the buffer
              return kizuna::secure_wipe(buf.ptr, buf.size * buf.itemsize);
          },
          py::arg("array"),
          R"pbdoc(
              Securely wipe a NumPy array's memory buffer.
              
              Uses platform-specific secure zeroing methods that cannot be
              optimized away by the compiler.
              
              Args:
                  array: NumPy array to wipe
              
              Returns:
                  0 on success, -1 on error
              
              Example:
                  >>> import numpy as np
                  >>> import kizuna_native
                  >>> arr = np.random.randn(1000)
                  >>> result = kizuna_native.secure_wipe(arr)
                  >>> assert result == 0
                  >>> assert np.all(arr == 0)
          )pbdoc");
    
    m.def("secure_wipe_multipass",
          [](py::array array, int num_passes) -> int {
              py::buffer_info buf = array.request();
              return kizuna::secure_wipe_multipass(buf.ptr, buf.size * buf.itemsize, num_passes);
          },
          py::arg("array"),
          py::arg("num_passes"),
          R"pbdoc(
              Securely wipe a NumPy array with multiple overwrite passes.
              
              Args:
                  array: NumPy array to wipe
                  num_passes: Number of overwrite passes (1-10)
              
              Returns:
                  0 on success, -1 on error
          )pbdoc");
    
    m.def("verify_wipe",
          [](py::array array) -> bool {
              py::buffer_info buf = array.request();
              return kizuna::verify_wipe(buf.ptr, buf.size * buf.itemsize);
          },
          py::arg("array"),
          R"pbdoc(
              Verify that a NumPy array's buffer contains only zeros.
              
              Args:
                  array: NumPy array to verify
              
              Returns:
                  True if all zeros, False otherwise
          )pbdoc");
    
    m.def("get_implementation_info",
          []() -> std::string {
#ifdef HAVE_SECUREZEROMEMORY
              return "Windows SecureZeroMemory";
#elif defined(HAVE_MEMSET_S)
              return "macOS/BSD memset_s";
#elif defined(HAVE_EXPLICIT_BZERO)
              return "Linux explicit_bzero";
#else
              return "Volatile pointer fallback";
#endif
          },
          R"pbdoc(
              Get information about the secure wipe implementation.
              
              Returns:
                  String describing the platform-specific method used
          )pbdoc");
}
