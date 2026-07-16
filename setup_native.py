"""Setup script for building native extensions.

Builds the C++ secure wiper extension using pybind11.
"""

import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

# Define extension
ext_modules = [
    Pybind11Extension(
        "kizuna_native",
        sources=["src/privacy/native/wiper.cpp"],
        cxx_std=17,
        include_dirs=[],
        define_macros=[],
    ),
]

# Platform-specific settings
if sys.platform == "win32":
    # Windows: Add Windows-specific definitions
    ext_modules[0].define_macros.append(("_WIN32", None))
elif sys.platform.startswith("linux"):
    # Linux: Enable position-independent code
    ext_modules[0].extra_compile_args = ["-fPIC"]
elif sys.platform == "darwin":
    # macOS: Set deployment target
    ext_modules[0].extra_compile_args = ["-mmacosx-version-min=10.14"]

setup(
    name="kizuna-native",
    version="1.0.0",
    author="Kizuna Team",
    description="Native secure memory operations for Kizuna Privacy Engine",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.10",
)
