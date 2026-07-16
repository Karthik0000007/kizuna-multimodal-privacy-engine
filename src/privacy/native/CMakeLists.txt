cmake_minimum_required(VERSION 3.15)
project(kizuna_native)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Create the native extension module
pybind11_add_module(kizuna_native
    wiper.cpp
)

# Platform-specific compiler flags for security
if(MSVC)
    # Windows: Enable security features
    target_compile_options(kizuna_native PRIVATE
        /W4           # Warning level 4
        /WX           # Treat warnings as errors
        /GS           # Buffer security check
        /sdl          # Additional security checks
    )
else()
    # Unix-like: GCC/Clang flags
    target_compile_options(kizuna_native PRIVATE
        -Wall
        -Wextra
        -Werror
        -fstack-protector-strong
        -D_FORTIFY_SOURCE=2
    )
    
    # Link flags for hardening
    if(NOT APPLE)
        target_link_options(kizuna_native PRIVATE
            -Wl,-z,relro
            -Wl,-z,now
        )
    endif()
endif()

# Installation
install(TARGETS kizuna_native
    LIBRARY DESTINATION .
)

# Print configuration summary
message(STATUS "==============================================")
message(STATUS "Kizuna Native Extension Configuration")
message(STATUS "==============================================")
message(STATUS "Python Version: ${Python_VERSION}")
message(STATUS "Python Executable: ${Python_EXECUTABLE}")
message(STATUS "pybind11 Version: ${pybind11_VERSION}")
message(STATUS "C++ Standard: C++${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "==============================================")
