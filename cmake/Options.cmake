# cmake-format: off
# NOTE:
# a) Do not modify this file to change option values. Options should be
#    configured using either a config.cmake file (refer the default file
#    inside the cmake folder), or by setting the required -DFLASHINFER_XXX
#    option through command-line.
#
# b) This file should only contain option definitions and should not contain
#    any other CMake commands.
#
# c) All new options should be defined here with a default value and a short
#    description.
#
# d) Add new options under the appropriate section.

# === BUILD COMPONENT OPTIONS ===
# Core FlashInfer kernel libraries (C++ libraries used by all components)
flashinfer_option(FLASHINFER_BUILD_KERNELS
  "Build and install kernel libraries (required for AOT PyTorch extensions)" OFF)

# PyTorch CUDA extensions for Python bindings
# These provide high-performance operations through precompiled kernels
flashinfer_option(FLASHINFER_AOT_TORCH_EXTS_CUDA
  "Build ahead-of-time compiled PyTorch CUDA extensions (requires FLASHINFER_BUILD_KERNELS)" OFF)

flashinfer_option(FLASHINFER_AOT_TORCH_EXTS_HIP
  "Build ahead-of-time compiled PyTorch HIP extensions (requires FLASHINFER_BUILD_KERNELS)" OFF)

flashinfer_option(FLASHINFER_TVM_BINDING "Build TVM binding support" OFF)
flashinfer_option(FLASHINFER_DISTRIBUTED "Build distributed support" OFF)
flashinfer_option(FLASHINFER_BUILD_WHEELS "Build distributed support" ON)

# === DATA TYPE OPTIONS ===
flashinfer_option(FLASHINFER_ENABLE_FP8 "Enable FP8 data type support" ON)
flashinfer_option(FLASHINFER_ENABLE_FP8_E4M3 "Enable FP8 E4M3 format specifically" ON)
flashinfer_option(FLASHINFER_ENABLE_FP8_E5M2 "Enable FP8 E5M2 format specifically" ON)
flashinfer_option(FLASHINFER_ENABLE_F16 "Enable F16 data type support" ON)
flashinfer_option(FLASHINFER_ENABLE_BF16 "Enable BF16 data type support" ON)

# === CODE GENERATION OPTIONS ===
flashinfer_option(FLASHINFER_GEN_HEAD_DIMS "Head dimensions to enable" 128 256)
flashinfer_option(FLASHINFER_GEN_POS_ENCODING_MODES "Position encoding modes to enable" 0)
flashinfer_option(FLASHINFER_GEN_MASK_MODES "Mask modes to enable" 0 1 2)
flashinfer_option(FLASHINFER_GEN_USE_FP16_QK_REDUCTIONS "Use FP16 for QK reductions" OFF)
flashinfer_option(FLASHINFER_SM90_ALLOWED_HEAD_DIMS "64,64" "128,128" "256,256" "192,128")

# === BUILD TYPE OPTIONS ===
flashinfer_option(FLASHINFER_UNITTESTS "Build unit tests" OFF)
flashinfer_option(FLASHINFER_CXX_BENCHMARKS "Build benchmarks" OFF)
flashinfer_option(FLASHINFER_DIST_UNITTESTS "Build distributed unit tests" OFF)

# === VERSION OPTIONS ===
# Custom version suffix for builds
flashinfer_option(FLASHINFER_VERSION_SUFFIX "Custom version suffix for builds" "")

# === FEATURE-SPECIFIC TESTS/BENCHMARKS ===
flashinfer_option(FLASHINFER_FP8_TESTS "Build FP8 tests" OFF)
flashinfer_option(FLASHINFER_FP8_BENCHMARKS "Build FP8 benchmarks" OFF)

# === ARCHITECTURE OPTIONS ===
flashinfer_option(FLASHINFER_CUDA_ARCHITECTURES "CUDA architectures to compile for" "")
flashinfer_option(FLASHINFER_MIN_CUDA_ARCH "Minimum CUDA architecture required (SM_XX)" 75)

# === PATH OPTIONS ===
flashinfer_option(FLASHINFER_CUTLASS_DIR "Path to CUTLASS installation" "")
flashinfer_option(FLASHINFER_TVM_SOURCE_DIR "Path to TVM source directory" "")

# === COMPILER OPTIONS ===
# Control the C++ ABI for PyTorch compatibility
flashinfer_option(FLASHINFER_USE_CXX11_ABI "Use the C++11 ABI for PyTorch compatibility" OFF)

# === PYTHON OPTIONS ===
flashinfer_option(FLASHINFER_PY_LIMITED_API "Use Python's limited API for better version compatibility" ON)
flashinfer_option(FLASHINFER_MIN_PYTHON_ABI "Minimum Python ABI version for limited API compatibility" "3.9")

# === CUDA OPTIONS ===
flashinfer_option(FLASHINFER_ENABLE_CUDA "Enable NVIDIA CUDA backend" ON)

# === HIP/ROCm OPTIONS ===
flashinfer_option(FLASHINFER_ENABLE_HIP "Enable AMD HIP/ROCm backend" OFF)
flashinfer_option(FLASHINFER_HIP_ARCHITECTURES "HIP architectures to compile for (gfx908, gfx90a, gfx942, etc)" "")

# === AUTO-DERIVED OPTIONS ===

# PyTorch extensions require kernels to be built
if((FLASHINFER_AOT_TORCH_EXTS_CUDA OR FLASHINFER_AOT_TORCH_EXTS_HIP) AND NOT FLASHINFER_BUILD_KERNELS)
  message(STATUS "Building AOT PyTorch extensions require FLASHINFER_BUILD_KERNELS, enabling it")
  set(FLASHINFER_BUILD_KERNELS ON CACHE BOOL "Build kernels (required by PyTorch extensions)" FORCE)
endif()

if(FLASHINFER_AOT_TORCH_EXTS_CUDA AND NOT FLASHINFER_ENABLE_CUDA)
  message(STATUS "FLASHINFER_AOT_TORCH_EXTS_CUDA requires FLASHINFER_ENABLE_CUDA, enabling it")
  set(FLASHINFER_ENABLE_CUDA ON CACHE BOOL "Build CUDA backend" FORCE)
endif()

if(FLASHINFER_AOT_TORCH_EXTS_HIP AND NOT FLASHINFER_ENABLE_HIP)
  message(STATUS "FLASHINFER_AOT_TORCH_EXTS_HIP requires FLASHINFER_ENABLE_HIP, enabling it")
  set(FLASHINFER_ENABLE_HIP ON CACHE BOOL "Build HIP backend" FORCE)
endif()

# Enabling both CUDA and HIP at the same time is not supported
if(FLASHINFER_ENABLE_HIP AND FLASHINFER_ENABLE_CUDA)
  message(FATAL_ERROR "Enabling both CUDA and HIP backends is not supported")
endif()

# Handle CUDA architectures
if(FLASHINFER_ENABLE_CUDA)
  if(FLASHINFER_CUDA_ARCHITECTURES)
    message(STATUS "CMAKE_CUDA_ARCHITECTURES set to ${FLASHINFER_CUDA_ARCHITECTURES}.")
  else()
    # No user-provided architectures, try to detect the CUDA archs based on where
    # the project is being built
    set(detected_archs "")
    detect_cuda_architectures(detected_archs)
    if(detected_archs)
      set(FLASHINFER_CUDA_ARCHITECTURES ${detected_archs} CACHE STRING
          "CUDA architectures" FORCE)
      message(STATUS "Setting FLASHINFER_CUDA_ARCHITECTURES to detected values: ${FLASHINFER_CUDA_ARCHITECTURES}")
    else()
      # No architectures detected, use safe defaults
      set(FLASHINFER_CUDA_ARCHITECTURES "75;80;86" CACHE STRING
          "CUDA architectures to compile for" FORCE)
      message(STATUS "No architectures detected, using defaults: ${FLASHINFER_CUDA_ARCHITECTURES}")
    endif()
  endif()

  set(CMAKE_CUDA_ARCHITECTURES ${FLASHINFER_CUDA_ARCHITECTURES})

  # Derive SM90 support automatically from CUDA architectures
  set(FLASHINFER_ENABLE_SM90 OFF CACHE INTERNAL "SM90 architecture support enabled")

  if(FLASHINFER_CUDA_ARCHITECTURES)
    string(REGEX MATCH "(^|;)90($|;|a)" FOUND_SM90 "${FLASHINFER_CUDA_ARCHITECTURES}")
    if(FOUND_SM90)
      set(FLASHINFER_ENABLE_SM90 ON CACHE INTERNAL "SM90 architecture support enabled")
      message(STATUS "Enabling SM90-specific optimizations based on CUDA architecture selection")
    endif()
  endif()
endif()

if(FLASHINFER_ENABLE_HIP)
  # Set the FLASHINFER_HIP_ARCHITECTURES variable
  if(DEFINED FLASHINFER_HIP_ARCHITECTURES AND
     NOT "${FLASHINFER_HIP_ARCHITECTURES}" STREQUAL "")
    message(STATUS "Using user-specified HIP architectures: ${FLASHINFER_HIP_ARCHITECTURES}")
  else()
    # Auto-detect architectures
    include(ConfigureHIPArchitectures)
    set(detected_hip_archs "")
    detect_hip_architectures(detected_hip_archs)
    if(detected_hip_archs)
      set(FLASHINFER_HIP_ARCHITECTURES ${detected_hip_archs} CACHE STRING
          "CUDA architectures" FORCE)
      message(STATUS "Setting FLASHINFER_HIP_ARCHITECTURES to detected values: ${FLASHINFER_HIP_ARCHITECTURES}")
    else()
      # Default to MI300X architecture
      set(FLASHINFER_HIP_ARCHITECTURES "gfx942")
      message(STATUS "No HIP architectures detected, using default: ${HIP_ARCHITECTURES}")
    endif()
  endif()

  set(CMAKE_HIP_ARCHITECTURES ${FLASHINFER_HIP_ARCHITECTURES})
endif()

# Handle automatic enabling of dependent features
if(FLASHINFER_FP8_TESTS)
  set(FLASHINFER_UNITTESTS ON CACHE BOOL "Tests enabled for FP8" FORCE)
endif()

if(FLASHINFER_FP8_BENCHMARKS)
  set(FLASHINFER_CXX_BENCHMARKS ON CACHE BOOL "Benchmarks enabled for FP8" FORCE)
endif()

if(FLASHINFER_DIST_UNITTESTS)
  set(FLASHINFER_UNITTESTS ON CACHE BOOL "Tests enabled for distributed" FORCE)
endif()

if(FLASHINFER_TVM_BINDING AND NOT FLASHINFER_BUILD_KERNELS)
  message(FATAL_ERROR "TVM binding requires FLASHINFER_BUILD_KERNELS to be ON")
endif()

if(FLASHINFER_ENABLE_FP8)
  # Enable both FP8 formats when FP8 is enabled
  set(FLASHINFER_ENABLE_FP8_E4M3 ON CACHE BOOL "Enable FP8 E4M3 format" FORCE)
  set(FLASHINFER_ENABLE_FP8_E5M2 ON CACHE BOOL "Enable FP8 E5M2 format" FORCE)
endif()

# Ensure FP8 is enabled for FP8 tests/benchmarks
if(FLASHINFER_FP8_TESTS OR FLASHINFER_FP8_BENCHMARKS)
  set(FLASHINFER_ENABLE_FP8 ON CACHE BOOL "FP8 enabled for tests/benchmarks" FORCE)
  set(FLASHINFER_ENABLE_FP8_E4M3 ON CACHE BOOL "FP8_E4M3 enabled for tests/benchmarks" FORCE)
endif()

# cmake-format: on
