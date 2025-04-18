# Top-level Python package configuration

# If building AOT PyTorch CUDA extensions
if(FLASHINFER_AOT_TORCH_EXTS_CUDA)
  message(STATUS "Building AOT PyTorch CUDA extensions")
  add_subdirectory(csrc)
endif()

# Ensure Python package version matches CMake version
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/flashinfer")

message(STATUS "Python package version set to: ${PROJECT_VERSION_FULL}")

# Prepare conditional template variables
set(TORCH_VERSION_ENTRY "")
if(DEFINED Torch_VERSION)
  set(TORCH_VERSION_ENTRY "\"torch\": \"${Torch_VERSION}\",")
endif()

set(CUDA_ARCH_ENTRY "")
if(CMAKE_CUDA_ARCHITECTURES)
  set(CUDA_ARCH_ENTRY
      "\"cuda_architectures\": \"${CMAKE_CUDA_ARCHITECTURES}\",")
endif()

# Generate unified configuration file
include(GenerateConfigPython)

flashinfer_generate_config_python(INSTALL_DIR "flashinfer")

# Install Python files
file(GLOB_RECURSE PYTHON_FILES "${CMAKE_CURRENT_SOURCE_DIR}/flashinfer/*.py")
foreach(file ${PYTHON_FILES})
  file(RELATIVE_PATH rel_path ${CMAKE_CURRENT_SOURCE_DIR} ${file})
  get_filename_component(rel_dir ${rel_path} DIRECTORY)
  install(FILES ${file} DESTINATION ${rel_dir})
endforeach()

message(STATUS "Vendored CUTLASS headers to be installed along with package "
               "at ${SKBUILD_PLATLIB_DIR}/flashinfer/include")

# Extract the two include paths from CUTLASS_INCLUDE_DIRS defined in
# Dependencies.cmake
list(GET CUTLASS_INCLUDE_DIRS 0 CUTLASS_MAIN_INCLUDE_DIR)
list(GET CUTLASS_INCLUDE_DIRS 1 CUTLASS_TOOLS_INCLUDE_DIR)

# Install main CUTLASS and CUTE headers
install(
  DIRECTORY "${CUTLASS_MAIN_INCLUDE_DIR}/"
  DESTINATION "${SKBUILD_PLATLIB_DIR}/flashinfer/include"
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.hpp"
  PATTERN "*.cuh"
  PATTERN "*.inl")

# Install CUTLASS tools/utils/include headers
install(
  DIRECTORY "${CUTLASS_TOOLS_INCLUDE_DIR}/"
  DESTINATION "${SKBUILD_PLATLIB_DIR}/flashinfer/include"
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.hpp"
  PATTERN "*.cuh"
  PATTERN "*.inl")

message(STATUS "TVM bindings to be installed along with package at "
               "${SKBUILD_PLATLIB_DIR}/flashinfer/tvm_binding")

# Install TVM binding files
install(
  DIRECTORY "${CMAKE_SOURCE_DIR}/tvm_binding/"
  DESTINATION "${SKBUILD_PLATLIB_DIR}/flashinfer/tvm_binding"
  FILES_MATCHING
  PATTERN "*.py"
  PATTERN "*.cc"
  PATTERN "*.cu"
  PATTERN "*.h"
  PATTERN "*.cuh")

message(STATUS "flashinfer Python package configured")
