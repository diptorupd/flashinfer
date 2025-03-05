cmake_minimum_required(VERSION 3.23.1)
project(flashinfer CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

file(GLOB_RECURSE FLASHINFER_GENERATORS
     ${PROJECT_SOURCE_DIR}/aot_build_utils/*.py)

add_subdirectory(libflashinfer)