cmake_minimum_required(VERSION 3.23.1)
project(flashinfer CUDA CXX)

file(GLOB_RECURSE FLASHINFER_GENERATORS
     ${PROJECT_SOURCE_DIR}/aot_build_utils/*.py)

add_subdirectory(libflashinfer)