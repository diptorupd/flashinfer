cmake_minimum_required(VERSION 3.21)
project(standalone_prefill_kernel CUDA CXX)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA specific optimizations
set(CMAKE_CUDA_ARCHITECTURES 80)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-line-info")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --optimize 3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --threads 0")

find_package(CUDAToolkit REQUIRED)
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
# Find FlashInfer find_package(flashinfer REQUIRED COMPONENTS Headers)

# Add the executable
add_executable(single_prefill_driver single_prefill_driver.cu)

set(FLASHINFER_INCLUDE_DIRS
    /home/diptodeb/devel/flashinfer/libflashinfer/include)
target_include_directories(single_prefill_driver
                           PRIVATE ${FLASHINFER_INCLUDE_DIRS})

# Link dependencies
target_link_libraries(single_prefill_driver PUBLIC CUDA::cudart
                                                   CUDA::cuda_driver)

# Include directories
target_include_directories(single_prefill_driver
                           PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

set_property(TARGET single_prefill_driver PROPERTY CUDA_SEPARABLE_COMPILATION
                                                   ON)
