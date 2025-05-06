// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <concepts>
#include <functional>
#include <string>
#include <string_view>
#include <utility>

namespace flashinfer
{
namespace launch_configurator
{

// Add a callback type for occupancy calculation
using OccupancyCalculator =
    std::function<int(void *kernel_func, int num_threads, size_t smem_size)>;

/// @brief GPU Hardware capabilities
struct GPUCapabilities
{
    std::string_view arch_name;   // "cuda", "hip"
    std::string_view device_name; // "gfx90a", "sm_80", etc.
    int compute_capability_major; // For NVIDIA/AMD architecure versioning
    int compute_capability_minor;
    int warp_size;             // Number of threads per warp
    int max_threads_per_block; // Maximum threads per block
    int max_shared_mem_per_sm; // Available shared memory per SM/CU
    int num_sm;                // Number of SMs/CUs
    int max_threads_per_sm;    // Maximum threads per SM/CU
    int max_blocks_per_sm;     // Maximum blocks per SM/CU
    int max_warps_per_sm;      // Maximum warps per SM/CU
};

/// @brief Flashinfer attention kernel configuration
struct KernelLaunchConfig
{
    uint32_t cta_tile_q;      // Tile size for query dimension
    uint32_t num_mma_q;       // Number of MMA units for query
    uint32_t num_mma_kv;      // Number of MMA units for key/value
    uint32_t num_warps_q;     // Number of warps for query
    uint32_t num_warps_kv;    // Number of warps for key/value
    uint32_t num_threads;     // Total threads per block
    uint32_t num_ctas_per_sm; // CTAs per SM
    uint32_t smem_bytes;      // Shared memory per thread block
    uint32_t num_chunks;      // Number of KV chunks needed
};

/// @brief A concept that defines the interface for any kernel launch
/// configurator used by Flashinfer.
template <typename T, typename ParamT>
concept LaunchConfigurator = requires(T configurator,
                                      const GPUCapabilities &hw,
                                      const ParamT &params,
                                      OccupancyCalculator calc) {
    {
        T::supports(hw)
    } -> std::same_as<bool>;
    {
        configurator.name()
    } -> std::convertible_to<std::string_view>;
    {
        configurator.set_occupancy_calculator(calc)
    } -> std::same_as<void>;
    {
        configurator.get_kernel_launch_config(hw, params, (void *)nullptr)
    } -> std::same_as<KernelLaunchConfig>;
};

/// @brief Parameters for the single prefill kernel that are used to determine
/// an optimal kernel launch configuration.
struct SinglePrefillParameters
{
    int64_t packed_qo_len;      // Query length * group_size
    uint32_t head_dim_qk;       // Size of query/key dimension
    uint32_t head_dim_vo;       // Size of value/output dimension
    uint32_t num_qo_heads;      // Number of query heads
    uint32_t num_kv_heads;      // Number of key/value heads
    uint32_t qo_len;            // Query sequence length
    uint32_t kv_len;            // KV sequence length
    size_t dtype_size_q;        // Byte size of query data type
    size_t dtype_size_kv;       // Byte size of key/value data type
    size_t dtype_size_o;        // Byte size of output data type
    bool use_fp16_qk_reduction; // Use FP16 for QK reduction
    int pos_encoding_mode;      // Positional encoding mode
};

/// @brief A concept that defines the interface for the single prefill kernel
/// launch configurator.
template <typename T>
concept SinglePrefillConfigurator =
    LaunchConfigurator<T, SinglePrefillParameters>;

} // namespace launch_configurator
} // namespace flashinfer
