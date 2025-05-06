// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "configurator_iface.hpp"

namespace flashinfer
{
namespace launch_configurator
{

template <typename Derived> class NvSinglePrefillBaseConfigurator
{
private:
    NvSinglePrefillBaseConfigurator() = default;

    // Friend declarations for all authorized derived classes
    friend NvSinglePrefillGenericConfigurator;
    friend NvSinglePrefillAmpereConfigurator;
    // Note: Add any future derived classes as friends here

    // Store the occupancy calculator
    OccupancyCalculator occupancy_calc_;

public:
    // Required by LaunchConfigurator concept
    void set_occupancy_calculator(OccupancyCalculator calc)
    {
        occupancy_calc_ = std::move(calc);
    }

    // Interface method that satisfies the concept
    KernelLaunchConfig
    get_kernel_launch_config(const GPUCapabilities &hw,
                             const SinglePrefillParameters &params,
                             void *kernel_func = nullptr) const
    {
        // Call the derived implementation if specialized
        return static_cast<const Derived *>(this)
            ->get_kernel_launch_config_impl(hw, params, kernel_func);
    }

    // Static methods called by the derived classes
    static bool supports_base(const GPUCapabilities &hw)
    {
        return hw.arch_name == "cuda";
    }

    static std::string_view name_base() { return "nvidia_base_single_prefill"; }

protected:
    // Helper methods available to all derived classes
    uint32_t determine_cta_tile_q(int64_t packed_qo_len,
                                  uint32_t head_dim) const
    {
        if (packed_qo_len <= 128)
            return 16;
        if (head_dim <= 64)
            return 128;
        return 64;
    }

    uint32_t determine_num_warps_q(uint32_t cta_tile_q) const
    {
        return (cta_tile_q > 16) ? 4 : 1;
    }

    uint32_t determine_num_warps_kv(uint32_t cta_tile_q) const
    {
        return 4 / determine_num_warps_q(cta_tile_q);
    }

    uint32_t determine_num_mma_q(uint32_t cta_tile_q) const
    {
        return (cta_tile_q > 64) ? 2 : 1;
    }

    void determine_shared_memory_config(const SinglePrefillParameters &params,
                                        const GPUCapabilities &hw,
                                        KernelLaunchConfig &config) const
    {

        size_t q_smem =
            config.cta_tile_q * params.head_dim_qk * params.dtype_size_q;
        size_t kv_smem = (params.head_dim_qk + params.head_dim_vo) * 16 *
                         config.num_warps_kv * params.dtype_size_kv;

        // Determine number of CTAs per SM
        config.num_ctas_per_sm =
            (hw.max_shared_mem_per_sm >= 2 * (q_smem + kv_smem)) ? 2 : 1;
        config.smem_bytes = hw.max_shared_mem_per_sm / config.num_ctas_per_sm;

        // Calculate max_num_mma_kv_reg
        uint32_t max_num_mma_kv_reg =
            (params.head_dim_vo >= 128 && config.num_mma_q == 2 &&
             params.pos_encoding_mode == 1 && // FIXME: magic number
             !params.use_fp16_qk_reduction)
                ? 2
                : (8 / config.num_mma_q);

        // Calculate max_num_mma_kv_smem
        uint32_t max_num_mma_kv_smem =
            (config.smem_bytes - q_smem) /
            ((params.head_dim_qk + params.head_dim_vo) * 16 *
             config.num_warps_kv * params.dtype_size_kv);

        // Set the final num_mma_kv
        config.num_mma_kv = std::min(max_num_mma_kv_smem, max_num_mma_kv_reg);
    }

    void determine_chunking_strategy(const SinglePrefillParameters &params,
                                     const GPUCapabilities &hw,
                                     void *kernel_func,
                                     KernelLaunchConfig &config) const
    {
        int num_blocks_per_sm;

        if (occupancy_calc_ && kernel_func) {
            // Use actual GPU runtime calculation
            num_blocks_per_sm = occupancy_calc_(kernel_func, config.num_threads,
                                                config.smem_bytes);
        }
        else {
            // Fallback to approximation
            num_blocks_per_sm =
                std::min(32 / (config.num_warps_q * config.num_warps_kv),
                         (int)(hw.max_shared_mem_per_sm / config.smem_bytes));
        }

        uint32_t max_num_kv_chunks =
            (num_blocks_per_sm * hw.num_sm) /
            (params.num_kv_heads *
             ceil_div(params.qo_len *
                          (params.num_qo_heads / params.num_kv_heads),
                      config.cta_tile_q));

        if (max_num_kv_chunks > 0) {
            uint32_t chunk_size =
                std::max(ceil_div(params.kv_len, max_num_kv_chunks), 256u);
            config.num_chunks = ceil_div(params.kv_len, chunk_size);
        }
        else {
            config.num_chunks = 0;
        }
    }

    // Helper functions
    uint32_t ceil_div(uint32_t a, uint32_t b) const { return (a + b - 1) / b; }

    // Default implementation of get_kernel_launch_config
    KernelLaunchConfig
    get_kernel_launch_config_impl(const GPUCapabilities &hw,
                                  const SinglePrefillParameters &params,
                                  void *kernel_func = nullptr) const
    {
        KernelLaunchConfig config;

        // Determine CTA tile size
        config.cta_tile_q =
            determine_cta_tile_q(params.packed_qo_len, params.head_dim_vo);

        // Configure warps
        config.num_warps_q = determine_num_warps_q(config.cta_tile_q);
        config.num_warps_kv = determine_num_warps_kv(config.cta_tile_q);

        // Configure MMA units
        config.num_mma_q = determine_num_mma_q(config.cta_tile_q);

        // Calculate total threads
        config.num_threads =
            (config.num_warps_q * config.num_warps_kv) * hw.warp_size;

        // Determine shared memory configuration
        determine_shared_memory_config(params, hw, config);

        // Calculate chunking strategy - pass kernel_func
        determine_chunking_strategy(params, hw, kernel_func, config);

        return config;
    }
};

// Concrete generic implementation for all NVIDIA GPUs
class NvSinglePrefillGenericConfigurator
    : public NvSinglePrefillBaseConfigurator<NvSinglePrefillGenericConfigurator>
{
public:
    // Satisfy the SinglePrefillModel concept
    static bool supports(const GPUCapabilities &hw)
    {
        return NvSinglePrefillBaseConfigurator::supports_base(hw);
    }

    static std::string_view name() { return "nvidia_generic_single_prefill"; }
};

// Specialized implementation for Ampere (SM80+)
class NvSinglePrefillAmpereConfigurator
    : public NvSinglePrefillBaseConfigurator<NvSinglePrefillAmpereConfigurator>
{
    using BaseConfigurator =
        NvSinglePrefillBaseConfigurator<NvSinglePrefillAmpereConfigurator>;

public:
    static bool supports(const GPUCapabilities &hw)
    {
        return hw.arch_name == "cuda" && hw.compute_capability_major >= 8;
    }

    static std::string_view name() { return "nvidia_ampere_single_prefill"; }

    // Customize any methods as needed
    KernelLaunchConfig
    get_kernel_launch_config_impl(const GPUCapabilities &hw,
                                  const SinglePrefillParameters &params,
                                  void *kernel_func = nullptr) const
    {
        // Get base configuration
        KernelLaunchConfig config =
            BaseConfigurator::get_kernel_launch_config_impl(hw, params,
                                                            kernel_func);

        if (params.head_dim_vo >= 128 && config.num_mma_q == 2 &&
            // FIXME: magic number
            params.pos_encoding_mode == 1 && !params.use_fp16_qk_reduction)
        {
            // Special case optimization for large models with RoPE on Ampere
            config.num_mma_kv = std::min(config.num_mma_kv, 2u);
        }

        return config;
    }
};

} // namespace launch_configurator
} // namespace flashinfer
