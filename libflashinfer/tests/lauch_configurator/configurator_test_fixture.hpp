// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "kernel_launch_configurator/configurator_iface.hpp"
#include "kernel_launch_configurator/configurator_registry.hpp"
#include "kernel_launch_configurator/nv_singleprefill_configurator.hpp"

namespace flashinfer
{
namespace launch_configurator
{
namespace testing
{

// Mock for occupancy calculator
class MockOccupancyCalculator
{
public:
    MOCK_METHOD(int, Calculate, (void *, int, size_t), ());

    OccupancyCalculator AsStdFunction()
    {
        return [this](void *kernel_func, int num_threads,
                      size_t smem_size) -> int {
            return this->Calculate(kernel_func, num_threads, smem_size);
        };
    }
};

// A dummy kernel function for testing
void DummyKernel() {}

// For testing
inline uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

// Base fixture for configurator tests
class ConfiguratorTestFixture : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Register configurators
        ConfiguratorRegistry::register_configurator<
            NvSinglePrefillGenericConfigurator, SinglePrefillParameters>(
            "nvidia_generic");
        ConfiguratorRegistry::register_configurator<
            NvSinglePrefillAmpereConfigurator, SinglePrefillParameters>(
            "nvidia_ampere");

        // Setup standard parameters for testing
        params.packed_qo_len = 1024;
        params.head_dim_qk = 128;
        params.head_dim_vo = 128;
        params.num_qo_heads = 32;
        params.num_kv_heads = 32;
        params.qo_len = 128;
        params.kv_len = 2048;
        params.dtype_size_q = 2; // FP16
        params.dtype_size_kv = 2;
        params.dtype_size_o = 2;
        params.use_fp16_qk_reduction = true;
        params.pos_encoding_mode = 1; // RoPE Llama

        // Setup standard GPU capabilities
        cuda_generic_caps.arch_name = "cuda";
        cuda_generic_caps.device_name = "sm_70";
        cuda_generic_caps.compute_capability_major = 7;
        cuda_generic_caps.compute_capability_minor = 0;
        cuda_generic_caps.warp_size = 32;
        cuda_generic_caps.max_threads_per_block = 1024;
        cuda_generic_caps.max_shared_mem_per_sm = 98304; // 96KB
        cuda_generic_caps.num_sm = 80;

        cuda_ampere_caps = cuda_generic_caps;
        cuda_ampere_caps.device_name = "sm_80";
        cuda_ampere_caps.compute_capability_major = 8;
        cuda_ampere_caps.compute_capability_minor = 0;
        cuda_ampere_caps.max_shared_mem_per_sm = 163840; // 160KB for Ampere

        // Get pointer to our dummy kernel
        kernel_func = (void *)DummyKernel;
    }

    SinglePrefillParameters params;
    GPUCapabilities cuda_generic_caps;
    GPUCapabilities cuda_ampere_caps;
    MockOccupancyCalculator mock_occupancy;
    void *kernel_func;
};

} // namespace testing
} // namespace launch_configurator
} // namespace flashinfer
