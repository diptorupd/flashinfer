// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "configurator_test_fixture.hpp"

using namespace flashinfer::launch_configurator;
using namespace flashinfer::launch_configurator::testing;
using ::testing::Return;

class ConfiguratorSpecializationTest : public ConfiguratorTestFixture
{
};

TEST_F(ConfiguratorSpecializationTest, AmpereOptimizationsForLargeModels)
{
    // Create generic and Ampere configurators
    NvSinglePrefillGenericConfigurator generic_configurator;
    NvSinglePrefillAmpereConfigurator ampere_configurator;

    // Setup parameters for large model with RoPE
    params.head_dim_vo = 128;
    params.use_fp16_qk_reduction = false;
    params.pos_encoding_mode = 1; // RoPE Llama
    params.packed_qo_len = 256;   // Ensure we get num_mma_q = 2

    // Get configurations
    KernelLaunchConfig generic_config =
        generic_configurator.get_kernel_launch_config(cuda_ampere_caps, params,
                                                      kernel_func);
    KernelLaunchConfig ampere_config =
        ampere_configurator.get_kernel_launch_config(cuda_ampere_caps, params,
                                                     kernel_func);

    // Verify Ampere special case optimization
    EXPECT_EQ(generic_config.num_mma_q, 2); // Both should have num_mma_q = 2
    EXPECT_EQ(ampere_config.num_mma_q, 2);

    // But Ampere should limit num_mma_kv to 2 for this case
    EXPECT_EQ(ampere_config.num_mma_kv, 2);
    EXPECT_GE(generic_config.num_mma_kv, ampere_config.num_mma_kv);
}

TEST_F(ConfiguratorSpecializationTest, SharedMemoryConfigurationDifferences)
{
    // Create generic and Ampere configurators
    NvSinglePrefillGenericConfigurator generic_configurator;
    NvSinglePrefillAmpereConfigurator ampere_configurator;

    // Setup parameters that highlight differences in shared memory usage
    params.head_dim_qk = 128;
    params.head_dim_vo = 128;

    // Get configurations
    KernelLaunchConfig generic_config =
        generic_configurator.get_kernel_launch_config(cuda_ampere_caps, params,
                                                      kernel_func);
    KernelLaunchConfig ampere_config =
        ampere_configurator.get_kernel_launch_config(cuda_ampere_caps, params,
                                                     kernel_func);

    // Both should calculate shared memory requirements
    EXPECT_GT(generic_config.smem_bytes, 0);
    EXPECT_GT(ampere_config.smem_bytes, 0);

    // Configurations should be tuned for the architecture
    EXPECT_GE(ampere_config.num_ctas_per_sm, generic_config.num_ctas_per_sm);
}

TEST_F(ConfiguratorSpecializationTest,
       HopperSpecificOptimizationsAreNotInAmpere)
{
    // This test verifies that Ampere doesn't include Hopper-specific
    // optimizations Create a hypothetical Hopper GPU
    GPUCapabilities cuda_hopper_caps = cuda_ampere_caps;
    cuda_hopper_caps.device_name = "sm_90";
    cuda_hopper_caps.compute_capability_major = 9;
    cuda_hopper_caps.compute_capability_minor = 0;

    // Since we don't have a Hopper configurator yet, we expect Ampere
    // configurator to be used instead
    auto hopper_configurator =
        ConfiguratorRegistry::create_single_prefill_configurator(
            cuda_hopper_caps, mock_occupancy.AsStdFunction());

    ASSERT_NE(hopper_configurator, nullptr);
    // Should fall back to Ampere configurator
    EXPECT_EQ(hopper_configurator->name(), "nvidia_ampere_single_prefill");
}
