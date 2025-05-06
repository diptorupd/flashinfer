// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "configurator_test_fixture.hpp"

using namespace flashinfer::launch_configurator;
using namespace flashinfer::launch_configurator::testing;
using ::testing::Return;

class ConfiguratorCoreTest : public ConfiguratorTestFixture
{
};

TEST_F(ConfiguratorCoreTest, GenericConfiguratorBasicParameterSelection)
{
    // Create a generic configurator
    NvSinglePrefillGenericConfigurator configurator;

    // Get configuration
    KernelLaunchConfig config = configurator.get_kernel_launch_config(
        cuda_generic_caps, params, kernel_func);

    // Validate basic parameters
    EXPECT_EQ(config.cta_tile_q, 64);   // Should be 64 for large head_dim
    EXPECT_EQ(config.num_warps_q, 4);   // Should be 4 for cta_tile_q > 16
    EXPECT_EQ(config.num_warps_kv, 1);  // Should be 1 for cta_tile_q > 16
    EXPECT_EQ(config.num_mma_q, 1);     // Should be 1 for cta_tile_q <= 64
    EXPECT_EQ(config.num_threads, 128); // 4 * 1 * 32
}

TEST_F(ConfiguratorCoreTest, ConfigurationForSmallQLength)
{
    // Test small query length
    params.packed_qo_len = 64;
    params.qo_len = 8;

    NvSinglePrefillGenericConfigurator configurator;
    KernelLaunchConfig config = configurator.get_kernel_launch_config(
        cuda_generic_caps, params, kernel_func);

    EXPECT_EQ(config.cta_tile_q, 16);  // Should be 16 for small qo_len
    EXPECT_EQ(config.num_warps_q, 1);  // Should be 1 for cta_tile_q = 16
    EXPECT_EQ(config.num_warps_kv, 4); // Should be 4 for cta_tile_q = 16
}

TEST_F(ConfiguratorCoreTest, ConfigurationForSmallHeadDim)
{
    // Test small head dimension
    params.head_dim_qk = 32;
    params.head_dim_vo = 32;

    NvSinglePrefillGenericConfigurator configurator;
    KernelLaunchConfig config = configurator.get_kernel_launch_config(
        cuda_generic_caps, params, kernel_func);

    EXPECT_EQ(config.cta_tile_q, 128); // Should be 128 for head_dim <= 64
}

TEST_F(ConfiguratorCoreTest, ChunkingBehaviorForLargeKVLength)
{
    // Test large KV length that needs chunking
    params.kv_len = 32768; // 32K tokens

    NvSinglePrefillGenericConfigurator configurator;
    KernelLaunchConfig config = configurator.get_kernel_launch_config(
        cuda_generic_caps, params, kernel_func);

    // Should have multiple chunks
    EXPECT_GT(config.num_chunks, 1);
    // Minimum chunk size should be at least 256
    EXPECT_GE(ceil_div(params.kv_len, config.num_chunks), 256u);
}
