// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "configurator_test_fixture.hpp"

using namespace flashinfer::launch_configurator;
using namespace flashinfer::launch_configurator::testing;
using ::testing::_;
using ::testing::Return;

class ConfiguratorOccupancyTest : public ConfiguratorTestFixture
{
};

TEST_F(ConfiguratorOccupancyTest, OccupancyCalculatorIsUsedWhenAvailable)
{
    // Create a generic configurator
    NvSinglePrefillGenericConfigurator configurator;

    // Setup the mock occupancy calculator to return a specific value
    EXPECT_CALL(mock_occupancy, Calculate(kernel_func, _, _))
        .WillOnce(Return(4)); // Simulate 4 blocks per SM

    // Set the occupancy calculator
    configurator.set_occupancy_calculator(mock_occupancy.AsStdFunction());

    // Get configuration
    KernelLaunchConfig config = configurator.get_kernel_launch_config(
        cuda_generic_caps, params, kernel_func);

    // With 4 blocks per SM and 80 SMs, expect more chunks than with fallback
    int expected_blocks = 4 * 80; // 4 blocks per SM * 80 SMs
    int expected_blocks_per_head = expected_blocks / params.num_kv_heads;
    int expected_max_chunks =
        expected_blocks_per_head / ceil_div(params.qo_len, config.cta_tile_q);

    // Assert we're using the provided occupancy calculator result
    EXPECT_GT(config.num_chunks, 0);
    EXPECT_LE(config.num_chunks, expected_max_chunks);
}

TEST_F(ConfiguratorOccupancyTest, FallbackWhenOccupancyCalculatorNotAvailable)
{
    // Create a generic configurator without setting occupancy calculator
    NvSinglePrefillGenericConfigurator configurator;

    // Get configuration - should use fallback approximation
    KernelLaunchConfig config = configurator.get_kernel_launch_config(
        cuda_generic_caps, params, nullptr);

    // Should still have valid config with fallback approximation
    EXPECT_GT(config.num_chunks, 0);
}

TEST_F(ConfiguratorOccupancyTest, FallbackWhenOccupancyCalcReturnsZero)
{
    // Create a generic configurator
    NvSinglePrefillGenericConfigurator configurator;

    // Setup the mock to return an invalid value
    EXPECT_CALL(mock_occupancy, Calculate(kernel_func, _, _))
        .WillOnce(Return(0)); // Simulate failure

    // Set the occupancy calculator
    configurator.set_occupancy_calculator(mock_occupancy.AsStdFunction());

    // Get configuration
    KernelLaunchConfig config = configurator.get_kernel_launch_config(
        cuda_generic_caps, params, kernel_func);

    // Should still have valid config with fallback approximation
    EXPECT_GT(config.num_chunks, 0);
}
