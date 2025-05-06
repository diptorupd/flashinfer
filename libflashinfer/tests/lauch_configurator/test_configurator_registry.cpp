// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "configurator_test_fixture.hpp"

using namespace flashinfer::launch_configurator;
using namespace flashinfer::launch_configurator::testing;
using ::testing::Return;

class ConfiguratorRegistryTest : public ConfiguratorTestFixture
{
};

TEST_F(ConfiguratorRegistryTest, SelectsCorrectConfiguratorForGPU)
{
    auto turing_configurator =
        ConfiguratorRegistry::create_single_prefill_configurator(
            cuda_generic_caps, mock_occupancy.AsStdFunction());

    ASSERT_NE(turing_configurator, nullptr);
    EXPECT_EQ(turing_configurator->name(), "nvidia_generic_single_prefill");

    auto ampere_configurator =
        ConfiguratorRegistry::create_single_prefill_configurator(
            cuda_ampere_caps, mock_occupancy.AsStdFunction());

    ASSERT_NE(ampere_configurator, nullptr);
    EXPECT_EQ(ampere_configurator->name(), "nvidia_ampere_single_prefill");
}

TEST_F(ConfiguratorRegistryTest, FallbackBehavior)
{
    GPUCapabilities amd_caps = cuda_generic_caps;
    amd_caps.arch_name = "hip";

    auto no_match = ConfiguratorRegistry::create_single_prefill_configurator(
        amd_caps, mock_occupancy.AsStdFunction());
    EXPECT_EQ(no_match, nullptr);

    auto any_match = ConfiguratorRegistry::create_single_prefill_configurator(
        amd_caps, mock_occupancy.AsStdFunction(), FallbackPolicy::ANY_MODEL);
    ASSERT_NE(any_match, nullptr);
}
