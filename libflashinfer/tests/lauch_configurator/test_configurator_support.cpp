// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "configurator_test_fixture.hpp"

using namespace flashinfer::launch_configurator;
using namespace flashinfer::launch_configurator::testing;
using ::testing::Return;

class ConfiguratorSupportTest : public ConfiguratorTestFixture
{
};

TEST_F(ConfiguratorSupportTest, GenericConfiguratorSupportsCorrectGPUs)
{
    EXPECT_TRUE(
        NvSinglePrefillGenericConfigurator::supports(cuda_generic_caps));
    EXPECT_TRUE(NvSinglePrefillGenericConfigurator::supports(cuda_ampere_caps));

    GPUCapabilities amd_caps = cuda_generic_caps;
    amd_caps.arch_name = "hip";
    EXPECT_FALSE(NvSinglePrefillGenericConfigurator::supports(amd_caps));
}

TEST_F(ConfiguratorSupportTest, AmpereConfiguratorSupportsCorrectGPUs)
{
    EXPECT_FALSE(
        NvSinglePrefillAmpereConfigurator::supports(cuda_generic_caps));
    EXPECT_TRUE(NvSinglePrefillAmpereConfigurator::supports(cuda_ampere_caps));

    GPUCapabilities amd_caps = cuda_ampere_caps;
    amd_caps.arch_name = "hip";
    EXPECT_FALSE(NvSinglePrefillAmpereConfigurator::supports(amd_caps));
}
