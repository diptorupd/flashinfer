// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "configurator_iface.hpp"
#include <functional>
#include <memory>
#include <vector>

namespace flashinfer
{
namespace launch_configurator
{

class NvSinglePrefillGenericConfigurator;

/// @brief Describes the fallback policy for launch configurator selection
enum class FallbackPolicy
{
    NONE,             // No fallback, return nullptr
    ANY_CONFIGURATOR, // Use any available configurator
    SAME_ARCH_ONLY    // Only use configurators for same architecture
};

// Factory class for launch configurators with dynamic registration
class LaunchConfiguratorRegistry
{
private:
    template <typename ConfiguratorT, typename ParamT> struct ConfiguratorInfo
    {
        std::function<std::unique_ptr<ConfiguratorT>()> factory;
        std::function<bool(const GPUCapabilities &)> supports_fn;
        std::string name;
    };

    // Specialized registry for different kernel types
    template <typename ConfiguratorT, typename ParamT>
    static std::vector<ConfiguratorInfo<ConfiguratorT, ParamT>> &get_registry()
    {
        static std::vector<ConfiguratorInfo<ConfiguratorT, ParamT>> registry;
        return registry;
    }

    /// @brief Returns the best kernel launch configuration for the given GPU
    ///
    /// The function searches for the best match based on the following
    /// priority:
    /// 1. Exact match (specific GPU identifier)
    /// 2. Family match (same GPU family)
    /// 3. Architecture match (e.g. CUDA vs. HIP)
    /// 4. Cross-architecture fallback (e.g. use CUDA configurations for HIP)
    /// 5. Error out if no match is found
    /// @tparam ConfiguratorT
    /// @tparam ParamT
    /// @param hw Instance of GPUCapabilities representing the target GPU
    /// @return Unique pointer to the configurator if match is found, nullptr
    /// otherwise
    ///
    template <typename ConfiguratorT, typename ParamT>
    static std::unique_ptr<ConfiguratorT>
    create_for_gpu(const GPUCapabilities &hw,
                   OccupancyCalculator occupancy_calc = nullptr,
                   FallbackPolicy policy = FallbackPolicy::SAME_ARCH_ONLY)
    {
        auto &registry = get_registry<ConfiguratorT, ParamT>();

        // No configurators registered
        if (registry.empty()) {
            return nullptr;
        }

        std::unique_ptr<ConfiguratorT> family_match = nullptr;
        std::unique_ptr<ConfiguratorT> arch_match = nullptr;

        for (const auto &info : registry) {
            if (!info.supports_fn(hw)) {
                continue;
            }

            // Check if this is an exact match by examining the configurator
            // name. Exact match would contain specific GPU identifier.
            if (info.name.find(hw.device_name) != std::string::npos) {
                auto configurator = info.factory();
                if (occupancy_calc) {
                    configurator->set_occupancy_calculator(occupancy_calc);
                }
                return configurator;
            }

            // Architecture match (same GPU family)
            if (hw.arch_name == "cuda" &&
                info.name.find("nvidia") != std::string::npos)
            {
                family_match = info.factory();
            }
            else if (hw.arch_name == "hip" &&
                     info.name.find("amd") != std::string::npos)
            {
                family_match = info.factory();
            }
            else {
                // Generic match for the architecture
                arch_match = info.factory();
            }
        }

        // Return the most specific match found
        if (family_match) {
            if (occupancy_calc) {
                family_match->set_occupancy_calculator(occupancy_calc);
            }
            return family_match;
        }

        if (arch_match) {
            if (occupancy_calc) {
                arch_match->set_occupancy_calculator(occupancy_calc);
            }
            return arch_match;
        }

        switch (policy) {
        case FallbackPolicy::ANY_CONFIGURATOR:
            auto configurator = registry[0].factory();
            if (occupancy_calc) {
                configurator->set_occupancy_calculator(occupancy_calc);
            }
            return configurator;

        case FallbackPolicy::NONE:
        case FallbackPolicy::SAME_ARCH_ONLY:
        default:
            return nullptr;
        }
    }

public:
    // Register a kernel launch configurator type for a specific kernel
    template <typename ConfiguratorT, typename ParamT>
    static void register_configurator(std::string name)
    {
        static_assert(LaunchConfigurator<ConfiguratorT, ParamT>,
                      "Must satisfy LaunchConfigurator concept");

        auto &registry = get_registry<ConfiguratorT, ParamT>();
        registry.push_back({[]() { return std::make_unique<ConfiguratorT>(); },
                            [](const GPUCapabilities &hw) {
                                return ConfiguratorT::supports(hw);
                            },
                            std::move(name)});
    }

    /// @brief Returns the best launch configurator for SinglePrefill kernels
    /// for a given GPU.
    ///
    /// This specialized factory ensures type safety by constraining the
    /// returned configurator to satisfy the SinglePrefillConfigurator concept.
    ///
    /// @tparam ConfiguratorT The configurator type. Must satisfy the
    ///                       SinglePrefillConfigurator concept.
    /// @param hw Instance of GPUCapabilities representing the target GPU
    /// @param occupancy_calc Optional occupancy calculator function
    /// @param policy Fallback policy for configurator selection
    /// @return Unique pointer to the configurator if match is found, nullptr
    ///         otherwise
    template <typename ConfiguratorT = NvSinglePrefillGenericConfigurator>
        requires SinglePrefillConfigurator<ConfiguratorT>
    static std::unique_ptr<ConfiguratorT> get_single_prefill_configurator(
        const GPUCapabilities &hw,
        OccupancyCalculator occupancy_calc = nullptr,
        FallbackPolicy policy = FallbackPolicy::SAME_ARCH_ONLY)
    {
        return create_for_gpu<ConfiguratorT, SinglePrefillParameters>(
            hw, occupancy_calc, policy);
    }

    // TODO: Additional factory methods for other kernel types can be added here
    // For example:
    /*
    template <typename ConfiguratorT>
        requires BatchPrefillConfigurator<ConfiguratorT>
    static std::unique_ptr<ConfiguratorT>
    get_batch_prefill_configurator(...);

    template <typename ConfiguratorT>
        requires DecodeConfigurator<ConfiguratorT>
    static std::unique_ptr<ConfiguratorT> get_decode_configurator(...);
    */
};

// Hardware detection utility
GPUCapabilities detect_gpu();

} // namespace launch_configurator
} // namespace flashinfer
