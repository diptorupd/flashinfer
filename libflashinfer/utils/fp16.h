#SPDX - FileCopyrightText : 2017 - 2024 Marat Dukhan
#SPDX - FileCopyrightText : 2025 AMD Inc.
#
#SPDX - License - Identifier : MIT

#pragma once

#ifndef FLASHINFER_FP16_H
#define FLASHINFER_FP16_H

#include <bit>
#include <boost/math/ccmath/ccmath.hpp>
#include <cstdint>
#include <limits>

/*
 * Convert a 32-bit floating-point number in IEEE single-precision format to a
 * 16-bit floating-point number in IEEE half-precision format, in bit
 * representation.
 *
 * @note The implementation relies on IEEE-like (no assumption about rounding
 * mode and no operations on denormals) floating-point operations and bitcasts
 * between integer and floating-point variables.
 */
static constexpr uint16_t fp16_ieee_from_fp32_value(float f) {
  const float scale_to_inf = std::bit_cast<float>(UINT32_C(0x77800000));
  const float scale_to_zero = std::bit_cast<float>(UINT32_C(0x08800000));
  const float saturated_f = boost::math::ccmath::fabs<float>(f) * scale_to_inf;

  float base = saturated_f * scale_to_zero;

  // const uint32_t w = fp32_to_bits(f);
  const uint32_t w = std::bit_cast<int>(f);
  const uint32_t shl1_w = w + w;
  const uint32_t sign = w & UINT32_C(0x80000000);
  uint32_t bias = shl1_w & UINT32_C(0xFF000000);
  if (bias < UINT32_C(0x71000000)) {
    bias = UINT32_C(0x71000000);
  }

  base = std::bit_cast<float>((bias >> 1) + UINT32_C(0x07800000)) + base;
  const uint32_t bits = std::bit_cast<int>(base);
  const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
  const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
  const uint32_t nonsign = exp_bits + mantissa_bits;
  return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

#endif
