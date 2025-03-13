//SPDX - FileCopyrightText : 2025 Advanced Micro Devices, Inc. 
//
//SPDX - License - Identifier : Apache-2.0

#include <gtest/gtest.h>
#include <flashinfer/fp16.h>

#include <limits>
#include <cmath>


namespace
{
bool approximatelyEqual(float expected, float actual, float epsilon = 1e-6f) {
    return std::fabs(expected - actual) < epsilon;
}

struct Fp16Fp32CompParam
{   
    float fp32_val;
    uint16_t fp16_val;
};

}

/// Test some basic cases for fp16 to fp32 conversion and vice-versa.
///
struct TestFp32ToFP16Basic : public ::testing::TestWithParam<Fp16Fp32CompParam> {

    TestFp32ToFP16Basic()
    {
        auto actual_fp16 = fp16_ieee_from_fp32_value(GetParam().fp32_val);
        auto expected_fp16 = GetParam().fp16_val;
        EXPECT_EQ(expected_fp16, actual_fp16);
        EXPECT_TRUE(approximatelyEqual(GetParam().fp32_val, fp16_ieee_to_fp32_value(actual_fp16)));
    }
};

TEST_P(TestFp32ToFP16Basic, ChkFpConversions) {}


INSTANTIATE_TEST_SUITE_P(
    Fp32ToFP16BasicConversion,
    TestFp32ToFP16Basic,
    ::testing::Values(
        Fp16Fp32CompParam{1.0f, 0x3c00},
        Fp16Fp32CompParam{-1.0f, 0xbc00},
        Fp16Fp32CompParam{0.0f, 0x0000},
        Fp16Fp32CompParam{65504.0f, 0x7bff}
    ));

struct TestFp32ToFP16SpecialCases : public ::testing::Test
{
    bool approximatelyEqual(float expected, float actual, float epsilon = 1e-6f) {
        return std::fabs(expected - actual) < epsilon;
      }
};

TEST_F(TestFp32ToFP16SpecialCases, ChkSmallFloat) {
  float input = 1e-4f;
  uint16_t actual_fp16 = fp16_ieee_from_fp32_value(input);
  EXPECT_TRUE(approximatelyEqual(input, fp16_ieee_to_fp32_value(actual_fp16)));
}

TEST_F(TestFp32ToFP16SpecialCases, ChkNaN) {
    float input = std::nanf("");
    uint16_t actual_fp16 = fp16_ieee_from_fp32_value(input);
    EXPECT_NE(0, actual_fp16 & 0x7c00); // Check if it's NaN
    EXPECT_TRUE(std::isnan(fp16_ieee_to_fp32_value(actual_fp16)));
}

TEST_F(TestFp32ToFP16SpecialCases,ChkInfinity) {
    float input = std::numeric_limits<float>::infinity();
    uint16_t actual_fp16 = fp16_ieee_from_fp32_value(input);
    EXPECT_EQ(0x7c00, actual_fp16); // Check if it's positive infinity
    EXPECT_TRUE(std::isinf(fp16_ieee_to_fp32_value(actual_fp16)));
}

TEST_F(TestFp32ToFP16SpecialCases, ChkNegInfinity) {
    float input = -std::numeric_limits<float>::infinity();
    uint16_t actual_fp16 = fp16_ieee_from_fp32_value(input);
    EXPECT_EQ(0xfc00, actual_fp16); // Check if it's negative infinity
    EXPECT_TRUE(std::isinf(fp16_ieee_to_fp32_value(actual_fp16)));
}

TEST_F(TestFp32ToFP16SpecialCases, ChkMaxFloat) {
    float input = std::numeric_limits<float>::max();
    uint16_t actual_fp16 = fp16_ieee_from_fp32_value(input);
    EXPECT_EQ(0x7c00, actual_fp16); // Expect infinity due to overflow
}

TEST_F(TestFp32ToFP16SpecialCases, ChkMinFloat) {
    float input = std::numeric_limits<float>::min();
    uint16_t actual_fp16 = fp16_ieee_from_fp32_value(input);
    EXPECT_NE(0, actual_fp16);
}