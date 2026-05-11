#include <gtest/gtest.h>
#include <pointcloud/half_float.h>

#include <cmath>
#include <limits>

namespace {

constexpr float kTol = 1e-3f;

// ============================================================================
// floatToHalf / halfToFloat round-trip
// ============================================================================

TEST(HalfFloat, ZeroRoundTrips) {
  uint16_t h = pointcloud::floatToHalf(0.0f);
  float f = pointcloud::halfToFloat(h);
  EXPECT_EQ(f, 0.0f);
}

TEST(HalfFloat, NegativeZeroRoundTrips) {
  uint16_t h = pointcloud::floatToHalf(-0.0f);
  float f = pointcloud::halfToFloat(h);
  EXPECT_EQ(f, -0.0f);
  // Check sign bit is set.
  EXPECT_TRUE(std::signbit(f));
}

TEST(HalfFloat, OneRoundTrips) {
  uint16_t h = pointcloud::floatToHalf(1.0f);
  float f = pointcloud::halfToFloat(h);
  EXPECT_NEAR(f, 1.0f, kTol);
}

TEST(HalfFloat, SmallPositiveRoundTrips) {
  float val = 0.5f;
  uint16_t h = pointcloud::floatToHalf(val);
  float f = pointcloud::halfToFloat(h);
  EXPECT_NEAR(f, val, kTol);
}

TEST(HalfFloat, LargeValueRoundTrips) {
  // Half-float max is ~65504.
  float val = 1024.0f;
  uint16_t h = pointcloud::floatToHalf(val);
  float f = pointcloud::halfToFloat(h);
  EXPECT_NEAR(f, val, 1.0f);
}

TEST(HalfFloat, NegativeRoundTrips) {
  float val = -3.14f;
  uint16_t h = pointcloud::floatToHalf(val);
  float f = pointcloud::halfToFloat(h);
  // Half-float has ~3 decimal digits of precision.
  EXPECT_NEAR(f, val, 2e-3f);
}

// ============================================================================
// Special values
// ============================================================================

TEST(HalfFloat, PositiveInfinity) {
  float inf = std::numeric_limits<float>::infinity();
  uint16_t h = pointcloud::floatToHalf(inf);
  float f = pointcloud::halfToFloat(h);
  EXPECT_TRUE(std::isinf(f));
  EXPECT_GT(f, 0.0f);
}

TEST(HalfFloat, NegativeInfinity) {
  float ninf = -std::numeric_limits<float>::infinity();
  uint16_t h = pointcloud::floatToHalf(ninf);
  float f = pointcloud::halfToFloat(h);
  EXPECT_TRUE(std::isinf(f));
  EXPECT_LT(f, 0.0f);
}

TEST(HalfFloat, NaNRoundTrips) {
  float nan = std::numeric_limits<float>::quiet_NaN();
  uint16_t h = pointcloud::floatToHalf(nan);
  float f = pointcloud::halfToFloat(h);
  EXPECT_TRUE(std::isnan(f));
}

TEST(HalfFloat, OverflowBecomesInf) {
  // Value larger than half-float max (~65504) should become infinity.
  float huge = 100000.0f;
  uint16_t h = pointcloud::floatToHalf(huge);
  float f = pointcloud::halfToFloat(h);
  EXPECT_TRUE(std::isinf(f));
}

TEST(HalfFloat, VerySmallDenorm) {
  // Very small value should round to zero or a denorm.
  float tiny = 1e-8f;
  uint16_t h = pointcloud::floatToHalf(tiny);
  float f = pointcloud::halfToFloat(h);
  EXPECT_GE(f, 0.0f);
  EXPECT_LT(f, 1e-4f);
}

}  // namespace
