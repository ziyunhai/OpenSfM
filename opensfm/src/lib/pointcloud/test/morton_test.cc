#include <gtest/gtest.h>
#include <pointcloud/morton.h>

namespace {

// ============================================================================
// mortonEncode / mortonDecode round-trip
// ============================================================================

TEST(Morton, OriginRoundTrips) {
  uint64_t code = pointcloud::mortonEncode(0, 0, 0);
  EXPECT_EQ(code, 0u);

  uint32_t x, y, z;
  pointcloud::mortonDecode(code, x, y, z);
  EXPECT_EQ(x, 0u);
  EXPECT_EQ(y, 0u);
  EXPECT_EQ(z, 0u);
}

TEST(Morton, UnitCoordinatesRoundTrip) {
  uint32_t x, y, z;

  pointcloud::mortonDecode(pointcloud::mortonEncode(1, 0, 0), x, y, z);
  EXPECT_EQ(x, 1u);
  EXPECT_EQ(y, 0u);
  EXPECT_EQ(z, 0u);

  pointcloud::mortonDecode(pointcloud::mortonEncode(0, 1, 0), x, y, z);
  EXPECT_EQ(x, 0u);
  EXPECT_EQ(y, 1u);
  EXPECT_EQ(z, 0u);

  pointcloud::mortonDecode(pointcloud::mortonEncode(0, 0, 1), x, y, z);
  EXPECT_EQ(x, 0u);
  EXPECT_EQ(y, 0u);
  EXPECT_EQ(z, 1u);
}

TEST(Morton, ArbitraryValuesRoundTrip) {
  uint32_t ix = 12345, iy = 67890, iz = 11111;
  uint64_t code = pointcloud::mortonEncode(ix, iy, iz);

  uint32_t x, y, z;
  pointcloud::mortonDecode(code, x, y, z);
  EXPECT_EQ(x, ix);
  EXPECT_EQ(y, iy);
  EXPECT_EQ(z, iz);
}

TEST(Morton, MaxCoordinateRoundTrips) {
  uint32_t max_val = pointcloud::kMortonMaxCoord;
  uint64_t code = pointcloud::mortonEncode(max_val, max_val, max_val);

  uint32_t x, y, z;
  pointcloud::mortonDecode(code, x, y, z);
  EXPECT_EQ(x, max_val);
  EXPECT_EQ(y, max_val);
  EXPECT_EQ(z, max_val);
}

// ============================================================================
// Bit interleaving properties
// ============================================================================

TEST(Morton, XBitIsInLowestPosition) {
  // mortonEncode(1, 0, 0) should have bit 0 set (x is in lowest position).
  uint64_t code = pointcloud::mortonEncode(1, 0, 0);
  EXPECT_EQ(code & 1u, 1u);
}

TEST(Morton, YBitIsInSecondPosition) {
  // mortonEncode(0, 1, 0) should have bit 1 set (y is shifted by 1).
  uint64_t code = pointcloud::mortonEncode(0, 1, 0);
  EXPECT_EQ(code & 0b010, 0b010u);
}

TEST(Morton, ZBitIsInThirdPosition) {
  // mortonEncode(0, 0, 1) should have bit 2 set (z is shifted by 2).
  uint64_t code = pointcloud::mortonEncode(0, 0, 1);
  EXPECT_EQ(code & 0b100, 0b100u);
}

// ============================================================================
// spreadBits3 / compactBits3 round-trip
// ============================================================================

TEST(Morton, SpreadCompactRoundTrip) {
  for (uint64_t v : {0u, 1u, 42u, 1000u, (1u << 21) - 1}) {
    EXPECT_EQ(pointcloud::compactBits3(pointcloud::spreadBits3(v)), v);
  }
}

// ============================================================================
// Ordering property
// ============================================================================

TEST(Morton, SpatialOrderingPreserved) {
  // Points close in space should have closer Morton codes than far points.
  uint64_t near1 = pointcloud::mortonEncode(10, 10, 10);
  uint64_t near2 = pointcloud::mortonEncode(11, 10, 10);
  uint64_t far = pointcloud::mortonEncode(1000, 1000, 1000);

  uint64_t diff_near = (near1 > near2) ? near1 - near2 : near2 - near1;
  uint64_t diff_far = (near1 > far) ? near1 - far : far - near1;

  EXPECT_LT(diff_near, diff_far);
}

// ============================================================================
// quantise
// ============================================================================

TEST(Morton, QuantiseMinReturnsZero) {
  float rangeInv = 1.0f / 10.0f;
  EXPECT_EQ(pointcloud::quantise(0.0f, 0.0f, rangeInv), 0u);
}

TEST(Morton, QuantiseMaxReturnsMaxCoord) {
  float rangeInv = 1.0f / 10.0f;
  EXPECT_EQ(pointcloud::quantise(10.0f, 0.0f, rangeInv),
            pointcloud::kMortonMaxCoord);
}

TEST(Morton, QuantiseMidReturnsMidCoord) {
  float rangeInv = 1.0f / 10.0f;
  uint32_t mid = pointcloud::quantise(5.0f, 0.0f, rangeInv);
  // Should be approximately half of kMortonMaxCoord.
  uint32_t expected = pointcloud::kMortonMaxCoord / 2;
  EXPECT_NEAR(static_cast<double>(mid), static_cast<double>(expected), 1.0);
}

TEST(Morton, QuantiseClampsBelow) {
  float rangeInv = 1.0f / 10.0f;
  EXPECT_EQ(pointcloud::quantise(-5.0f, 0.0f, rangeInv), 0u);
}

TEST(Morton, QuantiseClampsAbove) {
  float rangeInv = 1.0f / 10.0f;
  EXPECT_EQ(pointcloud::quantise(20.0f, 0.0f, rangeInv),
            pointcloud::kMortonMaxCoord);
}

}  // namespace
