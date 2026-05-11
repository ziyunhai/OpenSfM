#include <geometry/rotation.h>
#include <gtest/gtest.h>

#include <cmath>

namespace {

constexpr double kTol = 1e-10;

// ============================================================================
// RotationFromOpk
// ============================================================================

TEST(Rotation, ZeroOpkIncludesCameraConvention) {
  // OPK = (0,0,0) should give Rc = diag(1, -1, -1).
  Mat3d R = geometry::RotationFromOpk(0.0, 0.0, 0.0);

  Mat3d expected;
  expected << 1, 0, 0, 0, -1, 0, 0, 0, -1;
  EXPECT_TRUE(R.isApprox(expected, kTol));
}

TEST(Rotation, RotationFromOpkIsOrthogonal) {
  Mat3d R = geometry::RotationFromOpk(0.3, 0.5, 0.7);

  EXPECT_TRUE((R * R.transpose()).isApprox(Mat3d::Identity(), kTol));
  EXPECT_NEAR(R.determinant(), 1.0, kTol);
}

TEST(Rotation, PureKappaRotatesAroundZ) {
  double kappa = M_PI / 4.0;
  Mat3d R = geometry::RotationFromOpk(0.0, 0.0, kappa);

  // Should be orthogonal.
  EXPECT_TRUE((R * R.transpose()).isApprox(Mat3d::Identity(), kTol));
  EXPECT_NEAR(R.determinant(), 1.0, kTol);
}

// ============================================================================
// OpkFromRotation: round-trip
// ============================================================================

TEST(Rotation, OpkRoundTripZero) {
  Mat3d R = geometry::RotationFromOpk(0.0, 0.0, 0.0);
  Vec3d opk = geometry::OpkFromRotation(R);

  EXPECT_NEAR(opk(0), 0.0, kTol);
  EXPECT_NEAR(opk(1), 0.0, kTol);
  EXPECT_NEAR(opk(2), 0.0, kTol);
}

TEST(Rotation, OpkRoundTripArbitrary) {
  double omega = 0.15, phi = 0.25, kappa = 0.35;
  Mat3d R = geometry::RotationFromOpk(omega, phi, kappa);
  Vec3d opk = geometry::OpkFromRotation(R);

  EXPECT_NEAR(opk(0), omega, kTol);
  EXPECT_NEAR(opk(1), phi, kTol);
  EXPECT_NEAR(opk(2), kappa, kTol);
}

TEST(Rotation, OpkRoundTripNegativeAngles) {
  double omega = -0.2, phi = -0.1, kappa = -0.3;
  Mat3d R = geometry::RotationFromOpk(omega, phi, kappa);
  Vec3d opk = geometry::OpkFromRotation(R);

  EXPECT_NEAR(opk(0), omega, kTol);
  EXPECT_NEAR(opk(1), phi, kTol);
  EXPECT_NEAR(opk(2), kappa, kTol);
}

}  // namespace
