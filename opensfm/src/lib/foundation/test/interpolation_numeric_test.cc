#include <foundation/interpolation.h>
#include <foundation/numeric.h>
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <array>
#include <cmath>

namespace {

// ============================================================================
// BilinearInterpolation
// ============================================================================

TEST(BilinearInterpolation, ReturnsExactValueAtIntegerCoords) {
  Eigen::MatrixXf img(3, 3);
  img << 1, 2, 3, 4, 5, 6, 7, 8, 9;

  EXPECT_NEAR(foundation::BilinearInterpolation(img, 0.0f, 0.0f), 1.0f, 1e-6f);
  EXPECT_NEAR(foundation::BilinearInterpolation(img, 1.0f, 1.0f), 5.0f, 1e-6f);
}

TEST(BilinearInterpolation, InterpolatesMidpointCorrectly) {
  Eigen::MatrixXf img(2, 2);
  img << 0, 2, 4, 6;

  // Center of the 2x2 image: average of all 4 corners = 3.0
  EXPECT_NEAR(foundation::BilinearInterpolation(img, 0.5f, 0.5f), 3.0f, 1e-6f);
}

TEST(BilinearInterpolation, InterpolatesAlongEdge) {
  Eigen::MatrixXf img(2, 2);
  img << 0, 10, 0, 10;

  // Midpoint along top row
  EXPECT_NEAR(foundation::BilinearInterpolation(img, 0.0f, 0.5f), 5.0f, 1e-6f);
}

TEST(BilinearInterpolation, ReturnsDefaultForOutOfBounds) {
  Eigen::MatrixXf img(2, 2);
  img << 1, 2, 3, 4;

  EXPECT_NEAR(foundation::BilinearInterpolation(img, -0.1f, 0.0f), 0.0f, 1e-6f);
  EXPECT_NEAR(foundation::BilinearInterpolation(img, 0.0f, -0.1f), 0.0f, 1e-6f);
  EXPECT_NEAR(foundation::BilinearInterpolation(img, 1.0f, 0.0f, -1.0f), -1.0f,
              1e-6f);
}

// ============================================================================
// Sign
// ============================================================================

TEST(Sign, PositiveReturnsOne) { EXPECT_EQ(foundation::Sign(5.0), 1.0); }

TEST(Sign, NegativeReturnsMinusOne) { EXPECT_EQ(foundation::Sign(-3.0), -1.0); }

TEST(Sign, ZeroReturnsOne) { EXPECT_EQ(foundation::Sign(0.0), 1.0); }

TEST(Sign, WorksWithIntegers) {
  EXPECT_EQ(foundation::Sign(-7), -1);
  EXPECT_EQ(foundation::Sign(3), 1);
}

// ============================================================================
// SkewMatrix
// ============================================================================

TEST(SkewMatrix, ProducesCrossProductMatrix) {
  Eigen::Vector3d v(1.0, 2.0, 3.0);
  Eigen::Matrix3d S = foundation::SkewMatrix(v);

  // S * w should equal v x w for any w
  Eigen::Vector3d w(4.0, 5.0, 6.0);
  Eigen::Vector3d cross = v.cross(w);
  Eigen::Vector3d skew_product = S * w;

  EXPECT_TRUE(skew_product.isApprox(cross, 1e-12));
}

TEST(SkewMatrix, IsSkewSymmetric) {
  Eigen::Vector3d v(1.0, 2.0, 3.0);
  Eigen::Matrix3d S = foundation::SkewMatrix(v);

  EXPECT_TRUE(S.isApprox(-S.transpose(), 1e-12));
}

// ============================================================================
// ClosestRotationMatrix
// ============================================================================

TEST(ClosestRotationMatrix, IdentityRemainsIdentity) {
  Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d R = foundation::ClosestRotationMatrix(I);

  EXPECT_TRUE(R.isApprox(I, 1e-12));
}

TEST(ClosestRotationMatrix, ScaledIdentityBecomesRotation) {
  Eigen::Matrix3d scaled = 2.5 * Eigen::Matrix3d::Identity();
  Eigen::Matrix3d R = foundation::ClosestRotationMatrix(scaled);

  // Result should be a valid rotation: R^T R = I and det(R) = 1
  EXPECT_TRUE((R.transpose() * R).isApprox(Eigen::Matrix3d::Identity(), 1e-12));
  EXPECT_NEAR(R.determinant(), 1.0, 1e-12);
}

TEST(ClosestRotationMatrix, NoisyRotationIsCorrected) {
  // Start with a known rotation (90 deg around Z)
  Eigen::Matrix3d R_true;
  R_true << 0, -1, 0, 1, 0, 0, 0, 0, 1;

  // Add small noise
  Eigen::Matrix3d noisy = R_true + 0.05 * Eigen::Matrix3d::Random();
  Eigen::Matrix3d R = foundation::ClosestRotationMatrix(noisy);

  EXPECT_TRUE((R.transpose() * R).isApprox(Eigen::Matrix3d::Identity(), 1e-12));
  EXPECT_NEAR(R.determinant(), 1.0, 1e-12);
  // Should be close to the original rotation
  EXPECT_LT((R - R_true).norm(), 0.2);
}

// ============================================================================
// SolveAX0
// ============================================================================

TEST(SolveAX0, FindsNullspaceOfRankDeficientMatrix) {
  // 3x3 matrix with rank 2: last row is sum of first two
  Eigen::Matrix3d A;
  A << 1, 0, 0, 0, 1, 0, 1, 1, 0;

  Eigen::Vector3d x;
  bool ok = foundation::SolveAX0(A, &x);
  EXPECT_TRUE(ok);

  // x should be in the nullspace: A * x ~ 0
  EXPECT_LT((A * x).norm(), 1e-10);
}

TEST(SolveAX0, ReturnsFalseForUnderdetermined) {
  Eigen::MatrixXd A(2, 3);
  A << 1, 0, 0, 0, 1, 0;

  Eigen::VectorXd x;
  bool ok = foundation::SolveAX0(A, &x);
  EXPECT_FALSE(ok);
}

// ============================================================================
// SolveQuartic
// ============================================================================

TEST(SolveQuartic, FindsRootsOfKnownPolynomial) {
  // (x-1)(x-2)(x-3)(x-4) = x^4 - 10x^3 + 35x^2 - 50x + 24
  std::array<double, 5> coeffs = {24.0, -50.0, 35.0, -10.0, 1.0};
  std::array<double, 4> roots;

  bool ok = foundation::SolveQuartic(coeffs, roots);
  ASSERT_TRUE(ok);

  // Sort roots for comparison
  std::sort(roots.begin(), roots.end());
  EXPECT_NEAR(roots[0], 1.0, 1e-6);
  EXPECT_NEAR(roots[1], 2.0, 1e-6);
  EXPECT_NEAR(roots[2], 3.0, 1e-6);
  EXPECT_NEAR(roots[3], 4.0, 1e-6);
}

TEST(SolveQuartic, RefineImprovesPrecision) {
  std::array<double, 5> coeffs = {24.0, -50.0, 35.0, -10.0, 1.0};
  std::array<double, 4> roots;

  foundation::SolveQuartic(coeffs, roots);
  auto refined = foundation::RefineQuarticRoots(coeffs, roots);

  std::sort(refined.begin(), refined.end());
  EXPECT_NEAR(refined[0], 1.0, 1e-10);
  EXPECT_NEAR(refined[1], 2.0, 1e-10);
  EXPECT_NEAR(refined[2], 3.0, 1e-10);
  EXPECT_NEAR(refined[3], 4.0, 1e-10);
}

}  // namespace
