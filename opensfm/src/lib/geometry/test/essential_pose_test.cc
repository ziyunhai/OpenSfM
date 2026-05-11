#include <geometry/essential.h>
#include <geometry/relative_pose.h>
#include <gtest/gtest.h>

#include <Eigen/Geometry>
#include <cmath>

namespace {

constexpr double kTol = 1e-8;

// Build a synthetic two-view scene: camera 2 is translated along X with a
// small rotation.  Returns bearing pairs (normalised) from N 3D points.
class TwoViewFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    R2 = Eigen::AngleAxisd(0.05, Vec3d::UnitY()).toRotationMatrix();
    t2 = Vec3d(1.0, 0.0, 0.0);

    // Random-ish 3D points in front of both cameras.
    // Need at least 9 for EssentialNPoints (9×9 system).
    std::vector<Vec3d> points = {
        {0.5, 0.1, 4.0},   {-0.3, 0.4, 3.5}, {0.2, -0.5, 5.0}, {0.8, 0.3, 6.0},
        {-0.1, -0.2, 4.5}, {0.0, 0.6, 3.0},  {-0.7, 0.0, 5.5}, {0.4, -0.3, 4.2},
        {0.6, 0.5, 3.8},   {-0.4, -0.1, 4.8}};

    const int n = static_cast<int>(points.size());
    x1.resize(n, 3);
    x2.resize(n, 3);
    for (int i = 0; i < n; ++i) {
      Vec3d b1 = points[i].normalized();
      Vec3d b2 = (R2 * points[i] + t2).normalized();
      x1.row(i) = b1;
      x2.row(i) = b2;
    }
  }

  Mat3d R2;
  Vec3d t2;
  Eigen::Matrix<double, -1, 3> x1, x2;
};

// ============================================================================
// EssentialNPoints
// ============================================================================

TEST_F(TwoViewFixture, EssentialNPointsReturnsOneMatrix) {
  auto Es = geometry::EssentialNPoints(x1, x2);

  EXPECT_EQ(Es.size(), 1u);
}

TEST_F(TwoViewFixture, EssentialNPointsSatisfiesEpipolarConstraint) {
  auto Es = geometry::EssentialNPoints(x1, x2);
  ASSERT_EQ(Es.size(), 1u);

  const auto& E = Es[0];
  // x2^T * E * x1 ≈ 0 for all correspondences.
  for (int i = 0; i < x1.rows(); ++i) {
    double val = x2.row(i) * E * x1.row(i).transpose();
    EXPECT_NEAR(val, 0.0, 1e-6);
  }
}

// ============================================================================
// EssentialFivePoints
// ============================================================================

TEST_F(TwoViewFixture, EssentialFivePointsReturnsModels) {
  // Use first 5 correspondences.
  Eigen::Matrix<double, 5, 3> x1_5 = x1.topRows(5);
  Eigen::Matrix<double, 5, 3> x2_5 = x2.topRows(5);

  auto Es = geometry::EssentialFivePoints(x1_5, x2_5);

  // Five-point algorithm returns up to 10 solutions.
  EXPECT_GE(Es.size(), 1u);
  EXPECT_LE(Es.size(), 10u);
}

TEST_F(TwoViewFixture, EssentialFivePointsAtLeastOneSatisfiesEpipolar) {
  Eigen::Matrix<double, 5, 3> x1_5 = x1.topRows(5);
  Eigen::Matrix<double, 5, 3> x2_5 = x2.topRows(5);

  auto Es = geometry::EssentialFivePoints(x1_5, x2_5);
  ASSERT_GE(Es.size(), 1u);

  // At least one solution should satisfy the epipolar constraint.
  bool found_good = false;
  for (const auto& E : Es) {
    double max_err = 0.0;
    for (int i = 0; i < 5; ++i) {
      double val = std::abs(x2_5.row(i) * E * x1_5.row(i).transpose());
      max_err = std::max(max_err, val);
    }
    if (max_err < 1e-6) {
      found_good = true;
      break;
    }
  }
  EXPECT_TRUE(found_good);
}

// ============================================================================
// RelativePoseFromEssential
// ============================================================================

TEST_F(TwoViewFixture, RelativePoseFromEssentialRecoversMotion) {
  // Construct the true essential matrix: E = [t]_x * R
  Mat3d tx;
  tx << 0, -t2(2), t2(1), t2(2), 0, -t2(0), -t2(1), t2(0), 0;
  Mat3d E_true = tx * R2;

  auto RT = geometry::RelativePoseFromEssential(E_true, x1, x2);

  Mat3d R_est = RT.block<3, 3>(0, 0);
  Vec3d t_est = RT.block<3, 1>(0, 3).normalized();
  Vec3d t_true = t2.normalized();

  // Rotation should match.
  EXPECT_TRUE(R_est.isApprox(R2, 1e-6));
  // Translation direction should match (up to sign).
  EXPECT_NEAR(std::abs(t_est.dot(t_true)), 1.0, 1e-6);
}

// ============================================================================
// RelativeRotationNPoints
// ============================================================================

TEST_F(TwoViewFixture, RelativeRotationNPointsRecoversRotation) {
  // For rotation-only, use bearings without translation.
  const int n = 8;
  Eigen::Matrix<double, -1, 3> b1(n, 3), b2(n, 3);
  std::vector<Vec3d> points = {
      {0.5, 0.1, 4.0},   {-0.3, 0.4, 3.5}, {0.2, -0.5, 5.0}, {0.8, 0.3, 6.0},
      {-0.1, -0.2, 4.5}, {0.0, 0.6, 3.0},  {-0.7, 0.0, 5.5}, {0.4, -0.3, 4.2}};

  Mat3d R_pure = Eigen::AngleAxisd(0.1, Vec3d::UnitZ()).toRotationMatrix();
  for (int i = 0; i < n; ++i) {
    b1.row(i) = points[i].normalized();
    b2.row(i) = (R_pure * points[i]).normalized();
  }

  Mat3d R_est = geometry::RelativeRotationNPoints(b1, b2);

  EXPECT_TRUE(R_est.isApprox(R_pure, 1e-6));
}

// ============================================================================
// Error handling
// ============================================================================

TEST(Essential, MismatchedSizesThrows) {
  Eigen::Matrix<double, 5, 3> x1 = Eigen::Matrix<double, 5, 3>::Random();
  Eigen::Matrix<double, 3, 3> x2 = Eigen::Matrix<double, 3, 3>::Random();

  EXPECT_THROW(geometry::EssentialNPoints(x1, x2), std::runtime_error);
  EXPECT_THROW(geometry::EssentialFivePoints(x1, x2), std::runtime_error);
}

TEST(RelativePose, MismatchedSizesThrows) {
  Eigen::Matrix<double, 5, 3> x1 = Eigen::Matrix<double, 5, 3>::Random();
  Eigen::Matrix<double, 3, 3> x2 = Eigen::Matrix<double, 3, 3>::Random();

  Mat3d E = Mat3d::Identity();
  EXPECT_THROW(geometry::RelativePoseFromEssential(E, x1, x2),
               std::runtime_error);
}

}  // namespace
