#include <geometry/similarity.h>
#include <gtest/gtest.h>

#include <cmath>

namespace {

constexpr double kTol = 1e-12;

// ============================================================================
// Construction and accessors
// ============================================================================

TEST(Similarity, DefaultIsIdentityTransform) {
  geometry::Similarity sim;

  EXPECT_TRUE(sim.Translation().isApprox(Vec3d::Zero(), kTol));
  EXPECT_TRUE(sim.Rotation().isApprox(Vec3d::Zero(), kTol));
  EXPECT_DOUBLE_EQ(sim.Scale(), 1.0);
  EXPECT_TRUE(sim.IsValid());
}

TEST(Similarity, ConstructFromAngleAxisAndTranslation) {
  Vec3d rot(0.1, 0.2, 0.3);
  Vec3d trans(1.0, 2.0, 3.0);
  double scale = 2.0;

  geometry::Similarity sim(rot, trans, scale);

  EXPECT_TRUE(sim.Rotation().isApprox(rot, kTol));
  EXPECT_TRUE(sim.Translation().isApprox(trans, kTol));
  EXPECT_DOUBLE_EQ(sim.Scale(), scale);
  EXPECT_TRUE(sim.IsValid());
}

// ============================================================================
// Transform
// ============================================================================

TEST(Similarity, IdentityTransformIsNoOp) {
  geometry::Similarity sim;
  Vec3d point(1.0, 2.0, 3.0);

  EXPECT_TRUE(sim.Transform(point).isApprox(point, kTol));
}

TEST(Similarity, PureTranslationShiftsPoint) {
  Vec3d trans(10.0, 20.0, 30.0);
  Vec3d zero = Vec3d::Zero();
  geometry::Similarity sim(zero, trans, 1.0);

  Vec3d point(1.0, 2.0, 3.0);
  Vec3d expected = point + trans;

  EXPECT_TRUE(sim.Transform(point).isApprox(expected, kTol));
}

TEST(Similarity, PureScaleScalesPoint) {
  Vec3d zero = Vec3d::Zero();
  geometry::Similarity sim(zero, zero, 3.0);

  Vec3d point(1.0, 2.0, 3.0);
  Vec3d expected = 3.0 * point;

  EXPECT_TRUE(sim.Transform(point).isApprox(expected, kTol));
}

// ============================================================================
// Inverse
// ============================================================================

TEST(Similarity, InverseUndoesTransform) {
  Vec3d rot(0.1, 0.2, 0.3);
  Vec3d trans(1.0, 2.0, 3.0);
  double scale = 2.5;

  geometry::Similarity sim(rot, trans, scale);
  geometry::Similarity inv = sim.Inverse();

  Vec3d point(4.0, 5.0, 6.0);
  Vec3d transformed = sim.Transform(point);
  Vec3d recovered = inv.Transform(transformed);

  EXPECT_TRUE(recovered.isApprox(point, 1e-10));
}

TEST(Similarity, InverseScaleIsReciprocal) {
  Vec3d zero = Vec3d::Zero();
  geometry::Similarity sim(zero, zero, 4.0);
  geometry::Similarity inv = sim.Inverse();

  EXPECT_NEAR(inv.Scale(), 0.25, kTol);
}

// ============================================================================
// Setters
// ============================================================================

TEST(Similarity, SettersUpdateValues) {
  geometry::Similarity sim;

  Vec3d new_trans(5.0, 6.0, 7.0);
  sim.SetTranslation(new_trans);
  EXPECT_TRUE(sim.Translation().isApprox(new_trans, kTol));

  Vec3d new_rot(0.5, 0.6, 0.7);
  sim.SetRotation(new_rot);
  EXPECT_TRUE(sim.Rotation().isApprox(new_rot, kTol));

  sim.SetScale(3.14);
  EXPECT_DOUBLE_EQ(sim.Scale(), 3.14);
}

// ============================================================================
// IsValid
// ============================================================================

TEST(Similarity, InvalidWithNaNScale) {
  geometry::Similarity sim;
  sim.SetScale(std::nan(""));

  EXPECT_FALSE(sim.IsValid());
}

TEST(Similarity, InvalidWithInfScale) {
  geometry::Similarity sim;
  sim.SetScale(std::numeric_limits<double>::infinity());

  EXPECT_FALSE(sim.IsValid());
}

}  // namespace
