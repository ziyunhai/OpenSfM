#include <features/matching.h>
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <map>
#include <set>

namespace {

// ============================================================================
// compute_vlad_descriptor
// ============================================================================

TEST(VladDescriptor, SingleFeatureAssignedToNearestCenter) {
  // 2 centers in 3D, 1 feature closest to center 0
  MatXf centers(2, 3);
  centers.row(0) << 1.0f, 0.0f, 0.0f;
  centers.row(1) << 0.0f, 1.0f, 0.0f;

  MatXf features(1, 3);
  features.row(0) << 1.1f, 0.2f, 0.0f;

  VecXf vlad = features::compute_vlad_descriptor(features, centers);

  ASSERT_EQ(vlad.size(), 6);  // 2 centers * 3 dims
  // Residual for center 0: (1.1-1, 0.2-0, 0-0) = (0.1, 0.2, 0)
  EXPECT_NEAR(vlad(0), 0.1f, 1e-5f);
  EXPECT_NEAR(vlad(1), 0.2f, 1e-5f);
  EXPECT_NEAR(vlad(2), 0.0f, 1e-5f);
  // Center 1 gets no assignment -> zeros
  EXPECT_NEAR(vlad(3), 0.0f, 1e-5f);
  EXPECT_NEAR(vlad(4), 0.0f, 1e-5f);
  EXPECT_NEAR(vlad(5), 0.0f, 1e-5f);
}

TEST(VladDescriptor, MultipleFeaturesAccumulateResiduals) {
  MatXf centers(1, 2);
  centers.row(0) << 0.0f, 0.0f;

  MatXf features(3, 2);
  features.row(0) << 1.0f, 2.0f;
  features.row(1) << 3.0f, 4.0f;
  features.row(2) << -1.0f, -1.0f;

  VecXf vlad = features::compute_vlad_descriptor(features, centers);

  ASSERT_EQ(vlad.size(), 2);
  // Sum of residuals: (1+3-1, 2+4-1) = (3, 5)
  EXPECT_NEAR(vlad(0), 3.0f, 1e-5f);
  EXPECT_NEAR(vlad(1), 5.0f, 1e-5f);
}

TEST(VladDescriptor, ThrowsOnEmptyCenters) {
  MatXf centers(0, 3);
  MatXf features(1, 3);
  features.row(0) << 1.0f, 0.0f, 0.0f;

  EXPECT_THROW(features::compute_vlad_descriptor(features, centers),
               std::runtime_error);
}

// ============================================================================
// compute_vlad_distances
// ============================================================================

TEST(VladDistances, ReturnsDistancesToOtherImages) {
  std::map<std::string, VecXf> descriptors;
  VecXf d1(2);
  d1 << 1.0f, 0.0f;
  VecXf d2(2);
  d2 << 0.0f, 1.0f;
  VecXf d3(2);
  d3 << 1.0f, 1.0f;

  descriptors["img1"] = d1;
  descriptors["img2"] = d2;
  descriptors["img3"] = d3;

  std::set<std::string> others = {"img2", "img3"};
  auto [distances, names] =
      features::compute_vlad_distances(descriptors, "img1", others);

  ASSERT_EQ(distances.size(), 2);
  ASSERT_EQ(names.size(), 2);

  // Find img2 and img3 in results
  for (size_t i = 0; i < names.size(); ++i) {
    if (names[i] == "img2") {
      // dist(d1, d2) = ||(1,0)-(0,1)|| = sqrt(2)
      EXPECT_NEAR(distances[i], std::sqrt(2.0), 1e-6);
    } else if (names[i] == "img3") {
      // dist(d1, d3) = ||(1,0)-(1,1)|| = 1
      EXPECT_NEAR(distances[i], 1.0, 1e-6);
    } else {
      FAIL() << "Unexpected image: " << names[i];
    }
  }
}

TEST(VladDistances, ReturnsEmptyForUnknownImage) {
  std::map<std::string, VecXf> descriptors;
  std::set<std::string> others = {"img2"};
  auto [distances, names] =
      features::compute_vlad_distances(descriptors, "nonexistent", others);

  EXPECT_TRUE(distances.empty());
  EXPECT_TRUE(names.empty());
}

TEST(VladDistances, SkipsSelfInOthers) {
  std::map<std::string, VecXf> descriptors;
  VecXf d1(2);
  d1 << 1.0f, 0.0f;
  descriptors["img1"] = d1;

  std::set<std::string> others = {"img1"};
  auto [distances, names] =
      features::compute_vlad_distances(descriptors, "img1", others);

  EXPECT_TRUE(distances.empty());
  EXPECT_TRUE(names.empty());
}

}  // namespace
