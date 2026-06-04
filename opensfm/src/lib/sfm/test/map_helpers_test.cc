#include <geometry/camera.h>
#include <geometry/pose.h>
#include <gtest/gtest.h>
#include <map/map.h>
#include <map/observation.h>
#include <sfm/map_helpers.h>

#include <cmath>

namespace {

// Helper: build a toy map with n cameras at known positions looking at points.
// Places cameras on a circle of radius `cam_radius` at height 0, all looking
// at the origin.
class MapHelpersFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    auto cam = geometry::Camera::CreatePerspectiveCamera(0.5, 0.0, 0.0);
    cam.id = "cam";
    cam.width = 640;
    cam.height = 480;
    map.CreateCamera(cam);

    map::RigCamera rig_camera;
    rig_camera.id = "rig";
    map.CreateRigCamera(rig_camera);
  }

  // Create a shot looking from `origin` toward `target`.
  void AddShot(const std::string& id, const Vec3d& origin,
               const Vec3d& target) {
    // Build a look-at rotation (camera z = forward = target - origin).
    Vec3d forward = (target - origin).normalized();
    Vec3d right = forward.cross(Vec3d::UnitY()).normalized();
    if (right.norm() < 1e-6) {
      right = forward.cross(Vec3d::UnitX()).normalized();
    }
    Vec3d up = right.cross(forward).normalized();

    Mat3d R_world;
    R_world.col(0) = right;
    R_world.col(1) = up;
    R_world.col(2) = forward;

    // R_cam = R_world^T, t_cam = -R_cam * origin
    Mat3d R_cam = R_world.transpose();
    Vec3d t_cam = -R_cam * origin;

    geometry::Pose pose(R_cam, t_cam);
    map.CreateRigInstance(id);
    map.CreateShot(id, "cam", "rig", id, pose);
  }

  map::Map map;
};

// ============================================================================
// RemoveIsolatedPoints
// ============================================================================

TEST_F(MapHelpersFixture, RemoveIsolatedPointsRemovesOutlier) {
  // Create a cluster of close points.
  for (int i = 0; i < 20; ++i) {
    Vec3d pos(0.01 * (i % 5), 0.01 * (i / 5), 5.0);
    map.CreateLandmark(std::to_string(i), pos);
  }
  // Add one far-away isolated point.
  map.CreateLandmark("outlier", Vec3d(100.0, 100.0, 100.0));

  ASSERT_EQ(map.NumberOfLandmarks(), 21u);

  int removed = sfm::map_helpers::RemoveIsolatedPoints(map, 7);

  // The outlier should have been removed.
  EXPECT_GE(removed, 1);
  EXPECT_FALSE(map.HasLandmark("outlier"));
}

TEST_F(MapHelpersFixture, RemoveIsolatedPointsKeepsCluster) {
  // Create a uniform spherical cluster — nothing should be removed.
  // Spherical arrangement avoids edge effects that a grid has.
  for (int i = 0; i < 30; ++i) {
    double theta = 2.0 * M_PI * i / 30.0;
    Vec3d pos(0.01 * std::cos(theta), 0.01 * std::sin(theta), 5.0);
    map.CreateLandmark(std::to_string(i), pos);
  }

  int removed = sfm::map_helpers::RemoveIsolatedPoints(map, 7);

  EXPECT_EQ(removed, 0);
  EXPECT_EQ(map.NumberOfLandmarks(), 30u);
}

TEST_F(MapHelpersFixture, RemoveIsolatedPointsTooFewReturnsZero) {
  // Fewer points than k → should return 0 without crashing.
  for (int i = 0; i < 3; ++i) {
    map.CreateLandmark(std::to_string(i), Vec3d::Random());
  }

  int removed = sfm::map_helpers::RemoveIsolatedPoints(map, 7);

  EXPECT_EQ(removed, 0);
}

// ============================================================================
// FilterBadlyConditionedPoints
// ============================================================================

TEST_F(MapHelpersFixture, FilterBadlyConditionedRemovesNarrowBaseline) {
  // Two cameras very close together → narrow baseline → badly conditioned.
  AddShot("shot0", Vec3d(0.0, 0.0, 0.0), Vec3d(0.0, 0.0, 5.0));
  AddShot("shot1", Vec3d(0.001, 0.0, 0.0), Vec3d(0.0, 0.0, 5.0));

  Vec3d point_pos(0.0, 0.0, 5.0);
  auto& lm = map.CreateLandmark("lm0", point_pos);

  auto& shot0 = map.GetShot("shot0");
  auto& shot1 = map.GetShot("shot1");
  map::Observation obs0(320, 240, 0.5, 128, 128, 128, 0);
  map::Observation obs1(320, 240, 0.5, 128, 128, 128, 0);
  map.AddObservation(&shot0, &lm, obs0);
  map.AddObservation(&shot1, &lm, obs1);

  // With a tiny baseline angle (< default 1 degree), the point should be
  // filtered.
  int removed = sfm::map_helpers::FilterBadlyConditionedPoints(map, 1.0);

  EXPECT_GE(removed, 1);
}

TEST_F(MapHelpersFixture, FilterBadlyConditionedKeepsWellConditioned) {
  // Two cameras with a wide baseline.
  AddShot("shot0", Vec3d(-2.0, 0.0, 0.0), Vec3d(0.0, 0.0, 5.0));
  AddShot("shot1", Vec3d(2.0, 0.0, 0.0), Vec3d(0.0, 0.0, 5.0));

  Vec3d point_pos(0.0, 0.0, 5.0);
  auto& lm = map.CreateLandmark("lm0", point_pos);

  auto& shot0 = map.GetShot("shot0");
  auto& shot1 = map.GetShot("shot1");
  // Approximate image projections for the point
  map::Observation obs0(320, 240, 0.5, 128, 128, 128, 0);
  map::Observation obs1(320, 240, 0.5, 128, 128, 128, 0);
  map.AddObservation(&shot0, &lm, obs0);
  map.AddObservation(&shot1, &lm, obs1);

  int removed = sfm::map_helpers::FilterBadlyConditionedPoints(map, 1.0);

  EXPECT_EQ(removed, 0);
  EXPECT_TRUE(map.HasLandmark("lm0"));
}

}  // namespace
