#include <gtest/gtest.h>
#include <pointcloud/tile_io.h>

#include <cstdlib>
#include <filesystem>
#include <string>

namespace {

namespace fs = std::filesystem;

class TileIOFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a unique temp directory.
    tmp_dir = fs::temp_directory_path() / "pointcloud_test_XXXXXX";
    tmp_dir = fs::path(mkdtemp(const_cast<char*>(tmp_dir.string().c_str())));
  }

  void TearDown() override { fs::remove_all(tmp_dir); }

  fs::path tmp_dir;
};

// ============================================================================
// tileFilePath
// ============================================================================

TEST(TileIO, TileFilePathBuildsCorrectName) {
  std::string path = pointcloud::tileFilePath("/some/dir", "r03");
  EXPECT_EQ(path, "/some/dir/r03.bin");
}

TEST(TileIO, TileFilePathRoot) {
  std::string path = pointcloud::tileFilePath("/dir", "r");
  EXPECT_EQ(path, "/dir/r.bin");
}

// ============================================================================
// writeTile / readTile round-trip
// ============================================================================

TEST_F(TileIOFixture, WriteReadTileRoundTrips) {
  pointcloud::TileHeader header{};
  header.magic = pointcloud::kTileMagic;
  header.version = pointcloud::kTileVersion;
  header.numPoints = 3;
  header.childMask = 0b00000101;
  header.aabbMin[0] = -1.0f;
  header.aabbMin[1] = -2.0f;
  header.aabbMin[2] = -3.0f;
  header.aabbMax[0] = 1.0f;
  header.aabbMax[1] = 2.0f;
  header.aabbMax[2] = 3.0f;
  header.spacing = 0.5f;
  header.depth = 2;

  std::vector<pointcloud::PointRecord> points(3);
  points[0] = {100, 200, 300, 10, 20, 30, 255, 128, 64, 500, {}};
  points[1] = {400, 500, 600, -10, -20, -30, 0, 0, 0, 100, {}};
  points[2] = {700, 800, 900, 0, 0, 127, 100, 200, 50, 250, {}};

  bool ok = pointcloud::writeTile(tmp_dir.string(), "r", header, points);
  ASSERT_TRUE(ok);

  pointcloud::TileHeader read_header{};
  std::vector<pointcloud::PointRecord> read_points;
  ok = pointcloud::readTile(tmp_dir.string(), "r", read_header, read_points);
  ASSERT_TRUE(ok);

  EXPECT_EQ(read_header.magic, header.magic);
  EXPECT_EQ(read_header.version, header.version);
  EXPECT_EQ(read_header.numPoints, header.numPoints);
  EXPECT_EQ(read_header.childMask, header.childMask);
  EXPECT_FLOAT_EQ(read_header.aabbMin[0], header.aabbMin[0]);
  EXPECT_FLOAT_EQ(read_header.aabbMax[2], header.aabbMax[2]);
  EXPECT_EQ(read_header.depth, header.depth);

  ASSERT_EQ(read_points.size(), 3u);
  EXPECT_EQ(read_points[0].px, 100);
  EXPECT_EQ(read_points[0].r, 255);
  EXPECT_EQ(read_points[1].nx, -10);
  EXPECT_EQ(read_points[2].nz, 127);
}

// ============================================================================
// readTileHeader
// ============================================================================

TEST_F(TileIOFixture, ReadTileHeaderOnly) {
  pointcloud::TileHeader header{};
  header.magic = pointcloud::kTileMagic;
  header.version = pointcloud::kTileVersion;
  header.numPoints = 1;
  header.childMask = 0;
  header.spacing = 1.0f;
  header.depth = 0;

  std::vector<pointcloud::PointRecord> points(1);
  points[0] = {};

  ASSERT_TRUE(pointcloud::writeTile(tmp_dir.string(), "r", header, points));

  pointcloud::TileHeader read_header{};
  bool ok = pointcloud::readTileHeader(tmp_dir.string(), "r", read_header);
  ASSERT_TRUE(ok);
  EXPECT_EQ(read_header.numPoints, 1u);
  EXPECT_EQ(read_header.magic, pointcloud::kTileMagic);
}

// ============================================================================
// writeMetadata / readMetadata round-trip
// ============================================================================

TEST_F(TileIOFixture, WriteReadMetadataRoundTrips) {
  pointcloud::OctreeMetadata meta{};
  meta.totalPoints = 42000;
  meta.maxDepth = 5;
  meta.rootSpacing = 1.5f;
  meta.aabbMin = {-10.0f, -20.0f, -30.0f};
  meta.aabbMax = {10.0f, 20.0f, 30.0f};
  meta.maxPointsPerTile = 50000;

  bool ok = pointcloud::writeMetadata(tmp_dir.string(), meta);
  ASSERT_TRUE(ok);

  pointcloud::OctreeMetadata read_meta{};
  ok = pointcloud::readMetadata(tmp_dir.string(), read_meta);
  ASSERT_TRUE(ok);

  EXPECT_EQ(read_meta.totalPoints, 42000u);
  EXPECT_EQ(read_meta.maxDepth, 5);
  EXPECT_FLOAT_EQ(read_meta.rootSpacing, 1.5f);
  EXPECT_FLOAT_EQ(read_meta.aabbMin[0], -10.0f);
  EXPECT_FLOAT_EQ(read_meta.aabbMax[2], 30.0f);
  EXPECT_EQ(read_meta.maxPointsPerTile, 50000u);
}

// ============================================================================
// Error handling
// ============================================================================

TEST(TileIO, ReadNonexistentTileReturnsFalse) {
  pointcloud::TileHeader header{};
  std::vector<pointcloud::PointRecord> points;
  bool ok = pointcloud::readTile("/nonexistent/dir", "r", header, points);
  EXPECT_FALSE(ok);
}

TEST(TileIO, ReadNonexistentMetadataReturnsFalse) {
  pointcloud::OctreeMetadata meta{};
  bool ok = pointcloud::readMetadata("/nonexistent/dir", meta);
  EXPECT_FALSE(ok);
}

}  // namespace
