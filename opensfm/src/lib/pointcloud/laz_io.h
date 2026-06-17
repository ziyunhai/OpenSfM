#pragma once

/// LAZ (LASzip-compressed LAS) reader and writer.
///
/// This is the ONLY translation unit that depends on LASzip (`<laszip_api.h>`).
/// It reuses the LAS field conventions from `las_detail` (PDRF 7, classification
/// = label, RGB left-justified 16-bit, normals as 3×f32 Extra Bytes, integer
/// scale/offset positions) and only swaps the storage for the LASzip codec.

#include <cstdint>
#include <string>

#include <pointcloud/las_io.h>
#include <pointcloud/point_cloud_io.h>

namespace pointcloud {

class LAZReader : public PointCloudReader {
 public:
  explicit LAZReader(const std::string& path);
  ~LAZReader() override;
  bool ok() const { return ok_; }

  uint64_t totalCount() const override { return count_; }
  bool hasCount() const override { return true; }
  PointAttributes attributes() const override { return attrs_; }
  bool hasAabb() const override { return true; }
  void aabb(double outMin[3], double outMax[3]) const override;
  bool readChunk(uint64_t maxPoints, PointChunk& out) override;
  void rewind() override;

 private:
  void* laszip_{nullptr};  // laszip_POINTER
  bool ok_{false};
  bool opened_{false};
  std::string path_;
  las_detail::RecordLayout layout_;
  uint64_t count_{0};
  uint64_t cursor_{0};
  double aabbMin_[3]{0, 0, 0};
  double aabbMax_[3]{0, 0, 0};
  PointAttributes attrs_;
};

class LAZWriter : public PointCloudWriter {
 public:
  LAZWriter(const std::string& path, const PointCloudHeader& header);
  ~LAZWriter() override;
  bool ok() const { return ok_; }

  bool writeChunk(const PointChunk& chunk) override;
  bool finalize() override;

 private:
  void* laszip_{nullptr};  // laszip_POINTER
  bool ok_{false};
  bool opened_{false};
  uint64_t written_{0};
  las_detail::RecordLayout layout_;
};

}  // namespace pointcloud
