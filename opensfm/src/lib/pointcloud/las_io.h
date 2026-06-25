#pragma once

/// LAS 1.4 reader and writer (dependency-free).
///
/// Uses Point Data Record Format 7 (legacy XYZ int32 + intensity + return bits
/// + classification + RGB).  Our per-point `class` label maps to the native
/// LAS classification byte; per-point normals are carried in an "Extra Bytes"
/// VLR (three float32 dimensions nx/ny/nz, appended to each record).  Positions
/// use the LAS integer scale/offset model for precision.
///
/// The record codec in `las_detail` is shared with the LAZ back-end (LAZ only
/// swaps the compressed byte source).

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include <pointcloud/point_cloud_io.h>

namespace pointcloud {

namespace las_detail {

// Layout of a single PDRF-7 point record (+ optional f32 normal extra bytes).
struct RecordLayout {
  uint16_t recordLength = 36;  // 36 (PDRF7 base) or 48 (+ nx,ny,nz f32)
  bool hasNormals = false;
  double scale[3] = {0.001, 0.001, 0.001};
  double offset[3] = {0.0, 0.0, 0.0};

  // Field byte offsets within a record.
  static constexpr int kX = 0, kY = 4, kZ = 8;
  static constexpr int kClass = 16;
  static constexpr int kRed = 30, kGreen = 32, kBlue = 34;
  static constexpr int kBaseSize = 36;
  static constexpr int kNormalOffset = 36;  // when hasNormals
};

void encodeRecord(const RecordLayout& layout, const double pos[3],
                  const float nrm[3], const uint8_t col[3], uint8_t label,
                  uint8_t* dst);
void decodeRecord(const RecordLayout& layout, const uint8_t* src, double pos[3],
                  float nrm[3], uint8_t col[3], uint8_t& label);

// Build the 375-byte LAS 1.4 public header (counts/min-max patched later).
std::vector<uint8_t> buildPublicHeader(const RecordLayout& layout,
                                       uint32_t numVlrs,
                                       uint32_t offsetToPointData);
// Build the Extra Bytes VLR (54-byte header + 3×192-byte nx/ny/nz f32 descrs).
std::vector<uint8_t> buildNormalExtraBytesVlr();
// Build the OGC Coordinate System WKT VLR (LASF_Projection, record id 2112).
// The payload is the null-terminated WKT.  Empty input → empty vector.
std::vector<uint8_t> buildWktVlr(const std::string& wkt);

constexpr int kHeaderSize = 375;
constexpr int kVlrHeaderSize = 54;
constexpr int kExtraByteDescrSize = 192;

}  // namespace las_detail

class LASReader : public PointCloudReader {
 public:
  explicit LASReader(const std::string& path);
  bool ok() const { return ok_; }

  uint64_t totalCount() const override { return count_; }
  bool hasCount() const override { return true; }
  PointAttributes attributes() const override { return attrs_; }
  bool hasAabb() const override { return true; }
  void aabb(double outMin[3], double outMax[3]) const override;
  bool readChunk(uint64_t maxPoints, PointChunk& out) override;
  void rewind() override;

 private:
  std::ifstream in_;
  bool ok_{false};
  las_detail::RecordLayout layout_;
  uint64_t count_{0};
  uint64_t cursor_{0};
  uint64_t pointDataOffset_{0};
  double aabbMin_[3]{0, 0, 0};
  double aabbMax_[3]{0, 0, 0};
  PointAttributes attrs_;
};

class LASWriter : public PointCloudWriter {
 public:
  LASWriter(const std::string& path, const PointCloudHeader& header);
  bool ok() const { return out_.good(); }

  bool writeChunk(const PointChunk& chunk) override;
  bool finalize() override;

 private:
  std::ofstream out_;
  las_detail::RecordLayout layout_;
  uint64_t written_{0};
  uint32_t numVlrs_{0};
  uint32_t pointDataOffset_{0};
  double aabbMin_[3]{1e30, 1e30, 1e30};
  double aabbMax_[3]{-1e30, -1e30, -1e30};
  bool headerOk_{false};
};

}  // namespace pointcloud
