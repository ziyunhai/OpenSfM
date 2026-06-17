#pragma once

/// Point-cloud read/write abstraction for the dense pipeline.
///
/// A small class hierarchy with concrete PLY / LAS / LAZ back-ends, selected by
/// file extension via `makeReader` / `makeWriter`.  Readers stream points in
/// chunks (so multi-gigabyte clouds never need to be fully resident); binary
/// PLY additionally exposes a memory-mapped random-access fast path used by the
/// out-of-core octree builder.
///
/// Positions are carried as `double` so the LAS scale/offset integer encoding
/// round-trips losslessly; they are narrowed to `float` only at tile encode.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace pointcloud {

/// Which optional attributes a stream carries.
struct PointAttributes {
  bool hasNormals{false};
  bool hasColors{false};
  bool hasLabels{false};  // class / classification
};

/// A chunk of points in SoA layout.
struct PointChunk {
  uint64_t count{0};
  std::vector<double> positions;  // 3*count, world-space
  std::vector<float> normals;     // 3*count or empty
  std::vector<uint8_t> colors;    // 3*count or empty
  std::vector<uint8_t> labels;    // count or empty

  void clear();
  /// Resize buffers for `n` points, allocating only the present attributes.
  void resize(uint64_t n, const PointAttributes& attrs);
};

/// Describes a cloud a writer must produce.
struct PointCloudHeader {
  uint64_t pointCount{0};  // 0 == unknown / streaming (patched on finalize)
  PointAttributes attrs;
  bool hasAabb{false};
  double aabbMin[3]{0, 0, 0};
  double aabbMax[3]{0, 0, 0};
  // LAS scale/offset hints (0 scale → auto-derive). Ignored by PLY.
  double scale[3]{0, 0, 0};
  double offset[3]{0, 0, 0};
};

class PointCloudReader {
 public:
  /// Random-access description of a binary point body (mmap fast path).
  /// Only binary PLY fills this; LAS/LAZ leave it invalid (streaming only).
  struct MappedBody {
    const uint8_t* base{nullptr};  // mapped file base
    uint64_t bodyOffset{0};        // byte offset of the first record
    uint32_t recordStride{0};      // bytes per point record
    uint64_t count{0};
    // Per-field byte offset within a record (−1 == absent).
    int posOffset{-1};
    int posType{0};  // 0 == float32, 1 == float64
    int nrmOffset{-1};
    int colOffset{-1};
    int lblOffset{-1};
    bool valid() const { return base != nullptr && recordStride != 0; }
  };

  virtual ~PointCloudReader() = default;

  /// Total point count if known cheaply (PLY/LAS headers), else 0.
  virtual uint64_t totalCount() const = 0;
  virtual bool hasCount() const = 0;
  virtual PointAttributes attributes() const = 0;
  /// True if a bounding box is known cheaply (LAS header); PLY returns false.
  virtual bool hasAabb() const = 0;
  virtual void aabb(double outMin[3], double outMax[3]) const = 0;

  /// Read up to `maxPoints` more points sequentially. Returns false at EOF
  /// (and leaves `out` empty).
  virtual bool readChunk(uint64_t maxPoints, PointChunk& out) = 0;
  /// Restart sequential reading from the first point.
  virtual void rewind() = 0;

  /// Random-access body (binary PLY only); default: not available.
  virtual MappedBody mappedBody() { return MappedBody{}; }
};

class PointCloudWriter {
 public:
  virtual ~PointCloudWriter() = default;
  /// Append a chunk of points. Returns false on I/O error.
  virtual bool writeChunk(const PointChunk& chunk) = 0;
  /// Patch header counts, flush, close. Returns false on I/O error.
  virtual bool finalize() = 0;
};

/// Lowercased file extension including the dot (e.g. ".ply"); "" if none.
std::string fileExtensionLower(const std::string& path);

/// Open a reader for `path`, dispatched by extension. Returns nullptr on
/// unsupported extension or open/parse failure.
std::unique_ptr<PointCloudReader> makeReader(const std::string& path);

/// Open a writer for `path`, dispatched by extension. Returns nullptr on
/// unsupported extension or open failure.
std::unique_ptr<PointCloudWriter> makeWriter(const std::string& path,
                                             const PointCloudHeader& header);

}  // namespace pointcloud
