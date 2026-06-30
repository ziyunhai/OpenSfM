#pragma once

/// Binary/ASCII PLY reader and writer.
///
/// `PLYReader` memory-maps the file (via vendored mio) and decodes points by a
/// parsed property table — supporting our canonical dense layout
/// (x y z nx ny nz red green blue class, binary little-endian, 28-byte rows)
/// plus reasonable variants.  For the canonical binary layout it also exposes a
/// `mappedBody()` random-access view used by the out-of-core octree.
///
/// `PLYWriter` writes the canonical 28-byte binary layout (zeros where an
/// attribute is absent), byte-identical to `opensfm/io.py::point_cloud_to_ply`.

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

// mio::mmap_source comes bundled inside the vendored csv2 single-header
// (the same copy map/tracks_manager.cc uses) — avoid a second mio copy.
#include <csv2/csv2.hpp>
#include <pointcloud/point_cloud_io.h>

namespace pointcloud {

class PLYReader : public PointCloudReader {
 public:
  explicit PLYReader(const std::string& path);
  bool ok() const { return ok_; }

  uint64_t totalCount() const override { return count_; }
  bool hasCount() const override { return true; }
  PointAttributes attributes() const override { return attrs_; }
  bool hasAabb() const override { return false; }
  void aabb(double outMin[3], double outMax[3]) const override;
  bool readChunk(uint64_t maxPoints, PointChunk& out) override;
  void rewind() override { cursor_ = 0; }
  MappedBody mappedBody() override;

 private:
  enum class Format { Ascii, BinaryLE, BinaryBE };

  // One parsed PLY property: byte offset + element type.
  struct Property {
    std::string name;
    int typeCode{0};  // see typeSizeOf
    int size{0};
    uint64_t offset{0};  // byte offset within a binary record
  };

  bool parseHeader();
  // Decode the i-th point (binary) from a mapped record pointer.
  void decodeRecord(const uint8_t* rec, double pos[3], float nrm[3],
                    uint8_t col[3], uint8_t& lbl) const;

  mio::mmap_source map_;
  bool ok_{false};
  Format format_{Format::BinaryLE};
  uint64_t count_{0};
  uint64_t dataOffset_{0};  // byte offset of first record
  uint32_t recordStride_{0};
  PointAttributes attrs_;

  // Resolved field offsets within a record (−1 == absent). For binary.
  int posOff_[3]{-1, -1, -1};
  int posType_{0};  // 0 == f32, 1 == f64
  int nrmOff_[3]{-1, -1, -1};
  int nrmType_{0};
  int colOff_[3]{-1, -1, -1};
  int lblOff_{-1};
  int lblType_{0};

  // Sequential read state.
  uint64_t cursor_{0};     // next point index
  uint64_t asciiByte_{0};  // byte offset into the mapped body (ASCII path)
  std::vector<Property> properties_;  // declaration order (ASCII column order)
};

class PLYWriter : public PointCloudWriter {
 public:
  PLYWriter(const std::string& path, const PointCloudHeader& header);
  bool ok() const { return out_.good(); }

  bool writeChunk(const PointChunk& chunk) override;
  bool finalize() override;

 private:
  std::ofstream out_;
  std::string path_;
  uint64_t written_{0};
  bool headerWritten_{false};
  std::streampos countPos_{0};  // file offset of the vertex-count digits
  void writeHeaderPlaceholder();
};

}  // namespace pointcloud
