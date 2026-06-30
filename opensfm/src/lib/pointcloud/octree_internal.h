#pragma once

/// Internal kernel shared by the in-core builder (`octree_builder.cc`) and the
/// out-of-core builder (`ooc_octree.cc`).
///
/// The two builders differ only in HOW point data is reached (raw NumPy SoA
/// pointers vs. a memory-mapped PLY) and in HOW points are ordered/partitioned
/// before recursion (one global sort vs. external bucketing).  Everything that
/// determines the on-disk tile bytes — octant split, child AABBs, Morton key,
/// LOD subsample, and `makePointRecord` — lives here so both paths produce
/// format-identical tiles.
///
/// This header is library-internal; the Qt viewer only ever includes
/// `tile_format.h` / `tile_io.h` / `half_float.h`.

#include <cstdint>
#include <string>
#include <vector>

#include <pointcloud/octree_builder.h>
#include <pointcloud/tile_format.h>

namespace pointcloud {

// ── Indexed point source ─────────────────────────────────────────────────────
//
// Abstracts random access to point attributes by global index so the in-core
// recursion and the out-of-core recursion share one encoder + traversal.
struct PointSource {
  virtual ~PointSource() = default;
  virtual void position(uint64_t i, float out[3]) const = 0;
  /// Returns false when the source carries no normals (out left untouched).
  virtual bool normal(uint64_t i, float out[3]) const = 0;
  /// Returns false when the source carries no colors (out left untouched).
  virtual bool color(uint64_t i, uint8_t out[3]) const = 0;
  /// Per-point splat radius, or `dflt` when the source has none.
  virtual float radius(uint64_t i, float dflt) const = 0;
};

/// Adapter over the raw SoA pointers in `BuilderInput` (the in-core path).
class RawPointSource : public PointSource {
 public:
  explicit RawPointSource(const BuilderInput& in) : in_(in) {}

  void position(uint64_t i, float out[3]) const override {
    const float* p = in_.positions + i * 3;
    out[0] = p[0];
    out[1] = p[1];
    out[2] = p[2];
  }
  bool normal(uint64_t i, float out[3]) const override {
    if (!in_.normals) {
      return false;
    }
    const float* n = in_.normals + i * 3;
    out[0] = n[0];
    out[1] = n[1];
    out[2] = n[2];
    return true;
  }
  bool color(uint64_t i, uint8_t out[3]) const override {
    if (!in_.colors) {
      return false;
    }
    const uint8_t* c = in_.colors + i * 3;
    out[0] = c[0];
    out[1] = c[1];
    out[2] = c[2];
    return true;
  }
  float radius(uint64_t i, float dflt) const override {
    return in_.radii ? in_.radii[i] : dflt;
  }

 private:
  const BuilderInput& in_;
};

// ── Axis-aligned bounding box (float, matches tile/world precision) ───────────

struct AABB {
  float min[3]{1e30f, 1e30f, 1e30f};
  float max[3]{-1e30f, -1e30f, -1e30f};

  void expand(float x, float y, float z);
  float extent(int axis) const { return max[axis] - min[axis]; }
  float maxExtent() const;
  /// Symmetrically expand the two shorter axes so the box is a cube.
  void cubePad();
};

// ── Shared geometry kernels ──────────────────────────────────────────────────

/// Octant of a point relative to a node center (center-split; matches the
/// viewer's traversal).  Bit 0 = x, 1 = y, 2 = z; boundary rule is `>=`.
inline int octantOf(const float p[3], float cx, float cy, float cz) {
  int oct = 0;
  if (p[0] >= cx) {
    oct |= 1;
  }
  if (p[1] >= cy) {
    oct |= 2;
  }
  if (p[2] >= cz) {
    oct |= 4;
  }
  return oct;
}

/// AABB of child octant `oct` within [min, max] (center-split).
void childAabb(const float min[3], const float max[3], int oct, float outMin[3],
               float outMax[3]);

// ── Deterministic Morton sort entry ──────────────────────────────────────────

struct SortEntry {
  uint64_t morton;
  uint64_t index;
};

/// Total order on (morton, index) — deterministic tie-break so the in-core and
/// out-of-core builders agree regardless of chunking/threading.
inline bool sortEntryLess(const SortEntry& a, const SortEntry& b) {
  return (a.morton != b.morton) ? (a.morton < b.morton) : (a.index < b.index);
}

// ── Tile encoding + recursion ────────────────────────────────────────────────

/// Encode one point into a tile `PointRecord` (half-float position relative to
/// the tile AABB, int8-snorm normal, RGB, half-float radius).
PointRecord makePointRecord(uint64_t idx, const PointSource& src,
                            const float tileMin[3], const float tileMax[3],
                            float defaultRadius);

/// Stride subsample of a Morton-ordered index range into `out` (≤ lodCount).
void strideSubsample(const std::vector<uint64_t>& indices, uint32_t lodCount,
                     std::vector<uint64_t>& out);

/// Recursively build + write the subtree rooted at `key` from a Morton-sorted
/// index slice.  Shared by the in-core builder (key "r", depth 0) and the
/// out-of-core per-bucket builder (key = bucket node key, depth = splitDepth).
void buildNode(const std::string& key, int depth,
               std::vector<uint64_t>& indices, const float aabbMin[3],
               const float aabbMax[3], const PointSource& src,
               const OctreeBuilderConfig& config, OctreeMetadata& meta,
               int& nodesWritten, const ProgressCallback& progress);

}  // namespace pointcloud
