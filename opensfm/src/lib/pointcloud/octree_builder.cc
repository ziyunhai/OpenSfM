#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include <pointcloud/half_float.h>
#include <pointcloud/morton.h>
#include <pointcloud/octree_builder.h>
#include <pointcloud/octree_internal.h>
#include <pointcloud/tile_io.h>

namespace pointcloud {

// ── AABB ─────────────────────────────────────────────────────────────────────

void AABB::expand(float x, float y, float z) {
  if (x < min[0]) {
    min[0] = x;
  }
  if (y < min[1]) {
    min[1] = y;
  }
  if (z < min[2]) {
    min[2] = z;
  }
  if (x > max[0]) {
    max[0] = x;
  }
  if (y > max[1]) {
    max[1] = y;
  }
  if (z > max[2]) {
    max[2] = z;
  }
}

float AABB::maxExtent() const {
  return std::max({extent(0), extent(1), extent(2)});
}

void AABB::cubePad() {
  float e = maxExtent();
  for (int a = 0; a < 3; ++a) {
    float diff = e - extent(a);
    min[a] -= diff * 0.5f;
    max[a] += diff * 0.5f;
  }
}

// ── Shared geometry kernels ──────────────────────────────────────────────────

void childAabb(const float min[3], const float max[3], int oct, float outMin[3],
               float outMax[3]) {
  float cx = (min[0] + max[0]) * 0.5f;
  float cy = (min[1] + max[1]) * 0.5f;
  float cz = (min[2] + max[2]) * 0.5f;
  outMin[0] = (oct & 1) ? cx : min[0];
  outMin[1] = (oct & 2) ? cy : min[1];
  outMin[2] = (oct & 4) ? cz : min[2];
  outMax[0] = (oct & 1) ? max[0] : cx;
  outMax[1] = (oct & 2) ? max[1] : cy;
  outMax[2] = (oct & 4) ? max[2] : cz;
}

// ── Tile encoding ────────────────────────────────────────────────────────────

PointRecord makePointRecord(uint64_t idx, const PointSource& src,
                            const float tileMin[3], const float tileMax[3],
                            float defaultRadius) {
  PointRecord rec{};

  float p[3];
  src.position(idx, p);

  // Relative position within the tile AABB → [0,1] → half-float.
  float rangeX = tileMax[0] - tileMin[0];
  float rangeY = tileMax[1] - tileMin[1];
  float rangeZ = tileMax[2] - tileMin[2];
  if (rangeX < 1e-12f) {
    rangeX = 1.0f;
  }
  if (rangeY < 1e-12f) {
    rangeY = 1.0f;
  }
  if (rangeZ < 1e-12f) {
    rangeZ = 1.0f;
  }

  rec.px = floatToHalf((p[0] - tileMin[0]) / rangeX);
  rec.py = floatToHalf((p[1] - tileMin[1]) / rangeY);
  rec.pz = floatToHalf((p[2] - tileMin[2]) / rangeZ);

  // Normal (snorm int8).
  float n[3];
  if (src.normal(idx, n)) {
    auto clampSnorm = [](float v) -> int8_t {
      return static_cast<int8_t>(
          std::max(-127.0f, std::min(127.0f, v * 127.0f)));
    };
    rec.nx = clampSnorm(n[0]);
    rec.ny = clampSnorm(n[1]);
    rec.nz = clampSnorm(n[2]);
  }

  // Color.
  uint8_t c[3];
  if (src.color(idx, c)) {
    rec.r = c[0];
    rec.g = c[1];
    rec.b = c[2];
  }

  // Radius.
  rec.radius = floatToHalf(src.radius(idx, defaultRadius));

  std::memset(rec._pad, 0, sizeof(rec._pad));
  return rec;
}

void strideSubsample(const std::vector<uint64_t>& indices, uint32_t lodCount,
                     std::vector<uint64_t>& out) {
  out.clear();
  uint64_t n = indices.size();
  if (lodCount == 0 || n == 0) {
    return;
  }
  if (static_cast<uint64_t>(lodCount) >= n) {
    out = indices;
    return;
  }
  out.reserve(lodCount);
  float stride = static_cast<float>(n) / static_cast<float>(lodCount);
  for (uint32_t i = 0; i < lodCount; ++i) {
    out.push_back(indices[static_cast<size_t>(i * stride)]);
  }
}

// ── Recursive octree construction ────────────────────────────────────────────

void buildNode(const std::string& key, int depth,
               std::vector<uint64_t>& indices, const float aabbMin[3],
               const float aabbMax[3], const PointSource& src,
               const OctreeBuilderConfig& config, OctreeMetadata& meta,
               int& nodesWritten, const ProgressCallback& progress) {
  if (indices.empty()) {
    return;
  }

  float extent = std::max({aabbMax[0] - aabbMin[0], aabbMax[1] - aabbMin[1],
                           aabbMax[2] - aabbMin[2]});

  bool isLeaf =
      (indices.size() <= config.maxPointsPerTile) || (depth >= config.maxDepth);

  if (isLeaf) {
    // Write all points into this tile.  Spacing reflects the tile content.
    float spacing =
        extent /
        std::sqrt(static_cast<float>(std::max(size_t(1), indices.size())));
    TileHeader hdr{};
    hdr.magic = kTileMagic;
    hdr.version = kTileVersion;
    hdr.numPoints = static_cast<uint32_t>(indices.size());
    hdr.childMask = 0;
    std::copy_n(aabbMin, 3, hdr.aabbMin);
    std::copy_n(aabbMax, 3, hdr.aabbMax);
    hdr.spacing = spacing;
    hdr.depth = static_cast<uint32_t>(depth);

    std::vector<PointRecord> records;
    records.reserve(indices.size());
    for (uint64_t idx : indices) {
      records.push_back(makePointRecord(idx, src, aabbMin, aabbMax, spacing));
    }

    writeTile(config.outputDir, key, hdr, records);
    meta.totalPoints += indices.size();
    if (depth > meta.maxDepth) {
      meta.maxDepth = depth;
    }

    ++nodesWritten;
    if (progress) {
      progress(nodesWritten, 0);
    }
    return;
  }

  // ── Inner node: subdivide into 8 octants ──

  float cx = (aabbMin[0] + aabbMax[0]) * 0.5f;
  float cy = (aabbMin[1] + aabbMax[1]) * 0.5f;
  float cz = (aabbMin[2] + aabbMax[2]) * 0.5f;

  // Partition (stable: preserves Morton order within each child).
  std::vector<uint64_t> childIndices[8];
  for (uint64_t idx : indices) {
    float p[3];
    src.position(idx, p);
    childIndices[octantOf(p, cx, cy, cz)].push_back(idx);
  }

  // LOD subsample of this inner node (stride over Morton order → uniform).
  std::vector<uint64_t> lodIndices;
  strideSubsample(indices, config.lodSampleCount, lodIndices);

  uint32_t childMask = 0;
  for (int oct = 0; oct < 8; ++oct) {
    if (!childIndices[oct].empty()) {
      childMask |= (1u << oct);
    }
  }

  // Write this inner node's LOD tile.  Spacing reflects the LOD subsample
  // stored in this tile (the traversal uses it to decide when children are
  // needed for more detail).
  float spacing =
      extent /
      std::sqrt(static_cast<float>(std::max(size_t(1), lodIndices.size())));
  TileHeader hdr{};
  hdr.magic = kTileMagic;
  hdr.version = kTileVersion;
  hdr.numPoints = static_cast<uint32_t>(lodIndices.size());
  hdr.childMask = childMask;
  std::copy_n(aabbMin, 3, hdr.aabbMin);
  std::copy_n(aabbMax, 3, hdr.aabbMax);
  hdr.spacing = spacing;
  hdr.depth = static_cast<uint32_t>(depth);

  std::vector<PointRecord> records;
  records.reserve(lodIndices.size());
  for (uint64_t idx : lodIndices) {
    records.push_back(makePointRecord(idx, src, aabbMin, aabbMax, spacing));
  }

  writeTile(config.outputDir, key, hdr, records);
  meta.totalPoints += lodIndices.size();
  if (depth > meta.maxDepth) {
    meta.maxDepth = depth;
  }

  ++nodesWritten;
  if (progress) {
    progress(nodesWritten, 0);
  }

  // Free the LOD buffer before recursing (children own the bulk memory).
  lodIndices.clear();
  lodIndices.shrink_to_fit();

  // Recurse into children.
  for (int oct = 0; oct < 8; ++oct) {
    if (childIndices[oct].empty()) {
      continue;
    }
    float childMin[3], childMax[3];
    childAabb(aabbMin, aabbMax, oct, childMin, childMax);
    std::string childKey = key + std::to_string(oct);
    buildNode(childKey, depth + 1, childIndices[oct], childMin, childMax, src,
              config, meta, nodesWritten, progress);
  }
}

// ── Public API: in-core builder ──────────────────────────────────────────────

OctreeMetadata buildOctree(const BuilderInput& input,
                           const OctreeBuilderConfig& config,
                           ProgressCallback progress) {
  OctreeMetadata meta{};

  if (input.numPoints == 0 || input.positions == nullptr) {
    return meta;
  }

  RawPointSource src(input);

  // 1. Global AABB + cube-pad.
  AABB aabb;
  for (uint64_t i = 0; i < input.numPoints; ++i) {
    const float* p = input.positions + i * 3;
    aabb.expand(p[0], p[1], p[2]);
  }
  aabb.cubePad();

  meta.aabbMin = {aabb.min[0], aabb.min[1], aabb.min[2]};
  meta.aabbMax = {aabb.max[0], aabb.max[1], aabb.max[2]};
  meta.maxPointsPerTile = config.maxPointsPerTile;

  float rootExtent = aabb.maxExtent();
  uint64_t rootLodCount =
      std::min(static_cast<uint64_t>(config.lodSampleCount), input.numPoints);
  meta.rootSpacing =
      rootExtent /
      std::sqrt(static_cast<float>(std::max(rootLodCount, uint64_t(1))));

  // 2. Morton codes (over the global cube).
  float rangeInv[3];
  for (int a = 0; a < 3; ++a) {
    float ext = aabb.max[a] - aabb.min[a];
    rangeInv[a] = (ext > 1e-12f) ? (1.0f / ext) : 0.0f;
  }

  std::vector<SortEntry> entries(input.numPoints);
  for (uint64_t i = 0; i < input.numPoints; ++i) {
    const float* p = input.positions + i * 3;
    uint32_t qx = quantise(p[0], aabb.min[0], rangeInv[0]);
    uint32_t qy = quantise(p[1], aabb.min[1], rangeInv[1]);
    uint32_t qz = quantise(p[2], aabb.min[2], rangeInv[2]);
    entries[i].morton = mortonEncode(qx, qy, qz);
    entries[i].index = i;
  }

  // 3. Deterministic sort by (morton, index).
  std::sort(entries.begin(), entries.end(), sortEntryLess);

  std::vector<uint64_t> sortedIndices(input.numPoints);
  for (uint64_t i = 0; i < input.numPoints; ++i) {
    sortedIndices[i] = entries[i].index;
  }
  std::vector<SortEntry>().swap(entries);  // free

  // 4. Build octree recursively and write tiles.
  int nodesWritten = 0;
  buildNode("r", 0, sortedIndices, aabb.min, aabb.max, src, config, meta,
            nodesWritten, progress);

  // 5. Write metadata.
  writeMetadata(config.outputDir, meta);

  return meta;
}

}  // namespace pointcloud
