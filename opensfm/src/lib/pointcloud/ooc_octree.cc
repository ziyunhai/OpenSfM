/// Out-of-core octree builder.
///
/// Builds the same tile set as `buildOctree` but with RAM bounded independently
/// of the point count.  Point data is reached through a memory-mapped binary
/// PLY (random access by index); non-mmappable inputs (LAS/LAZ) are first
/// spilled to a temporary binary PLY.
///
/// Strategy (Schütz 2020-style external build):
///   Pass 1  global AABB (stream).
///   Small/medium clouds (≤ maxBucketPoints): build fully in-core from mmap,
///           reusing `buildNode` — byte-identical to the in-core builder.
///   Large clouds:
///     Pass 2  bucket every point to its depth-D octree node (center-split),
///             spilling (morton,index) pairs to per-bucket temp files.
///     Pass 3  build each depth-D subtree independently from its bucket; emit a
///             bounded LOD sample per bucket root for Pass 4.
///     Pass 4  build the shallow tree (depths D-1 … 0) bottom-up from the
///             bounded bucket samples.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <pointcloud/half_float.h>
#include <pointcloud/morton.h>
#include <pointcloud/octree_builder.h>
#include <pointcloud/octree_internal.h>
#include <pointcloud/point_cloud_io.h>
#include <pointcloud/tile_io.h>

namespace pointcloud {

namespace {

static_assert(sizeof(SortEntry) == 16, "SortEntry must be 16 bytes for spill");

namespace fs = std::filesystem;

// ── Point source over a memory-mapped binary body ────────────────────────────

class MmapPointSource : public PointSource {
 public:
  explicit MmapPointSource(const PointCloudReader::MappedBody& mb) : mb_(mb) {}

  void position(uint64_t i, float out[3]) const override {
    const uint8_t* rec = recPtr(i) + mb_.posOffset;
    if (mb_.posType == 1) {
      double d[3];
      std::memcpy(d, rec, 24);
      out[0] = static_cast<float>(d[0]);
      out[1] = static_cast<float>(d[1]);
      out[2] = static_cast<float>(d[2]);
    } else {
      std::memcpy(out, rec, 12);
    }
  }
  bool normal(uint64_t i, float out[3]) const override {
    if (mb_.nrmOffset < 0) {
      return false;
    }
    std::memcpy(out, recPtr(i) + mb_.nrmOffset, 12);
    return true;
  }
  bool color(uint64_t i, uint8_t out[3]) const override {
    if (mb_.colOffset < 0) {
      return false;
    }
    const uint8_t* p = recPtr(i) + mb_.colOffset;
    out[0] = p[0];
    out[1] = p[1];
    out[2] = p[2];
    return true;
  }
  float radius(uint64_t /*i*/, float dflt) const override { return dflt; }

 private:
  const uint8_t* recPtr(uint64_t i) const {
    return mb_.base + mb_.bodyOffset + i * mb_.recordStride;
  }
  PointCloudReader::MappedBody mb_;
};

// ── Spill (morton,index) entry files ─────────────────────────────────────────

std::string bktPath(const std::string& dir, const std::string& key) {
  return dir + "/bkt_" + key + ".idx";
}
std::string samplePath(const std::string& dir, const std::string& key) {
  return dir + "/sample_" + key + ".idx";
}

void writeEntries(const std::string& path, const std::vector<SortEntry>& e) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) {
    return;
  }
  if (!e.empty()) {
    std::fwrite(e.data(), sizeof(SortEntry), e.size(), f);
  }
  std::fclose(f);
}

std::vector<SortEntry> readEntries(const std::string& path) {
  std::vector<SortEntry> e;
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) {
    return e;
  }
  std::fseek(f, 0, SEEK_END);
  long sz = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (sz > 0) {
    e.resize(static_cast<size_t>(sz) / sizeof(SortEntry));
    if (std::fread(e.data(), sizeof(SortEntry), e.size(), f) != e.size()) {
      e.clear();
    }
  }
  std::fclose(f);
  return e;
}

uint64_t fileEntryCount(const std::string& path) {
  std::error_code ec;
  auto sz = fs::file_size(path, ec);
  return ec ? 0 : static_cast<uint64_t>(sz) / sizeof(SortEntry);
}

// ── Bounded pool of append-mode bucket writers (Pass 2) ──────────────────────

class BucketWriterPool {
 public:
  BucketWriterPool(std::string dir, size_t maxOpen)
      : dir_(std::move(dir)), maxOpen_(maxOpen) {}
  ~BucketWriterPool() { closeAll(); }

  void append(const std::string& key, const SortEntry& e) {
    FILE* f = get(key);
    std::fwrite(&e, sizeof(SortEntry), 1, f);
  }

  void closeAll() {
    for (auto& kv : open_) {
      if (kv.second) {
        std::fclose(kv.second);
      }
    }
    open_.clear();
    lru_.clear();
  }

  const std::set<std::string>& keys() const { return created_; }

 private:
  FILE* get(const std::string& key) {
    auto it = open_.find(key);
    if (it != open_.end()) {
      touch(key);
      return it->second;
    }
    if (open_.size() >= maxOpen_) {
      evictOldest();
    }
    bool firstTime = created_.insert(key).second;
    FILE* f = std::fopen(bktPath(dir_, key).c_str(), firstTime ? "wb" : "ab");
    open_[key] = f;
    lru_.push_back(key);
    return f;
  }
  void touch(const std::string& key) {
    for (auto it = lru_.begin(); it != lru_.end(); ++it) {
      if (*it == key) {
        lru_.erase(it);
        break;
      }
    }
    lru_.push_back(key);
  }
  void evictOldest() {
    if (lru_.empty()) {
      return;
    }
    std::string victim = lru_.front();
    lru_.erase(lru_.begin());
    auto it = open_.find(victim);
    if (it != open_.end()) {
      if (it->second) {
        std::fclose(it->second);
      }
      open_.erase(it);
    }
  }

  std::string dir_;
  size_t maxOpen_;
  std::unordered_map<std::string, FILE*> open_;
  std::vector<std::string> lru_;  // front = oldest
  std::set<std::string> created_;
};

// ── Geometry helpers keyed on the octree-key string ──────────────────────────

void aabbForKey(const std::string& key, const float rootMin[3],
                const float rootMax[3], float outMin[3], float outMax[3]) {
  std::copy_n(rootMin, 3, outMin);
  std::copy_n(rootMax, 3, outMax);
  for (size_t i = 1; i < key.size(); ++i) {  // skip leading 'r'
    int oct = key[i] - '0';
    float cmin[3], cmax[3];
    childAabb(outMin, outMax, oct, cmin, cmax);
    std::copy_n(cmin, 3, outMin);
    std::copy_n(cmax, 3, outMax);
  }
}

std::string bucketKeyForPoint(const float p[3], const float rootMin[3],
                              const float rootMax[3], uint32_t depth) {
  std::string key = "r";
  float mn[3], mx[3];
  std::copy_n(rootMin, 3, mn);
  std::copy_n(rootMax, 3, mx);
  for (uint32_t d = 0; d < depth; ++d) {
    float cx = (mn[0] + mx[0]) * 0.5f;
    float cy = (mn[1] + mx[1]) * 0.5f;
    float cz = (mn[2] + mx[2]) * 0.5f;
    int oct = octantOf(p, cx, cy, cz);
    key += static_cast<char>('0' + oct);
    float cmin[3], cmax[3];
    childAabb(mn, mx, oct, cmin, cmax);
    std::copy_n(cmin, 3, mn);
    std::copy_n(cmax, 3, mx);
  }
  return key;
}

// Write a tile from a list of (morton,index) entries (inner LOD or upper node).
void writeTileFromEntries(const std::string& outputDir, const std::string& key,
                          int depth, const float nodeMin[3],
                          const float nodeMax[3],
                          const std::vector<SortEntry>& ents, uint32_t childMask,
                          const PointSource& src, OctreeMetadata& meta) {
  float extent = std::max({nodeMax[0] - nodeMin[0], nodeMax[1] - nodeMin[1],
                           nodeMax[2] - nodeMin[2]});
  float spacing =
      extent / std::sqrt(static_cast<float>(std::max(size_t(1), ents.size())));
  TileHeader hdr{};
  hdr.magic = kTileMagic;
  hdr.version = kTileVersion;
  hdr.numPoints = static_cast<uint32_t>(ents.size());
  hdr.childMask = childMask;
  std::copy_n(nodeMin, 3, hdr.aabbMin);
  std::copy_n(nodeMax, 3, hdr.aabbMax);
  hdr.spacing = spacing;
  hdr.depth = static_cast<uint32_t>(depth);

  std::vector<PointRecord> recs;
  recs.reserve(ents.size());
  for (const auto& e : ents) {
    recs.push_back(makePointRecord(e.index, src, nodeMin, nodeMax, spacing));
  }
  writeTile(outputDir, key, hdr, recs);
  meta.totalPoints += ents.size();
  if (depth > meta.maxDepth) {
    meta.maxDepth = depth;
  }
}

// ── Pass-3 bucket builder (recursive for over-full buckets) ──────────────────

struct OocContext {
  const OctreeBuilderConfig* config;
  const PointSource* src;
  std::string tempDir;
  uint32_t splitDepth;
  uint64_t maxBucketPoints;
  float rootMin[3];
  float rootMax[3];
  OctreeMetadata* meta;
  int nodesWritten{0};
};

// Stride-pick ≤ lodCount entries from a sorted entry list (Morton order).
std::vector<SortEntry> sampleEntries(const std::vector<SortEntry>& sorted,
                                     uint32_t lodCount) {
  std::vector<SortEntry> out;
  uint64_t n = sorted.size();
  if (lodCount == 0 || n == 0) {
    return out;
  }
  if (static_cast<uint64_t>(lodCount) >= n) {
    return sorted;
  }
  out.reserve(lodCount);
  float stride = static_cast<float>(n) / static_cast<float>(lodCount);
  for (uint32_t i = 0; i < lodCount; ++i) {
    out.push_back(sorted[static_cast<size_t>(i * stride)]);
  }
  return out;
}

// Build the subtree rooted at `key` (depth `depth`) from spill file `idxPath`
// holding `count` entries.  When `emitSample`, writes the node's bounded LOD
// sample to sample_<key>.idx for Pass 4.
void buildBucket(OocContext& ctx, const std::string& key, int depth,
                 const std::string& idxPath, uint64_t count,
                 const float nodeMin[3], const float nodeMax[3],
                 bool emitSample) {
  if (count == 0) {
    return;
  }
  const OctreeBuilderConfig& config = *ctx.config;
  const PointSource& src = *ctx.src;

  // Fits in RAM (and not a degenerate huge leaf) → exact in-core subtree.
  if (count <= ctx.maxBucketPoints) {
    std::vector<SortEntry> entries = readEntries(idxPath);
    std::sort(entries.begin(), entries.end(), sortEntryLess);
    std::vector<uint64_t> indices;
    indices.reserve(entries.size());
    for (const auto& e : entries) {
      indices.push_back(e.index);
    }
    float mn[3], mx[3];
    std::copy_n(nodeMin, 3, mn);
    std::copy_n(nodeMax, 3, mx);
    buildNode(key, depth, indices, mn, mx, src, config, *ctx.meta,
              ctx.nodesWritten, nullptr);
    if (emitSample) {
      writeEntries(samplePath(ctx.tempDir, key),
                   sampleEntries(entries, config.lodSampleCount));
    }
    return;
  }

  // Over-full node.  If we cannot subdivide further, write a bounded leaf
  // sample (degenerate: many ~coincident points) rather than OOM.
  float cx = (nodeMin[0] + nodeMax[0]) * 0.5f;
  float cy = (nodeMin[1] + nodeMax[1]) * 0.5f;
  float cz = (nodeMin[2] + nodeMax[2]) * 0.5f;

  // Stream the spill file: partition into 8 child files + collect a stride
  // sample in file order.
  FILE* in = std::fopen(idxPath.c_str(), "rb");
  if (!in) {
    return;
  }
  std::string childPaths[8];
  FILE* childF[8] = {nullptr};
  uint64_t childCount[8] = {0};
  bool subdividable = depth < config.maxDepth;
  for (int o = 0; o < 8 && subdividable; ++o) {
    childPaths[o] = bktPath(ctx.tempDir, key + static_cast<char>('0' + o));
    childF[o] = std::fopen(childPaths[o].c_str(), "wb");
  }

  std::vector<SortEntry> sample;
  uint32_t lod = config.lodSampleCount;
  if (lod > 0) {
    sample.reserve(lod);
  }
  uint64_t pickK = 0;
  uint64_t nextPick = (lod > 0) ? (pickK * count) / lod : count;

  const size_t kBlock = 1u << 16;
  std::vector<SortEntry> buf(kBlock);
  uint64_t i = 0;
  size_t got;
  while ((got = std::fread(buf.data(), sizeof(SortEntry), kBlock, in)) > 0) {
    for (size_t b = 0; b < got; ++b, ++i) {
      const SortEntry& e = buf[b];
      if (subdividable) {
        float p[3];
        src.position(e.index, p);
        int oct = octantOf(p, cx, cy, cz);
        std::fwrite(&e, sizeof(SortEntry), 1, childF[oct]);
        ++childCount[oct];
      }
      if (lod > 0 && pickK < lod && i == nextPick) {
        sample.push_back(e);
        ++pickK;
        nextPick = (pickK * count) / lod;
      }
    }
  }
  std::fclose(in);
  for (int o = 0; o < 8; ++o) {
    if (childF[o]) {
      std::fclose(childF[o]);
    }
  }

  // Order the bounded sample by (morton, index) for a spatially-uniform tile.
  std::sort(sample.begin(), sample.end(), sortEntryLess);

  uint32_t childMask = 0;
  for (int o = 0; o < 8; ++o) {
    if (childCount[o] > 0) {
      childMask |= (1u << o);
    }
  }

  writeTileFromEntries(config.outputDir, key, depth, nodeMin, nodeMax, sample,
                       childMask, src, *ctx.meta);
  ++ctx.nodesWritten;
  if (emitSample) {
    writeEntries(samplePath(ctx.tempDir, key), sample);
  }

  // Recurse into children, then delete their spill files.
  for (int o = 0; o < 8; ++o) {
    if (childCount[o] == 0) {
      continue;
    }
    float cmin[3], cmax[3];
    childAabb(nodeMin, nodeMax, o, cmin, cmax);
    buildBucket(ctx, key + static_cast<char>('0' + o), depth + 1, childPaths[o],
                childCount[o], cmin, cmax, /*emitSample=*/false);
    std::error_code ec;
    fs::remove(childPaths[o], ec);
  }
}

// ── Pass 4: shallow tree (depths D-1 … 0) from bucket samples ─────────────────

void buildUpperLevels(OocContext& ctx,
                      const std::set<std::string>& bucketKeys) {
  const OctreeBuilderConfig& config = *ctx.config;
  // Live keys at the current depth (start at depth D = bucket roots).
  std::set<std::string> live = bucketKeys;

  for (int d = static_cast<int>(ctx.splitDepth) - 1; d >= 0; --d) {
    // Parents at depth d of the live (depth d+1) nodes.
    std::set<std::string> parents;
    for (const auto& k : live) {
      parents.insert(k.substr(0, k.size() - 1));
    }
    std::set<std::string> newLive;
    for (const auto& pkey : parents) {
      // Merge child samples + childMask.
      std::vector<SortEntry> merged;
      uint32_t childMask = 0;
      for (int o = 0; o < 8; ++o) {
        std::string ckey = pkey + static_cast<char>('0' + o);
        if (!live.count(ckey)) {
          continue;
        }
        childMask |= (1u << o);
        std::vector<SortEntry> cs = readEntries(samplePath(ctx.tempDir, ckey));
        merged.insert(merged.end(), cs.begin(), cs.end());
      }
      std::sort(merged.begin(), merged.end(), sortEntryLess);
      std::vector<SortEntry> sample =
          sampleEntries(merged, config.lodSampleCount);

      float nmin[3], nmax[3];
      aabbForKey(pkey, ctx.rootMin, ctx.rootMax, nmin, nmax);
      writeTileFromEntries(config.outputDir, pkey, d, nmin, nmax, sample,
                           childMask, *ctx.src, *ctx.meta);
      ++ctx.nodesWritten;
      writeEntries(samplePath(ctx.tempDir, pkey), sample);
      newLive.insert(pkey);
    }
    live.swap(newLive);
  }
}

}  // namespace

// ── Public entry point ───────────────────────────────────────────────────────

OctreeMetadata buildOctreeFromFile(const std::string& cloudPath,
                                   const OocConfig& oconfig,
                                   ProgressCallback progress) {
  OctreeMetadata meta{};
  const OctreeBuilderConfig& config = oconfig.base;

  std::string tempDir =
      oconfig.tempDir.empty() ? (config.outputDir + "/_ooc_tmp") : oconfig.tempDir;
  std::error_code ec;
  fs::create_directories(tempDir, ec);

  // Open the cloud; obtain a random-access mmap view.  Non-mmappable inputs
  // (LAS/LAZ) are spilled to a temporary canonical PLY first.
  auto reader = makeReader(cloudPath);
  if (!reader) {
    fs::remove_all(tempDir, ec);
    return meta;
  }
  std::string spillPly;
  PointCloudReader::MappedBody mb = reader->mappedBody();
  std::unique_ptr<PointCloudReader> spillReader;
  if (!mb.valid()) {
    spillPly = tempDir + "/_spill.ply";
    PointCloudHeader h;
    h.attrs = reader->attributes();
    auto w = makeWriter(spillPly, h);
    if (!w) {
      fs::remove_all(tempDir, ec);
      return meta;
    }
    PointChunk c;
    while (reader->readChunk(1u << 20, c)) {
      w->writeChunk(c);
    }
    w->finalize();
    spillReader = makeReader(spillPly);
    if (!spillReader) {
      fs::remove_all(tempDir, ec);
      return meta;
    }
    mb = spillReader->mappedBody();
  }
  if (!mb.valid() || mb.count == 0) {
    fs::remove_all(tempDir, ec);
    return meta;
  }

  MmapPointSource src(mb);
  const uint64_t n = mb.count;

  // Pass 1: global AABB + cube-pad.
  AABB aabb;
  for (uint64_t i = 0; i < n; ++i) {
    float p[3];
    src.position(i, p);
    aabb.expand(p[0], p[1], p[2]);
  }
  aabb.cubePad();

  meta.aabbMin = {aabb.min[0], aabb.min[1], aabb.min[2]};
  meta.aabbMax = {aabb.max[0], aabb.max[1], aabb.max[2]};
  meta.maxPointsPerTile = config.maxPointsPerTile;
  float rootExtent = aabb.maxExtent();
  uint64_t rootLodCount =
      std::min(static_cast<uint64_t>(config.lodSampleCount), n);
  meta.rootSpacing =
      rootExtent /
      std::sqrt(static_cast<float>(std::max(rootLodCount, uint64_t(1))));

  float rangeInv[3];
  for (int a = 0; a < 3; ++a) {
    float ext = aabb.max[a] - aabb.min[a];
    rangeInv[a] = (ext > 1e-12f) ? (1.0f / ext) : 0.0f;
  }

  // Small/medium: exact in-core build from the mmap (≤ maxBucketPoints).
  if (n <= oconfig.maxBucketPoints) {
    std::vector<SortEntry> entries(n);
    for (uint64_t i = 0; i < n; ++i) {
      float p[3];
      src.position(i, p);
      uint32_t qx = quantise(p[0], aabb.min[0], rangeInv[0]);
      uint32_t qy = quantise(p[1], aabb.min[1], rangeInv[1]);
      uint32_t qz = quantise(p[2], aabb.min[2], rangeInv[2]);
      entries[i].morton = mortonEncode(qx, qy, qz);
      entries[i].index = i;
    }
    std::sort(entries.begin(), entries.end(), sortEntryLess);
    std::vector<uint64_t> indices(n);
    for (uint64_t i = 0; i < n; ++i) {
      indices[i] = entries[i].index;
    }
    std::vector<SortEntry>().swap(entries);
    int nodesWritten = 0;
    buildNode("r", 0, indices, aabb.min, aabb.max, src, config, meta,
              nodesWritten, progress);
    writeMetadata(config.outputDir, meta);
    fs::remove_all(tempDir, ec);
    return meta;
  }

  // Large cloud: external bucket build.
  // Pass 2: bucket every point to its depth-D node.
  {
    BucketWriterPool pool(tempDir, /*maxOpen=*/256);
    for (uint64_t i = 0; i < n; ++i) {
      float p[3];
      src.position(i, p);
      uint32_t qx = quantise(p[0], aabb.min[0], rangeInv[0]);
      uint32_t qy = quantise(p[1], aabb.min[1], rangeInv[1]);
      uint32_t qz = quantise(p[2], aabb.min[2], rangeInv[2]);
      SortEntry e;
      e.morton = mortonEncode(qx, qy, qz);
      e.index = i;
      std::string key =
          bucketKeyForPoint(p, aabb.min, aabb.max, oconfig.splitDepth);
      pool.append(key, e);
    }
    pool.closeAll();

    OocContext ctx;
    ctx.config = &config;
    ctx.src = &src;
    ctx.tempDir = tempDir;
    ctx.splitDepth = oconfig.splitDepth;
    ctx.maxBucketPoints = oconfig.maxBucketPoints;
    std::copy_n(aabb.min, 3, ctx.rootMin);
    std::copy_n(aabb.max, 3, ctx.rootMax);
    ctx.meta = &meta;

    // Pass 3: build each depth-D subtree.
    std::set<std::string> bucketKeys = pool.keys();
    for (const auto& key : bucketKeys) {
      std::string path = bktPath(tempDir, key);
      uint64_t cnt = fileEntryCount(path);
      float nmin[3], nmax[3];
      aabbForKey(key, aabb.min, aabb.max, nmin, nmax);
      buildBucket(ctx, key, static_cast<int>(oconfig.splitDepth), path, cnt,
                  nmin, nmax, /*emitSample=*/true);
      fs::remove(path, ec);
    }

    // Pass 4: shallow tree from bucket samples.
    buildUpperLevels(ctx, bucketKeys);
  }

  writeMetadata(config.outputDir, meta);
  fs::remove_all(tempDir, ec);
  return meta;
}

}  // namespace pointcloud
