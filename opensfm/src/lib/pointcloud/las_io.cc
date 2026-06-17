#include <pointcloud/las_io.h>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace pointcloud {
namespace las_detail {

namespace {

// Little-endian scalar put/get (host is LE on all supported platforms).
template <typename T>
void put(uint8_t* p, T v) {
  std::memcpy(p, &v, sizeof(T));
}
template <typename T>
T get(const uint8_t* p) {
  T v;
  std::memcpy(&v, p, sizeof(T));
  return v;
}

int32_t toIntCoord(double world, double offset, double scale) {
  double s = (scale != 0.0) ? scale : 1.0;
  double v = std::llround((world - offset) / s);
  if (v > 2147483647.0) v = 2147483647.0;
  if (v < -2147483648.0) v = -2147483648.0;
  return static_cast<int32_t>(v);
}

}  // namespace

void encodeRecord(const RecordLayout& L, const double pos[3], const float nrm[3],
                  const uint8_t col[3], uint8_t label, uint8_t* dst) {
  std::memset(dst, 0, L.recordLength);
  put<int32_t>(dst + RecordLayout::kX, toIntCoord(pos[0], L.offset[0], L.scale[0]));
  put<int32_t>(dst + RecordLayout::kY, toIntCoord(pos[1], L.offset[1], L.scale[1]));
  put<int32_t>(dst + RecordLayout::kZ, toIntCoord(pos[2], L.offset[2], L.scale[2]));
  // byte 14: return number (1) | number of returns (1) << 4.
  dst[14] = 0x11;
  dst[RecordLayout::kClass] = label;
  // GPS time (offset 22, double) left 0.
  // RGB stored 16-bit, left-justified (value << 8) — the common convention.
  put<uint16_t>(dst + RecordLayout::kRed, static_cast<uint16_t>(col[0]) << 8);
  put<uint16_t>(dst + RecordLayout::kGreen, static_cast<uint16_t>(col[1]) << 8);
  put<uint16_t>(dst + RecordLayout::kBlue, static_cast<uint16_t>(col[2]) << 8);
  if (L.hasNormals) {
    put<float>(dst + RecordLayout::kNormalOffset + 0, nrm[0]);
    put<float>(dst + RecordLayout::kNormalOffset + 4, nrm[1]);
    put<float>(dst + RecordLayout::kNormalOffset + 8, nrm[2]);
  }
}

void decodeRecord(const RecordLayout& L, const uint8_t* src, double pos[3],
                  float nrm[3], uint8_t col[3], uint8_t& label) {
  pos[0] = get<int32_t>(src + RecordLayout::kX) * L.scale[0] + L.offset[0];
  pos[1] = get<int32_t>(src + RecordLayout::kY) * L.scale[1] + L.offset[1];
  pos[2] = get<int32_t>(src + RecordLayout::kZ) * L.scale[2] + L.offset[2];
  label = src[RecordLayout::kClass];
  col[0] = static_cast<uint8_t>(get<uint16_t>(src + RecordLayout::kRed) >> 8);
  col[1] = static_cast<uint8_t>(get<uint16_t>(src + RecordLayout::kGreen) >> 8);
  col[2] = static_cast<uint8_t>(get<uint16_t>(src + RecordLayout::kBlue) >> 8);
  if (L.hasNormals) {
    nrm[0] = get<float>(src + RecordLayout::kNormalOffset + 0);
    nrm[1] = get<float>(src + RecordLayout::kNormalOffset + 4);
    nrm[2] = get<float>(src + RecordLayout::kNormalOffset + 8);
  } else {
    nrm[0] = nrm[1] = nrm[2] = 0.0f;
  }
}

std::vector<uint8_t> buildPublicHeader(const RecordLayout& L, uint32_t numVlrs,
                                       uint32_t offsetToPointData) {
  std::vector<uint8_t> h(kHeaderSize, 0);
  std::memcpy(h.data() + 0, "LASF", 4);
  put<uint16_t>(h.data() + 6, 1);  // global encoding: GPS standard time
  h[24] = 1;                       // version major
  h[25] = 4;                       // version minor
  const char* sw = "OpenSfM-Desktop pointcloud";
  std::memcpy(h.data() + 58, sw, std::strlen(sw));
  put<uint16_t>(h.data() + 94, static_cast<uint16_t>(kHeaderSize));
  put<uint32_t>(h.data() + 96, offsetToPointData);
  put<uint32_t>(h.data() + 100, numVlrs);
  h[104] = 7;  // point data record format
  put<uint16_t>(h.data() + 105, L.recordLength);
  // Legacy point count (107) and 1.4 point count (247) patched on finalize.
  for (int a = 0; a < 3; ++a) {
    put<double>(h.data() + 131 + a * 8, L.scale[a]);
    put<double>(h.data() + 155 + a * 8, L.offset[a]);
  }
  // Min/max (179..227) patched on finalize.
  return h;
}

std::vector<uint8_t> buildNormalExtraBytesVlr() {
  const int payload = 3 * kExtraByteDescrSize;  // nx, ny, nz
  std::vector<uint8_t> v(kVlrHeaderSize + payload, 0);
  // VLR header.
  std::memcpy(v.data() + 2, "LASF_Spec", 9);          // user id
  put<uint16_t>(v.data() + 18, 4);                    // record id = ExtraBytes
  put<uint16_t>(v.data() + 20, static_cast<uint16_t>(payload));
  const char* desc = "normals";
  std::memcpy(v.data() + 22, desc, std::strlen(desc));
  // Three LASattribute descriptors (192 bytes each).
  const char* names[3] = {"nx", "ny", "nz"};
  for (int i = 0; i < 3; ++i) {
    uint8_t* a = v.data() + kVlrHeaderSize + i * kExtraByteDescrSize;
    a[2] = 9;  // data_type = float (f32)
    a[3] = 0;  // options = 0 (raw value, no scale/offset/no_data)
    std::memcpy(a + 4, names[i], std::strlen(names[i]));  // name[32] at +4
    // description[32] at +160.
    std::memcpy(a + 160, names[i], std::strlen(names[i]));
  }
  return v;
}

}  // namespace las_detail

// ── LASReader ────────────────────────────────────────────────────────────────

LASReader::LASReader(const std::string& path)
    : in_(path, std::ios::binary) {
  using namespace las_detail;
  if (!in_) return;

  uint8_t h[kHeaderSize];
  in_.read(reinterpret_cast<char*>(h), kHeaderSize);
  if (!in_ || std::memcmp(h, "LASF", 4) != 0) return;

  auto getU16 = [&](int o) { uint16_t v; std::memcpy(&v, h + o, 2); return v; };
  auto getU32 = [&](int o) { uint32_t v; std::memcpy(&v, h + o, 4); return v; };
  auto getU64 = [&](int o) { uint64_t v; std::memcpy(&v, h + o, 8); return v; };
  auto getF64 = [&](int o) { double v; std::memcpy(&v, h + o, 8); return v; };

  uint8_t pdrf = h[104];
  layout_.recordLength = getU16(105);
  for (int a = 0; a < 3; ++a) {
    layout_.scale[a] = getF64(131 + a * 8);
    layout_.offset[a] = getF64(155 + a * 8);
  }
  aabbMax_[0] = getF64(179); aabbMin_[0] = getF64(187);
  aabbMax_[1] = getF64(195); aabbMin_[1] = getF64(203);
  aabbMax_[2] = getF64(211); aabbMin_[2] = getF64(219);

  pointDataOffset_ = getU32(96);
  uint32_t numVlrs = getU32(100);
  count_ = getU64(247);
  if (count_ == 0) count_ = getU32(107);  // fall back to legacy count

  // RGB present for PDRF 2,3,5,7,8,10; classification present for all here.
  attrs_.hasColors = (pdrf == 2 || pdrf == 3 || pdrf == 5 || pdrf == 7 ||
                      pdrf == 8 || pdrf == 10);
  attrs_.hasLabels = true;

  // Walk VLRs to detect the normals Extra Bytes record.
  uint64_t pos = kHeaderSize;
  for (uint32_t i = 0; i < numVlrs; ++i) {
    in_.seekg(static_cast<std::streamoff>(pos));
    uint8_t vh[kVlrHeaderSize];
    in_.read(reinterpret_cast<char*>(vh), kVlrHeaderSize);
    if (!in_) break;
    char userId[17] = {0};
    std::memcpy(userId, vh + 2, 16);
    uint16_t recordId = 0; std::memcpy(&recordId, vh + 18, 2);
    uint16_t recLen = 0; std::memcpy(&recLen, vh + 20, 2);
    if (std::string(userId) == "LASF_Spec" && recordId == 4) {
      // ExtraBytes; if it declares >=3 dims and the record is long enough to
      // carry 3 f32 after the base, treat them as nx/ny/nz.
      int nDescr = recLen / kExtraByteDescrSize;
      if (nDescr >= 3 &&
          layout_.recordLength >= RecordLayout::kBaseSize + 12) {
        layout_.hasNormals = true;
        attrs_.hasNormals = true;
      }
    }
    pos += kVlrHeaderSize + recLen;
  }

  in_.seekg(static_cast<std::streamoff>(pointDataOffset_));
  cursor_ = 0;
  ok_ = true;
}

void LASReader::aabb(double outMin[3], double outMax[3]) const {
  for (int a = 0; a < 3; ++a) { outMin[a] = aabbMin_[a]; outMax[a] = aabbMax_[a]; }
}

void LASReader::rewind() {
  in_.clear();
  in_.seekg(static_cast<std::streamoff>(pointDataOffset_));
  cursor_ = 0;
}

bool LASReader::readChunk(uint64_t maxPoints, PointChunk& out) {
  out.clear();
  if (!ok_ || cursor_ >= count_ || maxPoints == 0) return false;
  const uint64_t n = std::min(maxPoints, count_ - cursor_);
  out.resize(n, attrs_);

  const uint32_t stride = layout_.recordLength;
  std::vector<uint8_t> buf(static_cast<size_t>(n) * stride);
  in_.read(reinterpret_cast<char*>(buf.data()),
           static_cast<std::streamsize>(buf.size()));
  if (!in_) return false;

  for (uint64_t k = 0; k < n; ++k) {
    double pos[3]; float nrm[3]{0, 0, 0}; uint8_t col[3]{0, 0, 0}; uint8_t lbl = 0;
    las_detail::decodeRecord(layout_, buf.data() + k * stride, pos, nrm, col, lbl);
    out.positions[3 * k + 0] = pos[0];
    out.positions[3 * k + 1] = pos[1];
    out.positions[3 * k + 2] = pos[2];
    if (attrs_.hasNormals) { out.normals[3 * k] = nrm[0]; out.normals[3 * k + 1] = nrm[1]; out.normals[3 * k + 2] = nrm[2]; }
    if (attrs_.hasColors) { out.colors[3 * k] = col[0]; out.colors[3 * k + 1] = col[1]; out.colors[3 * k + 2] = col[2]; }
    if (attrs_.hasLabels) out.labels[k] = lbl;
  }
  cursor_ += n;
  out.count = n;
  return true;
}

// ── LASWriter ────────────────────────────────────────────────────────────────

LASWriter::LASWriter(const std::string& path, const PointCloudHeader& header)
    : out_(path, std::ios::binary) {
  using namespace las_detail;
  if (!out_) return;

  layout_.hasNormals = header.attrs.hasNormals;
  layout_.recordLength = static_cast<uint16_t>(
      RecordLayout::kBaseSize + (layout_.hasNormals ? 12 : 0));
  for (int a = 0; a < 3; ++a) {
    layout_.scale[a] = (header.scale[a] > 0.0) ? header.scale[a] : 0.001;
    // Offset near the data when a bbox is known; else 0 (fine for local
    // topocentric coordinates: ±2.1e6 m at 1 mm scale).
    layout_.offset[a] = header.hasAabb ? std::floor(header.aabbMin[a]) : 0.0;
  }

  numVlrs_ = layout_.hasNormals ? 1u : 0u;
  uint32_t vlrBytes =
      layout_.hasNormals ? (kVlrHeaderSize + 3 * kExtraByteDescrSize) : 0u;
  pointDataOffset_ = kHeaderSize + vlrBytes;

  auto h = buildPublicHeader(layout_, numVlrs_, pointDataOffset_);
  out_.write(reinterpret_cast<const char*>(h.data()),
             static_cast<std::streamsize>(h.size()));
  if (layout_.hasNormals) {
    auto vlr = buildNormalExtraBytesVlr();
    out_.write(reinterpret_cast<const char*>(vlr.data()),
               static_cast<std::streamsize>(vlr.size()));
  }
  headerOk_ = out_.good();
}

bool LASWriter::writeChunk(const PointChunk& chunk) {
  if (!out_.good() || !headerOk_) return false;
  const uint64_t n = chunk.count;
  if (n == 0) return true;

  const bool hasN = !chunk.normals.empty();
  const bool hasC = !chunk.colors.empty();
  const bool hasL = !chunk.labels.empty();
  const uint32_t stride = layout_.recordLength;

  std::vector<uint8_t> buf(static_cast<size_t>(n) * stride);
  for (uint64_t k = 0; k < n; ++k) {
    double pos[3] = {chunk.positions[3 * k], chunk.positions[3 * k + 1],
                     chunk.positions[3 * k + 2]};
    float nrm[3] = {hasN ? chunk.normals[3 * k] : 0.0f,
                    hasN ? chunk.normals[3 * k + 1] : 0.0f,
                    hasN ? chunk.normals[3 * k + 2] : 0.0f};
    uint8_t col[3] = {hasC ? chunk.colors[3 * k] : (uint8_t)0,
                      hasC ? chunk.colors[3 * k + 1] : (uint8_t)0,
                      hasC ? chunk.colors[3 * k + 2] : (uint8_t)0};
    uint8_t lbl = hasL ? chunk.labels[k] : 0;
    las_detail::encodeRecord(layout_, pos, nrm, col, lbl, buf.data() + k * stride);
    for (int a = 0; a < 3; ++a) {
      aabbMin_[a] = std::min(aabbMin_[a], pos[a]);
      aabbMax_[a] = std::max(aabbMax_[a], pos[a]);
    }
  }
  out_.write(reinterpret_cast<const char*>(buf.data()),
             static_cast<std::streamsize>(buf.size()));
  written_ += n;
  return out_.good();
}

bool LASWriter::finalize() {
  if (!out_.good()) return false;
  // Patch legacy + 1.4 point counts and the bbox min/max.
  auto putAt = [&](std::streamoff off, const void* data, size_t bytes) {
    out_.seekp(off);
    out_.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
  };
  // LAS 1.4: legacy 32-bit count must be 0 for point formats 6-10 (we use
  // PDRF 7); the extended 64-bit count is authoritative.
  uint32_t legacy = 0;
  putAt(107, &legacy, 4);
  putAt(247, &written_, 8);
  if (written_ > 0) {
    double maxx = aabbMax_[0], minx = aabbMin_[0], maxy = aabbMax_[1],
           miny = aabbMin_[1], maxz = aabbMax_[2], minz = aabbMin_[2];
    putAt(179, &maxx, 8); putAt(187, &minx, 8);
    putAt(195, &maxy, 8); putAt(203, &miny, 8);
    putAt(211, &maxz, 8); putAt(219, &minz, 8);
  }
  out_.flush();
  bool good = out_.good();
  out_.close();
  return good;
}

}  // namespace pointcloud
