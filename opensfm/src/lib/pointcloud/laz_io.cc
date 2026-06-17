#include <pointcloud/laz_io.h>

#include <laszip_api.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace pointcloud {

namespace {

// Cast helpers for the opaque laszip_POINTER stored as void*.
inline laszip_POINTER LZ(void* p) { return static_cast<laszip_POINTER>(p); }

}  // namespace

// ── LAZReader ────────────────────────────────────────────────────────────────

LAZReader::LAZReader(const std::string& path) : path_(path) {
  if (laszip_create(&laszip_)) {
    throw std::runtime_error("LASzip reader: laszip_create failed");
  }
  laszip_BOOL is_compressed = 0;
  if (laszip_open_reader(LZ(laszip_), path.c_str(), &is_compressed)) {
    laszip_CHAR* e = nullptr;
    std::string err = "unknown error";
    if (laszip_get_error(LZ(laszip_), &e) == 0 && e) err = e;
    laszip_destroy(LZ(laszip_));
    laszip_ = nullptr;
    throw std::runtime_error("LASzip reader (open_reader): " + err);
  }
  opened_ = true;

  laszip_header* header = nullptr;
  if (laszip_get_header_pointer(LZ(laszip_), &header) || !header) return;

  layout_.recordLength = header->point_data_record_length;
  layout_.scale[0] = header->x_scale_factor;
  layout_.scale[1] = header->y_scale_factor;
  layout_.scale[2] = header->z_scale_factor;
  layout_.offset[0] = header->x_offset;
  layout_.offset[1] = header->y_offset;
  layout_.offset[2] = header->z_offset;
  aabbMin_[0] = header->min_x; aabbMax_[0] = header->max_x;
  aabbMin_[1] = header->min_y; aabbMax_[1] = header->max_y;
  aabbMin_[2] = header->min_z; aabbMax_[2] = header->max_z;

  count_ = header->extended_number_of_point_records
               ? header->extended_number_of_point_records
               : header->number_of_point_records;

  const uint8_t pdrf = header->point_data_format;
  attrs_.hasColors = (pdrf == 2 || pdrf == 3 || pdrf == 5 || pdrf == 7 ||
                      pdrf == 8 || pdrf == 10);
  attrs_.hasLabels = true;
  // Normals present if the record is long enough for 3×f32 extra bytes and an
  // ExtraBytes VLR (record id 4) is declared.
  bool extraBytesVlr = false;
  for (uint32_t i = 0; i < header->number_of_variable_length_records; ++i) {
    if (header->vlrs && header->vlrs[i].record_id == 4 &&
        std::strncmp(header->vlrs[i].user_id, "LASF_Spec", 9) == 0) {
      extraBytesVlr = true;
    }
  }
  if (extraBytesVlr &&
      layout_.recordLength >= las_detail::RecordLayout::kBaseSize + 12) {
    layout_.hasNormals = true;
    attrs_.hasNormals = true;
  }
  ok_ = true;
}

LAZReader::~LAZReader() {
  if (laszip_) {
    if (opened_) laszip_close_reader(LZ(laszip_));
    laszip_destroy(LZ(laszip_));
  }
}

void LAZReader::aabb(double outMin[3], double outMax[3]) const {
  for (int a = 0; a < 3; ++a) { outMin[a] = aabbMin_[a]; outMax[a] = aabbMax_[a]; }
}

void LAZReader::rewind() {
  if (ok_) {
    laszip_seek_point(LZ(laszip_), 0);
    cursor_ = 0;
  }
}

bool LAZReader::readChunk(uint64_t maxPoints, PointChunk& out) {
  out.clear();
  if (!ok_ || cursor_ >= count_ || maxPoints == 0) return false;
  const uint64_t n = std::min(maxPoints, count_ - cursor_);
  out.resize(n, attrs_);

  laszip_point* p = nullptr;
  if (laszip_get_point_pointer(LZ(laszip_), &p) || !p) return false;

  for (uint64_t k = 0; k < n; ++k) {
    if (laszip_read_point(LZ(laszip_))) return false;
    out.positions[3 * k + 0] = p->X * layout_.scale[0] + layout_.offset[0];
    out.positions[3 * k + 1] = p->Y * layout_.scale[1] + layout_.offset[1];
    out.positions[3 * k + 2] = p->Z * layout_.scale[2] + layout_.offset[2];
    if (attrs_.hasColors) {
      out.colors[3 * k + 0] = static_cast<uint8_t>(p->rgb[0] >> 8);
      out.colors[3 * k + 1] = static_cast<uint8_t>(p->rgb[1] >> 8);
      out.colors[3 * k + 2] = static_cast<uint8_t>(p->rgb[2] >> 8);
    }
    if (attrs_.hasLabels) {
      // PDRF 6-10 carry the class in extended_classification.
      uint8_t cls = p->extended_classification
                        ? p->extended_classification
                        : p->classification;
      out.labels[k] = cls;
    }
    if (attrs_.hasNormals && p->extra_bytes && p->num_extra_bytes >= 12) {
      float nrm[3];
      std::memcpy(nrm, p->extra_bytes, 12);
      out.normals[3 * k + 0] = nrm[0];
      out.normals[3 * k + 1] = nrm[1];
      out.normals[3 * k + 2] = nrm[2];
    }
  }
  cursor_ += n;
  out.count = n;
  return true;
}

// ── LAZWriter ────────────────────────────────────────────────────────────────

LAZWriter::LAZWriter(const std::string& path, const PointCloudHeader& header) {
  using namespace las_detail;

  // On any laszip failure, surface the real error (instead of a generic
  // "cannot open") by throwing — the factory propagates it to Python.
  auto fail = [&](const char* what) {
    std::string err = "unknown error";
    if (laszip_) {
      laszip_CHAR* e = nullptr;
      if (laszip_get_error(LZ(laszip_), &e) == 0 && e) err = e;
      laszip_destroy(LZ(laszip_));
      laszip_ = nullptr;
    }
    throw std::runtime_error(std::string("LASzip writer (") + what + "): " + err);
  };

  if (laszip_create(&laszip_)) {
    throw std::runtime_error("LASzip writer: laszip_create failed");
  }

  layout_.hasNormals = header.attrs.hasNormals;
  layout_.recordLength = static_cast<uint16_t>(
      RecordLayout::kBaseSize + (layout_.hasNormals ? 12 : 0));
  for (int a = 0; a < 3; ++a) {
    layout_.scale[a] = (header.scale[a] > 0.0) ? header.scale[a] : 0.001;
    layout_.offset[a] = header.hasAabb ? header.aabbMin[a] : 0.0;
  }

  laszip_header* h = nullptr;
  if (laszip_get_header_pointer(LZ(laszip_), &h) || !h) fail("get_header_pointer");
  h->version_major = 1;
  h->version_minor = 4;
  h->global_encoding = 1;  // GPS standard time
  h->header_size = 375;
  // laszip_create defaults the header to LAS 1.2 (offset_to_point_data = 227).
  // It never re-derives the base offset from header_size; laszip_add_vlr only
  // *increments* it.  So seed it with our 1.4 header size, and each add_vlr
  // below brings it to header_size + Σ VLR sizes (what open_writer validates).
  h->offset_to_point_data = 375;
  h->point_data_format = 7;
  h->point_data_record_length = layout_.recordLength;
  h->x_scale_factor = layout_.scale[0];
  h->y_scale_factor = layout_.scale[1];
  h->z_scale_factor = layout_.scale[2];
  h->x_offset = layout_.offset[0];
  h->y_offset = layout_.offset[1];
  h->z_offset = layout_.offset[2];
  // LAS 1.4: for point formats 6-10 the LEGACY counts MUST be zero; only the
  // extended (64-bit) counts are used.  laszip enforces this on open.
  h->number_of_point_records = 0;
  h->extended_number_of_point_records = header.pointCount;
  h->extended_number_of_points_by_return[0] = header.pointCount;

  // Normals → Extra Bytes VLR (descriptor payload only; laszip writes the VLR
  // header). Reuse the LAS descriptor builder (skip its 54-byte VLR header).
  if (layout_.hasNormals) {
    std::vector<uint8_t> vlr = buildNormalExtraBytesVlr();
    const uint8_t* descr = vlr.data() + kVlrHeaderSize;
    laszip_U16 descrLen = static_cast<laszip_U16>(vlr.size() - kVlrHeaderSize);
    if (laszip_add_vlr(LZ(laszip_), "LASF_Spec", 4, descrLen, "normals", descr))
      fail("add_vlr");
  }

  if (laszip_open_writer(LZ(laszip_), path.c_str(), /*compress=*/1))
    fail("open_writer");
  opened_ = true;
  ok_ = true;
}

LAZWriter::~LAZWriter() {
  if (laszip_) {
    if (opened_) laszip_close_writer(LZ(laszip_));
    laszip_destroy(LZ(laszip_));
  }
}

bool LAZWriter::writeChunk(const PointChunk& chunk) {
  if (!ok_) return false;
  const uint64_t n = chunk.count;
  if (n == 0) return true;
  const bool hasN = !chunk.normals.empty();
  const bool hasC = !chunk.colors.empty();
  const bool hasL = !chunk.labels.empty();

  laszip_point* p = nullptr;
  if (laszip_get_point_pointer(LZ(laszip_), &p) || !p) return false;

  auto toInt = [](double world, double off, double scale) -> laszip_I32 {
    double s = (scale != 0.0) ? scale : 1.0;
    double v = (world - off) / s;
    v = std::max(-2147483648.0, std::min(2147483647.0, v));
    return static_cast<laszip_I32>(v < 0 ? v - 0.5 : v + 0.5);
  };

  for (uint64_t k = 0; k < n; ++k) {
    p->X = toInt(chunk.positions[3 * k + 0], layout_.offset[0], layout_.scale[0]);
    p->Y = toInt(chunk.positions[3 * k + 1], layout_.offset[1], layout_.scale[1]);
    p->Z = toInt(chunk.positions[3 * k + 2], layout_.offset[2], layout_.scale[2]);
    p->intensity = 0;
    uint8_t lbl = hasL ? chunk.labels[k] : 0;
    p->classification = lbl;            // legacy field
    p->extended_classification = lbl;   // PDRF 6-10 field
    if (hasC) {
      p->rgb[0] = static_cast<laszip_U16>(chunk.colors[3 * k + 0]) << 8;
      p->rgb[1] = static_cast<laszip_U16>(chunk.colors[3 * k + 1]) << 8;
      p->rgb[2] = static_cast<laszip_U16>(chunk.colors[3 * k + 2]) << 8;
    } else {
      p->rgb[0] = p->rgb[1] = p->rgb[2] = 0;
    }
    if (layout_.hasNormals && p->extra_bytes && p->num_extra_bytes >= 12) {
      float nrm[3] = {hasN ? chunk.normals[3 * k + 0] : 0.0f,
                      hasN ? chunk.normals[3 * k + 1] : 0.0f,
                      hasN ? chunk.normals[3 * k + 2] : 0.0f};
      std::memcpy(p->extra_bytes, nrm, 12);
    }
    if (laszip_write_point(LZ(laszip_))) return false;
  }
  written_ += n;
  return true;
}

bool LAZWriter::finalize() {
  if (!ok_) return false;
  bool good = (laszip_close_writer(LZ(laszip_)) == 0);
  opened_ = false;
  return good;
}

}  // namespace pointcloud
