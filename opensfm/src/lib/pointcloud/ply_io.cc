#include <pointcloud/ply_io.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <sstream>
#include <string>
#include <system_error>

namespace pointcloud {

namespace {

// Element type codes for PLY scalar properties.
enum TypeCode { F32 = 0, F64 = 1, U8 = 2, I8 = 3, U16 = 4, I16 = 5, U32 = 6, I32 = 7 };

int typeCodeOf(const std::string& t) {
  if (t == "float" || t == "float32") return F32;
  if (t == "double" || t == "float64") return F64;
  if (t == "uchar" || t == "uint8") return U8;
  if (t == "char" || t == "int8") return I8;
  if (t == "ushort" || t == "uint16") return U16;
  if (t == "short" || t == "int16") return I16;
  if (t == "uint" || t == "uint32") return U32;
  if (t == "int" || t == "int32") return I32;
  return -1;
}

int typeSize(int code) {
  switch (code) {
    case F32: case U32: case I32: return 4;
    case F64: return 8;
    case U8: case I8: return 1;
    case U16: case I16: return 2;
    default: return 0;
  }
}

// Read a little-endian scalar of `code` as double (host is little-endian on
// all supported platforms: x86_64 / arm64).
double readDoubleLE(const uint8_t* p, int code) {
  switch (code) {
    case F32: { float v; std::memcpy(&v, p, 4); return static_cast<double>(v); }
    case F64: { double v; std::memcpy(&v, p, 8); return v; }
    case U8:  return static_cast<double>(*p);
    case I8:  return static_cast<double>(static_cast<int8_t>(*p));
    case U16: { uint16_t v; std::memcpy(&v, p, 2); return v; }
    case I16: { int16_t v; std::memcpy(&v, p, 2); return v; }
    case U32: { uint32_t v; std::memcpy(&v, p, 4); return v; }
    case I32: { int32_t v; std::memcpy(&v, p, 4); return v; }
    default: return 0.0;
  }
}

uint32_t readU32LE(const uint8_t* p, int code) {
  return static_cast<uint32_t>(readDoubleLE(p, code));
}

// Field roles a property name can map to.
enum class Field { None, PosX, PosY, PosZ, NrmX, NrmY, NrmZ, ColR, ColG, ColB, Label };

Field classifyField(const std::string& n) {
  if (n == "x") return Field::PosX;
  if (n == "y") return Field::PosY;
  if (n == "z") return Field::PosZ;
  if (n == "nx") return Field::NrmX;
  if (n == "ny") return Field::NrmY;
  if (n == "nz") return Field::NrmZ;
  if (n == "red" || n == "r" || n == "diffuse_red") return Field::ColR;
  if (n == "green" || n == "g" || n == "diffuse_green") return Field::ColG;
  if (n == "blue" || n == "b" || n == "diffuse_blue") return Field::ColB;
  if (n == "class" || n == "classification" || n == "label" ||
      n == "scalar_Classification")
    return Field::Label;
  return Field::None;
}

}  // namespace

// ── PLYReader ────────────────────────────────────────────────────────────────

PLYReader::PLYReader(const std::string& path) {
  std::error_code ec;
  map_ = mio::make_mmap_source(path, ec);
  if (ec || map_.size() == 0) {
    ok_ = false;
    return;
  }
  ok_ = parseHeader();
}

bool PLYReader::parseHeader() {
  const char* data = map_.data();
  const uint64_t size = map_.size();

  // Read ASCII header lines until "end_header".
  uint64_t pos = 0;
  auto readLine = [&](std::string& line) -> bool {
    if (pos >= size) return false;
    uint64_t start = pos;
    while (pos < size && data[pos] != '\n') ++pos;
    uint64_t end = pos;
    if (end > start && data[end - 1] == '\r') --end;  // tolerate CRLF
    line.assign(data + start, data + end);
    if (pos < size) ++pos;  // consume '\n'
    return true;
  };

  std::string line;
  if (!readLine(line) || line.substr(0, 3) != "ply") return false;

  uint64_t recordOffset = 0;
  while (readLine(line)) {
    std::istringstream ss(line);
    std::string tok;
    ss >> tok;
    if (tok == "format") {
      std::string fmt;
      ss >> fmt;
      if (fmt == "ascii") format_ = Format::Ascii;
      else if (fmt == "binary_little_endian") format_ = Format::BinaryLE;
      else if (fmt == "binary_big_endian") format_ = Format::BinaryBE;
      else return false;
    } else if (tok == "element") {
      std::string what; uint64_t n = 0;
      ss >> what >> n;
      if (what == "vertex") count_ = n;
      // Other elements (e.g. "face") are ignored; their properties would
      // follow, but dense clouds have only a vertex element.
    } else if (tok == "property") {
      std::string t1;
      ss >> t1;
      if (t1 == "list") {
        // Face lists etc. — unsupported here; bail out cleanly.
        return false;
      }
      std::string name;
      ss >> name;
      int code = typeCodeOf(t1);
      if (code < 0) return false;
      Property prop;
      prop.name = name;
      prop.typeCode = code;
      prop.size = typeSize(code);
      prop.offset = recordOffset;
      properties_.push_back(prop);
      recordOffset += prop.size;
    } else if (tok == "end_header") {
      break;
    }
    // comment / obj_info lines ignored.
  }

  if (format_ == Format::BinaryBE) return false;  // unsupported (rare)

  dataOffset_ = pos;
  recordStride_ = static_cast<uint32_t>(recordOffset);

  // Resolve field offsets.
  for (const auto& p : properties_) {
    switch (classifyField(p.name)) {
      case Field::PosX: posOff_[0] = static_cast<int>(p.offset); posType_ = (p.typeCode == F64) ? 1 : 0; break;
      case Field::PosY: posOff_[1] = static_cast<int>(p.offset); break;
      case Field::PosZ: posOff_[2] = static_cast<int>(p.offset); break;
      case Field::NrmX: nrmOff_[0] = static_cast<int>(p.offset); nrmType_ = (p.typeCode == F64) ? 1 : 0; break;
      case Field::NrmY: nrmOff_[1] = static_cast<int>(p.offset); break;
      case Field::NrmZ: nrmOff_[2] = static_cast<int>(p.offset); break;
      case Field::ColR: colOff_[0] = static_cast<int>(p.offset); break;
      case Field::ColG: colOff_[1] = static_cast<int>(p.offset); break;
      case Field::ColB: colOff_[2] = static_cast<int>(p.offset); break;
      case Field::Label: lblOff_ = static_cast<int>(p.offset); lblType_ = p.typeCode; break;
      case Field::None: break;
    }
  }

  if (posOff_[0] < 0 || posOff_[1] < 0 || posOff_[2] < 0) return false;
  attrs_.hasNormals = (nrmOff_[0] >= 0 && nrmOff_[1] >= 0 && nrmOff_[2] >= 0);
  attrs_.hasColors = (colOff_[0] >= 0 && colOff_[1] >= 0 && colOff_[2] >= 0);
  attrs_.hasLabels = (lblOff_ >= 0);

  asciiByte_ = dataOffset_;
  return true;
}

void PLYReader::aabb(double outMin[3], double outMax[3]) const {
  for (int a = 0; a < 3; ++a) { outMin[a] = 0.0; outMax[a] = 0.0; }
}

void PLYReader::decodeRecord(const uint8_t* rec, double pos[3], float nrm[3],
                             uint8_t col[3], uint8_t& lbl) const {
  int psize = (posType_ == 1) ? 8 : 4;
  int pcode = (posType_ == 1) ? F64 : F32;
  for (int a = 0; a < 3; ++a)
    pos[a] = readDoubleLE(rec + posOff_[a], pcode);
  (void)psize;
  if (attrs_.hasNormals) {
    int ncode = (nrmType_ == 1) ? F64 : F32;
    for (int a = 0; a < 3; ++a)
      nrm[a] = static_cast<float>(readDoubleLE(rec + nrmOff_[a], ncode));
  }
  if (attrs_.hasColors) {
    for (int a = 0; a < 3; ++a)
      col[a] = static_cast<uint8_t>(readU32LE(rec + colOff_[a], U8));
  }
  if (attrs_.hasLabels) {
    lbl = static_cast<uint8_t>(readU32LE(rec + lblOff_, lblType_));
  }
}

bool PLYReader::readChunk(uint64_t maxPoints, PointChunk& out) {
  out.clear();
  if (!ok_ || cursor_ >= count_ || maxPoints == 0) return false;

  const uint64_t n = std::min(maxPoints, count_ - cursor_);
  out.resize(n, attrs_);

  const uint8_t* base = reinterpret_cast<const uint8_t*>(map_.data());

  if (format_ == Format::BinaryLE) {
    const uint8_t* body = base + dataOffset_;
    for (uint64_t k = 0; k < n; ++k) {
      const uint8_t* rec = body + (cursor_ + k) * recordStride_;
      double pos[3]; float nrm[3]{0, 0, 0}; uint8_t col[3]{0, 0, 0}; uint8_t lbl = 0;
      decodeRecord(rec, pos, nrm, col, lbl);
      out.positions[3 * k + 0] = pos[0];
      out.positions[3 * k + 1] = pos[1];
      out.positions[3 * k + 2] = pos[2];
      if (attrs_.hasNormals) { out.normals[3 * k + 0] = nrm[0]; out.normals[3 * k + 1] = nrm[1]; out.normals[3 * k + 2] = nrm[2]; }
      if (attrs_.hasColors) { out.colors[3 * k + 0] = col[0]; out.colors[3 * k + 1] = col[1]; out.colors[3 * k + 2] = col[2]; }
      if (attrs_.hasLabels) out.labels[k] = lbl;
    }
  } else {
    // ASCII: parse `n` whitespace-separated rows from the body, routing each
    // column to its field by declaration order.
    const uint64_t size = map_.size();
    uint64_t bp = asciiByte_;
    for (uint64_t k = 0; k < n; ++k) {
      double pos[3]{0, 0, 0}; float nrm[3]{0, 0, 0}; uint8_t col[3]{0, 0, 0}; uint8_t lbl = 0;
      for (const auto& prop : properties_) {
        // Skip whitespace.
        while (bp < size && std::isspace(static_cast<unsigned char>(map_.data()[bp]))) ++bp;
        uint64_t s = bp;
        while (bp < size && !std::isspace(static_cast<unsigned char>(map_.data()[bp]))) ++bp;
        std::string tok(map_.data() + s, map_.data() + bp);
        double val = tok.empty() ? 0.0 : std::strtod(tok.c_str(), nullptr);
        switch (classifyField(prop.name)) {
          case Field::PosX: pos[0] = val; break;
          case Field::PosY: pos[1] = val; break;
          case Field::PosZ: pos[2] = val; break;
          case Field::NrmX: nrm[0] = static_cast<float>(val); break;
          case Field::NrmY: nrm[1] = static_cast<float>(val); break;
          case Field::NrmZ: nrm[2] = static_cast<float>(val); break;
          case Field::ColR: col[0] = static_cast<uint8_t>(val); break;
          case Field::ColG: col[1] = static_cast<uint8_t>(val); break;
          case Field::ColB: col[2] = static_cast<uint8_t>(val); break;
          case Field::Label: lbl = static_cast<uint8_t>(val); break;
          case Field::None: break;
        }
      }
      out.positions[3 * k + 0] = pos[0]; out.positions[3 * k + 1] = pos[1]; out.positions[3 * k + 2] = pos[2];
      if (attrs_.hasNormals) { out.normals[3 * k + 0] = nrm[0]; out.normals[3 * k + 1] = nrm[1]; out.normals[3 * k + 2] = nrm[2]; }
      if (attrs_.hasColors) { out.colors[3 * k + 0] = col[0]; out.colors[3 * k + 1] = col[1]; out.colors[3 * k + 2] = col[2]; }
      if (attrs_.hasLabels) out.labels[k] = lbl;
    }
    asciiByte_ = bp;
  }

  cursor_ += n;
  out.count = n;
  return true;
}

PLYReader::MappedBody PLYReader::mappedBody() {
  MappedBody mb;
  // Random access only for binary little-endian with contiguous same-type
  // xyz (the canonical dense layout). Otherwise leave invalid.
  if (!ok_ || format_ != Format::BinaryLE) return mb;
  int psize = (posType_ == 1) ? 8 : 4;
  bool posContig = (posOff_[1] == posOff_[0] + psize) &&
                   (posOff_[2] == posOff_[0] + 2 * psize);
  if (!posContig) return mb;

  mb.base = reinterpret_cast<const uint8_t*>(map_.data());
  mb.bodyOffset = dataOffset_;
  mb.recordStride = recordStride_;
  mb.count = count_;
  mb.posOffset = posOff_[0];
  mb.posType = posType_;
  // Normals exposed only if contiguous f32 (what MmapPointSource expects).
  if (attrs_.hasNormals && nrmType_ == 0 &&
      nrmOff_[1] == nrmOff_[0] + 4 && nrmOff_[2] == nrmOff_[0] + 8) {
    mb.nrmOffset = nrmOff_[0];
  }
  if (attrs_.hasColors && colOff_[1] == colOff_[0] + 1 &&
      colOff_[2] == colOff_[0] + 2) {
    mb.colOffset = colOff_[0];
  }
  if (attrs_.hasLabels && lblType_ == U8) {
    mb.lblOffset = lblOff_;
  }
  return mb;
}

// ── PLYWriter ────────────────────────────────────────────────────────────────

// Fixed-width vertex-count placeholder so the count can be patched on finalize.
namespace {
constexpr int kCountDigits = 20;
}

PLYWriter::PLYWriter(const std::string& path, const PointCloudHeader& header)
    : out_(path, std::ios::binary), path_(path) {
  (void)header;  // canonical layout is fixed; attrs default to zeros.
  if (out_.good()) writeHeaderPlaceholder();
}

void PLYWriter::writeHeaderPlaceholder() {
  out_ << "ply\n"
       << "format binary_little_endian 1.0\n"
       << "element vertex ";
  countPos_ = out_.tellp();
  for (int i = 0; i < kCountDigits; ++i) out_ << '0';
  out_ << "\n"
       << "property float x\n"
       << "property float y\n"
       << "property float z\n"
       << "property float nx\n"
       << "property float ny\n"
       << "property float nz\n"
       << "property uchar red\n"
       << "property uchar green\n"
       << "property uchar blue\n"
       << "property uchar class\n"
       << "end_header\n";
  headerWritten_ = true;
}

bool PLYWriter::writeChunk(const PointChunk& chunk) {
  if (!out_.good() || !headerWritten_) return false;
  const uint64_t n = chunk.count;
  if (n == 0) return true;

  const bool hasN = !chunk.normals.empty();
  const bool hasC = !chunk.colors.empty();
  const bool hasL = !chunk.labels.empty();

  std::vector<uint8_t> buf(n * 28);
  for (uint64_t k = 0; k < n; ++k) {
    uint8_t* rec = buf.data() + k * 28;
    float xyz[3] = {static_cast<float>(chunk.positions[3 * k + 0]),
                    static_cast<float>(chunk.positions[3 * k + 1]),
                    static_cast<float>(chunk.positions[3 * k + 2])};
    float nrm[3] = {hasN ? chunk.normals[3 * k + 0] : 0.0f,
                    hasN ? chunk.normals[3 * k + 1] : 0.0f,
                    hasN ? chunk.normals[3 * k + 2] : 0.0f};
    std::memcpy(rec + 0, xyz, 12);
    std::memcpy(rec + 12, nrm, 12);
    rec[24] = hasC ? chunk.colors[3 * k + 0] : 0;
    rec[25] = hasC ? chunk.colors[3 * k + 1] : 0;
    rec[26] = hasC ? chunk.colors[3 * k + 2] : 0;
    rec[27] = hasL ? chunk.labels[k] : 0;
  }
  out_.write(reinterpret_cast<const char*>(buf.data()),
             static_cast<std::streamsize>(buf.size()));
  written_ += n;
  return out_.good();
}

bool PLYWriter::finalize() {
  if (!out_.good()) return false;
  // Patch the vertex count (zero-padded, fixed width).
  std::ostringstream ss;
  ss.width(kCountDigits);
  ss.fill('0');
  ss << written_;
  std::string digits = ss.str();
  out_.seekp(countPos_);
  out_.write(digits.data(), static_cast<std::streamsize>(digits.size()));
  out_.flush();
  bool good = out_.good();
  out_.close();
  return good;
}

}  // namespace pointcloud
