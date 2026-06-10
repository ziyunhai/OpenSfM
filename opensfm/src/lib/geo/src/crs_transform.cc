#include <geo/crs_transform.h>
#include <proj.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace geo {

// ── Parsing helpers ──────────────────────────────────────────────────────────

static std::string trim(const std::string& s) {
  auto start = s.find_first_not_of(" \t\r\n");
  if (start == std::string::npos) {
    return {};
  }
  auto end = s.find_last_not_of(" \t\r\n");
  return s.substr(start, end - start + 1);
}

static std::string toLower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return s;
}

static std::string toUpper(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  return s;
}

static bool startsWithCI(const std::string& s, const std::string& prefix) {
  if (s.size() < prefix.size()) {
    return false;
  }
  return toLower(s.substr(0, prefix.size())) == toLower(prefix);
}

/// Convert "WGS84 UTM 32N" to a PROJ4 definition.
static std::string parseUtmProjectionString(const std::string& line) {
  std::istringstream iss(line);
  std::string w1, w2, zone;
  iss >> w1 >> w2 >> zone;

  zone = toUpper(zone);

  int zoneNumber = 0;
  std::string hemisphere = "north";

  if (!zone.empty() && zone.back() == 'N') {
    zoneNumber = std::stoi(zone.substr(0, zone.size() - 1));
    hemisphere = "north";
  } else if (!zone.empty() && zone.back() == 'S') {
    zoneNumber = std::stoi(zone.substr(0, zone.size() - 1));
    hemisphere = "south";
  } else {
    zoneNumber = std::stoi(zone);
    hemisphere = "north";
  }

  return "+proj=utm +zone=" + std::to_string(zoneNumber) + " +" + hemisphere +
         " +ellps=WGS84 +datum=WGS84 +units=m +no_defs";
}

std::string ParseGcpProjectionString(const std::string& line) {
  std::string s = trim(line);

  if (s == "WGS84" || s == "EPSG:4326") {
    return {};  // identity
  }

  if (startsWithCI(s, "WGS84 UTM")) {
    return parseUtmProjectionString(s);
  }

  if (s.find("+proj") != std::string::npos || startsWithCI(s, "EPSG:")) {
    return s;
  }

  // Unknown — treat as identity with a warning (caller can check).
  return {};
}

// ── CrsTransform implementation ──────────────────────────────────────────────

struct CrsTransform::Impl {
  PJ_CONTEXT* ctx = nullptr;
  PJ* transform = nullptr;
  bool identity = true;

  ~Impl() {
    if (transform) {
      proj_destroy(transform);
    }
    if (ctx) {
      proj_context_destroy(ctx);
    }
  }
};

CrsTransform::CrsTransform(const std::string& projString, bool cdnEnabled,
                           const std::string& gridCacheDir)
    : m_impl(std::make_unique<Impl>()) {
  if (projString.empty()) {
    m_impl->identity = true;
    return;
  }

  m_impl->identity = false;
  m_impl->ctx = proj_context_create();
  if (!m_impl->ctx) {
    return;
  }

  if (cdnEnabled) {
    proj_context_set_enable_network(m_impl->ctx, 1);
  } else {
    proj_context_set_enable_network(m_impl->ctx, 0);
  }

  if (!gridCacheDir.empty()) {
    // Set search paths:
    std::vector<std::string> searchPaths;
    searchPaths.push_back(gridCacheDir);
    const char* proj_data = std::getenv("PROJ_DATA");
    if (!proj_data) {
      proj_data = std::getenv("PROJ_LIB");
    }
    if (proj_data) {
      searchPaths.push_back(proj_data);
    }
    std::vector<const char*> c_paths;
    for (const auto& p : searchPaths) {
      c_paths.push_back(p.c_str());
    }
    proj_context_set_search_paths(m_impl->ctx, static_cast<int>(c_paths.size()),
                                  c_paths.data());

    // Also set grid cache filename to be in this directory
    std::string cacheDbPath = gridCacheDir + "/cache.db";
    proj_grid_cache_set_enable(m_impl->ctx, 1);
    proj_grid_cache_set_filename(m_impl->ctx, cacheDbPath.c_str());
  }

  // Create a transformation pipeline: source CRS → WGS-84 (lat/lon/alt).
  m_impl->transform = proj_create_crs_to_crs(m_impl->ctx, projString.c_str(),
                                             "EPSG:4979", nullptr);

  if (!m_impl->transform) {
    return;
  }

  // Normalize output axis order to lon, lat (east, north) for consistency.
  PJ* normalized =
      proj_normalize_for_visualization(m_impl->ctx, m_impl->transform);
  if (normalized) {
    proj_destroy(m_impl->transform);
    m_impl->transform = normalized;
  }
}

CrsTransform::~CrsTransform() = default;
CrsTransform::CrsTransform(CrsTransform&&) noexcept = default;
CrsTransform& CrsTransform::operator=(CrsTransform&&) noexcept = default;

bool CrsTransform::isIdentity() const { return m_impl->identity; }

bool CrsTransform::isValid() const {
  return m_impl->identity || m_impl->transform != nullptr;
}

bool CrsTransform::transform(double easting, double northing, double alt,
                             double& lat, double& lon, double& out_alt) const {
  if (m_impl->identity) {
    // WGS-84 input: easting = longitude, northing = latitude.
    lon = easting;
    lat = northing;
    out_alt = alt;
    return true;
  }

  if (!m_impl->transform) {
    return false;
  }

  PJ_COORD input = proj_coord(easting, northing, alt, 0);
  PJ_COORD output = proj_trans(m_impl->transform, PJ_FWD, input);

  if (output.xyzt.x == HUGE_VAL) {
    return false;
  }

  // After normalize_for_visualization: x=lon, y=lat.
  lon = output.xy.x;
  lat = output.xy.y;
  out_alt = output.xyz.z;
  return true;
}

bool CrsTransform::inverseTransform(double lat, double lon, double alt,
                                    double& easting, double& northing,
                                    double& out_alt) const {
  if (m_impl->identity) {
    // WGS-84: easting = longitude, northing = latitude.
    easting = lon;
    northing = lat;
    out_alt = alt;
    return true;
  }

  if (!m_impl->transform) {
    return false;
  }

  // After normalize_for_visualization the forward direction uses
  // x=lon, y=lat ordering, so the inverse input is (lon, lat, alt).
  PJ_COORD input = proj_coord(lon, lat, alt, 0);
  PJ_COORD output = proj_trans(m_impl->transform, PJ_INV, input);

  if (output.xyzt.x == HUGE_VAL) {
    return false;
  }

  easting = output.xyz.x;
  northing = output.xyz.y;
  out_alt = output.xyz.z;
  return true;
}

}  // namespace geo
