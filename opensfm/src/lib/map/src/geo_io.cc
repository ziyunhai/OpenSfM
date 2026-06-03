#include <geo/crs_transform.h>
#include <map/geo_io.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>

namespace map {

static std::string trim(const std::string& s) {
  auto start = s.find_first_not_of(" \t\r\n");
  if (start == std::string::npos) {
    return {};
  }
  auto end = s.find_last_not_of(" \t\r\n");
  return s.substr(start, end - start + 1);
}

static std::vector<std::string> divide(const std::string& str,
                                       const std::string& delimiter) {
  std::vector<std::string> tokens;
  size_t lastPos = str.find_first_not_of(delimiter, 0);
  size_t pos = str.find_first_of(delimiter, lastPos);
  while (std::string::npos != pos || std::string::npos != lastPos) {
    tokens.push_back(str.substr(lastPos, pos - lastPos));
    lastPos = str.find_first_not_of(delimiter, pos);
    pos = str.find_first_of(delimiter, lastPos);
  }
  return tokens;
}

struct NumberToken {
  double value;
  size_t original_index;
};

struct StdCandidate {
  size_t start_idx;
  size_t len;
  double lat_std;
  double lon_std;
  double alt_std;
};

struct YprCandidate {
  size_t start_idx;
  double yaw;
  double pitch;
  double roll;
};

static bool parseDouble(const std::string& token, double& val) {
  try {
    size_t pos;
    val = std::stod(token, &pos);
    if (pos == token.size()) {
      return true;
    }
  } catch (...) {
  }
  return false;
}

static bool areConsecutive(const std::vector<NumberToken>& list,
                           size_t start_idx, size_t count,
                           size_t filename_idx) {
  for (size_t i = 1; i < count; ++i) {
    size_t prev_idx = list[start_idx + i - 1].original_index;
    size_t curr_idx = list[start_idx + i].original_index;
    if (curr_idx == prev_idx + 1) {
      continue;
    }
    if (filename_idx != std::string::npos && curr_idx == prev_idx + 2 &&
        filename_idx == prev_idx + 1) {
      continue;
    }
    return false;
  }
  return true;
}

static bool tryXyz(const std::vector<NumberToken>& list, size_t start_idx,
                   const geo::CrsTransform& transform, double& out_lat,
                   double& out_lon, double& out_alt, size_t filename_idx) {
  if (start_idx + 2 >= list.size()) {
    return false;
  }
  if (!areConsecutive(list, start_idx, 3, filename_idx)) {
    return false;
  }

  double x = list[start_idx].value;
  double y = list[start_idx + 1].value;
  double z = list[start_idx + 2].value;

  if (transform.isIdentity()) {
    if (std::abs(x) <= 90.0 && std::abs(y) <= 180.0) {
      out_lat = x;
      out_lon = y;
      out_alt = z;
      return true;
    } else if (std::abs(y) <= 90.0 && std::abs(x) <= 180.0) {
      out_lon = x;
      out_lat = y;
      out_alt = z;
      return true;
    }
    return false;
  } else {
    double lat = 0, lon = 0, alt = 0;
    if (transform.transform(x, y, z, lat, lon, alt)) {
      if (lat >= -90.0 && lat <= 90.0 && lon >= -180.0 && lon <= 180.0) {
        out_lat = lat;
        out_lon = lon;
        out_alt = alt;
        return true;
      }
    }
    return false;
  }
}

std::vector<GeolocationData> ParseGeolocationFile(
    const std::string& content, const std::vector<std::string>& dataset_images,
    const std::string& crs, bool cdnEnabled, const std::string& gridCacheDir) {
  std::vector<GeolocationData> parsed_data;

  std::unordered_set<std::string> valid_images(dataset_images.begin(),
                                               dataset_images.end());

  std::string proj = geo::ParseGcpProjectionString(crs);
  geo::CrsTransform transform(proj, cdnEnabled, gridCacheDir);

  char chosen_delim = '\0';
  for (char delim : {',', '\t', ' '}) {
    std::istringstream iss(content);
    std::string line;
    bool found = false;
    int check_count = 0;
    while (std::getline(iss, line) && check_count < 20) {
      if (line.empty() || line.front() == '#') {
        continue;
      }
      check_count++;
      auto tokens = divide(line, std::string(1, delim));
      for (const auto& token : tokens) {
        std::string t = trim(token);
        if (valid_images.count(t) > 0) {
          chosen_delim = delim;
          found = true;
          break;
        }
      }
      if (found) {
        break;
      }
    }
    if (found) {
      break;
    }
  }

  if (chosen_delim == '\0') {
    chosen_delim = ',';  // fallback
  }

  std::string delim_str(1, chosen_delim);

  std::istringstream iss(content);
  std::string line;
  while (std::getline(iss, line)) {
    if (line.empty() || line.front() == '#') {
      continue;
    }

    auto tokens = divide(line, delim_str);

    std::string filename;
    size_t filename_idx = std::string::npos;
    for (size_t i = 0; i < tokens.size(); ++i) {
      std::string t = trim(tokens[i]);
      if (valid_images.count(t) > 0) {
        filename = t;
        filename_idx = i;
        break;
      }
    }

    if (filename_idx == std::string::npos) {
      continue;
    }

    std::vector<NumberToken> num_tokens;
    for (size_t i = 0; i < tokens.size(); ++i) {
      if (i == filename_idx) {
        continue;
      }
      double val;
      if (parseDouble(trim(tokens[i]), val)) {
        num_tokens.push_back({val, i});
      }
    }

    GeolocationData data;
    data.filename = filename;

    bool xyz_found = false;
    size_t xyz_start_idx = 0;
    double lat = 0, lon = 0, alt = 0;

    if (num_tokens.size() >= 3) {
      bool ok_first =
          tryXyz(num_tokens, 0, transform, lat, lon, alt, filename_idx);
      bool ok_last = false;
      if (num_tokens.size() >= 6) {
        ok_last = tryXyz(num_tokens, num_tokens.size() - 3, transform, lat, lon,
                         alt, filename_idx);
      }

      if (ok_first) {
        tryXyz(num_tokens, 0, transform, lat, lon, alt, filename_idx);
        xyz_start_idx = 0;
        xyz_found = true;
      } else if (ok_last) {
        tryXyz(num_tokens, num_tokens.size() - 3, transform, lat, lon, alt,
               filename_idx);
        xyz_start_idx = num_tokens.size() - 3;
        xyz_found = true;
      }
    }

    if (!xyz_found) {
      continue;
    }

    data.lat = lat;
    data.lon = lon;
    data.alt = alt;
    data.has_lla = true;

    std::vector<NumberToken> remaining;
    for (size_t i = 0; i < num_tokens.size(); ++i) {
      if (i >= xyz_start_idx && i < xyz_start_idx + 3) {
        continue;
      }
      remaining.push_back(num_tokens[i]);
    }

    std::vector<StdCandidate> std_candidates;
    // Try triplet first
    if (remaining.size() >= 3) {
      for (size_t i = 0; i + 2 < remaining.size(); ++i) {
        if (areConsecutive(remaining, i, 3, filename_idx)) {
          if (std::abs(remaining[i].value) < 0.5 &&
              std::abs(remaining[i + 1].value) < 0.5 &&
              std::abs(remaining[i + 2].value) < 0.5) {
            std_candidates.push_back({i, 3, std::abs(remaining[i].value),
                                      std::abs(remaining[i + 1].value),
                                      std::abs(remaining[i + 2].value)});
          }
        }
      }
    }

    // Try pair next
    if (remaining.size() >= 2) {
      for (size_t i = 0; i + 1 < remaining.size(); ++i) {
        if (areConsecutive(remaining, i, 2, filename_idx)) {
          if (std::abs(remaining[i].value) < 0.5 &&
              std::abs(remaining[i + 1].value) < 0.5) {
            std_candidates.push_back({i, 2, std::abs(remaining[i].value),
                                      std::abs(remaining[i].value),
                                      std::abs(remaining[i + 1].value)});
          }
        }
      }
    }

    std::vector<YprCandidate> ypr_candidates;
    if (remaining.size() >= 3) {
      for (size_t i = 0; i + 2 < remaining.size(); ++i) {
        if (areConsecutive(remaining, i, 3, filename_idx)) {
          ypr_candidates.push_back({i, remaining[i].value,
                                    remaining[i + 1].value,
                                    remaining[i + 2].value});
        }
      }
    }

    bool pair_found = false;
    StdCandidate best_std;
    YprCandidate best_ypr;

    for (const auto& sc : std_candidates) {
      for (const auto& yc : ypr_candidates) {
        if (sc.start_idx + sc.len <= yc.start_idx ||
            yc.start_idx + 3 <= sc.start_idx) {
          best_std = sc;
          best_ypr = yc;
          pair_found = true;
          break;
        }
      }
      if (pair_found) {
        break;
      }
    }

    if (pair_found) {
      data.lat_std = best_std.lat_std;
      data.lon_std = best_std.lon_std;
      data.alt_std = best_std.alt_std;
      data.has_std = true;

      data.yaw = best_ypr.yaw;
      data.pitch = best_ypr.pitch;
      data.roll = best_ypr.roll;
      data.has_ypr = true;
    } else {
      if (!std_candidates.empty()) {
        const auto& sc = std_candidates.front();
        data.lat_std = sc.lat_std;
        data.lon_std = sc.lon_std;
        data.alt_std = sc.alt_std;
        data.has_std = true;
      } else if (!ypr_candidates.empty()) {
        const auto& yc = ypr_candidates.front();
        data.yaw = yc.yaw;
        data.pitch = yc.pitch;
        data.roll = yc.roll;
        data.has_ypr = true;
      }
    }

    parsed_data.push_back(data);
  }

  return parsed_data;
}

}  // namespace map