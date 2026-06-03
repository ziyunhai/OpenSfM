#pragma once

#include <string>
#include <vector>
#include <optional>
#include <unordered_set>

namespace map {

struct GeolocationData {
    std::string filename;
    
    bool has_lla{false};
    double lat{0};
    double lon{0};
    double alt{0};
    
    bool has_std{false};
    double lat_std{0};
    double lon_std{0};
    double alt_std{0};
    
    bool has_ypr{false};
    double yaw{0};
    double pitch{0};
    double roll{0};
    
    bool has_opk{false};
    double omega{0};
    double phi{0};
    double kappa{0};
};

std::vector<GeolocationData> ParseGeolocationFile(const std::string& content,
                                                  const std::vector<std::string>& dataset_images,
                                                  const std::string& crs);

} // namespace map
