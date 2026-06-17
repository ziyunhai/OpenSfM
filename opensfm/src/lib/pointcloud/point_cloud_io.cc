#include <pointcloud/point_cloud_io.h>

#include <algorithm>
#include <cctype>

#include <pointcloud/ply_io.h>
// LAS / LAZ back-ends are added in later milestones (M4 / M5).
#if defined(POINTCLOUD_HAVE_LAS)
#include <pointcloud/las_io.h>
#endif
#if defined(POINTCLOUD_HAVE_LAZ)
#include <pointcloud/laz_io.h>
#endif

namespace pointcloud {

void PointChunk::clear() {
  count = 0;
  positions.clear();
  normals.clear();
  colors.clear();
  labels.clear();
}

void PointChunk::resize(uint64_t n, const PointAttributes& attrs) {
  count = n;
  positions.assign(n * 3, 0.0);
  normals.assign(attrs.hasNormals ? n * 3 : 0, 0.0f);
  colors.assign(attrs.hasColors ? n * 3 : 0, 0);
  labels.assign(attrs.hasLabels ? n : 0, 0);
}

std::string fileExtensionLower(const std::string& path) {
  auto slash = path.find_last_of("/\\");
  auto dot = path.find_last_of('.');
  if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) {
    return "";
  }
  std::string ext = path.substr(dot);
  std::transform(ext.begin(), ext.end(), ext.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return ext;
}

std::unique_ptr<PointCloudReader> makeReader(const std::string& path) {
  const std::string ext = fileExtensionLower(path);
  if (ext == ".ply") {
    auto r = std::make_unique<PLYReader>(path);
    return r->ok() ? std::unique_ptr<PointCloudReader>(std::move(r)) : nullptr;
  }
#if defined(POINTCLOUD_HAVE_LAS)
  if (ext == ".las") {
    auto r = std::make_unique<LASReader>(path);
    return r->ok() ? std::unique_ptr<PointCloudReader>(std::move(r)) : nullptr;
  }
#endif
#if defined(POINTCLOUD_HAVE_LAZ)
  if (ext == ".laz") {
    auto r = std::make_unique<LAZReader>(path);
    return r->ok() ? std::unique_ptr<PointCloudReader>(std::move(r)) : nullptr;
  }
#endif
  return nullptr;
}

std::unique_ptr<PointCloudWriter> makeWriter(const std::string& path,
                                             const PointCloudHeader& header) {
  const std::string ext = fileExtensionLower(path);
  if (ext == ".ply") {
    auto w = std::make_unique<PLYWriter>(path, header);
    return w->ok() ? std::unique_ptr<PointCloudWriter>(std::move(w)) : nullptr;
  }
#if defined(POINTCLOUD_HAVE_LAS)
  if (ext == ".las") {
    auto w = std::make_unique<LASWriter>(path, header);
    return w->ok() ? std::unique_ptr<PointCloudWriter>(std::move(w)) : nullptr;
  }
#endif
#if defined(POINTCLOUD_HAVE_LAZ)
  if (ext == ".laz") {
    auto w = std::make_unique<LAZWriter>(path, header);
    return w->ok() ? std::unique_ptr<PointCloudWriter>(std::move(w)) : nullptr;
  }
#endif
  return nullptr;
}

}  // namespace pointcloud
