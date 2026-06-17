#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <stdexcept>

#include <pointcloud/octree_builder.h>
#include <pointcloud/point_cloud_io.h>
#include <pointcloud/tile_format.h>

namespace py = pybind11;

namespace {

// Convert one PointChunk to a Python tuple (positions f64, normals f32|None,
// colors u8|None, labels u8|None). Empty optional attributes become None.
py::object chunkToPy(const pointcloud::PointChunk& c) {
  const uint64_t n = c.count;
  py::array_t<double> pos({static_cast<py::ssize_t>(n), py::ssize_t(3)});
  std::memcpy(pos.mutable_data(), c.positions.data(), n * 3 * sizeof(double));

  auto opt = [&](const auto& vec, int cols, auto sample) -> py::object {
    using T = decltype(sample);
    if (vec.empty()) return py::none();
    if (cols == 1) {
      py::array_t<T> a({static_cast<py::ssize_t>(n)});
      std::memcpy(a.mutable_data(), vec.data(), vec.size() * sizeof(T));
      return std::move(a);
    }
    py::array_t<T> a({static_cast<py::ssize_t>(n), py::ssize_t(cols)});
    std::memcpy(a.mutable_data(), vec.data(), vec.size() * sizeof(T));
    return std::move(a);
  };

  return py::make_tuple(std::move(pos), opt(c.normals, 3, float{}),
                        opt(c.colors, 3, uint8_t{}), opt(c.labels, 1, uint8_t{}));
}

// Fill a PointChunk from numpy arrays for the writer path.
pointcloud::PointChunk chunkFromPy(
    py::array_t<double, py::array::c_style | py::array::forcecast> positions,
    py::array_t<float, py::array::c_style | py::array::forcecast> normals,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> colors,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> labels) {
  auto pb = positions.request();
  if (pb.ndim != 2 || pb.shape[1] != 3) {
    throw std::runtime_error("positions must be an (N, 3) array");
  }
  pointcloud::PointChunk c;
  c.count = static_cast<uint64_t>(pb.shape[0]);
  c.positions.assign(static_cast<const double*>(pb.ptr),
                     static_cast<const double*>(pb.ptr) + c.count * 3);
  if (normals.size() > 0) {
    auto b = normals.request();
    c.normals.assign(static_cast<const float*>(b.ptr),
                     static_cast<const float*>(b.ptr) + c.count * 3);
  }
  if (colors.size() > 0) {
    auto b = colors.request();
    c.colors.assign(static_cast<const uint8_t*>(b.ptr),
                    static_cast<const uint8_t*>(b.ptr) + c.count * 3);
  }
  if (labels.size() > 0) {
    auto b = labels.request();
    c.labels.assign(static_cast<const uint8_t*>(b.ptr),
                    static_cast<const uint8_t*>(b.ptr) + c.count);
  }
  return c;
}

}  // namespace

PYBIND11_MODULE(pypointcloud, m) {
  m.doc() = "Octree point cloud tiling for large point cloud visualization.";

  // ── OctreeBuilderConfig ──
  py::class_<pointcloud::OctreeBuilderConfig>(m, "OctreeBuilderConfig")
      .def(py::init<>())
      .def_readwrite("max_points_per_tile",
                     &pointcloud::OctreeBuilderConfig::maxPointsPerTile)
      .def_readwrite("max_depth",
                     &pointcloud::OctreeBuilderConfig::maxDepth)
      .def_readwrite("lod_sample_count",
                     &pointcloud::OctreeBuilderConfig::lodSampleCount)
      .def_readwrite("output_dir",
                     &pointcloud::OctreeBuilderConfig::outputDir);

  // ── OctreeMetadata ──
  py::class_<pointcloud::OctreeMetadata>(m, "OctreeMetadata")
      .def(py::init<>())
      .def_readonly("total_points",
                    &pointcloud::OctreeMetadata::totalPoints)
      .def_readonly("max_depth", &pointcloud::OctreeMetadata::maxDepth)
      .def_readonly("root_spacing",
                    &pointcloud::OctreeMetadata::rootSpacing)
      .def_readonly("aabb_min", &pointcloud::OctreeMetadata::aabbMin)
      .def_readonly("aabb_max", &pointcloud::OctreeMetadata::aabbMax);

  // ── build_octree ──
  m.def(
      "build_octree",
      [](py::array_t<float, py::array::c_style | py::array::forcecast>
             positions,
         py::array_t<float, py::array::c_style | py::array::forcecast> normals,
         py::array_t<uint8_t, py::array::c_style | py::array::forcecast> colors,
         py::array_t<float, py::array::c_style | py::array::forcecast> radii,
         const pointcloud::OctreeBuilderConfig& config)
          -> pointcloud::OctreeMetadata {
        auto pos_buf = positions.request();
        if (pos_buf.ndim != 2 || pos_buf.shape[1] != 3) {
          throw std::runtime_error("positions must be an (N, 3) float32 array");
        }
        uint64_t n = static_cast<uint64_t>(pos_buf.shape[0]);

        pointcloud::BuilderInput input;
        input.positions = static_cast<const float*>(pos_buf.ptr);
        input.numPoints = n;

        // Normals.
        if (normals.size() > 0) {
          auto nrm_buf = normals.request();
          if (nrm_buf.ndim == 2 && nrm_buf.shape[0] == static_cast<long>(n) &&
              nrm_buf.shape[1] == 3) {
            input.normals = static_cast<const float*>(nrm_buf.ptr);
          }
        }

        // Colors.
        if (colors.size() > 0) {
          auto col_buf = colors.request();
          if (col_buf.ndim == 2 && col_buf.shape[0] == static_cast<long>(n) &&
              col_buf.shape[1] == 3) {
            input.colors = static_cast<const uint8_t*>(col_buf.ptr);
          }
        }

        // Radii.
        if (radii.size() > 0) {
          auto rad_buf = radii.request();
          if (rad_buf.ndim == 1 && rad_buf.shape[0] == static_cast<long>(n)) {
            input.radii = static_cast<const float*>(rad_buf.ptr);
          }
        }

        py::gil_scoped_release release;
        return pointcloud::buildOctree(input, config);
      },
      py::arg("positions"), py::arg("normals") = py::array_t<float>(),
      py::arg("colors") = py::array_t<uint8_t>(),
      py::arg("radii") = py::array_t<float>(), py::arg("config") = py::none(),
      R"(Build an octree tile set from a flat point cloud.

Args:
    positions: (N, 3) float32 array of XYZ positions.
    normals: (N, 3) float32 array of normals (optional).
    colors: (N, 3) uint8 array of RGB colors (optional).
    radii: (N,) float32 array of splat radii (optional).
    config: OctreeBuilderConfig with output_dir set.

Returns:
    OctreeMetadata describing the generated octree.
)");

  // ── Out-of-core octree from a file (bounded RAM) ──
  m.def(
      "build_octree_from_file",
      [](const std::string& cloud_path,
         const pointcloud::OctreeBuilderConfig& config, uint32_t split_depth,
         uint64_t max_bucket_points,
         const std::string& temp_dir) -> pointcloud::OctreeMetadata {
        pointcloud::OocConfig oc;
        oc.base = config;
        oc.splitDepth = split_depth;
        oc.maxBucketPoints = max_bucket_points;
        oc.tempDir = temp_dir;
        py::gil_scoped_release release;
        return pointcloud::buildOctreeFromFile(cloud_path, oc);
      },
      py::arg("cloud_path"), py::arg("config"), py::arg("split_depth") = 4,
      py::arg("max_bucket_points") = static_cast<uint64_t>(8000000),
      py::arg("temp_dir") = std::string(),
      R"(Build an octree tile set from a point-cloud FILE with bounded RAM.

The cloud (.ply/.las/.laz) is read out-of-core (binary PLY is memory-mapped),
so peak memory is independent of the point count. Output tiles are
format-identical to build_octree.

Args:
    cloud_path: path to a .ply / .las / .laz point cloud.
    config: OctreeBuilderConfig with output_dir set.
    split_depth: octree depth at which points are bucketed to disk.
    max_bucket_points: a bucket larger than this is recursively re-bucketed.
    temp_dir: scratch directory (default: output_dir/_ooc_tmp).

Returns:
    OctreeMetadata describing the generated octree.
)");

  // ── PointCloud IO: header + chunked reader/writer ──
  py::class_<pointcloud::PointCloudHeader>(m, "PointCloudHeader")
      .def(py::init<>())
      .def_readwrite("point_count", &pointcloud::PointCloudHeader::pointCount)
      .def_property(
          "has_normals",
          [](const pointcloud::PointCloudHeader& h) { return h.attrs.hasNormals; },
          [](pointcloud::PointCloudHeader& h, bool v) { h.attrs.hasNormals = v; })
      .def_property(
          "has_colors",
          [](const pointcloud::PointCloudHeader& h) { return h.attrs.hasColors; },
          [](pointcloud::PointCloudHeader& h, bool v) { h.attrs.hasColors = v; })
      .def_property(
          "has_labels",
          [](const pointcloud::PointCloudHeader& h) { return h.attrs.hasLabels; },
          [](pointcloud::PointCloudHeader& h, bool v) { h.attrs.hasLabels = v; });

  py::class_<pointcloud::PointCloudReader>(m, "PointCloudReader")
      .def_property_readonly("total_count",
                             &pointcloud::PointCloudReader::totalCount)
      .def_property_readonly("has_count",
                             &pointcloud::PointCloudReader::hasCount)
      .def("attributes",
           [](pointcloud::PointCloudReader& r) {
             auto a = r.attributes();
             return py::make_tuple(a.hasNormals, a.hasColors, a.hasLabels);
           })
      .def("rewind", &pointcloud::PointCloudReader::rewind)
      .def(
          "read_chunk",
          [](pointcloud::PointCloudReader& r, uint64_t max_points) -> py::object {
            pointcloud::PointChunk c;
            bool more;
            {
              py::gil_scoped_release release;
              more = r.readChunk(max_points, c);
            }
            if (!more) return py::none();
            return chunkToPy(c);
          },
          py::arg("max_points"),
          "Read up to max_points more points → (positions f64 (n,3), normals "
          "f32 (n,3)|None, colors u8 (n,3)|None, labels u8 (n,)|None); None at EOF.");

  py::class_<pointcloud::PointCloudWriter>(m, "PointCloudWriter")
      .def(
          "write_chunk",
          [](pointcloud::PointCloudWriter& w,
             py::array_t<double, py::array::c_style | py::array::forcecast> positions,
             py::array_t<float, py::array::c_style | py::array::forcecast> normals,
             py::array_t<uint8_t, py::array::c_style | py::array::forcecast> colors,
             py::array_t<uint8_t, py::array::c_style | py::array::forcecast> labels) {
            auto c = chunkFromPy(positions, normals, colors, labels);
            py::gil_scoped_release release;
            return w.writeChunk(c);
          },
          py::arg("positions"), py::arg("normals") = py::array_t<float>(),
          py::arg("colors") = py::array_t<uint8_t>(),
          py::arg("labels") = py::array_t<uint8_t>())
      .def("finalize", &pointcloud::PointCloudWriter::finalize);

  m.def(
      "open_reader",
      [](const std::string& path) -> std::unique_ptr<pointcloud::PointCloudReader> {
        auto r = pointcloud::makeReader(path);
        if (!r) {
          throw std::runtime_error("Cannot open point cloud for reading: " + path);
        }
        return r;
      },
      py::arg("path"), "Open a .ply/.las/.laz cloud for chunked reading.");

  m.def(
      "open_writer",
      [](const std::string& path,
         const pointcloud::PointCloudHeader& header)
          -> std::unique_ptr<pointcloud::PointCloudWriter> {
        auto w = pointcloud::makeWriter(path, header);
        if (!w) {
          throw std::runtime_error("Cannot open point cloud for writing: " + path);
        }
        return w;
      },
      py::arg("path"), py::arg("header") = pointcloud::PointCloudHeader(),
      "Open a .ply/.las/.laz cloud for chunked writing.");

  // ── Convenience whole-cloud helpers (for small clouds / tests) ──
  m.def(
      "read_point_cloud",
      [](const std::string& path) -> py::object {
        auto r = pointcloud::makeReader(path);
        if (!r) throw std::runtime_error("Cannot open point cloud: " + path);
        pointcloud::PointChunk all;
        all.resize(0, r->attributes());
        pointcloud::PointChunk c;
        {
          py::gil_scoped_release release;
          while (r->readChunk(1u << 20, c)) {
            all.positions.insert(all.positions.end(), c.positions.begin(),
                                 c.positions.end());
            all.normals.insert(all.normals.end(), c.normals.begin(),
                               c.normals.end());
            all.colors.insert(all.colors.end(), c.colors.begin(), c.colors.end());
            all.labels.insert(all.labels.end(), c.labels.begin(), c.labels.end());
            all.count += c.count;
          }
        }
        return chunkToPy(all);
      },
      py::arg("path"),
      "Load an entire cloud → (positions f64, normals|None, colors|None, "
      "labels|None). Use open_reader for large clouds.");

  m.def(
      "write_point_cloud",
      [](const std::string& path,
         py::array_t<double, py::array::c_style | py::array::forcecast> positions,
         py::array_t<float, py::array::c_style | py::array::forcecast> normals,
         py::array_t<uint8_t, py::array::c_style | py::array::forcecast> colors,
         py::array_t<uint8_t, py::array::c_style | py::array::forcecast> labels) {
        auto c = chunkFromPy(positions, normals, colors, labels);
        pointcloud::PointCloudHeader h;
        h.pointCount = c.count;
        h.attrs.hasNormals = !c.normals.empty();
        h.attrs.hasColors = !c.colors.empty();
        h.attrs.hasLabels = !c.labels.empty();
        auto w = pointcloud::makeWriter(path, h);
        if (!w) throw std::runtime_error("Cannot open point cloud: " + path);
        py::gil_scoped_release release;
        return w->writeChunk(c) && w->finalize();
      },
      py::arg("path"), py::arg("positions"),
      py::arg("normals") = py::array_t<float>(),
      py::arg("colors") = py::array_t<uint8_t>(),
      py::arg("labels") = py::array_t<uint8_t>(),
      "Write an entire cloud to .ply/.las/.laz (format by extension).");
}
