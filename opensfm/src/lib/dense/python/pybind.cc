#include <dense/depthmap_bind.h>
#include <dense/openmvs_exporter.h>
#include <foundation/python_types.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

PYBIND11_MODULE(pydense, m) {
  py::class_<dense::OpenMVSExporter>(m, "OpenMVSExporter")
      .def(py::init())
      .def("add_camera", &dense::OpenMVSExporter::AddCamera)
      .def("add_shot", &dense::OpenMVSExporter::AddShot)
      .def("add_point", &dense::OpenMVSExporter::AddPoint)
      .def("export", &dense::OpenMVSExporter::Export);

  py::class_<dense::SVOFuserWrapper>(m, "SVOFuser")
      .def(py::init())
      .def("set_voxel_size", &dense::SVOFuserWrapper::SetVoxelSize)
      .def("set_trunc_factor", &dense::SVOFuserWrapper::SetTruncFactor)
      .def("set_min_weight", &dense::SVOFuserWrapper::SetMinWeight)
      .def("set_device", &dense::SVOFuserWrapper::SetDevice)
      .def("set_bbox", &dense::SVOFuserWrapper::SetBBox, py::arg("min_world"),
           py::arg("max_world"))
      .def_static("is_gpu_available", &dense::SVOFuserWrapper::IsGPUAvailable)
      .def("count_voxels", &dense::SVOFuserWrapper::CountVoxels)
      .def("add_view", &dense::SVOFuserWrapper::AddView, py::arg("K"),
           py::arg("R"), py::arg("t"), py::arg("depth"), py::arg("normal"),
           py::arg("color"), py::arg("mask"),
           py::arg("confidence") = py::none(), py::arg("name") = "")
      .def("fuse", &dense::SVOFuserWrapper::Fuse)
      .def("fuse_only", &dense::SVOFuserWrapper::FuseOnly)
      .def("refine", &dense::SVOFuserWrapper::Refine, py::arg("color_iters"),
           py::arg("joint_iters"), py::arg("lambda_reg"))
      .def("extract_points", &dense::SVOFuserWrapper::ExtractPoints);

  py::class_<dense::DepthmapClusterEstimatorWrapper>(m,
                                                     "DepthmapClusterEstimator")
      .def(py::init())
      .def("set_max_iterations",
           &dense::DepthmapClusterEstimatorWrapper::SetMaxIterations)
      .def("set_patch_size",
           &dense::DepthmapClusterEstimatorWrapper::SetPatchSize)
      .def("set_max_image_size",
           &dense::DepthmapClusterEstimatorWrapper::SetMaxImageSize)
      .def("set_hierarchy_levels",
           &dense::DepthmapClusterEstimatorWrapper::SetHierarchyLevels)
      .def("set_sigma_spatial",
           &dense::DepthmapClusterEstimatorWrapper::SetSigmaSpatial)
      .def("set_sigma_color",
           &dense::DepthmapClusterEstimatorWrapper::SetSigmaColor)
      .def("set_top_k", &dense::DepthmapClusterEstimatorWrapper::SetTopK)
      .def("set_use_census",
           &dense::DepthmapClusterEstimatorWrapper::SetUseCensus)
      .def("set_smooth_weight",
           &dense::DepthmapClusterEstimatorWrapper::SetSmoothWeight)
      .def("set_checkerboard_filter",
           &dense::DepthmapClusterEstimatorWrapper::SetCheckerboardFilter)
      .def("set_speckle_min_size",
           &dense::DepthmapClusterEstimatorWrapper::SetSpeckleMinSize)
      .def("set_gap_max_size",
           &dense::DepthmapClusterEstimatorWrapper::SetGapMaxSize)
      .def("set_geom_consistency_weight",
           &dense::DepthmapClusterEstimatorWrapper::SetGeomConsistencyWeight)
      .def("set_device", &dense::DepthmapClusterEstimatorWrapper::SetDevice)
      .def("begin_ref_view",
           &dense::DepthmapClusterEstimatorWrapper::BeginRefView)
      .def("add_source_view",
           &dense::DepthmapClusterEstimatorWrapper::AddSourceView)
      .def("set_sfm_points",
           &dense::DepthmapClusterEstimatorWrapper::SetSfMPoints)
      .def("add_geom_link",
           &dense::DepthmapClusterEstimatorWrapper::AddGeomLink)
      .def("run", &dense::DepthmapClusterEstimatorWrapper::Run)
      .def("clear", &dense::DepthmapClusterEstimatorWrapper::Clear)
      .def_static("is_available",
                  &dense::DepthmapClusterEstimatorWrapper::IsAvailable)
      .def_static("gpu_memory_bytes",
                  &dense::DepthmapClusterEstimatorWrapper::GpuMemoryBytes)
      .def_static("num_devices",
                  &dense::DepthmapClusterEstimatorWrapper::NumDevices)
      .def_static("device_name",
                  &dense::DepthmapClusterEstimatorWrapper::DeviceName)
      .def_static("device_memory_bytes",
                  &dense::DepthmapClusterEstimatorWrapper::DeviceMemoryBytes)
      .def_static("device_is_gpu",
                  &dense::DepthmapClusterEstimatorWrapper::DeviceIsGPU)
      .def_static(
          "device_available_memory",
          &dense::DepthmapClusterEstimatorWrapper::DeviceAvailableMemory)
      .def_static("reserve_device_memory",
                  &dense::DepthmapClusterEstimatorWrapper::ReserveDeviceMemory)
      .def_static("release_device_memory",
                  &dense::DepthmapClusterEstimatorWrapper::ReleaseDeviceMemory);

  py::class_<dense::DepthmapCleanerWrapper>(m, "GPUDepthmapCleaner")
      .def(py::init())
      .def("set_same_depth_threshold",
           &dense::DepthmapCleanerWrapper::SetSameDepthThreshold)
      .def("set_min_consistent_views",
           &dense::DepthmapCleanerWrapper::SetMinConsistentViews)
      .def("set_device", &dense::DepthmapCleanerWrapper::SetDevice)
      .def("add_view", &dense::DepthmapCleanerWrapper::AddView)
      .def("clean", &dense::DepthmapCleanerWrapper::Clean)
      .def("clear", &dense::DepthmapCleanerWrapper::Clear)
      .def_static("is_available", &dense::DepthmapCleanerWrapper::IsAvailable);

  py::class_<dense::DSMRasterizerWrapper>(m, "DSMRasterizer")
      .def(py::init())
      .def("set_gsd", &dense::DSMRasterizerWrapper::SetGSD)
      .def("set_bbox", &dense::DSMRasterizerWrapper::SetBBox, py::arg("min_xy"),
           py::arg("max_xy"))
      .def("set_device", &dense::DSMRasterizerWrapper::SetDevice)
      .def("set_mode_threshold", &dense::DSMRasterizerWrapper::SetModeThreshold)
      .def("set_min_count", &dense::DSMRasterizerWrapper::SetMinCount)
      .def("set_bilateral", &dense::DSMRasterizerWrapper::SetBilateral,
           py::arg("enabled"), py::arg("radius"), py::arg("range_sigma"))
      .def("begin", &dense::DSMRasterizerWrapper::Begin)
      .def("scatter", &dense::DSMRasterizerWrapper::Scatter, py::arg("K"),
           py::arg("R"), py::arg("t"), py::arg("depth"), py::arg("normal"),
           py::arg("confidence"))
      .def("update_modes", &dense::DSMRasterizerWrapper::UpdateModes)
      .def("finish", &dense::DSMRasterizerWrapper::Finish)
      .def_static("is_available", &dense::DSMRasterizerWrapper::IsAvailable);
}
