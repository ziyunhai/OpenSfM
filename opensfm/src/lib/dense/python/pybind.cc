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
      .def("set_num_levels", &dense::SVOFuserWrapper::SetNumLevels)
      .def("set_decimate_flat", &dense::SVOFuserWrapper::SetDecimateFat)
      .def("set_edge_threshold", &dense::SVOFuserWrapper::SetEdgeThreshold)
      .def("set_min_count", &dense::SVOFuserWrapper::SetMinCount)
      .def("set_relative_min_weight",
           &dense::SVOFuserWrapper::SetRelativeMinWeight)
      .def("set_dsm_wall_cull_nz", &dense::SVOFuserWrapper::SetDSMWallCullNz)
      .def("set_bbox", &dense::SVOFuserWrapper::SetBBox, py::arg("min_world"),
           py::arg("max_world"))
      .def_static("is_gpu_available", &dense::SVOFuserWrapper::IsGPUAvailable)
      .def("capacity", &dense::SVOFuserWrapper::Capacity)
      .def("release_refine_buffers",
           &dense::SVOFuserWrapper::ReleaseRefineBuffers)
      .def("count_voxels", &dense::SVOFuserWrapper::CountVoxels)
      .def("add_view", &dense::SVOFuserWrapper::AddView, py::arg("K"),
           py::arg("R"), py::arg("t"), py::arg("depth"), py::arg("normal"),
           py::arg("color"), py::arg("mask"),
           py::arg("confidence") = py::none(), py::arg("name") = "")
      .def("fuse", &dense::SVOFuserWrapper::Fuse)
      .def("fuse_only", &dense::SVOFuserWrapper::FuseOnly)
      .def("refine_geometry", &dense::SVOFuserWrapper::RefineGeometry,
           py::arg("iters"), py::arg("lambda_reg"),
           py::arg("neighbors") =
               std::map<std::string, std::vector<std::string>>(),
           py::arg("lambda_anchor") = 0.0f, py::arg("early_stop_rel") = 0.0f)
      .def("extract_and_bake", &dense::SVOFuserWrapper::ExtractAndBake)
      .def("extract_mesh", &dense::SVOFuserWrapper::ExtractMesh)
      .def("prune_by_visibility", &dense::SVOFuserWrapper::PruneByVisibility,
           py::arg("iterations"), py::arg("carve_margin"),
           py::arg("carve_threshold"), py::arg("support_min"))
      .def("extract_points", &dense::SVOFuserWrapper::ExtractPoints)
      .def("render_dsm_ortho", &dense::SVOFuserWrapper::RenderDSMOrtho,
           py::arg("origin_x"), py::arg("origin_y"), py::arg("gsd"),
           py::arg("width"), py::arg("height"), py::arg("z_min"),
           py::arg("z_max"))
      .def("bake_colors", &dense::SVOFuserWrapper::BakeColorsStandalone,
           py::arg("points"), py::arg("normals"), py::arg("n_final") = 2,
           py::arg("irls_iters") = 3, py::arg("relax_occlusion") = py::none(),
           py::arg("dsm_occ") = py::none(), py::arg("dsm_origin_x") = 0.0f,
           py::arg("dsm_origin_y") = 0.0f, py::arg("dsm_gsd") = 0.0f,
           py::arg("dsm_max_z") = 0.0f);

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
      .def("set_anchor_views",
           &dense::DepthmapClusterEstimatorWrapper::SetAnchorViews)
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
      .def("set_carving_threshold",
           &dense::DepthmapCleanerWrapper::SetCarvingThreshold)
      .def("set_max_carved_views",
           &dense::DepthmapCleanerWrapper::SetMaxCarvedViews)
      .def("set_grazing_cos_threshold",
           &dense::DepthmapCleanerWrapper::SetGrazingCosThreshold)
      .def("add_view", &dense::DepthmapCleanerWrapper::AddView)
      .def("add_view_with_normal",
           &dense::DepthmapCleanerWrapper::AddViewWithNormal)
      .def("clean", &dense::DepthmapCleanerWrapper::Clean)
      .def("clear", &dense::DepthmapCleanerWrapper::Clear)
      .def_static("is_available", &dense::DepthmapCleanerWrapper::IsAvailable);

  py::class_<dense::GPUDiffuserWrapper>(m, "GPUDiffuser")
      .def(py::init())
      .def("set_device", &dense::GPUDiffuserWrapper::SetDevice)
      .def("diffuse", &dense::GPUDiffuserWrapper::Diffuse, py::arg("guide"),
           py::arg("iterations"), py::arg("kappa"), py::arg("dt"))
      .def("upload_grid", &dense::GPUDiffuserWrapper::UploadGrid,
           py::arg("grid"))
      .def("snap_edges", &dense::GPUDiffuserWrapper::SnapEdges, py::arg("dsm"),
           py::arg("guide"), py::arg("iterations"), py::arg("radius"),
           py::arg("sigma_spatial"), py::arg("sigma_range"))
      .def("shock_filter", &dense::GPUDiffuserWrapper::ShockFilter,
           py::arg("dsm"), py::arg("iterations"), py::arg("win"), py::arg("dt"),
           py::arg("coherence"), py::arg("gsd"), py::arg("edge_slope"))
      .def("gated_median", &dense::GPUDiffuserWrapper::GatedMedian,
           py::arg("ortho"), py::arg("valid"), py::arg("threshold"));
}
