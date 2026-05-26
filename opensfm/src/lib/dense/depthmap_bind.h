#pragma once

#include <foundation/python_types.h>

// Conditional compilation: if OpenCL is not available, provide a stub
// that raises a clear error at runtime.
#ifdef OPENSFM_HAVE_OPENCL

#include <dense/cleaner.h>
#include <dense/cluster.h>
#include <dense/dsm_opencl_kernels.h>
#include <pybind11/eigen.h>

namespace dense {

// Row-major (N, 3) double matrix for SfM points (matches numpy C-order).
using PointsMatrix = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;

class DepthmapClusterEstimatorWrapper {
 public:
  void SetMaxIterations(int n) { params_.max_iterations = n; }
  void SetPatchSize(int size) { params_.patch_size = size; }
  void SetMaxImageSize(int size) { params_.max_image_size = size; }
  void SetHierarchyLevels(int levels) { params_.hierarchy_levels = levels; }
  void SetSigmaSpatial(float v) { params_.sigma_spatial = v; }
  void SetSigmaColor(float v) { params_.sigma_color = v; }
  void SetTopK(int k) { params_.top_k = k; }
  void SetUseCensus(bool v) { params_.use_census = v; }
  void SetSmoothWeight(float w) { params_.smooth_weight = w; }
  void SetEdgeWeight(float w) { params_.edge_weight = w; }
  void SetEscapeDepthRatio(float r) { params_.escape_depth_ratio = r; }
  void SetCenterColorWeight(float w) { params_.center_color_weight = w; }
  void SetVarianceGate(float v) { params_.variance_gate = v; }
  void SetAnchorViews(int n) { params_.anchor_views = n; }
  void SetFarGradientThreshold(float t) { params_.far_gradient_threshold = t; }
  void SetSegmentationEnabled(bool v) { params_.segmentation_enabled = v; }
  void SetSLICGridStep(int v) { params_.slic_grid_step = v; }
  void SetSLICCompactness(float v) { params_.slic_compactness = v; }
  void SetDebugDir(const std::string& dir) { params_.debug_dir = dir; }
  void SetDebugShotId(const std::string& id) { params_.debug_shot_id = id; }
  void SetCheckerboardFilter(bool v) { params_.checkerboard_filter = v; }
  void SetSpeckleMinSize(int v) { params_.speckle_min_size = v; }
  void SetGapMaxSize(int v) { params_.gap_max_size = v; }
  void SetGeomConsistencyWeight(float w) {
    cluster_.SetGeomConsistencyWeight(w);
  }
  void SetDevice(int idx) { cluster_.SetDevice(idx); }

  int BeginRefView(const Mat3d& K, const Mat3d& R, const Vec3d& t,
                   Eigen::Ref<const ImageU8> image, float depth_min,
                   float depth_max) {
    return cluster_.BeginRefView(K, R, t, image, depth_min, depth_max);
  }

  void AddSourceView(const Mat3d& K, const Mat3d& R, const Vec3d& t,
                     Eigen::Ref<const ImageU8> image) {
    cluster_.AddSourceView(K, R, t, image);
  }

  void SetSfMPoints(Eigen::Ref<const PointsMatrix> points) {
    cluster_.SetSfMPoints(points);
  }

  void AddGeomLink(int ref_idx, int source_view_idx, int from_ref_idx) {
    cluster_.AddGeomLink(ref_idx, source_view_idx, from_ref_idx);
  }

  py::list Run() {
    cluster_.SetParams(params_);
    std::vector<DepthmapResult> results;
    {
      py::gil_scoped_release release;
      cluster_.Run(&results);
    }
    py::list retn;
    for (auto& r : results) {
      const int h = static_cast<int>(r.depth.rows());
      const int w = static_cast<int>(r.depth.cols());
      py::list entry;
      entry.append(foundation::py_array_from_data(r.depth.data(), h, w));
      entry.append(foundation::py_array_from_data(r.normal.data(), h, w, 3));
      entry.append(foundation::py_array_from_data(r.cost.data(), h, w));
      entry.append(foundation::py_array_from_data(r.confidence.data(), h, w));
      retn.append(entry);
    }
    return retn;
  }

  void Clear() { cluster_.Clear(); }

  static bool IsAvailable() {
    return opencl::CLContext::Instance().IsAvailable();
  }

  static size_t GpuMemoryBytes() {
    return DepthmapClusterEstimator::GpuMemoryBytes();
  }

  static int NumDevices() { return opencl::CLContext::Instance().NumDevices(); }

  static std::string DeviceName(int idx) {
    return opencl::CLContext::Instance().Device(idx).name();
  }

  static size_t DeviceMemoryBytes(int idx) {
    return opencl::CLContext::Instance().Device(idx).GlobalMemSize();
  }

  static bool DeviceIsGPU(int idx) {
    return opencl::CLContext::Instance().Device(idx).IsGPU();
  }

  static size_t DeviceAvailableMemory(int idx) {
    return opencl::CLContext::Instance().Device(idx).AvailableMemory();
  }

  static void ReserveDeviceMemory(int idx, size_t bytes) {
    opencl::CLContext::Instance().Device(idx).ReserveMemory(bytes);
  }

  static void ReleaseDeviceMemory(int idx, size_t bytes) {
    opencl::CLContext::Instance().Device(idx).ReleaseMemory(bytes);
  }

 private:
  DepthmapClusterEstimator cluster_;
  DepthmapParams params_;
};

class DepthmapCleanerWrapper {
 public:
  void SetSameDepthThreshold(float t) { cleaner_.SetSameDepthThreshold(t); }
  void SetMinConsistentViews(int n) { cleaner_.SetMinConsistentViews(n); }
  void SetDevice(int idx) { cleaner_.SetDevice(idx); }
  void SetCarvingThreshold(float t) { cleaner_.SetCarvingThreshold(t); }
  void SetMaxCarvedViews(int n) { cleaner_.SetMaxCarvedViews(n); }
  void SetGrazingCosThreshold(float t) { cleaner_.SetGrazingCosThreshold(t); }
  void SetEdgeDepthRatio(float r) { cleaner_.SetEdgeDepthRatio(r); }

  void AddView(const Mat3d& K, const Mat3d& R, const Vec3d& t,
               Eigen::Ref<const ImageF> depth) {
    cleaner_.AddView(K, R, t, depth);
  }

  void AddViewWithNormal(const Mat3d& K, const Mat3d& R, const Vec3d& t,
                         Eigen::Ref<const ImageF> depth,
                         Eigen::Ref<const PixelData3f> normal) {
    cleaner_.AddView(K, R, t, depth, normal);
  }

  void Clear() { cleaner_.Clear(); }

  foundation::pyarray_f Clean(int ref_idx,
                              const py::array_t<int>& neighbor_ids) {
    std::vector<int> nbrs(neighbor_ids.data(),
                          neighbor_ids.data() + neighbor_ids.size());
    cv::Mat cleaned;
    {
      py::gil_scoped_release release;
      cleaned = cleaner_.Clean(ref_idx, nbrs);
    }
    return foundation::py_array_from_data(cleaned.ptr<float>(0), cleaned.rows,
                                          cleaned.cols);
  }

  foundation::pyarray_int ComputeSLIC(Eigen::Ref<const ImageU8> gray,
                                      int grid_step, float compactness) {
    cv::Mat gray_cv(static_cast<int>(gray.rows()),
                    static_cast<int>(gray.cols()), CV_8U,
                    const_cast<uint8_t*>(gray.data()));
    cv::Mat labels;
    {
      py::gil_scoped_release release;
      labels = cleaner_.ComputeSLIC(gray_cv, grid_step, compactness);
    }
    return foundation::py_array_from_data(labels.ptr<int>(0), labels.rows,
                                          labels.cols);
  }

  foundation::pyarray_f FilterMahalanobis(Eigen::Ref<const ImageF> depth,
                                          const Mat3d& K, float mahal_threshold,
                                          int window_radius) {
    cv::Mat depth_cv(static_cast<int>(depth.rows()),
                     static_cast<int>(depth.cols()), CV_32F,
                     const_cast<float*>(depth.data()));
    cv::Mat result;
    {
      py::gil_scoped_release release;
      result = cleaner_.FilterMahalanobis(depth_cv, K, mahal_threshold,
                                          window_radius);
    }
    return foundation::py_array_from_data(result.ptr<float>(0), result.rows,
                                          result.cols);
  }

  static bool IsAvailable() {
    return opencl::CLContext::Instance().IsAvailable();
  }

 private:
  GPUDepthmapCleaner cleaner_;
};

}  // namespace dense

#else  // !OPENSFM_HAVE_OPENCL

#include <dense/fuser.h>
#include <pybind11/eigen.h>

namespace dense {

// Same type alias as in the OpenCL-enabled section above.
using PointsMatrix = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;

/// Stub DepthmapClusterEstimator when OpenCL is not compiled in.
class DepthmapClusterEstimatorWrapper {
 public:
  void SetMaxIterations(int) {}
  void SetPatchSize(int) {}
  void SetMaxImageSize(int) {}
  void SetHierarchyLevels(int) {}
  void SetSigmaSpatial(float) {}
  void SetSigmaColor(float) {}
  void SetTopK(int) {}
  void SetUseCensus(bool) {}
  void SetSmoothWeight(float) {}
  void SetEdgeWeight(float) {}
  void SetEscapeDepthRatio(float) {}
  void SetCenterColorWeight(float) {}
  void SetVarianceGate(float) {}
  void SetAnchorViews(int) {}
  void SetFarGradientThreshold(float) {}
  void SetSegmentationEnabled(bool) {}
  void SetSLICGridStep(int) {}
  void SetSLICCompactness(float) {}
  void SetDebugDir(const std::string&) {}
  void SetDebugShotId(const std::string&) {}
  void SetCheckerboardFilter(bool) {}
  void SetSpeckleMinSize(int) {}
  void SetGapMaxSize(int) {}
  void SetGeomConsistencyWeight(float) {}
  void SetDevice(int) {}
  int BeginRefView(const Mat3d&, const Mat3d&, const Vec3d&,
                   Eigen::Ref<const ImageU8>, float, float) {
    throw std::runtime_error("PatchMatch: OpenCL not available");
  }
  void AddSourceView(const Mat3d&, const Mat3d&, const Vec3d&,
                     Eigen::Ref<const ImageU8>) {
    throw std::runtime_error("PatchMatch: OpenCL not available");
  }
  void SetSfMPoints(Eigen::Ref<const PointsMatrix>) {}
  void AddGeomLink(int, int, int) {}
  py::list Run() {
    throw std::runtime_error("PatchMatch: OpenCL not available");
  }
  void Clear() {}
  static bool IsAvailable() { return false; }
  static size_t GpuMemoryBytes() { return 0; }
  static int NumDevices() { return 0; }
  static std::string DeviceName(int) { return ""; }
  static size_t DeviceMemoryBytes(int) { return 0; }
  static bool DeviceIsGPU(int) { return false; }
  static size_t DeviceAvailableMemory(int) { return 0; }
  static void ReserveDeviceMemory(int, size_t) {}
  static void ReleaseDeviceMemory(int, size_t) {}
};

/// Stub GPUDepthmapCleaner when OpenCL is not compiled in.
class DepthmapCleanerWrapper {
 public:
  void SetSameDepthThreshold(float) {}
  void SetMinConsistentViews(int) {}
  void SetDevice(int) {}
  void SetCarvingThreshold(float) {}
  void SetMaxCarvedViews(int) {}
  void SetGrazingCosThreshold(float) {}
  void SetEdgeDepthRatio(float) {}
  void AddView(const Mat3d&, const Mat3d&, const Vec3d&,
               Eigen::Ref<const ImageF>) {
    throw std::runtime_error("GPUDepthmapCleaner: OpenCL not available");
  }
  void AddViewWithNormal(const Mat3d&, const Mat3d&, const Vec3d&,
                         Eigen::Ref<const ImageF>,
                         Eigen::Ref<const PixelData3f>) {
    throw std::runtime_error("GPUDepthmapCleaner: OpenCL not available");
  }
  foundation::pyarray_f Clean(int, const py::array_t<int>&) {
    throw std::runtime_error("GPUDepthmapCleaner: OpenCL not available");
  }
  foundation::pyarray_int ComputeSLIC(Eigen::Ref<const ImageF>, int, float) {
    throw std::runtime_error("GPUDepthmapCleaner: OpenCL not available");
  }
  foundation::pyarray_f FilterMahalanobis(Eigen::Ref<const ImageF>,
                                          const Mat3d&, float, int) {
    throw std::runtime_error("GPUDepthmapCleaner: OpenCL not available");
  }
  void Clear() {}
  static bool IsAvailable() { return false; }
};

}  // namespace dense

#endif  // OPENSFM_HAVE_OPENCL

#include <dense/svo_fuser.h>
#include <pybind11/eigen.h>

namespace dense {

// ---- SVOFuser wrapper ----

class SVOFuserWrapper {
 public:
  void SetVoxelSize(float size) { sf_.SetVoxelSize(size); }
  void SetTruncFactor(float factor) { sf_.SetTruncFactor(factor); }
  void SetMinWeight(float w) { sf_.SetMinWeight(w); }
  void SetDevice(int device_idx) { sf_.SetDevice(device_idx); }
  void SetNumLevels(int n) { sf_.SetNumLevels(n); }
  void SetDecimateFat(int n) { sf_.SetDecimateFat(static_cast<uint32_t>(n)); }
  void SetEdgeThreshold(float t) { sf_.SetEdgeThreshold(t); }
  void SetMinCount(int n) { sf_.SetMinCount(n); }
  void SetRelativeMinWeight(float w) { sf_.SetRelativeMinWeight(w); }
  void SetBBox(const Vec3f& min_world, const Vec3f& max_world) {
    sf_.SetBBox(min_world, max_world);
  }
  static bool IsGPUAvailable() { return SVOFuser::IsGPUAvailable(); }

  uint32_t CountVoxels() {
    py::gil_scoped_release release;
    return sf_.CountVoxels();
  }

  void AddView(const Mat3d& K, const Mat3d& R, const Vec3d& t,
               Eigen::Ref<const ImageF> depth,
               const py::array_t<float, py::array::c_style>& normal,
               const py::array_t<uint8_t, py::array::c_style>& color,
               Eigen::Ref<const ImageU8> mask,
               const py::object& confidence = py::none(),
               const std::string& name = "") {
    const int h = static_cast<int>(depth.rows());
    const int w = static_cast<int>(depth.cols());
    if (normal.ndim() != 3 || normal.shape(0) != h || normal.shape(1) != w) {
      throw std::invalid_argument(
          "depth and normal must have matching shapes.");
    }
    if (color.ndim() != 3 || color.shape(0) != h || color.shape(1) != w) {
      throw std::invalid_argument("depth and color must have matching shapes.");
    }
    Eigen::Map<const PixelData3f> n_e(normal.data(), 3, h * w);
    Eigen::Map<const PixelData3u8> c_e(color.data(), 3, h * w);

    ImageF weight;
    if (!confidence.is_none()) {
      auto conf_arr = confidence.cast<py::array_t<float, py::array::c_style>>();
      if (conf_arr.ndim() == 2 && conf_arr.shape(0) == h &&
          conf_arr.shape(1) == w) {
        weight = Eigen::Map<const ImageF>(conf_arr.data(), h, w);
      }
    }
    sf_.AddView(K, R, t, depth, n_e, c_e, mask, weight, name);
  }

  py::list Fuse() {
    std::vector<Vec3f> points;
    std::vector<Vec3f> normals;
    std::vector<Vec3<uint8_t>> colors;

    {
      py::gil_scoped_release release;
      sf_.Fuse(&points, &normals, &colors);
    }

    const int n = static_cast<int>(points.size());
    py::list retn;
    retn.append(foundation::py_array_from_data(points.data()->data(), n, 3));
    retn.append(foundation::py_array_from_data(normals.data()->data(), n, 3));
    retn.append(foundation::py_array_from_data(colors.data()->data(), n, 3));
    return retn;
  }

  // Split API: Fuse (no extract), Refine, ExtractPoints.
  void FuseOnly() {
    py::gil_scoped_release release;
    sf_.Fuse();
  }

  void RefineGeometry(int iters, float lambda_reg) {
    py::gil_scoped_release release;
    sf_.RefineGeometry(iters, lambda_reg);
  }

  // Extract surface points and bake colors from images in one call.
  // Requires PrepareRefinement still active (images on GPU).
  py::list ExtractAndBake() {
    std::vector<Vec3f> points;
    std::vector<Vec3f> normals;
    std::vector<Vec3<uint8_t>> colors;

    {
      py::gil_scoped_release release;
      sf_.ExtractPoints(&points, &normals, &colors);
      sf_.BakeColors(points, normals, &colors);
    }

    const int n = static_cast<int>(points.size());
    py::list retn;
    retn.append(foundation::py_array_from_data(points.data()->data(), n, 3));
    retn.append(foundation::py_array_from_data(normals.data()->data(), n, 3));
    retn.append(foundation::py_array_from_data(colors.data()->data(), n, 3));
    return retn;
  }

  void PruneByVisibility(int iterations, float carve_margin,
                         int carve_threshold, int support_min) {
    py::gil_scoped_release release;
    sf_.PruneByVisibility(iterations, carve_margin, carve_threshold,
                          support_min);
  }

  py::list ExtractPoints() {
    std::vector<Vec3f> points;
    std::vector<Vec3f> normals;
    std::vector<Vec3<uint8_t>> colors;

    {
      py::gil_scoped_release release;
      sf_.ExtractPoints(&points, &normals, &colors);
    }

    const int n = static_cast<int>(points.size());
    py::list retn;
    retn.append(foundation::py_array_from_data(points.data()->data(), n, 3));
    retn.append(foundation::py_array_from_data(normals.data()->data(), n, 3));
    retn.append(foundation::py_array_from_data(colors.data()->data(), n, 3));
    return retn;
  }

 private:
  SVOFuser sf_;
};

// ---- DSM rasterizer wrapper (streaming mode-seeking + bilateral) ----

#ifdef OPENSFM_HAVE_OPENCL

class DSMRasterizerWrapper {
 public:
  void SetGSD(float gsd) { gsd_ = gsd; }
  void SetBBox(const Eigen::Vector2f& min_xy, const Eigen::Vector2f& max_xy) {
    origin_x_ = min_xy.x();
    origin_y_ = min_xy.y();
    max_x_ = max_xy.x();
    max_y_ = max_xy.y();
    has_bbox_ = true;
  }
  void SetDevice(int idx) { device_idx_ = idx; }
  void SetModeThreshold(float meters) { mode_threshold_ = meters; }
  void SetMinCount(int n) { min_count_ = n; }
  void SetMinNormalZ(float hard_gate, float soft_upper) {
    min_normal_z_ = hard_gate;
    soft_upper_nz_ = soft_upper;
  }
  void SetBilateral(bool enabled, int radius, float range_sigma) {
    bilateral_enabled_ = enabled;
    bilateral_radius_ = radius;
    bilateral_range_ = range_sigma;
  }

  // Allocate GPU grids and prepare for scatter.
  void Begin() {
    if (!has_bbox_ || gsd_ <= 0.0f) {
      throw std::runtime_error(
          "DSMRasterizer: must set GSD and bbox before Begin()");
    }

    grid_w_ = static_cast<int>(std::ceil((max_x_ - origin_x_) / gsd_));
    grid_h_ = static_cast<int>(std::ceil((max_y_ - origin_y_) / gsd_));
    if (grid_w_ <= 0 || grid_h_ <= 0) {
      throw std::runtime_error("DSMRasterizer: degenerate grid size");
    }

    ncells_ = static_cast<size_t>(grid_h_) * grid_w_;

    auto& dev = opencl::CLContext::Instance().Device(device_idx_);
    prog_ = dev.GetOrBuildProgram("dsm_mode_v2", kDSMKernelSource);

    auto& ctx = dev.context();
    auto& q = dev.queue();

    // N_MODES=3, K_BUF=10
    cl_mode_z_ =
        cl::Buffer(ctx, CL_MEM_READ_WRITE, 3 * ncells_ * sizeof(int32_t));
    cl_mode_weight_ =
        cl::Buffer(ctx, CL_MEM_READ_WRITE, 3 * ncells_ * sizeof(int32_t));
    cl_mode_count_ =
        cl::Buffer(ctx, CL_MEM_READ_WRITE, 3 * ncells_ * sizeof(int32_t));
    cl_buf_z_ =
        cl::Buffer(ctx, CL_MEM_READ_WRITE, 10 * ncells_ * sizeof(int32_t));
    cl_buf_w_ =
        cl::Buffer(ctx, CL_MEM_READ_WRITE, 10 * ncells_ * sizeof(int32_t));
    cl_buf_pos_ = cl::Buffer(ctx, CL_MEM_READ_WRITE, ncells_ * sizeof(int32_t));
    cl_out_a_ = cl::Buffer(ctx, CL_MEM_READ_WRITE, ncells_ * sizeof(float));
    cl_out_b_ = cl::Buffer(ctx, CL_MEM_READ_WRITE, ncells_ * sizeof(float));
    cl_valid_ = cl::Buffer(ctx, CL_MEM_READ_WRITE, ncells_ * sizeof(uint8_t));
    cl_guide_ = cl::Buffer(ctx, CL_MEM_READ_WRITE, ncells_ * sizeof(float));

    ClearAll();
  }

  // Scatter one depthmap view into the mode grid (confidence-weighted).
  void Scatter(const Mat3d& K, const Mat3d& R, const Vec3d& t,
               Eigen::Ref<const ImageF> depth,
               const py::array_t<float, py::array::c_style>& normal,
               Eigen::Ref<const ImageF> confidence) {
    const int h = static_cast<int>(depth.rows());
    const int w = static_cast<int>(depth.cols());
    const int npx = h * w;

    if (normal.ndim() != 3 || normal.shape(0) != h || normal.shape(1) != w ||
        normal.shape(2) != 3) {
      throw std::invalid_argument("normal must be (H, W, 3) float32");
    }

    // Pack camera params: K_inv(9) + R_inv(9) + t(3) = 21 floats row-major
    Eigen::Matrix3f Kinvf = K.cast<float>().inverse();
    Eigen::Matrix3f Rinvf = R.cast<float>().transpose();
    Eigen::Vector3f tf = t.cast<float>();

    float cam[21];
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        cam[i * 3 + j] = Kinvf(i, j);
        cam[9 + i * 3 + j] = Rinvf(i, j);
      }
    }
    cam[18] = tf.x();
    cam[19] = tf.y();
    cam[20] = tf.z();

    auto& dev = opencl::CLContext::Instance().Device(device_idx_);
    auto& ctx = dev.context();
    auto& q = dev.queue();

    // Upload view data
    cl::Buffer cl_depth(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        npx * sizeof(float), const_cast<float*>(depth.data()));
    cl::Buffer cl_normal(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                         npx * 3 * sizeof(float),
                         const_cast<float*>(normal.data()));
    cl::Buffer cl_conf(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       npx * sizeof(float),
                       const_cast<float*>(confidence.data()));
    cl::Buffer cl_cam(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                      21 * sizeof(float), cam);

    const float inv_gsd = 1.0f / gsd_;
    const int ncells_i = static_cast<int>(ncells_);
    const int fp_threshold = static_cast<int>(mode_threshold_ * 8192.0f);
    const size_t global = ((static_cast<size_t>(npx) + 255) / 256) * 256;

    cl::Kernel k(prog_, "dsm_scatter");
    k.setArg(0, cl_depth);
    k.setArg(1, cl_normal);
    k.setArg(2, cl_conf);
    k.setArg(3, cl_cam);
    k.setArg(4, cl_mode_z_);
    k.setArg(5, cl_mode_weight_);
    k.setArg(6, cl_mode_count_);
    k.setArg(7, cl_buf_z_);
    k.setArg(8, cl_buf_w_);
    k.setArg(9, cl_buf_pos_);
    k.setArg(10, origin_x_);
    k.setArg(11, origin_y_);
    k.setArg(12, inv_gsd);
    k.setArg(13, grid_w_);
    k.setArg(14, grid_h_);
    k.setArg(15, w);
    k.setArg(16, h);
    k.setArg(17, ncells_i);
    k.setArg(18, fp_threshold);
    k.setArg(19, min_normal_z_);
    k.setArg(20, soft_upper_nz_);

    py::gil_scoped_release release;
    q.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(global),
                           cl::NDRange(256));
    q.finish();
  }

  // Analyse ring buffers and promote clusters to mode slots.
  void UpdateModes() {
    auto& dev = opencl::CLContext::Instance().Device(device_idx_);
    auto& q = dev.queue();
    const int ncells_i = static_cast<int>(ncells_);
    const int fp_threshold = static_cast<int>(mode_threshold_ * 8192.0f);
    const int min_buf_samples = 3;
    const size_t global = ((ncells_ + 255) / 256) * 256;

    cl::Kernel k(prog_, "dsm_update_modes");
    k.setArg(0, cl_mode_z_);
    k.setArg(1, cl_mode_weight_);
    k.setArg(2, cl_mode_count_);
    k.setArg(3, cl_buf_z_);
    k.setArg(4, cl_buf_w_);
    k.setArg(5, cl_buf_pos_);
    k.setArg(6, ncells_i);
    k.setArg(7, fp_threshold);
    k.setArg(8, min_buf_samples);

    py::gil_scoped_release release;
    q.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(global),
                           cl::NDRange(256));
    q.finish();
  }

  // Finalize: pick highest mode → out_z + validity mask.
  // Does NOT apply bilateral (that's done separately after diffusion).
  foundation::pyarray_f Finish() {
    auto& dev = opencl::CLContext::Instance().Device(device_idx_);
    auto& q = dev.queue();
    const int ncells_i = static_cast<int>(ncells_);
    const size_t global1d = ((ncells_ + 255) / 256) * 256;

    // Finalize → cl_out_a_ + cl_valid_
    cl::Kernel k_fin(prog_, "dsm_finalize");
    k_fin.setArg(0, cl_mode_z_);
    k_fin.setArg(1, cl_mode_weight_);
    k_fin.setArg(2, cl_mode_count_);
    k_fin.setArg(3, cl_out_a_);
    k_fin.setArg(4, cl_valid_);
    k_fin.setArg(5, ncells_i);
    k_fin.setArg(6, min_count_);
    {
      py::gil_scoped_release release;
      q.enqueueNDRangeKernel(k_fin, cl::NullRange, cl::NDRange(global1d),
                             cl::NDRange(256));
      q.finish();
    }

    // Download grid
    const size_t float_bytes = ncells_ * sizeof(float);
    std::vector<float> host_grid(ncells_);
    q.enqueueReadBuffer(cl_out_a_, CL_TRUE, 0, float_bytes, host_grid.data());

    return foundation::py_array_from_data(host_grid.data(), grid_h_, grid_w_);
  }

  // Get the validity mask (1 = observed, 0 = hole). Call after Finish().
  py::array_t<uint8_t> GetValidityMask() {
    auto& dev = opencl::CLContext::Instance().Device(device_idx_);
    auto& q = dev.queue();
    std::vector<uint8_t> host_mask(ncells_);
    q.enqueueReadBuffer(cl_valid_, CL_TRUE, 0, ncells_, host_mask.data());
    auto result = py::array_t<uint8_t>({grid_h_, grid_w_});
    std::memcpy(result.mutable_data(), host_mask.data(), ncells_);
    return result;
  }

  // Run Perona-Malik diffusion on the current grid (cl_out_a_).
  // guide_np: gradient magnitude array (same size as grid), or empty for
  //           self-guided mode (computes gradient from current grid).
  // Returns the diffused grid.
  foundation::pyarray_f Diffuse(
      const py::array_t<float, py::array::c_style>& guide_np, int iterations,
      float kappa, float dt) {
    auto& dev = opencl::CLContext::Instance().Device(device_idx_);
    auto& q = dev.queue();
    const size_t gx = ((static_cast<size_t>(grid_w_) + 15) / 16) * 16;
    const size_t gy = ((static_cast<size_t>(grid_h_) + 15) / 16) * 16;

    // Upload or compute guide
    if (guide_np.size() > 0) {
      if (guide_np.size() != static_cast<ssize_t>(ncells_)) {
        throw std::invalid_argument(
            "guide must have same number of cells as grid");
      }
      q.enqueueWriteBuffer(cl_guide_, CL_TRUE, 0, ncells_ * sizeof(float),
                           const_cast<float*>(guide_np.data()));
    } else {
      // Self-guided: compute gradient magnitude from cl_out_a_
      cl::Kernel k_grad(prog_, "dsm_gradient_magnitude");
      k_grad.setArg(0, cl_out_a_);
      k_grad.setArg(1, cl_guide_);
      k_grad.setArg(2, grid_w_);
      k_grad.setArg(3, grid_h_);
      q.enqueueNDRangeKernel(k_grad, cl::NullRange, cl::NDRange(gx, gy),
                             cl::NDRange(16, 16));
      q.finish();
    }

    // Ping-pong diffusion: cl_out_a_ ↔ cl_out_b_
    cl::Buffer* src = &cl_out_a_;
    cl::Buffer* dst = &cl_out_b_;

    {
      py::gil_scoped_release release;
      for (int iter = 0; iter < iterations; ++iter) {
        cl::Kernel k_diff(prog_, "dsm_diffuse");
        k_diff.setArg(0, *src);
        k_diff.setArg(1, *dst);
        k_diff.setArg(2, cl_guide_);
        k_diff.setArg(3, cl_valid_);
        k_diff.setArg(4, grid_w_);
        k_diff.setArg(5, grid_h_);
        k_diff.setArg(6, kappa);
        k_diff.setArg(7, dt);
        q.enqueueNDRangeKernel(k_diff, cl::NullRange, cl::NDRange(gx, gy),
                               cl::NDRange(16, 16));
        q.finish();
        std::swap(src, dst);
      }
    }

    // Result is in *src after the last swap.
    // Copy back to cl_out_a_ if needed.
    if (src != &cl_out_a_) {
      q.enqueueCopyBuffer(*src, cl_out_a_, 0, 0, ncells_ * sizeof(float));
      q.finish();
    }

    // Download
    std::vector<float> host_grid(ncells_);
    q.enqueueReadBuffer(cl_out_a_, CL_TRUE, 0, ncells_ * sizeof(float),
                        host_grid.data());
    return foundation::py_array_from_data(host_grid.data(), grid_h_, grid_w_);
  }

  // Compute gradient magnitude of the current grid on GPU.
  // Call after Finish() or Diffuse(). Returns float32 array.
  foundation::pyarray_f ComputeGradient() {
    auto& dev = opencl::CLContext::Instance().Device(device_idx_);
    auto& q = dev.queue();
    const size_t gx = ((static_cast<size_t>(grid_w_) + 15) / 16) * 16;
    const size_t gy = ((static_cast<size_t>(grid_h_) + 15) / 16) * 16;

    cl::Kernel k_grad(prog_, "dsm_gradient_magnitude");
    k_grad.setArg(0, cl_out_a_);
    k_grad.setArg(1, cl_guide_);
    k_grad.setArg(2, grid_w_);
    k_grad.setArg(3, grid_h_);
    {
      py::gil_scoped_release release;
      q.enqueueNDRangeKernel(k_grad, cl::NullRange, cl::NDRange(gx, gy),
                             cl::NDRange(16, 16));
      q.finish();
    }

    std::vector<float> host_grad(ncells_);
    q.enqueueReadBuffer(cl_guide_, CL_TRUE, 0, ncells_ * sizeof(float),
                        host_grad.data());
    return foundation::py_array_from_data(host_grad.data(), grid_h_, grid_w_);
  }

  // Upsample a coarse grid by factor 2 into the current fine grid on GPU.
  // coarse_np must be (coarse_h, coarse_w). Result stored in cl_out_b_
  // and returned. Does not affect cl_out_a_ (the finalized grid).
  foundation::pyarray_f UpsampleNN(
      const py::array_t<float, py::array::c_style>& coarse_np, int coarse_w,
      int coarse_h) {
    auto& dev = opencl::CLContext::Instance().Device(device_idx_);
    auto& ctx = dev.context();
    auto& q = dev.queue();

    const size_t coarse_cells = static_cast<size_t>(coarse_w) * coarse_h;
    cl::Buffer cl_coarse(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                         coarse_cells * sizeof(float),
                         const_cast<float*>(coarse_np.data()));

    const size_t gx = ((static_cast<size_t>(grid_w_) + 15) / 16) * 16;
    const size_t gy = ((static_cast<size_t>(grid_h_) + 15) / 16) * 16;

    cl::Kernel k_up(prog_, "dsm_upsample_nn");
    k_up.setArg(0, cl_coarse);
    k_up.setArg(1, cl_out_b_);
    k_up.setArg(2, coarse_w);
    k_up.setArg(3, coarse_h);
    k_up.setArg(4, grid_w_);
    k_up.setArg(5, grid_h_);
    {
      py::gil_scoped_release release;
      q.enqueueNDRangeKernel(k_up, cl::NullRange, cl::NDRange(gx, gy),
                             cl::NDRange(16, 16));
      q.finish();
    }

    std::vector<float> host_grid(ncells_);
    q.enqueueReadBuffer(cl_out_b_, CL_TRUE, 0, ncells_ * sizeof(float),
                        host_grid.data());
    return foundation::py_array_from_data(host_grid.data(), grid_h_, grid_w_);
  }

  // Apply bilateral filter on cl_out_a_ → result.
  foundation::pyarray_f ApplyBilateral() {
    auto& dev = opencl::CLContext::Instance().Device(device_idx_);
    auto& q = dev.queue();

    if (!bilateral_enabled_ || bilateral_radius_ <= 0) {
      // Just download cl_out_a_ as-is
      std::vector<float> host_grid(ncells_);
      q.enqueueReadBuffer(cl_out_a_, CL_TRUE, 0, ncells_ * sizeof(float),
                          host_grid.data());
      return foundation::py_array_from_data(host_grid.data(), grid_h_, grid_w_);
    }

    cl::Kernel k_bi(prog_, "dsm_bilateral");
    k_bi.setArg(0, cl_out_a_);
    k_bi.setArg(1, cl_out_b_);
    k_bi.setArg(2, grid_w_);
    k_bi.setArg(3, grid_h_);
    k_bi.setArg(4, bilateral_radius_);
    const float sigma_s = static_cast<float>(bilateral_radius_) / 2.0f;
    const float inv_2ss = 1.0f / (2.0f * sigma_s * sigma_s);
    const float inv_2sr = 1.0f / (2.0f * bilateral_range_ * bilateral_range_);
    k_bi.setArg(5, inv_2ss);
    k_bi.setArg(6, inv_2sr);

    const size_t gx = ((static_cast<size_t>(grid_w_) + 15) / 16) * 16;
    const size_t gy = ((static_cast<size_t>(grid_h_) + 15) / 16) * 16;
    {
      py::gil_scoped_release release;
      q.enqueueNDRangeKernel(k_bi, cl::NullRange, cl::NDRange(gx, gy),
                             cl::NDRange(16, 16));
      q.finish();
    }

    std::vector<float> host_grid(ncells_);
    q.enqueueReadBuffer(cl_out_b_, CL_TRUE, 0, ncells_ * sizeof(float),
                        host_grid.data());
    return foundation::py_array_from_data(host_grid.data(), grid_h_, grid_w_);
  }

  int grid_width() const { return grid_w_; }
  int grid_height() const { return grid_h_; }

  static bool IsAvailable() {
    return opencl::CLContext::Instance().IsAvailable();
  }

 private:
  void ClearAll() {
    auto& dev = opencl::CLContext::Instance().Device(device_idx_);
    auto& q = dev.queue();
    const int ncells_i = static_cast<int>(ncells_);
    cl::Kernel k(prog_, "dsm_clear");
    k.setArg(0, cl_mode_z_);
    k.setArg(1, cl_mode_weight_);
    k.setArg(2, cl_mode_count_);
    k.setArg(3, cl_buf_z_);
    k.setArg(4, cl_buf_w_);
    k.setArg(5, cl_buf_pos_);
    k.setArg(6, ncells_i);
    const size_t global = ((ncells_ + 255) / 256) * 256;
    q.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(global),
                           cl::NDRange(256));
    q.finish();
  }

  float gsd_ = 0.0f;
  float origin_x_ = 0.0f;
  float origin_y_ = 0.0f;
  float max_x_ = 0.0f;
  float max_y_ = 0.0f;
  bool has_bbox_ = false;
  int device_idx_ = 0;
  float mode_threshold_ = 1.0f;
  int min_count_ = 3;
  float min_normal_z_ = 0.2f;
  float soft_upper_nz_ = 0.7f;
  int bilateral_radius_ = 2;
  float bilateral_range_ = 0.3f;
  bool bilateral_enabled_ = true;

  int grid_w_ = 0;
  int grid_h_ = 0;
  size_t ncells_ = 0;
  cl::Program prog_;
  cl::Buffer cl_mode_z_;
  cl::Buffer cl_mode_weight_;
  cl::Buffer cl_mode_count_;
  cl::Buffer cl_buf_z_;
  cl::Buffer cl_buf_w_;
  cl::Buffer cl_buf_pos_;
  cl::Buffer cl_out_a_;
  cl::Buffer cl_out_b_;
  cl::Buffer cl_valid_;
  cl::Buffer cl_guide_;
};

#else  // !OPENSFM_HAVE_OPENCL

class DSMRasterizerWrapper {
 public:
  void SetGSD(float) {}
  void SetBBox(const Eigen::Vector2f&, const Eigen::Vector2f&) {}
  void SetDevice(int) {}
  void SetModeThreshold(float) {}
  void SetMinCount(int) {}
  void SetMinNormalZ(float, float) {}
  void SetBilateral(bool, int, float) {}
  void Begin() {
    throw std::runtime_error("DSMRasterizer: OpenCL not available");
  }
  void Scatter(const Mat3d&, const Mat3d&, const Vec3d&,
               Eigen::Ref<const ImageF>,
               const py::array_t<float, py::array::c_style>&,
               Eigen::Ref<const ImageF>) {
    throw std::runtime_error("DSMRasterizer: OpenCL not available");
  }
  void UpdateModes() {
    throw std::runtime_error("DSMRasterizer: OpenCL not available");
  }
  foundation::pyarray_f Finish() {
    throw std::runtime_error("DSMRasterizer: OpenCL not available");
  }
  py::array_t<uint8_t> GetValidityMask() {
    throw std::runtime_error("DSMRasterizer: OpenCL not available");
  }
  foundation::pyarray_f Diffuse(const py::array_t<float, py::array::c_style>&,
                                int, float, float) {
    throw std::runtime_error("DSMRasterizer: OpenCL not available");
  }
  foundation::pyarray_f ComputeGradient() {
    throw std::runtime_error("DSMRasterizer: OpenCL not available");
  }
  foundation::pyarray_f UpsampleNN(
      const py::array_t<float, py::array::c_style>&, int, int) {
    throw std::runtime_error("DSMRasterizer: OpenCL not available");
  }
  foundation::pyarray_f ApplyBilateral() {
    throw std::runtime_error("DSMRasterizer: OpenCL not available");
  }
  int grid_width() const { return 0; }
  int grid_height() const { return 0; }
  static bool IsAvailable() { return false; }
};

#endif  // OPENSFM_HAVE_OPENCL

}  // namespace dense
