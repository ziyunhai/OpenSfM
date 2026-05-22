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

  void Refine(int color_iters, int joint_iters, float lambda_reg) {
    py::gil_scoped_release release;
    sf_.Refine(color_iters, joint_iters, lambda_reg);
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
    prog_ = dev.GetOrBuildProgram("dsm_mode_v1", kDSMKernelSource);

    auto& ctx = dev.context();

    // N_MODES=3, K_BUF=10
    cl_mode_z_ =
        cl::Buffer(ctx, CL_MEM_READ_WRITE, 3 * ncells_ * sizeof(int32_t));
    cl_mode_count_ =
        cl::Buffer(ctx, CL_MEM_READ_WRITE, 3 * ncells_ * sizeof(int32_t));
    cl_buf_z_ =
        cl::Buffer(ctx, CL_MEM_READ_WRITE, 10 * ncells_ * sizeof(int32_t));
    cl_buf_pos_ = cl::Buffer(ctx, CL_MEM_READ_WRITE, ncells_ * sizeof(int32_t));
    cl_out_a_ = cl::Buffer(ctx, CL_MEM_READ_WRITE, ncells_ * sizeof(float));
    cl_out_b_ = cl::Buffer(ctx, CL_MEM_READ_WRITE, ncells_ * sizeof(float));

    ClearAll();
  }

  // Scatter one depthmap view into the mode grid.
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
    const float min_normal_z = 0.35f;
    const size_t global = ((static_cast<size_t>(npx) + 255) / 256) * 256;

    cl::Kernel k(prog_, "dsm_scatter");
    k.setArg(0, cl_depth);
    k.setArg(1, cl_normal);
    k.setArg(2, cl_conf);
    k.setArg(3, cl_cam);
    k.setArg(4, cl_mode_z_);
    k.setArg(5, cl_mode_count_);
    k.setArg(6, cl_buf_z_);
    k.setArg(7, cl_buf_pos_);
    k.setArg(8, origin_x_);
    k.setArg(9, origin_y_);
    k.setArg(10, inv_gsd);
    k.setArg(11, grid_w_);
    k.setArg(12, grid_h_);
    k.setArg(13, w);
    k.setArg(14, h);
    k.setArg(15, ncells_i);
    k.setArg(16, fp_threshold);
    k.setArg(17, min_normal_z);

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
    k.setArg(1, cl_mode_count_);
    k.setArg(2, cl_buf_z_);
    k.setArg(3, cl_buf_pos_);
    k.setArg(4, ncells_i);
    k.setArg(5, fp_threshold);
    k.setArg(6, min_buf_samples);

    py::gil_scoped_release release;
    q.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(global),
                           cl::NDRange(256));
    q.finish();
  }

  // Finalize: pick highest mode, optional bilateral, download result.
  foundation::pyarray_f Finish() {
    auto& dev = opencl::CLContext::Instance().Device(device_idx_);
    auto& q = dev.queue();
    const int ncells_i = static_cast<int>(ncells_);
    const size_t global1d = ((ncells_ + 255) / 256) * 256;

    // Finalize → cl_out_a_
    cl::Kernel k_fin(prog_, "dsm_finalize");
    k_fin.setArg(0, cl_mode_z_);
    k_fin.setArg(1, cl_mode_count_);
    k_fin.setArg(2, cl_out_a_);
    k_fin.setArg(3, ncells_i);
    k_fin.setArg(4, min_count_);
    {
      py::gil_scoped_release release;
      q.enqueueNDRangeKernel(k_fin, cl::NullRange, cl::NDRange(global1d),
                             cl::NDRange(256));
      q.finish();
    }

    // Bilateral filter: cl_out_a_ → cl_out_b_
    cl::Buffer* result_buf = &cl_out_a_;
    if (bilateral_enabled_ && bilateral_radius_ > 0) {
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
      result_buf = &cl_out_b_;
    }

    // Download
    const size_t float_bytes = ncells_ * sizeof(float);
    std::vector<float> host_grid(ncells_);
    q.enqueueReadBuffer(*result_buf, CL_TRUE, 0, float_bytes, host_grid.data());

    return foundation::py_array_from_data(host_grid.data(), grid_h_, grid_w_);
  }

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
    k.setArg(1, cl_mode_count_);
    k.setArg(2, cl_buf_z_);
    k.setArg(3, cl_buf_pos_);
    k.setArg(4, ncells_i);
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
  int bilateral_radius_ = 2;
  float bilateral_range_ = 0.3f;
  bool bilateral_enabled_ = true;

  int grid_w_ = 0;
  int grid_h_ = 0;
  size_t ncells_ = 0;
  cl::Program prog_;
  cl::Buffer cl_mode_z_;
  cl::Buffer cl_mode_count_;
  cl::Buffer cl_buf_z_;
  cl::Buffer cl_buf_pos_;
  cl::Buffer cl_out_a_;
  cl::Buffer cl_out_b_;
};

#else  // !OPENSFM_HAVE_OPENCL

class DSMRasterizerWrapper {
 public:
  void SetGSD(float) {}
  void SetBBox(const Eigen::Vector2f&, const Eigen::Vector2f&) {}
  void SetDevice(int) {}
  void SetModeThreshold(float) {}
  void SetMinCount(int) {}
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
  static bool IsAvailable() { return false; }
};

#endif  // OPENSFM_HAVE_OPENCL

}  // namespace dense
