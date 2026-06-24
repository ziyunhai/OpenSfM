#pragma once

#include <foundation/python_types.h>

// Conditional compilation: if OpenCL is not available, provide a stub
// that raises a clear error at runtime.
#ifdef OPENSFM_HAVE_OPENCL

#include <dense/cleaner.h>
#include <dense/cluster.h>
#include <dense/diffusion_opencl_kernels.h>
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
  void SetAnchorViews(int n) { params_.anchor_views = n; }
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
  void SetAnchorViews(int) {}
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
  void SetDSMWallCullNz(float nz) { sf_.SetDSMWallCullNz(nz); }
  void SetBBox(const Vec3f& min_world, const Vec3f& max_world) {
    sf_.SetBBox(min_world, max_world);
  }
  static bool IsGPUAvailable() { return SVOFuser::IsGPUAvailable(); }

  // GPU hash-table capacity (slots) currently allocated; 0 if not yet fused.
  uint32_t Capacity() const { return sf_.Capacity(); }

  // Free refine images + grad/adam, keep the hash table (for Pass-2 reuse).
  void ReleaseRefineBuffers() {
    py::gil_scoped_release release;
    sf_.ReleaseRefineBuffers();
  }

  uint32_t CountVoxels() {
    py::gil_scoped_release release;
    return sf_.CountVoxels();
  }

  void AddView(
      const Mat3d& K, const Mat3d& R, const Vec3d& t,
      const py::array_t<float, py::array::c_style | py::array::forcecast>&
          depth,
      const py::array_t<float, py::array::c_style | py::array::forcecast>&
          normal,
      const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>&
          color,
      const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>&
          mask,
      const py::object& confidence = py::none(), const std::string& name = "") {
    if (depth.ndim() != 2) {
      throw std::invalid_argument("depth must be a 2-D array.");
    }
    const int h = static_cast<int>(depth.shape(0));
    const int w = static_cast<int>(depth.shape(1));
    if (normal.ndim() != 3 || normal.shape(0) != h || normal.shape(1) != w ||
        normal.shape(2) != 3) {
      throw std::invalid_argument(
          "depth and normal must have matching shapes.");
    }
    if (color.ndim() != 3 || color.shape(0) != h || color.shape(1) != w ||
        color.shape(2) != 3) {
      throw std::invalid_argument("depth and color must have matching shapes.");
    }
    const bool has_mask = mask.size() > 0;
    if (has_mask &&
        (mask.ndim() != 2 || mask.shape(0) != h || mask.shape(1) != w)) {
      throw std::invalid_argument("depth and mask must have matching shapes.");
    }

    // Borrow the pixel buffers instead of copying them into the fuser: it keeps
    // only non-owning Eigen maps over this memory.  Retain a reference to each
    // array so it outlives every count/fuse/refine/extract/bake/render call on
    // this fuser, even if the caller releases its own view cache first.  The
    // c_style|forcecast arrays are already contiguous in the expected dtype (no
    // copy); if a copy were ever needed, we retain that copy, so the mapped
    // pointer stays valid either way.  Absent optional buffers map to size 0.
    retained_.push_back(depth);
    retained_.push_back(normal);
    retained_.push_back(color);
    if (has_mask) {
      retained_.push_back(mask);
    }

    const float* weight_ptr = nullptr;
    if (!confidence.is_none()) {
      auto conf = confidence.cast<
          py::array_t<float, py::array::c_style | py::array::forcecast>>();
      if (conf.ndim() == 2 && conf.shape(0) == h && conf.shape(1) == w) {
        retained_.push_back(conf);
        weight_ptr = conf.data();
      }
    }

    const Eigen::Index npix = static_cast<Eigen::Index>(h) * w;
    sf_.AddView(K, R, t, Eigen::Map<const ImageF>(depth.data(), h, w),
                Eigen::Map<const PixelData3f>(normal.data(), 3, npix),
                Eigen::Map<const PixelData3u8>(color.data(), 3, npix),
                Eigen::Map<const ImageU8>(has_mask ? mask.data() : nullptr,
                                          has_mask ? h : 0, has_mask ? w : 0),
                Eigen::Map<const ImageF>(weight_ptr, weight_ptr ? h : 0,
                                         weight_ptr ? w : 0),
                name);
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

  void RefineGeometry(
      int iters, float lambda_reg,
      const std::map<std::string, std::vector<std::string>>& neighbors,
      float lambda_anchor, float early_stop_rel) {
    py::gil_scoped_release release;
    sf_.RefineGeometry(iters, lambda_reg, neighbors, lambda_anchor,
                       early_stop_rel);
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

  py::tuple RenderDSMOrtho(float origin_x, float origin_y, float gsd, int width,
                           int height, float z_min, float z_max) {
    std::vector<float> dsm;
    std::vector<uint8_t> ortho_rgba;
    std::vector<float> normals;

    {
      py::gil_scoped_release release;
      sf_.RenderDSMOrtho(origin_x, origin_y, gsd, width, height, z_min, z_max,
                         &dsm, &ortho_rgba, &normals);
    }

    const size_t ncells = static_cast<size_t>(width) * height;

    // DSM: float32 [height, width]
    py::array_t<float> dsm_arr({height, width});
    std::memcpy(dsm_arr.mutable_data(), dsm.data(), ncells * sizeof(float));

    // Ortho: unpack RGBA (uint32) → RGB (uint8 H×W×3)
    py::array_t<uint8_t> ortho_arr({height, width, 3});
    auto* dst = ortho_arr.mutable_data();
    const auto* src = reinterpret_cast<const uint32_t*>(ortho_rgba.data());
    for (size_t i = 0; i < ncells; ++i) {
      uint32_t rgba = src[i];
      dst[i * 3 + 0] = static_cast<uint8_t>(rgba & 0xFF);
      dst[i * 3 + 1] = static_cast<uint8_t>((rgba >> 8) & 0xFF);
      dst[i * 3 + 2] = static_cast<uint8_t>((rgba >> 16) & 0xFF);
    }

    // Normals: float32 [height, width, 3]
    py::array_t<float> nrm_arr({height, width, 3});
    std::memcpy(nrm_arr.mutable_data(), normals.data(),
                ncells * 3 * sizeof(float));

    return py::make_tuple(dsm_arr, ortho_arr, nrm_arr);
  }

  py::array_t<uint8_t> BakeColorsStandalone(
      const py::array_t<float, py::array::c_style>& points,
      const py::array_t<float, py::array::c_style>& normals, int n_final,
      int irls_iters, const py::object& relax_occlusion,
      const py::object& dsm_occ, float dsm_origin_x, float dsm_origin_y,
      float dsm_gsd, float dsm_max_z) {
    if (points.ndim() != 2 || points.shape(1) != 3) {
      throw std::invalid_argument("points must be (N, 3)");
    }
    if (normals.ndim() != 2 || normals.shape(1) != 3) {
      throw std::invalid_argument("normals must be (N, 3)");
    }
    const int n = static_cast<int>(points.shape(0));
    if (normals.shape(0) != n) {
      throw std::invalid_argument("points and normals must have same length");
    }

    // Copy into Eigen-mapped vectors expected by BakeColors.
    std::vector<Vec3f> pts(n);
    std::vector<Vec3f> nrm(n);
    std::memcpy(pts.data(), points.data(), n * 3 * sizeof(float));
    std::memcpy(nrm.data(), normals.data(), n * 3 * sizeof(float));

    // Optional per-point occlusion-relax flags (1 = skip occlusion test).
    std::vector<uint8_t> relax;
    if (!relax_occlusion.is_none()) {
      auto arr =
          relax_occlusion.cast<py::array_t<uint8_t, py::array::c_style>>();
      if (static_cast<int>(arr.size()) != n) {
        throw std::invalid_argument("relax_occlusion must have length N");
      }
      relax.resize(n);
      std::memcpy(relax.data(), arr.data(), n);
    }

    // Optional DSM heightfield (H, W) float for per-view horizon occlusion.
    std::vector<float> dsm;
    int dsm_h = 0, dsm_w = 0;
    if (!dsm_occ.is_none()) {
      auto arr = dsm_occ.cast<py::array_t<float, py::array::c_style>>();
      if (arr.ndim() != 2) {
        throw std::invalid_argument("dsm_occ must be a 2D (H, W) array");
      }
      dsm_h = static_cast<int>(arr.shape(0));
      dsm_w = static_cast<int>(arr.shape(1));
      dsm.resize(static_cast<size_t>(dsm_h) * dsm_w);
      std::memcpy(dsm.data(), arr.data(), dsm.size() * sizeof(float));
    }

    std::vector<Vec3<uint8_t>> colors(n, Vec3<uint8_t>(128, 128, 128));

    {
      py::gil_scoped_release release;
      sf_.BakeColors(pts, nrm, &colors, n_final, irls_iters,
                     relax.empty() ? nullptr : &relax,
                     dsm.empty() ? nullptr : &dsm, dsm_w, dsm_h, dsm_origin_x,
                     dsm_origin_y, dsm_gsd, dsm_max_z);
    }

    py::array_t<uint8_t> colors_arr({n, 3});
    std::memcpy(colors_arr.mutable_data(), colors.data(),
                n * 3 * sizeof(uint8_t));
    return colors_arr;
  }

 private:
  SVOFuser sf_;
  // Keep-alive references to the numpy buffers borrowed by AddView, so they
  // outlive the fuser (the maps inside sf_ are non-owning).  Released when this
  // wrapper — i.e. the Python SVOFuser object — is destroyed.
  std::vector<py::object> retained_;
};

// ---- DSM rasterizer wrapper (streaming mode-seeking + bilateral) ----

#ifdef OPENSFM_HAVE_OPENCL

class GPUDiffuserWrapper {
 public:
  void SetDevice(int idx) { device_idx_ = idx; }

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

  // Upload an external grid for post-processing (diffusion, bilateral).
  // Allocates GPU buffers sized to match the input grid.
  // NaN cells in the input are marked as non-observed (evolve under diffusion).
  void UploadGrid(const py::array_t<float, py::array::c_style>& grid_np) {
    if (grid_np.ndim() != 2) {
      throw std::invalid_argument("UploadGrid: input must be 2D float32");
    }
    grid_h_ = static_cast<int>(grid_np.shape(0));
    grid_w_ = static_cast<int>(grid_np.shape(1));
    ncells_ = static_cast<size_t>(grid_h_) * grid_w_;
    if (ncells_ == 0) {
      throw std::invalid_argument("UploadGrid: empty grid");
    }

    auto& dev = opencl::CLContext::Instance().Device(device_idx_);
    prog_ = dev.GetOrBuildProgram("diffusion", kDiffusionKernelSource);
    auto& ctx = dev.context();
    auto& q = dev.queue();

    // Allocate GPU buffers
    cl_out_a_ = cl::Buffer(ctx, CL_MEM_READ_WRITE, ncells_ * sizeof(float));
    cl_out_b_ = cl::Buffer(ctx, CL_MEM_READ_WRITE, ncells_ * sizeof(float));
    cl_valid_ = cl::Buffer(ctx, CL_MEM_READ_WRITE, ncells_ * sizeof(uint8_t));
    cl_guide_ = cl::Buffer(ctx, CL_MEM_READ_WRITE, ncells_ * sizeof(float));

    // Build host validity mask and replace NaN with 0 for upload
    const float* src = grid_np.data();
    std::vector<float> host_grid(ncells_);
    std::vector<uint8_t> host_valid(ncells_);
    for (size_t i = 0; i < ncells_; ++i) {
      if (std::isnan(src[i])) {
        host_grid[i] = NAN;  // Preserve NaN; diffusion kernel handles it.
        host_valid[i] = 0;
      } else {
        host_grid[i] = src[i];
        host_valid[i] = 1;
      }
    }

    q.enqueueWriteBuffer(cl_out_a_, CL_TRUE, 0, ncells_ * sizeof(float),
                         host_grid.data());
    q.enqueueWriteBuffer(cl_valid_, CL_TRUE, 0, ncells_ * sizeof(uint8_t),
                         host_valid.data());
  }

  // Image-guided DSM edge snapping: joint (cross) bilateral filter of the DSM
  // (2D float32, NaN = no-data) guided by the ortho colour (HxWx3 float in
  // [0,1]).  Collapses fattened roof/ground height ramps onto the photo-sharp
  // colour edge.  Self-contained (allocates its own buffers); NaN holes are
  // preserved.  Returns the sharpened DSM (HxW float32).
  foundation::pyarray_f SnapEdges(
      const py::array_t<float, py::array::c_style>& dsm_np,
      const py::array_t<float, py::array::c_style>& guide_np, int iterations,
      int radius, float sigma_spatial, float sigma_range) {
    if (dsm_np.ndim() != 2) {
      throw std::invalid_argument("SnapEdges: dsm must be 2D float32");
    }
    const int gh = static_cast<int>(dsm_np.shape(0));
    const int gw = static_cast<int>(dsm_np.shape(1));
    const size_t nc = static_cast<size_t>(gh) * gw;
    if (guide_np.size() != static_cast<ssize_t>(nc * 3)) {
      throw std::invalid_argument(
          "SnapEdges: guide must be HxWx3 matching dsm");
    }

    auto& dev = opencl::CLContext::Instance().Device(device_idx_);
    cl::Program prog =
        dev.GetOrBuildProgram("diffusion", kDiffusionKernelSource);
    auto& ctx = dev.context();
    auto& q = dev.queue();

    // Split NaN -> validity mask (holes pass through, never seed).
    const float* src = dsm_np.data();
    std::vector<float> host_dsm(nc);
    std::vector<uint8_t> host_valid(nc);
    for (size_t i = 0; i < nc; ++i) {
      if (std::isnan(src[i])) {
        host_dsm[i] = NAN;
        host_valid[i] = 0;
      } else {
        host_dsm[i] = src[i];
        host_valid[i] = 1;
      }
    }

    cl::Buffer a(ctx, CL_MEM_READ_WRITE, nc * sizeof(float));
    cl::Buffer b(ctx, CL_MEM_READ_WRITE, nc * sizeof(float));
    cl::Buffer cl_valid(ctx, CL_MEM_READ_WRITE, nc * sizeof(uint8_t));
    cl::Buffer cl_guide(ctx, CL_MEM_READ_ONLY, nc * 3 * sizeof(float));
    q.enqueueWriteBuffer(a, CL_TRUE, 0, nc * sizeof(float), host_dsm.data());
    q.enqueueWriteBuffer(cl_valid, CL_TRUE, 0, nc * sizeof(uint8_t),
                         host_valid.data());
    q.enqueueWriteBuffer(cl_guide, CL_TRUE, 0, nc * 3 * sizeof(float),
                         const_cast<float*>(guide_np.data()));

    const float inv_s = 1.0f / (2.0f * sigma_spatial * sigma_spatial);
    const float inv_r = 1.0f / (2.0f * sigma_range * sigma_range);
    const size_t gx = ((static_cast<size_t>(gw) + 15) / 16) * 16;
    const size_t gy = ((static_cast<size_t>(gh) + 15) / 16) * 16;

    cl::Buffer* s = &a;
    cl::Buffer* d = &b;
    {
      py::gil_scoped_release release;
      for (int it = 0; it < iterations; ++it) {
        cl::Kernel k(prog, "dsm_joint_bilateral");
        k.setArg(0, *s);
        k.setArg(1, *d);
        k.setArg(2, cl_guide);
        k.setArg(3, cl_valid);
        k.setArg(4, gw);
        k.setArg(5, gh);
        k.setArg(6, radius);
        k.setArg(7, inv_s);
        k.setArg(8, inv_r);
        q.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(gx, gy),
                               cl::NDRange(16, 16));
        q.finish();
        std::swap(s, d);
      }
    }

    std::vector<float> out(nc);
    q.enqueueReadBuffer(*s, CL_TRUE, 0, nc * sizeof(float), out.data());
    return foundation::py_array_from_data(out.data(), gh, gw);
  }

  // Coherence-enhancing shock filter of the DSM (2D float32, NaN = no-data).
  // Sharpens fattened height ramps into steps without any guide (breaks the
  // ortho<->DSM chicken-and-egg).  Self-contained; NaN cells are preserved.
  // Returns the sharpened DSM (HxW float32).
  foundation::pyarray_f ShockFilter(
      const py::array_t<float, py::array::c_style>& dsm_np, int iterations,
      int win, float dt, float coherence, float gsd, float edge_slope) {
    if (dsm_np.ndim() != 2) {
      throw std::invalid_argument("ShockFilter: dsm must be 2D float32");
    }
    const int gh = static_cast<int>(dsm_np.shape(0));
    const int gw = static_cast<int>(dsm_np.shape(1));
    const size_t nc = static_cast<size_t>(gh) * gw;

    auto& dev = opencl::CLContext::Instance().Device(device_idx_);
    cl::Program prog =
        dev.GetOrBuildProgram("diffusion", kDiffusionKernelSource);
    auto& ctx = dev.context();
    auto& q = dev.queue();

    const float* src = dsm_np.data();
    std::vector<float> host_dsm(nc);
    std::vector<uint8_t> host_valid(nc);
    for (size_t i = 0; i < nc; ++i) {
      if (std::isnan(src[i])) {
        host_dsm[i] = NAN;
        host_valid[i] = 0;
      } else {
        host_dsm[i] = src[i];
        host_valid[i] = 1;
      }
    }

    cl::Buffer a(ctx, CL_MEM_READ_WRITE, nc * sizeof(float));
    cl::Buffer b(ctx, CL_MEM_READ_WRITE, nc * sizeof(float));
    cl::Buffer cl_valid(ctx, CL_MEM_READ_WRITE, nc * sizeof(uint8_t));
    q.enqueueWriteBuffer(a, CL_TRUE, 0, nc * sizeof(float), host_dsm.data());
    q.enqueueWriteBuffer(cl_valid, CL_TRUE, 0, nc * sizeof(uint8_t),
                         host_valid.data());

    const size_t gx = ((static_cast<size_t>(gw) + 15) / 16) * 16;
    const size_t gy = ((static_cast<size_t>(gh) + 15) / 16) * 16;
    cl::Buffer* s = &a;
    cl::Buffer* d = &b;
    {
      py::gil_scoped_release release;
      for (int it = 0; it < iterations; ++it) {
        cl::Kernel k(prog, "dsm_shock");
        k.setArg(0, *s);
        k.setArg(1, *d);
        k.setArg(2, cl_valid);
        k.setArg(3, gw);
        k.setArg(4, gh);
        k.setArg(5, win);
        k.setArg(6, dt);
        k.setArg(7, coherence);
        k.setArg(8, gsd);
        k.setArg(9, edge_slope);
        q.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(gx, gy),
                               cl::NDRange(16, 16));
        q.finish();
        std::swap(s, d);
      }
    }

    std::vector<float> out(nc);
    q.enqueueReadBuffer(*s, CL_TRUE, 0, nc * sizeof(float), out.data());
    return foundation::py_array_from_data(out.data(), gh, gw);
  }

  // Gated 3x3 median despeckle of an RGB ortho (HxWx3 float, 0-255), guided by
  // a validity mask (HxW uint8, 0 = no-data passthrough).  Returns the
  // despeckled ortho (HxWx3 float).
  py::array_t<float> GatedMedian(
      const py::array_t<float, py::array::c_style>& ortho_np,
      const py::array_t<uint8_t, py::array::c_style>& valid_np,
      float threshold) {
    if (ortho_np.ndim() != 3 || ortho_np.shape(2) != 3) {
      throw std::invalid_argument("GatedMedian: ortho must be HxWx3 float32");
    }
    const int gh = static_cast<int>(ortho_np.shape(0));
    const int gw = static_cast<int>(ortho_np.shape(1));
    const size_t nc = static_cast<size_t>(gh) * gw;
    if (valid_np.size() != static_cast<ssize_t>(nc)) {
      throw std::invalid_argument(
          "GatedMedian: valid must be HxW matching ortho");
    }

    auto& dev = opencl::CLContext::Instance().Device(device_idx_);
    cl::Program prog =
        dev.GetOrBuildProgram("diffusion", kDiffusionKernelSource);
    auto& ctx = dev.context();
    auto& q = dev.queue();

    cl::Buffer cl_in(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     nc * 3 * sizeof(float),
                     const_cast<float*>(ortho_np.data()));
    cl::Buffer cl_out(ctx, CL_MEM_WRITE_ONLY, nc * 3 * sizeof(float));
    cl::Buffer cl_valid(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        nc * sizeof(uint8_t),
                        const_cast<uint8_t*>(valid_np.data()));

    const size_t gx = ((static_cast<size_t>(gw) + 15) / 16) * 16;
    const size_t gy = ((static_cast<size_t>(gh) + 15) / 16) * 16;
    {
      py::gil_scoped_release release;
      cl::Kernel k(prog, "ortho_gated_median");
      k.setArg(0, cl_in);
      k.setArg(1, cl_out);
      k.setArg(2, cl_valid);
      k.setArg(3, gw);
      k.setArg(4, gh);
      k.setArg(5, threshold);
      q.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(gx, gy),
                             cl::NDRange(16, 16));
      q.finish();
    }

    py::array_t<float> result({gh, gw, 3});
    q.enqueueReadBuffer(cl_out, CL_TRUE, 0, nc * 3 * sizeof(float),
                        result.mutable_data());
    return result;
  }

 private:
  int device_idx_ = 0;
  size_t ncells_ = 0;
  int grid_w_ = 0;
  int grid_h_ = 0;

  cl::Program prog_;
  cl::Buffer cl_out_a_;
  cl::Buffer cl_out_b_;
  cl::Buffer cl_valid_;
  cl::Buffer cl_guide_;
};

#else  // !OPENSFM_HAVE_OPENCL

class GPUDiffuserWrapper {
 public:
  void SetDevice(int) {}
  foundation::pyarray_f Diffuse(const py::array_t<float, py::array::c_style>&,
                                int, float, float) {
    throw std::runtime_error("GPUDiffuser: OpenCL not available");
  }
  void UploadGrid(const py::array_t<float, py::array::c_style>&) {
    throw std::runtime_error("GPUDiffuser: OpenCL not available");
  }
  foundation::pyarray_f SnapEdges(const py::array_t<float, py::array::c_style>&,
                                  const py::array_t<float, py::array::c_style>&,
                                  int, int, float, float) {
    throw std::runtime_error("GPUDiffuser: OpenCL not available");
  }
  foundation::pyarray_f ShockFilter(
      const py::array_t<float, py::array::c_style>&, int, int, float, float,
      float, float) {
    throw std::runtime_error("GPUDiffuser: OpenCL not available");
  }
  py::array_t<float> GatedMedian(
      const py::array_t<float, py::array::c_style>&,
      const py::array_t<uint8_t, py::array::c_style>&, float) {
    throw std::runtime_error("GPUDiffuser: OpenCL not available");
  }
};

#endif  // OPENSFM_HAVE_OPENCL

}  // namespace dense
