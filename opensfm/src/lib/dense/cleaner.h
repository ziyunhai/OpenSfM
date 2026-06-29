#pragma once

#ifdef OPENSFM_HAVE_OPENCL

#include <dense/fuser.h>
#include <dense/opencl_utils.h>

#include <opencv2/opencv.hpp>
#include <random>
#include <string>
#include <vector>

namespace dense {

/// Standalone GPU depthmap cleaner with edge-aware space carving.
///
/// Extends bilateral geometric consistency with:
/// - Space carving: neighbors that see further vote to carve foreground pixels
/// - Edge detection: pixels at depth discontinuities get stricter thresholds
/// - Grazing angle: pixels with near-perpendicular normals are penalized
class GPUDepthmapCleaner {
 public:
  GPUDepthmapCleaner();
  ~GPUDepthmapCleaner();

  // --- Core parameters ---
  void SetSameDepthThreshold(float t);
  void SetMinConsistentViews(int n);
  void SetDevice(int device_idx);

  // --- Space carving parameters ---
  /// Relative depth margin for carve votes (e.g. 0.05 = 5% further = carve).
  void SetCarvingThreshold(float t);
  /// Max carve votes before pixel is killed (e.g. 2).
  void SetMaxCarvedViews(int n);

  // --- Edge-aware parameters ---
  /// Cosine threshold for grazing angle detection (below = grazing, e.g. 0.2).
  void SetGrazingCosThreshold(float t);
  /// Depth ratio threshold for edge detection (e.g. 1.10 = 10% discontinuity).
  void SetEdgeDepthRatio(float r);

  /// Add a view with depth only (normals will not be used for this view).
  int AddView(const Mat3d& K, const Mat3d& R, const Vec3d& t,
              const ImageF& depth);

  /// Add a view with depth and per-pixel normal (camera-frame, HxWx3).
  int AddView(const Mat3d& K, const Mat3d& R, const Vec3d& t,
              const ImageF& depth, const PixelData3f& normal);

  cv::Mat Clean(int ref_idx, const std::vector<int>& neighbor_ids);

  /// Remove all views and release GPU buffers.
  void Clear();

 private:
  struct ViewEntry {
    cv::Mat depth;   // CV_32F — kept as cv::Mat for OpenCL upload.
    cv::Mat normal;  // CV_32FC3 (camera-frame), empty if not provided.
    Mat3d K;
    Mat3d R;
    Vec3d t;
    int width = 0;
    int height = 0;
  };
  std::vector<ViewEntry> views_;

  // Core parameters.
  float same_depth_threshold_ = 0.01f;
  int min_consistent_views_ = 3;
  int device_idx_ = 0;

  // Space carving parameters.
  float carving_threshold_ = 0.05f;
  int max_carved_views_ = 2;

  // Edge-aware parameters.
  float grazing_cos_threshold_ = 0.2f;

  bool kernel_built_ = false;
  cl::Program program_;
  cl::Kernel k_clean_;

  // Persistent GPU resources — reused across Clean() calls.
  cl::Buffer cl_cameras_;
  cl::Buffer cl_clean_depth_;
  cl::Buffer cl_ref_normal_;
  size_t cl_cameras_bytes_ = 0;
  size_t cl_clean_depth_bytes_ = 0;
  size_t cl_ref_normal_bytes_ = 0;

  // Cached image2d_t pool — reallocated only when dimensions change.
  static constexpr int kMaxCleanSources = 16;
  cl::Image2D cl_ref_depth_img_;
  std::vector<cl::Image2D> cl_src_depth_imgs_;
  int cl_img_w_ = 0;
  int cl_img_h_ = 0;
};

}  // namespace dense

#endif  // OPENSFM_HAVE_OPENCL
