#include <dense/cleaner.h>

#ifdef OPENSFM_HAVE_OPENCL

#include <dense/opencl_kernels.h>

#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <unordered_map>

namespace dense {

// =====================================================================
// GPUDepthmapCleaner implementation.
// =====================================================================

GPUDepthmapCleaner::GPUDepthmapCleaner() = default;

void GPUDepthmapCleaner::SetSameDepthThreshold(float t) {
  same_depth_threshold_ = t;
}

void GPUDepthmapCleaner::SetMinConsistentViews(int n) {
  min_consistent_views_ = n;
}

void GPUDepthmapCleaner::SetDevice(int device_idx) { device_idx_ = device_idx; }

void GPUDepthmapCleaner::SetCarvingThreshold(float t) {
  carving_threshold_ = t;
}

void GPUDepthmapCleaner::SetMaxCarvedViews(int n) { max_carved_views_ = n; }

void GPUDepthmapCleaner::SetGrazingCosThreshold(float t) {
  grazing_cos_threshold_ = t;
}

void GPUDepthmapCleaner::SetEdgeDepthRatio(float r) { edge_depth_ratio_ = r; }

int GPUDepthmapCleaner::AddView(const Mat3d& K, const Mat3d& R, const Vec3d& t,
                                const ImageF& depth) {
  ViewEntry v;
  v.depth =
      cv::Mat(static_cast<int>(depth.rows()), static_cast<int>(depth.cols()),
              CV_32F, const_cast<float*>(depth.data()))
          .clone();
  v.K = K;
  v.R = R;
  v.t = t;
  v.width = static_cast<int>(depth.cols());
  v.height = static_cast<int>(depth.rows());
  // v.normal remains empty — no normal for this view.
  views_.push_back(std::move(v));
  return static_cast<int>(views_.size()) - 1;
}

int GPUDepthmapCleaner::AddView(const Mat3d& K, const Mat3d& R, const Vec3d& t,
                                const ImageF& depth,
                                const PixelData3f& normal) {
  ViewEntry v;
  const int h = static_cast<int>(depth.rows());
  const int w = static_cast<int>(depth.cols());
  v.depth = cv::Mat(h, w, CV_32F, const_cast<float*>(depth.data())).clone();
  // PixelData3f is (3, H*W) column-major = interleaved [nx,ny,nz, nx,ny,nz,...]
  // which maps directly to CV_32FC3 of shape (H, W).
  v.normal = cv::Mat(h, w, CV_32FC3, const_cast<float*>(normal.data())).clone();
  v.K = K;
  v.R = R;
  v.t = t;
  v.width = w;
  v.height = h;
  views_.push_back(std::move(v));
  return static_cast<int>(views_.size()) - 1;
}

cv::Mat GPUDepthmapCleaner::Clean(int ref_idx,
                                  const std::vector<int>& neighbor_ids) {
  auto& dev = opencl::CLContext::Instance().Device(device_idx_);
  cl::Context& cl_ctx = dev.context();
  cl::CommandQueue& queue = dev.queue();

  if (!kernel_built_) {
    program_ = dev.BuildProgram(kCleanKernelSource);
    k_clean_ = cl::Kernel(program_, "acmmp_clean_depthmap");
    kernel_built_ = true;
  }

  if (ref_idx < 0 || ref_idx >= static_cast<int>(views_.size())) {
    throw std::out_of_range("Clean: ref_idx " + std::to_string(ref_idx) +
                            " out of range [0, " +
                            std::to_string(views_.size()) + ")");
  }
  for (int i = 0; i < static_cast<int>(neighbor_ids.size()); ++i) {
    if (neighbor_ids[i] < 0 ||
        neighbor_ids[i] >= static_cast<int>(views_.size())) {
      throw std::out_of_range("Clean: neighbor_ids[" + std::to_string(i) +
                              "] = " + std::to_string(neighbor_ids[i]) +
                              " out of range [0, " +
                              std::to_string(views_.size()) + ")");
    }
  }

  auto& ref = views_[ref_idx];
  const int w = ref.width;
  const int h = ref.height;
  const int npix = w * h;
  const int num_neighbors = static_cast<int>(neighbor_ids.size());
  const int num_views = 1 + num_neighbors;
  const bool has_normal = !ref.normal.empty();

  // --- Ensure image2d_t pool matches current dimensions ---
  if (w != cl_img_w_ || h != cl_img_h_) {
    cl_int err;
    cl::ImageFormat fmt(CL_R, CL_FLOAT);
    cl_ref_depth_img_ =
        cl::Image2D(cl_ctx, CL_MEM_READ_ONLY, fmt, w, h, 0, nullptr, &err);
    opencl::CheckCL(err, "Clean: ref depth Image2D create");
    cl_src_depth_imgs_.clear();
    cl_src_depth_imgs_.reserve(kMaxCleanSources);
    for (int i = 0; i < kMaxCleanSources; ++i) {
      cl::Image2D img(cl_ctx, CL_MEM_READ_ONLY, fmt, w, h, 0, nullptr, &err);
      opencl::CheckCL(err, "Clean: src depth Image2D create");
      cl_src_depth_imgs_.push_back(std::move(img));
    }
    cl_img_w_ = w;
    cl_img_h_ = h;
  }

  // --- Upload depth data into cached images ---
  cl::array<cl::size_type, 3> origin = {{0, 0, 0}};
  cl::array<cl::size_type, 3> region = {
      {static_cast<cl::size_type>(w), static_cast<cl::size_type>(h), 1}};
  queue.enqueueWriteImage(cl_ref_depth_img_, CL_TRUE, origin, region,
                          ref.depth.step[0], 0, ref.depth.data);

  for (int vi = 0; vi < num_neighbors && vi < kMaxCleanSources; ++vi) {
    auto& other = views_[neighbor_ids[vi]];
    cv::Mat other_depth;
    if (other.depth.cols != w || other.depth.rows != h) {
      cv::resize(other.depth, other_depth, cv::Size(w, h), 0, 0,
                 cv::INTER_LINEAR);
    } else {
      other_depth = other.depth;
    }
    queue.enqueueWriteImage(cl_src_depth_imgs_[vi], CL_TRUE, origin, region,
                            other_depth.step[0], 0, other_depth.data);
  }

  // --- Upload ref normal buffer (float3 per pixel) ---
  const size_t normal_bytes = static_cast<size_t>(npix) * 3 * sizeof(float);
  if (has_normal) {
    if (normal_bytes > cl_ref_normal_bytes_) {
      cl_ref_normal_ = cl::Buffer(cl_ctx, CL_MEM_READ_ONLY, normal_bytes);
      cl_ref_normal_bytes_ = normal_bytes;
    }
    // Ensure normal is contiguous and correct size.
    cv::Mat norm_upload = ref.normal;
    if (norm_upload.cols != w || norm_upload.rows != h) {
      cv::resize(norm_upload, norm_upload, cv::Size(w, h), 0, 0,
                 cv::INTER_LINEAR);
    }
    if (!norm_upload.isContinuous()) {
      norm_upload = norm_upload.clone();
    }
    queue.enqueueWriteBuffer(cl_ref_normal_, CL_FALSE, 0, normal_bytes,
                             norm_upload.data);
  } else {
    // Allocate a dummy buffer if not yet allocated (kernel still needs arg).
    if (cl_ref_normal_bytes_ == 0) {
      cl_ref_normal_ = cl::Buffer(cl_ctx, CL_MEM_READ_ONLY, sizeof(float));
      cl_ref_normal_bytes_ = sizeof(float);
    }
  }

  // --- Upload cameras (small buffer, safe as __global) ---
  std::vector<CLCamera> cameras(num_views);
  cameras[0] = MakeCLCamera(ref.K, ref.R, ref.t, w, h);
  for (int vi = 0; vi < num_neighbors; ++vi) {
    auto& other = views_[neighbor_ids[vi]];
    cameras[vi + 1] = MakeCLCamera(other.K, other.R, other.t, w, h);
  }
  const size_t cameras_bytes = cameras.size() * sizeof(CLCamera);
  if (cameras_bytes > cl_cameras_bytes_) {
    cl_cameras_ = cl::Buffer(cl_ctx, CL_MEM_READ_ONLY, cameras_bytes);
    cl_cameras_bytes_ = cameras_bytes;
  }
  queue.enqueueWriteBuffer(cl_cameras_, CL_FALSE, 0, cameras_bytes,
                           cameras.data());

  // --- Output buffer ---
  const size_t clean_bytes = npix * sizeof(float);
  if (clean_bytes > cl_clean_depth_bytes_) {
    cl_clean_depth_ = cl::Buffer(cl_ctx, CL_MEM_WRITE_ONLY, clean_bytes);
    cl_clean_depth_bytes_ = clean_bytes;
  }

  // --- Set kernel arguments ---
  int effective_min = std::min(min_consistent_views_, num_views);
  int has_normal_int = has_normal ? 1 : 0;

  int arg = 0;
  k_clean_.setArg(arg++, cl_ref_depth_img_);
  for (int i = 0; i < kMaxCleanSources; ++i) {
    k_clean_.setArg(arg++, cl_src_depth_imgs_[i]);
  }
  k_clean_.setArg(arg++, cl_cameras_);
  k_clean_.setArg(arg++, cl_clean_depth_);
  k_clean_.setArg(arg++, cl_ref_normal_);
  k_clean_.setArg(arg++, w);
  k_clean_.setArg(arg++, h);
  k_clean_.setArg(arg++, num_views);
  k_clean_.setArg(arg++, same_depth_threshold_);
  k_clean_.setArg(arg++, effective_min);
  k_clean_.setArg(arg++, carving_threshold_);
  k_clean_.setArg(arg++, max_carved_views_);
  k_clean_.setArg(arg++, grazing_cos_threshold_);
  k_clean_.setArg(arg++, edge_depth_ratio_);
  k_clean_.setArg(arg++, has_normal_int);

  cl::NDRange global(static_cast<size_t>((w + 15) / 16 * 16),
                     static_cast<size_t>((h + 15) / 16 * 16));
  cl::NDRange local(16, 16);
  queue.enqueueNDRangeKernel(k_clean_, cl::NullRange, global, local);

  queue.finish();

  cv::Mat cleaned(h, w, CV_32F);
  queue.enqueueReadBuffer(cl_clean_depth_, CL_TRUE, 0, clean_bytes,
                          cleaned.ptr<float>());
  return cleaned;
}

// =====================================================================
// SLIC segmentation on GPU — reuses kSLICKernelSource from
// opencl_kernels.h.  Returns label map and stores it internally.
// =====================================================================
cv::Mat GPUDepthmapCleaner::ComputeSLIC(const cv::Mat& gray, int grid_step,
                                        float compactness) {
  auto& dev = opencl::CLContext::Instance().Device(device_idx_);
  cl::Context& cl_ctx = dev.context();
  cl::CommandQueue& queue = dev.queue();
  cl_int err;

  // Build SLIC kernels once.
  if (!slic_built_) {
    slic_program_ = dev.GetOrBuildProgram("slic", kSLICKernelSource);
    k_slic_init_ = cl::Kernel(slic_program_, "slic_init_centers", &err);
    opencl::CheckCL(err, "slic_init_centers");
    k_slic_assign_ = cl::Kernel(slic_program_, "slic_assign_pixels", &err);
    opencl::CheckCL(err, "slic_assign_pixels");
    k_slic_update_ = cl::Kernel(slic_program_, "slic_update_centers", &err);
    opencl::CheckCL(err, "slic_update_centers");
    slic_built_ = true;
  }

  const int w = gray.cols;
  const int h = gray.rows;
  const int npix = w * h;

  // Convert to float [0,1].
  cv::Mat float_img;
  gray.convertTo(float_img, CV_32F, 1.0 / 255.0);

  // Upload as Image2D.
  cl::ImageFormat fmt(CL_R, CL_FLOAT);
  cl::Image2D cl_img(cl_ctx, CL_MEM_READ_ONLY, fmt, w, h, 0, nullptr, &err);
  opencl::CheckCL(err, "SLIC image upload");
  cl::array<cl::size_type, 3> origin = {{0, 0, 0}};
  cl::array<cl::size_type, 3> region = {
      {static_cast<cl::size_type>(w), static_cast<cl::size_type>(h), 1}};
  queue.enqueueWriteImage(cl_img, CL_TRUE, origin, region, float_img.step[0], 0,
                          float_img.data);

  // Grid dimensions.
  const int centers_x = (w + grid_step - 1) / grid_step;
  const int centers_y = (h + grid_step - 1) / grid_step;
  const int num_centers = centers_x * centers_y;

  // Allocate buffers.
  const size_t center_size = 16;  // float x,y,intensity + int count
  cl::Buffer cl_centers(cl_ctx, CL_MEM_READ_WRITE, center_size * num_centers,
                        nullptr, &err);
  opencl::CheckCL(err, "SLIC centers");
  cl::Buffer cl_labels(cl_ctx, CL_MEM_READ_WRITE, sizeof(int) * npix, nullptr,
                       &err);
  opencl::CheckCL(err, "SLIC labels");

  // Init centers.
  {
    int arg = 0;
    k_slic_init_.setArg(arg++, cl_centers);
    k_slic_init_.setArg(arg++, cl_img);
    k_slic_init_.setArg(arg++, w);
    k_slic_init_.setArg(arg++, h);
    k_slic_init_.setArg(arg++, grid_step);
    k_slic_init_.setArg(arg++, centers_x);
    k_slic_init_.setArg(arg++, centers_y);
    cl::NDRange global(static_cast<size_t>((centers_x + 15) / 16 * 16),
                       static_cast<size_t>((centers_y + 15) / 16 * 16));
    queue.enqueueNDRangeKernel(k_slic_init_, cl::NullRange, global,
                               cl::NDRange(16, 16));
  }

  // Iterate: assign + update.
  for (int it = 0; it < 5; ++it) {
    {
      int arg = 0;
      k_slic_assign_.setArg(arg++, cl_centers);
      k_slic_assign_.setArg(arg++, cl_labels);
      k_slic_assign_.setArg(arg++, cl_img);
      k_slic_assign_.setArg(arg++, w);
      k_slic_assign_.setArg(arg++, h);
      k_slic_assign_.setArg(arg++, grid_step);
      k_slic_assign_.setArg(arg++, centers_x);
      k_slic_assign_.setArg(arg++, centers_y);
      k_slic_assign_.setArg(arg++, compactness);
      cl::NDRange global(static_cast<size_t>((w + 15) / 16 * 16),
                         static_cast<size_t>((h + 15) / 16 * 16));
      queue.enqueueNDRangeKernel(k_slic_assign_, cl::NullRange, global,
                                 cl::NDRange(16, 16));
    }
    {
      int arg = 0;
      k_slic_update_.setArg(arg++, cl_centers);
      k_slic_update_.setArg(arg++, cl_labels);
      k_slic_update_.setArg(arg++, cl_img);
      k_slic_update_.setArg(arg++, w);
      k_slic_update_.setArg(arg++, h);
      k_slic_update_.setArg(arg++, grid_step);
      k_slic_update_.setArg(arg++, centers_x);
      k_slic_update_.setArg(arg++, centers_y);
      cl::NDRange global(static_cast<size_t>((centers_x + 15) / 16 * 16),
                         static_cast<size_t>((centers_y + 15) / 16 * 16));
      queue.enqueueNDRangeKernel(k_slic_update_, cl::NullRange, global,
                                 cl::NDRange(16, 16));
    }
  }

  // Read back labels.
  queue.finish();
  slic_labels_ = cv::Mat(h, w, CV_32S);
  queue.enqueueReadBuffer(cl_labels, CL_TRUE, 0, sizeof(int) * npix,
                          slic_labels_.ptr<int>());
  return slic_labels_.clone();
}

// =====================================================================
// Mahalanobis segment-aware depth filter — GPU dispatch.
// =====================================================================
cv::Mat GPUDepthmapCleaner::FilterMahalanobis(const cv::Mat& depth,
                                              const Mat3d& K,
                                              float mahal_threshold,
                                              int window_radius) {
  if (slic_labels_.empty()) {
    throw std::runtime_error("FilterMahalanobis: must call ComputeSLIC first");
  }

  auto& dev = opencl::CLContext::Instance().Device(device_idx_);
  cl::Context& cl_ctx = dev.context();
  cl::CommandQueue& queue = dev.queue();
  cl_int err;

  // Build Mahalanobis kernel once.
  if (!mahal_built_) {
    mahal_program_ = dev.BuildProgram(kMahalanobisKernelSource);
    k_mahal_filter_ = cl::Kernel(mahal_program_, "mahalanobis_filter", &err);
    opencl::CheckCL(err, "mahalanobis_filter kernel");
    mahal_built_ = true;
  }

  const int w = depth.cols;
  const int h = depth.rows;
  const int npix = w * h;

  // Resize labels to match depth if needed.
  cv::Mat labels = slic_labels_;
  if (labels.cols != w || labels.rows != h) {
    cv::resize(labels, labels, cv::Size(w, h), 0, 0, cv::INTER_NEAREST);
  }

  // Upload depth, labels.
  cl::Buffer cl_depth_in(cl_ctx, CL_MEM_READ_ONLY, sizeof(float) * npix,
                         nullptr, &err);
  opencl::CheckCL(err, "mahal depth_in");
  cl::Buffer cl_depth_out(cl_ctx, CL_MEM_WRITE_ONLY, sizeof(float) * npix,
                          nullptr, &err);
  opencl::CheckCL(err, "mahal depth_out");
  cl::Buffer cl_labels(cl_ctx, CL_MEM_READ_ONLY, sizeof(int) * npix, nullptr,
                       &err);
  opencl::CheckCL(err, "mahal labels");

  cv::Mat depth_cont = depth.isContinuous() ? depth : depth.clone();
  cv::Mat labels_cont = labels.isContinuous() ? labels : labels.clone();

  queue.enqueueWriteBuffer(cl_depth_in, CL_TRUE, 0, sizeof(float) * npix,
                           depth_cont.ptr<float>());
  queue.enqueueWriteBuffer(cl_labels, CL_TRUE, 0, sizeof(int) * npix,
                           labels_cont.ptr<int>());

  // Extract intrinsics.
  float fx = static_cast<float>(K(0, 0));
  float fy = static_cast<float>(K(1, 1));
  float cx = static_cast<float>(K(0, 2));
  float cy = static_cast<float>(K(1, 2));

  // Set kernel args.
  int arg = 0;
  k_mahal_filter_.setArg(arg++, cl_depth_in);
  k_mahal_filter_.setArg(arg++, cl_depth_out);
  k_mahal_filter_.setArg(arg++, cl_labels);
  k_mahal_filter_.setArg(arg++, fx);
  k_mahal_filter_.setArg(arg++, fy);
  k_mahal_filter_.setArg(arg++, cx);
  k_mahal_filter_.setArg(arg++, cy);
  k_mahal_filter_.setArg(arg++, w);
  k_mahal_filter_.setArg(arg++, h);
  k_mahal_filter_.setArg(arg++, window_radius);
  k_mahal_filter_.setArg(arg++, mahal_threshold);

  cl::NDRange global(static_cast<size_t>((w + 15) / 16 * 16),
                     static_cast<size_t>((h + 15) / 16 * 16));
  queue.enqueueNDRangeKernel(k_mahal_filter_, cl::NullRange, global,
                             cl::NDRange(16, 16));
  queue.finish();

  cv::Mat result(h, w, CV_32F);
  queue.enqueueReadBuffer(cl_depth_out, CL_TRUE, 0, sizeof(float) * npix,
                          result.ptr<float>());
  return result;
}

void GPUDepthmapCleaner::Clear() {
  views_.clear();
  // Release GPU resources.
  cl_cameras_ = cl::Buffer();
  cl_clean_depth_ = cl::Buffer();
  cl_ref_normal_ = cl::Buffer();
  cl_cameras_bytes_ = 0;
  cl_clean_depth_bytes_ = 0;
  cl_ref_normal_bytes_ = 0;
  cl_ref_depth_img_ = cl::Image2D();
  cl_src_depth_imgs_.clear();
  cl_img_w_ = 0;
  cl_img_h_ = 0;
  slic_labels_ = cv::Mat();
}

GPUDepthmapCleaner::~GPUDepthmapCleaner() {
  k_clean_ = cl::Kernel();
  program_ = cl::Program();
}

}  // namespace dense

#endif  // OPENSFM_HAVE_OPENCL
