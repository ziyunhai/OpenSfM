#include <dense/estimator.h>

#ifdef OPENSFM_HAVE_OPENCL

#include <dense/opencl_kernels.h>

// CL_DEVICE_IMAGE_PITCH_ALIGNMENT is part of cl_khr_image2d_from_buffer
// but only formally in OpenCL 2.0+ headers.  Apple's 1.2 driver supports
// the extension, so define the enum value if the header doesn't provide it.
#ifndef CL_DEVICE_IMAGE_PITCH_ALIGNMENT
#define CL_DEVICE_IMAGE_PITCH_ALIGNMENT 0x104A
#endif

#include <foundation/logging.h>

#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <sstream>
#include <unordered_map>

namespace dense {

// =====================================================================
// DepthmapEstimator implementation.
// =====================================================================

DepthmapEstimator::DepthmapEstimator() = default;
DepthmapEstimator::~DepthmapEstimator() = default;

void DepthmapEstimator::AddView(const Mat3d& K, const Mat3d& R, const Vec3d& t,
                                const ImageU8& image) {
  Ks_.push_back(K);
  Rs_.push_back(R);
  ts_.push_back(t);
  images_.emplace_back(cv::Mat(static_cast<int>(image.rows()),
                               static_cast<int>(image.cols()), CV_8U,
                               const_cast<uint8_t*>(image.data()))
                           .clone());
}

void DepthmapEstimator::SetDepthRange(float min_depth, float max_depth) {
  params_.depth_min = min_depth;
  params_.depth_max = max_depth;
}

void DepthmapEstimator::SetParams(const DepthmapParams& params) {
  params_ = params;
}

void DepthmapEstimator::SetGeomConsistencyWeight(float weight) {
  geom_weight_ = weight;
}

void DepthmapEstimator::SetDevice(int device_idx) { device_idx_ = device_idx; }

void DepthmapEstimator::SetPreviousDepths(
    const std::vector<std::pair<int, ImageF>>& depths) {
  prev_depth_entries_ = depths;
}

void DepthmapEstimator::ClearPreviousDepths() { prev_depth_entries_.clear(); }

void DepthmapEstimator::UploadPreviousDepths(int width, int height) {
  auto& dev = opencl::CLContext::Instance().Device(device_idx_);
  auto& queue = dev.queue();

  const int npix = width * height;

  if (prev_depth_entries_.empty() || geom_weight_ <= 0.0f) {
    // No geometric consistency — zero out mask.
    cl_prev_depth_mask_ = 0u;
    return;
  }

  // Build the flat buffer: prev_depths[src_idx * npix + y * w + x].
  // Only slots with actual depth data are filled; others stay zero.
  std::vector<float> flat(kMaxSources * npix, 0.0f);
  cl_uint mask = 0u;

  for (const auto& entry : prev_depth_entries_) {
    int src_idx = entry.first;
    if (src_idx < 0 || src_idx >= kMaxSources) {
      continue;
    }

    // Resize to current level resolution if needed.
    const ImageF& src_depth = entry.second;
    const float* depth_ptr;
    ImageF resized_depth;
    int dw = static_cast<int>(src_depth.cols());
    int dh = static_cast<int>(src_depth.rows());
    if (dw != width || dh != height) {
      cv::Mat src_cv(dh, dw, CV_32F, const_cast<float*>(src_depth.data()));
      cv::Mat dst_cv;
      cv::resize(src_cv, dst_cv, cv::Size(width, height), 0, 0,
                 cv::INTER_LINEAR);
      resized_depth = Eigen::Map<ImageF>(dst_cv.ptr<float>(), height, width);
      depth_ptr = resized_depth.data();
    } else {
      depth_ptr = src_depth.data();
    }

    // Copy into the flat buffer at the correct slot.
    const size_t offset = static_cast<size_t>(src_idx) * npix;
    std::memcpy(&flat[offset], depth_ptr, npix * sizeof(float));
    mask |= (1u << src_idx);
  }

  // Upload to GPU.
  const size_t buf_bytes = kMaxSources * npix * sizeof(float);
  queue.enqueueWriteBuffer(cl_prev_depths_, CL_TRUE, 0, buf_bytes, flat.data());
  cl_prev_depth_mask_ = mask;
}

void DepthmapEstimator::SetSfMPoints(
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& points) {
  const int num_points = static_cast<int>(points.rows());
  sfm_points_.clear();
  sfm_points_.reserve(num_points);
  for (int i = 0; i < num_points; ++i) {
    sfm_points_.push_back(points.row(i).transpose());
  }
  {
    std::ostringstream oss;
    oss << "[PatchMatch] Received " << sfm_points_.size()
        << " SfM points for planar prior";
    foundation::LogDebug("dense", oss.str());
  }
}

void DepthmapEstimator::Run(DepthmapResult* result) {
  int N = Prepare();
  for (int level = 0; level < N; ++level) {
    RunLevel(level, N, result);
  }
}

int DepthmapEstimator::Prepare() {
  if (images_.empty()) {
    throw std::runtime_error("DepthmapEstimator: no views added");
  }

  if (!opencl::CLContext::Instance().IsAvailable()) {
    throw std::runtime_error(
        "DepthmapEstimator: OpenCL is not available on this system");
  }

  params_.num_images = static_cast<int>(images_.size());
  const int full_w = images_[0].cols;
  const int full_h = images_[0].rows;
  const int N = std::max(1, params_.hierarchy_levels);

  // Build kernels once.
  if (!cl_initialised_) {
    BuildKernels();
    cl_initialised_ = true;
  }

  // Save original full-resolution data.
  orig_images_ = images_;
  orig_Ks_ = Ks_;
  prev_level_result_ = DepthmapResult();
  prev_level_w_ = 0;
  prev_level_h_ = 0;
  prepared_ = true;

  return N;
}

void DepthmapEstimator::RunLevel(int level, int total_levels,
                                 DepthmapResult* result) {
  if (!prepared_) {
    throw std::runtime_error(
        "DepthmapEstimator: Prepare() must be called first");
  }

  const int N = total_levels;
  const int full_w = orig_images_[0].cols;
  const int full_h = orig_images_[0].rows;

  // Scale factor: level 0 = 1/2^(N-1), ..., level N-1 = 1.0
  const double scale = 1.0 / std::pow(2.0, N - 1 - level);

  if (scale < 0.999) {
    // Build scaled images from the originals.
    std::vector<cv::Mat> scaled_images;
    std::vector<Mat3d> scaled_Ks;
    images_ = orig_images_;
    Ks_ = orig_Ks_;
    BuildImagePyramid(static_cast<float>(scale), scaled_images, scaled_Ks);
    images_ = scaled_images;
    Ks_ = scaled_Ks;
  } else {
    images_ = orig_images_;
    Ks_ = orig_Ks_;
  }

  const int w = images_[0].cols;
  const int h = images_[0].rows;

  UploadData();

  // SLIC segmentation: compute per-pixel segment labels for propagation gating.
  RunSLIC(w, h);

  if (level == 0 && prev_level_w_ == 0) {
    RandomInit(w, h);
  } else if (prev_level_w_ > 0) {
    UpsampleDepthNormal(prev_level_result_, prev_level_w_, prev_level_h_, w, h);
  }

  // Upload previous-scale depth maps for geometric consistency.
  UploadPreviousDepths(w, h);

  // Compute SfM-based planar prior for this pyramid level.
  const bool have_prior = !sfm_points_.empty();
  if (have_prior) {
    ComputeSfMPlanarPrior(w, h);
  }

  // ---- PatchMatch iterations with prior re-seeding ----
  PriorReinit(w, h);
  for (int i = 0; i < params_.max_iterations; ++i) {
    RunIteration(i, w, h);
  }
  if (params_.checkerboard_filter) {
    RunCheckerboardFilter(w, h);
  }

  auto& dev = opencl::CLContext::Instance().Device(device_idx_);
  dev.queue().finish();

  const int npix = w * h;
  std::vector<float> costs_dbg(npix);
  dev.queue().enqueueReadBuffer(cl_costs_, CL_TRUE, 0, sizeof(float) * npix,
                                costs_dbg.data());

  // Save for upsampling to the next level (no median filter).
  if (level < N - 1) {
    ReadBackResults(&prev_level_result_, w, h, /*apply_median=*/false);
    prev_level_w_ = w;
    prev_level_h_ = h;
  }

  // For the last level, read back final results.
  if (level == N - 1) {
    // Restore original data.
    images_ = orig_images_;
    Ks_ = orig_Ks_;
    ReadBackResults(result, full_w, full_h, /*apply_median=*/false);
  } else {
    // Return intermediate result (useful for cluster orchestration).
    ReadBackResults(result, w, h, /*apply_median=*/false);
  }
}

// =====================================================================
// Build OpenCL kernels
// =====================================================================
void DepthmapEstimator::BuildKernels() {
  auto& dev = opencl::CLContext::Instance().Device(device_idx_);
  program_ = dev.GetOrBuildProgram("patchmatch", kPatchMatchKernelSource);

  cl_int err;
  k_random_init_ = cl::Kernel(program_, "acmmp_random_init", &err);
  opencl::CheckCL(err, "kernel acmmp_random_init");

  k_patchmatch_red_ = cl::Kernel(program_, "acmmp_patchmatch", &err);
  opencl::CheckCL(err, "kernel acmmp_patchmatch (red)");

  k_patchmatch_black_ = cl::Kernel(program_, "acmmp_patchmatch", &err);
  opencl::CheckCL(err, "kernel acmmp_patchmatch (black)");

  k_upsample_ = cl::Kernel(program_, "acmmp_upsample", &err);
  opencl::CheckCL(err, "kernel acmmp_upsample");

  k_prior_reinit_ = cl::Kernel(program_, "acmmp_prior_reinit", &err);
  opencl::CheckCL(err, "kernel acmmp_prior_reinit");

  k_checkerboard_filter_red_ =
      cl::Kernel(program_, "acmmp_checkerboard_filter", &err);
  opencl::CheckCL(err, "kernel acmmp_checkerboard_filter (red)");

  k_checkerboard_filter_black_ =
      cl::Kernel(program_, "acmmp_checkerboard_filter", &err);
  opencl::CheckCL(err, "kernel acmmp_checkerboard_filter (black)");

  // SLIC segmentation kernels (separate program).
  if (params_.segmentation_enabled) {
    slic_program_ = dev.GetOrBuildProgram("slic", kSLICKernelSource);
    k_slic_init_ = cl::Kernel(slic_program_, "slic_init_centers", &err);
    opencl::CheckCL(err, "kernel slic_init_centers");
    k_slic_assign_ = cl::Kernel(slic_program_, "slic_assign_pixels", &err);
    opencl::CheckCL(err, "kernel slic_assign_pixels");
    k_slic_update_ = cl::Kernel(slic_program_, "slic_update_centers", &err);
    opencl::CheckCL(err, "kernel slic_update_centers");
    slic_enabled_ = true;
  }
}

// =====================================================================
// Upload images and cameras to OpenCL device memory
// =====================================================================
void DepthmapEstimator::UploadData() {
  auto& dev = opencl::CLContext::Instance().Device(device_idx_);
  auto& ctx = dev.context();
  auto& queue = dev.queue();
  cl_int err;

  const int num = static_cast<int>(images_.size());
  const int w = images_[0].cols;
  const int h = images_[0].rows;
  const int npix = w * h;

  const int kTotalSlots = 1 + kMaxSources;  // 1 ref + MAX_SOURCES src

  // Upload reference image as a separate Image2D.
  {
    cv::Mat fimg;
    images_[0].convertTo(fimg, CV_32F, 1.0 / 255.0);
    cv::Mat himg;
    fimg.convertTo(himg, CV_16F);

    cl::ImageFormat fmt(CL_R, CL_HALF_FLOAT);
    cl_ref_img_ = cl::Image2D(ctx, CL_MEM_READ_ONLY, fmt, himg.cols, himg.rows,
                              0, nullptr, &err);
    opencl::CheckCL(err, "Image2D ref upload (half)");
    cl::array<cl::size_type, 3> origin = {{0, 0, 0}};
    cl::array<cl::size_type, 3> region = {
        {static_cast<cl::size_type>(himg.cols),
         static_cast<cl::size_type>(himg.rows), 1}};
    queue.enqueueWriteImage(cl_ref_img_, CL_TRUE, origin, region, himg.step[0],
                            0, himg.data);
  }

  // Upload source images as Image2DArray (branchless texture indexing).
  // Always allocate kMaxSources layers; unused layers are zeroed.
  // Format: CL_HALF_FLOAT — halves texture bandwidth, enables read_imageh,
  // and unlocks double-rate FP16 texture units on Apple Silicon.
  {
    cl::ImageFormat fmt(CL_R, CL_HALF_FLOAT);
    cl_src_images_array_ = cl::Image2DArray(
        ctx, CL_MEM_READ_ONLY, fmt, static_cast<cl::size_type>(kMaxSources),
        static_cast<cl::size_type>(w), static_cast<cl::size_type>(h), 0, 0,
        nullptr, &err);
    opencl::CheckCL(err, "Image2DArray src_images (half)");

    // Write each source image to its layer.
    for (int i = 0; i < kMaxSources; i++) {
      cv::Mat fimg;
      if (i + 1 < num) {
        images_[i + 1].convertTo(fimg, CV_32F, 1.0 / 255.0);
      } else {
        // Unused layer: zero-filled image at correct size.
        fimg = cv::Mat::zeros(h, w, CV_32F);
      }
      cv::Mat himg;
      fimg.convertTo(himg, CV_16F);
      cl::array<cl::size_type, 3> origin = {
          {0, 0, static_cast<cl::size_type>(i)}};
      cl::array<cl::size_type, 3> region = {
          {static_cast<cl::size_type>(w), static_cast<cl::size_type>(h), 1}};
      queue.enqueueWriteImage(cl_src_images_array_, CL_TRUE, origin, region,
                              himg.step[0], 0, himg.data);
    }
  }

  // Upload cameras.
  std::vector<CLCamera> cams;
  for (int i = 0; i < num; i++) {
    cams.push_back(
        MakeCLCamera(Ks_[i], Rs_[i], ts_[i], images_[i].cols, images_[i].rows));
  }
  // Pad to kTotalSlots.
  while (static_cast<int>(cams.size()) < kTotalSlots) {
    cams.push_back(cams.back());
  }

  cl_cameras_ = cl::Buffer(ctx, CL_MEM_READ_ONLY,
                           sizeof(CLCamera) * cams.size(), nullptr, &err);
  opencl::CheckCL(err, "cameras buffer");
  queue.enqueueWriteBuffer(cl_cameras_, CL_TRUE, 0,
                           sizeof(CLCamera) * cams.size(), cams.data());

  // Compute per-source-view angle weights.
  // Weight = sin(triangulation_angle) so that wide-baseline views dominate.
  // Camera center = -R^T * t.
  {
    Vec3d c0 = -Rs_[0].transpose() * ts_[0];
    std::vector<float> aw(kMaxSources, 0.0f);
    for (int i = 0; i < std::min(num - 1, kMaxSources); i++) {
      Vec3d ci = -Rs_[i + 1].transpose() * ts_[i + 1];
      Vec3d baseline = ci - c0;
      double blen = baseline.norm();
      if (blen < 1e-12) {
        aw[i] = 0.01f;  // degenerate: near-zero baseline
      } else {
        // Optical axis of ref camera is 3rd row of R (z-axis in cam frame).
        Vec3d z0 = Rs_[0].row(2).transpose();
        // Angle between baseline direction and ref optical axis.
        double cos_angle = std::abs(z0.dot(baseline / blen));
        // sin(angle) emphasises views with perpendicular baselines.
        double sin_angle = std::sqrt(1.0 - cos_angle * cos_angle);
        aw[i] = static_cast<float>(std::max(sin_angle, 0.01));
      }
    }
    cl_angle_weights_ = cl::Buffer(ctx, CL_MEM_READ_ONLY,
                                   sizeof(float) * kMaxSources, nullptr, &err);
    opencl::CheckCL(err, "angle_weights buffer");
    queue.enqueueWriteBuffer(cl_angle_weights_, CL_TRUE, 0,
                             sizeof(float) * kMaxSources, aw.data());
  }

  // Allocate segment labels buffer (full-res ints).
  // Filled with -1 (no gating) by default; RunSLIC() writes real labels.
  {
    std::vector<int> neg_ones(npix, -1);
    cl_segment_labels_ =
        cl::Buffer(ctx, CL_MEM_READ_ONLY, sizeof(int) * npix, nullptr, &err);
    opencl::CheckCL(err, "segment_labels buffer");
    queue.enqueueWriteBuffer(cl_segment_labels_, CL_TRUE, 0, sizeof(int) * npix,
                             neg_ones.data());
  }

  // Allocate plane hypotheses, costs, rand states.
  // Detect cl_khr_image2d_from_buffer for zero-copy buffer↔image aliasing.
  // {
  //   std::string extensions = dev.device().getInfo<CL_DEVICE_EXTENSIONS>();
  //   use_image_from_buffer_ =
  //       (extensions.find("cl_khr_image2d_from_buffer") != std::string::npos);
  // }

  if (use_image_from_buffer_) {
    // Query row pitch alignment (in pixels).  Buffer rows must be aligned.
    cl_uint pitch_align = 1;
    dev.device().getInfo(CL_DEVICE_IMAGE_PITCH_ALIGNMENT, &pitch_align);
    if (pitch_align < 1) {
      pitch_align = 1;
    }

    size_t aligned_w_planes =
        ((w + pitch_align - 1) / pitch_align) * pitch_align;
    size_t aligned_w_costs =
        ((w + pitch_align - 1) / pitch_align) * pitch_align;
    size_t planes_pitch = aligned_w_planes * sizeof(cl_float4);
    size_t costs_pitch = aligned_w_costs * sizeof(cl_float);

    // Allocate buffers with pitch-aligned rows.
    cl_plane_hypotheses_ =
        cl::Buffer(ctx, CL_MEM_READ_WRITE, planes_pitch * h, nullptr, &err);
    opencl::CheckCL(err, "plane_hypotheses buffer (aliased)");

    cl_costs_ =
        cl::Buffer(ctx, CL_MEM_READ_WRITE, costs_pitch * h, nullptr, &err);
    opencl::CheckCL(err, "costs buffer (aliased)");

    // Create Image2D objects aliasing the buffer memory (zero-copy).
    cl::ImageFormat fmt_rgba(CL_RGBA, CL_FLOAT);
    cl_planes_img_ = cl::Image2D(ctx, fmt_rgba, cl_plane_hypotheses_, w, h,
                                 planes_pitch, &err);
    opencl::CheckCL(err, "planes_img from buffer");

    cl::ImageFormat fmt_r(CL_R, CL_FLOAT);
    cl_costs_img_ = cl::Image2D(ctx, fmt_r, cl_costs_, w, h, costs_pitch, &err);
    opencl::CheckCL(err, "costs_img from buffer");
  } else {
    // Fallback: separate buffer + image, synced via enqueueCopyBufferToImage.
    cl_plane_hypotheses_ = cl::Buffer(ctx, CL_MEM_READ_WRITE,
                                      sizeof(cl_float4) * npix, nullptr, &err);
    opencl::CheckCL(err, "plane_hypotheses buffer");

    cl_costs_ =
        cl::Buffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * npix, nullptr, &err);
    opencl::CheckCL(err, "costs buffer");

    cl::ImageFormat fmt_rgba(CL_RGBA, CL_FLOAT);
    cl_planes_img_ =
        cl::Image2D(ctx, CL_MEM_READ_WRITE, fmt_rgba, w, h, 0, nullptr, &err);
    opencl::CheckCL(err, "planes_img");

    cl::ImageFormat fmt_r(CL_R, CL_FLOAT);
    cl_costs_img_ =
        cl::Image2D(ctx, CL_MEM_READ_WRITE, fmt_r, w, h, 0, nullptr, &err);
    opencl::CheckCL(err, "costs_img");
  }

  cl_rand_states_ = cl::Buffer(ctx, CL_MEM_READ_WRITE, sizeof(cl_uint2) * npix,
                               nullptr, &err);
  opencl::CheckCL(err, "rand_states buffer");

  cl_selected_views_ =
      cl::Buffer(ctx, CL_MEM_READ_WRITE, sizeof(cl_uint) * npix, nullptr, &err);
  opencl::CheckCL(err, "selected_views buffer");

  // Allocate prior buffers (zeroed — populated later if planar prior is used).
  cl_prior_planes_ = cl::Buffer(ctx, CL_MEM_READ_WRITE,
                                sizeof(cl_float4) * npix, nullptr, &err);
  opencl::CheckCL(err, "prior_planes buffer");

  cl_plane_masks_ =
      cl::Buffer(ctx, CL_MEM_READ_WRITE, sizeof(cl_uint) * npix, nullptr, &err);
  opencl::CheckCL(err, "plane_masks buffer");

  // Zero-fill mask and selected_views buffers so PatchMatch starts
  // without prior influence and with uniform view priors.  Without
  // this, upsampled levels would read garbage selected_views in the
  // first PatchMatch iteration, biasing view selection randomly.
  std::vector<cl_uint> zeros(npix, 0u);
  queue.enqueueWriteBuffer(cl_plane_masks_, CL_TRUE, 0, sizeof(cl_uint) * npix,
                           zeros.data());

  queue.enqueueWriteBuffer(cl_selected_views_, CL_TRUE, 0,
                           sizeof(cl_uint) * npix, zeros.data());

  // Allocate previous-depth buffer for geometric consistency.
  // MAX_SOURCES * npix floats, initially zeroed.
  const size_t prev_depth_bytes = kMaxSources * npix * sizeof(float);
  cl_prev_depths_ =
      cl::Buffer(ctx, CL_MEM_READ_ONLY, prev_depth_bytes, nullptr, &err);
  opencl::CheckCL(err, "prev_depths buffer");

  std::vector<float> prev_zeros(kMaxSources * npix, 0.0f);
  queue.enqueueWriteBuffer(cl_prev_depths_, CL_TRUE, 0, prev_depth_bytes,
                           prev_zeros.data());
  cl_prev_depth_mask_ = 0u;
}

// =====================================================================
// Random initialisation on GPU
// =====================================================================
void DepthmapEstimator::RandomInit(int width, int height) {
  auto& dev = opencl::CLContext::Instance().Device(device_idx_);
  auto& queue = dev.queue();

  int half_patch = params_.patch_size / 2;
  int arg = 0;
  k_random_init_.setArg(arg++, cl_plane_hypotheses_);
  k_random_init_.setArg(arg++, cl_costs_);
  k_random_init_.setArg(arg++, cl_rand_states_);
  k_random_init_.setArg(arg++, cl_selected_views_);
  k_random_init_.setArg(arg++, cl_cameras_);
  k_random_init_.setArg(arg++, cl_ref_img_);
  k_random_init_.setArg(arg++, cl_src_images_array_);
  k_random_init_.setArg(arg++, width);
  k_random_init_.setArg(arg++, height);
  k_random_init_.setArg(arg++, params_.depth_min);
  k_random_init_.setArg(arg++, params_.depth_max);
  k_random_init_.setArg(arg++, params_.num_images);
  k_random_init_.setArg(arg++, half_patch);
  k_random_init_.setArg(arg++, params_.sigma_spatial);
  k_random_init_.setArg(arg++, params_.sigma_color);
  k_random_init_.setArg(arg++, params_.top_k);
  k_random_init_.setArg(arg++, params_.use_census ? 1.0f : 0.0f);
  k_random_init_.setArg(arg++, params_.center_color_weight);

  cl::NDRange global(static_cast<size_t>((width + 15) / 16 * 16),
                     static_cast<size_t>((height + 15) / 16 * 16));
  cl::NDRange local(16, 16);
  queue.enqueueNDRangeKernel(k_random_init_, cl::NullRange, global, local);
  // queue.finish();

  SyncBuffersToImages(width, height);
}

// =====================================================================
// Copy plane/cost buffers to their image2d_t counterparts for texture
// cache reads.  When cl_khr_image2d_from_buffer is active, the images
// alias the buffers directly — no copy needed, in-order queue ordering
// guarantees visibility between kernels.
// =====================================================================
void DepthmapEstimator::SyncBuffersToImages(int width, int height) {
  if (use_image_from_buffer_) {
    return;  // zero-copy: images alias buffers
  }

  auto& queue = opencl::CLContext::Instance().Device(device_idx_).queue();
  cl::array<cl::size_type, 3> origin = {{0, 0, 0}};
  cl::array<cl::size_type, 3> region = {{static_cast<cl::size_type>(width),
                                         static_cast<cl::size_type>(height),
                                         1}};

  queue.enqueueCopyBufferToImage(cl_plane_hypotheses_, cl_planes_img_, 0,
                                 origin, region);
  queue.enqueueCopyBufferToImage(cl_costs_, cl_costs_img_, 0, origin, region);
}

// =====================================================================
// GPU-side planar prior re-initialisation (matches reference
// RandomInitialization with planar_prior=true).
// =====================================================================
void DepthmapEstimator::PriorReinit(int width, int height) {
  auto& dev = opencl::CLContext::Instance().Device(device_idx_);
  auto& queue = dev.queue();

  SyncBuffersToImages(width, height);

  int half_patch = params_.patch_size / 2;
  int arg = 0;
  k_prior_reinit_.setArg(arg++, cl_plane_hypotheses_);
  k_prior_reinit_.setArg(arg++, cl_costs_);
  k_prior_reinit_.setArg(arg++, cl_planes_img_);
  k_prior_reinit_.setArg(arg++, cl_costs_img_);
  k_prior_reinit_.setArg(arg++, cl_rand_states_);
  k_prior_reinit_.setArg(arg++, cl_selected_views_);
  k_prior_reinit_.setArg(arg++, cl_cameras_);
  k_prior_reinit_.setArg(arg++, cl_ref_img_);
  k_prior_reinit_.setArg(arg++, cl_src_images_array_);
  k_prior_reinit_.setArg(arg++, width);
  k_prior_reinit_.setArg(arg++, height);
  k_prior_reinit_.setArg(arg++, params_.depth_min);
  k_prior_reinit_.setArg(arg++, params_.depth_max);
  k_prior_reinit_.setArg(arg++, params_.num_images);
  k_prior_reinit_.setArg(arg++, half_patch);
  k_prior_reinit_.setArg(arg++, params_.sigma_spatial);
  k_prior_reinit_.setArg(arg++, params_.sigma_color);
  k_prior_reinit_.setArg(arg++, params_.top_k);
  k_prior_reinit_.setArg(arg++, params_.use_census ? 1.0f : 0.0f);
  k_prior_reinit_.setArg(arg++, cl_prior_planes_);
  k_prior_reinit_.setArg(arg++, cl_plane_masks_);
  // Cost-parity arguments: match acmmp_patchmatch's total cost.
  k_prior_reinit_.setArg(arg++, params_.smooth_weight);
  k_prior_reinit_.setArg(arg++, cl_prev_depths_);
  k_prior_reinit_.setArg(arg++, cl_prev_depth_mask_);
  k_prior_reinit_.setArg(arg++, geom_weight_);
  k_prior_reinit_.setArg(arg++, params_.center_color_weight);

  cl::NDRange global(static_cast<size_t>((width + 15) / 16 * 16),
                     static_cast<size_t>((height + 15) / 16 * 16));
  cl::NDRange local(16, 16);
  queue.enqueueNDRangeKernel(k_prior_reinit_, cl::NullRange, global, local);
  // queue.finish();
}

// =====================================================================
// One PatchMatch iteration (red then black)
// =====================================================================
void DepthmapEstimator::RunIteration(int iter, int width, int height) {
  auto& queue = opencl::CLContext::Instance().Device(device_idx_).queue();
  int half_patch = params_.patch_size / 2;

  auto set_pm_args = [&](cl::Kernel& k, int color_flag) {
    int arg = 0;
    k.setArg(arg++, cl_plane_hypotheses_);
    k.setArg(arg++, cl_costs_);
    k.setArg(arg++, cl_planes_img_);
    k.setArg(arg++, cl_costs_img_);
    k.setArg(arg++, cl_rand_states_);
    k.setArg(arg++, cl_selected_views_);
    k.setArg(arg++, cl_cameras_);
    k.setArg(arg++, cl_ref_img_);
    k.setArg(arg++, cl_src_images_array_);
    k.setArg(arg++, width);
    k.setArg(arg++, height);
    k.setArg(arg++, params_.depth_min);
    k.setArg(arg++, params_.depth_max);
    k.setArg(arg++, params_.num_images);
    k.setArg(arg++, half_patch);
    k.setArg(arg++, params_.sigma_spatial);
    k.setArg(arg++, params_.sigma_color);
    k.setArg(arg++, params_.top_k);
    k.setArg(arg++, params_.use_census ? 1.0f : 0.0f);
    k.setArg(arg++, color_flag);
    k.setArg(arg++, iter);
    float decay =
        1.0f - 0.9f * static_cast<float>(iter) /
                   static_cast<float>(std::max(params_.max_iterations - 1, 1));
    k.setArg(arg++, params_.smooth_weight * decay);
    // Geometric consistency arguments.
    k.setArg(arg++, cl_prev_depths_);
    k.setArg(arg++, cl_prev_depth_mask_);
    k.setArg(arg++, geom_weight_);
    // Planar prior arguments.
    k.setArg(arg++, cl_prior_planes_);
    k.setArg(arg++, cl_plane_masks_);
    // Edge-stopping propagation gate.
    k.setArg(arg++, params_.edge_weight);
    // Depth-discontinuity escape heuristic.
    k.setArg(arg++, params_.escape_depth_ratio);
    // Center-pixel color consensus and variance gate.
    k.setArg(arg++, params_.center_color_weight);
    k.setArg(arg++, params_.variance_gate);
    // View-selection anchoring and far-propagation gradient gate.
    k.setArg(arg++, params_.anchor_views);
    k.setArg(arg++, params_.far_gradient_threshold);
    // Per-source-view angle weights (sin of triangulation angle).
    k.setArg(arg++, cl_angle_weights_);
    // SLIC segment labels (per-pixel, full-res).
    k.setArg(arg++, cl_segment_labels_);
  };

  // Dense checkerboard: launch half-width grid so all SIMD lanes are active.
  int half_width = (width + 1) / 2;
  cl::NDRange global(static_cast<size_t>((half_width + 15) / 16 * 16),
                     static_cast<size_t>((height + 15) / 16 * 16));
  cl::NDRange local(16, 16);

  // Red pass (x+y even)
  SyncBuffersToImages(width, height);
  set_pm_args(k_patchmatch_red_, 0);
  queue.enqueueNDRangeKernel(k_patchmatch_red_, cl::NullRange, global, local);
  // queue.finish();

  // Black pass (x+y odd)
  SyncBuffersToImages(width, height);
  set_pm_args(k_patchmatch_black_, 1);
  queue.enqueueNDRangeKernel(k_patchmatch_black_, cl::NullRange, global, local);
  // queue.finish();
}

// =====================================================================
// Checkerboard median filter (21 neighbours, red then black)
// =====================================================================
void DepthmapEstimator::RunCheckerboardFilter(int width, int height) {
  auto& queue = opencl::CLContext::Instance().Device(device_idx_).queue();

  auto set_filter_args = [&](cl::Kernel& k, int color_flag) {
    int arg = 0;
    k.setArg(arg++, cl_plane_hypotheses_);
    k.setArg(arg++, cl_costs_);
    k.setArg(arg++, cl_planes_img_);
    k.setArg(arg++, cl_costs_img_);
    k.setArg(arg++, cl_cameras_);
    k.setArg(arg++, width);
    k.setArg(arg++, height);
    k.setArg(arg++, color_flag);
  };

  // Dense checkerboard: launch half-width grid so all SIMD lanes are active.
  int half_width = (width + 1) / 2;
  cl::NDRange global(static_cast<size_t>((half_width + 15) / 16 * 16),
                     static_cast<size_t>((height + 15) / 16 * 16));
  cl::NDRange local(16, 16);

  // Filter red pixels (reading from black neighbours).
  SyncBuffersToImages(width, height);
  set_filter_args(k_checkerboard_filter_red_, 0);
  queue.enqueueNDRangeKernel(k_checkerboard_filter_red_, cl::NullRange, global,
                             local);
  // queue.finish();

  // Filter black pixels (reading from red neighbours).
  SyncBuffersToImages(width, height);
  set_filter_args(k_checkerboard_filter_black_, 1);
  queue.enqueueNDRangeKernel(k_checkerboard_filter_black_, cl::NullRange,
                             global, local);
  // queue.finish();
}

// =====================================================================
// Build image pyramid (downscale by factor)
// =====================================================================
void DepthmapEstimator::BuildImagePyramid(float scale,
                                          std::vector<cv::Mat>& scaled_images,
                                          std::vector<Mat3d>& scaled_Ks) const {
  scaled_images.clear();
  scaled_Ks.clear();

  for (size_t i = 0; i < images_.size(); i++) {
    int new_w = static_cast<int>(images_[i].cols * scale);
    int new_h = static_cast<int>(images_[i].rows * scale);
    new_w = std::max(new_w, 1);
    new_h = std::max(new_h, 1);

    cv::Mat scaled;
    cv::resize(images_[i], scaled, cv::Size(new_w, new_h), 0, 0,
               cv::INTER_AREA);
    scaled_images.push_back(scaled);

    float sx = static_cast<float>(new_w) / images_[i].cols;
    float sy = static_cast<float>(new_h) / images_[i].rows;

    Mat3d K = Ks_[i];
    K(0, 0) *= sx;
    K(0, 2) *= sx;
    K(1, 1) *= sy;
    K(1, 2) *= sy;
    scaled_Ks.push_back(K);
  }
}

// =====================================================================
// Upsample depth/normal from coarse to full resolution
// =====================================================================
void DepthmapEstimator::UpsampleDepthNormal(const DepthmapResult& coarse,
                                            int src_w, int src_h, int dst_w,
                                            int dst_h) {
  auto& dev = opencl::CLContext::Instance().Device(device_idx_);
  auto& ctx = dev.context();
  auto& queue = dev.queue();
  cl_int err;

  int src_npix = src_w * src_h;
  int dst_npix = dst_w * dst_h;

  // Upload coarse planes.
  // Coarse result has depth + normal; we need to pack as float4 planes.
  std::vector<cl_float4> src_planes(src_npix);
  std::vector<float> src_costs(src_npix, 2.0f);

  CLCamera cam0 = MakeCLCamera(Ks_[0], Rs_[0], ts_[0], dst_w, dst_h);

  for (int y = 0; y < src_h; y++) {
    for (int x = 0; x < src_w; x++) {
      int idx = y * src_w + x;
      float d = coarse.depth(y, x);
      Vec3f n = coarse.normal.col(idx);
      float fx = cam0.K[0] * static_cast<float>(src_w) / dst_w;
      float fy = cam0.K[4] * static_cast<float>(src_h) / dst_h;
      float cx_s = cam0.K[2] * static_cast<float>(src_w) / dst_w;
      float cy_s = cam0.K[5] * static_cast<float>(src_h) / dst_h;
      Vec3f pt_cam(d * (x - cx_s) / fx, d * (y - cy_s) / fy, d);
      float dp = -n.dot(pt_cam);
      src_planes[idx] = {n.x(), n.y(), n.z(), dp};
      src_costs[idx] = coarse.cost(y, x);
    }
  }

  cl::Buffer cl_src_planes(ctx, CL_MEM_READ_ONLY, sizeof(cl_float4) * src_npix,
                           nullptr, &err);
  opencl::CheckCL(err, "upsample src_planes");
  queue.enqueueWriteBuffer(cl_src_planes, CL_TRUE, 0,
                           sizeof(cl_float4) * src_npix, src_planes.data());

  cl::Buffer cl_src_costs(ctx, CL_MEM_READ_ONLY, sizeof(float) * src_npix,
                          nullptr, &err);
  opencl::CheckCL(err, "upsample src_costs");
  queue.enqueueWriteBuffer(cl_src_costs, CL_TRUE, 0, sizeof(float) * src_npix,
                           src_costs.data());

  int arg = 0;
  k_upsample_.setArg(arg++, cl_plane_hypotheses_);
  k_upsample_.setArg(arg++, cl_costs_);
  k_upsample_.setArg(arg++, cl_src_planes);
  k_upsample_.setArg(arg++, cl_src_costs);
  k_upsample_.setArg(arg++, cl_cameras_);
  k_upsample_.setArg(arg++, cl_rand_states_);
  k_upsample_.setArg(arg++, dst_w);
  k_upsample_.setArg(arg++, dst_h);
  k_upsample_.setArg(arg++, src_w);
  k_upsample_.setArg(arg++, src_h);
  k_upsample_.setArg(arg++, params_.depth_min);
  k_upsample_.setArg(arg++, params_.depth_max);

  cl::NDRange global(static_cast<size_t>((dst_w + 15) / 16 * 16),
                     static_cast<size_t>((dst_h + 15) / 16 * 16));
  cl::NDRange local(16, 16);
  queue.enqueueNDRangeKernel(k_upsample_, cl::NullRange, global, local);
  // queue.finish();
}

// =====================================================================
// SfM-based planar prior (CPU-side Delaunay + SVD plane fitting)
//
// Unlike the old depthmap-derived ComputePlanarPrior, this method uses
// known SfM 3D points that were triangulated during reconstruction:
//   1. Project sfm_points_ into the current reference camera to get
//      2D pixel positions + depths (using Ks_[0], Rs_[0], ts_[0]).
//   2. Filter: keep only projections inside the image with valid depth.
//   3. Delaunay triangulation of the 2D projections.
//   4. For each triangle, fit a plane via SVD through the 3 vertices in
//      camera coordinates.  All pixels inside share the same plane.
//   5. Parametric rasterization of triangles into a mask.
//   6. Validate depth from plane against [depth_min, depth_max].
//   7. Upload prior_planes and plane_masks to GPU.
// =====================================================================
void DepthmapEstimator::ComputeSfMPlanarPrior(int w, int h) {
  CLCamera cam0 = MakeCLCamera(Ks_[0], Rs_[0], ts_[0], w, h);
  const float fx = cam0.K[0], fy = cam0.K[4], cx = cam0.K[2], cy = cam0.K[5];

  // R and t from the reference camera (world-to-camera transform).
  const Mat3d& R = Rs_[0];
  const Vec3d& t = ts_[0];

  // ---- 1. Project SfM points into the reference camera ----
  struct Proj {
    cv::Point2f px;  // 2D pixel coordinate
    float depth;     // depth in camera coordinates (Z_cam)
    float cam[3];    // camera-space 3D coordinate
  };
  std::vector<Proj> projections;
  projections.reserve(sfm_points_.size());

  for (const auto& wp : sfm_points_) {
    // World-to-camera: X_cam = R * X_world + t
    Vec3d cam = R * wp + t;
    double xc = cam.x(), yc = cam.y(), zc = cam.z();

    if (zc <= 0.0) {
      continue;  // behind camera
    }

    float depth = static_cast<float>(zc);
    if (depth < params_.depth_min || depth > params_.depth_max) {
      continue;
    }

    // Project to pixel coordinates.
    float u = static_cast<float>(fx * (xc / zc) + cx);
    float v = static_cast<float>(fy * (yc / zc) + cy);

    // Must be inside image with a small margin.
    if (u < 0.0f || u >= w || v < 0.0f || v >= h) {
      continue;
    }

    Proj p;
    p.px = cv::Point2f(u, v);
    p.depth = depth;
    p.cam[0] = static_cast<float>(xc);
    p.cam[1] = static_cast<float>(yc);
    p.cam[2] = static_cast<float>(zc);
    projections.push_back(p);
  }

  {
    std::ostringstream oss;
    oss << "[PatchMatch]   SfM planar prior: " << projections.size() << "/"
        << sfm_points_.size() << " points project into " << w << "x" << h
        << " image";
    foundation::LogDebug("dense", oss.str());
  }

  if (projections.size() < 4) {
    foundation::LogDebug(
        "dense", "[PatchMatch]   Too few projected points, skipping SfM prior");
    // Upload empty masks so PriorReinit is a no-op.
    int npix = w * h;
    std::vector<cl_uint> masks_buf(npix, 0u);
    auto& queue = opencl::CLContext::Instance().Device(device_idx_).queue();
    queue.enqueueWriteBuffer(cl_plane_masks_, CL_TRUE, 0,
                             sizeof(cl_uint) * npix, masks_buf.data());
    return;
  }

  // ---- 2. Delaunay triangulation ----
  cv::Rect imageRC(0, 0, w, h);
  cv::Subdiv2D subdiv(imageRC);
  // Map pixel position → projection index for vertex depth lookup.
  std::unordered_map<int, int> px_to_proj;
  for (int i = 0; i < static_cast<int>(projections.size()); ++i) {
    const auto& p = projections[i];
    subdiv.insert(p.px);
    // Key: quantised pixel (int coords).
    int key = static_cast<int>(p.px.y) * w + static_cast<int>(p.px.x);
    px_to_proj[key] = i;
  }

  std::vector<cv::Vec6f> raw_triangles;
  subdiv.getTriangleList(raw_triangles);

  // ---- 3. SVD plane fitting per triangle ----
  struct TriangleData {
    cv::Point pt1, pt2, pt3;
    cl_float4 plane;  // (nx, ny, nz, d) in camera coords
  };
  std::vector<TriangleData> triangles;

  for (const auto& tri : raw_triangles) {
    cv::Point pt1(static_cast<int>(tri[0]), static_cast<int>(tri[1]));
    cv::Point pt2(static_cast<int>(tri[2]), static_cast<int>(tri[3]));
    cv::Point pt3(static_cast<int>(tri[4]), static_cast<int>(tri[5]));

    if (!imageRC.contains(pt1) || !imageRC.contains(pt2) ||
        !imageRC.contains(pt3)) {
      continue;
    }

    // Look up camera-space 3D coords for each vertex.
    auto it1 = px_to_proj.find(pt1.y * w + pt1.x);
    auto it2 = px_to_proj.find(pt2.y * w + pt2.x);
    auto it3 = px_to_proj.find(pt3.y * w + pt3.x);
    if (it1 == px_to_proj.end() || it2 == px_to_proj.end() ||
        it3 == px_to_proj.end()) {
      continue;
    }

    const float* X1 = projections[it1->second].cam;
    const float* X2 = projections[it2->second].cam;
    const float* X3 = projections[it3->second].cam;

    // ---- Triangle quality gates ----
    // Reject triangles spanning depth discontinuities: if the deepest
    // vertex is more than 1.5× the shallowest, the planar assumption
    // across the triangle is unreliable (e.g. building edge vs sky).
    {
      float d1 = projections[it1->second].depth;
      float d2 = projections[it2->second].depth;
      float d3 = projections[it3->second].depth;
      float d_min = std::min({d1, d2, d3});
      float d_max = std::max({d1, d2, d3});
      if (d_min > 0.0f && d_max / d_min > 1.5f) {
        continue;
      }
    }

    // Reject very large triangles where the SfM surface is speculative.
    {
      float ax = static_cast<float>(pt2.x - pt1.x);
      float ay = static_cast<float>(pt2.y - pt1.y);
      float bx = static_cast<float>(pt3.x - pt1.x);
      float by = static_cast<float>(pt3.y - pt1.y);
      float area = 0.5f * std::abs(ax * by - ay * bx);
      if (area > 20000.0f) {
        continue;
      }
    }

    // Fit plane n·X + d = 0 via SVD null-space.
    Eigen::Matrix<float, 3, 4> A;
    A << X1[0], X1[1], X1[2], 1.0f, X2[0], X2[1], X2[2], 1.0f, X3[0], X3[1],
        X3[2], 1.0f;
    Eigen::JacobiSVD<Eigen::Matrix<float, 3, 4>> svd(A, Eigen::ComputeFullV);
    Eigen::Vector4f B = svd.matrixV().col(3);
    float norm2 = B.head<3>().norm();
    if (norm2 < 1e-8f) {
      continue;
    }
    if (B(3) < 0) {
      norm2 = -norm2;  // ensure d > 0
    }
    B /= norm2;

    TriangleData td;
    td.pt1 = pt1;
    td.pt2 = pt2;
    td.pt3 = pt3;
    td.plane = {{B(0), B(1), B(2), B(3)}};
    triangles.push_back(td);
  }

  if (triangles.empty()) {
    int npix = w * h;
    std::vector<cl_uint> masks_buf(npix, 0u);
    auto& queue = opencl::CLContext::Instance().Device(device_idx_).queue();
    queue.enqueueWriteBuffer(cl_plane_masks_, CL_TRUE, 0,
                             sizeof(cl_uint) * npix, masks_buf.data());
    // Clear host-side prior data.
    prior_depths_ = ImageF::Zero(h, w);
    triangle_areas_ = ImageF::Zero(h, w);
    prior_mask_ = ImageU8::Zero(h, w);
    return;
  }

  // ---- 4. Compute triangle pixel areas ----
  std::vector<float> tri_areas(triangles.size());
  for (size_t ti = 0; ti < triangles.size(); ++ti) {
    const auto& td = triangles[ti];
    // Pixel-space triangle area via cross product.
    float ax = static_cast<float>(td.pt2.x - td.pt1.x);
    float ay = static_cast<float>(td.pt2.y - td.pt1.y);
    float bx = static_cast<float>(td.pt3.x - td.pt1.x);
    float by = static_cast<float>(td.pt3.y - td.pt1.y);
    tri_areas[ti] = 0.5f * std::abs(ax * by - ay * bx);
  }

  // ---- 5. Rasterize triangles ----
  ImageF mask_tri = ImageF::Zero(h, w);

  for (size_t ti = 0; ti < triangles.size(); ++ti) {
    const auto& td = triangles[ti];
    float L01 = std::hypot(static_cast<float>(td.pt1.x - td.pt2.x),
                           static_cast<float>(td.pt1.y - td.pt2.y));
    float L02 = std::hypot(static_cast<float>(td.pt1.x - td.pt3.x),
                           static_cast<float>(td.pt1.y - td.pt3.y));
    float L12 = std::hypot(static_cast<float>(td.pt2.x - td.pt3.x),
                           static_cast<float>(td.pt2.y - td.pt3.y));
    float max_edge = std::max(L01, std::max(L02, L12));
    if (max_edge < 1.0f) {
      continue;
    }
    float step_f = 1.0f / max_edge;

    for (float p = 0.0f; p < 1.0f; p += step_f) {
      for (float q = 0.0f; q < 1.0f - p; q += step_f) {
        int px = static_cast<int>(p * td.pt1.x + q * td.pt2.x +
                                  (1.0f - p - q) * td.pt3.x);
        int py = static_cast<int>(p * td.pt1.y + q * td.pt2.y +
                                  (1.0f - p - q) * td.pt3.y);
        if (px >= 0 && px < w && py >= 0 && py < h) {
          mask_tri(py, px) = static_cast<float>(ti + 1);
        }
      }
    }
  }

  // ---- 6. Build prior buffers and validate depths ----
  int npix = w * h;
  std::vector<cl_float4> prior_planes_buf(npix, {{0.0f, 0.0f, -1.0f, 0.0f}});
  std::vector<cl_uint> masks_buf(npix, 0u);
  int filled = 0;

  // Also store host-side data for confidence computation.
  prior_depths_ = ImageF::Zero(h, w);
  triangle_areas_ = ImageF::Zero(h, w);
  prior_mask_ = ImageU8::Zero(h, w);

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      int idx = y * w + x;
      float mask_val = mask_tri(y, x);
      if (mask_val > 0.0f) {
        int tri_id = static_cast<int>(mask_val) - 1;
        const cl_float4& plane = triangles[tri_id].plane;

        Vec3f n_plane(plane.s[0], plane.s[1], plane.s[2]);
        Vec3f ray((x - cx) / fx, (y - cy) / fy, 1.0f);
        float denom = n_plane.dot(ray);
        float d = (std::abs(denom) > 1e-6f) ? -plane.s[3] / denom : 0.0f;

        if (d >= params_.depth_min && d <= params_.depth_max) {
          prior_planes_buf[idx] = plane;
          masks_buf[idx] = 1u;
          prior_depths_(y, x) = d;
          triangle_areas_(y, x) = tri_areas[tri_id];
          prior_mask_(y, x) = 1;
          filled++;
        }
      }
    }
  }

  {
    std::ostringstream oss;
    oss << "[PatchMatch]   SfM prior: " << triangles.size() << " triangles, "
        << filled << "/" << npix << " pixels masked";
    foundation::LogDebug("dense", oss.str());
  }

  // ---- 6. Upload prior planes and masks to GPU ----
  auto& queue = opencl::CLContext::Instance().Device(device_idx_).queue();
  queue.enqueueWriteBuffer(cl_prior_planes_, CL_TRUE, 0,
                           sizeof(cl_float4) * npix, prior_planes_buf.data());
  queue.enqueueWriteBuffer(cl_plane_masks_, CL_TRUE, 0, sizeof(cl_uint) * npix,
                           masks_buf.data());
}

// =====================================================================
// Read results from GPU back to host
// =====================================================================
void DepthmapEstimator::ReadBackResults(DepthmapResult* result, int width,
                                        int height, bool apply_median) {
  auto& queue = opencl::CLContext::Instance().Device(device_idx_).queue();
  int npix = width * height;

  std::vector<cl_float4> planes(npix);
  std::vector<float> costs(npix);

  queue.enqueueReadBuffer(cl_plane_hypotheses_, CL_TRUE, 0,
                          sizeof(cl_float4) * npix, planes.data());
  queue.enqueueReadBuffer(cl_costs_, CL_TRUE, 0, sizeof(float) * npix,
                          costs.data());

  result->depth = ImageF::Zero(height, width);
  result->normal = PixelData3f::Zero(3, width * height);
  result->cost = ImageF::Zero(height, width);

  CLCamera cam0 = MakeCLCamera(Ks_[0], Rs_[0], ts_[0], width, height);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = y * width + x;
      auto& p = planes[idx];

      // Recover depth from plane.
      Vec3f n_plane(p.s[0], p.s[1], p.s[2]);

      // Guard against NaN/Inf from GPU readback.
      if (!std::isfinite(p.s[0]) || !std::isfinite(p.s[1]) ||
          !std::isfinite(p.s[2]) || !std::isfinite(p.s[3])) {
        result->cost(y, x) = 0.0f;
        continue;
      }

      Vec3f ray((x - cam0.K[2]) / cam0.K[0], (y - cam0.K[5]) / cam0.K[4], 1.0f);
      float denom = n_plane.dot(ray);
      float depth = (std::abs(denom) > 1e-6f) ? -p.s[3] / denom : 0.0f;

      // Reject NaN/Inf depths.
      if (!std::isfinite(depth)) {
        depth = 0.0f;
      }

      if (depth > 0.0f && depth >= params_.depth_min &&
          depth <= params_.depth_max) {
        result->depth(y, x) = depth;
        float len = n_plane.norm();
        if (len > 1e-6f) {
          result->normal.col(idx) = n_plane / len;
        }
      }
      float cost = costs[idx];
      result->cost(y, x) = std::isfinite(cost) ? cost : 0.0f;
    }
  }

  // Median filter for post-processing.
  // Use a 10% relative threshold to avoid stripping textureless surfaces
  // (asphalt, walls) where depth estimates are noisy but close to correct.
  int total_valid = (result->depth.array() > 0.0f).count();

  int removed = 0;
  if (apply_median) {
    // cv::medianBlur requires cv::Mat; create a zero-copy header over Eigen
    // data.
    cv::Mat depth_cv(height, width, CV_32F, result->depth.data());
    cv::Mat depth_filtered;
    cv::medianBlur(depth_cv, depth_filtered, 5);
    Eigen::Map<ImageF> filtered(depth_filtered.ptr<float>(), height, width);
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        float d = result->depth(y, x);
        float m = filtered(y, x);
        if (d > 0.0f && m > 0.0f && std::abs(d - m) / d > 0.10f) {
          result->depth(y, x) = 0.0f;
          removed++;
        }
      }
    }
  }

  // ---- Speckle removal: BFS flood-fill, remove small segments ----
  if (params_.speckle_min_size > 0) {
    RemoveSmallSegments(result->depth, result->normal, width, height,
                        params_.speckle_min_size, 0.01f);
  }

  // ---- Gap interpolation: fill small holes row-wise + column-wise ----
  if (params_.gap_max_size > 0) {
    GapInterpolation(result->depth, result->normal, width, height,
                     params_.gap_max_size, 0.025f);
  }

  // ---- Surface confidence: logistic on photometric cost ----
  // P(surface) = sigmoid(-(cost - c0) / lambda)
  // Cost already captures match quality; prior is used for PatchMatch guidance,
  // not for post-hoc judgment (avoids double-counting and artifacts in large
  // Delaunay triangles).
  result->confidence = ImageF::Zero(height, width);
  const float c0 = 0.5f;
  const float inv_lambda = 5.0f;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (result->depth(y, x) <= 0.0f) {
        continue;
      }
      float c = result->cost(y, x);
      result->confidence(y, x) =
          1.0f / (1.0f + std::exp((c - c0) * inv_lambda));
    }
  }
}

// =====================================================================
// Speckle removal: BFS flood-fill to find connected components of
// similar depth, then zero out components smaller than min_segment_size.
// depth_diff_threshold is the relative depth difference threshold.
// =====================================================================
void DepthmapEstimator::RemoveSmallSegments(ImageF& depth, PixelData3f& normal,
                                            int width, int height,
                                            int min_segment_size,
                                            float depth_diff_threshold) {
  const int npix = width * height;
  std::vector<bool> visited(npix, false);
  std::vector<int> segment;
  segment.reserve(1024);

  // 4-connectivity offsets: right, left, down, up
  const int dx[4] = {1, -1, 0, 0};
  const int dy[4] = {0, 0, 1, -1};

  int total_removed = 0;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = y * width + x;
      if (visited[idx] || depth(y, x) <= 0.0f) {
        visited[idx] = true;
        continue;
      }

      // BFS flood fill
      segment.clear();
      segment.push_back(idx);
      visited[idx] = true;
      int head = 0;

      while (head < static_cast<int>(segment.size())) {
        int cur = segment[head++];
        int cy = cur / width;
        int cx = cur % width;
        float d_cur = depth(cy, cx);

        for (int dir = 0; dir < 4; ++dir) {
          int nx = cx + dx[dir];
          int ny = cy + dy[dir];
          if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
            continue;
          }
          int nidx = ny * width + nx;
          if (visited[nidx]) {
            continue;
          }
          float d_nbr = depth(ny, nx);
          if (d_nbr <= 0.0f) {
            visited[nidx] = true;
            continue;
          }
          // Relative depth similarity check
          float max_d = std::max(d_cur, d_nbr);
          if (std::abs(d_cur - d_nbr) / max_d < depth_diff_threshold) {
            visited[nidx] = true;
            segment.push_back(nidx);
          }
        }
      }

      // Remove small segments
      if (static_cast<int>(segment.size()) < min_segment_size) {
        for (int pidx : segment) {
          int py = pidx / width;
          int px = pidx % width;
          depth(py, px) = 0.0f;
          normal.col(pidx).setZero();
        }
        total_removed += static_cast<int>(segment.size());
      }
    }
  }

  if (total_removed > 0) {
    {
      std::ostringstream oss;
      oss << "[PatchMatch] Speckle removal: " << total_removed
          << " pixels removed (min segment " << min_segment_size << ")";
      foundation::LogDebug("dense", oss.str());
    }
  }
}

// =====================================================================
// Gap interpolation: scan row-wise then column-wise, linearly
// interpolate depth and normal for small gaps bounded by similar depths.
// depth_diff_threshold is the relative depth difference threshold for
// the gap boundary pixels.
// =====================================================================
void DepthmapEstimator::GapInterpolation(ImageF& depth, PixelData3f& normal,
                                         int width, int height,
                                         int max_gap_size,
                                         float depth_diff_threshold) {
  int total_filled = 0;

  // 1. Row-wise (horizontal gaps)
  for (int y = 0; y < height; ++y) {
    int gap_start = -1;
    for (int x = 0; x < width; ++x) {
      float d = depth(y, x);
      if (d <= 0.0f) {
        if (gap_start < 0) {
          gap_start = x;
        }
        continue;
      }
      // We hit a valid pixel after a gap
      if (gap_start >= 0) {
        int gap_len = x - gap_start;
        int left_x = gap_start - 1;
        if (left_x >= 0 && gap_len <= max_gap_size) {
          float d_left = depth(y, left_x);
          float d_right = d;
          float max_d = std::max(d_left, d_right);
          if (std::abs(d_left - d_right) / max_d < depth_diff_threshold) {
            // Linearly interpolate depth and normal
            Vec3f n_left = normal.col(y * width + left_x);
            Vec3f n_right = normal.col(y * width + x);
            for (int gx = gap_start; gx < x; ++gx) {
              float t = static_cast<float>(gx - left_x) /
                        static_cast<float>(x - left_x);
              depth(y, gx) = d_left + t * (d_right - d_left);
              Vec3f n_interp = (1.0f - t) * n_left + t * n_right;
              float nlen = n_interp.norm();
              if (nlen > 1e-6f) {
                n_interp /= nlen;
              }
              normal.col(y * width + gx) = n_interp;
              ++total_filled;
            }
          }
        }
      }
      gap_start = -1;
    }
  }

  // 2. Column-wise (vertical gaps)
  for (int x = 0; x < width; ++x) {
    int gap_start = -1;
    for (int y = 0; y < height; ++y) {
      float d = depth(y, x);
      if (d <= 0.0f) {
        if (gap_start < 0) {
          gap_start = y;
        }
        continue;
      }
      if (gap_start >= 0) {
        int gap_len = y - gap_start;
        int top_y = gap_start - 1;
        if (top_y >= 0 && gap_len <= max_gap_size) {
          float d_top = depth(top_y, x);
          float d_bottom = d;
          float max_d = std::max(d_top, d_bottom);
          if (std::abs(d_top - d_bottom) / max_d < depth_diff_threshold) {
            Vec3f n_top = normal.col(top_y * width + x);
            Vec3f n_bottom = normal.col(y * width + x);
            for (int gy = gap_start; gy < y; ++gy) {
              float t = static_cast<float>(gy - top_y) /
                        static_cast<float>(y - top_y);
              // Only fill if still empty (don't overwrite row-wise fills)
              if (depth(gy, x) <= 0.0f) {
                depth(gy, x) = d_top + t * (d_bottom - d_top);
                Vec3f n_interp = (1.0f - t) * n_top + t * n_bottom;
                float nlen = n_interp.norm();
                if (nlen > 1e-6f) {
                  n_interp /= nlen;
                }
                normal.col(gy * width + x) = n_interp;
                ++total_filled;
              }
            }
          }
        }
      }
      gap_start = -1;
    }
  }

  if (total_filled > 0) {
    {
      std::ostringstream oss;
      oss << "[PatchMatch] Gap interpolation: " << total_filled
          << " pixels filled (max gap " << max_gap_size << ")";
      foundation::LogDebug("dense", oss.str());
    }
  }
}

void DepthmapEstimator::ReleaseGpuBuffers() {
  // Release GPU buffers (host-pinned memory on many OpenCL implementations).
  for (int i = 0; i < static_cast<int>(cl_images_.size()); i++) {
    cl_images_[i] = cl::Image2D();
  }
  cl_images_.clear();
  cl_ref_img_ = cl::Image2D();
  cl_src_images_array_ = cl::Image2DArray();
  cl_cameras_ = cl::Buffer();
  cl_plane_hypotheses_ = cl::Buffer();
  cl_costs_ = cl::Buffer();
  cl_rand_states_ = cl::Buffer();
  cl_selected_views_ = cl::Buffer();
  cl_prior_planes_ = cl::Buffer();
  cl_plane_masks_ = cl::Buffer();
  cl_prev_depths_ = cl::Buffer();
  cl_prev_depth_mask_ = 0u;
  cl_angle_weights_ = cl::Buffer();
}

void DepthmapEstimator::ReleaseBuffers() {
  ReleaseGpuBuffers();

  // Release CPU image copies.
  images_.clear();
  orig_images_.clear();
  Ks_.clear();
  orig_Ks_.clear();
  Rs_.clear();
  ts_.clear();
  sfm_points_.clear();
  prev_depth_entries_.clear();
  prev_level_result_ = DepthmapResult();
  prev_level_w_ = 0;
  prev_level_h_ = 0;
  prepared_ = false;
}

// =====================================================================
// Run SLIC superpixel segmentation on the reference image and upload
// per-pixel labels to cl_segment_labels_ for propagation gating.
// Runs at half-resolution for speed, then upsamples labels to full-res.
// =====================================================================
void DepthmapEstimator::RunSLIC(int width, int height) {
  if (!slic_enabled_) {
    // Segmentation disabled: labels stay -1 (no gating).
    return;
  }

  auto& dev = opencl::CLContext::Instance().Device(device_idx_);
  auto& ctx = dev.context();
  auto& queue = dev.queue();
  cl_int err;

  // Work at half resolution for speed.
  const int hw = width / 2;
  const int hh = height / 2;
  const int grid_step = params_.slic_grid_step;
  const float compactness = params_.slic_compactness;

  // Downscale reference image to half-res float.
  cv::Mat half_img;
  cv::resize(images_[0], half_img, cv::Size(hw, hh), 0, 0, cv::INTER_AREA);
  cv::Mat half_float;
  half_img.convertTo(half_float, CV_32F, 1.0 / 255.0);

  // Upload half-res image as Image2D (CL_R, CL_FLOAT).
  cl::ImageFormat fmt(CL_R, CL_FLOAT);
  cl::Image2D slic_img(ctx, CL_MEM_READ_ONLY, fmt, hw, hh, 0, nullptr, &err);
  opencl::CheckCL(err, "SLIC half-res image");
  cl::array<cl::size_type, 3> origin = {{0, 0, 0}};
  cl::array<cl::size_type, 3> region = {
      {static_cast<cl::size_type>(hw), static_cast<cl::size_type>(hh), 1}};
  queue.enqueueWriteImage(slic_img, CL_TRUE, origin, region, half_float.step[0],
                          0, half_float.data);

  // Grid dimensions.
  const int centers_x = (hw + grid_step - 1) / grid_step;
  const int centers_y = (hh + grid_step - 1) / grid_step;
  const int num_centers = centers_x * centers_y;

  // SLICCenter struct: float x, y, intensity; int count → 16 bytes.
  const size_t center_size = 16;
  cl::Buffer cl_centers(ctx, CL_MEM_READ_WRITE, center_size * num_centers,
                        nullptr, &err);
  opencl::CheckCL(err, "SLIC centers buffer");

  // Labels buffer at half-res.
  const int hpix = hw * hh;
  cl::Buffer cl_labels(ctx, CL_MEM_READ_WRITE, sizeof(int) * hpix, nullptr,
                       &err);
  opencl::CheckCL(err, "SLIC labels buffer");

  // Initialize centers on grid.
  {
    int arg = 0;
    k_slic_init_.setArg(arg++, cl_centers);
    k_slic_init_.setArg(arg++, slic_img);
    k_slic_init_.setArg(arg++, hw);
    k_slic_init_.setArg(arg++, hh);
    k_slic_init_.setArg(arg++, grid_step);
    k_slic_init_.setArg(arg++, centers_x);
    k_slic_init_.setArg(arg++, centers_y);

    cl::NDRange global(static_cast<size_t>((centers_x + 15) / 16 * 16),
                       static_cast<size_t>((centers_y + 15) / 16 * 16));
    cl::NDRange local(16, 16);
    queue.enqueueNDRangeKernel(k_slic_init_, cl::NullRange, global, local);
  }

  // Iterate: assign + update (5 iterations is standard for SLIC).
  const int slic_iters = 5;
  for (int it = 0; it < slic_iters; ++it) {
    // Assign pixels to nearest center.
    {
      int arg = 0;
      k_slic_assign_.setArg(arg++, cl_centers);
      k_slic_assign_.setArg(arg++, cl_labels);
      k_slic_assign_.setArg(arg++, slic_img);
      k_slic_assign_.setArg(arg++, hw);
      k_slic_assign_.setArg(arg++, hh);
      k_slic_assign_.setArg(arg++, grid_step);
      k_slic_assign_.setArg(arg++, centers_x);
      k_slic_assign_.setArg(arg++, centers_y);
      k_slic_assign_.setArg(arg++, compactness);

      cl::NDRange global(static_cast<size_t>((hw + 15) / 16 * 16),
                         static_cast<size_t>((hh + 15) / 16 * 16));
      cl::NDRange local(16, 16);
      queue.enqueueNDRangeKernel(k_slic_assign_, cl::NullRange, global, local);
    }

    // Update centers.
    {
      int arg = 0;
      k_slic_update_.setArg(arg++, cl_centers);
      k_slic_update_.setArg(arg++, cl_labels);
      k_slic_update_.setArg(arg++, slic_img);
      k_slic_update_.setArg(arg++, hw);
      k_slic_update_.setArg(arg++, hh);
      k_slic_update_.setArg(arg++, grid_step);
      k_slic_update_.setArg(arg++, centers_x);
      k_slic_update_.setArg(arg++, centers_y);

      cl::NDRange global(static_cast<size_t>((centers_x + 15) / 16 * 16),
                         static_cast<size_t>((centers_y + 15) / 16 * 16));
      cl::NDRange local(16, 16);
      queue.enqueueNDRangeKernel(k_slic_update_, cl::NullRange, global, local);
    }
  }

  // Read back half-res labels.
  queue.finish();
  std::vector<int> half_labels(hpix);
  queue.enqueueReadBuffer(cl_labels, CL_TRUE, 0, sizeof(int) * hpix,
                          half_labels.data());

  // Connectivity enforcement: BFS relabel small disconnected components.
  // Any fragment with fewer pixels than grid_step*grid_step/4 gets merged
  // into its largest adjacent segment.
  {
    const int min_size = grid_step * grid_step / 4;
    std::vector<int> component(hpix, -1);
    std::vector<int> stack;
    stack.reserve(min_size * 2);
    int next_comp = 0;

    for (int i = 0; i < hpix; ++i) {
      if (component[i] >= 0) {
        continue;
      }
      // BFS flood fill for this connected component.
      int label = half_labels[i];
      stack.clear();
      stack.push_back(i);
      component[i] = next_comp;
      int head = 0;
      while (head < static_cast<int>(stack.size())) {
        int p = stack[head++];
        int px = p % hw;
        int py = p / hw;
        const int dx[] = {-1, 1, 0, 0};
        const int dy[] = {0, 0, -1, 1};
        for (int d = 0; d < 4; ++d) {
          int nx = px + dx[d];
          int ny = py + dy[d];
          if (nx < 0 || nx >= hw || ny < 0 || ny >= hh) {
            continue;
          }
          int ni = ny * hw + nx;
          if (component[ni] >= 0) {
            continue;
          }
          if (half_labels[ni] != label) {
            continue;
          }
          component[ni] = next_comp;
          stack.push_back(ni);
        }
      }
      int comp_size = static_cast<int>(stack.size()) - 0;  // all pushed
      // If too small, relabel the entire component to the majority neighbor.
      if (comp_size < min_size) {
        // Find most frequent adjacent label.
        std::unordered_map<int, int> adj_count;
        for (int si = 0; si < static_cast<int>(stack.size()); ++si) {
          int p = stack[si];
          int px_c = p % hw;
          int py_c = p / hw;
          const int dx2[] = {-1, 1, 0, 0};
          const int dy2[] = {0, 0, -1, 1};
          for (int d = 0; d < 4; ++d) {
            int nx = px_c + dx2[d];
            int ny = py_c + dy2[d];
            if (nx < 0 || nx >= hw || ny < 0 || ny >= hh) {
              continue;
            }
            int ni = ny * hw + nx;
            if (half_labels[ni] != label) {
              adj_count[half_labels[ni]]++;
            }
          }
        }
        int best_adj = label;
        int best_cnt = 0;
        for (auto& kv : adj_count) {
          if (kv.second > best_cnt) {
            best_cnt = kv.second;
            best_adj = kv.first;
          }
        }
        // Relabel.
        for (int si = 0; si < static_cast<int>(stack.size()); ++si) {
          half_labels[stack[si]] = best_adj;
        }
      }
      next_comp++;
    }
  }

  // Bottom-up RAG merge: greedily merge adjacent segments whose mean
  // intensities differ by less than a threshold.  This collapses the
  // over-segmented SLIC grid into perceptually coherent regions.
  // {
  //   const float* img_data = reinterpret_cast<const float*>(half_float.data);

  //   // 1. Compact-relabel so labels are 0..N-1.
  //   std::unordered_map<int, int> label_remap;
  //   int n_seg = 0;
  //   for (int i = 0; i < hpix; ++i) {
  //     int lbl = half_labels[i];
  //     if (label_remap.find(lbl) == label_remap.end()) {
  //       label_remap[lbl] = n_seg++;
  //     }
  //     half_labels[i] = label_remap[lbl];
  //   }

  //   // 2. Compute per-segment mean intensity and pixel count.
  //   std::vector<double> seg_sum(n_seg, 0.0);
  //   std::vector<int> seg_count(n_seg, 0);
  //   for (int i = 0; i < hpix; ++i) {
  //     int s = half_labels[i];
  //     seg_sum[s] += img_data[i];
  //     seg_count[s]++;
  //   }
  //   std::vector<float> seg_mean(n_seg);
  //   for (int s = 0; s < n_seg; ++s) {
  //     seg_mean[s] = seg_count[s] > 0
  //                       ? static_cast<float>(seg_sum[s] / seg_count[s])
  //                       : 0.0f;
  //   }

  //   // 3. Build adjacency set (undirected edges between neighboring
  //   segments).
  //   // Use a union-find to track merges efficiently.
  //   std::vector<int> parent(n_seg);
  //   std::iota(parent.begin(), parent.end(), 0);
  //   auto find = [&](int x) {
  //     while (parent[x] != x) {
  //       parent[x] = parent[parent[x]];
  //       x = parent[x];
  //     }
  //     return x;
  //   };

  //   // Collect unique adjacent pairs.
  //   std::set<std::pair<int, int>> adj_set;
  //   for (int y = 0; y < hh; ++y) {
  //     for (int x = 0; x < hw; ++x) {
  //       int idx = y * hw + x;
  //       int s = half_labels[idx];
  //       if (x + 1 < hw) {
  //         int s2 = half_labels[idx + 1];
  //         if (s != s2) {
  //           int a = std::min(s, s2), b = std::max(s, s2);
  //           adj_set.insert({a, b});
  //         }
  //       }
  //       if (y + 1 < hh) {
  //         int s2 = half_labels[idx + hw];
  //         if (s != s2) {
  //           int a = std::min(s, s2), b = std::max(s, s2);
  //           adj_set.insert({a, b});
  //         }
  //       }
  //     }
  //   }

  //   // 4. Sort edges by color distance (ascending) and merge greedily.
  //   // Merge threshold: segments with mean intensity difference < this
  //   // are considered the same surface.  0.08 ≈ 20/255 intensity step.
  //   const float merge_threshold = 0.08f;

  //   struct Edge {
  //     float dist;
  //     int a, b;
  //   };
  //   std::vector<Edge> edges;
  //   edges.reserve(adj_set.size());
  //   for (auto& [a, b] : adj_set) {
  //     float d = std::abs(seg_mean[a] - seg_mean[b]);
  //     edges.push_back({d, a, b});
  //   }
  //   std::sort(edges.begin(), edges.end(),
  //             [](const Edge& x, const Edge& y) { return x.dist < y.dist; });

  //   // Greedy merge: merge the closest pair, update the merged segment's
  //   // mean as a weighted average, then continue.
  //   for (auto& e : edges) {
  //     if (e.dist > merge_threshold) break;
  //     int ra = find(e.a);
  //     int rb = find(e.b);
  //     if (ra == rb) continue;
  //     // Check current merged means (they may have shifted from merges).
  //     float mean_a = seg_mean[ra];
  //     float mean_b = seg_mean[rb];
  //     if (std::abs(mean_a - mean_b) > merge_threshold) continue;
  //     // Merge rb into ra.
  //     int ca = seg_count[ra], cb = seg_count[rb];
  //     seg_mean[ra] = (mean_a * ca + mean_b * cb) / (ca + cb);
  //     seg_count[ra] = ca + cb;
  //     parent[rb] = ra;
  //   }

  //   // 5. Apply final relabeling.
  //   for (int i = 0; i < hpix; ++i) {
  //     half_labels[i] = find(half_labels[i]);
  //   }

  //   // Log merge stats.
  //   std::set<int> unique_labels(half_labels.begin(), half_labels.end());
  //   std::cerr << "[SLIC] After RAG merge: " << unique_labels.size()
  //             << " segments (from " << n_seg << ")\n";
  // }

  // Upsample labels to full-res (nearest neighbor).
  const int npix = width * height;
  std::vector<int> full_labels(npix);
  const float sx = static_cast<float>(hw) / width;
  const float sy = static_cast<float>(hh) / height;
  for (int y = 0; y < height; ++y) {
    int hy = std::min(static_cast<int>(y * sy), hh - 1);
    for (int x = 0; x < width; ++x) {
      int hx = std::min(static_cast<int>(x * sx), hw - 1);
      full_labels[y * width + x] = half_labels[hy * hw + hx];
    }
  }

  // Upload to GPU.
  queue.enqueueWriteBuffer(cl_segment_labels_, CL_TRUE, 0, sizeof(int) * npix,
                           full_labels.data());

  // Debug: save random-colored segmentation image.
  if (!params_.debug_dir.empty()) {
    // Assign a random but deterministic color to each segment label.
    std::unordered_map<int, cv::Vec3b> color_map;
    std::mt19937 color_rng(42);
    cv::Mat debug_img(height, width, CV_8UC3);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        int lbl = full_labels[y * width + x];
        auto it = color_map.find(lbl);
        if (it == color_map.end()) {
          uint8_t r = static_cast<uint8_t>(color_rng() % 200 + 55);
          uint8_t g = static_cast<uint8_t>(color_rng() % 200 + 55);
          uint8_t b = static_cast<uint8_t>(color_rng() % 200 + 55);
          color_map[lbl] = cv::Vec3b(b, g, r);
          it = color_map.find(lbl);
        }
        debug_img.at<cv::Vec3b>(y, x) = it->second;
      }
    }
    std::string fname = params_.debug_dir + "/slic_labels";
    if (!params_.debug_shot_id.empty()) {
      fname += "_" + params_.debug_shot_id;
    }
    fname += ".png";
    cv::imwrite(fname, debug_img);
    {
      std::ostringstream oss;
      oss << "[SLIC] Saved debug segmentation to " << fname << " ("
          << color_map.size() << " segments)";
      foundation::LogDebug("dense", oss.str());
    }
  }
}

}  // namespace dense

#endif  // OPENSFM_HAVE_OPENCL
