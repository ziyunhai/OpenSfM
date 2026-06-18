#ifdef OPENSFM_HAVE_OPENCL

#include <dense/svo_opencl.h>
#include <dense/svo_opencl_kernels.h>
#include <foundation/logging.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>
#include <vector>

namespace dense {

SVOIntegratorCL::SVOIntegratorCL(int device_idx) : device_idx_(device_idx) {}

void SVOIntegratorCL::BuildKernels() {
  if (kernels_built_) {
    return;
  }

  auto& dev = opencl::CLContext::Instance().Device(device_idx_);
  program_ = dev.GetOrBuildProgram("svo", kSVOKernelSource);

  cl_int err;
  k_clear_ = cl::Kernel(program_, "svo_clear_table", &err);
  opencl::CheckCL(err, "kernel svo_clear_table");

  k_integrate_ = cl::Kernel(program_, "svo_integrate", &err);
  opencl::CheckCL(err, "kernel svo_integrate");

  k_count_clear_ = cl::Kernel(program_, "svo_clear_count_table", &err);
  opencl::CheckCL(err, "kernel svo_clear_count_table");

  k_count_ = cl::Kernel(program_, "svo_count", &err);
  opencl::CheckCL(err, "kernel svo_count");

  k_extract_ = cl::Kernel(program_, "svo_extract_points", &err);
  opencl::CheckCL(err, "kernel svo_extract_points");

  k_extract_fill_ = cl::Kernel(program_, "svo_extract_fill", &err);
  opencl::CheckCL(err, "kernel svo_extract_fill");

  k_refine_clear_ = cl::Kernel(program_, "svo_refine_clear", &err);
  opencl::CheckCL(err, "kernel svo_refine_clear");

  k_refine_accumulate_ = cl::Kernel(program_, "svo_refine_accumulate", &err);
  opencl::CheckCL(err, "kernel svo_refine_accumulate");

  k_refine_update_ = cl::Kernel(program_, "svo_refine_update", &err);
  opencl::CheckCL(err, "kernel svo_refine_update");

  k_bake_colors_ = cl::Kernel(program_, "svo_bake_colors", &err);
  opencl::CheckCL(err, "kernel svo_bake_colors");

  k_raycast_guided_ = cl::Kernel(program_, "svo_raycast_guided", &err);
  opencl::CheckCL(err, "kernel svo_raycast_guided");

  k_raycast_ = cl::Kernel(program_, "svo_raycast", &err);
  opencl::CheckCL(err, "kernel svo_raycast");

  // 1-byte dummy buffer used as placeholder when normal/color/mask is absent.
  cl_dummy_ =
      cl::Buffer(dev.context(), static_cast<cl_mem_flags>(CL_MEM_READ_ONLY),
                 static_cast<cl::size_type>(1), nullptr, &err);
  opencl::CheckCL(err, "dummy buffer");

  kernels_built_ = true;
}

void SVOIntegratorCL::Initialize(uint32_t capacity) {
  BuildKernels();

  // Release counting-pass resources (if any) before allocating the
  // much larger integration table — avoids holding both in GPU memory.
  if (count_capacity_ > 0) {
    cl_count_table_ = cl::Buffer();
    cl_counter_ = cl::Buffer();
    count_capacity_ = 0;
    count_mask_ = 0;
  }

  // Round up to power of two.
  uint32_t cap = 1;
  while (cap < capacity) {
    cap <<= 1;
  }

  // Clamp to device memory limits.
  auto& dev = opencl::CLContext::Instance().Device(device_idx_);
  const size_t max_alloc = dev.device().getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
  const size_t avail = dev.AvailableMemory();
  // Use at most 60% of available memory or the max single-alloc size.
  const size_t mem_limit = std::min(max_alloc, avail * 6 / 10);
  const uint32_t max_slots = static_cast<uint32_t>(std::min<size_t>(
      mem_limit / sizeof(GPUVoxelSlot), std::numeric_limits<uint32_t>::max()));
  // Round max_slots down to power of two.
  uint32_t max_cap = 1;
  while (max_cap <= max_slots / 2) {
    max_cap <<= 1;
  }
  if (cap > max_cap) {
    {
      std::ostringstream oss;
      oss << "[SVOIntegratorCL] Clamping capacity from " << cap << " to "
          << max_cap << " (max_alloc=" << (max_alloc >> 20)
          << "MB, avail=" << (avail >> 20) << "MB)";
      foundation::LogWarning("dense", oss.str());
    }
    cap = max_cap;
  }

  capacity_ = cap;
  capacity_mask_ = cap - 1;

  auto& ctx = dev.context();
  auto& queue = dev.queue();

  const size_t table_bytes =
      static_cast<size_t>(capacity_) * sizeof(GPUVoxelSlot);
  {
    std::ostringstream oss;
    oss << "[SVOIntegratorCL] GPU hash table: " << capacity_ << " slots ("
        << (table_bytes >> 20) << " MB) on " << dev.name();
    foundation::LogInfo("dense", oss.str());
  }
  cl_int err;
  cl_table_ = cl::Buffer(ctx, CL_MEM_READ_WRITE, table_bytes, nullptr, &err);
  opencl::CheckCL(err, "hash table buffer");

  cl_camera_ =
      cl::Buffer(ctx, CL_MEM_READ_ONLY, sizeof(SVOCameraGPU), nullptr, &err);
  opencl::CheckCL(err, "camera buffer");

  // Clear the table.
  k_clear_.setArg(0, cl_table_);
  k_clear_.setArg(1, capacity_);

  const size_t global = ((capacity_ + 255u) / 256u) * 256u;
  queue.enqueueNDRangeKernel(k_clear_, cl::NullRange, cl::NDRange(global),
                             cl::NDRange(256));

  // Allocate and zero the overflow counter.
  cl_overflow_ =
      cl::Buffer(ctx, CL_MEM_READ_WRITE, sizeof(uint32_t), nullptr, &err);
  opencl::CheckCL(err, "overflow counter buffer");
  uint32_t zero = 0;
  queue.enqueueWriteBuffer(cl_overflow_, CL_TRUE, 0, sizeof(uint32_t), &zero);

  queue.finish();
}

void SVOIntegratorCL::EnsureFrameBuffers(int rows, int cols, bool has_normal,
                                         bool has_mask, bool has_weight) {
  auto& ctx = opencl::CLContext::Instance().Device(device_idx_).context();
  cl_int err;
  const size_t npix = static_cast<size_t>(rows) * cols;

  const size_t d_bytes = npix * sizeof(float);
  if (d_bytes > depth_bytes_) {
    cl_depth_ = cl::Buffer(ctx, CL_MEM_READ_ONLY, d_bytes, nullptr, &err);
    opencl::CheckCL(err, "depth buffer");
    depth_bytes_ = d_bytes;
  }
  if (has_normal) {
    const size_t n_bytes = npix * 3 * sizeof(float);
    if (n_bytes > normal_bytes_) {
      cl_normal_ = cl::Buffer(ctx, CL_MEM_READ_ONLY, n_bytes, nullptr, &err);
      opencl::CheckCL(err, "normal buffer");
      normal_bytes_ = n_bytes;
    }
  }
  if (has_mask) {
    const size_t m_bytes = npix * sizeof(uint8_t);
    if (m_bytes > mask_bytes_) {
      cl_mask_ = cl::Buffer(ctx, CL_MEM_READ_ONLY, m_bytes, nullptr, &err);
      opencl::CheckCL(err, "mask buffer");
      mask_bytes_ = m_bytes;
    }
  }
  if (has_weight) {
    const size_t w_bytes = npix * sizeof(float);
    if (w_bytes > weight_bytes_) {
      cl_weight_ = cl::Buffer(ctx, CL_MEM_READ_ONLY, w_bytes, nullptr, &err);
      opencl::CheckCL(err, "weight buffer");
      weight_bytes_ = w_bytes;
    }
  }
}

// Copy an Eigen ColMajor 3×3 matrix into a row-major float[9] array.
static void ToRowMajor(const Mat3f& M, float* out) {
  // ColMajor data of M^T == RowMajor data of M.
  const Mat3f Mt = M.transpose();
  std::memcpy(out, Mt.data(), 9 * sizeof(float));
}

void SVOIntegratorCL::Integrate(const Mat3f& K, const Mat3f& R, const Vec3f& t,
                                const float* depth, int rows, int cols,
                                const float* normal, const uint8_t* color,
                                const uint8_t* mask, const float* weight,
                                float voxel_size, float trunc_dist,
                                const Eigen::Vector3i* bbox_min,
                                const Eigen::Vector3i* bbox_max,
                                const std::vector<RefTableInfo>* ref_tables,
                                float ref_min_weight) {
  if (capacity_ == 0) {
    throw std::runtime_error("SVOIntegratorCL: Initialize() not called");
  }
  (void)color;  // Color no longer integrated; baked separately.

  auto& dev = opencl::CLContext::Instance().Device(device_idx_);
  auto& queue = dev.queue();

  const bool has_normal = (normal != nullptr);
  const bool has_mask = (mask != nullptr);
  const bool has_weight = (weight != nullptr);

  EnsureFrameBuffers(rows, cols, has_normal, has_mask, has_weight);

  const size_t npix = static_cast<size_t>(rows) * cols;

  // Upload depth.
  queue.enqueueWriteBuffer(cl_depth_, CL_FALSE, 0, npix * sizeof(float), depth);
  // Upload optional arrays.
  if (has_normal) {
    queue.enqueueWriteBuffer(cl_normal_, CL_FALSE, 0, npix * 3 * sizeof(float),
                             normal);
  }
  if (has_mask) {
    queue.enqueueWriteBuffer(cl_mask_, CL_FALSE, 0, npix * sizeof(uint8_t),
                             mask);
  }
  if (has_weight) {
    queue.enqueueWriteBuffer(cl_weight_, CL_FALSE, 0, npix * sizeof(float),
                             weight);
  }

  // Prepare camera parameters (row-major matrices).
  SVOCameraGPU cam{};
  const Mat3f Kinv = K.inverse();
  const Mat3f Rinv = R.transpose();
  const Vec3f cam_pos = -Rinv * t;

  ToRowMajor(Kinv, cam.Kinv);
  ToRowMajor(Rinv, cam.Rinv);
  ToRowMajor(R, cam.R);
  cam.t[0] = t.x();
  cam.t[1] = t.y();
  cam.t[2] = t.z();
  cam.cam_pos[0] = cam_pos.x();
  cam.cam_pos[1] = cam_pos.y();
  cam.cam_pos[2] = cam_pos.z();

  queue.enqueueWriteBuffer(cl_camera_, CL_FALSE, 0, sizeof(SVOCameraGPU), &cam);

  // Finish all uploads before launching the kernel.
  queue.finish();

  // Set kernel arguments.
  int arg = 0;
  k_integrate_.setArg(arg++, cl_table_);
  k_integrate_.setArg(arg++, capacity_mask_);
  k_integrate_.setArg(arg++, cl_overflow_);
  k_integrate_.setArg(arg++, cl_depth_);
  k_integrate_.setArg(arg++, has_normal ? cl_normal_ : cl_dummy_);
  k_integrate_.setArg(arg++, has_mask ? cl_mask_ : cl_dummy_);
  k_integrate_.setArg(arg++, has_weight ? cl_weight_ : cl_dummy_);
  k_integrate_.setArg(arg++, static_cast<cl_int>(has_normal));
  k_integrate_.setArg(arg++, static_cast<cl_int>(has_mask));
  k_integrate_.setArg(arg++, static_cast<cl_int>(has_weight));
  k_integrate_.setArg(arg++, cl_camera_);
  k_integrate_.setArg(arg++, static_cast<cl_int>(rows));
  k_integrate_.setArg(arg++, static_cast<cl_int>(cols));
  k_integrate_.setArg(arg++, trunc_dist);
  k_integrate_.setArg(arg++, voxel_size);
  k_integrate_.setArg(arg++, 1.0f / voxel_size);

  // Bounding box arguments (voxel integer coordinates).
  const bool has_bbox = (bbox_min != nullptr && bbox_max != nullptr);
  k_integrate_.setArg(arg++,
                      static_cast<cl_int>(has_bbox ? (*bbox_min).x() : 0));
  k_integrate_.setArg(arg++,
                      static_cast<cl_int>(has_bbox ? (*bbox_min).y() : 0));
  k_integrate_.setArg(arg++,
                      static_cast<cl_int>(has_bbox ? (*bbox_min).z() : 0));
  k_integrate_.setArg(arg++,
                      static_cast<cl_int>(has_bbox ? (*bbox_max).x() : 0));
  k_integrate_.setArg(arg++,
                      static_cast<cl_int>(has_bbox ? (*bbox_max).y() : 0));
  k_integrate_.setArg(arg++,
                      static_cast<cl_int>(has_bbox ? (*bbox_max).z() : 0));
  k_integrate_.setArg(arg++, static_cast<cl_int>(has_bbox ? 1 : 0));

  // Multi-level reference table check arguments.
  const int n_ref =
      ref_tables ? static_cast<int>(std::min(ref_tables->size(), size_t(2)))
                 : 0;
  k_integrate_.setArg(arg++, static_cast<cl_int>(n_ref));

  if (n_ref >= 1) {
    k_integrate_.setArg(arg++, (*ref_tables)[0].buffer);
    k_integrate_.setArg(arg++, (*ref_tables)[0].mask);
    k_integrate_.setArg(arg++, (*ref_tables)[0].inv_voxel_size);
  } else {
    k_integrate_.setArg(arg++, cl_dummy_);
    k_integrate_.setArg(arg++, static_cast<cl_uint>(0));
    k_integrate_.setArg(arg++, 1.0f);
  }

  if (n_ref >= 2) {
    k_integrate_.setArg(arg++, (*ref_tables)[1].buffer);
    k_integrate_.setArg(arg++, (*ref_tables)[1].mask);
    k_integrate_.setArg(arg++, (*ref_tables)[1].inv_voxel_size);
  } else {
    k_integrate_.setArg(arg++, cl_dummy_);
    k_integrate_.setArg(arg++, static_cast<cl_uint>(0));
    k_integrate_.setArg(arg++, 1.0f);
  }

  k_integrate_.setArg(arg++, static_cast<float>(ref_min_weight * kWeightScale));

  // Launch: one work-item per pixel.
  cl::NDRange global(static_cast<size_t>((cols + 15) / 16 * 16),
                     static_cast<size_t>((rows + 15) / 16 * 16));
  cl::NDRange local(16, 16);
  queue.enqueueNDRangeKernel(k_integrate_, cl::NullRange, global, local);
  queue.finish();
}

// ── Counting pass ─────────────────────────────────────────────────────

void SVOIntegratorCL::InitializeCounting(uint32_t capacity) {
  BuildKernels();

  // Round up to power of two.
  uint32_t cap = 1;
  while (cap < capacity) {
    cap <<= 1;
  }

  // Clamp to device memory limits.
  auto& dev = opencl::CLContext::Instance().Device(device_idx_);
  const size_t max_alloc = dev.device().getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
  const size_t avail = dev.AvailableMemory();
  const size_t mem_limit = std::min(max_alloc, avail * 6 / 10);
  const uint32_t max_slots = static_cast<uint32_t>(std::min<size_t>(
      mem_limit / sizeof(GPUCountSlot), std::numeric_limits<uint32_t>::max()));
  uint32_t max_cap = 1;
  while (max_cap <= max_slots / 2) {
    max_cap <<= 1;
  }
  if (cap > max_cap) {
    {
      std::ostringstream oss;
      oss << "[SVOIntegratorCL] Clamping count capacity from " << cap << " to "
          << max_cap;
      foundation::LogWarning("dense", oss.str());
    }
    cap = max_cap;
  }

  count_capacity_ = cap;
  count_mask_ = cap - 1;

  auto& ctx = dev.context();
  auto& queue = dev.queue();

  const size_t table_bytes =
      static_cast<size_t>(count_capacity_) * sizeof(GPUCountSlot);
  {
    std::ostringstream oss;
    oss << "[SVOIntegratorCL] Count table: " << count_capacity_ << " slots ("
        << (table_bytes >> 20) << " MB)";
    foundation::LogInfo("dense", oss.str());
  }

  cl_int err;
  cl_count_table_ =
      cl::Buffer(ctx, CL_MEM_READ_WRITE, table_bytes, nullptr, &err);
  opencl::CheckCL(err, "count table buffer");

  // Atomic counter for unique inserts (single uint32).
  cl_counter_ =
      cl::Buffer(ctx, CL_MEM_READ_WRITE, sizeof(uint32_t), nullptr, &err);
  opencl::CheckCL(err, "counter buffer");

  // Overflow counter for counting pass.
  cl_count_overflow_ =
      cl::Buffer(ctx, CL_MEM_READ_WRITE, sizeof(uint32_t), nullptr, &err);
  opencl::CheckCL(err, "count overflow buffer");

  // Camera params buffer (shared with integration path).
  cl_camera_ =
      cl::Buffer(ctx, CL_MEM_READ_ONLY, sizeof(SVOCameraGPU), nullptr, &err);
  opencl::CheckCL(err, "camera buffer (counting)");

  // Zero the counters.
  uint32_t zero = 0;
  queue.enqueueWriteBuffer(cl_counter_, CL_FALSE, 0, sizeof(uint32_t), &zero);
  queue.enqueueWriteBuffer(cl_count_overflow_, CL_FALSE, 0, sizeof(uint32_t),
                           &zero);

  // Clear the counting table.
  k_count_clear_.setArg(0, cl_count_table_);
  k_count_clear_.setArg(1, count_capacity_);
  const size_t global = ((count_capacity_ + 255u) / 256u) * 256u;
  queue.enqueueNDRangeKernel(k_count_clear_, cl::NullRange, cl::NDRange(global),
                             cl::NDRange(256));
  queue.finish();
}

void SVOIntegratorCL::Count(const Mat3f& K, const Mat3f& R, const Vec3f& t,
                            const float* depth, int rows, int cols,
                            const uint8_t* mask, float voxel_size,
                            float trunc_dist, const Eigen::Vector3i* bbox_min,
                            const Eigen::Vector3i* bbox_max) {
  if (count_capacity_ == 0) {
    throw std::runtime_error(
        "SVOIntegratorCL: InitializeCounting() not called");
  }

  auto& dev = opencl::CLContext::Instance().Device(device_idx_);
  auto& queue = dev.queue();

  const bool has_mask = (mask != nullptr);
  // Reuse the depth/mask frame buffers from the integration path.
  EnsureFrameBuffers(rows, cols, /*has_normal=*/false, has_mask,
                     /*has_weight=*/false);

  const size_t npix = static_cast<size_t>(rows) * cols;

  // Upload depth.
  queue.enqueueWriteBuffer(cl_depth_, CL_FALSE, 0, npix * sizeof(float), depth);
  if (has_mask) {
    queue.enqueueWriteBuffer(cl_mask_, CL_FALSE, 0, npix * sizeof(uint8_t),
                             mask);
  }

  // Camera params (reuse cl_camera_ buffer).
  SVOCameraGPU cam{};
  const Mat3f Kinv = K.inverse();
  const Mat3f Rinv = R.transpose();
  const Vec3f cam_pos = -Rinv * t;
  ToRowMajor(Kinv, cam.Kinv);
  ToRowMajor(Rinv, cam.Rinv);
  ToRowMajor(R, cam.R);
  cam.t[0] = t.x();
  cam.t[1] = t.y();
  cam.t[2] = t.z();
  cam.cam_pos[0] = cam_pos.x();
  cam.cam_pos[1] = cam_pos.y();
  cam.cam_pos[2] = cam_pos.z();

  queue.enqueueWriteBuffer(cl_camera_, CL_FALSE, 0, sizeof(SVOCameraGPU), &cam);
  queue.finish();

  // Set kernel arguments.
  int arg = 0;
  k_count_.setArg(arg++, cl_count_table_);
  k_count_.setArg(arg++, count_mask_);
  k_count_.setArg(arg++, cl_counter_);
  k_count_.setArg(arg++, cl_count_overflow_);
  k_count_.setArg(arg++, cl_depth_);
  k_count_.setArg(arg++, has_mask ? cl_mask_ : cl_dummy_);
  k_count_.setArg(arg++, static_cast<cl_int>(has_mask));
  k_count_.setArg(arg++, cl_camera_);
  k_count_.setArg(arg++, static_cast<cl_int>(rows));
  k_count_.setArg(arg++, static_cast<cl_int>(cols));
  k_count_.setArg(arg++, trunc_dist);
  k_count_.setArg(arg++, voxel_size);
  k_count_.setArg(arg++, 1.0f / voxel_size);

  // Bounding box arguments (voxel integer coordinates).
  const bool has_bbox_c = (bbox_min != nullptr && bbox_max != nullptr);
  k_count_.setArg(arg++, static_cast<cl_int>(has_bbox_c ? (*bbox_min).x() : 0));
  k_count_.setArg(arg++, static_cast<cl_int>(has_bbox_c ? (*bbox_min).y() : 0));
  k_count_.setArg(arg++, static_cast<cl_int>(has_bbox_c ? (*bbox_min).z() : 0));
  k_count_.setArg(arg++, static_cast<cl_int>(has_bbox_c ? (*bbox_max).x() : 0));
  k_count_.setArg(arg++, static_cast<cl_int>(has_bbox_c ? (*bbox_max).y() : 0));
  k_count_.setArg(arg++, static_cast<cl_int>(has_bbox_c ? (*bbox_max).z() : 0));
  k_count_.setArg(arg++, static_cast<cl_int>(has_bbox_c ? 1 : 0));

  cl::NDRange global(static_cast<size_t>((cols + 15) / 16 * 16),
                     static_cast<size_t>((rows + 15) / 16 * 16));
  cl::NDRange local(16, 16);
  queue.enqueueNDRangeKernel(k_count_, cl::NullRange, global, local);
  queue.finish();
}

uint32_t SVOIntegratorCL::GetUniqueCount() const {
  if (count_capacity_ == 0) {
    return 0;
  }
  auto& dev = opencl::CLContext::Instance().Device(device_idx_);
  auto& queue = dev.queue();
  uint32_t count = 0;
  queue.enqueueReadBuffer(cl_counter_, CL_TRUE, 0, sizeof(uint32_t), &count);

  // Also read the counting overflow counter.
  uint32_t overflow = 0;
  queue.enqueueReadBuffer(cl_count_overflow_, CL_TRUE, 0, sizeof(uint32_t),
                          &overflow);
  if (overflow > 0) {
    {
      std::ostringstream oss;
      oss << "[SVOIntegratorCL] WARNING: count pass dropped " << overflow
          << " insertions (table too full)";
      foundation::LogWarning("dense", oss.str());
    }
  }
  return count;
}

uint32_t SVOIntegratorCL::GetOverflowCount() const {
  if (capacity_ == 0) {
    return 0;
  }
  auto& dev = opencl::CLContext::Instance().Device(device_idx_);
  auto& queue = dev.queue();
  uint32_t overflow = 0;
  queue.enqueueReadBuffer(cl_overflow_, CL_TRUE, 0, sizeof(uint32_t),
                          &overflow);
  return overflow;
}

void SVOIntegratorCL::ResetOverflowCounter() {
  if (capacity_ == 0) {
    return;
  }
  auto& dev = opencl::CLContext::Instance().Device(device_idx_);
  auto& queue = dev.queue();
  uint32_t zero = 0;
  queue.enqueueWriteBuffer(cl_overflow_, CL_TRUE, 0, sizeof(uint32_t), &zero);
}

VoxelMap SVOIntegratorCL::Download() const {
  if (capacity_ == 0) {
    return {};
  }

  auto& dev = opencl::CLContext::Instance().Device(device_idx_);
  auto& queue = dev.queue();

  std::vector<GPUVoxelSlot> host_table(capacity_);
  queue.enqueueReadBuffer(cl_table_, CL_TRUE, 0,
                          capacity_ * sizeof(GPUVoxelSlot), host_table.data());

  VoxelMap voxels;
  int occupied = 0;
  int merged = 0;

  for (uint32_t i = 0; i < capacity_; ++i) {
    const auto& slot = host_table[i];
    if (slot.key_ab == kEmptyKey || slot.key_c == kKeyCUninit ||
        slot.count == 0) {
      continue;
    }

    ++occupied;

    const int kx = static_cast<int>((slot.key_ab >> 16) & 0xFFFF) - 32768;
    const int ky = static_cast<int>(slot.key_ab & 0xFFFF) - 32768;
    const int kz = slot.key_c;

    // Values are pre-multiplied by weight in the kernel, so divide by
    // sum_weight (not count) to get weighted averages.
    const float sw = static_cast<float>(slot.sum_weight);
    if (sw < 1.0f) {
      continue;  // degenerate — no meaningful weight
    }
    const float inv_sw = 1.0f / sw;
    const float inv_fp_sw = inv_sw / static_cast<float>(kFPScale);

    const float avg_tsdf = static_cast<float>(slot.sum_tsdf) * inv_fp_sw;
    const float avg_nx = static_cast<float>(slot.sum_nx) * inv_fp_sw;
    const float avg_ny = static_cast<float>(slot.sum_ny) * inv_fp_sw;
    const float avg_nz = static_cast<float>(slot.sum_nz) * inv_fp_sw;

    SVOVoxel v;
    v.tsdf = PackSnorm16(std::clamp(avg_tsdf, -1.0f, 1.0f));
    // Use sum_weight (in kWeightScale units) for the weight so that
    // confidence-weighted observations accumulate correctly.
    v.weight = static_cast<uint16_t>(
        std::min(static_cast<float>(slot.sum_weight), 65535.0f));
    v.nx = PackSnorm16(std::clamp(avg_nx, -1.0f, 1.0f));
    v.ny = PackSnorm16(std::clamp(avg_ny, -1.0f, 1.0f));
    v.nz = PackSnorm16(std::clamp(avg_nz, -1.0f, 1.0f));
    // Color is baked separately via svo_bake_colors; store gray placeholder.
    v.r = 128;
    v.g = 128;
    v.b = 128;

    VoxelCoord coord{kx, ky, kz};
    auto [it, inserted] = voxels.try_emplace(coord, v);
    if (!inserted) {
      // Merge duplicate slot created by hash contention.
      auto& e = it->second;
      const float ew = static_cast<float>(e.weight);
      const float nw = static_cast<float>(v.weight);
      const float tw = ew + nw;
      const float inv_tw = 1.0f / tw;
      e.tsdf = PackSnorm16(
          (UnpackSnorm16(e.tsdf) * ew + UnpackSnorm16(v.tsdf) * nw) * inv_tw);
      e.nx = PackSnorm16((UnpackSnorm16(e.nx) * ew + UnpackSnorm16(v.nx) * nw) *
                         inv_tw);
      e.ny = PackSnorm16((UnpackSnorm16(e.ny) * ew + UnpackSnorm16(v.ny) * nw) *
                         inv_tw);
      e.nz = PackSnorm16((UnpackSnorm16(e.nz) * ew + UnpackSnorm16(v.nz) * nw) *
                         inv_tw);
      auto clamp_u8 = [](float x) -> uint8_t {
        return static_cast<uint8_t>(std::clamp(x, 0.0f, 255.0f));
      };
      e.r = clamp_u8(
          (static_cast<float>(e.r) * ew + static_cast<float>(v.r) * nw) *
          inv_tw);
      e.g = clamp_u8(
          (static_cast<float>(e.g) * ew + static_cast<float>(v.g) * nw) *
          inv_tw);
      e.b = clamp_u8(
          (static_cast<float>(e.b) * ew + static_cast<float>(v.b) * nw) *
          inv_tw);
      e.weight = static_cast<uint16_t>(std::min(tw, 65535.0f));
      ++merged;
    }
  }

  return voxels;
}

void SVOIntegratorCL::ExtractPoints(float min_weight, float voxel_size,
                                    uint32_t decimate_flat,
                                    float edge_threshold, int min_count,
                                    float relative_min_weight,
                                    std::vector<Vec3f>* points,
                                    std::vector<Vec3f>* normals,
                                    std::vector<Vec3<uint8_t>>* colors) {
  points->clear();
  normals->clear();
  colors->clear();

  if (capacity_ == 0) {
    return;
  }

  auto& dev = opencl::CLContext::Instance().Device(device_idx_);
  auto& ctx = dev.context();
  auto& queue = dev.queue();

  // Allocate output buffers.  Surface points are typically 5-10% of voxels;
  // capacity/4 is a generous upper bound.
  const size_t max_alloc = dev.device().getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
  // Each point needs 3*float(pos) + 3*float(norm) + 3*uint8(color) = 27 bytes.
  const uint32_t mem_limit_pts = static_cast<uint32_t>(std::min<size_t>(
      max_alloc / (3 * sizeof(float)), std::numeric_limits<uint32_t>::max()));
  const uint32_t max_output = std::min(capacity_ / 4, mem_limit_pts);

  const size_t pts_bytes = static_cast<size_t>(max_output) * 3 * sizeof(float);
  const size_t clr_bytes =
      static_cast<size_t>(max_output) * 3 * sizeof(uint8_t);

  {
    std::ostringstream oss;
    oss << "[SVOIntegratorCL] ExtractPoints: max_output=" << max_output << " ("
        << ((pts_bytes * 2 + clr_bytes) >> 20) << " MB buffers)";
    foundation::LogDebug("dense", oss.str());
  }

  cl_int err;
  cl::Buffer cl_out_pts(ctx, CL_MEM_WRITE_ONLY, pts_bytes, nullptr, &err);
  opencl::CheckCL(err, "extract points buffer");
  cl::Buffer cl_out_nrm(ctx, CL_MEM_WRITE_ONLY, pts_bytes, nullptr, &err);
  opencl::CheckCL(err, "extract normals buffer");
  cl::Buffer cl_out_clr(ctx, CL_MEM_WRITE_ONLY, clr_bytes, nullptr, &err);
  opencl::CheckCL(err, "extract colors buffer");
  cl::Buffer cl_out_counter(ctx, CL_MEM_READ_WRITE, sizeof(uint32_t), nullptr,
                            &err);
  opencl::CheckCL(err, "extract counter buffer");

  // Zero the counter.
  uint32_t zero = 0;
  queue.enqueueWriteBuffer(cl_out_counter, CL_TRUE, 0, sizeof(uint32_t), &zero);

  // min_weight is in SVOVoxel weight units (already scaled by kWeightScale).
  // The GPU stores sum_weight in WEIGHT_SCALE (128) units.
  const float min_weight_scaled = min_weight * kWeightScale;

  // Set kernel arguments.
  int arg = 0;
  k_extract_.setArg(arg++, cl_table_);
  k_extract_.setArg(arg++, capacity_mask_);
  k_extract_.setArg(arg++, capacity_);
  k_extract_.setArg(arg++, min_weight_scaled);
  k_extract_.setArg(arg++, voxel_size);
  k_extract_.setArg(arg++, static_cast<cl_uint>(decimate_flat));
  k_extract_.setArg(arg++, edge_threshold);
  k_extract_.setArg(arg++, static_cast<cl_int>(min_count));
  k_extract_.setArg(arg++, relative_min_weight);
  k_extract_.setArg(arg++, cl_out_pts);
  k_extract_.setArg(arg++, cl_out_nrm);
  k_extract_.setArg(arg++, cl_out_clr);
  k_extract_.setArg(arg++, cl_out_counter);
  k_extract_.setArg(arg++, max_output);

  // Launch: one work-item per hash table slot.
  const size_t global = ((static_cast<size_t>(capacity_) + 255u) / 256u) * 256u;
  queue.enqueueNDRangeKernel(k_extract_, cl::NullRange, cl::NDRange(global),
                             cl::NDRange(256));
  queue.finish();

  // Read back the count.
  uint32_t n_points = 0;
  queue.enqueueReadBuffer(cl_out_counter, CL_TRUE, 0, sizeof(uint32_t),
                          &n_points);
  if (n_points > max_output) {
    {
      std::ostringstream oss;
      oss << "[SVOIntegratorCL] Warning: extraction buffer overflow, "
          << n_points << " > " << max_output << ". Clamping.";
      foundation::LogWarning("dense", oss.str());
    }
    n_points = max_output;
  }

  {
    std::ostringstream oss;
    oss << "[SVOIntegratorCL] GPU extracted " << n_points << " surface points";
    foundation::LogInfo("dense", oss.str());
  }

  if (n_points == 0) {
    return;
  }

  // Read back only the used portion.
  const size_t read_pts = static_cast<size_t>(n_points) * 3 * sizeof(float);
  const size_t read_clr = static_cast<size_t>(n_points) * 3 * sizeof(uint8_t);

  std::vector<float> host_pts(n_points * 3);
  std::vector<float> host_nrm(n_points * 3);
  std::vector<uint8_t> host_clr(n_points * 3);

  queue.enqueueReadBuffer(cl_out_pts, CL_FALSE, 0, read_pts, host_pts.data());
  queue.enqueueReadBuffer(cl_out_nrm, CL_FALSE, 0, read_pts, host_nrm.data());
  queue.enqueueReadBuffer(cl_out_clr, CL_FALSE, 0, read_clr, host_clr.data());
  queue.finish();

  // Convert to output vectors.
  points->resize(n_points);
  normals->resize(n_points);
  colors->resize(n_points);

  for (uint32_t i = 0; i < n_points; ++i) {
    (*points)[i] =
        Vec3f(host_pts[i * 3], host_pts[i * 3 + 1], host_pts[i * 3 + 2]);
    (*normals)[i] =
        Vec3f(host_nrm[i * 3], host_nrm[i * 3 + 1], host_nrm[i * 3 + 2]);
    (*colors)[i] = Vec3<uint8_t>(host_clr[i * 3], host_clr[i * 3 + 1],
                                 host_clr[i * 3 + 2]);
  }
}

// ── Multi-level fill extraction ───────────────────────────────────────

void SVOIntegratorCL::ExtractFill(const cl::Buffer& fine_table,
                                  uint32_t fine_mask, float min_weight,
                                  float coarse_voxel_size,
                                  float fine_voxel_size, int level_shift,
                                  std::vector<Vec3f>* points,
                                  std::vector<Vec3f>* normals,
                                  std::vector<Vec3<uint8_t>>* colors) {
  points->clear();
  normals->clear();
  colors->clear();

  if (capacity_ == 0) {
    return;
  }

  auto& dev = opencl::CLContext::Instance().Device(device_idx_);
  auto& ctx = dev.context();
  auto& queue = dev.queue();

  // Budget: each coarse crossing can emit up to (1<<level_shift)^2 * 3 points.
  // Use capacity/4 as a safe upper bound (same heuristic as ExtractPoints).
  const size_t max_alloc = dev.device().getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
  const uint32_t mem_limit_pts = static_cast<uint32_t>(std::min<size_t>(
      max_alloc / (3 * sizeof(float)), std::numeric_limits<uint32_t>::max()));
  const uint32_t max_output = std::min(capacity_ / 4, mem_limit_pts);

  const size_t pts_bytes = static_cast<size_t>(max_output) * 3 * sizeof(float);
  const size_t clr_bytes =
      static_cast<size_t>(max_output) * 3 * sizeof(uint8_t);

  {
    std::ostringstream oss;
    oss << "[SVOIntegratorCL] ExtractFill L" << level_shift
        << ": max_output=" << max_output << " ("
        << ((pts_bytes * 2 + clr_bytes) >> 20) << " MB buffers)";
    foundation::LogDebug("dense", oss.str());
  }

  cl_int err;
  cl::Buffer cl_out_pts(ctx, CL_MEM_WRITE_ONLY, pts_bytes, nullptr, &err);
  opencl::CheckCL(err, "extract_fill points buffer");
  cl::Buffer cl_out_nrm(ctx, CL_MEM_WRITE_ONLY, pts_bytes, nullptr, &err);
  opencl::CheckCL(err, "extract_fill normals buffer");
  cl::Buffer cl_out_clr(ctx, CL_MEM_WRITE_ONLY, clr_bytes, nullptr, &err);
  opencl::CheckCL(err, "extract_fill colors buffer");
  cl::Buffer cl_out_counter(ctx, CL_MEM_READ_WRITE, sizeof(uint32_t), nullptr,
                            &err);
  opencl::CheckCL(err, "extract_fill counter buffer");

  uint32_t zero = 0;
  queue.enqueueWriteBuffer(cl_out_counter, CL_TRUE, 0, sizeof(uint32_t), &zero);

  const float min_weight_scaled = min_weight * kWeightScale;

  int arg = 0;
  k_extract_fill_.setArg(arg++, cl_table_);
  k_extract_fill_.setArg(arg++, capacity_mask_);
  k_extract_fill_.setArg(arg++, capacity_);
  k_extract_fill_.setArg(arg++, fine_table);
  k_extract_fill_.setArg(arg++, fine_mask);
  k_extract_fill_.setArg(arg++, min_weight_scaled);
  k_extract_fill_.setArg(arg++, coarse_voxel_size);
  k_extract_fill_.setArg(arg++, fine_voxel_size);
  k_extract_fill_.setArg(arg++, level_shift);
  k_extract_fill_.setArg(arg++, cl_out_pts);
  k_extract_fill_.setArg(arg++, cl_out_nrm);
  k_extract_fill_.setArg(arg++, cl_out_clr);
  k_extract_fill_.setArg(arg++, cl_out_counter);
  k_extract_fill_.setArg(arg++, max_output);

  const size_t global = ((static_cast<size_t>(capacity_) + 255u) / 256u) * 256u;
  queue.enqueueNDRangeKernel(k_extract_fill_, cl::NullRange,
                             cl::NDRange(global), cl::NDRange(256));
  queue.finish();

  uint32_t n_points = 0;
  queue.enqueueReadBuffer(cl_out_counter, CL_TRUE, 0, sizeof(uint32_t),
                          &n_points);
  if (n_points > max_output) {
    {
      std::ostringstream oss;
      oss << "[SVOIntegratorCL] Warning: extract_fill buffer overflow, "
          << n_points << " > " << max_output << ". Clamping.";
      foundation::LogWarning("dense", oss.str());
    }
    n_points = max_output;
  }

  {
    std::ostringstream oss;
    oss << "[SVOIntegratorCL] GPU extract_fill L" << level_shift << ": "
        << n_points << " fill points";
    foundation::LogInfo("dense", oss.str());
  }

  if (n_points == 0) {
    return;
  }

  const size_t read_pts = static_cast<size_t>(n_points) * 3 * sizeof(float);
  const size_t read_clr = static_cast<size_t>(n_points) * 3 * sizeof(uint8_t);

  std::vector<float> host_pts(n_points * 3);
  std::vector<float> host_nrm(n_points * 3);
  std::vector<uint8_t> host_clr(n_points * 3);

  queue.enqueueReadBuffer(cl_out_pts, CL_FALSE, 0, read_pts, host_pts.data());
  queue.enqueueReadBuffer(cl_out_nrm, CL_FALSE, 0, read_pts, host_nrm.data());
  queue.enqueueReadBuffer(cl_out_clr, CL_FALSE, 0, read_clr, host_clr.data());
  queue.finish();

  points->resize(n_points);
  normals->resize(n_points);
  colors->resize(n_points);

  for (uint32_t i = 0; i < n_points; ++i) {
    (*points)[i] =
        Vec3f(host_pts[i * 3], host_pts[i * 3 + 1], host_pts[i * 3 + 2]);
    (*normals)[i] =
        Vec3f(host_nrm[i * 3], host_nrm[i * 3 + 1], host_nrm[i * 3 + 2]);
    (*colors)[i] = Vec3<uint8_t>(host_clr[i * 3], host_clr[i * 3 + 1],
                                 host_clr[i * 3 + 2]);
  }
}

// ── Photometric refinement ────────────────────────────────────────────

void SVOIntegratorCL::PrepareRefinement(
    const std::vector<SVOCameraGPU>& cameras,
    const std::vector<RefineViewSrc>& views, int img_width, int img_height,
    int n_views) {
  if (capacity_ == 0) {
    throw std::runtime_error(
        "SVOIntegratorCL::PrepareRefinement: Initialize() + Integrate() "
        "must be called first");
  }

  auto& dev = opencl::CLContext::Instance().Device(device_idx_);
  auto& ctx = dev.context();
  auto& queue = dev.queue();
  cl_int err;

  n_refine_views_ = n_views;
  refine_img_width_ = img_width;
  refine_img_height_ = img_height;

  // Free any prior refinement working set before reallocating
  cl_color_images_ = cl::Image2DArray();
  cl_tsdf_depths_ = cl::Image2DArray();
  cl_clean_depths_ = cl::Image2DArray();
  cl_cameras_array_ = cl::Buffer();

  // Allocate gradient buffer: 1 float per slot.
  const size_t grad_bytes = static_cast<size_t>(capacity_) * sizeof(float);
  cl_refine_grad_ =
      cl::Buffer(ctx, CL_MEM_READ_WRITE, grad_bytes, nullptr, &err);
  opencl::CheckCL(err, "refine grad buffer");
  cl_refine_grad_w_ =
      cl::Buffer(ctx, CL_MEM_READ_WRITE, grad_bytes, nullptr, &err);
  opencl::CheckCL(err, "refine grad buffer");

  // Allocate Adam buffer: 2 floats per slot (m_d, v_d).
  const size_t adam_bytes = static_cast<size_t>(capacity_) * 2 * sizeof(float);
  cl_refine_adam_ =
      cl::Buffer(ctx, CL_MEM_READ_WRITE, adam_bytes, nullptr, &err);
  opencl::CheckCL(err, "refine adam buffer");

  // Upload color + depth as image2d_array_t, streaming each view's slice
  // straight from the caller's borrowed buffers via enqueueWriteImage.  This
  // avoids materialising a full per-cluster packed copy on the host (the old
  // path held ~3 GB of packed_colors/masks/depths + an RGBA expansion); the
  // only host scratch now is one reused npix*4 RGBA row-block.
  const size_t npix = static_cast<size_t>(img_width) * img_height;

  // Color: CL_RGBA CL_UNORM_INT8, alpha channel = validity mask.
  cl::ImageFormat color_fmt(CL_RGBA, CL_UNORM_INT8);
  cl_color_images_ = cl::Image2DArray(
      ctx, CL_MEM_READ_ONLY, color_fmt, static_cast<cl::size_type>(n_views),
      static_cast<cl::size_type>(img_width),
      static_cast<cl::size_type>(img_height), 0, 0, nullptr, &err);
  opencl::CheckCL(err, "refine color image2d_array");

  // TSDF-rendered depth array, pre-filled with clean depths.  The clean depths
  // serve as initial hints for guided narrow-band raycast, avoiding blind
  // marching from 0.1→100m (~2000 steps vs ~12 steps).
  cl::ImageFormat depth_fmt(CL_R, CL_FLOAT);
  cl_tsdf_depths_ = cl::Image2DArray(
      ctx, CL_MEM_READ_WRITE, depth_fmt, static_cast<cl::size_type>(n_views),
      static_cast<cl::size_type>(img_width),
      static_cast<cl::size_type>(img_height), 0, 0, nullptr, &err);
  opencl::CheckCL(err, "refine tsdf depth image2d_array");

  // Immutable copy of clean depths for bake occlusion (not modified by
  // refinement raycasts).  Clean depths represent the front-most surface
  // each camera actually observed — ideal for occlusion testing.
  cl_clean_depths_ = cl::Image2DArray(
      ctx, CL_MEM_READ_ONLY, depth_fmt, static_cast<cl::size_type>(n_views),
      static_cast<cl::size_type>(img_width),
      static_cast<cl::size_type>(img_height), 0, 0, nullptr, &err);
  opencl::CheckCL(err, "clean depth image2d_array for bake occlusion");

  // One reused RGBA scratch row-block (npix*4 bytes, independent of n_views):
  // RGB from the view's color (black if absent), alpha from its validity mask
  // (all-valid if absent) — byte-identical to the old packed→RGBA expansion.
  std::vector<uint8_t> rgba_slice(npix * 4);
  const cl::array<cl::size_type, 3> region = {
      {static_cast<cl::size_type>(img_width),
       static_cast<cl::size_type>(img_height), 1}};
  int upload_v = -1;  // last view attempted (for error reporting)
  try {
    for (int v = 0; v < n_views; ++v) {
      upload_v = v;
      const cl::array<cl::size_type, 3> origin = {
          {0, 0, static_cast<cl::size_type>(v)}};
      const uint8_t* color = views[v].color;
      const uint8_t* mask = views[v].mask;
      const float* depth = views[v].depth;
      if (depth == nullptr) {
        throw std::runtime_error(
            "PrepareRefinement: view " + std::to_string(v) +
            " has a null depth buffer (every view must carry a depth map)");
      }
      for (size_t p = 0; p < npix; ++p) {
        rgba_slice[p * 4 + 0] = color ? color[p * 3 + 0] : 0;
        rgba_slice[p * 4 + 1] = color ? color[p * 3 + 1] : 0;
        rgba_slice[p * 4 + 2] = color ? color[p * 3 + 2] : 0;
        rgba_slice[p * 4 + 3] = mask ? mask[p] : 255;
      }
      // Blocking writes (CL_TRUE): the color write must finish before
      // rgba_slice is overwritten for the next view; depth streams from the
      // (stable) borrowed buffer into both the mutable and immutable depth
      // arrays.
      queue.enqueueWriteImage(cl_color_images_, CL_TRUE, origin, region, 0, 0,
                              rgba_slice.data());
      queue.enqueueWriteImage(cl_tsdf_depths_, CL_TRUE, origin, region, 0, 0,
                              const_cast<float*>(depth));
      queue.enqueueWriteImage(cl_clean_depths_, CL_TRUE, origin, region, 0, 0,
                              const_cast<float*>(depth));
    }
  } catch (const cl::Error& e) {
    // The bare cl2.hpp message is just the API name ("clEnqueueWriteImage").
    // Most drivers defer image-array allocation to first use, so an
    // out-of-memory condition surfaces here rather than at creation — decode
    // the code and report the requested vs available GPU memory so the cause
    // (OOM vs invalid argument) is unambiguous.
    const size_t per_array_bytes = static_cast<size_t>(n_views) * npix * 4;
    std::ostringstream oss;
    oss << "PrepareRefinement image upload failed at view " << upload_v << "/"
        << n_views << " (" << img_width << "x" << img_height
        << "): " << e.what() << " (CL code " << e.err()
        << "). Requested GPU: " << ((3 * per_array_bytes) >> 20)
        << " MB images (color+2×depth) + "
        << ((grad_bytes * 2 + adam_bytes) >> 20) << " MB grad/adam; device "
        << dev.name() << " has " << (dev.GlobalMemSize() >> 20) << " MB total, "
        << (dev.AvailableMemory() >> 20)
        << " MB tracked-free. CL codes: -4=MEM_OBJECT_ALLOCATION_FAILURE, "
           "-5=OUT_OF_RESOURCES, -6=OUT_OF_HOST_MEMORY, -30=INVALID_VALUE.";
    throw std::runtime_error(oss.str());
  }

  // Upload camera array.
  const size_t cam_bytes = static_cast<size_t>(n_views) * sizeof(SVOCameraGPU);
  cl_cameras_array_ =
      cl::Buffer(ctx, CL_MEM_READ_ONLY, cam_bytes, nullptr, &err);
  opencl::CheckCL(err, "refine cameras buffer");
  queue.enqueueWriteBuffer(cl_cameras_array_, CL_FALSE, 0, cam_bytes,
                           cameras.data());

  // Clear gradient and Adam buffers.
  queue.finish();
  {
    const size_t global =
        ((static_cast<size_t>(capacity_) + 255u) / 256u) * 256u;
    k_refine_clear_.setArg(0, cl_refine_grad_);
    k_refine_clear_.setArg(1, cl_refine_grad_w_);
    k_refine_clear_.setArg(2, cl_refine_adam_);
    k_refine_clear_.setArg(3, capacity_);
    k_refine_clear_.setArg(4, static_cast<cl_int>(1));  // clear Adam too
    queue.enqueueNDRangeKernel(k_refine_clear_, cl::NullRange,
                               cl::NDRange(global), cl::NDRange(256));
    queue.finish();
  }

  {
    std::ostringstream oss;
    oss << "[SVOIntegratorCL] PrepareRefinement: " << n_views << " views ("
        << img_width << "x" << img_height << "), "
        << ((grad_bytes + adam_bytes) >> 20) << " MB grad+adam";
    foundation::LogInfo("dense", oss.str());
  }

  refine_prepared_ = true;
}

void SVOIntegratorCL::ReleaseRefineBuffers() {
  cl_refine_grad_ = cl::Buffer();
  cl_refine_grad_w_ = cl::Buffer();
  cl_refine_adam_ = cl::Buffer();
  cl_color_images_ = cl::Image2DArray();
  cl_tsdf_depths_ = cl::Image2DArray();
  cl_clean_depths_ = cl::Image2DArray();
  cl_cameras_array_ = cl::Buffer();
  n_refine_views_ = 0;
  refine_img_width_ = 0;
  refine_img_height_ = 0;
  refine_prepared_ = false;
}

void SVOIntegratorCL::RefineGeometry(int iters, float lambda_reg,
                                     float voxel_size, float trunc_dist,
                                     float min_weight,
                                     const std::vector<int32_t>& neighbor_data,
                                     int max_neighbors) {
  if (!refine_prepared_) {
    throw std::runtime_error(
        "SVOIntegratorCL::RefineGeometry: PrepareRefinement() not called");
  }

  auto& dev = opencl::CLContext::Instance().Device(device_idx_);
  auto& ctx = dev.context();
  auto& queue = dev.queue();
  cl_int err;

  refine_voxel_size_ = voxel_size;
  refine_min_weight_ = min_weight;

  const float inv_voxel_size = 1.0f / voxel_size;
  const float min_weight_scaled = min_weight * kWeightScale;
  const float lr_sdf =
      50.0f * voxel_size;       // 1.0 for voxel=0.02: ~0.1 step/iter (clipped)
  const float beta1 = 0.7f;     // unused, kept for kernel API compat
  const float epsilon = 1e-8f;  // unused
  const int half_patch = 5;
  const float inv_sigma_s2 = -0.125f;  // sigma_spatial = 1 px
  // Texture-confidence floor (patch-variance units, gray²).
  const float tex_floor = 0.05f;
  // Edge-aware regularization tuning (used only when lambda_reg > 0).
  // edge_k: TSDF jump (normalized units) above which diffusion stops, i.e. the
  //   crease/corner threshold — smaller keeps more edges, larger smooths more.
  const float edge_k = 0.3f;

  const int W = refine_img_width_;
  const int H = refine_img_height_;
  const size_t update_global =
      ((static_cast<size_t>(capacity_) + 255u) / 256u) * 256u;

  // Temp buffer for raycast depth (1 view at a time, then copied into the
  // image array slice).
  const size_t depth_buf_bytes = static_cast<size_t>(W) * H * sizeof(float);
  cl::Buffer cl_raycast_depth(ctx, CL_MEM_READ_WRITE, depth_buf_bytes, nullptr,
                              &err);
  opencl::CheckCL(err, "refine raycast depth buffer");

  // Temp buffer for hit_slot (needed by svo_raycast, unused here).
  cl::Buffer cl_raycast_hit(ctx, CL_MEM_READ_WRITE,
                            static_cast<size_t>(W) * H * sizeof(uint32_t),
                            nullptr, &err);
  opencl::CheckCL(err, "refine raycast hit buffer");

  // Upload neighbor buffer.
  std::vector<int32_t> neighbor_data_work = neighbor_data;
  const size_t nb_bytes =
      static_cast<size_t>(n_refine_views_) * max_neighbors * sizeof(int32_t);
  cl::Buffer cl_neighbors(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          nb_bytes, neighbor_data_work.data(), &err);
  opencl::CheckCL(err, "refine neighbor buffer");

  // Per-iteration convergence instrumentation (opt-in, env-gated so it costs
  // nothing in production): when OPENSFM_REFINE_LOG_STATS is set, read the
  // accumulated gradient back each iteration and log the descent statistics so
  // we can judge where the minimisation plateaus and whether early-stopping is
  // safe.  Read-only — does not perturb the descent.
  const bool log_stats = std::getenv("OPENSFM_REFINE_LOG_STATS") != nullptr;
  std::vector<float> grad_cpu, gradw_cpu;
  if (log_stats) {
    grad_cpu.resize(capacity_);
    gradw_cpu.resize(capacity_);
  }
  double prev_rms = 0.0;

  for (int iter = 0; iter < iters; ++iter) {
    // ---- Step 1: Guided narrow-band raycast for all views ----
    // Uses cl_tsdf_depths_ as depth hints (clean depths on iter 0, then
    // previous iteration's rendered depths). Search margin = trunc_dist +
    // 2*voxel_size.
    const float search_margin = trunc_dist + 2.0f * voxel_size;

    for (int vi = 0; vi < n_refine_views_; ++vi) {
      // Create sub-buffer for this camera (svo_raycast_guided takes
      // __constant).
      const size_t cam_offset = vi * sizeof(SVOCameraGPU);
      cl_buffer_region region{cam_offset, sizeof(SVOCameraGPU)};
      cl::Buffer cam_sub = cl_cameras_array_.createSubBuffer(
          CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
      if (err != CL_SUCCESS) {
        SVOCameraGPU single_cam;
        queue.enqueueReadBuffer(cl_cameras_array_, CL_TRUE, cam_offset,
                                sizeof(SVOCameraGPU), &single_cam);
        queue.enqueueWriteBuffer(cl_camera_, CL_TRUE, 0, sizeof(SVOCameraGPU),
                                 &single_cam);
        cam_sub = cl_camera_;
      }

      int rarg = 0;
      k_raycast_guided_.setArg(rarg++, cl_table_);
      k_raycast_guided_.setArg(rarg++, capacity_mask_);
      k_raycast_guided_.setArg(rarg++, cl_raycast_depth);
      k_raycast_guided_.setArg(rarg++, cl_raycast_hit);
      k_raycast_guided_.setArg(rarg++, cam_sub);
      k_raycast_guided_.setArg(rarg++, cl_tsdf_depths_);
      k_raycast_guided_.setArg(rarg++, static_cast<cl_int>(vi));
      k_raycast_guided_.setArg(rarg++, static_cast<cl_int>(H));
      k_raycast_guided_.setArg(rarg++, static_cast<cl_int>(W));
      k_raycast_guided_.setArg(rarg++, voxel_size);
      k_raycast_guided_.setArg(rarg++, inv_voxel_size);
      k_raycast_guided_.setArg(rarg++, search_margin);
      k_raycast_guided_.setArg(
          rarg++, min_weight);  // raw units; kernel multiplies by WEIGHT_SCALE

      cl::NDRange global_rc(static_cast<size_t>((W + 15) / 16 * 16),
                            static_cast<size_t>((H + 15) / 16 * 16));
      queue.enqueueNDRangeKernel(k_raycast_guided_, cl::NullRange, global_rc,
                                 cl::NDRange(16, 16));

      // Copy raycast depth buffer → tsdf_depths image array slice vi.
      std::array<size_t, 3> origin = {0, 0, static_cast<size_t>(vi)};
      std::array<size_t, 3> region_sz = {static_cast<size_t>(W),
                                         static_cast<size_t>(H), 1};
      queue.enqueueCopyBufferToImage(cl_raycast_depth, cl_tsdf_depths_, 0,
                                     origin, region_sz);
    }
    queue.finish();

    // ---- Step 2: Accumulate gradient from each source view ----
    for (int vi = 0; vi < n_refine_views_; ++vi) {
      int arg = 0;
      k_refine_accumulate_.setArg(arg++, cl_table_);
      k_refine_accumulate_.setArg(arg++, capacity_mask_);
      k_refine_accumulate_.setArg(arg++, cl_refine_grad_);
      k_refine_accumulate_.setArg(arg++, cl_refine_grad_w_);
      k_refine_accumulate_.setArg(arg++, cl_color_images_);
      k_refine_accumulate_.setArg(arg++, cl_tsdf_depths_);
      k_refine_accumulate_.setArg(arg++, cl_cameras_array_);
      k_refine_accumulate_.setArg(arg++, cl_neighbors);
      k_refine_accumulate_.setArg(arg++, static_cast<cl_int>(max_neighbors));
      k_refine_accumulate_.setArg(arg++, static_cast<cl_int>(n_refine_views_));
      k_refine_accumulate_.setArg(arg++, static_cast<cl_int>(vi));
      k_refine_accumulate_.setArg(arg++, static_cast<cl_int>(W));
      k_refine_accumulate_.setArg(arg++, static_cast<cl_int>(H));
      k_refine_accumulate_.setArg(arg++, voxel_size);
      k_refine_accumulate_.setArg(arg++, inv_voxel_size);
      k_refine_accumulate_.setArg(arg++, trunc_dist);
      k_refine_accumulate_.setArg(arg++, min_weight_scaled);
      k_refine_accumulate_.setArg(arg++, static_cast<cl_int>(half_patch));
      k_refine_accumulate_.setArg(arg++, inv_sigma_s2);
      k_refine_accumulate_.setArg(arg++, tex_floor);

      const size_t gw = static_cast<size_t>((W + 15) / 16 * 16);
      const size_t gh = static_cast<size_t>((H + 15) / 16 * 16);
      queue.enqueueNDRangeKernel(k_refine_accumulate_, cl::NullRange,
                                 cl::NDRange(gw, gh), cl::NDRange(16, 16));
    }
    queue.finish();

    // ---- Diagnostic: per-iteration descent statistics (env-gated) ----
    // Read grad and its per-voxel pixel count BEFORE the update kernel clears
    // them, and report the per-voxel mean gradient magnitude |grad/grad_w| —
    // exactly the force the update applies.  As the surface reaches
    // photo-consistency this decays toward a floor; the rms/prev ratio makes
    // the plateau (the natural early-stop point) obvious.
    if (log_stats) {
      queue.enqueueReadBuffer(cl_refine_grad_, CL_TRUE, 0,
                              capacity_ * sizeof(float), grad_cpu.data());
      queue.enqueueReadBuffer(cl_refine_grad_w_, CL_TRUE, 0,
                              capacity_ * sizeof(float), gradw_cpu.data());
      int n_active = 0;
      float max_abs = 0.0f;
      double sum_abs = 0.0, sum_sq = 0.0;
      for (uint32_t gi = 0; gi < capacity_; ++gi) {
        float w = gradw_cpu[gi];
        if (w <= 0.0f) {
          continue;
        }
        float g = grad_cpu[gi] / (w + 1e-6f);  // mirror update kernel
        float a = std::abs(g);
        ++n_active;
        sum_abs += a;
        sum_sq += static_cast<double>(g) * g;
        if (a > max_abs) {
          max_abs = a;
        }
      }
      const double mean_abs = n_active > 0 ? sum_abs / n_active : 0.0;
      const double rms = n_active > 0 ? std::sqrt(sum_sq / n_active) : 0.0;
      const double ratio = prev_rms > 0.0 ? rms / prev_rms : 1.0;
      prev_rms = rms;
      std::ostringstream oss;
      oss << "[SVORefine] iter " << (iter + 1) << "/" << iters
          << " grad: nact=" << n_active << " rms=" << rms
          << " mean=" << mean_abs << " max=" << max_abs
          << " rms/prev=" << ratio;
      foundation::LogInfo("dense", oss.str());
    }

    // ---- Step 3: Adam update on SDF ----
    {
      int arg = 0;
      k_refine_update_.setArg(arg++, cl_table_);
      k_refine_update_.setArg(arg++, capacity_mask_);
      k_refine_update_.setArg(arg++, capacity_);
      k_refine_update_.setArg(arg++, cl_refine_grad_);
      k_refine_update_.setArg(arg++, cl_refine_grad_w_);
      k_refine_update_.setArg(arg++, cl_refine_adam_);
      k_refine_update_.setArg(arg++, min_weight_scaled);
      k_refine_update_.setArg(arg++, voxel_size);
      k_refine_update_.setArg(arg++, lambda_reg);
      k_refine_update_.setArg(arg++, lr_sdf);
      k_refine_update_.setArg(arg++, beta1);
      k_refine_update_.setArg(arg++, epsilon);
      k_refine_update_.setArg(arg++, static_cast<cl_int>(iter));
      k_refine_update_.setArg(arg++, static_cast<cl_int>(n_refine_views_));
      k_refine_update_.setArg(arg++, edge_k);
      queue.enqueueNDRangeKernel(k_refine_update_, cl::NullRange,
                                 cl::NDRange(update_global), cl::NDRange(256));
      queue.finish();
    }

    if (!log_stats && ((iter + 1) % 5 == 0 || iter == iters - 1)) {
      std::ostringstream oss;
      oss << "[SVOIntegratorCL] RefineGeometry iter " << (iter + 1) << "/"
          << iters;
      foundation::LogInfo("dense", oss.str());
    }
  }

  // Keep color images and cameras alive for BakeColors.
  // Release grad/adam.
  cl_refine_grad_ = cl::Buffer();
  cl_refine_grad_w_ = cl::Buffer();
  cl_refine_adam_ = cl::Buffer();
}

void SVOIntegratorCL::BakeColors(const std::vector<Vec3f>& points,
                                 const std::vector<Vec3f>& normals,
                                 std::vector<Vec3<uint8_t>>* out_colors,
                                 int n_final, int irls_iters,
                                 const std::vector<uint8_t>* relax_occ) {
  if (!refine_prepared_) {
    throw std::runtime_error(
        "SVOIntegratorCL::BakeColors: PrepareRefinement() not called");
  }

  auto& dev = opencl::CLContext::Instance().Device(device_idx_);
  auto& ctx = dev.context();
  auto& queue = dev.queue();
  cl_int err;

  const size_t M = points.size();
  if (M == 0) {
    out_colors->clear();
    return;
  }

  // Upload points and normals (as flat float arrays).
  const size_t pts_bytes = M * 3 * sizeof(float);
  cl::Buffer cl_pts(
      ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, pts_bytes,
      const_cast<float*>(reinterpret_cast<const float*>(points.data())), &err);
  opencl::CheckCL(err, "bake points buffer");

  cl::Buffer cl_nrm(
      ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, pts_bytes,
      const_cast<float*>(reinterpret_cast<const float*>(normals.data())), &err);
  opencl::CheckCL(err, "bake normals buffer");

  // Optional per-point occlusion-relax flags (1 = skip the occlusion test for
  // that point — used for interpolated filled-DSM cells).
  const bool has_relax = relax_occ && relax_occ->size() == M;
  cl::Buffer cl_relax;
  if (has_relax) {
    cl_relax = cl::Buffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, M,
                          const_cast<uint8_t*>(relax_occ->data()), &err);
    opencl::CheckCL(err, "bake relax buffer");
  } else {
    cl_relax = cl::Buffer(ctx, CL_MEM_READ_ONLY, 1, nullptr, &err);
    opencl::CheckCL(err, "bake relax dummy buffer");
  }

  // Output color buffer.
  const size_t color_bytes = M * 3;
  cl::Buffer cl_out(ctx, CL_MEM_WRITE_ONLY, color_bytes, nullptr, &err);
  opencl::CheckCL(err, "bake output buffer");

  // Use clean depth maps for occlusion check in color baking.
  // Clean depths represent the front-most surface each camera actually
  // observed — guaranteed correct for occlusion without re-raycasting.
  const int W = refine_img_width_;
  const int H = refine_img_height_;

  // Dispatch svo_bake_colors kernel.
  {
    const float occlusion_margin = 3.0f * refine_voxel_size_;
    int arg = 0;
    k_bake_colors_.setArg(arg++, cl_pts);
    k_bake_colors_.setArg(arg++, cl_nrm);
    k_bake_colors_.setArg(arg++, cl_out);
    k_bake_colors_.setArg(arg++, cl_color_images_);
    k_bake_colors_.setArg(arg++, cl_clean_depths_);
    k_bake_colors_.setArg(arg++, cl_cameras_array_);
    k_bake_colors_.setArg(arg++, static_cast<cl_int>(n_refine_views_));
    k_bake_colors_.setArg(arg++, static_cast<cl_int>(W));
    k_bake_colors_.setArg(arg++, static_cast<cl_int>(H));
    k_bake_colors_.setArg(arg++, occlusion_margin);
    k_bake_colors_.setArg(arg++, static_cast<cl_int>(n_final));
    k_bake_colors_.setArg(arg++, static_cast<cl_int>(irls_iters));
    k_bake_colors_.setArg(arg++, cl_relax);
    k_bake_colors_.setArg(arg++, static_cast<cl_int>(has_relax ? 1 : 0));
    k_bake_colors_.setArg(arg++, static_cast<cl_int>(M));

    const size_t global = ((M + 255) / 256) * 256;
    queue.enqueueNDRangeKernel(k_bake_colors_, cl::NullRange,
                               cl::NDRange(global), cl::NDRange(256));
    queue.finish();
  }

  // Read back colors.
  out_colors->resize(M);
  queue.enqueueReadBuffer(cl_out, CL_TRUE, 0, color_bytes, out_colors->data());

  // Release all refinement resources.
  cl_color_images_ = cl::Image2DArray();
  cl_tsdf_depths_ = cl::Image2DArray();
  cl_clean_depths_ = cl::Image2DArray();
  cl_cameras_array_ = cl::Buffer();
  refine_prepared_ = false;

  {
    std::ostringstream oss;
    oss << "[SVOIntegratorCL] BakeColors: " << M << " points colored";
    foundation::LogInfo("dense", oss.str());
  }
}

// =====================================================================
// Visibility pruning implementation
// =====================================================================

void SVOIntegratorCL::InitializeVisibilityPruning() {
  BuildKernels();

  if (capacity_ == 0) {
    throw std::runtime_error(
        "InitializeVisibilityPruning: hash table not initialized");
  }

  auto& dev = opencl::CLContext::Instance().Device(device_idx_);
  cl::Context& ctx = dev.context();
  cl::CommandQueue& queue = dev.queue();
  cl_int err;

  const size_t slots_bytes = capacity_ * sizeof(int32_t);

  cl_carve_count_ =
      cl::Buffer(ctx, CL_MEM_READ_WRITE, slots_bytes, nullptr, &err);
  opencl::CheckCL(err, "carve_count buffer");

  cl_support_count_ =
      cl::Buffer(ctx, CL_MEM_READ_WRITE, slots_bytes, nullptr, &err);
  opencl::CheckCL(err, "support_count buffer");

  // Clear both counters.
  k_clear_votes_.setArg(0, cl_carve_count_);
  k_clear_votes_.setArg(1, cl_support_count_);
  k_clear_votes_.setArg(2, capacity_);

  size_t global = ((capacity_ + 255) / 256) * 256;
  queue.enqueueNDRangeKernel(k_clear_votes_, cl::NullRange, cl::NDRange(global),
                             cl::NDRange(256));
  queue.finish();

  visibility_initialized_ = true;
}

void SVOIntegratorCL::RaycastAndVote(const Mat3f& K, const Mat3f& R,
                                     const Vec3f& t, const float* clean_depth,
                                     int rows, int cols, float voxel_size,
                                     float min_depth, float max_depth,
                                     float min_weight, float carve_margin) {
  if (!visibility_initialized_) {
    throw std::runtime_error(
        "RaycastAndVote: call InitializeVisibilityPruning first");
  }

  auto& dev = opencl::CLContext::Instance().Device(device_idx_);
  cl::Context& ctx = dev.context();
  cl::CommandQueue& queue = dev.queue();
  cl_int err;

  const size_t npix = static_cast<size_t>(rows) * cols;
  const size_t pix_float_bytes = npix * sizeof(float);
  const size_t pix_uint_bytes = npix * sizeof(uint32_t);
  const float inv_voxel_size = 1.0f / voxel_size;

  // Ensure per-frame buffers are large enough.
  if (npix > raycast_pixels_) {
    cl_rendered_depth_ =
        cl::Buffer(ctx, CL_MEM_READ_WRITE, pix_float_bytes, nullptr, &err);
    opencl::CheckCL(err, "rendered_depth buffer");
    cl_hit_slot_ =
        cl::Buffer(ctx, CL_MEM_READ_WRITE, pix_uint_bytes, nullptr, &err);
    opencl::CheckCL(err, "hit_slot buffer");
    cl_clean_depth_ =
        cl::Buffer(ctx, CL_MEM_READ_ONLY, pix_float_bytes, nullptr, &err);
    opencl::CheckCL(err, "clean_depth buffer");
    raycast_pixels_ = npix;
  }

  // Upload clean depth.
  queue.enqueueWriteBuffer(cl_clean_depth_, CL_FALSE, 0, pix_float_bytes,
                           clean_depth);

  // Upload camera.
  SVOCameraGPU cam_gpu;
  Eigen::Matrix3f Kinv = K.inverse();
  Eigen::Matrix3f Rinv = R.transpose();
  Eigen::Vector3f cam_pos = -Rinv * t;
  for (int i = 0; i < 9; ++i) {
    cam_gpu.Kinv[i] = Kinv.data()[i];
  }
  for (int i = 0; i < 9; ++i) {
    cam_gpu.Rinv[i] = Rinv.data()[i];
  }
  for (int i = 0; i < 9; ++i) {
    cam_gpu.R[i] = R.data()[i];
  }
  for (int i = 0; i < 3; ++i) {
    cam_gpu.t[i] = t[i];
  }
  for (int i = 0; i < 3; ++i) {
    cam_gpu.cam_pos[i] = cam_pos[i];
  }
  for (int i = 0; i < 3; ++i) {
    cam_gpu._pad[i] = 0.0f;
  }

  // Reuse cl_camera_ buffer (already allocated by Integrate).
  if (!cl_camera_()) {
    cl_camera_ =
        cl::Buffer(ctx, CL_MEM_READ_ONLY, sizeof(SVOCameraGPU), nullptr, &err);
    opencl::CheckCL(err, "camera buffer for raycast");
  }
  queue.enqueueWriteBuffer(cl_camera_, CL_FALSE, 0, sizeof(SVOCameraGPU),
                           &cam_gpu);

  // --- Dispatch svo_raycast ---
  int arg = 0;
  k_raycast_.setArg(arg++, cl_table_);
  k_raycast_.setArg(arg++, capacity_mask_);
  k_raycast_.setArg(arg++, cl_rendered_depth_);
  k_raycast_.setArg(arg++, cl_hit_slot_);
  k_raycast_.setArg(arg++, cl_camera_);
  k_raycast_.setArg(arg++, rows);
  k_raycast_.setArg(arg++, cols);
  k_raycast_.setArg(arg++, voxel_size);
  k_raycast_.setArg(arg++, inv_voxel_size);
  k_raycast_.setArg(arg++, min_depth);
  k_raycast_.setArg(arg++, max_depth);
  k_raycast_.setArg(arg++, min_weight);

  cl::NDRange global_2d(((cols + 15) / 16) * 16, ((rows + 15) / 16) * 16);
  cl::NDRange local_2d(16, 16);
  queue.enqueueNDRangeKernel(k_raycast_, cl::NullRange, global_2d, local_2d);

  // --- Dispatch svo_carve_vote ---
  arg = 0;
  k_carve_vote_.setArg(arg++, cl_rendered_depth_);
  k_carve_vote_.setArg(arg++, cl_hit_slot_);
  k_carve_vote_.setArg(arg++, cl_clean_depth_);
  k_carve_vote_.setArg(arg++, cl_carve_count_);
  k_carve_vote_.setArg(arg++, cl_support_count_);
  k_carve_vote_.setArg(arg++, rows);
  k_carve_vote_.setArg(arg++, cols);
  k_carve_vote_.setArg(arg++, carve_margin);
  k_carve_vote_.setArg(arg++, capacity_mask_);

  queue.enqueueNDRangeKernel(k_carve_vote_, cl::NullRange, global_2d, local_2d);
  queue.finish();
}

void SVOIntegratorCL::Prune(int carve_threshold, int support_min,
                            float weight_penalty_per_vote) {
  if (!visibility_initialized_) {
    throw std::runtime_error("Prune: call InitializeVisibilityPruning first");
  }

  auto& dev = opencl::CLContext::Instance().Device(device_idx_);
  cl::CommandQueue& queue = dev.queue();

  int weight_penalty =
      static_cast<int>(weight_penalty_per_vote * 128.0f);  // WEIGHT_SCALE

  int arg = 0;
  k_prune_.setArg(arg++, cl_table_);
  k_prune_.setArg(arg++, cl_carve_count_);
  k_prune_.setArg(arg++, cl_support_count_);
  k_prune_.setArg(arg++, capacity_);
  k_prune_.setArg(arg++, carve_threshold);
  k_prune_.setArg(arg++, support_min);
  k_prune_.setArg(arg++, weight_penalty);

  size_t global = ((capacity_ + 255) / 256) * 256;
  queue.enqueueNDRangeKernel(k_prune_, cl::NullRange, cl::NDRange(global),
                             cl::NDRange(256));
  queue.finish();
}

void SVOIntegratorCL::ClearVotes() {
  if (!visibility_initialized_) {
    return;
  }

  auto& dev = opencl::CLContext::Instance().Device(device_idx_);
  cl::CommandQueue& queue = dev.queue();

  k_clear_votes_.setArg(0, cl_carve_count_);
  k_clear_votes_.setArg(1, cl_support_count_);
  k_clear_votes_.setArg(2, capacity_);

  size_t global = ((capacity_ + 255) / 256) * 256;
  queue.enqueueNDRangeKernel(k_clear_votes_, cl::NullRange, cl::NDRange(global),
                             cl::NDRange(256));
  queue.finish();
}

void SVOIntegratorCL::RenderDSMOrtho(float origin_x, float origin_y, float gsd,
                                     int width, int height, float z_min,
                                     float /*z_max*/, float voxel_size,
                                     float min_weight, float wall_cull_nz,
                                     std::vector<float>* dsm_out,
                                     std::vector<uint8_t>* ortho_out,
                                     std::vector<float>* normals_out) {
  auto& dev = opencl::CLContext::Instance().Device(device_idx_);
  auto& ctx = dev.context();
  cl::CommandQueue& queue = dev.queue();

  const size_t ncells = static_cast<size_t>(width) * height;
  const int32_t kEmptyZ = -1;

  // ---- Mesh one TSDF table's Surface Nets surface into a z-buffer. ----
  // Allocates a transient per-slot vertex buffer, runs the vertex + raster
  // passes.  Used for the fine level and each downsampled coarse level.
  auto mesh_level = [&](const cl::Buffer& table, uint32_t mask, uint32_t cap,
                        float vsize, const cl::Buffer& zbuf) {
    cl::Buffer vert(ctx, CL_MEM_READ_WRITE,
                    static_cast<size_t>(cap) * 3 * sizeof(float));
    queue.enqueueFillBuffer(zbuf, kEmptyZ, 0, ncells * sizeof(int32_t));
    // Cap a triangle's cell footprint generously above the cube pitch (1
    // voxel) to kill degenerate triangles; scales with the level's voxel.
    const int max_tri_cells = std::max(4, static_cast<int>(4.0f * vsize / gsd));
    const size_t g = ((static_cast<size_t>(cap) + 255) / 256) * 256;
    {
      cl::Kernel k(program_, "svo_dc_vertex");
      k.setArg(0, table);
      k.setArg(1, mask);
      k.setArg(2, cap);
      k.setArg(3, vert);
      k.setArg(4, vsize);
      k.setArg(5, min_weight);
      queue.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(g),
                                 cl::NDRange(256));
    }
    {
      cl::Kernel k(program_, "svo_dc_raster");
      k.setArg(0, table);
      k.setArg(1, mask);
      k.setArg(2, cap);
      k.setArg(3, vert);
      k.setArg(4, zbuf);
      k.setArg(5, origin_x);
      k.setArg(6, origin_y);
      k.setArg(7, gsd);
      k.setArg(8, width);
      k.setArg(9, height);
      k.setArg(10, z_min);
      k.setArg(11, max_tri_cells);
      k.setArg(12, wall_cull_nz);
      queue.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(g),
                                 cl::NDRange(256));
    }
  };

  // Fine level (L=0).
  cl::Buffer z_fine(ctx, CL_MEM_READ_WRITE, ncells * sizeof(int32_t));
  mesh_level(cl_table_, capacity_mask_, capacity_, voxel_size, z_fine);

  // Coarse fill levels: downsample the fine table by 2^L, mesh it, and let
  // its larger triangles cover the cells the finer levels left empty.
  const int kNumCoarse = 2;
  cl::Buffer z_coarse[2];
  cl::Buffer cl_overflow(ctx, CL_MEM_READ_WRITE, sizeof(uint32_t));
  for (int L = 1; L <= kNumCoarse; ++L) {
    // Coarse band ≈ 1/4 of the fine count per octave — size generously.
    uint32_t ccap = std::max<uint32_t>(capacity_ >> L, 1u << 16);
    uint32_t cmask = ccap - 1;
    cl::Buffer ctable(ctx, CL_MEM_READ_WRITE,
                      static_cast<size_t>(ccap) * sizeof(GPUVoxelSlot));
    const size_t cg = ((static_cast<size_t>(ccap) + 255) / 256) * 256;
    {
      cl::Kernel k(program_, "svo_clear_table");
      k.setArg(0, ctable);
      k.setArg(1, ccap);
      queue.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(cg),
                                 cl::NDRange(256));
    }
    const uint32_t zero = 0;
    queue.enqueueFillBuffer(cl_overflow, zero, 0, sizeof(uint32_t));
    {
      cl::Kernel k(program_, "svo_dc_downsample");
      k.setArg(0, cl_table_);
      k.setArg(1, capacity_);
      k.setArg(2, ctable);
      k.setArg(3, cmask);
      k.setArg(4, cl_overflow);
      k.setArg(5, L);
      const size_t fg = ((static_cast<size_t>(capacity_) + 255) / 256) * 256;
      queue.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(fg),
                                 cl::NDRange(256));
    }
    z_coarse[L - 1] =
        cl::Buffer(ctx, CL_MEM_READ_WRITE, ncells * sizeof(int32_t));
    mesh_level(ctable, cmask, ccap, voxel_size * static_cast<float>(1 << L),
               z_coarse[L - 1]);
  }

  cl::Buffer cl_dsm(ctx, CL_MEM_WRITE_ONLY, ncells * sizeof(float));
  cl::Buffer cl_ortho(ctx, CL_MEM_WRITE_ONLY, ncells * sizeof(uint32_t));
  cl::Buffer cl_normals(ctx, CL_MEM_WRITE_ONLY, ncells * 3 * sizeof(float));

  // Resolve the levels finest-first into a single z-buffer.
  cl::Buffer z_final(ctx, CL_MEM_READ_WRITE, ncells * sizeof(int32_t));
  {
    cl::Kernel k(program_, "svo_dc_resolve");
    k.setArg(0, z_fine);
    k.setArg(1, z_coarse[0]);
    k.setArg(2, z_coarse[1]);
    k.setArg(3, z_final);
    k.setArg(4, static_cast<uint32_t>(ncells));
    const size_t g = ((ncells + 255) / 256) * 256;
    queue.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(g),
                               cl::NDRange(256));
  }

  // 5x5 median despeckle (GPU).  Finalize + the Python ortho bake both read
  // this despeckled buffer, so DSM and ortho stay in sync.
  cl::Buffer z_median(ctx, CL_MEM_READ_WRITE, ncells * sizeof(int32_t));
  {
    cl::Kernel k(program_, "svo_dc_median");
    k.setArg(0, z_final);
    k.setArg(1, z_median);
    k.setArg(2, width);
    k.setArg(3, height);
    const size_t gx = ((static_cast<size_t>(width) + 15) / 16) * 16;
    const size_t gy = ((static_cast<size_t>(height) + 15) / 16) * 16;
    queue.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(gx, gy),
                               cl::NDRange(16, 16));
  }

  // Finalize: despeckled int z-buffer -> float DSM + per-cell normal.
  {
    cl::Kernel k(program_, "svo_dc_finalize");
    k.setArg(0, z_median);
    k.setArg(1, cl_dsm);
    k.setArg(2, cl_ortho);
    k.setArg(3, cl_normals);
    k.setArg(4, width);
    k.setArg(5, height);
    k.setArg(6, z_min);
    k.setArg(7, gsd);
    const size_t gx = ((static_cast<size_t>(width) + 15) / 16) * 16;
    const size_t gy = ((static_cast<size_t>(height) + 15) / 16) * 16;
    queue.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(gx, gy),
                               cl::NDRange(16, 16));
  }
  queue.finish();

  // Download results.
  dsm_out->resize(ncells);
  queue.enqueueReadBuffer(cl_dsm, CL_TRUE, 0, ncells * sizeof(float),
                          dsm_out->data());

  ortho_out->resize(ncells * 4);
  queue.enqueueReadBuffer(cl_ortho, CL_TRUE, 0, ncells * sizeof(uint32_t),
                          ortho_out->data());

  normals_out->resize(ncells * 3);
  queue.enqueueReadBuffer(cl_normals, CL_TRUE, 0, ncells * 3 * sizeof(float),
                          normals_out->data());
}

}  // namespace dense

#endif  // OPENSFM_HAVE_OPENCL
