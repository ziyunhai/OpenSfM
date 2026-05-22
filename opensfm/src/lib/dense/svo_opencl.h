#pragma once

#ifdef OPENSFM_HAVE_OPENCL

#include <dense/opencl_utils.h>
#include <dense/svo.h>
#include <foundation/types.h>

#include <cstdint>

namespace dense {

// Must match kernel-side VoxelSlot exactly (48 bytes).
struct GPUVoxelSlot {
  uint32_t key_ab;   // packed (x+32768)<<16 | (y+32768), EMPTY = 0xFFFFFFFF
  int32_t key_c;     // z coordinate, UNINIT = 0x80000000
  int32_t count;     // observation count
  int32_t sum_tsdf;  // fixed-point TSDF accumulator
  int32_t sum_nx;    // fixed-point normal.x accumulator
  int32_t sum_ny;
  int32_t sum_nz;
  int32_t sum_r;  // color accumulator
  int32_t sum_g;
  int32_t sum_b;
  int32_t sum_weight;  // accumulated confidence weight (WEIGHT_SCALE units)
  int32_t _pad;
};
static_assert(sizeof(GPUVoxelSlot) == 48, "GPUVoxelSlot must be 48 bytes");

static constexpr int32_t kKeyCUninit = static_cast<int32_t>(0x80000000);

// Must match kernel-side CountSlot exactly (8 bytes).
struct GPUCountSlot {
  uint32_t key_ab;
  int32_t key_c;
};
static_assert(sizeof(GPUCountSlot) == 8, "GPUCountSlot must be 8 bytes");

// Must match kernel-side SVOCamera exactly (144 bytes).
// All matrices stored in row-major order.
struct SVOCameraGPU {
  float Kinv[9];
  float Rinv[9];
  float R[9];
  float t[3];
  float cam_pos[3];
  float _pad[3];
};
static_assert(sizeof(SVOCameraGPU) == 144, "SVOCameraGPU must be 144 bytes");

static constexpr int kFPScale = 32768;
static constexpr uint32_t kEmptyKey = 0xFFFFFFFFu;

// GPU-accelerated TSDF integration into an open-addressing hash table.
//
// Usage:
//   SVOIntegratorCL integrator(device_idx);
//   integrator.Initialize(capacity);
//   for (each view) integrator.Integrate(...);
//   VoxelMap voxels = integrator.Download();
class SVOIntegratorCL {
 public:
  explicit SVOIntegratorCL(int device_idx = 0);

  // Allocate the GPU hash table.  |capacity| is rounded up to next power of 2.
  void Initialize(uint32_t capacity);

  // Integrate a single depthmap into the hash table on device.
  // Optional bbox_min/bbox_max clip integration to a sub-volume
  // (voxel integer coordinates).
  void Integrate(const Mat3f& K, const Mat3f& R, const Vec3f& t,
                 const float* depth, int rows, int cols, const float* normal,
                 const uint8_t* color, const uint8_t* mask, const float* weight,
                 float voxel_size, float trunc_dist,
                 const Eigen::Vector3i* bbox_min = nullptr,
                 const Eigen::Vector3i* bbox_max = nullptr);

  // Download the GPU hash table and convert to a CPU-side VoxelMap.
  VoxelMap Download() const;

  // Extract surface points directly on GPU — finds TSDF zero-crossings
  // in the hash table and interpolates position/normal/color.
  // Much faster than Download() + SVO::ExtractPoints() because only
  // the surface points (~5-10% of voxels) are transferred back.
  void ExtractPoints(float min_weight, float voxel_size,
                     std::vector<Vec3f>* points, std::vector<Vec3f>* normals,
                     std::vector<Vec3<uint8_t>>* colors);

  // Multi-level fill extraction: sub-sample coarse TSDF crossings at
  // fine-level density, emitting only where the fine table has no coverage.
  // |fine_table| and |fine_mask| come from a finer-level integrator.
  void ExtractFill(const cl::Buffer& fine_table, uint32_t fine_mask,
                   float min_weight, float coarse_voxel_size,
                   float fine_voxel_size, int level_shift,
                   std::vector<Vec3f>* points, std::vector<Vec3f>* normals,
                   std::vector<Vec3<uint8_t>>* colors);

  // Public accessors for the hash table buffer (needed for multi-level fill).
  const cl::Buffer& table_buffer() const { return cl_table_; }
  uint32_t capacity_mask() const { return capacity_mask_; }

  // --- Counting pass (lightweight dry-run) ---
  // Allocate a compact counting table (12 bytes/slot) to count unique
  // voxels across all views without accumulating TSDF / color data.
  void InitializeCounting(uint32_t capacity);

  // Run a count-only kernel for one depthmap (same ray-march, key-only
  // insertion into the counting table).
  // Optional bbox_min/bbox_max clip counting to a sub-volume.
  void Count(const Mat3f& K, const Mat3f& R, const Vec3f& t, const float* depth,
             int rows, int cols, const uint8_t* mask, float voxel_size,
             float trunc_dist, const Eigen::Vector3i* bbox_min = nullptr,
             const Eigen::Vector3i* bbox_max = nullptr);

  // Read back the unique voxel count from the GPU counter.
  uint32_t GetUniqueCount() const;

  // Read back the overflow (dropped contribution) count from the GPU counter.
  // Returns 0 if no overflow occurred (deterministic integration).
  uint32_t GetOverflowCount() const;

  // Reset the overflow counter to zero (call before each Integrate batch).
  void ResetOverflowCounter();

  uint32_t capacity() const { return capacity_; }

  // --- Photometric refinement ---
  // Per-view image descriptor for the packed color image buffer.
  struct ImageDesc {
    int32_t width;
    int32_t height;
    int32_t offset;  // byte offset into packed color buffer
    int32_t _pad;
  };
  static_assert(sizeof(ImageDesc) == 16, "ImageDesc must be 16 bytes");

  // Upload all view color images and cameras to GPU for refinement.
  // Must be called after Initialize() + Integrate() (hash table populated).
  void PrepareRefinement(const std::vector<SVOCameraGPU>& cameras,
                         const std::vector<uint8_t>& packed_colors,
                         const std::vector<float>& packed_depths,
                         const std::vector<ImageDesc>& image_descs,
                         int n_views);

  // Run photometric refinement on the GPU hash table in-place.
  // Phase 1: color-only (color_iters), Phase 2: joint SDF+color (joint_iters).
  // lambda_reg: initial regularization weight, decayed by lambda_decay per
  // iter.
  void Refine(int color_iters, int joint_iters, float lambda_reg,
              float lambda_decay, float voxel_size, float trunc_dist,
              float min_weight);

  // --- Visibility pruning ---
  // Initialize carve/support vote buffers (same capacity as hash table).
  void InitializeVisibilityPruning();

  // Raycast the hash table from a given camera, then compare with a clean
  // depth map to tally carve/support votes per voxel.
  void RaycastAndVote(const Mat3f& K, const Mat3f& R, const Vec3f& t,
                      const float* clean_depth, int rows, int cols,
                      float voxel_size, float min_depth, float max_depth,
                      float min_weight, float carve_margin);

  // Apply pruning based on accumulated votes.
  void Prune(int carve_threshold, int support_min,
             float weight_penalty_per_vote);

  // Clear vote counters (call between iterations if doing multiple passes).
  void ClearVotes();

 private:
  void BuildKernels();
  void EnsureFrameBuffers(int rows, int cols, bool has_normal, bool has_color,
                          bool has_mask, bool has_weight);

  int device_idx_;
  uint32_t capacity_ = 0;
  uint32_t capacity_mask_ = 0;

  cl::Program program_;
  cl::Kernel k_clear_;
  cl::Kernel k_integrate_;
  bool kernels_built_ = false;

  cl::Buffer cl_table_;
  cl::Buffer cl_camera_;
  cl::Buffer cl_depth_;
  cl::Buffer cl_normal_;
  cl::Buffer cl_color_;
  cl::Buffer cl_mask_;
  cl::Buffer cl_weight_;
  cl::Buffer cl_dummy_;     // 1-byte placeholder for null pointers
  cl::Buffer cl_overflow_;  // uint32 overflow counter (integration)

  // Counting pass buffers.
  cl::Kernel k_count_clear_;
  cl::Kernel k_count_;
  cl::Buffer cl_count_table_;
  cl::Buffer cl_counter_;
  cl::Buffer cl_count_overflow_;  // uint32 overflow counter (counting)
  uint32_t count_capacity_ = 0;
  uint32_t count_mask_ = 0;

  // Extraction kernel.
  cl::Kernel k_extract_;
  cl::Kernel k_extract_fill_;

  // Refinement kernels and buffers.
  cl::Kernel k_refine_clear_;
  cl::Kernel k_refine_accumulate_;
  cl::Kernel k_refine_update_;
  cl::Buffer cl_refine_grad_;  // 4 floats/slot: grad_d, grad_r, grad_g, grad_b
  cl::Buffer
      cl_refine_adam_;  // 8 floats/slot: m_d, v_d, m_r, v_r, m_g, v_g, m_b, v_b
  cl::Buffer cl_color_images_;   // packed RGB for all views
  cl::Buffer cl_depth_images_;   // packed depth maps for all views (visibility)
  cl::Buffer cl_cameras_array_;  // N × SVOCameraGPU
  cl::Buffer cl_image_descs_;    // N × ImageDesc
  int n_refine_views_ = 0;
  bool refine_prepared_ = false;

  size_t depth_bytes_ = 0;
  size_t normal_bytes_ = 0;
  size_t color_bytes_ = 0;
  size_t mask_bytes_ = 0;
  size_t weight_bytes_ = 0;

  // Visibility pruning kernels and buffers.
  cl::Kernel k_raycast_;
  cl::Kernel k_carve_vote_;
  cl::Kernel k_prune_;
  cl::Kernel k_clear_votes_;
  cl::Buffer cl_rendered_depth_;
  cl::Buffer cl_hit_slot_;
  cl::Buffer cl_clean_depth_;
  cl::Buffer cl_carve_count_;
  cl::Buffer cl_support_count_;
  size_t raycast_pixels_ = 0;
  bool visibility_initialized_ = false;
};

}  // namespace dense

#endif  // OPENSFM_HAVE_OPENCL
