#pragma once

#include <dense/fuser.h>  // reuse ImageF, PixelData3f, etc.

#include <memory>
#include <string>
#include <vector>

namespace dense {

// Forward-declare so we don't pull in OpenCL headers here.
class SVOIntegratorCL;

// SVOFuser: TSDF-based depthmap fusion on GPU via OpenCL.
//
// Usage: AddView() → CountVoxels() → Fuse() → [Refine()] → ExtractPoints().
// CountVoxels() must be called before Fuse() so the caller can check
// the voxel budget and split if necessary.
// After Fuse(), the hash table remains on GPU. Optionally call Refine()
// for photometric refinement, then ExtractPoints() to get the result.

class SVOFuser {
 public:
  SVOFuser();
  ~SVOFuser();

  // ------- Parameters -------
  void SetVoxelSize(float size);
  void SetTruncFactor(float factor);
  void SetMinWeight(float w);
  void SetDevice(int device_idx);
  void SetBBox(const Eigen::Vector3f& min_world,
               const Eigen::Vector3f& max_world);
  static bool IsGPUAvailable();

  // ------- Data -------
  void AddView(const Mat3d& K, const Mat3d& R, const Vec3d& t,
               const ImageF& depth, const PixelData3f& normal,
               const PixelData3u8& color, const ImageU8& mask,
               const ImageF& weight = ImageF(), const std::string& name = "");

  // ------- Run -------
  // Count unique voxels via a GPU counting pass.
  // Must be called before Fuse().
  uint32_t CountVoxels();

  // Integrate all views into the GPU hash table.
  // The hash table remains alive for Refine() and ExtractPoints().
  // Throws if CountVoxels() was not called first.
  void Fuse();

  // Photometric refinement on the GPU hash table (in-place).
  // Phase 1: color-only (color_iters), Phase 2: joint SDF+color (joint_iters).
  // Must be called after Fuse().
  void Refine(int color_iters, int joint_iters, float lambda_reg);

  // Visibility-based pruning of the TSDF hash table.
  // Raycasts the hash table from each integrated view, compares with its
  // clean depth map, and prunes voxels with too many carve votes.
  // Must be called after Fuse().
  // Parameters:
  //   iterations: number of raycast-vote-prune passes (typically 1-2)
  //   carve_margin: relative depth margin for carve votes (e.g. 0.05)
  //   carve_threshold: min carve votes to trigger pruning
  //   support_min: min support votes to be safe from pruning
  void PruneByVisibility(int iterations, float carve_margin,
                         int carve_threshold, int support_min);

  // Extract surface points from the (possibly refined) hash table.
  // Returns [points, normals, colors].
  void ExtractPoints(std::vector<Vec3f>* fused_points,
                     std::vector<Vec3f>* fused_normals,
                     std::vector<Vec3<uint8_t>>* fused_colors);

  // Legacy API: Fuse + ExtractPoints in one call (backward compat).
  void Fuse(std::vector<Vec3f>* fused_points, std::vector<Vec3f>* fused_normals,
            std::vector<Vec3<uint8_t>>* fused_colors);

 private:
  // Stored views (we keep the Eigen data alive in these buffers).
  struct StoredView {
    Mat3d K, R;
    Vec3d t;
    ImageF depth;
    PixelData3f normal;
    PixelData3u8 color;
    ImageU8 mask;
    ImageF weight;
    std::string name;
  };
  std::vector<StoredView> views_;

  float voxel_size_;
  float trunc_factor_;
  float min_weight_;
  int device_idx_ = 0;             // OpenCL device index
  uint32_t last_voxel_count_ = 0;  // cached from CountVoxels()
  bool has_bbox_ = false;
  Eigen::Vector3f bbox_min_world_ = Eigen::Vector3f::Zero();
  Eigen::Vector3f bbox_max_world_ = Eigen::Vector3f::Zero();

  // GPU integrator kept alive between Fuse() and Refine()/ExtractPoints().
  std::unique_ptr<SVOIntegratorCL> integrator_;
};

}  // namespace dense
