#include <dense/svo_fuser.h>
#include <dense/svo_opencl.h>
#include <foundation/logging.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace dense {

SVOFuser::SVOFuser()
    : voxel_size_(0.02f), trunc_factor_(4.0f), min_weight_(3.0f) {}

SVOFuser::~SVOFuser() = default;

void SVOFuser::SetVoxelSize(float size) { voxel_size_ = std::max(1e-6f, size); }

void SVOFuser::SetTruncFactor(float factor) {
  trunc_factor_ = std::max(1.0f, factor);
}

void SVOFuser::SetMinWeight(float w) { min_weight_ = std::max(0.0f, w); }

void SVOFuser::SetDevice(int device_idx) { device_idx_ = device_idx; }

uint32_t SVOFuser::Capacity() const {
  return integrator_ ? integrator_->capacity() : 0u;
}

void SVOFuser::ReleaseRefineBuffers() {
  if (integrator_) {
    integrator_->ReleaseRefineBuffers();
  }
}

void SVOFuser::SetNumLevels(int n) { num_levels_ = std::max(1, n); }

void SVOFuser::SetDecimateFat(uint32_t n) { decimate_flat_ = std::max(1u, n); }

void SVOFuser::SetEdgeThreshold(float t) {
  edge_threshold_ = std::max(0.0f, std::min(1.0f, t));
}

void SVOFuser::SetMinCount(int n) { min_count_ = std::max(1, n); }

void SVOFuser::SetRelativeMinWeight(float w) {
  relative_min_weight_ = std::max(0.0f, w);
}

void SVOFuser::SetDSMWallCullNz(float nz) {
  dsm_wall_cull_nz_ = std::min(std::max(0.0f, nz), 1.0f);
}

void SVOFuser::SetBBox(const Eigen::Vector3f& min_world,
                       const Eigen::Vector3f& max_world) {
  has_bbox_ = true;
  bbox_min_world_ = min_world;
  bbox_max_world_ = max_world;
}

bool SVOFuser::IsGPUAvailable() {
  return opencl::CLContext::Instance().IsAvailable();
}

void SVOFuser::AddView(const Mat3d& K, const Mat3d& R, const Vec3d& t,
                       Eigen::Map<const ImageF> depth,
                       Eigen::Map<const PixelData3f> normal,
                       Eigen::Map<const PixelData3u8> color,
                       Eigen::Map<const ImageU8> mask,
                       Eigen::Map<const ImageF> weight,
                       const std::string& name) {
  // Borrow the caller's buffers (the maps are non-owning); see StoredView.
  views_.emplace_back(K, R, t, depth, normal, color, mask, weight, name);
}

uint32_t SVOFuser::CountVoxels() {
  if (views_.empty()) {
    return 0;
  }

  if (!opencl::CLContext::Instance().IsAvailable()) {
    throw std::runtime_error(
        "SVOFuser: GPU (OpenCL) is required but not available");
  }

  const float trunc_dist = trunc_factor_ * voxel_size_;
  const float inv_vs = 1.0f / voxel_size_;
  Eigen::Vector3i bbox_min_v, bbox_max_v;
  if (has_bbox_) {
    bbox_min_v =
        (bbox_min_world_ * inv_vs).array().floor().cast<int>().matrix();
    bbox_max_v = (bbox_max_world_ * inv_vs).array().ceil().cast<int>().matrix();
  }
  const Eigen::Vector3i* bbox_min_ptr = has_bbox_ ? &bbox_min_v : nullptr;
  const Eigen::Vector3i* bbox_max_ptr = has_bbox_ ? &bbox_max_v : nullptr;

  SVOIntegratorCL integrator(device_idx_);

  uint64_t max_pixels = 0;
  for (const auto& sv : views_) {
    uint64_t px = static_cast<uint64_t>(sv.depth.rows()) * sv.depth.cols();
    if (px > max_pixels) {
      max_pixels = px;
    }
  }
  const float spread = std::sqrt(static_cast<float>(views_.size()));
  const float voxels_per_ray = std::ceil(4.0f * trunc_factor_);
  uint64_t count_est = static_cast<uint64_t>(static_cast<float>(max_pixels) *
                                             voxels_per_ray * spread);
  count_est = std::max<uint64_t>(count_est, 1u << 20);
  uint32_t count_cap =
      static_cast<uint32_t>(std::min<uint64_t>(count_est, 1u << 28));

  integrator.InitializeCounting(count_cap);

  for (size_t i = 0; i < views_.size(); ++i) {
    const auto& sv = views_[i];
    const int rows = static_cast<int>(sv.depth.rows());
    const int cols = static_cast<int>(sv.depth.cols());
    const Mat3f Kf = sv.K.cast<float>();
    const Mat3f Rf = sv.R.cast<float>();
    const Vec3f tf = sv.t.cast<float>();
    const uint8_t* mask_ptr = (sv.mask.size() > 0) ? sv.mask.data() : nullptr;
    integrator.Count(Kf, Rf, tf, sv.depth.data(), rows, cols, mask_ptr,
                     voxel_size_, trunc_dist, bbox_min_ptr, bbox_max_ptr);
  }

  uint32_t n_unique = integrator.GetUniqueCount();
  {
    std::ostringstream oss;
    oss << "[SVOFuser] Counted " << n_unique << " unique voxels";
    foundation::LogInfo("dense", oss.str());
  }
  last_voxel_count_ = n_unique;
  return n_unique;
}

void SVOFuser::Fuse() {
  if (views_.empty()) {
    return;
  }

  if (last_voxel_count_ == 0) {
    throw std::runtime_error(
        "SVOFuser::Fuse: CountVoxels() must be called before Fuse()");
  }

  if (!opencl::CLContext::Instance().IsAvailable()) {
    throw std::runtime_error(
        "SVOFuser: GPU (OpenCL) is required but not available");
  }

  const float trunc_dist = trunc_factor_ * voxel_size_;
  const float inv_vs = 1.0f / voxel_size_;
  Eigen::Vector3i bbox_min_v, bbox_max_v;
  if (has_bbox_) {
    bbox_min_v =
        (bbox_min_world_ * inv_vs).array().floor().cast<int>().matrix();
    bbox_max_v = (bbox_max_world_ * inv_vs).array().ceil().cast<int>().matrix();
  }
  const Eigen::Vector3i* bbox_min_ptr = has_bbox_ ? &bbox_min_v : nullptr;
  const Eigen::Vector3i* bbox_max_ptr = has_bbox_ ? &bbox_max_v : nullptr;

  uint32_t capacity = static_cast<uint32_t>(
      std::min<uint64_t>(static_cast<uint64_t>(last_voxel_count_) * 2,
                         std::numeric_limits<uint32_t>::max()));
  capacity = std::max<uint32_t>(capacity, 1u << 20);
  {
    std::ostringstream oss;
    oss << "[SVOFuser] Voxel count " << last_voxel_count_ << " -> capacity "
        << capacity << " (" << float(capacity) / last_voxel_count_ * 100
        << "%)";
    foundation::LogInfo("dense", oss.str());
  }

  integrator_ = std::make_unique<SVOIntegratorCL>(device_idx_);
  integrator_->Initialize(capacity);

  for (size_t i = 0; i < views_.size(); ++i) {
    const auto& sv = views_[i];
    const int rows = static_cast<int>(sv.depth.rows());
    const int cols = static_cast<int>(sv.depth.cols());
    const Mat3f Kf = sv.K.cast<float>();
    const Mat3f Rf = sv.R.cast<float>();
    const Vec3f tf = sv.t.cast<float>();
    const float* normal_ptr =
        (sv.normal.size() > 0) ? sv.normal.data() : nullptr;
    const uint8_t* color_ptr =
        (sv.color.size() > 0) ? sv.color.data() : nullptr;
    const uint8_t* mask_ptr = (sv.mask.size() > 0) ? sv.mask.data() : nullptr;
    const float* weight_ptr =
        (sv.weight.size() > 0) ? sv.weight.data() : nullptr;

    integrator_->Integrate(Kf, Rf, tf, sv.depth.data(), rows, cols, normal_ptr,
                           color_ptr, mask_ptr, weight_ptr, voxel_size_,
                           trunc_dist, bbox_min_ptr, bbox_max_ptr, nullptr,
                           0.0f);
  }

  // Check for overflow (dropped contributions due to hash table exhaustion).
  const uint32_t overflow = integrator_->GetOverflowCount();
  if (overflow > 0) {
    {
      std::ostringstream oss;
      oss << "[SVOFuser] WARNING: integration dropped " << overflow
          << " contributions (hash table overflow, capacity="
          << integrator_->capacity() << ")";
      foundation::LogWarning("dense", oss.str());
    }
  }

  foundation::LogInfo("dense",
                      "[SVOFuser] Fuse complete, hash table alive on GPU");
}

void SVOFuser::RefineGeometry(
    int iters, float lambda_reg,
    const std::map<std::string, std::vector<std::string>>& neighbors,
    float lambda_anchor, float early_stop_rel) {
  if (!integrator_) {
    throw std::runtime_error(
        "SVOFuser::RefineGeometry: Fuse() must be called before "
        "RefineGeometry()");
  }
  if (views_.empty()) {
    return;
  }

  if (neighbors.empty()) {
    throw std::runtime_error(
        "SVOFuser::RefineGeometry: neighbor view graph is required for "
        "RefineGeometry()");
  }

  const float trunc_dist = trunc_factor_ * voxel_size_;
  const int n_views = static_cast<int>(views_.size());

  // All views must have the same resolution.
  const int h0 = static_cast<int>(views_[0].depth.rows());
  const int w0 = static_cast<int>(views_[0].depth.cols());

  // Build packed camera array.
  std::vector<SVOCameraGPU> cameras(n_views);
  for (int i = 0; i < n_views; ++i) {
    const auto& sv = views_[i];
    const Mat3f Kf = sv.K.cast<float>();
    const Mat3f Rf = sv.R.cast<float>();
    const Vec3f tf = sv.t.cast<float>();
    const Mat3f Kinv = Kf.inverse();
    const Mat3f Rinv = Rf.transpose();
    const Vec3f cam_pos = -Rinv * tf;

    auto to_row_major = [](const Mat3f& M, float* out) {
      const Mat3f Mt = M.transpose();
      std::memcpy(out, Mt.data(), 9 * sizeof(float));
    };

    SVOCameraGPU& cam = cameras[i];
    to_row_major(Kinv, cam.Kinv);
    to_row_major(Rinv, cam.Rinv);
    to_row_major(Rf, cam.R);
    cam.t[0] = tf.x();
    cam.t[1] = tf.y();
    cam.t[2] = tf.z();
    cam.cam_pos[0] = cam_pos.x();
    cam.cam_pos[1] = cam_pos.y();
    cam.cam_pos[2] = cam_pos.z();
    cam._pad[0] = cam._pad[1] = cam._pad[2] = 0.0f;
  }

  // Borrow per-view color/mask/depth pointers; the integrator streams each
  // slice straight to the GPU, with no full per-cluster packed copy on the
  // host.  view_cache (and thus these maps) outlives the call.
  std::vector<RefineViewSrc> srcs(n_views);
  for (int i = 0; i < n_views; ++i) {
    const auto& sv = views_[i];
    srcs[i].color = (sv.color.size() > 0) ? sv.color.data() : nullptr;
    srcs[i].mask = (sv.mask.size() > 0) ? sv.mask.data() : nullptr;
    srcs[i].depth = sv.depth.data();
  }

  integrator_->PrepareRefinement(cameras, srcs, w0, h0, n_views);

  constexpr int kMaxRefineNeighbors = 4;
  std::vector<int32_t> neighbor_data;

  std::unordered_map<std::string, int> name_to_idx;
  name_to_idx.reserve(n_views * 2);
  for (int i = 0; i < n_views; ++i) {
    name_to_idx[views_[i].name] = i;
  }

  neighbor_data.assign(static_cast<size_t>(n_views) * kMaxRefineNeighbors, -1);
  int n_with_nbrs = 0;
  for (int i = 0; i < n_views; ++i) {
    auto it = neighbors.find(views_[i].name);
    if (it == neighbors.end()) {
      continue;
    }
    int k = 0;
    for (const std::string& nb_name : it->second) {
      if (k >= kMaxRefineNeighbors) {
        break;
      }
      if (nb_name == views_[i].name) {
        continue;  // skip self (best_neighbors[0] is the shot itself)
      }
      auto jt = name_to_idx.find(nb_name);
      if (jt == name_to_idx.end()) {
        continue;  // neighbor not loaded in this sub-volume
      }
      neighbor_data[static_cast<size_t>(i) * kMaxRefineNeighbors + k] =
          jt->second;
      ++k;
    }
    if (k > 0) {
      ++n_with_nbrs;
    }
  }

  std::ostringstream oss;
  oss << "[SVOFuser] RefineGeometry using SfM co-visibility neighbors: "
      << n_with_nbrs << "/" << n_views << " views have ≥1 neighbor in-volume";
  foundation::LogInfo("dense", oss.str());

  integrator_->RefineGeometry(iters, lambda_reg, voxel_size_, trunc_dist,
                              min_weight_, neighbor_data, kMaxRefineNeighbors,
                              lambda_anchor, early_stop_rel);

  {
    std::ostringstream oss;
    oss << "[SVOFuser] RefineGeometry complete (" << iters << " iterations)";
    foundation::LogInfo("dense", oss.str());
  }
}

void SVOFuser::BakeColors(std::vector<Vec3f>& points,
                          std::vector<Vec3f>& normals,
                          std::vector<Vec3<uint8_t>>* colors, int n_final,
                          int irls_iters,
                          const std::vector<uint8_t>* relax_occ,
                          const std::vector<float>* dsm_occ, int dsm_w,
                          int dsm_h, float dsm_origin_x, float dsm_origin_y,
                          float dsm_gsd, float dsm_max_z,
                          std::vector<uint8_t>* out_sharp) {
  if (!integrator_) {
    throw std::runtime_error(
        "SVOFuser::BakeColors: Fuse() must be called first");
  }
  if (views_.empty() || points.empty()) {
    return;
  }

  // If PrepareRefinement wasn't called (refine was skipped), upload images now.
  // We check by trying to call BakeColors — it will throw if not prepared.
  // Instead, we always ensure preparation here.
  const int n_views = static_cast<int>(views_.size());
  const int h0 = static_cast<int>(views_[0].depth.rows());
  const int w0 = static_cast<int>(views_[0].depth.cols());

  // Build camera array.
  std::vector<SVOCameraGPU> cameras(n_views);
  for (int i = 0; i < n_views; ++i) {
    const auto& sv = views_[i];
    const Mat3f Kf = sv.K.cast<float>();
    const Mat3f Rf = sv.R.cast<float>();
    const Vec3f tf = sv.t.cast<float>();
    const Mat3f Kinv = Kf.inverse();
    const Mat3f Rinv = Rf.transpose();
    const Vec3f cam_pos = -Rinv * tf;

    auto to_row_major = [](const Mat3f& M, float* out) {
      const Mat3f Mt = M.transpose();
      std::memcpy(out, Mt.data(), 9 * sizeof(float));
    };

    SVOCameraGPU& cam = cameras[i];
    to_row_major(Kinv, cam.Kinv);
    to_row_major(Rinv, cam.Rinv);
    to_row_major(Rf, cam.R);
    cam.t[0] = tf.x();
    cam.t[1] = tf.y();
    cam.t[2] = tf.z();
    cam.cam_pos[0] = cam_pos.x();
    cam.cam_pos[1] = cam_pos.y();
    cam.cam_pos[2] = cam_pos.z();
    cam._pad[0] = cam._pad[1] = cam._pad[2] = 0.0f;
  }

  // Borrow per-view color/mask/depth pointers (streamed per-slice in
  // PrepareRefinement) — no full per-cluster packed host copy.
  std::vector<RefineViewSrc> srcs(n_views);
  for (int i = 0; i < n_views; ++i) {
    const auto& sv = views_[i];
    srcs[i].color = (sv.color.size() > 0) ? sv.color.data() : nullptr;
    srcs[i].mask = (sv.mask.size() > 0) ? sv.mask.data() : nullptr;
    srcs[i].depth = sv.depth.data();
  }

  integrator_->PrepareRefinement(cameras, srcs, w0, h0, n_views);
  integrator_->BakeColors(points, normals, colors, n_final, irls_iters,
                          relax_occ, dsm_occ, dsm_w, dsm_h, dsm_origin_x,
                          dsm_origin_y, dsm_gsd, dsm_max_z, out_sharp);
}

void SVOFuser::PruneByVisibility(int iterations, float carve_margin,
                                 int carve_threshold, int support_min) {
  if (!integrator_) {
    throw std::runtime_error(
        "SVOFuser::PruneByVisibility: Fuse() must be called first");
  }
  if (views_.empty()) {
    return;
  }

  integrator_->InitializeVisibilityPruning();

  const float trunc_dist = voxel_size_ * trunc_factor_;
  // Weight penalty per excess carve vote: kill a voxel that was integrated
  // with min_weight_ if it gets carve_threshold + 1 excess votes.
  const float weight_penalty = min_weight_ / 2.0f;

  for (int iter = 0; iter < iterations; ++iter) {
    if (iter > 0) {
      integrator_->ClearVotes();
    }

    for (auto& view : views_) {
      const int rows = static_cast<int>(view.depth.rows());
      const int cols = static_cast<int>(view.depth.cols());
      if (rows == 0 || cols == 0) {
        continue;
      }

      // Compute depth range for this view.
      float min_depth = std::numeric_limits<float>::max();
      float max_depth = 0.0f;
      const float* dptr = view.depth.data();
      for (int i = 0; i < rows * cols; ++i) {
        if (dptr[i] > 0.0f) {
          min_depth = std::min(min_depth, dptr[i]);
          max_depth = std::max(max_depth, dptr[i]);
        }
      }
      if (max_depth <= 0.0f) {
        continue;
      }
      // Extend range by truncation band.
      min_depth = std::max(0.01f, min_depth - trunc_dist);
      max_depth += trunc_dist;

      // Convert camera from double to float.
      Mat3f Kf = view.K.cast<float>();
      Mat3f Rf = view.R.cast<float>();
      Vec3f tf = view.t.cast<float>();

      integrator_->RaycastAndVote(Kf, Rf, tf, view.depth.data(), rows, cols,
                                  voxel_size_, min_depth, max_depth,
                                  min_weight_, carve_margin);
    }

    integrator_->Prune(carve_threshold, support_min, weight_penalty);
    {
      std::ostringstream oss;
      oss << "[SVOFuser] Visibility prune iteration " << (iter + 1) << "/"
          << iterations << " complete";
      foundation::LogInfo("dense", oss.str());
    }
  }
}

void SVOFuser::ExtractPoints(std::vector<Vec3f>* fused_points,
                             std::vector<Vec3f>* fused_normals,
                             std::vector<Vec3<uint8_t>>* fused_colors) {
  if (!integrator_) {
    throw std::runtime_error(
        "SVOFuser::ExtractPoints: Fuse() must be called first");
  }

  // Phase 1: Extract fine level (L=0).
  integrator_->ExtractPoints(min_weight_, voxel_size_, decimate_flat_,
                             edge_threshold_, min_count_, relative_min_weight_,
                             fused_points, fused_normals, fused_colors);
  {
    std::ostringstream oss;
    oss << "[SVOFuser] Fine (L=0): " << fused_points->size()
        << " surface points";
    foundation::LogInfo("dense", oss.str());
  }

  // Phase 2: Integrate all coarse levels with coverage check, then extract.
  if (num_levels_ > 1 && !views_.empty()) {
    const float fine_voxel_size = voxel_size_;
    const cl::Buffer& fine_table = integrator_->table_buffer();
    const uint32_t fine_mask = integrator_->capacity_mask();

    // Keep all coarse integrators alive for cross-level reference checks.
    std::vector<std::unique_ptr<SVOIntegratorCL>> coarse_integrators;

    for (int L = 1; L < num_levels_; ++L) {
      const int level_shift = L;
      const float coarse_vs = fine_voxel_size * static_cast<float>(1 << L);
      const float coarse_trunc = trunc_factor_ * coarse_vs;

      {
        std::ostringstream oss;
        oss << "[SVOFuser] Coarse L=" << L << " voxel_size=" << coarse_vs
            << " trunc=" << coarse_trunc;
        foundation::LogInfo("dense", oss.str());
      }

      auto coarse_integrator = std::make_unique<SVOIntegratorCL>(device_idx_);

      const float coarse_inv_vs = 1.0f / coarse_vs;
      Eigen::Vector3i bbox_min_v, bbox_max_v;
      if (has_bbox_) {
        bbox_min_v = (bbox_min_world_ * coarse_inv_vs)
                         .array()
                         .floor()
                         .cast<int>()
                         .matrix();
        bbox_max_v = (bbox_max_world_ * coarse_inv_vs)
                         .array()
                         .ceil()
                         .cast<int>()
                         .matrix();
      }
      const Eigen::Vector3i* bbox_min_ptr = has_bbox_ ? &bbox_min_v : nullptr;
      const Eigen::Vector3i* bbox_max_ptr = has_bbox_ ? &bbox_max_v : nullptr;

      // Count pass for coarse level.
      uint32_t coarse_count_cap = static_cast<uint32_t>(
          std::min<uint64_t>(static_cast<uint64_t>(last_voxel_count_) /
                                     static_cast<uint64_t>(1 << (3 * L)) * 4 +
                                 (1u << 20),
                             std::numeric_limits<uint32_t>::max()));
      coarse_count_cap = std::max<uint32_t>(coarse_count_cap, 1u << 20);
      coarse_integrator->InitializeCounting(coarse_count_cap);

      for (size_t vi = 0; vi < views_.size(); ++vi) {
        const auto& sv = views_[vi];
        const int rows = static_cast<int>(sv.depth.rows());
        const int cols = static_cast<int>(sv.depth.cols());
        const Mat3f Kf = sv.K.cast<float>();
        const Mat3f Rf = sv.R.cast<float>();
        const Vec3f tf = sv.t.cast<float>();
        const uint8_t* mask_ptr =
            (sv.mask.size() > 0) ? sv.mask.data() : nullptr;

        coarse_integrator->Count(Kf, Rf, tf, sv.depth.data(), rows, cols,
                                 mask_ptr, coarse_vs, coarse_trunc,
                                 bbox_min_ptr, bbox_max_ptr);
      }

      uint32_t coarse_voxels = coarse_integrator->GetUniqueCount();
      {
        std::ostringstream oss;
        oss << "[SVOFuser] Coarse L=" << L << " counted " << coarse_voxels
            << " voxels";
        foundation::LogInfo("dense", oss.str());
      }

      if (coarse_voxels == 0) {
        continue;
      }

      // Allocate and integrate coarse level with reference check.
      uint32_t coarse_capacity = static_cast<uint32_t>(
          std::min<uint64_t>(static_cast<uint64_t>(coarse_voxels) * 2,
                             std::numeric_limits<uint32_t>::max()));
      coarse_capacity = std::max<uint32_t>(coarse_capacity, 1u << 20);

      coarse_integrator->Initialize(coarse_capacity);

      // Build reference table list: fine (L=0) + all previous coarse levels.
      std::vector<SVOIntegratorCL::RefTableInfo> ref_tables;
      ref_tables.push_back({fine_table, fine_mask, 1.0f / fine_voxel_size});
      for (size_t prev = 0; prev < coarse_integrators.size(); ++prev) {
        auto& prev_int = coarse_integrators[prev];
        float prev_vs = fine_voxel_size * static_cast<float>(1 << (prev + 1));
        ref_tables.push_back({prev_int->table_buffer(),
                              prev_int->capacity_mask(), 1.0f / prev_vs});
      }

      for (size_t vi = 0; vi < views_.size(); ++vi) {
        const auto& sv = views_[vi];
        const int rows = static_cast<int>(sv.depth.rows());
        const int cols = static_cast<int>(sv.depth.cols());
        const Mat3f Kf = sv.K.cast<float>();
        const Mat3f Rf = sv.R.cast<float>();
        const Vec3f tf = sv.t.cast<float>();
        const float* normal_ptr =
            (sv.normal.size() > 0) ? sv.normal.data() : nullptr;
        const uint8_t* color_ptr =
            (sv.color.size() > 0) ? sv.color.data() : nullptr;
        const uint8_t* mask_ptr =
            (sv.mask.size() > 0) ? sv.mask.data() : nullptr;
        const float* weight_ptr =
            (sv.weight.size() > 0) ? sv.weight.data() : nullptr;

        coarse_integrator->Integrate(
            Kf, Rf, tf, sv.depth.data(), rows, cols, normal_ptr, color_ptr,
            mask_ptr, weight_ptr, coarse_vs, coarse_trunc, bbox_min_ptr,
            bbox_max_ptr, &ref_tables, min_weight_);
      }

      // Keep this integrator alive for future levels to reference.
      coarse_integrators.push_back(std::move(coarse_integrator));
    }

    // Extract fill from each coarse level against fine table.
    for (int L = 1; L <= static_cast<int>(coarse_integrators.size()); ++L) {
      auto& coarse_int = coarse_integrators[L - 1];
      const float coarse_vs = fine_voxel_size * static_cast<float>(1 << L);
      const float coarse_min_weight = min_weight_ * static_cast<float>(1 << L);

      std::vector<Vec3f> fill_pts, fill_nrm;
      std::vector<Vec3<uint8_t>> fill_clr;

      coarse_int->ExtractFill(fine_table, fine_mask, coarse_min_weight,
                              coarse_vs, fine_voxel_size, L, &fill_pts,
                              &fill_nrm, &fill_clr);

      {
        std::ostringstream oss;
        oss << "[SVOFuser] Coarse L=" << L << " fill: " << fill_pts.size()
            << " points";
        foundation::LogInfo("dense", oss.str());
      }

      if (!fill_pts.empty()) {
        fused_points->insert(fused_points->end(), fill_pts.begin(),
                             fill_pts.end());
        fused_normals->insert(fused_normals->end(), fill_nrm.begin(),
                              fill_nrm.end());
        fused_colors->insert(fused_colors->end(), fill_clr.begin(),
                             fill_clr.end());
      }
    }

    // Release all coarse integrators.
    coarse_integrators.clear();
  }

  {
    std::ostringstream oss;
    oss << "[SVOFuser] Total ExtractPoints: " << fused_points->size()
        << " surface points (fine + fill)";
    foundation::LogInfo("dense", oss.str());
  }
}

// Legacy API: Fuse + ExtractPoints in one call.
void SVOFuser::Fuse(std::vector<Vec3f>* fused_points,
                    std::vector<Vec3f>* fused_normals,
                    std::vector<Vec3<uint8_t>>* fused_colors) {
  fused_points->clear();
  fused_normals->clear();
  fused_colors->clear();

  if (views_.empty()) {
    return;
  }

  Fuse();
  ExtractPoints(fused_points, fused_normals, fused_colors);
}

void SVOFuser::RenderDSMOrtho(float origin_x, float origin_y, float gsd,
                              int width, int height, float z_min, float z_max,
                              std::vector<float>* dsm_out,
                              std::vector<uint8_t>* ortho_out,
                              std::vector<float>* normals_out) {
  if (!integrator_) {
    throw std::runtime_error(
        "SVOFuser::RenderDSMOrtho: must call Fuse() first");
  }
  integrator_->RenderDSMOrtho(
      origin_x, origin_y, gsd, width, height, z_min, z_max, voxel_size_,
      min_weight_, dsm_wall_cull_nz_, dsm_out, ortho_out, normals_out);
}

void SVOFuser::ExtractMesh(std::vector<Vec3f>* verts,
                           std::vector<Vec3f>* normals, std::vector<int>* tris) {
  if (!integrator_) {
    throw std::runtime_error("SVOFuser::ExtractMesh: must call Fuse() first");
  }
  integrator_->ExtractMesh(min_weight_, voxel_size_, verts, normals, tris);
}

}  // namespace dense
