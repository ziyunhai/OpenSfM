#pragma once

// This file contains the OpenCL kernel source for PatchMatch, embedded as a C++
// raw string.  It is #included by acmmp.cc.
//
// The kernels implement the core PatchMatch stereo algorithm adapted from:
//   Xu et al., "Multi-Scale Geometric Consistency Guided and Planar Prior
//   Assisted Multi-View Stereo", TPAMI 2022.
//
// Major differences from the original CUDA code:
//   - Uses OpenCL 1.2 instead of CUDA.
//   - Uses a Philox-style PRNG instead of curand.
//   - Texture sampling uses read_imagef (CL_HALF_FLOAT storage,
//   hardware-converted).

#ifdef OPENSFM_HAVE_OPENCL

namespace dense {

inline constexpr int kMaxSources = 8;

inline const char* kPatchMatchKernelSource =
    R"CL(

// Maximum number of source images the kernels support.
#define MAX_SOURCES 8


// Nearest-neighbor sampler for reading planes/costs from image2d_t.
// Used to leverage the texture cache for scattered neighbor reads.
__constant sampler_t samp_planes = CLK_NORMALIZED_COORDS_FALSE
                                 | CLK_ADDRESS_CLAMP_TO_EDGE
                                 | CLK_FILTER_NEAREST;

// =====================================================================
// Camera struct – matches host-side layout.
// K is row-major 3x3, R is row-major 3x3, t is 3-vector.
// =====================================================================
typedef struct {
  float K[9];
  float R[9];
  float t[3];
  int width;
  int height;
} Camera;

// =====================================================================
// Plane hypothesis: (nx, ny, nz, depth)
// =====================================================================
typedef float4 PlaneHypothesis;

// =====================================================================
// Philox-2x32-10 PRNG (stateless, counter-based).
// =====================================================================
uint2 philox2x32_round(uint2 ctr, uint key) {
  uint lo = ctr.x;
  uint hi = ctr.y;
  // Philox S-box constant
  uint product_lo = lo * 0xD256D193u;
  uint product_hi = mul_hi(lo, 0xD256D193u);
  ctr.x = hi ^ key ^ product_hi;
  ctr.y = product_lo;
  return ctr;
}

float rand_float(uint2 *state, uint key) {
  // 4 rounds of Philox (standard for GPU PRNG, matches cuRAND quality).
  // 10 rounds was overkill for PatchMatch randomisation.
  for (int i = 0; i < 4; i++) {
    *state = philox2x32_round(*state, key + (uint)i * 0x9E3779B9u);
  }
  // Convert to [0, 1)
  return (float)(state->x) / 4294967296.0f;
}

// =====================================================================
// Project a 3D point onto a camera, returning (u, v, depth).
// =====================================================================
float3 project(float3 X, __global const Camera *cam) {
  // R * X + t
  float3 Xc;
  Xc.x = cam->R[0]*X.x + cam->R[1]*X.y + cam->R[2]*X.z + cam->t[0];
  Xc.y = cam->R[3]*X.x + cam->R[4]*X.y + cam->R[5]*X.z + cam->t[1];
  Xc.z = cam->R[6]*X.x + cam->R[7]*X.y + cam->R[8]*X.z + cam->t[2];
  // K * Xc
  float3 p;
  p.x = cam->K[0]*Xc.x + cam->K[1]*Xc.y + cam->K[2]*Xc.z;
  p.y = cam->K[3]*Xc.x + cam->K[4]*Xc.y + cam->K[5]*Xc.z;
  p.z = cam->K[6]*Xc.x + cam->K[7]*Xc.y + cam->K[8]*Xc.z;
  return p;
}

// =====================================================================
// Back-project pixel (x, y) at depth d to 3D using camera 0.
// =====================================================================
float3 backproject(float x, float y, float depth,
                   __global const Camera *cam) {
  // K_inv applied via direct formulas for simplicity
  float fx = cam->K[0];
  float fy = cam->K[4];
  float cx = cam->K[2];
  float cy = cam->K[5];
  float3 pt_cam;
  pt_cam.x = depth * (x - cx) / fx;
  pt_cam.y = depth * (y - cy) / fy;
  pt_cam.z = depth;
  // R^T * (pt_cam - t)
  float3 diff;
  diff.x = pt_cam.x - cam->t[0];
  diff.y = pt_cam.y - cam->t[1];
  diff.z = pt_cam.z - cam->t[2];
  float3 pt_world;
  pt_world.x = cam->R[0]*diff.x + cam->R[3]*diff.y + cam->R[6]*diff.z;
  pt_world.y = cam->R[1]*diff.x + cam->R[4]*diff.y + cam->R[7]*diff.z;
  pt_world.z = cam->R[2]*diff.x + cam->R[5]*diff.y + cam->R[8]*diff.z;
  return pt_world;
}

// =====================================================================
// Compute the depth of a plane at pixel (x, y)
// plane = (nx, ny, nz, d) where n.dot(X) + d = 0 in camera space.
// =====================================================================
float depth_from_plane(float x, float y, PlaneHypothesis plane,
                       __global const Camera *cam) {
  float fx = cam->K[0];
  float fy = cam->K[4];
  float cx = cam->K[2];
  float cy = cam->K[5];
  float denom = plane.x * (x - cx) / fx + plane.y * (y - cy) / fy + plane.z;
  if (fabs(denom) < 1e-6f) return 0.0f;
  return -plane.w / denom;
}

// =====================================================================
// Compute plane from depth and normal at pixel (x, y).
// =====================================================================
PlaneHypothesis plane_from_depth_normal(float x, float y, float depth,
                                        float3 normal,
                                        __global const Camera *cam) {
  float fx = cam->K[0];
  float fy = cam->K[4];
  float cx = cam->K[2];
  float cy = cam->K[5];
  float3 pt;
  pt.x = depth * (x - cx) / fx;
  pt.y = depth * (y - cy) / fy;
  pt.z = depth;
  float d = -dot(normal, pt);
  return (PlaneHypothesis)(normal.x, normal.y, normal.z, d);
}

// =====================================================================
// Bilateral ZNCC matching cost between reference and source patches.
// Uses sampler-based image reads for sub-pixel interpolation.
//
// Multi-scale approach: when a patch has low variance (textureless
// region), the effective radius grows by 1.5x, 2x, 2.5x, 3x while
// keeping the same number of sample points (strided sampling).
// This captures more context on uniform surfaces without extra cost.
//
// Returns the raw NCC correlation in [-1, 1] (higher = better match).
// Returns -1.0 when the match is invalid (degenerate geometry or
// insufficient patch variance at all scales).
// =====================================================================
float compute_ncc_at_stride(
    read_only image2d_t ref_img,
    read_only image2d_array_t src_images,
    int src_layer,
    float ref_center,
    float src_center,
    float src_cx, float src_cy,
    int cx, int cy,
    int half_patch,
    float stride,
    float dfdx_x, float dfdx_y,
    float dfdy_x, float dfdy_y,
    float spatial_factor,
    float color_factor
) {
  const sampler_t samp = CLK_NORMALIZED_COORDS_FALSE |
                         CLK_ADDRESS_CLAMP_TO_EDGE |
                         CLK_FILTER_LINEAR;

  // Center-shifted accumulation to avoid catastrophic cancellation.
  float sum_w = 0.0f;
  float sum_r = 0.0f;
  float sum_s = 0.0f;
  float sum_rr = 0.0f;
  float sum_ss = 0.0f;
  float sum_rs = 0.0f;


  for (int iy = -half_patch; iy <= half_patch; iy++) {
    for (int ix = -half_patch; ix <= half_patch; ix++) {
      // Strided sampling: sample at stride*ix, stride*iy in pixel space
      float fdx = stride * (float)ix;
      float fdy = stride * (float)iy;

      float rx = (float)cx + fdx;
      float ry = (float)cy + fdy;
      float ref_val = read_imagef(ref_img, samp, (float2)(rx + 0.5f, ry + 0.5f)).x;

      // Warp to source using the Jacobian
      float sx = src_cx + dfdx_x * fdx + dfdy_x * fdy;
      float sy = src_cy + dfdx_y * fdx + dfdy_y * fdy;
      float src_val = read_imagef(src_images, samp, (float4)(sx + 0.5f, sy + 0.5f, (float)src_layer, 0.0f)).x;

      // Bilateral weight: spatial × color.
      float r = ref_val - ref_center;
      float w = exp(fdx * fdx * spatial_factor + fdy * fdy * spatial_factor
                  + r * r * color_factor);
      float s = src_val - src_center;

      sum_r  += w * r;
      sum_s  += w * s;
      sum_rr += w * r * r;
      sum_ss += w * s * s;
      sum_rs += w * r * s;
      sum_w  += w;
    }
  }

  if (sum_w < 1e-6f) return -1.0f;

  float inv_w = 1.0f / sum_w;
  float mean_r = sum_r * inv_w;
  float mean_s = sum_s * inv_w;
  float var_ref = sum_rr * inv_w - mean_r * mean_r;
  float var_src = sum_ss * inv_w - mean_s * mean_s;

  if (var_ref < 1.5e-6f || var_src < 1.5e-6f) return -1.0f;

  float covar = sum_rs * inv_w - mean_r * mean_s;
  return covar / sqrt(var_ref * var_src);
}


// =====================================================================
// NCC using pre-computed warp parameters (single-scale).
//
// Takes the already-computed source projection and Jacobian, avoiding
// redundant depth_from_plane / backproject / project calls.
// Multi-scale removed — depth-prior blending handles textureless regions.
// =====================================================================
float compute_ncc_multiscale(
    read_only image2d_t ref_img,
    read_only image2d_array_t src_images,
    int src_layer,
    float ref_center,
    float src_center,
    float src_cx, float src_cy,
    int cx, int cy,
    int half_patch,
    float dfdx_x, float dfdx_y,
    float dfdy_x, float dfdy_y,
    float spatial_factor,
    float color_factor
) {
  float strides[3] = {1.0f, 2.0f, 3.0f};
  float best_ncc = -1.0f;
  for (int s = 0; s < 3; s++) {
    float ncc = compute_ncc_at_stride(
        ref_img, src_images, src_layer, ref_center, src_center,
        src_cx, src_cy, cx, cy, half_patch, strides[s],
        dfdx_x, dfdx_y, dfdy_x, dfdy_y,
        spatial_factor, color_factor);
    if (ncc > best_ncc) best_ncc = ncc;
    if (best_ncc > 0.1f) return best_ncc;
  }
  return best_ncc;
})CL"
    R"CL(

// =====================================================================
// Census-transform cost using pre-computed warp parameters.
//
// Takes the already-computed source projection and Jacobian, avoiding
// redundant depth_from_plane / backproject / project calls.
// Returns normalised Hamming distance in [0, 1] where 0 = perfect match.
// =====================================================================
float compute_census_inner(
    read_only image2d_t ref_img,
    read_only image2d_array_t src_images,
    int src_layer,
    float ref_center,
    float src_center,
    int cx, int cy,
    float src_cx, float src_cy,
    float dfdx_x, float dfdx_y,
    float dfdy_x, float dfdy_y
) {
  const sampler_t samp = CLK_NORMALIZED_COORDS_FALSE |
                         CLK_ADDRESS_CLAMP_TO_EDGE |
                         CLK_FILTER_LINEAR;

  uint ref_census = 0u;
  uint src_census = 0u;
  int bit = 0;

  for (int dy = -2; dy <= 2; dy++) {
    for (int dx = -2; dx <= 2; dx++) {
      if (dx == 0 && dy == 0) continue;

      float rx = (float)(cx + dx);
      float ry = (float)(cy + dy);
      float ref_val = read_imagef(ref_img, samp, (float2)(rx + 0.5f, ry + 0.5f)).x;

      float sx = src_cx + dfdx_x * dx + dfdy_x * dy;
      float sy = src_cy + dfdx_y * dx + dfdy_y * dy;
      float src_val = read_imagef(src_images, samp, (float4)(sx + 0.5f, sy + 0.5f, (float)src_layer, 0.0f)).x;

      if (ref_val > ref_center) ref_census |= (1u << bit);
      if (src_val > src_center) src_census |= (1u << bit);
      bit++;
    }
  }

  uint xor_val = ref_census ^ src_census;
  int hamming = popcount(xor_val);

  // Minimum contrast: reject uniform-vs-uniform matches (see
  // compute_census_cost for rationale).
  if (popcount(ref_census) < 3 && popcount(src_census) < 3) return 1.0f;

  return (float)hamming / 24.0f;
}

// =====================================================================
// Combined NCC+Census cost for a single source image, using
// pre-computed reference-side geometry (world points).
//
// This is the core optimisation: the reference camera geometry
// (depth_from_plane, backproject at center/dx/dy) is computed ONCE
// in compute_cost_vector and shared across all source images.
// Each call here only does the source-specific projection (3 project
// calls) plus NCC/Census — eliminating ~75% of redundant geometry.
// =====================================================================
float compute_single_cost(
    read_only image2d_t ref_img,
    read_only image2d_array_t src_images,
    int src_layer,
    __global const Camera *src_cam,
    float3 center_world,
    float3 world_dx,
    float3 world_dy,
    float ref_center,
    int cx, int cy,
    int half_patch,
    float spatial_factor,
    float color_factor,
    float census_weight,
    float plane_depth
) {
  const sampler_t samp = CLK_NORMALIZED_COORDS_FALSE |
                         CLK_ADDRESS_CLAMP_TO_EDGE |
                         CLK_FILTER_LINEAR;

  // Project to source (only per-source work)
  float3 src_proj = project(center_world, src_cam);
  if (src_proj.z < 1e-6f) return 2.0f;
  float src_cx = src_proj.x / src_proj.z;
  float src_cy = src_proj.y / src_proj.z;

  // Jacobian (source-specific)
  float3 proj_dx = project(world_dx, src_cam);
  float3 proj_dy = project(world_dy, src_cam);
  if (proj_dx.z < 1e-6f || proj_dy.z < 1e-6f) return 2.0f;

  float dfdx_x = proj_dx.x / proj_dx.z - src_cx;
  float dfdx_y = proj_dx.y / proj_dx.z - src_cy;
  float dfdy_x = proj_dy.x / proj_dy.z - src_cx;
  float dfdy_y = proj_dy.y / proj_dy.z - src_cy;

  float src_center = read_imagef(src_images, samp,
      (float4)(src_cx + 0.5f, src_cy + 0.5f, (float)src_layer, 0.0f)).x;

  // NCC (single-scale, using pre-computed warp)
  float ncc = compute_ncc_multiscale(ref_img, src_images, src_layer,
      ref_center, src_center, src_cx, src_cy, cx, cy, half_patch,
      dfdx_x, dfdx_y, dfdy_x, dfdy_y, spatial_factor, color_factor);

  if (ncc <= -0.5f) {
    if (census_weight > 1e-6f) {
      return compute_census_inner(ref_img, src_images, src_layer, ref_center, src_center,
          cx, cy, src_cx, src_cy, dfdx_x, dfdx_y, dfdy_x, dfdy_y);
    }
    return 2.0f;
  }

  return 1.0f - ncc;
}

// =====================================================================
// Compute per-view cost vector: one cost per source image.
// Returns the number of valid costs written.
//
// FUSED OPTIMIZATION: The bilateral weight exp(spatial + color)
// depends ONLY on the reference image.  By fusing all source views
// into a single pass over the reference patch (at stride=1), we
// compute the expensive exp() ONCE per sample instead of N times.
// This eliminates ~7/8 of all exp() calls in the common path.
//
// For sources that need larger strides (textureless regions), the
// code falls back to per-source compute_ncc_at_stride individually.
// =====================================================================
int compute_cost_vector(
    read_only image2d_t ref_img,
    __global const Camera *cameras,
    int num_images,
    int x, int y,
    PlaneHypothesis plane,
    int half_patch,
    float sigma_spatial,
    float sigma_color,
    float census_weight,
    float *out_costs,
    read_only image2d_array_t src_images,
    float center_color_weight
) {
  int n_src = min(num_images - 1, MAX_SOURCES);
  const sampler_t samp = CLK_NORMALIZED_COORDS_FALSE |
                         CLK_ADDRESS_CLAMP_TO_EDGE |
                         CLK_FILTER_LINEAR;

  // ---- Reference geometry: computed ONCE for all source images ----
  float depth = depth_from_plane((float)x, (float)y, plane, &cameras[0]);
  if (depth <= 0.0f) {
    for (int i = 0; i < n_src; i++) out_costs[i] = 2.0f;
    return n_src;
  }

  float depth_dx = depth_from_plane((float)(x+1), (float)y, plane, &cameras[0]);
  float depth_dy = depth_from_plane((float)x, (float)(y+1), plane, &cameras[0]);
  if (depth_dx <= 0.0f || depth_dy <= 0.0f) {
    for (int i = 0; i < n_src; i++) out_costs[i] = 2.0f;
    return n_src;
  }

  float3 center_world = backproject((float)x, (float)y, depth, &cameras[0]);
  float3 world_dx = backproject((float)(x+1), (float)y, depth_dx, &cameras[0]);
  float3 world_dy = backproject((float)x, (float)(y+1), depth_dy, &cameras[0]);

  float ref_center = read_imagef(ref_img, samp, (float2)(x + 0.5f, y + 0.5f)).x;

  float spatial_factor = -1.0f / (2.0f * sigma_spatial * sigma_spatial);
  float color_factor = -1.0f / (2.0f * sigma_color * sigma_color);

  // ---- Pre-project all sources: store Jacobians and center values ----
  float src_cxs[MAX_SOURCES], src_cys[MAX_SOURCES];
  float jdx_x[MAX_SOURCES], jdx_y[MAX_SOURCES];
  float jdy_x[MAX_SOURCES], jdy_y[MAX_SOURCES];
  float src_centers[MAX_SOURCES];
  int src_valid[MAX_SOURCES];

  for (int i = 0; i < n_src; i++) {
    float3 src_proj = project(center_world, &cameras[i+1]);
    if (src_proj.z < 1e-6f) { src_valid[i] = 0; out_costs[i] = 2.0f; continue; }
    src_cxs[i] = src_proj.x / src_proj.z;
    src_cys[i] = src_proj.y / src_proj.z;

    float3 proj_dx = project(world_dx, &cameras[i+1]);
    float3 proj_dy = project(world_dy, &cameras[i+1]);
    if (proj_dx.z < 1e-6f || proj_dy.z < 1e-6f) {
      src_valid[i] = 0; out_costs[i] = 2.0f; continue;
    }

    jdx_x[i] = proj_dx.x / proj_dx.z - src_cxs[i];
    jdx_y[i] = proj_dx.y / proj_dx.z - src_cys[i];
    jdy_x[i] = proj_dy.x / proj_dy.z - src_cxs[i];
    jdy_y[i] = proj_dy.y / proj_dy.z - src_cys[i];

    src_centers[i] = read_imagef(src_images, samp,
        (float4)(src_cxs[i] + 0.5f, src_cys[i] + 0.5f, (float)i, 0.0f)).x;
    src_valid[i] = 1;
  }

  // ---- FUSED NCC at stride=1: one exp() per sample for all sources ----
  // Shared reference-side accumulators:
  float sum_w = 0.0f, sum_wr = 0.0f, sum_wrr = 0.0f;
  // Per-source accumulators:
  float sum_ws[MAX_SOURCES], sum_wss[MAX_SOURCES], sum_wrs[MAX_SOURCES];
  for (int i = 0; i < MAX_SOURCES; i++) {
    sum_ws[i] = 0.0f; sum_wss[i] = 0.0f; sum_wrs[i] = 0.0f;
  }

  for (int iy = -half_patch; iy <= half_patch; iy++) {
    float ry = (float)(y + iy);

    // Hoist row-constant source base coordinates out of inner loop.
    float base_sx[MAX_SOURCES], base_sy[MAX_SOURCES];
    for (int s = 0; s < n_src; s++) {
      if (!src_valid[s]) continue;
      base_sx[s] = src_cxs[s] + jdy_x[s] * (float)iy;
      base_sy[s] = src_cys[s] + jdy_y[s] * (float)iy;
    }

    for (int ix = -half_patch; ix <= half_patch; ix++) {
      float rx = (float)(x + ix);

      // --- Reference: read ONCE, compute weight ONCE for all sources ---
      float ref_val = read_imagef(ref_img, samp, (float2)(rx + 0.5f, ry + 0.5f)).x;
      float r = ref_val - ref_center;
      float w = exp(spatial_factor * (float)(ix*ix + iy*iy)
                  + color_factor * r * r);

      sum_w += w;
      float wr = w * r;
      sum_wr += wr;
      sum_wrr += wr * r;

      // --- All valid sources: only source-specific work here ---
      for (int s = 0; s < n_src; s++) {
        if (!src_valid[s]) continue;
        float sx = base_sx[s] + jdx_x[s] * (float)ix;
        float sy = base_sy[s] + jdx_y[s] * (float)ix;
        float src_val = read_imagef(src_images, samp,
            (float4)(sx + 0.5f, sy + 0.5f, (float)s, 0.0f)).x;
        float sv = src_val - src_centers[s];
        sum_ws[s] += w * sv;
        sum_wss[s] += w * sv * sv;
        sum_wrs[s] += wr * sv;
      }
    }
  }

  // ---- Compute NCC from accumulated stats; fallback for poor matches ----
  if (sum_w < 1e-6f) {
    for (int i = 0; i < n_src; i++) out_costs[i] = 2.0f;
    return n_src;
  }

  float inv_w = 1.0f / sum_w;
  float mean_r = sum_wr * inv_w;
  float var_ref = sum_wrr * inv_w - mean_r * mean_r;

  for (int s = 0; s < n_src; s++) {
    if (!src_valid[s]) continue;

    float ncc_s1 = -1.0f;  // stride=1 NCC result

    if (var_ref >= 1.5e-6f) {
      float mean_s = sum_ws[s] * inv_w;
      float var_src = sum_wss[s] * inv_w - mean_s * mean_s;
      if (var_src >= 1.5e-6f) {
        float covar = sum_wrs[s] * inv_w - mean_r * mean_s;
        ncc_s1 = covar / sqrt(var_ref * var_src);
      }
    }

    // Fast path: good match at stride=1 (majority of cases)
    if (ncc_s1 > 0.1f) {
      out_costs[s] = 1.0f - ncc_s1;
      continue;
    }

    // Slow path: try larger strides individually (rare: textureless)
    float best_ncc = ncc_s1;
    float strides_fallback[2] = {2.0f, 3.0f};
    for (int st = 0; st < 2; st++) {
      float ncc = compute_ncc_at_stride(ref_img, src_images, s,
          ref_center, src_centers[s], src_cxs[s], src_cys[s],
          x, y, half_patch, strides_fallback[st],
          jdx_x[s], jdx_y[s], jdy_x[s], jdy_y[s],
          spatial_factor, color_factor);
      if (ncc > best_ncc) best_ncc = ncc;
      if (best_ncc > 0.1f) break;
    }

    if (best_ncc <= -0.5f) {
      out_costs[s] = (census_weight > 1e-6f)
          ? compute_census_inner(ref_img, src_images, s,
                ref_center, src_centers[s], x, y,
                src_cxs[s], src_cys[s],
                jdx_x[s], jdx_y[s], jdy_x[s], jdy_y[s])
          : 2.0f;
    } else {
      out_costs[s] = 1.0f - best_ncc;
    }
  }


  // Center-pixel color consensus penalty: if the projected center in a
  // source view has significantly different intensity, the hypothesis
  // likely maps to the wrong surface.  Penalty ramps linearly from 0
  // (within 1x sigma_color) to center_color_weight (at 2x sigma_color).
  if (center_color_weight > 0.0f) {
    float inv_sc = 1.0f / max(sigma_color, 1e-6f);
    for (int s = 0; s < n_src; s++) {
      if (out_costs[s] >= 2.0f) continue;  // already invalid
      float cc_diff = fabs(ref_center - src_centers[s]);
      float cc_excess = cc_diff * inv_sc - 1.0f;
      if (cc_excess > 0.0f) {
        out_costs[s] += center_color_weight * fmin(cc_excess, 1.0f);
      }
    }
  }

  return n_src;
})CL"
    R"CL(

// =====================================================================
// Compute initial multi-view cost and selected views bitmask.
// Evaluates all source images, picks top-k, returns mean cost and
// stores a bitmask of which views were selected.
// =====================================================================
float compute_initial_cost_and_views(
    read_only image2d_t ref_img,
    __global const Camera *cameras,
    int num_images,
    int x, int y,
    PlaneHypothesis plane,
    int half_patch,
    float sigma_spatial,
    float sigma_color,
    int top_k,
    float census_weight,
    uint *out_selected,
    read_only image2d_array_t src_images,
    float center_color_weight
) {
  float cost_vector[MAX_SOURCES];
  float cost_copy[MAX_SOURCES];
  int n_src = min(num_images - 1, MAX_SOURCES);
  for (int i = 0; i < MAX_SOURCES; i++) {
    cost_vector[i] = 2.0f;
    cost_copy[i] = 2.0f;
  }

  compute_cost_vector(ref_img, cameras, num_images, x, y, plane,
                      half_patch, sigma_spatial, sigma_color, census_weight,
                      cost_vector, src_images, center_color_weight);

  int num_valid = 0;
  for (int i = 0; i < n_src; i++) {
    cost_copy[i] = cost_vector[i];
    if (cost_vector[i] < 2.0f) num_valid++;
  }

  // Sort the cost_vector ascending (insertion sort)
  for (int i = 1; i < n_src; i++) {
    float tmp = cost_vector[i];
    int j = i;
    for (; j >= 1 && tmp < cost_vector[j-1]; j--)
      cost_vector[j] = cost_vector[j-1];
    cost_vector[j] = tmp;
  }

  *out_selected = 0u;
  int k = min(min(top_k, num_valid), n_src);
  if (k <= 0) return 2.0f;

  float total = 0.0f;
  for (int i = 0; i < k; i++) total += cost_vector[i];
  float threshold = cost_vector[k - 1];

  for (int i = 0; i < n_src; i++) {
    if (cost_copy[i] <= threshold) {
      *out_selected |= (1u << i);
    }
  }
  return total / (float)k;
}

// =====================================================================
// Convert PDF to CDF in-place.
// =====================================================================
void pdf_to_cdf(float *probs, int n) {
  float sum = 0.0f;
  for (int i = 0; i < n; i++) sum += probs[i];
  if (sum < 1e-12f) return;
  float inv = 1.0f / sum;
  float cum = 0.0f;
  for (int i = 0; i < n; i++) {
    cum += probs[i] * inv;
    probs[i] = cum;
  }
}

// =====================================================================
// Bilateral-gated smoothness cost.
//
// Measures how well a plane hypothesis agrees with its spatial
// neighbours in depth and normal orientation.  A bilateral gate on
// relative depth difference preserves discontinuities: neighbours
// at very different depths contribute near-zero weight.
//
// Samples the 4-connected + 4 diagonal neighbours (up to 8).
// Returns a cost in [0, 1] where 0 = perfectly smooth.
// =====================================================================
float compute_smoothness_cost(
    read_only image2d_t planes_img,
    __global const Camera *cam,
    PlaneHypothesis cur_plane,
    int x, int y, int width, int height
) {
  float cur_depth = depth_from_plane((float)x, (float)y, cur_plane, cam);
  if (cur_depth <= 0.0f) return 0.0f;

  float3 cur_n = normalize((float3)(cur_plane.x, cur_plane.y, cur_plane.z));

  // Bilateral sigma: 2% relative depth.  Neighbours beyond this
  // relative distance are treated as belonging to a different surface.
  const float sigma_rel = 0.02f;

  // 4-connected offsets only.  Diagonal neighbours have the *same*
  // checkerboard parity, so reading them during a red/black pass is a
  // data race (they're being written concurrently by other work-items).
  const int dx[4] = {-1,  1,  0,  0};
  const int dy[4] = { 0,  0, -1,  1};

  float total_weight = 0.0f;
  float total_cost = 0.0f;

  for (int k = 0; k < 4; k++) {
    int nx = x + dx[k];
    int ny = y + dy[k];
    if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;

    PlaneHypothesis nb = read_imagef(planes_img, samp_planes, (int2)(nx, ny));
    float nb_depth = depth_from_plane((float)nx, (float)ny, nb, cam);
    if (nb_depth <= 0.0f) continue;

    // Bilateral gate: exponential falloff on relative depth difference
    float rel_diff = fabs(cur_depth - nb_depth) / cur_depth;
    float w = exp(-rel_diff * rel_diff / (2.0f * sigma_rel * sigma_rel));

    // Normal disagreement: 1 - dot(n_cur, n_nb), in [0, 2]
    float3 nb_n = normalize((float3)(nb.x, nb.y, nb.z));
    float normal_cost = 1.0f - dot(cur_n, nb_n);  // 0 = same, 2 = opposite

    // Depth surface deviation: does the neighbour's depth agree with
    // what our plane predicts at the neighbour's pixel position?
    float predicted = depth_from_plane((float)nx, (float)ny, cur_plane, cam);
    float depth_dev = (predicted > 0.0f)
        ? fabs(nb_depth - predicted) / cur_depth
        : 0.0f;

    // Combined: 50% normal, 50% depth deviation, both in [0, ~1]
    float cost = 0.5f * clamp(normal_cost, 0.0f, 1.0f)
               + 0.5f * clamp(depth_dev * 10.0f, 0.0f, 1.0f);

    total_cost += w * cost;
    total_weight += w;
  }

  if (total_weight < 1e-6f) return 0.0f;
  return total_cost / total_weight;
}

// =====================================================================
// Compute surface normal from the depth gradient of 4-connected neighbors.
//
// from how depth changes spatially across neighboring pixels.  This
// acts as an implicit geometric regularizer — biasing normals toward
// being consistent with the local depth surface — without any explicit
// smoothness term.
//
// The normal is computed as:
//   dx = depth[x+1] - depth[x-1]   (central difference)
//   dy = depth[y+1] - depth[y-1]
//   n  = normalize(fx*dx, fy*dy, (cx-x)*dx + (cy-y)*dy - depth)
//
// Returns (0,0,0) if the normal cannot be computed (border pixels
// or neighbors with invalid depths).
// =====================================================================
float3 compute_surface_normal(
    read_only image2d_t planes_img,
    __global const Camera *cam,
    int x, int y, int width, int height,
    float center_depth
) {
  if (center_depth <= 0.0f) return (float3)(0.0f, 0.0f, 0.0f);
  if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1)
    return (float3)(0.0f, 0.0f, 0.0f);

  float fx = cam->K[0];
  float fy = cam->K[4];
  float cx = cam->K[2];
  float cy = cam->K[5];

  // Read 4-connected neighbor depths via texture cache.
  // These are opposite-color pixels in the checkerboard, so they
  // were written in the previous half-iteration — safe to read.
  PlaneHypothesis pl_l = read_imagef(planes_img, samp_planes, (int2)(x - 1, y));
  PlaneHypothesis pl_r = read_imagef(planes_img, samp_planes, (int2)(x + 1, y));
  PlaneHypothesis pl_u = read_imagef(planes_img, samp_planes, (int2)(x, y - 1));
  PlaneHypothesis pl_d = read_imagef(planes_img, samp_planes, (int2)(x, y + 1));

  float d_l = depth_from_plane((float)(x - 1), (float)y, pl_l, cam);
  float d_r = depth_from_plane((float)(x + 1), (float)y, pl_r, cam);
  float d_u = depth_from_plane((float)x, (float)(y - 1), pl_u, cam);
  float d_d = depth_from_plane((float)x, (float)(y + 1), pl_d, cam);

  if (d_l <= 0.0f || d_r <= 0.0f || d_u <= 0.0f || d_d <= 0.0f)
    return (float3)(0.0f, 0.0f, 0.0f);

  float dx = d_r - d_l;
  float dy = d_d - d_u;

  float3 n;
  n.x = fx * dx;
  n.y = fy * dy;
  n.z = (cx - (float)x) * dx + (cy - (float)y) * dy - center_depth;

  float len = length(n);
  if (len < 1e-8f) return (float3)(0.0f, 0.0f, 0.0f);
  n = n / len;

  // Ensure normal faces camera (z < 0 in camera space)
  if (n.z > 0.0f) n = -n;

  return n;
}

// =====================================================================
// Sample a previous-scale depth map stored in a flat global buffer.
// prev_depths layout: [MAX_SOURCES * width * height] floats.
// Uses nearest-neighbor sampling (avoids blurring depth discontinuities).
// =====================================================================
float sample_prev_depth(
    __global const float *prev_depths,
    int src_idx,
    float u, float v,
    int width, int height
) {
  int ix = clamp((int)(u + 0.5f), 0, width - 1);
  int iy = clamp((int)(v + 0.5f), 0, height - 1);
  return prev_depths[src_idx * width * height + iy * width + ix];
}

// =====================================================================
// Geometric consistency cost (matches reference ComputeGeomConsistencyCost).
//
// For a plane hypothesis at pixel (px, py) in the reference camera:
//   1. Compute depth from plane, backproject to 3D world point.
//   2. Project into source camera → (src_x, src_y).
//   3. Look up previous-scale depth at (src_x, src_y) in source view.
//   4. Backproject source pixel with that depth to 3D.
//   5. Re-project to reference camera → (back_x, back_y).
//   6. Return 2D reprojection error ||p - back||.
//
// If the hypothesis is geometrically consistent with the previous depth,
// the forward-backward roundtrip returns to the original pixel (cost ≈ 0).
// =====================================================================
float compute_geom_consistency_cost(
    __global const Camera *ref_cam,
    __global const Camera *src_cam,
    PlaneHypothesis plane,
    int px, int py,
    __global const float *prev_depths,
    int src_idx,
    int pd_width, int pd_height
) {
  const float max_cost = 5.0f;

  // 1. Depth from plane at reference pixel
  float depth = depth_from_plane((float)px, (float)py, plane, ref_cam);
  if (depth <= 0.0f) return max_cost;

  // 2. Backproject to world
  float3 world_pt = backproject((float)px, (float)py, depth, ref_cam);

  // 3. Project to source camera
  float3 src_proj = project(world_pt, src_cam);
  if (src_proj.z < 1e-6f) return max_cost;
  float src_x = src_proj.x / src_proj.z;
  float src_y = src_proj.y / src_proj.z;

  // Bounds check
  if (src_x < 0.0f || src_x >= (float)pd_width ||
      src_y < 0.0f || src_y >= (float)pd_height)
    return max_cost;

  // 4. Look up previous depth at source pixel
  float src_depth = sample_prev_depth(prev_depths, src_idx,
                                      src_x, src_y, pd_width, pd_height);
  if (src_depth <= 0.0f) return max_cost;

  // 5. Backproject source pixel with previous depth to world
  float3 src_world = backproject(src_x, src_y, src_depth, src_cam);

  // 6. Re-project to reference camera
  float3 ref_proj = project(src_world, ref_cam);
  if (ref_proj.z < 1e-6f) return max_cost;
  float back_x = ref_proj.x / ref_proj.z;
  float back_y = ref_proj.y / ref_proj.z;

  // 7. 2D reprojection error
  float dx = (float)px - back_x;
  float dy = (float)py - back_y;
  return min(max_cost, sqrt(dx * dx + dy * dy));
}

// =====================================================================
// View-weighted cost with geometric consistency.
//
// For source views that have previous-scale depth maps (indicated by
// prev_depth_mask), the geometric consistency cost is added to the
// photometric cost before view-weighting, following the reference:
//   cost += view_weight * (photo + geom_weight * geom)
// =====================================================================
float compute_weighted_cost_geom(
    const float *cost_vec,
    const float *view_weights,
    float weight_norm,
    int n_src,
    __global const Camera *cameras,
    PlaneHypothesis plane,
    int x, int y,
    __global const float *prev_depths,
    uint prev_depth_mask,
    int pd_width, int pd_height,
    float geom_weight
) {
  if (weight_norm < 1e-6f) return 2.0f;
  float total = 0.0f;
  for (int i = 0; i < n_src; i++) {
    if (view_weights[i] > 0.0f) {
      float photo = cost_vec[i];
      float geom = 0.0f;
      if (geom_weight > 0.0f && (prev_depth_mask & (1u << i))) {
        geom = compute_geom_consistency_cost(
            &cameras[0], &cameras[i + 1],
            plane, x, y,
            prev_depths, i, pd_width, pd_height);
      }
      total += view_weights[i] * (photo + geom_weight * geom);
    }
  }
  return total / weight_norm;
}

// =====================================================================
// Monte Carlo importance-sampled cost (COLMAP-inspired).
//
// Instead of a deterministic weighted sum, builds a CDF from
// view_weights and draws num_samples random views.  This ensures no
// view is ever permanently excluded — even low-weight views have a
// chance of being sampled, breaking conspiracy feedback loops.
// =====================================================================
#define MC_NUM_SAMPLES 15

float compute_mc_cost(
    const float *cost_vec,
    const float *view_weights,
    float weight_norm,
    int n_src,
    __global const Camera *cameras,
    PlaneHypothesis plane,
    int x, int y,
    __global const float *prev_depths,
    uint prev_depth_mask,
    int pd_width, int pd_height,
    float geom_weight,
    uint2 *rng_state,
    uint rng_key
) {
  if (weight_norm < 1e-6f) return 2.0f;

  // Build CDF from view_weights.
  float cdf[MAX_SOURCES];
  float cum = 0.0f;
  for (int i = 0; i < n_src; i++) {
    cum += view_weights[i];
    cdf[i] = cum;
  }
  // cum == weight_norm at this point.
  float inv_norm = 1.0f / cum;

  // Draw MC_NUM_SAMPLES random views and accumulate cost.
  float total = 0.0f;
  int valid_samples = 0;
  for (int s = 0; s < MC_NUM_SAMPLES; s++) {
    float u = rand_float(rng_state, rng_key + (uint)s * 7919u);
    float threshold = u * cum;

    // Find view from CDF (linear scan — MAX_SOURCES is small).
    int sampled_view = n_src - 1;
    for (int i = 0; i < n_src; i++) {
      if (cdf[i] > threshold) {
        sampled_view = i;
        break;
      }
    }

    // Compute cost for sampled view.
    float photo = cost_vec[sampled_view];
    float geom = 0.0f;
    if (geom_weight > 0.0f && (prev_depth_mask & (1u << sampled_view))) {
      geom = compute_geom_consistency_cost(
          &cameras[0], &cameras[sampled_view + 1],
          plane, x, y,
          prev_depths, sampled_view, pd_width, pd_height);
    }
    total += photo + geom_weight * geom;
    valid_samples++;
  }

  return (valid_samples > 0) ? (total / (float)valid_samples) : 2.0f;
}
)CL"
    R"CL(

// =====================================================================
// Kernel: Random initialisation of plane hypotheses.
//
// Generates random depth + normal for each pixel, computes initial
// cost, and stores the per-pixel selected-views bitmask.
// =====================================================================
__kernel void acmmp_random_init(
    __global PlaneHypothesis *planes,
    __global float *costs,
    __global uint2 *rand_states,
    __global uint *selected_views,
    __global const Camera *cameras,
    read_only image2d_t ref_img,
    read_only image2d_array_t src_images,
    int width, int height,
    float depth_min, float depth_max,
    int num_images,
    int half_patch,
    float sigma_spatial,
    float sigma_color,
    int top_k,
    float census_weight,
    float center_color_weight
) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  if (x >= width || y >= height) return;

  int idx = y * width + x;

  // Initialise PRNG state from pixel coordinates
  uint2 state = (uint2)(idx * 1099087573u + 2654435769u,
                        idx * 2246822519u + 13u);
  uint key = (uint)(idx + 7919);

  // Random depth in log space
  float log_d_min = log(max(depth_min, 1e-6f));
  float log_d_max = log(max(depth_max, 1e-5f));
  float depth = exp(log_d_min + rand_float(&state, key) * (log_d_max - log_d_min));

  // Random normal (hemisphere facing camera)
  float3 normal;
  normal.x = rand_float(&state, key + 3u) * 2.0f - 1.0f;
  normal.y = rand_float(&state, key + 5u) * 2.0f - 1.0f;
  normal.z = -1.0f;
  normal = normalize(normal);

  PlaneHypothesis plane = plane_from_depth_normal(
      (float)x, (float)y, depth, normal, &cameras[0]);

  planes[idx] = plane;
  rand_states[idx] = state;

  // Compute initial cost and selected views
  uint sel = 0u;
  costs[idx] = compute_initial_cost_and_views(
      ref_img, cameras, num_images,
      x, y, plane, half_patch, sigma_spatial, sigma_color, top_k,
      census_weight, &sel, src_images, center_color_weight);
  selected_views[idx] = sel;
}

// =====================================================================
// Kernel: Planar-prior re-initialisation (cost-matched).
//
// Re-seeds masked pixels with the SfM Delaunay prior plane (±1% depth
// perturbation) and computes the SAME total cost as acmmp_patchmatch
// (photometric + smoothness + geometric consistency).  This ensures the
// comparison against the stored cost is fair — the prior can only replace
// an existing hypothesis when it genuinely matches better.
//
// Unlike the original implementation which used compute_initial_cost_and_views
// (photometric-only, fresh top-k view selection), this version:
//   1. Reuses the existing per-pixel selected_views for cost computation.
//   2. Includes smoothness and geometric consistency terms.
//   3. Only updates selected_views if the prior hypothesis is accepted.
// =====================================================================
__kernel void acmmp_prior_reinit(
    __global PlaneHypothesis *planes,
    __global float *costs,
    read_only image2d_t planes_img,
    read_only image2d_t costs_img,
    __global uint2 *rand_states,
    __global uint *selected_views,
    __global const Camera *cameras,
    read_only image2d_t ref_img,
    read_only image2d_array_t src_images,
    int width, int height,
    float depth_min, float depth_max,
    int num_images,
    int half_patch,
    float sigma_spatial,
    float sigma_color,
    int top_k,
    float census_weight,
    __global const PlaneHypothesis *prior_planes,
    __global const uint *plane_masks,
    float smooth_weight,
    __global const float *prev_depths,
    uint prev_depth_mask,
    float geom_weight,
    float center_color_weight
) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  if (x >= width || y >= height) return;

  int idx = y * width + x;

  // Only consider masked pixels (inside a Delaunay triangle).
  if (plane_masks[idx] == 0u) return;

  // Recover prior depth at this pixel.
  float depth_prior = depth_from_plane((float)x, (float)y,
                                       prior_planes[idx], &cameras[0]);
  if (depth_prior <= 0.0f || depth_prior < depth_min || depth_prior > depth_max)
    return;

  // Perturb prior depth by ±1% (reference: curand_uniform * 0.02 * d + 0.99 * d).
  uint2 state = rand_states[idx];
  uint key = (uint)(idx + 31337u);
  float depth_perturbed = depth_prior * 0.99f
      + rand_float(&state, key) * depth_prior * 0.02f;
  depth_perturbed = clamp(depth_perturbed, depth_min, depth_max);

  // Use prior normal directly.
  float3 prior_n = normalize((float3)(prior_planes[idx].x,
                                      prior_planes[idx].y,
                                      prior_planes[idx].z));

  PlaneHypothesis plane = plane_from_depth_normal(
      (float)x, (float)y, depth_perturbed, prior_n, &cameras[0]);

  // ---- Reuse the existing per-pixel view selection for cost ----
  int n_src = min(num_images - 1, MAX_SOURCES);
  uint cur_sel = selected_views[idx];

  // Derive uniform weights from the existing selection bitmask.
  float view_weights[MAX_SOURCES];
  float weight_norm = 0.0f;
  for (int i = 0; i < MAX_SOURCES; i++) {
    if (i < n_src && (cur_sel & (1u << i))) {
      view_weights[i] = 1.0f;
      weight_norm += 1.0f;
    } else {
      view_weights[i] = 0.0f;
    }
  }

  // If no views are selected yet (e.g. first iteration), fall back to
  // compute_initial_cost_and_views to bootstrap.
  if (weight_norm < 1e-6f) {
    uint sel = 0u;
    float cost = compute_initial_cost_and_views(
        ref_img, cameras, num_images,
        x, y, plane, half_patch, sigma_spatial, sigma_color, top_k,
        census_weight, &sel,
        src_images, center_color_weight);

    if (smooth_weight > 1e-6f) {
      cost += smooth_weight * compute_smoothness_cost(planes_img, &cameras[0],
          plane, x, y, width, height);
    }

    if (cost < costs[idx]) {
      planes[idx] = plane;
      costs[idx] = cost;
      selected_views[idx] = sel;
    }
    rand_states[idx] = state;
    return;
  }

  // ---- Compute the same total cost as acmmp_patchmatch ----
  float cost_vec[MAX_SOURCES];
  for (int i = 0; i < MAX_SOURCES; i++) cost_vec[i] = 2.0f;
  compute_cost_vector(ref_img, cameras, num_images, x, y, plane,
      half_patch, sigma_spatial, sigma_color, census_weight,
      cost_vec,
      src_images, center_color_weight);

  float cost = compute_weighted_cost_geom(cost_vec, view_weights, weight_norm,
      n_src, cameras, plane, x, y,
      prev_depths, prev_depth_mask, width, height, geom_weight);

  if (smooth_weight > 1e-6f) {
    cost += smooth_weight * compute_smoothness_cost(planes_img, &cameras[0],
        plane, x, y, width, height);
  }

  // Only replace the existing hypothesis if the prior-based plane
  // actually matches better under the SAME cost function.
  if (cost < costs[idx]) {
    planes[idx] = plane;
    costs[idx] = cost;
    // Keep existing view selection — prior only changes depth/normal.
  }
  rand_states[idx] = state;
}

// =====================================================================
// Kernel: Adaptive Checkerboard PatchMatch iteration.
//
// Implements the PatchMatch propagation from Xu et al. (TPAMI 2022):
//   1. Adaptive checkerboard sampling: 8 candidate positions
//      (4 near + 4 far) with long-range search.
//   2. Multi-hypothesis joint view selection: per-pixel view weights
//      via CDF-based sampling informed by neighbor view selections.
//   3. View-weighted cost aggregation: accept best candidate.
//   4. Multi-hypothesis refinement with 5 diverse plane candidates.
//
// color_flag: 0 = red pixels (x+y even), 1 = black pixels (x+y odd).
// =====================================================================
__kernel void acmmp_patchmatch(
    __global PlaneHypothesis *planes,
    __global float *costs,
    read_only image2d_t planes_img,
    read_only image2d_t costs_img,
    __global uint2 *rand_states,
    __global uint *selected_views,
    __global const Camera *cameras,
    read_only image2d_t ref_img,
    read_only image2d_array_t src_images,
    int width, int height,
    float depth_min, float depth_max,
    int num_images,
    int half_patch,
    float sigma_spatial,
    float sigma_color,
    int top_k,
    float census_weight,
    int color_flag,
    int iteration,
    float smooth_weight,
    __global const float *prev_depths,
    uint prev_depth_mask,
    float geom_weight,
    __global const PlaneHypothesis *prior_planes,
    __global const uint *plane_masks,
    float edge_weight,
    float escape_depth_ratio,
    float center_color_weight,
    float variance_gate,
    int anchor_views,
    float far_gradient_threshold,
    __global const float *angle_weights,
    __global const int *segment_labels
) {
  // Dense checkerboard indexing: grid is (half_width x height), reconstruct
  // true pixel coordinates so all SIMD lanes are active on Apple Silicon.
  int gid_x = get_global_id(0);
  int gid_y = get_global_id(1);
  int half_width = (width + 1) / 2;
  if (gid_x >= half_width || gid_y >= height) return;
  int x = gid_x * 2 + ((gid_y + color_flag) % 2);
  int y = gid_y;
  if (x >= width) return;

  int idx = y * width + x;
  uint2 state = rand_states[idx];
  uint key = (uint)(idx + iteration * 7919 + 42);
  int n_src = min(num_images - 1, MAX_SOURCES);
  int npix = width * height;

  // =================================================================
  // Step 1: Identify 8 candidate neighbor positions (adaptive search)
  // =================================================================
  // Layout: 0=near_up, 1=far_up, 2=near_down, 3=far_down,
  //         4=near_left, 5=far_left, 6=near_right, 7=far_right
  int positions[8];
  bool flags[8];
  float cost_array[8 * MAX_SOURCES];  // cost_array[cand * MAX_SOURCES + view]
  for (int i = 0; i < 8; i++) { positions[i] = idx; flags[i] = false; }
  for (int i = 0; i < 8 * MAX_SOURCES; i++) cost_array[i] = 2.0f;

  // --- Far up (start at y-3, search every 2 rows up to 23 rows) ---
  if (y > 2) {
    int best_py = y - 3;
    float best_c = read_imagef(costs_img, samp_planes, (int2)(x, best_py)).x;
    for (int i = 1; i < 11; i++) {
      int yy = y - 3 - 2 * i;
      if (yy < 0) break;
      float c = read_imagef(costs_img, samp_planes, (int2)(x, yy)).x;
      if (c < best_c) { best_c = c; best_py = yy; }
    }
    positions[1] = best_py * width + x; flags[1] = true;
    PlaneHypothesis best_plane = read_imagef(planes_img, samp_planes, (int2)(x, best_py));
    compute_cost_vector(ref_img, cameras, num_images, x, y,
        best_plane, half_patch, sigma_spatial, sigma_color, census_weight,
        &cost_array[1 * MAX_SOURCES],
        src_images, center_color_weight);
  }

  // --- Far down ---
  if (y < height - 3) {
    int best_py = y + 3;
    float best_c = read_imagef(costs_img, samp_planes, (int2)(x, best_py)).x;
    for (int i = 1; i < 11; i++) {
      int yy = y + 3 + 2 * i;
      if (yy >= height) break;
      float c = read_imagef(costs_img, samp_planes, (int2)(x, yy)).x;
      if (c < best_c) { best_c = c; best_py = yy; }
    }
    positions[3] = best_py * width + x; flags[3] = true;
    PlaneHypothesis best_plane = read_imagef(planes_img, samp_planes, (int2)(x, best_py));
    compute_cost_vector(ref_img, cameras, num_images, x, y,
        best_plane, half_patch, sigma_spatial, sigma_color, census_weight,
        &cost_array[3 * MAX_SOURCES],
        src_images, center_color_weight);
  }

  // --- Far left ---
  if (x > 2) {
    int best_px = x - 3;
    float best_c = read_imagef(costs_img, samp_planes, (int2)(best_px, y)).x;
    for (int i = 1; i < 11; i++) {
      int xx = x - 3 - 2 * i;
      if (xx < 0) break;
      float c = read_imagef(costs_img, samp_planes, (int2)(xx, y)).x;
      if (c < best_c) { best_c = c; best_px = xx; }
    }
    positions[5] = y * width + best_px; flags[5] = true;
    PlaneHypothesis best_plane = read_imagef(planes_img, samp_planes, (int2)(best_px, y));
    compute_cost_vector(ref_img, cameras, num_images, x, y,
        best_plane, half_patch, sigma_spatial, sigma_color, census_weight,
        &cost_array[5 * MAX_SOURCES],
        src_images, center_color_weight);
  }

  // --- Far right ---
  if (x < width - 3) {
    int best_px = x + 3;
    float best_c = read_imagef(costs_img, samp_planes, (int2)(best_px, y)).x;
    for (int i = 1; i < 11; i++) {
      int xx = x + 3 + 2 * i;
      if (xx >= width) break;
      float c = read_imagef(costs_img, samp_planes, (int2)(xx, y)).x;
      if (c < best_c) { best_c = c; best_px = xx; }
    }
    positions[7] = y * width + best_px; flags[7] = true;
    PlaneHypothesis best_plane = read_imagef(planes_img, samp_planes, (int2)(best_px, y));
    compute_cost_vector(ref_img, cameras, num_images, x, y,
        best_plane, half_patch, sigma_spatial, sigma_color, census_weight,
        &cost_array[7 * MAX_SOURCES],
        src_images, center_color_weight);
  }

  // --- Near up (with diagonal V-shape search) ---
  if (y > 0) {
    int best_px = x, best_py = y - 1;
    float best_c = read_imagef(costs_img, samp_planes, (int2)(best_px, best_py)).x;
    for (int i = 0; i < 3; i++) {
      if (y > 1 + i && x > i) {
        int px = x - i - 1, py = y - 2 - i;
        float c = read_imagef(costs_img, samp_planes, (int2)(px, py)).x;
        if (c < best_c) { best_c = c; best_px = px; best_py = py; }
      }
      if (y > 1 + i && x < width - 1 - i) {
        int px = x + i + 1, py = y - 2 - i;
        float c = read_imagef(costs_img, samp_planes, (int2)(px, py)).x;
        if (c < best_c) { best_c = c; best_px = px; best_py = py; }
      }
    }
    positions[0] = best_py * width + best_px; flags[0] = true;
    PlaneHypothesis best_plane = read_imagef(planes_img, samp_planes, (int2)(best_px, best_py));
    compute_cost_vector(ref_img, cameras, num_images, x, y,
        best_plane, half_patch, sigma_spatial, sigma_color, census_weight,
        &cost_array[0 * MAX_SOURCES],
        src_images, center_color_weight);
  }
)CL"
    R"CL(

  // --- Near down ---
  if (y < height - 1) {
    int best_px = x, best_py = y + 1;
    float best_c = read_imagef(costs_img, samp_planes, (int2)(best_px, best_py)).x;
    for (int i = 0; i < 3; i++) {
      if (y < height - 2 - i && x > i) {
        int px = x - i - 1, py = y + 2 + i;
        float c = read_imagef(costs_img, samp_planes, (int2)(px, py)).x;
        if (c < best_c) { best_c = c; best_px = px; best_py = py; }
      }
      if (y < height - 2 - i && x < width - 1 - i) {
        int px = x + i + 1, py = y + 2 + i;
        float c = read_imagef(costs_img, samp_planes, (int2)(px, py)).x;
        if (c < best_c) { best_c = c; best_px = px; best_py = py; }
      }
    }
    positions[2] = best_py * width + best_px; flags[2] = true;
    PlaneHypothesis best_plane = read_imagef(planes_img, samp_planes, (int2)(best_px, best_py));
    compute_cost_vector(ref_img, cameras, num_images, x, y,
        best_plane, half_patch, sigma_spatial, sigma_color, census_weight,
        &cost_array[2 * MAX_SOURCES],
        src_images, center_color_weight);
  }

  // --- Near left ---
  if (x > 0) {
    int best_px = x - 1, best_py = y;
    float best_c = read_imagef(costs_img, samp_planes, (int2)(best_px, best_py)).x;
    for (int i = 0; i < 3; i++) {
      if (x > 1 + i && y > i) {
        int px = x - 2 - i, py = y - i - 1;
        float c = read_imagef(costs_img, samp_planes, (int2)(px, py)).x;
        if (c < best_c) { best_c = c; best_px = px; best_py = py; }
      }
      if (x > 1 + i && y < height - 1 - i) {
        int px = x - 2 - i, py = y + i + 1;
        float c = read_imagef(costs_img, samp_planes, (int2)(px, py)).x;
        if (c < best_c) { best_c = c; best_px = px; best_py = py; }
      }
    }
    positions[4] = best_py * width + best_px; flags[4] = true;
    PlaneHypothesis best_plane = read_imagef(planes_img, samp_planes, (int2)(best_px, best_py));
    compute_cost_vector(ref_img, cameras, num_images, x, y,
        best_plane, half_patch, sigma_spatial, sigma_color, census_weight,
        &cost_array[4 * MAX_SOURCES],
        src_images, center_color_weight);
  }

  // --- Near right ---
  if (x < width - 1) {
    int best_px = x + 1, best_py = y;
    float best_c = read_imagef(costs_img, samp_planes, (int2)(best_px, best_py)).x;
    for (int i = 0; i < 3; i++) {
      if (x < width - 2 - i && y > i) {
        int px = x + 2 + i, py = y - i - 1;
        float c = read_imagef(costs_img, samp_planes, (int2)(px, py)).x;
        if (c < best_c) { best_c = c; best_px = px; best_py = py; }
      }
      if (x < width - 2 - i && y < height - 1 - i) {
        int px = x + 2 + i, py = y + i + 1;
        float c = read_imagef(costs_img, samp_planes, (int2)(px, py)).x;
        if (c < best_c) { best_c = c; best_px = px; best_py = py; }
      }
    }
    positions[6] = best_py * width + best_px; flags[6] = true;
    PlaneHypothesis best_plane = read_imagef(planes_img, samp_planes, (int2)(best_px, best_py));
    compute_cost_vector(ref_img, cameras, num_images, x, y,
        best_plane, half_patch, sigma_spatial, sigma_color, census_weight,
        &cost_array[6 * MAX_SOURCES],
        src_images, center_color_weight);
  }

  // =================================================================
  // Step 2: Diversity-Enforced View Selection
  // =================================================================
  //
  // Redesign rationale: The original ACMMP CDF sampling narrows the
  // view set across iterations via magic-number Gaussian kernels.  This
  // causes a conspiracy where a few low-baseline views all agree on a
  // leaked foreground depth, and other views that DISAGREE are starved
  // of weight.  The redesign:
  //   - Count how many candidates each view validates (cost < threshold)
  //   - Give weight = validation_count (flat, no Gaussian bias)
  //   - Anchor previous-iteration views with a minimum floor weight
  //   - Use a FIXED cost threshold (no iteration decay)
  //
  // This ensures views which consistently validate across spatial
  // neighbors get weight, but views that disagree are NOT suppressed.
  // =================================================================

  // 2a. Count per-view validations across the 8 spatial candidates
  float view_weights[MAX_SOURCES];
  for (int i = 0; i < MAX_SOURCES; i++) view_weights[i] = 0.0f;

  // Fixed threshold: a view validates a candidate if cost < 0.6
  // (i.e. NCC > 0.2).  No iteration-dependent decay.
  const float validation_threshold = 0.8f;

  for (int i = 0; i < n_src; i++) {
    float valid_count = 0.0f;
    for (int j = 0; j < 8; j++) {
      if (flags[j] && cost_array[j * MAX_SOURCES + i] < validation_threshold) {
        valid_count += 1.0f;
      }
    }
    // Weight = number of candidates this view validates.
    if (valid_count > 0.0f) {
      view_weights[i] = valid_count;
    }
  }

  // 2b. Anchoring: forcibly include views from the previous iteration.
  uint prev_sel = selected_views[idx];
  if (anchor_views > 0) {
    int anchored = 0;
    for (int i = 0; i < n_src && anchored < anchor_views; i++) {
      if ((prev_sel & (1u << i)) && view_weights[i] < 1e-6f) {
        view_weights[i] = 1.0f;
        anchored++;
      }
    }
  }

  // 2c. Multi-factor geometric priors (COLMAP-inspired).
  // Multiply each view's weight by:
  //   - Triangulation angle factor: angle_weights[i] = sin(baseline_angle)
  //     → promotes wide-baseline views, penalizes near-zero baselines.
  //   - Incident angle factor: exp(-(1-|cos_θ|)² / (2*σ²))
  //     → penalizes views that see the surface at grazing angles.
  //     Uses the current pixel's normal estimate so it adapts with depth.
  {
    // Get current normal from the existing plane hypothesis.
    PlaneHypothesis cur_plane = read_imagef(planes_img, samp_planes, (int2)(x, y));
    float3 cur_n = (float3)(cur_plane.x, cur_plane.y, cur_plane.z);
    float n_len = length(cur_n);
    bool have_normal = (n_len > 0.1f);
    if (have_normal) cur_n /= n_len;

    // Reference camera center: C0 = -R0^T * t0
    float3 C0;
    C0.x = -(cameras[0].R[0]*cameras[0].t[0] + cameras[0].R[3]*cameras[0].t[1] + cameras[0].R[6]*cameras[0].t[2]);
    C0.y = -(cameras[0].R[1]*cameras[0].t[0] + cameras[0].R[4]*cameras[0].t[1] + cameras[0].R[7]*cameras[0].t[2]);
    C0.z = -(cameras[0].R[2]*cameras[0].t[0] + cameras[0].R[5]*cameras[0].t[1] + cameras[0].R[8]*cameras[0].t[2]);

    // 3D point at current pixel using current depth.
    float cur_depth = depth_from_plane((float)x, (float)y, cur_plane, &cameras[0]);
    float3 point_3d = backproject((float)x, (float)y, cur_depth, &cameras[0]);

    const float inc_sigma = 0.6f;  // incident angle sigma (COLMAP default)
    const float inv_2sigma2 = -0.5f / (inc_sigma * inc_sigma);

    for (int i = 0; i < n_src; i++) {
      if (view_weights[i] < 1e-6f) continue;

      // (a) Triangulation angle factor: directly use precomputed sin(angle).
      float tri_factor = angle_weights[i];  // in [0.01, 1.0]
      view_weights[i] *= tri_factor;

      // (b) Incident angle factor: how frontal does source see the surface?
      if (have_normal && cur_depth > 0.0f) {
        // Source camera center: Ci = -Ri^T * ti
        float3 Ci;
        Ci.x = -(cameras[i+1].R[0]*cameras[i+1].t[0] + cameras[i+1].R[3]*cameras[i+1].t[1] + cameras[i+1].R[6]*cameras[i+1].t[2]);
        Ci.y = -(cameras[i+1].R[1]*cameras[i+1].t[0] + cameras[i+1].R[4]*cameras[i+1].t[1] + cameras[i+1].R[7]*cameras[i+1].t[2]);
        Ci.z = -(cameras[i+1].R[2]*cameras[i+1].t[0] + cameras[i+1].R[5]*cameras[i+1].t[1] + cameras[i+1].R[8]*cameras[i+1].t[2]);

        // Ray from 3D point to source camera.
        float3 ray_to_src = normalize(Ci - point_3d);
        float cos_inc = fabs(dot(cur_n, ray_to_src));
        // Penalize grazing views: factor → 0 as cos_inc → 0.
        float x_inc = 1.0f - cos_inc;
        float inc_factor = exp(x_inc * x_inc * inv_2sigma2);
        view_weights[i] *= inc_factor;
      }
    }
  }

  // 2d. Compute weight_norm and new_selected bitmask.
  uint new_selected = 0u;
  float weight_norm = 0.0f;
  for (int i = 0; i < n_src; i++) {
)CL"
    R"CL(
    if (view_weights[i] > 0.0f) {
      new_selected |= (1u << i);
      weight_norm += view_weights[i];
    }
  }

  // 2e. Fallback: if no views passed validation, use uniform weights
  // for all views that had valid projection (cost < 2.0).
  if (weight_norm < 1e-6f) {
    for (int i = 0; i < n_src; i++) {
      for (int j = 0; j < 8; j++) {
        if (flags[j] && cost_array[j * MAX_SOURCES + i] < 2.0f) {
          view_weights[i] = 1.0f;
          new_selected |= (1u << i);
          weight_norm += 1.0f;
          break;
        }
      }
    }
  }


  // =================================================================
  // Step 3: Compute view-weighted final costs for each candidate
  // =================================================================

  // Pre-fetch current pixel intensity for edge-gating (Perona-Malik).
  float cur_intensity = read_imagef(ref_img, samp_planes, (int2)(x, y)).x;

  // Variance gate: compute local intensity variance in a 3x3 patch.
  // If the patch is textureless (variance < threshold^2), propagation
  // from neighbors is unreliable — block it so random refinement can
  // find the true depth instead of copying leaked foreground.
  bool propagation_allowed = true;
  if (variance_gate > 0.0f) {
    float local_var = 0.0f;
    for (int vy = -1; vy <= 1; vy++) {
      for (int vx = -1; vx <= 1; vx++) {
        float v = read_imagef(ref_img, samp_planes, (int2)(x+vx, y+vy)).x;
        float d = v - cur_intensity;
        local_var += d * d;
      }
    }
    local_var *= (1.0f / 9.0f);
    propagation_allowed = (local_var >= variance_gate * variance_gate);
  }

  // =================================================================
  // Far-Propagation Gradient Gate: invalidate far candidates (1,3,5,7)
  // that cross a significant image edge between the candidate pixel
  // and the current pixel.  Sample 4 evenly-spaced points along the
  // line; if any has a large intensity jump, the candidate is blocked.
  // =================================================================
  if (far_gradient_threshold > 0.0f) {
    int far_indices[4] = {1, 3, 5, 7};
    for (int fi = 0; fi < 4; fi++) {
      int ci = far_indices[fi];
      if (!flags[ci]) continue;
      int cx = positions[ci] % width;
      int cy = positions[ci] / width;
      float max_grad = 0.0f;
      float prev_val = cur_intensity;
      for (int step = 1; step <= 4; step++) {
        float t = (float)step / 4.0f;
        float sx = (float)x + t * ((float)cx - (float)x);
        float sy = (float)y + t * ((float)cy - (float)y);
        float sv = read_imagef(ref_img, samp_planes, (int2)((int)(sx+0.5f), (int)(sy+0.5f))).x;
        float grad = fabs(sv - prev_val);
        if (grad > max_grad) max_grad = grad;
        prev_val = sv;
      }
      if (max_grad > far_gradient_threshold) {
        flags[ci] = false;
      }
    }
  }

  // =================================================================
  // Segment-boundary gate: block propagation from candidates that
  // belong to a different SLIC segment than the current pixel.
  // segment_labels[i] == -1 means segmentation is disabled.
  // =================================================================
  int cur_segment = segment_labels[idx];
  if (cur_segment >= 0) {
    for (int i = 0; i < 8; i++) {
      if (flags[i] && segment_labels[positions[i]] != cur_segment) {
        flags[i] = false;
      }
    }
  }

  float final_costs[8];
  for (int i = 0; i < 8; i++) {
    if (flags[i]) {
      PlaneHypothesis cand_plane = read_imagef(planes_img, samp_planes,
          (int2)(positions[i] % width, positions[i] / width));
      float photo = compute_mc_cost(&cost_array[i * MAX_SOURCES],
          view_weights, weight_norm, n_src,
          cameras, cand_plane, x, y,
          prev_depths, prev_depth_mask, width, height, geom_weight,
          &state, key + 600u + (uint)i * 31u);
      if (smooth_weight > 1e-6f) {
        float smooth = compute_smoothness_cost(planes_img, &cameras[0],
            cand_plane, x, y, width, height);
        photo += smooth_weight * smooth;
      }
      // Perona-Malik edge gate: penalize propagation across image edges.
      // sigma_color is the bilateral NCC color sigma — reuse it here.
      if (edge_weight > 1e-6f) {
        int cx = positions[i] % width;
        int cy = positions[i] / width;
        float nb_intensity = read_imagef(ref_img, samp_planes, (int2)(cx, cy)).x;
        float diff = cur_intensity - nb_intensity;
        float gate = exp(-diff * diff / (2.0f * sigma_color * sigma_color));
        // penalty in [0, edge_weight]: strong edge → full penalty
        photo += edge_weight * (1.0f - gate);
      }
      final_costs[i] = photo;
    } else {
      final_costs[i] = 2.0f;
    }
  }

  // Find best candidate
  int best_cand = 0;
  float best_cand_cost = final_costs[0];
  for (int i = 1; i < 8; i++) {
    if (final_costs[i] < best_cand_cost) {
      best_cand_cost = final_costs[i];
      best_cand = i;
    }
  }

  // Compute current pixel's cost with new view weights
  PlaneHypothesis plane_now = read_imagef(planes_img, samp_planes, (int2)(x, y));
  float cost_vec_now[MAX_SOURCES];
  for (int i = 0; i < MAX_SOURCES; i++) cost_vec_now[i] = 2.0f;
  compute_cost_vector(ref_img, cameras, num_images, x, y,
      plane_now, half_patch, sigma_spatial, sigma_color, census_weight,
      cost_vec_now,
      src_images, center_color_weight);
  float cost_now = compute_mc_cost(cost_vec_now, view_weights,
      weight_norm, n_src,
      cameras, plane_now, x, y,
      prev_depths, prev_depth_mask, width, height, geom_weight,
      &state, key + 700u);
  if (smooth_weight > 1e-6f) {
    cost_now += smooth_weight * compute_smoothness_cost(planes_img, &cameras[0],
        plane_now, x, y, width, height);
  }
  costs[idx] = cost_now;

  float depth_now = depth_from_plane((float)x, (float)y, plane_now, &cameras[0]);

  // =================================================================
  // Planar prior setup (hypothesis generation only)
  //
  // The prior provides diverse starting hypotheses near the Delaunay
  // surface (step 4), but acceptance is always pure photometric cost.
  // =================================================================
  bool have_prior = (plane_masks[idx] > 0u);
  float depth_prior = 0.0f;
  float3 prior_n = (float3)(0.0f, 0.0f, -1.0f);
  float depth_sigma = (depth_max - depth_min) / 32.0f;
  float angle_sigma_rad = M_PI_F * (5.0f / 180.0f);  // for hypothesis generation

  if (have_prior) {
    depth_prior = depth_from_plane((float)x, (float)y, prior_planes[idx], &cameras[0]);
    prior_n = normalize((float3)(prior_planes[idx].x, prior_planes[idx].y, prior_planes[idx].z));
  }

  // =================================================================
  // Step 3b: Accept best candidate (pure photometric cost)
  // =================================================================
  {
    if (propagation_allowed && flags[best_cand] && best_cand_cost < cost_now) {
      PlaneHypothesis best_cand_plane = read_imagef(planes_img, samp_planes,
          (int2)(positions[best_cand] % width, positions[best_cand] / width));
      float db = depth_from_plane((float)x, (float)y,
                                  best_cand_plane, &cameras[0]);
      if (db >= depth_min && db <= depth_max) {
        depth_now = db;
        plane_now = best_cand_plane;
        cost_now = best_cand_cost;
        selected_views[idx] = new_selected;
      }
    }
  }

  // =================================================================
  // Step 4: Multi-hypothesis refinement (5 diverse candidates)
  //
  // From Xu et al.: test 5 combinations of depth and normal:
  //   0: (random_depth,     current_normal)
  //   1: (current_depth,    random_normal)
  //   2: (random_depth,     random_normal)
  //   3: (current_depth,    perturbed_normal)
  //   4: (perturbed_depth,  current_normal)
  //
  // =================================================================
  {
    // Perturbation scales
    float depth_perturbation = 0.005f;
    float normal_perturbation = 0.01f * M_PI_F;
    float3 cur_normal = normalize((float3)(plane_now.x, plane_now.y, plane_now.z));

    float depth_rand, depth_perturbed;
    float3 normal_rand, normal_perturbed;

    // Generate random depth + normal.
    // When a prior exists, draw from a narrow Gaussian around the prior
    // (ACMMP: depth in [prior - 3*sigma, prior + 3*sigma], normal
    // perturbed from prior by angle_sigma = 5 degrees).
    if (have_prior && depth_prior > 0.0f) {
      depth_rand = depth_prior - 3.0f * depth_sigma
                 + rand_float(&state, key + 503u) * 6.0f * depth_sigma;
      depth_rand = clamp(depth_rand, depth_min, depth_max);
      float pa1 = (rand_float(&state, key + 504u) - 0.5f) * angle_sigma_rad;
      float pa2 = (rand_float(&state, key + 505u) - 0.5f) * angle_sigma_rad;
      normal_rand.x = prior_n.x + pa1;
      normal_rand.y = prior_n.y + pa2;
      normal_rand.z = prior_n.z;
      normal_rand = normalize(normal_rand);
      if (normal_rand.z > 0.0f) normal_rand.z = -normal_rand.z;
    } else {
      depth_rand = exp(log(max(depth_min, 1e-6f))
                   + rand_float(&state, key + 503u)
                   * (log(max(depth_max, 1e-5f)) - log(max(depth_min, 1e-6f))));
      normal_rand.x = rand_float(&state, key + 504u) * 2.0f - 1.0f;
      normal_rand.y = rand_float(&state, key + 505u) * 2.0f - 1.0f;
      normal_rand.z = -1.0f;
      normal_rand = normalize(normal_rand);
    }

    // Perturbed depth and normal (small perturbation of current)
    {
      float d_min_p = (1.0f - depth_perturbation) * depth_now;
      float d_max_p = (1.0f + depth_perturbation) * depth_now;
      depth_perturbed = d_min_p + rand_float(&state, key + 510u) * (d_max_p - d_min_p);
      depth_perturbed = clamp(depth_perturbed, depth_min, depth_max);
    }
    {
      float pa1 = (rand_float(&state, key + 520u) - 0.5f) * normal_perturbation;
      float pa2 = (rand_float(&state, key + 521u) - 0.5f) * normal_perturbation;
      normal_perturbed.x = cur_normal.x + pa1;
      normal_perturbed.y = cur_normal.y + pa2;
      normal_perturbed.z = cur_normal.z;
      normal_perturbed = normalize(normal_perturbed);
      if (normal_perturbed.z > 0.0f) normal_perturbed.z = -normal_perturbed.z;
    }

    // This acts as a geometric regularizer: biases normals toward being
    // consistent with the local depth surface, without explicit smoothness.
    float3 surface_normal = compute_surface_normal(planes_img, &cameras[0],
        x, y, width, height, depth_now);
    bool have_surface_normal = (length(surface_normal) > 0.5f);

    // 6 candidate hypotheses (5 original + surface normal from neighbors)
    //   0: (random_depth,     current_normal)
    //   1: (current_depth,    random_normal)
    //   2: (random_depth,     random_normal)
    //   3: (current_depth,    perturbed_normal)
    //   4: (perturbed_depth,  current_normal)
    //   5: (current_depth,    surface_normal)
    float  ref_depths[6]  = {depth_rand, depth_now, depth_rand,
                             depth_now, depth_perturbed, depth_now};
    float3 ref_normals[6];
    ref_normals[0] = cur_normal;
    ref_normals[1] = normal_rand;
    ref_normals[2] = normal_rand;
    ref_normals[3] = normal_perturbed;
    ref_normals[4] = cur_normal;
    ref_normals[5] = surface_normal;

    int n_cands = have_surface_normal ? 6 : 5;
    for (int h = 0; h < n_cands; h++) {
      PlaneHypothesis cand = plane_from_depth_normal(
          (float)x, (float)y, ref_depths[h], ref_normals[h], &cameras[0]);

      float cv[MAX_SOURCES];
      for (int ci = 0; ci < MAX_SOURCES; ci++) cv[ci] = 2.0f;
      compute_cost_vector(ref_img, cameras, num_images, x, y, cand,
          half_patch, sigma_spatial, sigma_color, census_weight,
          cv,
          src_images, center_color_weight);
      float temp_cost = compute_mc_cost(cv, view_weights,
          weight_norm, n_src,
          cameras, cand, x, y,
          prev_depths, prev_depth_mask, width, height, geom_weight,
          &state, key + 800u + (uint)h * 37u);
      if (smooth_weight > 1e-6f) {
        temp_cost += smooth_weight * compute_smoothness_cost(planes_img, &cameras[0],
            cand, x, y, width, height);
      }

      float depth_before = depth_from_plane((float)x, (float)y, cand, &cameras[0]);
      if (depth_before < depth_min || depth_before > depth_max) continue;

      if (temp_cost < cost_now) {
        depth_now = depth_before;
        plane_now = cand;
        cost_now = temp_cost;
      }
    }
  }

  // =================================================================
  // Step 5: Depth-discontinuity escape heuristic.
  //
  // If a direct neighbor has significantly deeper depth (by
  // escape_depth_ratio), the current pixel may be a background pixel
  // contaminated by foreground propagation.  Try the deeper neighbor's
  // plane as an escape candidate.
  // =================================================================
  if (escape_depth_ratio > 1.0f) {
    PlaneHypothesis nb_planes[4];
    nb_planes[0] = read_imagef(planes_img, samp_planes, (int2)(x - 1, y));
    nb_planes[1] = read_imagef(planes_img, samp_planes, (int2)(x + 1, y));
    nb_planes[2] = read_imagef(planes_img, samp_planes, (int2)(x, y - 1));
    nb_planes[3] = read_imagef(planes_img, samp_planes, (int2)(x, y + 1));

    float nb_depths[4];
    nb_depths[0] = depth_from_plane((float)(x-1), (float)y, nb_planes[0], &cameras[0]);
    nb_depths[1] = depth_from_plane((float)(x+1), (float)y, nb_planes[1], &cameras[0]);
    nb_depths[2] = depth_from_plane((float)x, (float)(y-1), nb_planes[2], &cameras[0]);
    nb_depths[3] = depth_from_plane((float)x, (float)(y+1), nb_planes[3], &cameras[0]);

    // Find the deepest valid neighbor.
    int best_nb = -1;
    float best_nb_depth = 0.0f;
    for (int ni = 0; ni < 4; ni++) {
      if (nb_depths[ni] > best_nb_depth) {
        best_nb_depth = nb_depths[ni];
        best_nb = ni;
      }
    }

    // Trigger escape if the deepest neighbor is significantly further.
    if (best_nb >= 0 && best_nb_depth > depth_now * escape_depth_ratio) {
      // Use the deep neighbor's plane evaluated at our pixel position.
      PlaneHypothesis esc_plane = nb_planes[best_nb];
      float esc_depth = depth_from_plane((float)x, (float)y, esc_plane, &cameras[0]);

      if (esc_depth >= depth_min && esc_depth <= depth_max) {
        float cv_esc[MAX_SOURCES];
        for (int ci = 0; ci < MAX_SOURCES; ci++) cv_esc[ci] = 2.0f;
        compute_cost_vector(ref_img, cameras, num_images, x, y, esc_plane,
            half_patch, sigma_spatial, sigma_color, census_weight,
            cv_esc, src_images, center_color_weight);
        float esc_cost = compute_mc_cost(cv_esc, view_weights,
            weight_norm, n_src,
            cameras, esc_plane, x, y,
            prev_depths, prev_depth_mask, width, height, geom_weight,
            &state, key + 900u);
        if (smooth_weight > 1e-6f) {
          esc_cost += smooth_weight * compute_smoothness_cost(planes_img,
              &cameras[0], esc_plane, x, y, width, height);
        }

        if (esc_cost < cost_now) {
          depth_now = esc_depth;
          plane_now = esc_plane;
          cost_now = esc_cost;
        }
      }
    }
  }

  planes[idx] = plane_now;
  costs[idx] = cost_now;
  rand_states[idx] = state;
})CL"
    R"CL(

// =====================================================================
// Kernel: Upsample plane hypotheses from a coarser resolution.
//
// For valid coarse pixels, the plane is recomputed at the fine pixel
// position and the cost is set to 2.0 so PatchMatch re-evaluates it
// at the fine resolution (coarse-level NCC costs are artificially low
// due to less high-frequency detail).
//
// For invalid coarse pixels (depth <= 0), a random plane is generated
// instead — otherwise those pixels would start with garbage data.
// =====================================================================
__kernel void acmmp_upsample(
    __global PlaneHypothesis *dst_planes,
    __global float *dst_costs,
    __global const PlaneHypothesis *src_planes,
    __global const float *src_costs,
    __global const Camera *cameras,
    __global uint2 *rand_states,
    int dst_width, int dst_height,
    int src_width, int src_height,
    float depth_min, float depth_max
) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  if (x >= dst_width || y >= dst_height) return;

  int dst_idx = y * dst_width + x;

  // Always initialise PRNG state for every pixel so that subsequent
  // PatchMatch iterations have well-defined random numbers.  Without
  // this, valid-upsampled pixels would keep whatever garbage was in
  // the (newly allocated) rand_states buffer.
  uint2 state = (uint2)(dst_idx * 1099087573u + 2654435769u,
                        dst_idx * 2246822519u + 13u);
  rand_states[dst_idx] = state;

  // Find corresponding coarse pixel (nearest-neighbor)
  float scale_x = (float)src_width / (float)dst_width;
  float scale_y = (float)src_height / (float)dst_height;
  int sx = (int)((float)x * scale_x);
  int sy = (int)((float)y * scale_y);
  sx = clamp(sx, 0, src_width - 1);
  sy = clamp(sy, 0, src_height - 1);
  int src_idx = sy * src_width + sx;

  PlaneHypothesis plane = src_planes[src_idx];

  // Recompute depth at the high-res pixel position
  float depth = depth_from_plane((float)x, (float)y, plane, &cameras[0]);

  if (depth > 0.0f) {
    // Valid coarse pixel: upsample the plane
    float3 normal = normalize((float3)(plane.x, plane.y, plane.z));
    dst_planes[dst_idx] = plane_from_depth_normal(
        (float)x, (float)y, depth, normal, &cameras[0]);
  } else {
    // Invalid coarse pixel: random initialisation
    uint key = (uint)(dst_idx + 7919);
    float log_d_min = log(max(depth_min, 1e-6f));
    float log_d_max = log(max(depth_max, 1e-5f));
    float rand_depth = exp(log_d_min + rand_float(&state, key) * (log_d_max - log_d_min));
    float3 normal;
    normal.x = rand_float(&state, key + 3u) * 2.0f - 1.0f;
    normal.y = rand_float(&state, key + 5u) * 2.0f - 1.0f;
    normal.z = -1.0f;
    normal = normalize(normal);
    dst_planes[dst_idx] = plane_from_depth_normal(
        (float)x, (float)y, rand_depth, normal, &cameras[0]);
    rand_states[dst_idx] = state;
  }

  // Always force PatchMatch to re-evaluate at the fine resolution.
  // Coarse-level NCC costs are artificially low and would prevent
  // PatchMatch from replacing wrong hypotheses.
  dst_costs[dst_idx] = 2.0f;
}

// =====================================================================
// Helper: insertion sort for small float arrays.
// =====================================================================
void sort_small_f(float *d, int n) {
  for (int i = 1; i < n; i++) {
    float tmp = d[i];
    int j;
    for (j = i; j >= 1 && tmp < d[j - 1]; j--)
      d[j] = d[j - 1];
    d[j] = tmp;
  }
}

// =====================================================================
// Kernel: Red-black checkerboard median filter (21 neighbours).
//
// After each PatchMatch red-black iteration, smooth depth estimates by
// replacing each pixel's depth with the median of up to 21 neighbours
// sampled in a cross + diagonal checkerboard pattern (all neighbours
// are of the opposite colour, so there are no read-write conflicts
// within a single colour pass).
//
// Ported from CheckerboardFilter in Xu et al. (TPAMI 2022).
// color_flag: 0 = red (x+y even), 1 = black (x+y odd).
// =====================================================================
__kernel void acmmp_checkerboard_filter(
    __global PlaneHypothesis *planes,
    __global float *costs,
    read_only image2d_t planes_img,
    read_only image2d_t costs_img,
    __global const Camera *cameras,
    int width, int height,
    int color_flag
) {
  // Dense checkerboard indexing: grid is (half_width x height), reconstruct
  // true pixel coordinates so all SIMD lanes are active on Apple Silicon.
  int gid_x = get_global_id(0);
  int gid_y = get_global_id(1);
  int half_width = (width + 1) / 2;
  if (gid_x >= half_width || gid_y >= height) return;
  int x = gid_x * 2 + ((gid_y + color_flag) % 2);
  int y = gid_y;
  if (x >= width) return;

  int center = y * width + x;

  // Skip pixels with very low cost (already well converged).
  if (read_imagef(costs_img, samp_planes, (int2)(x, y)).x < 0.001f) return;

  float filter_buf[21];
  int idx = 0;

  // Center pixel depth.
  filter_buf[idx++] = depth_from_plane((float)x, (float)y,
      read_imagef(planes_img, samp_planes, (int2)(x, y)), &cameras[0]);

  // --- Vertical axis (stride 1, 3, 5 in y) ---
  if (y > 0) {
    filter_buf[idx++] = depth_from_plane((float)x, (float)(y - 1),
        read_imagef(planes_img, samp_planes, (int2)(x, y - 1)), &cameras[0]);
  }
  if (y > 2) {
    filter_buf[idx++] = depth_from_plane((float)x, (float)(y - 3),
        read_imagef(planes_img, samp_planes, (int2)(x, y - 3)), &cameras[0]);
  }
  if (y > 4) {
    filter_buf[idx++] = depth_from_plane((float)x, (float)(y - 5),
        read_imagef(planes_img, samp_planes, (int2)(x, y - 5)), &cameras[0]);
  }
  if (y < height - 1) {
    filter_buf[idx++] = depth_from_plane((float)x, (float)(y + 1),
        read_imagef(planes_img, samp_planes, (int2)(x, y + 1)), &cameras[0]);
  }
  if (y < height - 3) {
    filter_buf[idx++] = depth_from_plane((float)x, (float)(y + 3),
        read_imagef(planes_img, samp_planes, (int2)(x, y + 3)), &cameras[0]);
  }
  if (y < height - 5) {
    filter_buf[idx++] = depth_from_plane((float)x, (float)(y + 5),
        read_imagef(planes_img, samp_planes, (int2)(x, y + 5)), &cameras[0]);
  }

  // --- Horizontal axis (stride 1, 3, 5 in x) ---
  if (x > 0) {
    filter_buf[idx++] = depth_from_plane((float)(x - 1), (float)y,
        read_imagef(planes_img, samp_planes, (int2)(x - 1, y)), &cameras[0]);
  }
  if (x > 2) {
    filter_buf[idx++] = depth_from_plane((float)(x - 3), (float)y,
        read_imagef(planes_img, samp_planes, (int2)(x - 3, y)), &cameras[0]);
  }
  if (x > 4) {
    filter_buf[idx++] = depth_from_plane((float)(x - 5), (float)y,
        read_imagef(planes_img, samp_planes, (int2)(x - 5, y)), &cameras[0]);
  }
  if (x < width - 1) {
    filter_buf[idx++] = depth_from_plane((float)(x + 1), (float)y,
        read_imagef(planes_img, samp_planes, (int2)(x + 1, y)), &cameras[0]);
  }
  if (x < width - 3) {
    filter_buf[idx++] = depth_from_plane((float)(x + 3), (float)y,
        read_imagef(planes_img, samp_planes, (int2)(x + 3, y)), &cameras[0]);
  }
  if (x < width - 5) {
    filter_buf[idx++] = depth_from_plane((float)(x + 5), (float)y,
        read_imagef(planes_img, samp_planes, (int2)(x + 5, y)), &cameras[0]);
  }

  // --- Diagonal-ish neighbours (checkerboard L-shaped offsets) ---
  if (y > 0 && x < width - 2) {
    filter_buf[idx++] = depth_from_plane((float)(x + 2), (float)(y - 1),
        read_imagef(planes_img, samp_planes, (int2)(x + 2, y - 1)), &cameras[0]);
  }
  if (y < height - 1 && x < width - 2) {
    filter_buf[idx++] = depth_from_plane((float)(x + 2), (float)(y + 1),
        read_imagef(planes_img, samp_planes, (int2)(x + 2, y + 1)), &cameras[0]);
  }
  if (y > 0 && x > 1) {
    filter_buf[idx++] = depth_from_plane((float)(x - 2), (float)(y - 1),
        read_imagef(planes_img, samp_planes, (int2)(x - 2, y - 1)), &cameras[0]);
  }
  if (y < height - 1 && x > 1) {
    filter_buf[idx++] = depth_from_plane((float)(x - 2), (float)(y + 1),
        read_imagef(planes_img, samp_planes, (int2)(x - 2, y + 1)), &cameras[0]);
  }
  if (x > 0 && y > 2) {
    filter_buf[idx++] = depth_from_plane((float)(x - 1), (float)(y - 2),
        read_imagef(planes_img, samp_planes, (int2)(x - 1, y - 2)), &cameras[0]);
  }
  if (x < width - 1 && y > 2) {
    filter_buf[idx++] = depth_from_plane((float)(x + 1), (float)(y - 2),
        read_imagef(planes_img, samp_planes, (int2)(x + 1, y - 2)), &cameras[0]);
  }
  if (x > 0 && y < height - 2) {
    filter_buf[idx++] = depth_from_plane((float)(x - 1), (float)(y + 2),
        read_imagef(planes_img, samp_planes, (int2)(x - 1, y + 2)), &cameras[0]);
  }
  if (x < width - 1 && y < height - 2) {
    filter_buf[idx++] = depth_from_plane((float)(x + 1), (float)(y + 2),
        read_imagef(planes_img, samp_planes, (int2)(x + 1, y + 2)), &cameras[0]);
  }

  // Bilateral depth gate: reject neighbors whose depth differs by more
  // than 5% from the center pixel.  At foreground-background transitions
  // this prevents the median from mixing depths from different surfaces,
  // which would otherwise crystallize wrong-depth blobs.
  {
    float percent = 0.03f;  // // was 0.15f
    float center_d = filter_buf[0];
    if (center_d > 0.0f) {
      float depth_lo = center_d * (1 - percent);
      float depth_hi = center_d * (1 + percent);
      int new_idx = 1;  // always keep center
      for (int i = 1; i < idx; i++) {
        if (filter_buf[i] > depth_lo && filter_buf[i] < depth_hi) {
          filter_buf[new_idx++] = filter_buf[i];
        }
      }
      idx = new_idx;
    }
  }

  // Insertion-sort and pick median.
  sort_small_f(filter_buf, idx);
  int mid = idx / 2;
  float median_depth;
  if (idx % 2 == 0) {
    median_depth = (filter_buf[mid - 1] + filter_buf[mid]) * 0.5f;
  } else {
    median_depth = filter_buf[mid];
  }

  // Reconstruct plane with center's normal but the median depth.
  if (median_depth > 0.0f) {
    PlaneHypothesis center_plane = read_imagef(planes_img, samp_planes, (int2)(x, y));
    float3 normal = normalize((float3)(center_plane.x,
                                       center_plane.y,
                                       center_plane.z));
    planes[center] = plane_from_depth_normal(
        (float)x, (float)y, median_depth, normal, &cameras[0]);
  }
}
)CL";

// =====================================================================
// SLIC superpixel segmentation kernel source.
//
// Three kernels implementing iterative SLIC on a grayscale image:
//   1. slic_init_centers: place initial centers on a regular grid
//   2. slic_assign_pixels: assign each pixel to nearest center (within 2S)
//   3. slic_update_centers: recompute center position + color as mean
//
// After convergence, segment labels are used to gate PatchMatch
// propagation — hypotheses cannot cross segment boundaries.
// =====================================================================
inline const char* kSLICKernelSource = R"CL(

// SLIC center: (x, y, intensity)
typedef struct {
  float x;
  float y;
  float intensity;
  int count;  // number of assigned pixels (for update)
} SLICCenter;

// =====================================================================
// Kernel: Initialize SLIC centers on a regular grid.
// Centers are placed at (S/2 + i*S, S/2 + j*S), sampling the image
// intensity at that position.
// =====================================================================
__kernel void slic_init_centers(
    __global SLICCenter *centers,
    read_only image2d_t img,
    int width, int height,
    int grid_step,
    int centers_x, int centers_y
) {
  int cx = get_global_id(0);
  int cy = get_global_id(1);
  if (cx >= centers_x || cy >= centers_y) return;

  int center_idx = cy * centers_x + cx;

  int px = grid_step / 2 + cx * grid_step;
  int py = grid_step / 2 + cy * grid_step;
  px = min(px, width - 1);
  py = min(py, height - 1);

  const sampler_t samp = CLK_NORMALIZED_COORDS_FALSE
                       | CLK_ADDRESS_CLAMP_TO_EDGE
                       | CLK_FILTER_NEAREST;

  float val = read_imagef(img, samp, (int2)(px, py)).x;

  centers[center_idx].x = (float)px;
  centers[center_idx].y = (float)py;
  centers[center_idx].intensity = val;
  centers[center_idx].count = 0;
}

// =====================================================================
// Kernel: Assign each pixel to its nearest center within a 2S window.
// Distance = sqrt((dx/S)^2 + (dy/S)^2 + (di*m/S)^2)  [SLIC distance]
// =====================================================================
__kernel void slic_assign_pixels(
    __global const SLICCenter *centers,
    __global int *labels,
    read_only image2d_t img,
    int width, int height,
    int grid_step,
    int centers_x, int centers_y,
    float compactness
) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  if (x >= width || y >= height) return;

  const sampler_t samp = CLK_NORMALIZED_COORDS_FALSE
                       | CLK_ADDRESS_CLAMP_TO_EDGE
                       | CLK_FILTER_NEAREST;

  float pixel_val = read_imagef(img, samp, (int2)(x, y)).x;

  // Which center cell does this pixel belong to?
  int cell_x = x / grid_step;
  int cell_y = y / grid_step;

  float best_dist = 1e10f;
  int best_label = 0;

  float inv_s = 1.0f / (float)grid_step;
  float m_over_s = compactness * inv_s;

  // Search 3x3 neighborhood of center cells
  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      int ncx = cell_x + dx;
      int ncy = cell_y + dy;
      if (ncx < 0 || ncx >= centers_x || ncy < 0 || ncy >= centers_y) continue;

      int ci = ncy * centers_x + ncx;
      float cx_pos = centers[ci].x;
      float cy_pos = centers[ci].y;
      float ci_val = centers[ci].intensity;

      float spatial_x = ((float)x - cx_pos) * inv_s;
      float spatial_y = ((float)y - cy_pos) * inv_s;
      float color_d = (pixel_val - ci_val) * m_over_s;

      float dist = spatial_x * spatial_x + spatial_y * spatial_y + color_d * color_d;

      if (dist < best_dist) {
        best_dist = dist;
        best_label = ci;
      }
    }
  }

  labels[y * width + x] = best_label;
}

// =====================================================================
// Kernel: Update centers by scanning their 2S×2S neighborhood and
// averaging all assigned pixels.  No atomics needed — each center
// independently scans its local region.
// =====================================================================
__kernel void slic_update_centers(
    __global SLICCenter *centers,
    __global const int *labels,
    read_only image2d_t img,
    int width, int height,
    int grid_step,
    int centers_x, int centers_y
) {
  int cx = get_global_id(0);
  int cy = get_global_id(1);
  if (cx >= centers_x || cy >= centers_y) return;

  int center_idx = cy * centers_x + cx;

  const sampler_t samp = CLK_NORMALIZED_COORDS_FALSE
                       | CLK_ADDRESS_CLAMP_TO_EDGE
                       | CLK_FILTER_NEAREST;

  // Scan 2S × 2S window around the current center position
  int cur_x = (int)(centers[center_idx].x + 0.5f);
  int cur_y = (int)(centers[center_idx].y + 0.5f);

  int x_min = max(0, cur_x - grid_step);
  int x_max = min(width - 1, cur_x + grid_step);
  int y_min = max(0, cur_y - grid_step);
  int y_max = min(height - 1, cur_y + grid_step);

  float sum_x = 0.0f, sum_y = 0.0f, sum_val = 0.0f;
  int count = 0;

  for (int py = y_min; py <= y_max; py++) {
    for (int px = x_min; px <= x_max; px++) {
      if (labels[py * width + px] == center_idx) {
        float val = read_imagef(img, samp, (int2)(px, py)).x;
        sum_x += (float)px;
        sum_y += (float)py;
        sum_val += val;
        count++;
      }
    }
  }

  if (count > 0) {
    centers[center_idx].x = sum_x / (float)count;
    centers[center_idx].y = sum_y / (float)count;
    centers[center_idx].intensity = sum_val / (float)count;
    centers[center_idx].count = count;
  }
}

)CL";

// =====================================================================
// Mahalanobis segment-aware depth filtering kernel.
//
// For each pixel with depth > 0:
//   1. Gather same-segment neighbors within a window
//   2. Backproject all gathered points to 3D
//   3. Compute robust mean + covariance (MAD-trimmed)
//   4. Compute Mahalanobis distance of center pixel's 3D point
//   5. Zero out if distance exceeds threshold
//
// This rejects points that don't belong to the local planar/ellipsoidal
// distribution of their segment — catching foreground smear that
// survived cross-view cleaning.
// =====================================================================
inline const char* kMahalanobisKernelSource = R"CL(

// Maximum points to gather in the window (bounded for register use).
#define MAHAL_MAX_GATHER 128

__kernel void mahalanobis_filter(
    __global const float *depth_in,   // input cleaned depth (W*H)
    __global float *depth_out,        // output filtered depth (W*H)
    __global const int *labels,       // SLIC segment labels (W*H)
    float fx, float fy, float cx, float cy,  // intrinsics
    int width, int height,
    int window_radius,                // half-size of gather window
    float mahal_threshold             // Mahalanobis distance threshold
) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  if (x >= width || y >= height) return;

  int idx = y * width + x;
  float d = depth_in[idx];

  // Pass-through for invalid pixels.
  if (d <= 0.0f) {
    depth_out[idx] = 0.0f;
    return;
  }

  int my_label = labels[idx];

  // Backproject center pixel to 3D.
  float3 center_pt;
  center_pt.x = d * ((float)x - cx) / fx;
  center_pt.y = d * ((float)y - cy) / fy;
  center_pt.z = d;

  // Gather same-segment neighbors.
  float3 pts[MAHAL_MAX_GATHER];
  int n_pts = 0;

  int x_min = max(0, x - window_radius);
  int x_max = min(width - 1, x + window_radius);
  int y_min = max(0, y - window_radius);
  int y_max = min(height - 1, y + window_radius);

  for (int ny = y_min; ny <= y_max && n_pts < MAHAL_MAX_GATHER; ny++) {
    for (int nx = x_min; nx <= x_max && n_pts < MAHAL_MAX_GATHER; nx++) {
      int ni = ny * width + nx;
      if (labels[ni] != my_label) continue;
      float nd = depth_in[ni];
      if (nd <= 0.0f) continue;
      pts[n_pts].x = nd * ((float)nx - cx) / fx;
      pts[n_pts].y = nd * ((float)ny - cy) / fy;
      pts[n_pts].z = nd;
      n_pts++;
    }
  }

  // Need minimum 4 points for a meaningful 3D covariance.
  if (n_pts < 4) {
    depth_out[idx] = d;
    return;
  }

  // Compute mean.
  float3 mean = (float3)(0.0f, 0.0f, 0.0f);
  for (int i = 0; i < n_pts; i++) {
    mean += pts[i];
  }
  mean /= (float)n_pts;

  // Compute initial residuals for MAD-based trimming.
  float residuals[MAHAL_MAX_GATHER];
  for (int i = 0; i < n_pts; i++) {
    float3 diff = pts[i] - mean;
    residuals[i] = dot(diff, diff);
  }

  // Simple MAD-trimming: find median residual, keep only points within
  // 3× median.  Use partial sort (find ~50th percentile via selection).
  // Approximate: use the n_pts/2-th smallest via bubble partial sort.
  float med_res = 0.0f;
  {
    int mid = n_pts / 2;
    // Partial selection: find mid-th smallest element.
    for (int i = 0; i <= mid; i++) {
      int min_idx = i;
      for (int j = i + 1; j < n_pts; j++) {
        if (residuals[j] < residuals[min_idx]) min_idx = j;
      }
      // Swap residuals and pts.
      float tmp_r = residuals[i];
      residuals[i] = residuals[min_idx];
      residuals[min_idx] = tmp_r;
      float3 tmp_p = pts[i];
      pts[i] = pts[min_idx];
      pts[min_idx] = tmp_p;
    }
    med_res = residuals[mid];
  }

  // Trim: keep only points with residual <= 3 * median.
  float trim_threshold = 3.0f * med_res;
  if (trim_threshold < 1e-12f) trim_threshold = 1e-12f;
  int n_inliers = 0;
  for (int i = 0; i < n_pts; i++) {
    if (residuals[i] <= trim_threshold) {
      pts[n_inliers] = pts[i];
      n_inliers++;
    }
  }

  if (n_inliers < 4) {
    depth_out[idx] = d;
    return;
  }

  // Recompute mean from inliers.
  mean = (float3)(0.0f, 0.0f, 0.0f);
  for (int i = 0; i < n_inliers; i++) {
    mean += pts[i];
  }
  mean /= (float)n_inliers;

  // Compute 3×3 covariance matrix (symmetric).
  // C = [cxx cxy cxz; cxy cyy cyz; cxz cyz czz]
  float cxx = 0.0f, cxy = 0.0f, cxz = 0.0f;
  float cyy = 0.0f, cyz = 0.0f, czz = 0.0f;
  for (int i = 0; i < n_inliers; i++) {
    float3 diff = pts[i] - mean;
    cxx += diff.x * diff.x;
    cxy += diff.x * diff.y;
    cxz += diff.x * diff.z;
    cyy += diff.y * diff.y;
    cyz += diff.y * diff.z;
    czz += diff.z * diff.z;
  }
  float inv_n = 1.0f / (float)(n_inliers - 1);
  cxx *= inv_n; cxy *= inv_n; cxz *= inv_n;
  cyy *= inv_n; cyz *= inv_n; czz *= inv_n;

  // Add small regularization for numerical stability.
  float reg = 1e-8f * (cxx + cyy + czz);
  cxx += reg; cyy += reg; czz += reg;

  // Invert 3×3 symmetric matrix via cofactors.
  float det = cxx * (cyy * czz - cyz * cyz)
            - cxy * (cxy * czz - cyz * cxz)
            + cxz * (cxy * cyz - cyy * cxz);

  if (fabs(det) < 1e-20f) {
    // Degenerate covariance — keep the pixel.
    depth_out[idx] = d;
    return;
  }
  float inv_det = 1.0f / det;

  // Inverse covariance (cofactor matrix / det).
  float ixx = (cyy * czz - cyz * cyz) * inv_det;
  float ixy = (cxz * cyz - cxy * czz) * inv_det;
  float ixz = (cxy * cyz - cxz * cyy) * inv_det;
  float iyy = (cxx * czz - cxz * cxz) * inv_det;
  float iyz = (cxy * cxz - cxx * cyz) * inv_det;
  float izz = (cxx * cyy - cxy * cxy) * inv_det;

  // Mahalanobis distance² of center pixel.
  float3 diff = center_pt - mean;
  float mahal_sq = diff.x * (ixx * diff.x + ixy * diff.y + ixz * diff.z)
                 + diff.y * (ixy * diff.x + iyy * diff.y + iyz * diff.z)
                 + diff.z * (ixz * diff.x + iyz * diff.y + izz * diff.z);

  // Reject if Mahalanobis distance exceeds threshold.
  float threshold_sq = mahal_threshold * mahal_threshold;
  if (mahal_sq > threshold_sq) {
    depth_out[idx] = 0.0f;
  } else {
    depth_out[idx] = d;
  }
}

)CL";

// =====================================================================
// Depthmap cleaning kernel: cross-view depth consistency check.
//
// For each pixel in the reference depth, backproject to 3D, project
// into each neighbor view, and check depth agreement.  Pixels without
// enough consistent neighbor views are zeroed out.
//
// The neighbor depth maps are stored as a flat buffer:
//   clean_depths[view_idx * width * height + y * width + x]
// where view_idx 0 = reference, 1..N-1 = neighbors.
// =====================================================================
inline const char* kCleanKernelSource = R"CL(

#define MAX_CLEAN_SOURCES 16

typedef struct {
  float K[9];
  float R[9];
  float t[3];
  int width;
  int height;
} Camera;

// Nearest-neighbor sampler for the reference depth (pixel-aligned reads).
__constant sampler_t samp_nn = CLK_NORMALIZED_COORDS_FALSE
                             | CLK_ADDRESS_CLAMP_TO_EDGE
                             | CLK_FILTER_NEAREST;

// Bilinear sampler for neighbor depth (sub-pixel reprojection).
__constant sampler_t samp_lin = CLK_NORMALIZED_COORDS_FALSE
                              | CLK_ADDRESS_CLAMP_TO_EDGE
                              | CLK_FILTER_LINEAR;

// Read neighbor depth from image2d_t by source index (0-based).
// Uses bilinear sampling with +0.5 pixel-center offset.
float read_src_depth(int src_idx, float u, float v,
                     read_only image2d_t s0,  read_only image2d_t s1,
                     read_only image2d_t s2,  read_only image2d_t s3,
                     read_only image2d_t s4,  read_only image2d_t s5,
                     read_only image2d_t s6,  read_only image2d_t s7,
                     read_only image2d_t s8,  read_only image2d_t s9,
                     read_only image2d_t s10, read_only image2d_t s11,
                     read_only image2d_t s12, read_only image2d_t s13,
                     read_only image2d_t s14, read_only image2d_t s15) {
  float2 coord = (float2)(u + 0.5f, v + 0.5f);
  switch (src_idx) {
    case  0: return read_imagef(s0,  samp_lin, coord).x;
    case  1: return read_imagef(s1,  samp_lin, coord).x;
    case  2: return read_imagef(s2,  samp_lin, coord).x;
    case  3: return read_imagef(s3,  samp_lin, coord).x;
    case  4: return read_imagef(s4,  samp_lin, coord).x;
    case  5: return read_imagef(s5,  samp_lin, coord).x;
    case  6: return read_imagef(s6,  samp_lin, coord).x;
    case  7: return read_imagef(s7,  samp_lin, coord).x;
    case  8: return read_imagef(s8,  samp_lin, coord).x;
    case  9: return read_imagef(s9,  samp_lin, coord).x;
    case 10: return read_imagef(s10, samp_lin, coord).x;
    case 11: return read_imagef(s11, samp_lin, coord).x;
    case 12: return read_imagef(s12, samp_lin, coord).x;
    case 13: return read_imagef(s13, samp_lin, coord).x;
    case 14: return read_imagef(s14, samp_lin, coord).x;
    default: return read_imagef(s15, samp_lin, coord).x;
  }
}

__kernel void acmmp_clean_depthmap(
    read_only image2d_t ref_depth_img,
    read_only image2d_t src0,  read_only image2d_t src1,
    read_only image2d_t src2,  read_only image2d_t src3,
    read_only image2d_t src4,  read_only image2d_t src5,
    read_only image2d_t src6,  read_only image2d_t src7,
    read_only image2d_t src8,  read_only image2d_t src9,
    read_only image2d_t src10, read_only image2d_t src11,
    read_only image2d_t src12, read_only image2d_t src13,
    read_only image2d_t src14, read_only image2d_t src15,
    __global const Camera *cameras,
    __global float *clean_depth,
    __global const float *ref_normal,  // HxWx3 camera-frame normals (or NULL-equivalent)
    int width, int height,
    int num_views,
    float same_depth_threshold,
    int min_consistent_views,
    float carving_threshold,
    int max_carved_views,
    float grazing_cos_threshold,
    float edge_depth_ratio,
    int has_normal
) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  if (x >= width || y >= height) return;

  int idx = y * width + x;

  float ref_depth = read_imagef(ref_depth_img, samp_nn, (int2)(x, y)).x;
  if (!(ref_depth > 0.0f)) {
    clean_depth[idx] = 0.0f;
    return;
  }

  // ---- Edge detection: check depth discontinuity in 3x3 neighborhood ----
  float d_min = ref_depth;
  float d_max = ref_depth;
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      if (dx == 0 && dy == 0) continue;
      int nx = x + dx;
      int ny = y + dy;
      if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
      float nd = read_imagef(ref_depth_img, samp_nn, (int2)(nx, ny)).x;
      if (nd > 0.0f) {
        d_min = fmin(d_min, nd);
        d_max = fmax(d_max, nd);
      }
    }
  }
  int is_edge = (d_min > 1e-6f && d_max / d_min > edge_depth_ratio) ? 1 : 0;

  // ---- Grazing angle detection from per-pixel normal ----
  int is_grazing = 0;
  if (has_normal) {
    float nnx = ref_normal[idx * 3];
    float nny = ref_normal[idx * 3 + 1];
    float nnz = ref_normal[idx * 3 + 2];
    float nlen = sqrt(nnx*nnx + nny*nny + nnz*nnz);
    if (nlen > 1e-6f) {
      // Normal is in camera frame; view ray in camera frame is (cam_x, cam_y, cam_z)/|...|
      Camera ref_cam_s = cameras[0];
      float fcx = ref_cam_s.K[0], ccx = ref_cam_s.K[2];
      float fcy = ref_cam_s.K[4], ccy = ref_cam_s.K[5];
      float ray_x = (x - ccx) / fcx;
      float ray_y = (y - ccy) / fcy;
      float ray_z = 1.0f;
      float rlen = sqrt(ray_x*ray_x + ray_y*ray_y + ray_z*ray_z);
      float cos_angle = fabs((nnx*ray_x + nny*ray_y + nnz*ray_z) / (nlen * rlen));
      is_grazing = (cos_angle < grazing_cos_threshold) ? 1 : 0;
    }
  }

  int is_suspicious = is_edge | is_grazing;

  Camera ref_cam = cameras[0];
  float fx = ref_cam.K[0], cx = ref_cam.K[2];
  float fy = ref_cam.K[4], cy = ref_cam.K[5];

  float cam_x = (x - cx) / fx * ref_depth;
  float cam_y = (y - cy) / fy * ref_depth;
  float cam_z = ref_depth;

  float pcam_x = cam_x - ref_cam.t[0];
  float pcam_y = cam_y - ref_cam.t[1];
  float pcam_z = cam_z - ref_cam.t[2];

  float wx = ref_cam.R[0]*pcam_x + ref_cam.R[3]*pcam_y + ref_cam.R[6]*pcam_z;
  float wy = ref_cam.R[1]*pcam_x + ref_cam.R[4]*pcam_y + ref_cam.R[7]*pcam_z;
  float wz = ref_cam.R[2]*pcam_x + ref_cam.R[5]*pcam_y + ref_cam.R[8]*pcam_z;

  int consistent = 1;
  int carved = 0;
  int n_src = min(num_views - 1, MAX_CLEAN_SOURCES);

  for (int v = 0; v < n_src; ++v) {
    Camera src_cam = cameras[v + 1];

    float sx = src_cam.R[0]*wx + src_cam.R[1]*wy + src_cam.R[2]*wz + src_cam.t[0];
    float sy = src_cam.R[3]*wx + src_cam.R[4]*wy + src_cam.R[5]*wz + src_cam.t[1];
    float sz = src_cam.R[6]*wx + src_cam.R[7]*wy + src_cam.R[8]*wz + src_cam.t[2];

    if (sz <= 1e-6f) continue;

    float su = src_cam.K[0] * sx / sz + src_cam.K[2];
    float sv_coord = src_cam.K[4] * sy / sz + src_cam.K[5];

    // Bounds check (1px margin for bilinear tap).
    if (su < 0.0f || sv_coord < 0.0f ||
        su >= (float)(width - 1) || sv_coord >= (float)(height - 1)) continue;

    // Hardware bilinear depth read via texture cache.
    float src_depth = read_src_depth(v, su, sv_coord,
        src0, src1, src2, src3, src4, src5, src6, src7,
        src8, src9, src10, src11, src12, src13, src14, src15);

    if (!(src_depth > 0.0f)) continue;

    // ---- Space carving: source sees significantly further ----
    if (src_depth > sz * (1.0f + carving_threshold)) {
      carved++;
      continue;
    }

    // Forward depth check.
    if (fabs(src_depth - sz) >= sz * same_depth_threshold) continue;

    // === Backward geometric consistency check ===
    float src_fx = src_cam.K[0], src_cx = src_cam.K[2];
    float src_fy = src_cam.K[4], src_cy = src_cam.K[5];

    float src_cam_x = (su - src_cx) / src_fx * src_depth;
    float src_cam_y = (sv_coord - src_cy) / src_fy * src_depth;
    float src_cam_z = src_depth;

    float spc_x = src_cam_x - src_cam.t[0];
    float spc_y = src_cam_y - src_cam.t[1];
    float spc_z = src_cam_z - src_cam.t[2];

    float wx2 = src_cam.R[0]*spc_x + src_cam.R[3]*spc_y + src_cam.R[6]*spc_z;
    float wy2 = src_cam.R[1]*spc_x + src_cam.R[4]*spc_y + src_cam.R[7]*spc_z;
    float wz2 = src_cam.R[2]*spc_x + src_cam.R[5]*spc_y + src_cam.R[8]*spc_z;

    float rx = ref_cam.R[0]*wx2 + ref_cam.R[1]*wy2 + ref_cam.R[2]*wz2 + ref_cam.t[0];
    float ry = ref_cam.R[3]*wx2 + ref_cam.R[4]*wy2 + ref_cam.R[5]*wz2 + ref_cam.t[1];
    float rz = ref_cam.R[6]*wx2 + ref_cam.R[7]*wy2 + ref_cam.R[8]*wz2 + ref_cam.t[2];

    if (rz <= 1e-6f) continue;

    float ru = fx * rx / rz + cx;
    float rv = fy * ry / rz + cy;

    if (fabs(ru - (float)x) + fabs(rv - (float)y) > 2.0f) continue;
    if (fabs(rz - ref_depth) >= ref_depth * same_depth_threshold) continue;

    consistent++;
  }

  // ---- Adaptive decision: suspicious pixels get stricter thresholds ----
  int eff_min = is_suspicious ? (min_consistent_views + 1) : min_consistent_views;
  int eff_max_carve = is_suspicious ? max(1, max_carved_views - 1) : max_carved_views;

  int keep = (consistent >= eff_min) && (carved <= eff_max_carve);
  clean_depth[idx] = keep ? ref_depth : 0.0f;
}

)CL";

}  // namespace dense

#endif  // OPENSFM_HAVE_OPENCL
