#pragma once

// OpenCL kernels for DSM (Digital Surface Model) rasterization.
//
// Multi-scale confidence-weighted mode-seeking rasterizer:
//   Each grid cell maintains N altitude modes with weighted accumulation.
//   Samples are weighted by confidence × smoothstep(normal_z) to suppress
//   facade leakage while preserving sloped surfaces.
//
//   Incoming samples are matched to existing modes; unmatched samples
//   accumulate in a per-cell ring buffer.  After each view, a separate
//   kernel analyses the buffer, detects clusters via sorted sliding-window,
//   and promotes them to mode slots.
//
//   Finalization picks the HIGHEST mode with sufficient support, yielding
//   a true Digital Surface Model (rooftops, canopy) rather than a terrain
//   model (ground).
//
//   The pipeline runs at 3 scales (4×, 2×, 1× GSD).  After each scale's
//   finalization, Perona-Malik anisotropic diffusion fills holes guided
//   by the coarser level's gradient (self-guided at coarsest level).
//   Finer scales overwrite coarser composites where they have valid data.

#ifdef OPENSFM_HAVE_OPENCL

namespace dense {

inline const char* kDSMKernelSource = R"CL(

// Fixed-point scales.
// Z: 8192 gives ~0.12 mm precision. Max Z ≈ 262 km.
// Weight: 1024 gives 10-bit fractional. Max weight/mode ≈ 2M.
#define FP_SCALE        8192
#define FP_WEIGHT_SCALE 1024
#define N_MODES         3
#define K_BUF           10

// Smoothstep helper (Hermite interpolation, clamped [0,1]).
inline float smoothstep_f(float edge0, float edge1, float x) {
    float t = clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// ----------------------------------------------------------------
// Clear all mode, weight, and buffer arrays to zero.
//
// mode_z      [N_MODES * ncells] — fixed-point weighted-sum Z
// mode_weight [N_MODES * ncells] — fixed-point accumulated weight
// mode_count  [N_MODES * ncells] — observation count per mode
// buf_z       [K_BUF * ncells]   — ring-buffer of unmatched Z samples
// buf_w       [K_BUF * ncells]   — ring-buffer of unmatched weights
// buf_pos     [ncells]           — write cursor for the ring buffer
// ----------------------------------------------------------------
__kernel void dsm_clear(
    __global int* mode_z,
    __global int* mode_weight,
    __global int* mode_count,
    __global int* buf_z,
    __global int* buf_w,
    __global int* buf_pos,
    const int num_cells)
{
    const int idx = get_global_id(0);
    if (idx >= num_cells) return;

    for (int m = 0; m < N_MODES; ++m) {
        mode_z[m * num_cells + idx] = 0;
        mode_weight[m * num_cells + idx] = 0;
        mode_count[m * num_cells + idx] = 0;
    }
    for (int k = 0; k < K_BUF; ++k) {
        buf_z[k * num_cells + idx] = 0;
        buf_w[k * num_cells + idx] = 0;
    }
    buf_pos[idx] = 0;
}

// ----------------------------------------------------------------
// Back-project a depthmap pixel to world space and scatter into the
// mode grid with confidence × normal weighting.
//
// Hard gate: wnz < min_normal_z → reject (fast cull of vertical walls).
// Soft weight: w = confidence × smoothstep(min_normal_z, soft_upper_nz, wnz)
//
// Each pixel either merges into the closest matching mode (atomic
// weighted update) or appends (z, w) to the ring buffer.
//
// cam_params layout (21 floats, row-major):
//   [0..8]   K_inv  (3×3 inverse intrinsics)
//   [9..17]  R_inv  (3×3 = R^T, camera-to-world rotation)
//   [18..20] t      (translation: X_cam = R * X_world + t)
// ----------------------------------------------------------------
__kernel void dsm_scatter(
    __global const float* depth,
    __global const float* normal,
    __global const float* confidence,
    __global const float* cam_params,
    __global volatile int* mode_z,
    __global volatile int* mode_weight,
    __global volatile int* mode_count,
    __global int* buf_z,
    __global int* buf_w,
    __global volatile int* buf_pos,
    const float origin_x,
    const float origin_y,
    const float inv_gsd,
    const int grid_w,
    const int grid_h,
    const int img_w,
    const int img_h,
    const int num_cells,
    const int fp_threshold,
    const float min_normal_z,
    const float soft_upper_nz)
{
    const int idx = get_global_id(0);
    if (idx >= img_w * img_h) return;

    const float d = depth[idx];
    if (d <= 0.0f) return;

    const int u = idx % img_w;
    const int v = idx / img_w;

    // K_inv (row-major)
    const float ki00 = cam_params[0], ki01 = cam_params[1], ki02 = cam_params[2];
    const float ki10 = cam_params[3], ki11 = cam_params[4], ki12 = cam_params[5];
    const float ki20 = cam_params[6], ki21 = cam_params[7], ki22 = cam_params[8];
    // R_inv (row-major)
    const float ri00 = cam_params[9],  ri01 = cam_params[10], ri02 = cam_params[11];
    const float ri10 = cam_params[12], ri11 = cam_params[13], ri12 = cam_params[14];
    const float ri20 = cam_params[15], ri21 = cam_params[16], ri22 = cam_params[17];
    // Translation
    const float tx = cam_params[18], ty = cam_params[19], tz = cam_params[20];

    // Back-project: cam_pt = K_inv * [u, v, 1] * depth
    const float fu = (float)u;
    const float fv = (float)v;
    const float cx = (ki00 * fu + ki01 * fv + ki02) * d;
    const float cy = (ki10 * fu + ki11 * fv + ki12) * d;
    const float cz = (ki20 * fu + ki21 * fv + ki22) * d;

    // World = R_inv * (cam_pt - t)
    const float bx = cx - tx, by = cy - ty, bz = cz - tz;
    const float wx = ri00 * bx + ri01 * by + ri02 * bz;
    const float wy = ri10 * bx + ri11 * by + ri12 * bz;
    const float wz = ri20 * bx + ri21 * by + ri22 * bz;

    // Grid cell
    const int gx = (int)floor((wx - origin_x) * inv_gsd);
    const int gy = (int)floor((wy - origin_y) * inv_gsd);
    if (gx < 0 || gx >= grid_w || gy < 0 || gy >= grid_h) return;

    const int cell = gy * grid_w + gx;

    // Normal Z in world frame — hard gate rejects near-vertical surfaces.
    const int nbase = idx * 3;
    const float cnx = normal[nbase];
    const float cny = normal[nbase + 1];
    const float cnz = normal[nbase + 2];
    const float wnz = ri20 * cnx + ri21 * cny + ri22 * cnz;
    if (wnz < min_normal_z) return;

    // Soft weight: confidence × smoothstep(min_nz, soft_upper_nz, wnz)
    const float conf = confidence[idx];
    const float nz_weight = smoothstep_f(min_normal_z, soft_upper_nz, wnz);
    const float sample_weight = conf * nz_weight;
    if (sample_weight <= 0.0f) return;

    // Fixed-point Z and weight.
    const int fp_z = (int)(wz * (float)FP_SCALE);
    const int fp_w = (int)(sample_weight * (float)FP_WEIGHT_SCALE);
    if (fp_w <= 0) return;

    // --- Try to match an existing mode ---
    // Compare against weighted-mean Z: mode_z[slot] / mode_weight[slot].
    int best_mode = -1;
    int best_dist = fp_threshold + 1;
    for (int m = 0; m < N_MODES; ++m) {
        const int slot = m * num_cells + cell;
        const int mw = mode_weight[slot];
        if (mw <= 0) continue;
        // Weighted mean Z (fixed-point): mode_z / mode_weight * FP_WEIGHT_SCALE
        // To compare in same units as fp_z, compute:
        //   mean_z_fp = mode_z[slot] / mode_weight[slot] * FP_WEIGHT_SCALE
        // But division is expensive; instead compare:
        //   |fp_z * mw - mode_z[slot] * FP_WEIGHT_SCALE| < fp_threshold * mw
        // Using 64-bit to avoid overflow.
        const long mz_sum = (long)mode_z[slot];
        const long dist_num = (long)fp_z * (long)mw
                            - mz_sum * (long)FP_WEIGHT_SCALE;
        const long abs_dist = (dist_num >= 0) ? dist_num : -dist_num;
        const long thresh_scaled = (long)fp_threshold * (long)mw;
        if (abs_dist < thresh_scaled) {
            // Approximate integer distance for best-match selection.
            const int dist_approx = (int)(abs_dist / (long)mw);
            if (dist_approx < best_dist) {
                best_dist = dist_approx;
                best_mode = m;
            }
        }
    }

    if (best_mode >= 0) {
        // Merge into the closest mode via atomic weighted update.
        const int slot = best_mode * num_cells + cell;
        atomic_add(&mode_count[slot], 1);
        atomic_add(&mode_weight[slot], fp_w);
        // CAS loop: mode_z += delta  (accumulate weighted Z sum)
        // Use 64-bit to avoid overflow: fp_z * fp_w can exceed int32.
        int old_val = mode_z[slot];
        const int delta = (int)((long)fp_z * (long)fp_w / (long)FP_WEIGHT_SCALE);
        for (;;) {
            const int new_val = old_val + delta;
            const int prev = atomic_cmpxchg(
                (volatile __global int*)&mode_z[slot], old_val, new_val);
            if (prev == old_val) break;
            old_val = prev;
        }
    } else {
        // No matching mode — append (z, w) to ring buffer.
        const int pos = atomic_add(&buf_pos[cell], 1);
        const int buf_slot = (pos % K_BUF) * num_cells + cell;
        buf_z[buf_slot] = fp_z;
        buf_w[buf_slot] = fp_w;
    }
}

// ----------------------------------------------------------------
// Analyse each cell's ring buffer to detect new mode clusters.
//
// Algorithm (per cell):
//   1. Read min(buf_pos, K_BUF) buffered samples into private memory.
//   2. Insertion-sort by Z (K_BUF ≤ 10 — very fast).
//   3. Sliding-window: find the longest subsequence where
//      sorted[end] − sorted[start] ≤ fp_threshold.
//   4. If the cluster has ≥ min_buf_samples entries:
//      a. Compute weighted cluster mean.
//      b. If it matches an existing mode → merge.
//      c. Else find a free slot (weight == 0) or the weakest slot
//         with less accumulated weight → replace.
//   5. Reset buf_pos to 0.
// ----------------------------------------------------------------
__kernel void dsm_update_modes(
    __global int* mode_z,
    __global int* mode_weight,
    __global int* mode_count,
    __global int* buf_z,
    __global int* buf_w,
    __global int* buf_pos,
    const int num_cells,
    const int fp_threshold,
    const int min_buf_samples)
{
    const int cell = get_global_id(0);
    if (cell >= num_cells) return;

    const int raw_pos = buf_pos[cell];
    if (raw_pos < min_buf_samples) return;

    // Read samples into private memory.
    const int n = (raw_pos < K_BUF) ? raw_pos : K_BUF;
    int samples_z[K_BUF];
    int samples_w[K_BUF];
    for (int k = 0; k < n; ++k) {
        samples_z[k] = buf_z[k * num_cells + cell];
        samples_w[k] = buf_w[k * num_cells + cell];
    }

    // Insertion sort by Z (co-sort weights).
    for (int i = 1; i < n; ++i) {
        const int key_z = samples_z[i];
        const int key_w = samples_w[i];
        int j = i - 1;
        while (j >= 0 && samples_z[j] > key_z) {
            samples_z[j + 1] = samples_z[j];
            samples_w[j + 1] = samples_w[j];
            --j;
        }
        samples_z[j + 1] = key_z;
        samples_w[j + 1] = key_w;
    }

    // Sliding window: find densest cluster within fp_threshold.
    int best_start = 0, best_len = 1;
    int start = 0;
    for (int end = 1; end < n; ++end) {
        while (samples_z[end] - samples_z[start] > fp_threshold) {
            ++start;
        }
        const int len = end - start + 1;
        if (len > best_len) {
            best_len = len;
            best_start = start;
        }
    }

    if (best_len < min_buf_samples) {
        buf_pos[cell] = 0;
        return;
    }

    // Weighted cluster mean.  Use long for intermediate products
    // to avoid int32 overflow (fp_z * fp_w can exceed 2^31).
    int cluster_wz_sum = 0;  // sum of (z * w / FP_WEIGHT_SCALE)
    int cluster_w_sum = 0;   // sum of weights
    for (int k = best_start; k < best_start + best_len; ++k) {
        const int w = samples_w[k];
        cluster_w_sum += w;
        cluster_wz_sum += (int)((long)samples_z[k] * (long)w
                                / (long)FP_WEIGHT_SCALE);
    }

    // Try to merge into an existing mode.
    int merge_slot = -1;
    int merge_dist = fp_threshold + 1;
    // Cluster mean in FP_SCALE units (use long to avoid overflow):
    const int cluster_mean_z = (cluster_w_sum > 0)
        ? (int)((long)cluster_wz_sum * (long)FP_WEIGHT_SCALE
                / (long)cluster_w_sum)
        : 0;

    for (int m = 0; m < N_MODES; ++m) {
        const int slot = m * num_cells + cell;
        const int mw = mode_weight[slot];
        if (mw <= 0) continue;
        // Use long to avoid overflow in mode_z * FP_WEIGHT_SCALE.
        const int mode_mean_z = (int)((long)mode_z[slot]
                                      * (long)FP_WEIGHT_SCALE / (long)mw);
        const int dist = abs(cluster_mean_z - mode_mean_z);
        if (dist < merge_dist) {
            merge_dist = dist;
            merge_slot = m;
        }
    }

    if (merge_slot >= 0 && merge_dist < fp_threshold) {
        // Merge cluster into existing mode.
        const int slot = merge_slot * num_cells + cell;
        mode_z[slot] += cluster_wz_sum;
        mode_weight[slot] += cluster_w_sum;
        mode_count[slot] += best_len;
    } else {
        // Find a free or weakest slot to place a new mode.
        int target = -1;
        int target_weight = cluster_w_sum;  // only replace if we're stronger
        for (int m = 0; m < N_MODES; ++m) {
            const int slot = m * num_cells + cell;
            const int mw = mode_weight[slot];
            if (mw == 0) {
                target = m;
                break;  // free slot — take it
            }
            if (mw < target_weight) {
                target_weight = mw;
                target = m;
            }
        }
        if (target >= 0) {
            const int slot = target * num_cells + cell;
            mode_z[slot] = cluster_wz_sum;
            mode_weight[slot] = cluster_w_sum;
            mode_count[slot] = best_len;
        }
    }

    // Reset buffer.
    buf_pos[cell] = 0;
}

// ----------------------------------------------------------------
// Finalize: pick the HIGHEST mode with count ≥ min_count.
// Output weighted-mean Z and a validity mask.
//
// out_z[cell] = weighted_mean_z (float) or NaN.
// out_valid[cell] = 1 if valid mode found, 0 otherwise.
// ----------------------------------------------------------------
__kernel void dsm_finalize(
    __global const int* mode_z,
    __global const int* mode_weight,
    __global const int* mode_count,
    __global float* out_z,
    __global unsigned char* out_valid,
    const int num_cells,
    const int min_count)
{
    const int cell = get_global_id(0);
    if (cell >= num_cells) return;

    float best_z = 0.0f;
    int found = 0;

    for (int m = 0; m < N_MODES; ++m) {
        const int slot = m * num_cells + cell;
        const int mc = mode_count[slot];
        if (mc < min_count) continue;
        const int mw = mode_weight[slot];
        if (mw <= 0) continue;
        // Weighted mean Z = (mode_z / mode_weight) * (FP_WEIGHT_SCALE / FP_SCALE)
        // = mode_z * FP_WEIGHT_SCALE / (mode_weight * FP_SCALE)
        const float mean_z = (float)mode_z[slot] * (float)FP_WEIGHT_SCALE
                           / ((float)mw * (float)FP_SCALE);
        if (!found || mean_z > best_z) {
            best_z = mean_z;
            found = 1;
        }
    }

    out_z[cell] = found ? best_z : NAN;
    out_valid[cell] = found ? (unsigned char)1 : (unsigned char)0;
}

// ----------------------------------------------------------------
// Edge-preserving bilateral filter on the DSM grid (float).
// 2D dispatch: global_id(0) = x, global_id(1) = y.
// ----------------------------------------------------------------
__kernel void dsm_bilateral(
    __global const float* input,
    __global float* output,
    const int width,
    const int height,
    const int radius,
    const float inv_2sigma_s_sq,
    const float inv_2sigma_r_sq)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    const float center = input[y * width + x];
    if (isnan(center)) {
        output[y * width + x] = NAN;
        return;
    }

    float sum_val = 0.0f;
    float sum_wt  = 0.0f;

    for (int dy = -radius; dy <= radius; ++dy) {
        const int ny = y + dy;
        if (ny < 0 || ny >= height) continue;
        for (int dx = -radius; dx <= radius; ++dx) {
            const int nx = x + dx;
            if (nx < 0 || nx >= width) continue;
            const float nval = input[ny * width + nx];
            if (isnan(nval)) continue;

            const float sd = (float)(dx * dx + dy * dy);
            const float rd = center - nval;
            const float wt = exp(-sd * inv_2sigma_s_sq
                                 - rd * rd * inv_2sigma_r_sq);
            sum_val += nval * wt;
            sum_wt  += wt;
        }
    }

    output[y * width + x] = (sum_wt > 0.0f) ? (sum_val / sum_wt) : NAN;
}

// ----------------------------------------------------------------
// Perona-Malik anisotropic diffusion — one iteration.
// Fills NaN holes while preserving edges (Dirichlet BC on observed
// cells, zero-flux at data boundary).
//
// 2D dispatch: global_id(0) = x, global_id(1) = y.
//
// guide: edge magnitude from coarser level (or self for coarsest).
//        Used as |∇| in the edge-stopping function.
// valid_mask: 1 where finalize produced a value (Dirichlet — frozen).
// ----------------------------------------------------------------
__kernel void dsm_diffuse(
    __global const float* input,
    __global float* output,
    __global const float* guide,
    __global const unsigned char* valid_mask,
    const int width,
    const int height,
    const float kappa,
    const float dt)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    const int idx = y * width + x;

    // Observed cells are frozen (Dirichlet boundary condition).
    if (valid_mask[idx]) {
        output[idx] = input[idx];
        return;
    }

    const float center = input[idx];
    // If this cell has never been touched (still NaN at start),
    // check if any neighbor is valid to seed diffusion.
    if (isnan(center)) {
        // Try to initialize from mean of valid neighbors.
        float sum = 0.0f;
        int count = 0;
        if (x > 0 && !isnan(input[idx - 1]))         { sum += input[idx - 1]; ++count; }
        if (x < width - 1 && !isnan(input[idx + 1])) { sum += input[idx + 1]; ++count; }
        if (y > 0 && !isnan(input[idx - width]))      { sum += input[idx - width]; ++count; }
        if (y < height - 1 && !isnan(input[idx + width])) { sum += input[idx + width]; ++count; }
        output[idx] = (count > 0) ? (sum / (float)count) : NAN;
        return;
    }

    // 4-neighbor anisotropic diffusion.
    // Edge-stopping: g(x) = 1 / (1 + (x/kappa)^2)  [Perona-Malik Type 1]
    const float inv_kappa_sq = 1.0f / (kappa * kappa);
    float flux_sum = 0.0f;

    // Helper: conductance at midpoint between cell and neighbor.
    // Use max of guide values at center and neighbor as edge magnitude.
    const float g_center = guide[idx];

    // East
    if (x < width - 1) {
        const float nval = input[idx + 1];
        if (!isnan(nval)) {
            const float g_n = guide[idx + 1];
            const float g_max = fmax(g_center, g_n);
            const float cond = 1.0f / (1.0f + g_max * g_max * inv_kappa_sq);
            flux_sum += cond * (nval - center);
        }
    }
    // West
    if (x > 0) {
        const float nval = input[idx - 1];
        if (!isnan(nval)) {
            const float g_n = guide[idx - 1];
            const float g_max = fmax(g_center, g_n);
            const float cond = 1.0f / (1.0f + g_max * g_max * inv_kappa_sq);
            flux_sum += cond * (nval - center);
        }
    }
    // South
    if (y < height - 1) {
        const float nval = input[idx + width];
        if (!isnan(nval)) {
            const float g_n = guide[idx + width];
            const float g_max = fmax(g_center, g_n);
            const float cond = 1.0f / (1.0f + g_max * g_max * inv_kappa_sq);
            flux_sum += cond * (nval - center);
        }
    }
    // North
    if (y > 0) {
        const float nval = input[idx - width];
        if (!isnan(nval)) {
            const float g_n = guide[idx - width];
            const float g_max = fmax(g_center, g_n);
            const float cond = 1.0f / (1.0f + g_max * g_max * inv_kappa_sq);
            flux_sum += cond * (nval - center);
        }
    }

    output[idx] = center + dt * flux_sum;
}

// ----------------------------------------------------------------
// Compute gradient magnitude (Sobel-like) of a float grid.
// Used to generate the diffusion guide from a DSM level.
// 2D dispatch: global_id(0) = x, global_id(1) = y.
// ----------------------------------------------------------------
__kernel void dsm_gradient_magnitude(
    __global const float* input,
    __global float* output,
    const int width,
    const int height)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    const int idx = y * width + x;
    const float c = input[idx];
    if (isnan(c)) {
        output[idx] = 0.0f;
        return;
    }

    // Central differences with NaN handling (fall back to one-sided).
    float gx = 0.0f, gy = 0.0f;

    const float left  = (x > 0) ? input[idx - 1] : NAN;
    const float right = (x < width - 1) ? input[idx + 1] : NAN;
    if (!isnan(left) && !isnan(right)) {
        gx = (right - left) * 0.5f;
    } else if (!isnan(right)) {
        gx = right - c;
    } else if (!isnan(left)) {
        gx = c - left;
    }

    const float up   = (y > 0) ? input[idx - width] : NAN;
    const float down = (y < height - 1) ? input[idx + width] : NAN;
    if (!isnan(up) && !isnan(down)) {
        gy = (down - up) * 0.5f;
    } else if (!isnan(down)) {
        gy = down - c;
    } else if (!isnan(up)) {
        gy = c - up;
    }

    output[idx] = sqrt(gx * gx + gy * gy);
}

// ----------------------------------------------------------------
// Nearest-neighbor upsample by factor 2: coarse → fine grid.
// 2D dispatch over FINE grid: global_id(0) = fine_x, global_id(1) = fine_y.
// ----------------------------------------------------------------
__kernel void dsm_upsample_nn(
    __global const float* coarse,
    __global float* fine,
    const int coarse_w,
    const int coarse_h,
    const int fine_w,
    const int fine_h)
{
    const int fx = get_global_id(0);
    const int fy = get_global_id(1);
    if (fx >= fine_w || fy >= fine_h) return;

    const int cx = fx / 2;
    const int cy = fy / 2;
    const int ci = min(cy, coarse_h - 1) * coarse_w + min(cx, coarse_w - 1);
    fine[fy * fine_w + fx] = coarse[ci];
}

)CL";

}  // namespace dense

#endif  // OPENSFM_HAVE_OPENCL
