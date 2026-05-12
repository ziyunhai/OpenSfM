#pragma once

// OpenCL kernels for DSM (Digital Surface Model) rasterization.
//
// Streaming mode-seeking rasterizer:
//   Each grid cell maintains N altitude modes (running mean + count).
//   Incoming samples are matched to existing modes; unmatched samples
//   accumulate in a per-cell ring buffer of K entries.  After each view,
//   a separate kernel analyses the buffer, detects clusters via sorted
//   sliding-window, and promotes them to mode slots.
//
//   Finalization picks the HIGHEST mode with sufficient support — yielding
//   a true Digital Surface Model (rooftops, canopy) rather than a terrain
//   model (ground).
//
// Followed by an optional edge-preserving bilateral filter.

#ifdef OPENSFM_HAVE_OPENCL

namespace dense {

inline const char* kDSMKernelSource = R"CL(

// Fixed-point scale: 8192 gives ~0.12 mm precision.
// Max mode_z before int32 overflow: 2^31 / 8192 ≈ 262 km → safe.
#define FP_SCALE     8192
#define N_MODES      3
#define K_BUF        10

// ----------------------------------------------------------------
// Clear all mode and buffer arrays to zero.
//
// mode_z     [N_MODES * ncells]  — fixed-point running-mean Z
// mode_count [N_MODES * ncells]  — observation count per mode
// buf_z      [K_BUF * ncells]    — ring-buffer of unmatched samples
// buf_pos    [ncells]            — write cursor for the ring buffer
// ----------------------------------------------------------------
__kernel void dsm_clear(
    __global int* mode_z,
    __global int* mode_count,
    __global int* buf_z,
    __global int* buf_pos,
    const int num_cells)
{
    const int idx = get_global_id(0);
    if (idx >= num_cells) return;

    for (int m = 0; m < N_MODES; ++m) {
        mode_z[m * num_cells + idx] = 0;
        mode_count[m * num_cells + idx] = 0;
    }
    for (int k = 0; k < K_BUF; ++k) {
        buf_z[k * num_cells + idx] = 0;
    }
    buf_pos[idx] = 0;
}

// ----------------------------------------------------------------
// Back-project a depthmap pixel to world space and scatter into the
// mode grid.  Each pixel either merges into the closest matching
// mode (atomic running-mean update) or appends to the ring buffer.
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
    __global volatile int* mode_count,
    __global int* buf_z,
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
    const float min_normal_z)
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

    // Normal Z in world frame — reject steep surfaces (walls).
    const int nbase = idx * 3;
    const float cnx = normal[nbase];
    const float cny = normal[nbase + 1];
    const float cnz = normal[nbase + 2];
    const float wnz = ri20 * cnx + ri21 * cny + ri22 * cnz;
    if (wnz < min_normal_z) return;

    // Fixed-point Z value for this sample.
    const int fp_z = (int)(wz * (float)FP_SCALE);

    const int cell = gy * grid_w + gx;

    // --- Try to match an existing mode ---
    int best_mode = -1;
    int best_dist = fp_threshold + 1;
    for (int m = 0; m < N_MODES; ++m) {
        const int slot = m * num_cells + cell;
        const int mc = mode_count[slot];
        if (mc <= 0) continue;
        const int mz = mode_z[slot];
        const int dist = abs(fp_z - mz);
        if (dist < best_dist) {
            best_dist = dist;
            best_mode = m;
        }
    }

    if (best_mode >= 0) {
        // Merge into the closest mode via atomic running-mean update.
        const int slot = best_mode * num_cells + cell;
        const int new_count = atomic_add(&mode_count[slot], 1) + 1;
        // CAS loop: mode_z ← old + (fp_z − old) / new_count
        int old_val = mode_z[slot];
        for (;;) {
            const int new_val = old_val + (fp_z - old_val) / new_count;
            const int prev = atomic_cmpxchg(
                (volatile __global int*)&mode_z[slot], old_val, new_val);
            if (prev == old_val) break;
            old_val = prev;
        }
    } else {
        // No matching mode — append to ring buffer.
        const int pos = atomic_add(&buf_pos[cell], 1);
        buf_z[(pos % K_BUF) * num_cells + cell] = fp_z;
    }
}

// ----------------------------------------------------------------
// Analyse each cell's ring buffer to detect new mode clusters.
//
// Algorithm (per cell):
//   1. Read min(buf_pos, K_BUF) buffered samples into private memory.
//   2. Insertion-sort (K_BUF ≤ 10 — very fast).
//   3. Sliding-window: find the longest subsequence where
//      sorted[end] − sorted[start] ≤ fp_threshold.
//   4. If the cluster has ≥ min_buf_samples entries:
//      a. Compute cluster mean.
//      b. If it matches an existing mode → merge.
//      c. Else find a free slot (count == 0) or the weakest slot
//         with fewer observations → replace.
//   5. Reset buf_pos to 0.
// ----------------------------------------------------------------
__kernel void dsm_update_modes(
    __global int* mode_z,
    __global int* mode_count,
    __global int* buf_z,
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
    int samples[K_BUF];
    for (int k = 0; k < n; ++k) {
        samples[k] = buf_z[k * num_cells + cell];
    }

    // Insertion sort.
    for (int i = 1; i < n; ++i) {
        const int key = samples[i];
        int j = i - 1;
        while (j >= 0 && samples[j] > key) {
            samples[j + 1] = samples[j];
            --j;
        }
        samples[j + 1] = key;
    }

    // Sliding window: find densest cluster within fp_threshold.
    int best_start = 0, best_len = 1;
    int start = 0;
    for (int end = 1; end < n; ++end) {
        while (samples[end] - samples[start] > fp_threshold) {
            ++start;
        }
        const int len = end - start + 1;
        if (len > best_len) {
            best_len = len;
            best_start = start;
        }
    }

    if (best_len < min_buf_samples) {
        // No significant cluster — just reset.
        buf_pos[cell] = 0;
        return;
    }

    // Cluster mean (integer arithmetic, no overflow for K_BUF ≤ 10).
    int cluster_sum = 0;
    for (int k = best_start; k < best_start + best_len; ++k) {
        cluster_sum += samples[k];
    }
    const int cluster_mean = cluster_sum / best_len;

    // Try to merge into an existing mode.
    int merge_slot = -1;
    int merge_dist = fp_threshold + 1;
    for (int m = 0; m < N_MODES; ++m) {
        const int slot = m * num_cells + cell;
        const int mc = mode_count[slot];
        if (mc <= 0) continue;
        const int dist = abs(cluster_mean - mode_z[slot]);
        if (dist < merge_dist) {
            merge_dist = dist;
            merge_slot = m;
        }
    }

    if (merge_slot >= 0) {
        // Merge cluster into existing mode.
        const int slot = merge_slot * num_cells + cell;
        const int old_count = mode_count[slot];
        const int new_count = old_count + best_len;
        // Weighted combination of old centre and cluster mean.
        mode_z[slot] = (mode_z[slot] * old_count + cluster_mean * best_len)
                       / new_count;
        mode_count[slot] = new_count;
    } else {
        // Find a free or weakest slot to place a new mode.
        int target = -1;
        int target_count = best_len;  // only replace if we're stronger
        for (int m = 0; m < N_MODES; ++m) {
            const int slot = m * num_cells + cell;
            const int mc = mode_count[slot];
            if (mc == 0) {
                target = m;
                break;  // free slot — take it
            }
            if (mc < target_count) {
                target_count = mc;
                target = m;
            }
        }
        if (target >= 0) {
            const int slot = target * num_cells + cell;
            mode_z[slot] = cluster_mean;
            mode_count[slot] = best_len;
        }
    }

    // Reset buffer.
    buf_pos[cell] = 0;
}

// ----------------------------------------------------------------
// Finalize: pick the HIGHEST mode with count ≥ min_count.
// Output float Z = mode_z / FP_SCALE.  NaN if no valid mode.
// ----------------------------------------------------------------
__kernel void dsm_finalize(
    __global const int* mode_z,
    __global const int* mode_count,
    __global float* out_z,
    const int num_cells,
    const int min_count)
{
    const int cell = get_global_id(0);
    if (cell >= num_cells) return;

    int best_z = 0;
    int found = 0;

    for (int m = 0; m < N_MODES; ++m) {
        const int slot = m * num_cells + cell;
        const int mc = mode_count[slot];
        if (mc < min_count) continue;
        const int mz = mode_z[slot];
        if (!found || mz > best_z) {
            best_z = mz;
            found = 1;
        }
    }

    out_z[cell] = found ? ((float)best_z / (float)FP_SCALE) : NAN;
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

)CL";

}  // namespace dense

#endif  // OPENSFM_HAVE_OPENCL
