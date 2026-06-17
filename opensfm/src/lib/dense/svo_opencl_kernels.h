#pragma once

// OpenCL kernel source for SVO (Sparse Voxel Octree) TSDF integration.
//
// The integration kernel parallelises over depthmap pixels.  Each work-item
// back-projects its pixel, ray-marches through the truncation band, and
// atomically accumulates TSDF / normal / color into a GPU open-addressing
// hash table.
//
// The hash table uses a packed (x,y) 32-bit key with a separate 32-bit z key
// and a "ready" flag to handle the multi-word-key race condition safely.
// Values are accumulated as fixed-point integers and converted to floating
// point on final download.

#ifdef OPENSFM_HAVE_OPENCL

namespace dense {

inline const char* kSVOKernelSource =
    R"CL(

// =====================================================================
// Constants
// =====================================================================
#define EMPTY_KEY 0xFFFFFFFFu
#define KEY_C_UNINIT ((int)0x80000000)
#define FP_SCALE  32768
#define MAX_PROBES 1024
#define WEIGHT_SCALE 128

// =====================================================================
// GPU hash table slot — must match host-side GPUVoxelSlot exactly.
// 9 x int32 = 36 bytes.  Color is baked separately post-extraction.
// =====================================================================
typedef struct {
    uint key_ab;      // packed (x+32768)<<16 | (y+32768), EMPTY = 0xFFFFFFFF
    int  key_c;       // z coordinate, KEY_C_UNINIT = 0x80000000 when unwritten
    int  count;       // observation count
    int  sum_tsdf;    // fixed-point TSDF accumulator
    int  sum_nx;      // fixed-point normal.x accumulator
    int  sum_ny;
    int  sum_nz;
    int  sum_weight;  // accumulated confidence weight (scale WEIGHT_SCALE)
    int  _pad;        // reserved
} VoxelSlot;

// =====================================================================
// Per-frame camera parameters — must match host-side SVOCameraGPU.
// All matrices are stored in **row-major** order.
// =====================================================================
typedef struct {
    float Kinv[9];    // inverse intrinsics
    float Rinv[9];    // R^T  (cam → world rotation)
    float R[9];       // world → cam rotation
    float t[3];       // translation  (world → cam)
    float cam_pos[3]; // camera centre in world = -R^T * t
    float _pad[3];
} SVOCamera;

// =====================================================================
// Hash helpers
// =====================================================================
uint pack_xy(int kx, int ky) {
    return ((uint)(kx + 32768) << 16) | (uint)(ky + 32768);
}

uint voxel_hash(int kx, int ky, int kz) {
    uint h = 2166136261u;
    h ^= (uint)kx; h *= 16777619u;
    h ^= (uint)ky; h *= 16777619u;
    h ^= (uint)kz; h *= 16777619u;
    return h;
}

// =====================================================================
// Atomically insert-or-accumulate a voxel contribution.
//
// All value sums (tsdf, normal, color) are PRE-MULTIPLIED by weight
// before accumulation so that Download can compute weighted averages
// by dividing by sum_weight.
//
// Protocol (32-bit CAS, no ready flag):
//   1. CAS on key_ab to claim an empty slot.
//   2. For a freshly claimed slot, write key_c with atomic_xchg +
//      mem_fence (transitions from KEY_C_UNINIT to valid kz).
//   3. For a slot that already has our key_ab, spin on key_c until it
//      leaves KEY_C_UNINIT.  On match, accumulate; otherwise probe.
//   4. Accumulation uses atomic_add on every field (commutative).
//
// Overflow counter is incremented on probe exhaustion or (extremely
// rare) spin exhaustion, making drops visible to the host.
// =====================================================================
void hash_accumulate(__global VoxelSlot* table, uint mask,
                     __global uint* overflow_counter,
                     int kx, int ky, int kz,
                     int tsdf_fp,
                     int nx_fp, int ny_fp, int nz_fp,
                     int weight)
{
    uint my_ab = pack_xy(kx, ky);
    if (my_ab == EMPTY_KEY) return;  // (32767,32767) cannot be stored

    // Pre-multiply values by weight for weighted averaging.
    int w_tsdf = tsdf_fp * weight;
    int w_nx   = nx_fp   * weight;
    int w_ny   = ny_fp   * weight;
    int w_nz   = nz_fp   * weight;

    uint h = voxel_hash(kx, ky, kz) & mask;

    for (int probe = 0; probe < MAX_PROBES; probe++) {
        uint slot = (h + (uint)probe) & mask;

        uint old = atomic_cmpxchg(&table[slot].key_ab, EMPTY_KEY, my_ab);

        if (old == EMPTY_KEY) {
            // Freshly claimed — write key_c, then accumulate.
            atomic_xchg(&table[slot].key_c, kz);
            mem_fence(CLK_GLOBAL_MEM_FENCE);

            atomic_add(&table[slot].count,      1);
            atomic_add(&table[slot].sum_weight, weight);
            atomic_add(&table[slot].sum_tsdf,   w_tsdf);
            atomic_add(&table[slot].sum_nx,     w_nx);
            atomic_add(&table[slot].sum_ny,     w_ny);
            atomic_add(&table[slot].sum_nz,     w_nz);
            return;
        }

        if (old == my_ab) {
            // Same (x,y) — spin until key_c is written.
            int stored_z = KEY_C_UNINIT;
            for (int s = 0; s < 32768; s++) {
                stored_z = atomic_add(&table[slot].key_c, 0);
                if (stored_z != KEY_C_UNINIT) break;
            }
            if (stored_z == KEY_C_UNINIT) {
                // Spin exhausted (practically impossible on Apple GPU).
                atomic_add(overflow_counter, 1u);
                continue;
            }
            if (stored_z == kz) {
                atomic_add(&table[slot].count,      1);
                atomic_add(&table[slot].sum_weight, weight);
                atomic_add(&table[slot].sum_tsdf,   w_tsdf);
                atomic_add(&table[slot].sum_nx,     w_nx);
                atomic_add(&table[slot].sum_ny,     w_ny);
                atomic_add(&table[slot].sum_nz,     w_nz);
                return;
            }
            // Different z — linear probe.
        }
        // Different key_ab — linear probe.
    }
    // Dropped (max probes exceeded — table too full).
    atomic_add(overflow_counter, 1u);
}

// =====================================================================
// Clear the hash table (one work-item per slot).
// =====================================================================
__kernel void svo_clear_table(__global VoxelSlot* table, uint capacity) {
    uint i = get_global_id(0);
    if (i >= capacity) return;
    table[i].key_ab     = EMPTY_KEY;
    table[i].key_c      = KEY_C_UNINIT;
    table[i].count      = 0;
    table[i].sum_tsdf   = 0;
    table[i].sum_nx     = 0;
    table[i].sum_ny     = 0;
    table[i].sum_nz     = 0;
    table[i].sum_weight = 0;
    table[i]._pad       = 0;
}


#define COS_THETA_MIN 0.15f   // ~81° — skip pixels at more grazing angles

// Forward declaration (defined after svo_extract_points).
uint hash_lookup(__global const VoxelSlot* table, uint mask,
                 int kx, int ky, int kz);

// =====================================================================
// Integrate one depthmap into the hash table.
// Global work size: (cols_padded, rows_padded).
// Supports optional multi-level reference check: if a finer level already
// has a well-weighted voxel at the pixel's depth position, skip integration.
// =====================================================================
__kernel void svo_integrate(
    __global VoxelSlot*    table,
    uint                   capacity_mask,
    __global uint*         overflow_counter,
    __global const float*  depth,
    __global const float*  normal_buf,
    __global const uchar*  mask_buf,
    __global const float*  weight_buf,
    int                    has_normal,
    int                    has_mask,
    int                    has_weight,
    __constant SVOCamera*  cam,
    int                    rows,
    int                    cols,
    float                  trunc,
    float                  voxel_size,
    float                  inv_voxel_size,
    int                    bbox_min_x,
    int                    bbox_min_y,
    int                    bbox_min_z,
    int                    bbox_max_x,
    int                    bbox_max_y,
    int                    bbox_max_z,
    int                    has_bbox,
    // --- Multi-level reference check (up to 2 finer tables) ---
    int                    n_ref_tables,
    __global const VoxelSlot* ref_table_0,
    uint                   ref_mask_0,
    float                  ref_inv_vs_0,
    __global const VoxelSlot* ref_table_1,
    uint                   ref_mask_1,
    float                  ref_inv_vs_1,
    float                  ref_min_weight)
{
    int c = get_global_id(0);
    int r = get_global_id(1);
    if (c >= cols || r >= rows) return;

    int idx = r * cols + c;

    if (has_mask && mask_buf[idx] == 0) return;

    float d = depth[idx];
    if (d <= 0.0f) return;

    // ---- Back-project pixel to camera frame ----
    float uc = (float)c, vr = (float)r;
    float3 p_cam;
    p_cam.x = d * (cam->Kinv[0]*uc + cam->Kinv[1]*vr + cam->Kinv[2]);
    p_cam.y = d * (cam->Kinv[3]*uc + cam->Kinv[4]*vr + cam->Kinv[5]);
    p_cam.z = d * (cam->Kinv[6]*uc + cam->Kinv[7]*vr + cam->Kinv[8]);

    // ---- Transform to world: Rinv * (p_cam - t) ----
    float3 diff = (float3)(
        p_cam.x - cam->t[0],
        p_cam.y - cam->t[1],
        p_cam.z - cam->t[2]);
    float3 p_world;
    p_world.x = cam->Rinv[0]*diff.x + cam->Rinv[1]*diff.y + cam->Rinv[2]*diff.z;
    p_world.y = cam->Rinv[3]*diff.x + cam->Rinv[4]*diff.y + cam->Rinv[5]*diff.z;
    p_world.z = cam->Rinv[6]*diff.x + cam->Rinv[7]*diff.y + cam->Rinv[8]*diff.z;

    // ---- Multi-level coverage check: skip if finer levels already cover ----
    if (n_ref_tables > 0) {
        int ref_kx_0 = (int)floor(p_world.x * ref_inv_vs_0);
        int ref_ky_0 = (int)floor(p_world.y * ref_inv_vs_0);
        int ref_kz_0 = (int)floor(p_world.z * ref_inv_vs_0);
        uint ref_slot_0 = hash_lookup(ref_table_0, ref_mask_0, ref_kx_0, ref_ky_0, ref_kz_0);
        if (ref_slot_0 != 0xFFFFFFFF) {
            float ref_sw_0 = (float)ref_table_0[ref_slot_0].sum_weight;
            if (ref_sw_0 >= ref_min_weight) return;
        }
        if (n_ref_tables > 1) {
            int ref_kx_1 = (int)floor(p_world.x * ref_inv_vs_1);
            int ref_ky_1 = (int)floor(p_world.y * ref_inv_vs_1);
            int ref_kz_1 = (int)floor(p_world.z * ref_inv_vs_1);
            uint ref_slot_1 = hash_lookup(ref_table_1, ref_mask_1, ref_kx_1, ref_ky_1, ref_kz_1);
            if (ref_slot_1 != 0xFFFFFFFF) {
                float ref_sw_1 = (float)ref_table_1[ref_slot_1].sum_weight;
                if (ref_sw_1 >= ref_min_weight) return;
            }
        }
    }

    // ---- Ray from camera centre to surface point ----
    float3 cp = (float3)(cam->cam_pos[0], cam->cam_pos[1], cam->cam_pos[2]);
    float3 ray = p_world - cp;
    float ray_len = length(ray);
    if (ray_len < 1e-8f) return;
    // Guard against float32 precision loss: when ray_len is so large
    // that tt += voxel_size is a no-op (below the ULP), the ray-march
    // loop becomes infinite.  FLT_EPSILON ~ 1.19e-7.
    if (ray_len * 1.2e-7f >= voxel_size) return;
    ray /= ray_len;

    float t_start = fmax(0.0f, ray_len - trunc);
    float t_end   = ray_len + trunc;

    // ---- Per-pixel normal in world frame ----
    float3 n_w = (float3)(0.0f, 0.0f, 0.0f);
    int nx_fp = 0, ny_fp = 0, nz_fp = 0;
    if (has_normal) {
        float3 n_cam = (float3)(
            normal_buf[idx*3], normal_buf[idx*3+1], normal_buf[idx*3+2]);
        float nlen = length(n_cam);
        if (nlen > 1e-6f) {
            n_cam /= nlen;
            n_w.x = cam->Rinv[0]*n_cam.x + cam->Rinv[1]*n_cam.y + cam->Rinv[2]*n_cam.z;
            n_w.y = cam->Rinv[3]*n_cam.x + cam->Rinv[4]*n_cam.y + cam->Rinv[5]*n_cam.z;
            n_w.z = cam->Rinv[6]*n_cam.x + cam->Rinv[7]*n_cam.y + cam->Rinv[8]*n_cam.z;
            nx_fp = (int)(n_w.x * (float)FP_SCALE);
            ny_fp = (int)(n_w.y * (float)FP_SCALE);
            nz_fp = (int)(n_w.z * (float)FP_SCALE);

            // cos(angle between ray and surface normal).
            // Normal should point toward camera (negative dot with ray).
            float cos_theta = fabs(dot(ray, n_w));
            if (cos_theta < COS_THETA_MIN) return;  // grazing — skip pixel
        }
    }

    // ---- Per-pixel weight: confidence ----
    float w_float = 1.0f;
    if (has_weight) {
        w_float *= weight_buf[idx];
    }
    int w_int = clamp((int)(w_float * (float)WEIGHT_SCALE),
                      1, WEIGHT_SCALE * 4);

    // ---- Ray-march through the truncation band ----
    for (float tt = t_start; tt <= t_end; tt += voxel_size) {
        float3 sample_pt = cp + tt * ray;

        // World → voxel coordinate
        int kx = (int)floor(sample_pt.x * inv_voxel_size);
        int ky = (int)floor(sample_pt.y * inv_voxel_size);
        int kz = (int)floor(sample_pt.z * inv_voxel_size);

        // Skip out-of-range coordinates (16-bit packing limit).
        if (kx < -32768 || kx > 32766 || ky < -32768 || ky > 32766)
            continue;

        // Skip voxels outside the bounding box (sub-volume clipping).
        if (has_bbox &&
            (kx < bbox_min_x || kx > bbox_max_x ||
             ky < bbox_min_y || ky > bbox_max_y ||
             kz < bbox_min_z || kz > bbox_max_z)) continue;

        // Voxel centre in world coordinates.
        float3 center = (float3)(
            ((float)kx + 0.5f) * voxel_size,
            ((float)ky + 0.5f) * voxel_size,
            ((float)kz + 0.5f) * voxel_size);

        // Project voxel centre into camera frame → get its depth.
        float3 p_cam_v;
        p_cam_v.x = cam->R[0]*center.x + cam->R[1]*center.y + cam->R[2]*center.z + cam->t[0];
        p_cam_v.y = cam->R[3]*center.x + cam->R[4]*center.y + cam->R[5]*center.z + cam->t[1];
        p_cam_v.z = cam->R[6]*center.x + cam->R[7]*center.y + cam->R[8]*center.z + cam->t[2];

        if (p_cam_v.z <= 0.0f) continue;

        float sdf = d - p_cam_v.z;
        if (sdf < -trunc) continue;

        float tsdf = fmin(sdf / trunc, 1.0f);
        int tsdf_fp = (int)(tsdf * (float)FP_SCALE);

        hash_accumulate(table, capacity_mask, overflow_counter,
                        kx, ky, kz,
                        tsdf_fp, nx_fp, ny_fp, nz_fp, w_int);
    }
})CL"
    R"CL(

// =====================================================================
// Lightweight counting hash table (12 bytes/slot vs 48 for integration).
// Used in a dry-run pass to count unique voxels before allocating the
// full integration table with the right capacity.
// =====================================================================
typedef struct {
    uint key_ab;   // packed (x+32768)<<16 | (y+32768), EMPTY = 0xFFFFFFFF
    int  key_c;    // z coordinate, KEY_C_UNINIT when unwritten
} CountSlot;

// Insert a voxel key into the counting table.  Increments *counter
// exactly once per unique (kx, ky, kz) coordinate.
void count_hash_insert(__global CountSlot* table, uint mask,
                       __global uint* counter,
                       __global uint* overflow_counter,
                       int kx, int ky, int kz)
{
    uint my_ab = pack_xy(kx, ky);
    if (my_ab == EMPTY_KEY) return;

    uint h = voxel_hash(kx, ky, kz) & mask;

    for (int probe = 0; probe < MAX_PROBES; probe++) {
        uint slot = (h + (uint)probe) & mask;

        uint old = atomic_cmpxchg(&table[slot].key_ab, EMPTY_KEY, my_ab);

        if (old == EMPTY_KEY) {
            // Freshly claimed slot — write z, count.
            atomic_xchg(&table[slot].key_c, kz);
            mem_fence(CLK_GLOBAL_MEM_FENCE);
            atomic_add(counter, 1u);
            return;
        }

        if (old == my_ab) {
            // Same (x,y) — spin until key_c leaves UNINIT.
            int stored_z = KEY_C_UNINIT;
            for (int s = 0; s < 32768; s++) {
                stored_z = atomic_add(&table[slot].key_c, 0);
                if (stored_z != KEY_C_UNINIT) break;
            }
            if (stored_z == KEY_C_UNINIT) {
                atomic_add(overflow_counter, 1u);
                continue;
            }
            if (stored_z == kz) return;  // already counted
        }
        // Different key — linear probe.
    }
    // Dropped (max probes exceeded).
    atomic_add(overflow_counter, 1u);
}

// Clear the counting table.
__kernel void svo_clear_count_table(__global CountSlot* table, uint capacity) {
    uint i = get_global_id(0);
    if (i >= capacity) return;
    table[i].key_ab = EMPTY_KEY;
    table[i].key_c  = KEY_C_UNINIT;
}

// =====================================================================
// Count unique voxels touched by one depthmap (dry-run ray-march).
// Same back-projection and ray-march as svo_integrate but only inserts
// keys into the lightweight counting table — no TSDF/color/normal.
// =====================================================================
__kernel void svo_count(
    __global CountSlot*    table,
    uint                   capacity_mask,
    __global uint*         counter,
    __global uint*         overflow_counter,
    __global const float*  depth,
    __global const uchar*  mask_buf,
    int                    has_mask,
    __constant SVOCamera*  cam,
    int                    rows,
    int                    cols,
    float                  trunc,
    float                  voxel_size,
    float                  inv_voxel_size,
    int                    bbox_min_x,
    int                    bbox_min_y,
    int                    bbox_min_z,
    int                    bbox_max_x,
    int                    bbox_max_y,
    int                    bbox_max_z,
    int                    has_bbox)
{
    int c = get_global_id(0);
    int r = get_global_id(1);
    if (c >= cols || r >= rows) return;

    int idx = r * cols + c;

    if (has_mask && mask_buf[idx] == 0) return;

    float d = depth[idx];
    if (d <= 0.0f) return;

    // ---- Back-project pixel to camera frame ----
    float uc = (float)c, vr = (float)r;
    float3 p_cam;
    p_cam.x = d * (cam->Kinv[0]*uc + cam->Kinv[1]*vr + cam->Kinv[2]);
    p_cam.y = d * (cam->Kinv[3]*uc + cam->Kinv[4]*vr + cam->Kinv[5]);
    p_cam.z = d * (cam->Kinv[6]*uc + cam->Kinv[7]*vr + cam->Kinv[8]);

    // ---- Transform to world: Rinv * (p_cam - t) ----
    float3 diff = (float3)(
        p_cam.x - cam->t[0],
        p_cam.y - cam->t[1],
        p_cam.z - cam->t[2]);
    float3 p_world;
    p_world.x = cam->Rinv[0]*diff.x + cam->Rinv[1]*diff.y + cam->Rinv[2]*diff.z;
    p_world.y = cam->Rinv[3]*diff.x + cam->Rinv[4]*diff.y + cam->Rinv[5]*diff.z;
    p_world.z = cam->Rinv[6]*diff.x + cam->Rinv[7]*diff.y + cam->Rinv[8]*diff.z;

    // ---- Ray from camera centre to surface point ----
    float3 cp = (float3)(cam->cam_pos[0], cam->cam_pos[1], cam->cam_pos[2]);
    float3 ray = p_world - cp;
    float ray_len = length(ray);
    if (ray_len < 1e-8f) return;
    if (ray_len * 1.2e-7f >= voxel_size) return;
    ray /= ray_len;

    float t_start = fmax(0.0f, ray_len - trunc);
    float t_end   = ray_len + trunc;

    // ---- Ray-march through the truncation band ----
    for (float tt = t_start; tt <= t_end; tt += voxel_size) {
        float3 sample_pt = cp + tt * ray;

        int kx = (int)floor(sample_pt.x * inv_voxel_size);
        int ky = (int)floor(sample_pt.y * inv_voxel_size);
        int kz = (int)floor(sample_pt.z * inv_voxel_size);

        if (kx < -32768 || kx > 32766 || ky < -32768 || ky > 32766)
            continue;

        // Skip voxels outside the bounding box (sub-volume clipping).
        if (has_bbox &&
            (kx < bbox_min_x || kx > bbox_max_x ||
             ky < bbox_min_y || ky > bbox_max_y ||
             kz < bbox_min_z || kz > bbox_max_z)) continue;

        // Voxel centre in world coordinates.
        float3 center = (float3)(
            ((float)kx + 0.5f) * voxel_size,
            ((float)ky + 0.5f) * voxel_size,
            ((float)kz + 0.5f) * voxel_size);

        // Project voxel centre into camera frame.
        float3 p_cam_v;
        p_cam_v.x = cam->R[0]*center.x + cam->R[1]*center.y + cam->R[2]*center.z + cam->t[0];
        p_cam_v.y = cam->R[3]*center.x + cam->R[4]*center.y + cam->R[5]*center.z + cam->t[1];
        p_cam_v.z = cam->R[6]*center.x + cam->R[7]*center.y + cam->R[8]*center.z + cam->t[2];

        if (p_cam_v.z <= 0.0f) continue;

        float sdf = d - p_cam_v.z;
        if (sdf < -trunc) continue;

        count_hash_insert(table, capacity_mask, counter, overflow_counter, kx, ky, kz);
    }
}

// =====================================================================
// GPU-side surface point extraction.
//
// One work-item per hash table slot.  For each occupied slot with
// sufficient weight, check the three positive-axis neighbours
// (+X, +Y, +Z) for a TSDF sign change.  When found, linearly
// interpolate position, normal, and color and write to the output
// arrays via an atomic counter.
//
// This replaces the expensive Download→ImportVoxels→ExtractPoints
// CPU pipeline and avoids transferring the entire hash table.
// =====================================================================

// Look up a voxel in the integration hash table by its integer
// coordinate.  Returns the slot index if found, 0xFFFFFFFF otherwise.
uint hash_lookup(__global const VoxelSlot* table, uint mask,
                 int kx, int ky, int kz)
{
    uint my_ab = pack_xy(kx, ky);
    if (my_ab == EMPTY_KEY) return 0xFFFFFFFF;

    uint h = voxel_hash(kx, ky, kz) & mask;

    for (int probe = 0; probe < MAX_PROBES; probe++) {
        uint slot_idx = (h + (uint)probe) & mask;
        uint stored_ab = table[slot_idx].key_ab;

        if (stored_ab == EMPTY_KEY) return 0xFFFFFFFF;

        if (stored_ab == my_ab) {
            int stored_z = table[slot_idx].key_c;
            if (stored_z == KEY_C_UNINIT) continue;
            if (stored_z == kz) return slot_idx;
        }
    }
    return 0xFFFFFFFF;
}

__kernel void svo_extract_points(
    __global const VoxelSlot* table,
    uint                      capacity_mask,
    uint                      capacity,
    float                     min_weight_scaled,
    float                     voxel_size,
    uint                      decimate_flat,
    float                     edge_threshold,
    int                       min_count,
    float                     relative_min_weight,
    __global float*           out_points,
    __global float*           out_normals,
    __global uchar*           out_colors,
    __global uint*            out_counter,
    uint                      max_output)
{
    uint i = get_global_id(0);
    if (i >= capacity) return;

    // Read slot A.
    uint key_ab_a = table[i].key_ab;
    if (key_ab_a == EMPTY_KEY) return;
    int count_a = table[i].count;
    if (count_a < min_count) return;

    float sw_a = (float)table[i].sum_weight;
    if (sw_a < min_weight_scaled) return;

    int kx = (int)((key_ab_a >> 16) & 0xFFFF) - 32768;
    int ky = (int)(key_ab_a & 0xFFFF) - 32768;
    int kz = table[i].key_c;

    float inv_sw_a     = 1.0f / sw_a;
    float inv_fp_sw_a  = inv_sw_a / (float)FP_SCALE;
    float tsdf_a       = (float)table[i].sum_tsdf * inv_fp_sw_a;

    // Pre-decode voxel A normal (reused for all 3 axes).
    float na_x = (float)table[i].sum_nx * inv_fp_sw_a;
    float na_y = (float)table[i].sum_ny * inv_fp_sw_a;
    float na_z = (float)table[i].sum_nz * inv_fp_sw_a;

    // Check +X, +Y, +Z neighbours for TSDF sign change.
    int off_x[3] = {1, 0, 0};
    int off_y[3] = {0, 1, 0};
    int off_z[3] = {0, 0, 1};

    for (int d = 0; d < 3; d++) {
        int nx = kx + off_x[d];
        int ny = ky + off_y[d];
        int nz = kz + off_z[d];

        uint nb_idx = hash_lookup(table, capacity_mask, nx, ny, nz);
        if (nb_idx == 0xFFFFFFFF) continue;

        int count_b = table[nb_idx].count;
        if (count_b < min_count) continue;

        float sw_b = (float)table[nb_idx].sum_weight;
        if (sw_b < min_weight_scaled) continue;

        float inv_sw_b    = 1.0f / sw_b;
        float inv_fp_sw_b = inv_sw_b / (float)FP_SCALE;
        float tsdf_b      = (float)table[nb_idx].sum_tsdf * inv_fp_sw_b;

        // Sign change?
        if ((tsdf_a > 0.0f) == (tsdf_b > 0.0f)) continue;
        if (tsdf_a == 0.0f || tsdf_b == 0.0f) continue;

        // Interpolation factor: tsdf_a + t*(tsdf_b - tsdf_a) = 0.
        float t = tsdf_a / (tsdf_a - tsdf_b);
        t = clamp(t, 0.0f, 1.0f);

        // Interpolated position.
        float pa_x = ((float)kx + 0.5f) * voxel_size;
        float pa_y = ((float)ky + 0.5f) * voxel_size;
        float pa_z = ((float)kz + 0.5f) * voxel_size;
        float pb_x = ((float)nx + 0.5f) * voxel_size;
        float pb_y = ((float)ny + 0.5f) * voxel_size;
        float pb_z = ((float)nz + 0.5f) * voxel_size;

        float px = pa_x + t * (pb_x - pa_x);
        float py = pa_y + t * (pb_y - pa_y);
        float pz = pa_z + t * (pb_z - pa_z);

        // Interpolated normal.
        float nb_x = (float)table[nb_idx].sum_nx * inv_fp_sw_b;
        float nb_y = (float)table[nb_idx].sum_ny * inv_fp_sw_b;
        float nb_z = (float)table[nb_idx].sum_nz * inv_fp_sw_b;
        float inx = na_x + t * (nb_x - na_x);
        float iny = na_y + t * (nb_y - na_y);
        float inz = na_z + t * (nb_z - na_z);
        float nlen = sqrt(inx*inx + iny*iny + inz*inz);
        if (nlen > 1e-6f) { inx /= nlen; iny /= nlen; inz /= nlen; }

        // Local adaptive weight threshold: require weight to be a fraction
        // of the local neighborhood maximum.  Skipped when relative_min_weight <= 0.
        if (relative_min_weight > 0.0f) {
            float local_max_w = fmax(sw_a, sw_b);
            // Sample 6-connected neighborhood for maximum weight.
            int nb6_dx[6] = {1, -1, 0,  0, 0,  0};
            int nb6_dy[6] = {0,  0, 1, -1, 0,  0};
            int nb6_dz[6] = {0,  0, 0,  0, 1, -1};
            for (int n = 0; n < 6; n++) {
                uint nidx = hash_lookup(table, capacity_mask,
                    kx + nb6_dx[n], ky + nb6_dy[n], kz + nb6_dz[n]);
                if (nidx != 0xFFFFFFFF)
                    local_max_w = fmax(local_max_w, (float)table[nidx].sum_weight);
            }
            float adaptive_min = fmax(min_weight_scaled, local_max_w * relative_min_weight);
            if (sw_a < adaptive_min || sw_b < adaptive_min) continue;
        }

        // Curvature-adaptive decimation: skip flat-region points.
        if (decimate_flat > 1u) {
            float cos_nn = na_x*nb_x + na_y*nb_y + na_z*nb_z;
            float edge_score = 1.0f - fabs(cos_nn);
            if (edge_score < edge_threshold) {
                // Flat region: deterministic spatial skip.
                if (voxel_hash(kx, ky, kz) % decimate_flat != 0u) continue;
            }
        }

        // Atomic output index.
        uint out_idx = atomic_add(out_counter, 1u);
        if (out_idx >= max_output) continue;

        out_points[out_idx * 3 + 0] = px;
        out_points[out_idx * 3 + 1] = py;
        out_points[out_idx * 3 + 2] = pz;
        out_normals[out_idx * 3 + 0] = inx;
        out_normals[out_idx * 3 + 1] = iny;
        out_normals[out_idx * 3 + 2] = inz;
        // Colors will be baked by svo_bake_colors after extraction.
        out_colors[out_idx * 3 + 0] = 128;
        out_colors[out_idx * 3 + 1] = 128;
        out_colors[out_idx * 3 + 2] = 128;
    }
}

// =====================================================================
// Multi-level fill-in extraction.
//
// One work-item per coarse hash table slot.  For each occupied coarse
// voxel with a TSDF sign change to a positive-axis neighbor, sub-sample
// the crossing plane at fine-level density and emit points only where
// the fine table has NO coverage (i.e. fill holes).
//
// Arguments:
//   coarse_table / coarse_mask / coarse_capacity — coarse integration table
//   fine_table / fine_mask — fine (L=0) integration table (read-only lookup)
//   min_weight_scaled — minimum sum_weight to consider a voxel valid
//   coarse_voxel_size — voxel size at coarse level
//   fine_voxel_size — voxel size at fine level
//   level_shift — log2(coarse_voxel_size / fine_voxel_size), i.e. 1<<level_shift sub-samples per axis
//   out_points/out_normals/out_colors/out_counter/max_output — output buffers
// =====================================================================
__kernel void svo_extract_fill(
    __global const VoxelSlot* coarse_table,
    uint                      coarse_mask,
    uint                      coarse_capacity,
    __global const VoxelSlot* fine_table,
    uint                      fine_mask,
    float                     min_weight_scaled,
    float                     coarse_voxel_size,
    float                     fine_voxel_size,
    int                       level_shift,
    __global float*           out_points,
    __global float*           out_normals,
    __global uchar*           out_colors,
    __global uint*            out_counter,
    uint                      max_output)
{
    uint i = get_global_id(0);
    if (i >= coarse_capacity) return;

    // Read coarse slot A.
    uint key_ab_a = coarse_table[i].key_ab;
    if (key_ab_a == EMPTY_KEY) return;
    int count_a = coarse_table[i].count;
    if (count_a == 0) return;

    float sw_a = (float)coarse_table[i].sum_weight;
    if (sw_a < min_weight_scaled) return;

    int kx = (int)((key_ab_a >> 16) & 0xFFFF) - 32768;
    int ky = (int)(key_ab_a & 0xFFFF) - 32768;
    int kz = coarse_table[i].key_c;

    float inv_sw_a    = 1.0f / sw_a;
    float inv_fp_sw_a = inv_sw_a / (float)FP_SCALE;
    float tsdf_a      = (float)coarse_table[i].sum_tsdf * inv_fp_sw_a;

    // Pre-decode coarse voxel A normal.
    float na_x = (float)coarse_table[i].sum_nx * inv_fp_sw_a;
    float na_y = (float)coarse_table[i].sum_ny * inv_fp_sw_a;
    float na_z = (float)coarse_table[i].sum_nz * inv_fp_sw_a;

    // Number of sub-samples per axis along the crossing plane.
    int subdiv = 1 << level_shift;

    // Check +X, +Y, +Z neighbours for TSDF sign change.
    int off_x[3] = {1, 0, 0};
    int off_y[3] = {0, 1, 0};
    int off_z[3] = {0, 0, 1};

    for (int d = 0; d < 3; d++) {
        int nx = kx + off_x[d];
        int ny = ky + off_y[d];
        int nz = kz + off_z[d];

        // Look up neighbor in coarse table.
        uint nb_idx = hash_lookup(coarse_table, coarse_mask, nx, ny, nz);
        if (nb_idx == 0xFFFFFFFF) continue;

        float sw_b = (float)coarse_table[nb_idx].sum_weight;
        if (sw_b < min_weight_scaled) continue;

        float inv_sw_b    = 1.0f / sw_b;
        float inv_fp_sw_b = inv_sw_b / (float)FP_SCALE;
        float tsdf_b      = (float)coarse_table[nb_idx].sum_tsdf * inv_fp_sw_b;

        // Sign change?
        if ((tsdf_a > 0.0f) == (tsdf_b > 0.0f)) continue;
        if (tsdf_a == 0.0f || tsdf_b == 0.0f) continue;

        // Interpolation factor along the edge.
        float t = tsdf_a / (tsdf_a - tsdf_b);
        t = clamp(t, 0.0f, 1.0f);

        // Coarse voxel centres.
        float pa_x = ((float)kx + 0.5f) * coarse_voxel_size;
        float pa_y = ((float)ky + 0.5f) * coarse_voxel_size;
        float pa_z = ((float)kz + 0.5f) * coarse_voxel_size;

        // Interpolated crossing point (coarse grid).
        float cx = pa_x + t * (float)off_x[d] * coarse_voxel_size;
        float cy = pa_y + t * (float)off_y[d] * coarse_voxel_size;
        float cz = pa_z + t * (float)off_z[d] * coarse_voxel_size;

        // Interpolated normal from coarse.
        float nb_x = (float)coarse_table[nb_idx].sum_nx * inv_fp_sw_b;
        float nb_y = (float)coarse_table[nb_idx].sum_ny * inv_fp_sw_b;
        float nb_z = (float)coarse_table[nb_idx].sum_nz * inv_fp_sw_b;
        float inx = na_x + t * (nb_x - na_x);
        float iny = na_y + t * (nb_y - na_y);
        float inz = na_z + t * (nb_z - na_z);
        float nlen = sqrt(inx*inx + iny*iny + inz*inz);
        if (nlen > 1e-6f) { inx /= nlen; iny /= nlen; inz /= nlen; }

        // Sub-sample the crossing plane at fine resolution.
        // The two axes perpendicular to direction d define the plane.
        // d=0 (edge along X): plane axes are Y, Z
        // d=1 (edge along Y): plane axes are X, Z
        // d=2 (edge along Z): plane axes are X, Y
        for (int s0 = 0; s0 < subdiv; s0++) {
            for (int s1 = 0; s1 < subdiv; s1++) {
                // Offset within the coarse voxel face, in world units.
                float off0 = ((float)s0 + 0.5f) * fine_voxel_size
                             - 0.5f * coarse_voxel_size;
                float off1 = ((float)s1 + 0.5f) * fine_voxel_size
                             - 0.5f * coarse_voxel_size;

                // Compute sub-sample world position.
                float sx, sy, sz;
                if (d == 0) {
                    sx = cx; sy = cy + off0; sz = cz + off1;
                } else if (d == 1) {
                    sx = cx + off0; sy = cy; sz = cz + off1;
                } else {
                    sx = cx + off0; sy = cy + off1; sz = cz;
                }

                // Convert to fine voxel coordinate and check fine table.
                float inv_fine_vs = 1.0f / fine_voxel_size;
                int fx = (int)floor(sx * inv_fine_vs);
                int fy = (int)floor(sy * inv_fine_vs);
                int fz = (int)floor(sz * inv_fine_vs);

                uint fine_slot = hash_lookup(fine_table, fine_mask, fx, fy, fz);
                if (fine_slot != 0xFFFFFFFF) {
                    // Fine table has this voxel — check if it has enough weight.
                    float fine_sw = (float)fine_table[fine_slot].sum_weight;
                    if (fine_sw >= min_weight_scaled) {
                        continue;  // Fine already covers this point.
                    }
                }

                // No fine coverage — emit this point.
                uint out_idx = atomic_add(out_counter, 1u);
                if (out_idx >= max_output) return;

                out_points[out_idx * 3 + 0] = sx;
                out_points[out_idx * 3 + 1] = sy;
                out_points[out_idx * 3 + 2] = sz;
                out_normals[out_idx * 3 + 0] = inx;
                out_normals[out_idx * 3 + 1] = iny;
                out_normals[out_idx * 3 + 2] = inz;
                // Colors baked post-extraction.
                out_colors[out_idx * 3 + 0] = 128;
                out_colors[out_idx * 3 + 1] = 128;
                out_colors[out_idx * 3 + 2] = 128;
            }
        }
    }
}
)CL"
    R"CL(

// =====================================================================
// Photometric refinement kernels (image2d_array_t path)
//
// All views share the same (W, H) resolution (same-resolution assumption).
// color_images: CL_RGBA CL_UNORM_INT8, one layer per view, [0,1] float4.
// tsdf_depths:  CL_R   CL_FLOAT, one layer per view, TSDF-rendered depth.
// =====================================================================

// ---- Atomic float add via CAS loop ----
void atomic_add_f(__global float* addr, float val) {
    union { uint u; float f; } expected, desired;
    expected.f = *addr;
    do {
        desired.f = expected.f + val;
        uint old = atomic_cmpxchg((__global volatile uint*)addr,
                                  expected.u, desired.u);
        if (old == expected.u) break;
        expected.u = old;
    } while (true);
}

// ---- Trilinear SDF sampling helpers (unchanged) ----
// Decode the weighted-average TSDF from a hash table slot.
float decode_tsdf(__global const VoxelSlot* table, uint slot_idx) {
    float sw = (float)table[slot_idx].sum_weight;
    if (sw < 1.0f) return 1.0f;  // empty → outside
    return (float)table[slot_idx].sum_tsdf / (sw * (float)FP_SCALE);
}

// Sample TSDF at an arbitrary world-space point via trilinear interpolation.
// Returns 1.0 if any of the 8 corner voxels are missing (unknown region).
float sample_sdf_trilinear(
    __global const VoxelSlot* table, uint mask,
    float3 p, float inv_vs, float min_weight_scaled)
{
    // Fractional voxel coordinates.
    float fx = p.x * inv_vs - 0.5f;
    float fy = p.y * inv_vs - 0.5f;
    float fz = p.z * inv_vs - 0.5f;
    int ix = (int)floor(fx);
    int iy = (int)floor(fy);
    int iz = (int)floor(fz);
    float tx = fx - (float)ix;
    float ty = fy - (float)iy;
    float tz = fz - (float)iz;

    float vals[8];
    int offsets[8][3] = {
        {0,0,0}, {1,0,0}, {0,1,0}, {1,1,0},
        {0,0,1}, {1,0,1}, {0,1,1}, {1,1,1}
    };
    for (int c = 0; c < 8; c++) {
        uint si = hash_lookup(table, mask,
                              ix + offsets[c][0],
                              iy + offsets[c][1],
                              iz + offsets[c][2]);
        if (si == 0xFFFFFFFF) return 1.0f;
        float sw = (float)table[si].sum_weight;
        if (sw < min_weight_scaled) return 1.0f;
        vals[c] = (float)table[si].sum_tsdf / (sw * (float)FP_SCALE);
    }

    // Trilinear interpolation.
    float c00 = vals[0] * (1.0f - tx) + vals[1] * tx;
    float c10 = vals[2] * (1.0f - tx) + vals[3] * tx;
    float c01 = vals[4] * (1.0f - tx) + vals[5] * tx;
    float c11 = vals[6] * (1.0f - tx) + vals[7] * tx;
    float c0  = c00 * (1.0f - ty) + c10 * ty;
    float c1  = c01 * (1.0f - ty) + c11 * ty;
    return c0 * (1.0f - tz) + c1 * tz;
}

// ---- SDF gradient via central differences ----
// Nearest-neighbor SDF sample: single hash lookup, no 8-corner requirement.
// Returns 1.0 if the voxel is missing or under-weighted.
float sample_sdf_nearest(
    __global const VoxelSlot* table, uint mask,
    float3 p, float inv_vs, float min_weight_scaled)
{
    int kx = (int)floor(p.x * inv_vs);
    int ky = (int)floor(p.y * inv_vs);
    int kz = (int)floor(p.z * inv_vs);
    uint si = hash_lookup(table, mask, kx, ky, kz);
    if (si == 0xFFFFFFFF) return 1.0f;
    float sw = (float)table[si].sum_weight;
    if (sw < min_weight_scaled) return 1.0f;
    return (float)table[si].sum_tsdf / (sw * (float)FP_SCALE);
}

float3 compute_sdf_gradient(
    __global const VoxelSlot* table, uint mask,
    float3 p, float inv_vs, float eps, float min_weight_scaled)
{
    float3 grad;
    grad.x = sample_sdf_nearest(table, mask,
                 p + (float3)(eps, 0.0f, 0.0f), inv_vs, min_weight_scaled)
           - sample_sdf_nearest(table, mask,
                 p - (float3)(eps, 0.0f, 0.0f), inv_vs, min_weight_scaled);
    grad.y = sample_sdf_nearest(table, mask,
                 p + (float3)(0.0f, eps, 0.0f), inv_vs, min_weight_scaled)
           - sample_sdf_nearest(table, mask,
                 p - (float3)(0.0f, eps, 0.0f), inv_vs, min_weight_scaled);
    grad.z = sample_sdf_nearest(table, mask,
                 p + (float3)(0.0f, 0.0f, eps), inv_vs, min_weight_scaled)
           - sample_sdf_nearest(table, mask,
                 p - (float3)(0.0f, 0.0f, eps), inv_vs, min_weight_scaled);
    return grad / (2.0f * eps);
}

// ---- image2d_array helpers ----

// Grayscale from RGBA UNORM_INT8 image array layer.
// color_images: CL_RGBA CL_UNORM_INT8, values in [0,1].
float gray_from_array(read_only image2d_array_t images,
                      const sampler_t samp,
                      float u, float v, int layer)
{
    float4 c = read_imagef(images, samp, (float4)(u + 0.5f, v + 0.5f, (float)layer, 0.0f));
    return (c.x + c.y + c.z) * 0.333333f;
}

// Color (rgb) from image array layer.
float3 color_from_array(read_only image2d_array_t images,
                        const sampler_t samp,
                        float u, float v, int layer)
{
    float4 c = read_imagef(images, samp, (float4)(u + 0.5f, v + 0.5f, (float)layer, 0.0f));
    return c.xyz;
}

// Grayscale image gradient via ±1 central diff, hardware bilinear.
float2 image_grad_array(read_only image2d_array_t images,
                        const sampler_t samp,
                        float u, float v, int layer)
{
    float gxp = gray_from_array(images, samp, u + 1.0f, v,        layer);
    float gxm = gray_from_array(images, samp, u - 1.0f, v,        layer);
    float gyp = gray_from_array(images, samp, u,         v + 1.0f, layer);
    float gym = gray_from_array(images, samp, u,         v - 1.0f, layer);
    return (float2)((gxp - gxm) * 0.5f, (gyp - gym) * 0.5f);
}

// Bilateral Gaussian-windowed ZNCC between two layers of the same image array.
//
// Replicates compute_ncc_at_stride from opencl_kernels.h but reads both
// ref and src from the same image2d_array_t via their respective layer indices.
//
// inv_sigma_s2: spatial weight exponent, e.g. -0.5 for sigma_s=1 pixel
//
// Returns (ncc_score, d2ncc, var_r, var_s):
//   ncc_score   ∈ [-1,1]
//   d2ncc       = ∂NCC/∂I_j at center pixel (src layer center), per Pons-Keriven eq.(13)
//               = w_center × (ref_norm_center - NCC×src_norm_center) / (σ_src × Σw)
//   var_r,var_s = windowed patch variances (texture/contrast confidence)
// Returns (-1,0,0,0) when either patch variance < 1.5e-6 (flat/uniform region).
float4 compute_zncc_array(
    read_only image2d_array_t images,
    int ref_layer, float ref_cx, float ref_cy,
    int src_layer, float src_cx, float src_cy,
    int half_patch,
    float inv_sigma_s2)
{
    const sampler_t samp = CLK_NORMALIZED_COORDS_FALSE |
                           CLK_ADDRESS_CLAMP_TO_EDGE |
                           CLK_FILTER_LINEAR;

    float ref_center = gray_from_array(images, samp, ref_cx, ref_cy, ref_layer);
    float src_center = gray_from_array(images, samp, src_cx, src_cy, src_layer);

    float sum_w  = 0.0f;
    float sum_r  = 0.0f, sum_s  = 0.0f;
    float sum_rr = 0.0f, sum_ss = 0.0f, sum_rs = 0.0f;
    float w_center = 0.0f;

    for (int iy = -half_patch; iy <= half_patch; iy++) {
        for (int ix = -half_patch; ix <= half_patch; ix++) {
            float fdx = (float)ix, fdy = (float)iy;
            float ref_g = gray_from_array(images, samp, ref_cx + fdx, ref_cy + fdy, ref_layer);
            float src_g = gray_from_array(images, samp, src_cx + fdx, src_cy + fdy, src_layer);

            float dr = ref_g - ref_center;
            float w = exp(inv_sigma_s2 * (fdx*fdx + fdy*fdy));

            float r = dr;                  // centered at ref_center
            float s = src_g - src_center;  // centered at src_center

            sum_r  += w * r;
            sum_s  += w * s;
            sum_rr += w * r * r;
            sum_ss += w * s * s;
            sum_rs += w * r * s;
            sum_w  += w;
            if (ix == 0 && iy == 0) w_center = w;
        }
    }

    if (sum_w < 1e-6f) return (float4)(-1.0f, 0.0f, 0.0f, 0.0f);

    float inv_w  = 1.0f / sum_w;
    float mean_r = sum_r * inv_w;
    float mean_s = sum_s * inv_w;
    float var_r  = sum_rr * inv_w - mean_r * mean_r;
    float var_s  = sum_ss * inv_w - mean_s * mean_s;

    if (var_r < 1.5e-6f || var_s < 1.5e-6f) return (float4)(-1.0f, 0.0f, 0.0f, 0.0f);

    float sigma_r = sqrt(var_r);
    float sigma_s = sqrt(var_s);
    float covar   = sum_rs * inv_w - mean_r * mean_s;
    float ncc     = covar / (sigma_r * sigma_s);

    // Pons-Keriven ∂₂M_CC at center pixel (eq.13):
    //   ∂₂M = w_center × (I_ref_norm_center − NCC × I_src_norm_center) / (σ_s × Σw)
    // where I_ref_norm_center = (ref_center − true_mean_ref) / σ_r = −mean_r / σ_r
    //       I_src_norm_center = (src_center − true_mean_src) / σ_s = −mean_s / σ_s
    float ref_norm_c = -mean_r / sigma_r;
    float src_norm_c = -mean_s / sigma_s;
    float d2ncc = w_center * (ref_norm_c - ncc * src_norm_c) / (sigma_s * sum_w);

    return (float4)(ncc, d2ncc, var_r, var_s);
}

)CL"
    R"CL(

// =====================================================================
// svo_refine_clear — zero gradient (1 float/slot) + Adam (2 floats/slot)
// =====================================================================
__kernel void svo_refine_clear(
    __global float* grad_buf,   // 1 float per slot
    __global float* grad_w_buf, // 1 float per slot
    __global float* adam_buf,   // 2 floats per slot
    uint capacity,
    int  clear_adam)            // 1 = also clear Adam state
{
    uint i = get_global_id(0);
    if (i >= capacity) return;
    grad_buf[i] = 0.0f;
    grad_w_buf[i] = 0.0f;
    if (clear_adam) {
        adam_buf[i * 2 + 0] = 0.0f;  // m_d
        adam_buf[i * 2 + 1] = 0.0f;  // v_d
    }
}

// =====================================================================
// svo_refine_accumulate — per-pixel bilateral ZNCC gradient accumulation
//
// Dispatched once per source view: global_size = (img_width, img_height).
// For each pixel, ray-march the TSDF to find the surface point, then
// project into all other views and accumulate the photometric SDF gradient
// using bilateral Gaussian-windowed ZNCC (Pons-Keriven 2005/PAMI 2007).
//
// color_images: CL_RGBA CL_UNORM_INT8, one slice per view.
// tsdf_depths:  CL_R   CL_FLOAT, one slice per view (TSDF-rendered; used
//               for occlusion — updated each iteration before this kernel).
// =====================================================================
__kernel void svo_refine_accumulate(
    __global const VoxelSlot* table,
    uint                      capacity_mask,
    __global float*           grad_buf,          // 1 float per slot
    __global float*           grad_w_buf,        // 1 float per slot
    read_only image2d_array_t color_images,      // CL_RGBA UNORM_INT8
    read_only image2d_array_t tsdf_depths,       // CL_R FLOAT
    __constant SVOCamera*     cameras,
    __global const int*       neighbor_buf,      // n_views × max_neighbors
    int                       max_neighbors,     // K (e.g. 4)
    int                       n_views,
    int                       src_view,
    int                       img_width,
    int                       img_height,
    float                     voxel_size,
    float                     inv_voxel_size,
    float                     trunc_dist,
    float                     min_weight_scaled,
    int                       half_patch,
    float                     inv_sigma_s2,
    float                     tex_floor)
{
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (col >= img_width || row >= img_height) return;

    // Border margin: half_patch + 1 gradient sample.
    int margin = half_patch + 2;
    if (col < margin || col >= img_width  - margin ||
        row < margin || row >= img_height - margin) return;

    // Nearest-neighbor sampler for depth lookup.
    const sampler_t nn_samp = CLK_NORMALIZED_COORDS_FALSE |
                              CLK_ADDRESS_CLAMP_TO_EDGE |
                              CLK_FILTER_NEAREST;

    // Depth of this pixel in the TSDF-rendered surface.
    float d = read_imagef(tsdf_depths, nn_samp,
                          (float4)((float)col + 0.5f, (float)row + 0.5f,
                                   (float)src_view, 0.0f)).x;
    if (d <= 0.0f) return;

    // ---- Back-project pixel to world ----
    __constant SVOCamera* cam = &cameras[src_view];
    float uc = (float)col, vr = (float)row;
    float3 p_cam;
    p_cam.x = d * (cam->Kinv[0]*uc + cam->Kinv[1]*vr + cam->Kinv[2]);
    p_cam.y = d * (cam->Kinv[3]*uc + cam->Kinv[4]*vr + cam->Kinv[5]);
    p_cam.z = d * (cam->Kinv[6]*uc + cam->Kinv[7]*vr + cam->Kinv[8]);

    float3 diff = (float3)(p_cam.x - cam->t[0],
                           p_cam.y - cam->t[1],
                           p_cam.z - cam->t[2]);
    float3 x_surface;
    x_surface.x = cam->Rinv[0]*diff.x + cam->Rinv[1]*diff.y + cam->Rinv[2]*diff.z;
    x_surface.y = cam->Rinv[3]*diff.x + cam->Rinv[4]*diff.y + cam->Rinv[5]*diff.z;
    x_surface.z = cam->Rinv[6]*diff.x + cam->Rinv[7]*diff.y + cam->Rinv[8]*diff.z;

    // ---- Surface normal via central-difference SDF gradient ----
    // eps must be >= 1.5*voxel_size so nearest-neighbor samples are guaranteed
    // to land in distinct voxels regardless of sub-voxel position.
    float eps = 1.5f * voxel_size;
    float3 grad_sdf = compute_sdf_gradient(table, capacity_mask,
                          x_surface, inv_voxel_size, eps, min_weight_scaled);
    float grad_len = length(grad_sdf);
    if (grad_len < 1e-6f) return;
    float3 normal = grad_sdf / grad_len;

    // ---- Find nearest voxel slot for gradient deposition ----
    // Trilinear 8-corner distribution is ideal but fails at truncation band
    // edges where neighbor voxels don't exist. Use single nearest slot.
    int nearest_kx = (int)floor(x_surface.x * inv_voxel_size);
    int nearest_ky = (int)floor(x_surface.y * inv_voxel_size);
    int nearest_kz = (int)floor(x_surface.z * inv_voxel_size);
    uint nearest_slot = hash_lookup(table, capacity_mask, nearest_kx, nearest_ky, nearest_kz);
    if (nearest_slot == 0xFFFFFFFF) return;

    // ---- Accumulate photometric gradient from neighbor views ----
    float total_grad_d = 0.0f;
    float total_w      = 0.0f;  // sum of texture-confidence weights

    const int nb_offset = src_view * max_neighbors;
    for (int k = 0; k < max_neighbors; k++) {
        int j = neighbor_buf[nb_offset + k];
        if (j < 0) break;  // sentinel: end of neighbor list

        // Project surface point into view j.
        __constant SVOCamera* cam_j = &cameras[j];
        float3 p_j;
        p_j.x = cam_j->R[0]*x_surface.x + cam_j->R[1]*x_surface.y + cam_j->R[2]*x_surface.z + cam_j->t[0];
        p_j.y = cam_j->R[3]*x_surface.x + cam_j->R[4]*x_surface.y + cam_j->R[5]*x_surface.z + cam_j->t[1];
        p_j.z = cam_j->R[6]*x_surface.x + cam_j->R[7]*x_surface.y + cam_j->R[8]*x_surface.z + cam_j->t[2];
        if (p_j.z <= 0.0f) continue;

        float inv_zj = 1.0f / p_j.z;
        // Recover fx, fy from Kinv: Kinv row-0 = (1/fx, 0, -cx/fx), so fx = 1/Kinv[0].
        float fx_j = 1.0f / (cam_j->Kinv[0] + 1e-12f);
        float fy_j = 1.0f / (cam_j->Kinv[4] + 1e-12f);
        float cx_j = -cam_j->Kinv[2] * fx_j;
        float cy_j = -cam_j->Kinv[5] * fy_j;
        float px = fx_j * p_j.x * inv_zj + cx_j;
        float py = fy_j * p_j.y * inv_zj + cy_j;

        // Bounds check (same resolution for all views).
        if (px < (float)margin || px >= (float)(img_width  - margin) ||
            py < (float)margin || py >= (float)(img_height - margin)) continue;

        // Validity mask check: alpha channel at nearest pixel.
        float alpha_j = read_imagef(color_images, nn_samp,
                            (float4)(px + 0.5f, py + 0.5f, (float)j, 0.0f)).w;
        if (alpha_j < 0.5f) continue;  // masked pixel

        // TSDF-rendered occlusion check: absolute depth margin (3 voxels).
        float tsdf_d = read_imagef(tsdf_depths, nn_samp,
                           (float4)(px + 0.5f, py + 0.5f, (float)j, 0.0f)).x;
        if (tsdf_d > 0.0f && p_j.z > tsdf_d + 3.0f * voxel_size) continue;

        // Viewing angle (cos between surface normal and view direction).
        float3 cam_j_pos = (float3)(cam_j->cam_pos[0], cam_j->cam_pos[1], cam_j->cam_pos[2]);
        float3 ray_j = normalize(x_surface - cam_j_pos);
        float cos_angle = fabs(dot(normal, ray_j));
        if (cos_angle < 0.15f) continue;

        // ---- Bilateral ZNCC between source and target views ----
        float4 zncc_result = compute_zncc_array(
            color_images,
            src_view, (float)col, (float)row,
            j, px, py,
            half_patch, inv_sigma_s2);

        float ncc   = zncc_result.x;
        float d2ncc = zncc_result.y;
        float var_r = zncc_result.z;
        float var_s = zncc_result.w;
        if (ncc < -0.5f) continue;  // failed or flat patch

        // ---- SDF gradient: Pons-Keriven eq.(14) ----
        // Normal in camera j frame.
        float n_cam_x = cam_j->R[0]*normal.x + cam_j->R[1]*normal.y + cam_j->R[2]*normal.z;
        float n_cam_y = cam_j->R[3]*normal.x + cam_j->R[4]*normal.y + cam_j->R[5]*normal.z;
        float n_cam_z = cam_j->R[6]*normal.x + cam_j->R[7]*normal.y + cam_j->R[8]*normal.z;

        // Projected normal derivative: full perspective Jacobian of the
        // projection (px,py) = (fx·X/z+cx, fy·Y/z+cy) w.r.t. moving the
        // surface point along the world normal (X,Y,z) += δ·n_cam:
        //   ∂px/∂δ = (1/z)·[fx·n_cam_x − (px−cx)·n_cam_z]
        //   ∂py/∂δ = (1/z)·[fy·n_cam_y − (py−cy)·n_cam_z]
        // The −(px−cx)·n_cam_z term is the dominant one for camera-facing
        // surfaces (n_cam ≈ (0,0,−1)); dropping it leaves an essentially
        // random gradient that destroys the surface.
        float du_dn = inv_zj * (fx_j * n_cam_x - (px - cx_j) * n_cam_z);
        float dv_dn = inv_zj * (fy_j * n_cam_y - (py - cy_j) * n_cam_z);

        // Image gradient ∇I_j at projected point.
        const sampler_t lin_samp = CLK_NORMALIZED_COORDS_FALSE |
                                   CLK_ADDRESS_CLAMP_TO_EDGE |
                                   CLK_FILTER_LINEAR;
        float2 ig = image_grad_array(color_images, lin_samp, px, py, j);

        // Chain rule: ∂E/∂φ = d2ncc × ∇I_j·(du_dn,dv_dn) × cos_angle / |∇φ|
        // where E = 1 - NCC. Update kernel does φ_new = φ - lr*grad → descent.
        float proj_grad = ig.x * du_dn + ig.y * dv_dn;
        float g = d2ncc * proj_grad * cos_angle / (grad_len + 1e-6f);

        // Texture confidence: the ZNCC derivative carries a 1/σ factor that
        // inflates low-contrast (low-texture) gradients to full magnitude even
        // though they are noise-dominated.  Weight each view by its patch
        // contrast (min variance over ref/src) to suppress that noise.
        float w_tex = fmin(var_r, var_s);
        total_grad_d += g * w_tex;  // accumulate weighted ∂E/∂φ for descent
        total_w      += w_tex;
    }

    if (total_w <= 0.0f) return;

    // Confidence-weighted mean with a denominator floor: when the total
    // texture confidence is small (low-texture surface point), dividing by
    // (total_w + tex_floor) shrinks the step so the voxel keeps the smooth
    // fused TSDF instead of speckling.  Well-textured points (total_w >>
    // tex_floor) are essentially unaffected — a plain weighted mean.
    total_grad_d /= (total_w + tex_floor);

    // Deposit gradient to nearest voxel slot.
    atomic_add_f(&grad_buf[nearest_slot], total_grad_d);
    atomic_add_f(&grad_w_buf[nearest_slot], 1.0f);  // count the number of accumulated gradients for this slot
}

)CL"
    R"CL(

// =====================================================================
// svo_refine_update — SGD-momentum update + edge-aware anisotropic reg
//
// One work-item per hash table slot.
// grad_buf: 1 float/slot.  adam_buf: 2 floats/slot (m_d, v_d).
// When lambda_reg == 0 only data-driven voxels update (fast path).  When
// lambda_reg > 0, a gravity-aware tangent-aligned diffusion runs on every
// surface voxel: it flattens facets into planes and sharpens the creases
// where they meet (edge-stop + tangential anisotropy), with a Z-up bias
// toward horizontal/vertical surfaces for man-made 90° structure.
// =====================================================================
__kernel void svo_refine_update(
    __global VoxelSlot* table,
    uint                capacity_mask,
    uint                capacity,
    __global float*     grad_buf,         // 1 float per slot
    __global float*     grad_w_buf,       // 1 float per slot
    __global float*     adam_buf,         // 2 floats per slot
    float               min_weight_scaled,
    float               voxel_size,
    float               lambda_reg,       // 0 = no regularization
    float               lr_sdf,
    float               beta1,
    float               epsilon,
    int                 iteration,
    int                 n_views,
    float               edge_k)           // edge-stop threshold (norm. TSDF units)
{
    uint i = get_global_id(0);
    if (i >= capacity) return;

    uint key_ab = table[i].key_ab;
    if (key_ab == EMPTY_KEY) return;
    if (table[i].count == 0) return;

    float sw = (float)table[i].sum_weight;
    if (sw < min_weight_scaled) return;

    float inv_fp_sw = 1.0f / (sw * (float)FP_SCALE);
    float tsdf = (float)table[i].sum_tsdf * inv_fp_sw;
    // Only update voxels within the truncation band.
    if (fabs(tsdf) > 4.f) return;

    float grad_d = grad_buf[i];
    float grad_w = grad_w_buf[i];
    // Clear for next iteration.
    grad_buf[i] = 0.0f;
    grad_w_buf[i] = 0.0f;

    grad_d /= (grad_w + 1e-6f);  // mean gradient if multiple pixels contributed

    // Fast path: with regularization off, only voxels touched by the data
    // term need updating.  With it on, the diffusion runs on every surface
    // voxel below (including low-texture ones the data term now suppresses),
    // so we must not early-out on a zero data gradient.
    if (lambda_reg <= 0.0f && fabs(grad_d) < 1e-10f) return;

    int kx = (int)((key_ab >> 16) & 0xFFFF) - 32768;
    int ky = (int)(key_ab & 0xFFFF) - 32768;
    int kz = table[i].key_c;

    // ---- Edge-aware anisotropic regularization (flatten facets, sharpen creases) ----
    if (lambda_reg > 0.0f) {
        // Gather the 6 face-neighbor TSDF values (+x,-x,+y,-y,+z,-z).
        const int dx[6] = {1,-1, 0, 0, 0, 0};
        const int dy[6] = {0, 0, 1,-1, 0, 0};
        const int dz[6] = {0, 0, 0, 0, 1,-1};
        float nval[6];
        int   npres[6];
        for (int nb = 0; nb < 6; nb++) {
            uint ni = hash_lookup(table, capacity_mask,
                                  kx + dx[nb], ky + dy[nb], kz + dz[nb]);
            if (ni == 0xFFFFFFFF) { npres[nb] = 0; nval[nb] = 0.0f; continue; }
            float nsw = (float)table[ni].sum_weight;
            if (nsw < min_weight_scaled) { npres[nb] = 0; nval[nb] = 0.0f; continue; }
            npres[nb] = 1;
            nval[nb] = (float)table[ni].sum_tsdf / (nsw * (float)FP_SCALE);
        }

        // Local surface normal n = ∇φ via central differences (one-sided at
        // missing neighbors).  φ increases toward the empty side, so n points
        // outward; only its direction matters here.
        float3 n = (float3)(0.0f, 0.0f, 0.0f);
        if      (npres[0] && npres[1]) n.x = nval[0] - nval[1];
        else if (npres[0])             n.x = nval[0] - tsdf;
        else if (npres[1])             n.x = tsdf    - nval[1];
        if      (npres[2] && npres[3]) n.y = nval[2] - nval[3];
        else if (npres[2])             n.y = nval[2] - tsdf;
        else if (npres[3])             n.y = tsdf    - nval[3];
        if      (npres[4] && npres[5]) n.z = nval[4] - nval[5];
        else if (npres[4])             n.z = nval[4] - tsdf;
        else if (npres[5])             n.z = tsdf    - nval[5];

        float nlen = length(n);
        if (nlen > 1e-6f) {
            n /= nlen;

            // Anisotropic weighted Laplacian: weight each axis-neighbor by its
            // tangentiality (1 − (axis·n)²) so diffusion runs ALONG the facet
            // (flattening it) but not ACROSS the normal (preserving the crease).
            // An edge-stop exp(−Δ²/K²) further halts diffusion across large TSDF
            // jumps (corners), sharpening rather than rounding them.
            float axis_n2[3] = {n.x*n.x, n.y*n.y, n.z*n.z};
            float wsum = 0.0f, acc = 0.0f;
            for (int nb = 0; nb < 6; nb++) {
                if (!npres[nb]) continue;
                float tang = 1.0f - axis_n2[nb >> 1];
                if (tang < 1e-4f) continue;
                float diff  = nval[nb] - tsdf;
                float estop = exp(-(diff * diff) / (edge_k * edge_k));
                float a = tang * estop;
                acc  += a * diff;
                wsum += a;
            }
            if (wsum > 1e-6f) {
                // Descent gradient pulling tsdf toward the anisotropic neighbor
                // mean: Δtsdf = +α(mean−tsdf) via new = tsdf − lr·grad.
                grad_d -= lambda_reg * 2.0f * (acc / wsum);
            }
        }
    }

    // Nothing moved this voxel (no data gradient, flat neighborhood).
    if (fabs(grad_d) < 1e-10f) return;

    // ---- Gradient clipping: prevent steps > 0.1 in normalized TSDF ----
    float max_grad = 0.25f;
    grad_d = clamp(grad_d, -max_grad, max_grad);

    // ---- SGD with momentum ----
    float m_d = adam_buf[i * 2 + 0];
    m_d = beta1 * m_d + (1.0f - beta1) * grad_d;
    adam_buf[i * 2 + 0] = m_d;

    float delta_d = lr_sdf * m_d;
    float new_tsdf = clamp(tsdf - delta_d, -1.0f, 1.0f);
    table[i].sum_tsdf = (int)(new_tsdf * sw * (float)FP_SCALE);
}

// =====================================================================
// Color-space helpers for photometric blending.
// CL_UNORM_INT8 samples are sRGB-encoded; mixing colors should happen in
// linear light.  Gamma 2.2 is an adequate approximation of the sRGB curve.
// =====================================================================
float3 srgb_to_linear(float3 c) {
    return pow(fmax(c, (float3)(0.0f)), (float3)(2.2f));
}
float3 linear_to_srgb(float3 c) {
    return pow(fmax(c, (float3)(0.0f)), (float3)(1.0f / 2.2f));
}

// =====================================================================
// svo_bake_colors — Robust multi-view color baking (consensus + sharpen)
//
// Dispatched 1D over M extracted surface points.  Two stages:
//   Phase 0 (consensus gate): gather all valid views, then estimate a
//     robust consensus color via median initialization + MAD scale +
//     iterated Tukey-biweight reweighting.  Gross outliers (occlusion
//     leaks, specular highlights, moved objects) receive a hard-zero
//     weight, so they neither bias the consensus nor pass the gate.
//   Phase 1 (sharpening): among the inlier views, pick the top n_final
//     by resolution (inverse ground-sample-distance) and blend them in
//     linear color space, weighted by resolution.  Selecting a few sharp
//     views — rather than averaging all of them — preserves texture
//     detail that an N-view weighted mean would low-pass into blur.
//
// color_images: CL_RGBA CL_UNORM_INT8, slice per view.
//   Alpha channel encodes validity mask (0 = invalid pixel).
// clean_depths: CL_R CL_FLOAT, slice per view (front-most clean depth).
// occlusion_margin: absolute world-space depth tolerance (e.g. 3*voxel).
// n_final:     number of sharpest inlier views to blend (1 or 2).
// irls_iters:  Tukey reweighting iterations for the consensus (e.g. 3).
// =====================================================================
__kernel void svo_bake_colors(
    __global const float*    points,
    __global const float*    normals,
    __global uchar*          out_colors,       // M × 3 RGB bytes
    read_only image2d_array_t color_images,    // RGBA with alpha = mask
    read_only image2d_array_t clean_depths,    // front-most clean depth
    __constant SVOCamera*    cameras,
    int                      n_views,
    int                      img_width,
    int                      img_height,
    float                    occlusion_margin,  // absolute depth margin
    int                      n_final,           // # sharpest views to blend
    int                      irls_iters,        // Tukey reweight iterations
    __global const uchar*    relax_occ,         // per-point: 1 = skip occlusion
    int                      has_relax,         // relax_occ buffer provided?
    int                      n_points)          // bound guard (M)
{
    uint gid = get_global_id(0);
    if (gid >= (uint)n_points) return;   // dispatch is padded to 256

    // Reliably-interpolated filled-DSM cells live where NO view got depth (that
    // is why they were holes), so their projection lands on masked (alpha=0)
    // pixels in every view.  For them, relax (skip) the depth-validity MASK gate
    // so the real image colour — which exists in the colour image even where the
    // cleaned depth does not — is still sampled.  Occlusion stays strict (it is
    // a no-op for true holes, where clean_d==0, but rejects badly-placed
    // geometry elsewhere); the flag is NOT set for extrapolated boundary cells.
    int relax_pt = has_relax && (relax_occ[gid] != 0);

    float3 x = (float3)(points[gid*3], points[gid*3+1], points[gid*3+2]);
    float3 norm = (float3)(normals[gid*3], normals[gid*3+1], normals[gid*3+2]);

    const sampler_t nn_samp  = CLK_NORMALIZED_COORDS_FALSE |
                               CLK_ADDRESS_CLAMP_TO_EDGE |
                               CLK_FILTER_NEAREST;
    const sampler_t lin_samp = CLK_NORMALIZED_COORDS_FALSE |
                               CLK_ADDRESS_CLAMP_TO_EDGE |
                               CLK_FILTER_LINEAR;

    // Gather phase: collect valid observations.
    // Private arrays; n_views is bounded by host to KMAX.
    const int KMAX = 36;
    float obs_r[36], obs_g[36], obs_b[36];
    float obs_w[36];    // geometric prior = cos^2(theta)
    float obs_res[36];  // resolution score = fx * cos(theta) / z  (~1/GSD)
    int n_valid = 0;

    for (int j = 0; j < n_views && j < KMAX; j++) {
        __constant SVOCamera* cam_j = &cameras[j];

        // Project point into view j.
        float3 p_j;
        p_j.x = cam_j->R[0]*x.x + cam_j->R[1]*x.y + cam_j->R[2]*x.z + cam_j->t[0];
        p_j.y = cam_j->R[3]*x.x + cam_j->R[4]*x.y + cam_j->R[5]*x.z + cam_j->t[1];
        p_j.z = cam_j->R[6]*x.x + cam_j->R[7]*x.y + cam_j->R[8]*x.z + cam_j->t[2];
        if (p_j.z <= 0.0f) continue;

        float inv_zj = 1.0f / p_j.z;
        float fx_j = 1.0f / (cam_j->Kinv[0] + 1e-12f);
        float fy_j = 1.0f / (cam_j->Kinv[4] + 1e-12f);
        float cx_j = -cam_j->Kinv[2] * fx_j;
        float cy_j = -cam_j->Kinv[5] * fy_j;
        float px = fx_j * p_j.x * inv_zj + cx_j;
        float py = fy_j * p_j.y * inv_zj + cy_j;

        if (px < 1.0f || px >= (float)(img_width  - 1) ||
            py < 1.0f || py >= (float)(img_height - 1)) continue;

        // Validity mask check: alpha channel at nearest pixel.
        float4 rgba_nn = read_imagef(color_images, nn_samp,
                             (float4)(px + 0.5f, py + 0.5f, (float)j, 0.0f));
        // Masked pixel (alpha=0 = no cleaned depth here).  Filled cells relax
        // this: their colour exists in the image even where depth does not.
        if (rgba_nn.w < 0.5f && !relax_pt) continue;

        // Occlusion check: absolute depth margin (not relative!).
        // If this view's clean surface is in front of our point, skip.
        float clean_d = read_imagef(clean_depths, nn_samp,
                           (float4)(px + 0.5f, py + 0.5f, (float)j, 0.0f)).x;
        if (clean_d > 0.0f && p_j.z > clean_d + occlusion_margin) continue;

        // Viewing angle: cos between normal and view direction.
        float3 cam_pos_j = (float3)(cam_j->cam_pos[0], cam_j->cam_pos[1], cam_j->cam_pos[2]);
        float3 view_dir  = normalize(cam_pos_j - x);
        float cos_theta  = dot(norm, view_dir);
        if (cos_theta < 0.2f) continue;  // grazing angle

        // Sample color with bilinear interpolation.
        float4 rgba = read_imagef(color_images, lin_samp,
                          (float4)(px + 0.5f, py + 0.5f, (float)j, 0.0f));

        obs_r[n_valid]   = rgba.x;
        obs_g[n_valid]   = rgba.y;
        obs_b[n_valid]   = rgba.z;
        obs_w[n_valid]   = cos_theta * cos_theta;        // geometric prior
        obs_res[n_valid] = fx_j * cos_theta * inv_zj;    // ~ pixels / world unit
        n_valid++;
    }

    // Fallback: no valid observation.  Emit black (0) — NOT grey (128) — so
    // the post-process ortho hole-fill (keyed on colour sum==0) fills these
    // cells from their neighbours instead of leaving a grey patch.
    if (n_valid == 0) {
        out_colors[gid * 3 + 0] = 0;
        out_colors[gid * 3 + 1] = 0;
        out_colors[gid * 3 + 2] = 0;
        return;
    }

    // Single observation: nothing to robustify or select — emit it directly.
    if (n_valid == 1) {
        out_colors[gid * 3 + 0] = (uchar)clamp(obs_r[0] * 255.0f, 0.0f, 255.0f);
        out_colors[gid * 3 + 1] = (uchar)clamp(obs_g[0] * 255.0f, 0.0f, 255.0f);
        out_colors[gid * 3 + 2] = (uchar)clamp(obs_b[0] * 255.0f, 0.0f, 255.0f);
        return;
    }

    // ---- Phase 0: robust consensus (median init + MAD + Tukey IRLS) ----
    float scratch[36];

    // Component-wise median initialization (insertion sort per channel).
    for (int i = 0; i < n_valid; i++) scratch[i] = obs_r[i];
    for (int i = 1; i < n_valid; i++) {
        float key = scratch[i]; int k = i - 1;
        while (k >= 0 && scratch[k] > key) { scratch[k+1] = scratch[k]; k--; }
        scratch[k+1] = key;
    }
    float mu_r = (n_valid & 1) ? scratch[n_valid/2]
                   : 0.5f*(scratch[n_valid/2 - 1] + scratch[n_valid/2]);

    for (int i = 0; i < n_valid; i++) scratch[i] = obs_g[i];
    for (int i = 1; i < n_valid; i++) {
        float key = scratch[i]; int k = i - 1;
        while (k >= 0 && scratch[k] > key) { scratch[k+1] = scratch[k]; k--; }
        scratch[k+1] = key;
    }
    float mu_g = (n_valid & 1) ? scratch[n_valid/2]
                   : 0.5f*(scratch[n_valid/2 - 1] + scratch[n_valid/2]);

    for (int i = 0; i < n_valid; i++) scratch[i] = obs_b[i];
    for (int i = 1; i < n_valid; i++) {
        float key = scratch[i]; int k = i - 1;
        while (k >= 0 && scratch[k] > key) { scratch[k+1] = scratch[k]; k--; }
        scratch[k+1] = key;
    }
    float mu_b = (n_valid & 1) ? scratch[n_valid/2]
                   : 0.5f*(scratch[n_valid/2 - 1] + scratch[n_valid/2]);

    // Robust scale: MAD of color-distance to the median.
    for (int i = 0; i < n_valid; i++) {
        float dr = obs_r[i] - mu_r;
        float dg = obs_g[i] - mu_g;
        float db = obs_b[i] - mu_b;
        scratch[i] = sqrt(dr*dr + dg*dg + db*db);
    }
    for (int i = 1; i < n_valid; i++) {
        float key = scratch[i]; int k = i - 1;
        while (k >= 0 && scratch[k] > key) { scratch[k+1] = scratch[k]; k--; }
        scratch[k+1] = key;
    }
    float mad = (n_valid & 1) ? scratch[n_valid/2]
                  : 0.5f*(scratch[n_valid/2 - 1] + scratch[n_valid/2]);
    // sigma in color-distance units. Floor ~ per-channel 8 grey levels:
    // sqrt(3) * 8/255 ≈ 0.054.
    float sigma = fmax(1.4826f * mad, 0.054f);
    const float c_tukey = 4.685f;
    float cs = c_tukey * sigma;

    // Iterated Tukey-biweight reweighting (re-estimate location from median).
    for (int it = 0; it < irls_iters; it++) {
        float sw = 0.0f, sr = 0.0f, sg = 0.0f, sb = 0.0f;
        for (int i = 0; i < n_valid; i++) {
            float dr = obs_r[i] - mu_r;
            float dg = obs_g[i] - mu_g;
            float db = obs_b[i] - mu_b;
            float u  = sqrt(dr*dr + dg*dg + db*db) / cs;
            float b  = (u < 1.0f) ? (1.0f - u*u) * (1.0f - u*u) : 0.0f;
            float wgt = obs_w[i] * b;   // geometric prior × robust weight
            sw += wgt;
            sr += wgt * obs_r[i];
            sg += wgt * obs_g[i];
            sb += wgt * obs_b[i];
        }
        if (sw > 1e-8f) { mu_r = sr/sw; mu_g = sg/sw; mu_b = sb/sw; }
    }

    // ---- Phase 1: smooth weighted blend over ALL consensus inliers ----
    // The previous hard "top-n_final views by resolution" argmax flipped the
    // chosen source image between adjacent pixels (the resolution ranking is
    // a near-tie among overlapping views), so neighbours latched onto images
    // with different exposure/colour -> patchy ortho seams.  Instead, blend
    // every inlier continuously with
    //     w = geometric prior (cos^2) × Tukey robustness × (res/res_max)^SHARP
    // Since cos and resolution vary SMOOTHLY across the surface, the blend
    // weights — and therefore the colour — vary smoothly between adjacent
    // pixels: no hard view-assignment discontinuity.  SHARP controls the
    // detail<->smoothness trade-off: higher concentrates weight on the
    // sharpest view (more texture, less smoothing); lower blends more views.
    const float SHARP = 4.0f;
    (void)n_final;  // superseded by the continuous blend

    float res_max = 1e-6f;
    for (int i = 0; i < n_valid; i++) {
        float dr = obs_r[i] - mu_r;
        float dg = obs_g[i] - mu_g;
        float db = obs_b[i] - mu_b;
        float u  = sqrt(dr*dr + dg*dg + db*db) / cs;
        if (u < 1.0f) res_max = fmax(res_max, obs_res[i]);
    }
    float inv_res_max = 1.0f / res_max;

    float3 out_lin = (float3)(0.0f, 0.0f, 0.0f);
    float wsum = 0.0f;
    for (int i = 0; i < n_valid; i++) {
        float dr = obs_r[i] - mu_r;
        float dg = obs_g[i] - mu_g;
        float db = obs_b[i] - mu_b;
        float u  = sqrt(dr*dr + dg*dg + db*db) / cs;
        if (u >= 1.0f) continue;  // outlier — gated out by consensus
        float robust = (1.0f - u*u) * (1.0f - u*u);
        float rn     = obs_res[i] * inv_res_max;   // in (0, 1]
        float w      = obs_w[i] * robust * pow(rn, SHARP);
        out_lin += srgb_to_linear((float3)(obs_r[i], obs_g[i], obs_b[i])) * w;
        wsum    += w;
    }

    if (wsum > 1e-8f) {
        out_lin /= wsum;
    } else {
        // Degenerate: every view gated out — fall back to consensus color.
        out_lin = srgb_to_linear((float3)(mu_r, mu_g, mu_b));
    }

    float3 out_srgb = linear_to_srgb(out_lin) * 255.0f;
    out_colors[gid * 3 + 0] = (uchar)clamp(out_srgb.x, 0.0f, 255.0f);
    out_colors[gid * 3 + 1] = (uchar)clamp(out_srgb.y, 0.0f, 255.0f);
    out_colors[gid * 3 + 2] = (uchar)clamp(out_srgb.z, 0.0f, 255.0f);
}

// =====================================================================
// Visibility pruning kernels: raycast -> carve_vote -> prune
// =====================================================================

// =====================================================================
// svo_raycast: Render a depth map from the TSDF hash table.
//
// For each pixel, march along the viewing ray and detect the first
// TSDF zero-crossing (sign change from positive to negative/zero).
// Outputs:
//   rendered_depth[row * cols + col]  -- depth at first surface (0 = no hit)
//   hit_slot[row * cols + col]        -- hash slot index of hit voxel (0xFFFFFFFF = miss)
//
// Global work size: (cols_padded, rows_padded)
// =====================================================================
__kernel void svo_raycast(
    __global const VoxelSlot* table,
    uint                      capacity_mask,
    __global float*           rendered_depth,
    __global uint*            hit_slot,
    __constant SVOCamera*     cam,
    int                       rows,
    int                       cols,
    float                     voxel_size,
    float                     inv_voxel_size,
    float                     min_depth,
    float                     max_depth,
    float                     min_weight
) {
    int c = get_global_id(0);
    int r = get_global_id(1);
    if (c >= cols || r >= rows) return;

    int idx = r * cols + c;
    rendered_depth[idx] = 0.0f;
    hit_slot[idx] = 0xFFFFFFFFu;

    // Camera centre and ray direction.
    float uc = (float)c, vr = (float)r;
    float3 dir_cam;
    dir_cam.x = cam->Kinv[0]*uc + cam->Kinv[1]*vr + cam->Kinv[2];
    dir_cam.y = cam->Kinv[3]*uc + cam->Kinv[4]*vr + cam->Kinv[5];
    dir_cam.z = cam->Kinv[6]*uc + cam->Kinv[7]*vr + cam->Kinv[8];

    // Rotate to world frame.
    float3 dir_w;
    dir_w.x = cam->Rinv[0]*dir_cam.x + cam->Rinv[1]*dir_cam.y + cam->Rinv[2]*dir_cam.z;
    dir_w.y = cam->Rinv[3]*dir_cam.x + cam->Rinv[4]*dir_cam.y + cam->Rinv[5]*dir_cam.z;
    dir_w.z = cam->Rinv[6]*dir_cam.x + cam->Rinv[7]*dir_cam.y + cam->Rinv[8]*dir_cam.z;
    float dir_len = length(dir_w);
    if (dir_len < 1e-8f) return;
    dir_w /= dir_len;

    float3 cp = (float3)(cam->cam_pos[0], cam->cam_pos[1], cam->cam_pos[2]);

    // Step through voxels along the ray.
    float prev_tsdf = 2.0f;  // Invalid sentinel (outside [-1, 1]).
    uint  prev_slot_idx = 0xFFFFFFFFu;
    float prev_t = min_depth;

    int min_weight_fp = (int)(min_weight * (float)WEIGHT_SCALE);

    for (float tt = min_depth; tt <= max_depth; tt += voxel_size) {
        float3 pos = cp + tt * dir_w;
        int kx = (int)floor(pos.x * inv_voxel_size);
        int ky = (int)floor(pos.y * inv_voxel_size);
        int kz = (int)floor(pos.z * inv_voxel_size);

        if (kx < -32768 || kx > 32766 || ky < -32768 || ky > 32766)
            continue;

        uint slot_idx = hash_lookup(table, capacity_mask, kx, ky, kz);
        if (slot_idx > capacity_mask) {
            prev_tsdf = 2.0f;
            prev_t = tt;
            continue;
        }

        // Check minimum weight.
        int sw = table[slot_idx].sum_weight;
        if (sw < min_weight_fp) {
            prev_tsdf = 2.0f;
            prev_t = tt;
            continue;
        }

        float tsdf = (float)table[slot_idx].sum_tsdf / ((float)sw * (float)FP_SCALE);

        // Detect zero-crossing: prev > 0, current <= 0.
        if (prev_tsdf > 0.0f && prev_tsdf <= 1.0f && tsdf <= 0.0f) {
            // Linear interpolation to find exact crossing.
            float t_cross = prev_t + (tt - prev_t) * prev_tsdf / (prev_tsdf - tsdf + 1e-8f);
            // Convert parametric distance to camera-frame depth.
            float3 hit_pos = cp + t_cross * dir_w;
            float3 hit_cam;
            hit_cam.x = cam->R[0]*hit_pos.x + cam->R[1]*hit_pos.y + cam->R[2]*hit_pos.z + cam->t[0];
            hit_cam.y = cam->R[3]*hit_pos.x + cam->R[4]*hit_pos.y + cam->R[5]*hit_pos.z + cam->t[1];
            hit_cam.z = cam->R[6]*hit_pos.x + cam->R[7]*hit_pos.y + cam->R[8]*hit_pos.z + cam->t[2];
            rendered_depth[idx] = hit_cam.z;
            hit_slot[idx] = prev_slot_idx;
            return;
        }

        prev_tsdf = tsdf;
        prev_slot_idx = slot_idx;
        prev_t = tt;
    }
}

// =====================================================================
// svo_raycast_guided: Narrow-band raycast using a per-pixel depth hint.
//
// Instead of marching from min_depth to max_depth (potentially thousands of
// steps), we use a depth hint (from clean depth or previous iteration) and
// only search within [hint - margin, hint + margin] along the ray.
// This reduces ~2000 steps/pixel to ~12 steps for typical trunc_dist.
//
// The depth_hints image provides per-pixel camera-frame depth from either
// the original clean depth maps or the previous iteration's rendered output.
//
// Global work size: (cols_padded, rows_padded)
// =====================================================================
__kernel void svo_raycast_guided(
    __global const VoxelSlot* table,
    uint                      capacity_mask,
    __global float*           rendered_depth,
    __global uint*            hit_slot,
    __constant SVOCamera*     cam,
    __read_only image2d_array_t depth_hints,  // [n_views × H × W], CL_R FLOAT
    int                       view_idx,
    int                       rows,
    int                       cols,
    float                     voxel_size,
    float                     inv_voxel_size,
    float                     search_margin,  // world-space half-range
    float                     min_weight
) {
    int c = get_global_id(0);
    int r = get_global_id(1);
    if (c >= cols || r >= rows) return;

    int idx = r * cols + c;
    rendered_depth[idx] = 0.0f;
    hit_slot[idx] = 0xFFFFFFFFu;

    // Read depth hint for this pixel.
    const sampler_t samp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    float4 hint_val = read_imagef(depth_hints, samp, (int4)(c, r, view_idx, 0));
    float z_hint = hint_val.x;
    if (z_hint <= 0.0f) return;  // No hint → skip (no known surface here).

    // Camera centre and ray direction.
    float uc = (float)c, vr = (float)r;
    float3 dir_cam;
    dir_cam.x = cam->Kinv[0]*uc + cam->Kinv[1]*vr + cam->Kinv[2];
    dir_cam.y = cam->Kinv[3]*uc + cam->Kinv[4]*vr + cam->Kinv[5];
    dir_cam.z = cam->Kinv[6]*uc + cam->Kinv[7]*vr + cam->Kinv[8];

    // Rotate to world frame.
    float3 dir_w;
    dir_w.x = cam->Rinv[0]*dir_cam.x + cam->Rinv[1]*dir_cam.y + cam->Rinv[2]*dir_cam.z;
    dir_w.y = cam->Rinv[3]*dir_cam.x + cam->Rinv[4]*dir_cam.y + cam->Rinv[5]*dir_cam.z;
    dir_w.z = cam->Rinv[6]*dir_cam.x + cam->Rinv[7]*dir_cam.y + cam->Rinv[8]*dir_cam.z;
    float dir_len = length(dir_w);
    if (dir_len < 1e-8f) return;
    dir_w /= dir_len;

    float3 cp = (float3)(cam->cam_pos[0], cam->cam_pos[1], cam->cam_pos[2]);

    // Convert camera-frame depth hint to ray parameter.
    // dz = dot(optical_axis_world, dir_w) = cos(angle to optical axis).
    // Since dir_cam.z ~ focal_length for normalized coords, and dir_w is
    // the normalized world ray: t = z / dz where dz = R[2,:] · dir_w.
    float dz = cam->R[6]*dir_w.x + cam->R[7]*dir_w.y + cam->R[8]*dir_w.z;
    if (fabs(dz) < 1e-8f) return;
    float t_center = z_hint / dz;

    // Convert world-space margin to ray-parameter margin.
    float t_margin = search_margin / fabs(dz);
    float t_min = fmax(t_center - t_margin, 0.1f);
    float t_max = t_center + t_margin;

    // Step through the narrow band.
    float prev_tsdf = 2.0f;
    uint  prev_slot_idx = 0xFFFFFFFFu;
    float prev_t = t_min;

    int min_weight_fp = (int)(min_weight * (float)WEIGHT_SCALE);

    for (float tt = t_min; tt <= t_max; tt += voxel_size) {
        float3 pos = cp + tt * dir_w;
        int kx = (int)floor(pos.x * inv_voxel_size);
        int ky = (int)floor(pos.y * inv_voxel_size);
        int kz = (int)floor(pos.z * inv_voxel_size);

        if (kx < -32768 || kx > 32766 || ky < -32768 || ky > 32766)
            continue;

        uint slot_idx = hash_lookup(table, capacity_mask, kx, ky, kz);
        if (slot_idx > capacity_mask) {
            prev_tsdf = 2.0f;
            prev_t = tt;
            continue;
        }

        int sw = table[slot_idx].sum_weight;
        if (sw < min_weight_fp) {
            prev_tsdf = 2.0f;
            prev_t = tt;
            continue;
        }

        float tsdf = (float)table[slot_idx].sum_tsdf / ((float)sw * (float)FP_SCALE);

        // Detect zero-crossing: prev > 0, current <= 0.
        if (prev_tsdf > 0.0f && prev_tsdf <= 1.0f && tsdf <= 0.0f) {
            float t_cross = prev_t + (tt - prev_t) * prev_tsdf / (prev_tsdf - tsdf + 1e-8f);
            float3 hit_pos = cp + t_cross * dir_w;
            float3 hit_cam;
            hit_cam.x = cam->R[0]*hit_pos.x + cam->R[1]*hit_pos.y + cam->R[2]*hit_pos.z + cam->t[0];
            hit_cam.y = cam->R[3]*hit_pos.x + cam->R[4]*hit_pos.y + cam->R[5]*hit_pos.z + cam->t[1];
            hit_cam.z = cam->R[6]*hit_pos.x + cam->R[7]*hit_pos.y + cam->R[8]*hit_pos.z + cam->t[2];
            rendered_depth[idx] = hit_cam.z;
            hit_slot[idx] = prev_slot_idx;
            return;
        }

        prev_tsdf = tsdf;
        prev_slot_idx = slot_idx;
        prev_t = tt;
    }
}


// =====================================================================
// DSM extraction by Surface Nets (dual contouring) + top-down raster.
//
// Replaces the per-column raycast.  Three passes:
//   1. svo_dc_vertex   — one Surface Nets vertex per surface cube.
//   2. svo_dc_raster   — emit the dual quads, scan-convert top-down into
//                        an integer max-z buffer.
//   3. svo_dc_finalize — int z-buffer -> float DSM + per-cell normal.
//
// Because every surface cube emits a vertex (no per-ray sampling), there
// is no miss-raycast / punch-through speckle.  The triangulated surface
// interpolates across small gaps; max-z keeps the topmost surface.
// =====================================================================

// Fixed-point Z encoding for the atomic max-z buffer (1 mm precision).
#define DSM_Z_FP 1000.0f

// Decode a slot's weighted-average TSDF.  Returns 0 (and leaves outputs
// unset) only if the slot is missing or empty.  A corner just needs to be
// PRESENT to give its sign — the surface test is the cube's sign change,
// not a per-corner weight gate (off-surface corners are seen by far fewer
// rays, so gating each one individually erases most of the surface).
int dc_tsdf(__global const VoxelSlot* table, uint slot, float* out,
            int* weight) {
    if (slot == 0xFFFFFFFF) return 0;
    int sw = table[slot].sum_weight;
    if (sw < 1) return 0;
    *out = (float)table[slot].sum_tsdf / ((float)sw * (float)FP_SCALE);
    *weight = sw;
    return 1;
}

// Look up a cube's Surface Nets vertex by its min-corner voxel.
// Returns 1 and fills *out if that cube produced a vertex (z != NaN).
int dc_vertex_lookup(__global const VoxelSlot* table, uint mask,
                     __global const float* vert_pos,
                     int kx, int ky, int kz, float3* out) {
    uint slot = hash_lookup(table, mask, kx, ky, kz);
    if (slot == 0xFFFFFFFF) return 0;
    float vz = vert_pos[slot * 3 + 2];
    if (isnan(vz)) return 0;
    out->x = vert_pos[slot * 3 + 0];
    out->y = vert_pos[slot * 3 + 1];
    out->z = vz;
    return 1;
}

// ---- Pass 1: one Surface Nets vertex per surface cube ----------------
// One work-item per hash slot.  The slot's voxel (kx,ky,kz) is the
// min-corner of a dual cube; the vertex is the mean of the cube's TSDF
// edge-crossings.  vert_pos[slot].z = NaN marks "no vertex".
__kernel void svo_dc_vertex(
    __global const VoxelSlot* table,
    uint                      capacity_mask,
    uint                      capacity,
    __global float*           vert_pos,
    const float               voxel_size,
    const float               min_weight)
{
    uint i = get_global_id(0);
    if (i >= capacity) return;

    // Default: no vertex (sentinel).  Written for every slot.
    vert_pos[i * 3 + 2] = NAN;

    uint key_ab = table[i].key_ab;
    if (key_ab == EMPTY_KEY) return;

    const int min_weight_fp = (int)(min_weight * (float)WEIGHT_SCALE);
    int kx = (int)((key_ab >> 16) & 0xFFFF) - 32768;
    int ky = (int)(key_ab & 0xFFFF) - 32768;
    int kz = table[i].key_c;

    // Corner index c = bit0(x) | bit1(y) | bit2(z).
    const int cox[8] = {0, 1, 0, 1, 0, 1, 0, 1};
    const int coy[8] = {0, 0, 1, 1, 0, 0, 1, 1};
    const int coz[8] = {0, 0, 0, 0, 1, 1, 1, 1};

    float tsdf[8];
    float cx[8], cy[8], cz[8];  // corner world centres
    int present[8];             // band voxels between rays are simply absent
    int max_w = 0;              // anchor the cube on its best-supported corner
    for (int c = 0; c < 8; c++) {
        int x = kx + cox[c], y = ky + coy[c], z = kz + coz[c];
        uint s = (c == 0) ? i : hash_lookup(table, capacity_mask, x, y, z);
        int wc = 0;
        present[c] = dc_tsdf(table, s, &tsdf[c], &wc);
        if (wc > max_w) max_w = wc;
        cx[c] = ((float)x + 0.5f) * voxel_size;
        cy[c] = ((float)y + 0.5f) * voxel_size;
        cz[c] = ((float)z + 0.5f) * voxel_size;
    }
    // Reject cubes not anchored to any well-supported surface (one strong
    // corner is enough; the occluded backside is legitimately low-weight).
    if (max_w < min_weight_fp) return;

    // 12 cube edges (corner pairs differing in exactly one bit).  TOLERANT:
    // only accumulate a crossing on edges whose BOTH endpoints are present.
    // The surface-crossing edge's endpoints are the best-observed voxels, so
    // they survive even where off-surface corners are absent — this fills the
    // per-cell holes the strict all-8 rule left, mirroring how
    // svo_extract_points needs only the 2 voxels across the crossing edge.
    const int e0[12] = {0, 2, 4, 6, 0, 1, 4, 5, 0, 1, 2, 3};
    const int e1[12] = {1, 3, 5, 7, 2, 3, 6, 7, 4, 5, 6, 7};

    float sx = 0.0f, sy = 0.0f, sz = 0.0f;
    int n = 0;
    for (int e = 0; e < 12; e++) {
        int a = e0[e], b = e1[e];
        if (!present[a] || !present[b]) continue;
        float ta = tsdf[a], tb = tsdf[b];
        // Straddle zero?  (treat 0 as inside)
        if ((ta <= 0.0f) == (tb <= 0.0f)) continue;
        float t = ta / (ta - tb);
        t = clamp(t, 0.0f, 1.0f);
        sx += cx[a] + t * (cx[b] - cx[a]);
        sy += cy[a] + t * (cy[b] - cy[a]);
        sz += cz[a] + t * (cz[b] - cz[a]);
        n++;
    }
    if (n == 0) return;  // cube fully inside/outside -> no surface

    float inv = 1.0f / (float)n;
    vert_pos[i * 3 + 0] = sx * inv;
    vert_pos[i * 3 + 1] = sy * inv;
    vert_pos[i * 3 + 2] = sz * inv;
}

// ---- Triangle scan-convert into the atomic max-z buffer --------------
void dc_raster_tri(__global int* zbuf, int grid_w, int grid_h,
                   float origin_x, float origin_y, float gsd,
                   float z_min, int max_tri_cells, float min_nz,
                   float3 a, float3 b, float3 c) {
    // Wall cull: skip near-vertical triangles.  For a top-down 2.5D DSM we
    // only want horizontal-ish surface patches (roofs/ground).  A near-vertical
    // wall quad has a ~1-cell XY footprint but its top sits at roof height, so
    // max-z paints roof height into the cells just outside the true roof
    // outline -> a 1-3 px dilation / dark ortho halo.  |nz|/|n| is the cosine
    // of the surface normal from vertical; below min_nz the patch is steeper
    // than acos(min_nz) from horizontal -> treat as wall and drop it.
    {
        float ux = b.x - a.x, uy = b.y - a.y, uz = b.z - a.z;
        float vx = c.x - a.x, vy = c.y - a.y, vz = c.z - a.z;
        float nx = uy * vz - uz * vy;
        float ny = uz * vx - ux * vz;
        float nz = ux * vy - uy * vx;
        float nlen = sqrt(nx * nx + ny * ny + nz * nz);
        if (nlen > 1e-12f && fabs(nz) < min_nz * nlen) return;
    }

    // World XY -> continuous cell coordinates.
    float ax = (a.x - origin_x) / gsd, ay = (a.y - origin_y) / gsd;
    float bx = (b.x - origin_x) / gsd, by = (b.y - origin_y) / gsd;
    float cx = (c.x - origin_x) / gsd, cy = (c.y - origin_y) / gsd;

    int minx = (int)floor(fmin(ax, fmin(bx, cx)));
    int maxx = (int)ceil(fmax(ax, fmax(bx, cx)));
    int miny = (int)floor(fmin(ay, fmin(by, cy)));
    int maxy = (int)ceil(fmax(ay, fmax(by, cy)));

    // Guard: reject degenerate / oversized triangles (kills stray bridges).
    if ((maxx - minx) > max_tri_cells || (maxy - miny) > max_tri_cells) return;

    minx = max(minx, 0);
    miny = max(miny, 0);
    maxx = min(maxx, grid_w - 1);
    maxy = min(maxy, grid_h - 1);
    if (minx > maxx || miny > maxy) return;

    float denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy);
    if (fabs(denom) < 1e-12f) return;
    float inv_denom = 1.0f / denom;

    for (int py = miny; py <= maxy; py++) {
        for (int px = minx; px <= maxx; px++) {
            float fx = (float)px + 0.5f;
            float fy = (float)py + 0.5f;
            float l0 = ((by - cy) * (fx - cx) + (cx - bx) * (fy - cy)) * inv_denom;
            float l1 = ((cy - ay) * (fx - cx) + (ax - cx) * (fy - cy)) * inv_denom;
            float l2 = 1.0f - l0 - l1;
            if (l0 < -1e-4f || l1 < -1e-4f || l2 < -1e-4f) continue;
            float wz = l0 * a.z + l1 * b.z + l2 * c.z;
            int zi = (int)((wz - z_min) * DSM_Z_FP);
            // Below the valid range -> not a surface; leave the cell empty
            // (NaN) rather than fabricating a z_min "floor".
            if (zi < 0) continue;
            atomic_max(&zbuf[py * grid_w + px], zi);
        }
    }
}

// ---- Pass 2: emit dual quads and rasterize top-down ------------------
// One work-item per hash slot that produced a vertex.  For each axis,
// when the TSDF straddles zero across the edge A->A+axis, the 4 cubes
// around that grid edge each have a vertex; connect them into a quad
// (2 triangles) and scan-convert into the max-z buffer.  Each interior
// grid edge is owned by exactly one work-item (its lower endpoint A).
__kernel void svo_dc_raster(
    __global const VoxelSlot* table,
    uint                      capacity_mask,
    uint                      capacity,
    __global const float*     vert_pos,
    __global int*             zbuf,
    const float               origin_x,
    const float               origin_y,
    const float               gsd,
    const int                 grid_w,
    const int                 grid_h,
    const float               z_min,
    const int                 max_tri_cells,
    const float               min_nz)
{
    uint i = get_global_id(0);
    if (i >= capacity) return;
    if (isnan(vert_pos[i * 3 + 2])) return;  // this cube has no vertex

    uint key_ab = table[i].key_ab;
    int kx = (int)((key_ab >> 16) & 0xFFFF) - 32768;
    int ky = (int)(key_ab & 0xFFFF) - 32768;
    int kz = table[i].key_c;

    float tA;
    int wA;
    dc_tsdf(table, i, &tA, &wA);  // present: this slot owns a vertex

    for (int axis = 0; axis < 3; axis++) {
        int bx = kx, by = ky, bz = kz;
        if (axis == 0) bx++;
        else if (axis == 1) by++;
        else bz++;

        uint sB = hash_lookup(table, capacity_mask, bx, by, bz);
        float tB;
        int wB;
        if (!dc_tsdf(table, sB, &tB, &wB)) continue;
        if ((tA <= 0.0f) == (tB <= 0.0f)) continue;  // no crossing this edge

        // Two perpendicular unit axes P, Q for this edge.
        int Px, Py, Pz, Qx, Qy, Qz;
        if (axis == 0)      { Px = 0; Py = 1; Pz = 0; Qx = 0; Qy = 0; Qz = 1; }
        else if (axis == 1) { Px = 1; Py = 0; Pz = 0; Qx = 0; Qy = 0; Qz = 1; }
        else                { Px = 1; Py = 0; Pz = 0; Qx = 0; Qy = 1; Qz = 0; }

        // 4 cubes sharing the edge: A, A-P, A-Q, A-P-Q (by min-corner).
        float3 v00, v10, v01, v11;
        if (!dc_vertex_lookup(table, capacity_mask, vert_pos,
                              kx, ky, kz, &v00)) continue;
        if (!dc_vertex_lookup(table, capacity_mask, vert_pos,
                              kx - Px, ky - Py, kz - Pz, &v10)) continue;
        if (!dc_vertex_lookup(table, capacity_mask, vert_pos,
                              kx - Qx, ky - Qy, kz - Qz, &v01)) continue;
        if (!dc_vertex_lookup(table, capacity_mask, vert_pos,
                              kx - Px - Qx, ky - Py - Qy, kz - Pz - Qz,
                              &v11)) continue;

        dc_raster_tri(zbuf, grid_w, grid_h, origin_x, origin_y, gsd,
                      z_min, max_tri_cells, min_nz, v00, v10, v11);
        dc_raster_tri(zbuf, grid_w, grid_h, origin_x, origin_y, gsd,
                      z_min, max_tri_cells, min_nz, v00, v11, v01);
    }
}

// ---- Pass 3: int z-buffer -> float DSM + per-cell normal -------------
// One work-item per grid cell.  Normal from DSM central differences.
__kernel void svo_dc_finalize(
    __global const int* zbuf,
    __global float*     dsm_out,
    __global uint*      ortho_out,
    __global float*     normals_out,
    const int           grid_w,
    const int           grid_h,
    const float         z_min,
    const float         gsd)
{
    int gx = get_global_id(0);
    int gy = get_global_id(1);
    if (gx >= grid_w || gy >= grid_h) return;

    int cell = gy * grid_w + gx;
    ortho_out[cell] = 0;

    int zi = zbuf[cell];
    if (zi < 0) {
        dsm_out[cell] = NAN;
        normals_out[cell * 3 + 0] = 0.0f;
        normals_out[cell * 3 + 1] = 0.0f;
        normals_out[cell * 3 + 2] = 0.0f;
        return;
    }

    const float inv_z = 1.0f / DSM_Z_FP;
    dsm_out[cell] = z_min + (float)zi * inv_z;

    float nx = 0.0f, ny = 0.0f, nz = 1.0f;
    if (gx > 0 && gx < grid_w - 1) {
        int zl = zbuf[cell - 1], zr = zbuf[cell + 1];
        if (zl >= 0 && zr >= 0)
            nx = -((float)(zr - zl) * inv_z) / (2.0f * gsd);
    }
    if (gy > 0 && gy < grid_h - 1) {
        int zd = zbuf[cell - grid_w], zu = zbuf[cell + grid_w];
        if (zd >= 0 && zu >= 0)
            ny = -((float)(zu - zd) * inv_z) / (2.0f * gsd);
    }
    float len = sqrt(nx * nx + ny * ny + nz * nz);
    normals_out[cell * 3 + 0] = nx / len;
    normals_out[cell * 3 + 1] = ny / len;
    normals_out[cell * 3 + 2] = nz / len;
}

// =====================================================================
// Multi-scale gap fill for the DSM mesh.
//
// The fine TSDF band is not solid — voxels between rays are absent, so the
// fine Surface Nets mesh TEARS at those gaps and a missing cube cannot be
// spanned by any fine triangle (salt-and-pepper holes).  Fix: build COARSER
// TSDF levels by downsampling the fine table by 2^L.  A coarse voxel
// aggregates up to 2^(3L) fine voxels, so it is present wherever ANY of them
// is (bridging the per-voxel gaps) and its summed weight clears the anchor
// even in low-confidence regions.  The coarse table is meshed with the SAME
// svo_dc_vertex/svo_dc_raster kernels; its (larger) triangles rasterize into
// the cells the finer levels left empty.  Finest level wins (preserve detail).
// Mirrors svo_extract_fill for the point cloud.
// =====================================================================

// Accumulate one fine slot's raw (already weight-premultiplied) sums into a
// coarse bucket.  Mirrors hash_accumulate's claim protocol; the zero crossing
// is scale-invariant so the summed TSDF needs no re-normalisation.
void dc_coarse_accumulate(__global VoxelSlot* table, uint mask,
                          __global uint* overflow, int kx, int ky, int kz,
                          int count, int sw, int st, int snx, int sny, int snz) {
    uint my_ab = pack_xy(kx, ky);
    if (my_ab == EMPTY_KEY) return;
    uint h = voxel_hash(kx, ky, kz) & mask;
    for (int probe = 0; probe < MAX_PROBES; probe++) {
        uint slot = (h + (uint)probe) & mask;
        uint old = atomic_cmpxchg(&table[slot].key_ab, EMPTY_KEY, my_ab);
        if (old == EMPTY_KEY) {
            atomic_xchg(&table[slot].key_c, kz);
            mem_fence(CLK_GLOBAL_MEM_FENCE);
            atomic_add(&table[slot].count,      count);
            atomic_add(&table[slot].sum_weight, sw);
            atomic_add(&table[slot].sum_tsdf,   st);
            atomic_add(&table[slot].sum_nx,     snx);
            atomic_add(&table[slot].sum_ny,     sny);
            atomic_add(&table[slot].sum_nz,     snz);
            return;
        }
        if (old == my_ab) {
            int stored_z = KEY_C_UNINIT;
            for (int s = 0; s < 32768; s++) {
                stored_z = atomic_add(&table[slot].key_c, 0);
                if (stored_z != KEY_C_UNINIT) break;
            }
            if (stored_z == KEY_C_UNINIT) { atomic_add(overflow, 1u); continue; }
            if (stored_z == kz) {
                atomic_add(&table[slot].count,      count);
                atomic_add(&table[slot].sum_weight, sw);
                atomic_add(&table[slot].sum_tsdf,   st);
                atomic_add(&table[slot].sum_nx,     snx);
                atomic_add(&table[slot].sum_ny,     sny);
                atomic_add(&table[slot].sum_nz,     snz);
                return;
            }
        }
    }
    atomic_add(overflow, 1u);
}

// floor(a / 2^sh) for signed a.  (Right-shift of a negative signed value is
// implementation-defined in OpenCL, and voxel coords can be negative.)
int dc_floor_shift(int a, int sh) {
    return (a >= 0) ? (a >> sh) : -(((-a) + ((1 << sh) - 1)) >> sh);
}

// Downsample the fine table into a coarse table by 2^level_shift.
// One work-item per fine slot; coarse coord = floor(fine / 2^L).
__kernel void svo_dc_downsample(
    __global const VoxelSlot* fine_table,
    uint                      fine_capacity,
    __global VoxelSlot*       coarse_table,
    uint                      coarse_mask,
    __global uint*            overflow,
    const int                 level_shift)
{
    uint i = get_global_id(0);
    if (i >= fine_capacity) return;
    uint key_ab = fine_table[i].key_ab;
    if (key_ab == EMPTY_KEY) return;

    int kx = (int)((key_ab >> 16) & 0xFFFF) - 32768;
    int ky = (int)(key_ab & 0xFFFF) - 32768;
    int kz = fine_table[i].key_c;

    int cx = dc_floor_shift(kx, level_shift);
    int cy = dc_floor_shift(ky, level_shift);
    int cz = dc_floor_shift(kz, level_shift);

    // Accumulate the fine voxel's AVERAGED (decoded) TSDF as one unit-weight
    // sample.  Summing the raw weight-premultiplied sums would overflow int32
    // when many high-weight fine voxels land in one coarse cell; the decoded
    // average is bounded to ±FP_SCALE and the DC sign/crossing is
    // scale-invariant, so this reproduces the coarse surface exactly while
    // staying safely within int32.  Coarse sum_weight = (#fine voxels)·
    // WEIGHT_SCALE, so the min_weight anchor needs ≥ min_weight fine voxels.
    int sw_i = fine_table[i].sum_weight;
    if (sw_i < 1) return;
    int avg_t = fine_table[i].sum_tsdf / sw_i;
    int avg_nx = fine_table[i].sum_nx / sw_i;
    int avg_ny = fine_table[i].sum_ny / sw_i;
    int avg_nz = fine_table[i].sum_nz / sw_i;

    dc_coarse_accumulate(coarse_table, coarse_mask, overflow, cx, cy, cz, 1,
                         WEIGHT_SCALE, avg_t, avg_nx, avg_ny, avg_nz);
}

// Composite per-level z-buffers into one, finest first (preserve detail).
__kernel void svo_dc_resolve(
    __global const int* z_fine,
    __global const int* z_c1,
    __global const int* z_c2,
    __global int*       z_out,
    const uint          ncells)
{
    uint i = get_global_id(0);
    if (i >= ncells) return;
    int z = z_fine[i];
    if (z < 0) z = z_c1[i];
    if (z < 0) z = z_c2[i];
    z_out[i] = z;
}

// 5x5 median despeckle of the resolved z-buffer.  Ignores empty (-1) cells in
// the window and never fills holes (an empty cell stays empty), so it removes
// isolated speckle while preserving real step edges and no-data borders.  The
// ortho is baked in Python from THIS despeckled DSM, so DSM<->ortho stay synced.
__kernel void svo_dc_median(
    __global const int* z_in,
    __global int*       z_out,
    const int           grid_w,
    const int           grid_h)
{
    int gx = get_global_id(0);
    int gy = get_global_id(1);
    if (gx >= grid_w || gy >= grid_h) return;
    int cell = gy * grid_w + gx;

    int self = z_in[cell];
    if (self < 0) { z_out[cell] = -1; return; }  // keep no-data empty

    // Gather valid neighbours into a sorted prefix (insertion sort, ≤25).
    int vals[81];
    int n = 0;
    for (int dy = -4; dy <= 4; dy++) {
        int yy = gy + dy;
        if (yy < 0 || yy >= grid_h) continue;
        for (int dx = -4; dx <= 4; dx++) {
            int xx = gx + dx;
            if (xx < 0 || xx >= grid_w) continue;
            int v = z_in[yy * grid_w + xx];
            if (v < 0) continue;
            int j = n;
            while (j > 0 && vals[j - 1] > v) { vals[j] = vals[j - 1]; j--; }
            vals[j] = v;
            n++;
        }
    }
    z_out[cell] = vals[n / 2];  // n >= 1 (self is valid)
}
)CL";

}  // namespace dense

#endif  // OPENSFM_HAVE_OPENCL
