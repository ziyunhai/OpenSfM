# Clean Depthmap — Detailed Algorithm & Data Flow

## Overview

The cleaning phase takes the raw depthmaps produced by PatchMatch (Phase 1) and
removes geometrically inconsistent pixels by cross-checking depth agreement
across multiple views. This is a **binary keep/discard** filter — it does not
modify depth values, only zeroes out unreliable pixels.

The cleaning runs on GPU via a single OpenCL kernel (`acmmp_clean_depthmap`).
Host-side orchestration lives in `GPUDepthmapCleaner` (cleaner.h/cc) and the
Python-side batching in `_clean_views_batched_gpu()` / `_gpu_clean_and_save_cluster()`
in `dense.py`.

---

## 1. Host-Side Orchestration (Python)

### 1.1 Dispatch (`_clean_views_batched_gpu`)

Cleaning is organized **per cluster** (same clusters as Phase 1) and
distributed across GPUs via a shared work queue.

**Threading model**: **one thread per GPU device** (not per-device ×
`_MAX_PARALLEL_PER_DEVICE` like Phase 1). This is because the
`GPUDepthmapCleaner` holds an OpenCL context that cannot be shared across
threads on the same device. Each thread reuses its `GPUDepthmapCleaner` across
clusters, calling `clear()` between batches (kernel stays compiled; only
view data is released).

```
for each GPU device:
    create one GPUDepthmapCleaner
    configure thresholds
    loop:
        pull next cluster from shared queue
        _gpu_clean_and_save_cluster(cleaner, cluster_shots)
```

### 1.2 Per-Cluster Work (`_gpu_clean_and_save_cluster`)

For each cluster of reference shots:

1. **Select neighbors** (Python-side): for each ref shot, take up to
   `max_neighbors = 16` (`kMaxCleanSources`) shots from `all_neighbors`
   (N-hop expansion from the covisibility graph). Only shots with existing
   raw depthmaps qualify. The `all_neighbors` list is pre-ranked by overlap
   quality.

2. **Compute union of needed shots**: cluster refs + all their neighbors.
   Deduplicated and sorted for deterministic cleaner indices.

3. **Load raw depthmaps** from disk (parallel I/O via
   `data.load_raw_depthmaps_parallel(ordered_shots)`). Each raw depthmap
   returns `(depth, normal, score, neighbor_ids, neighbors, confidence)`.
   Only `depth` is sent to the GPU cleaner; `normal` and `confidence` are
   kept CPU-side for the ref shots (reattached to the clean output).

4. **Feed the GPU cleaner**: `cleaner.add_view(K, R, t, depth)` for each
   shot in the ordered list. This stores the depth as `cv::Mat CV_32F` and
   camera params on the host. **No GPU upload yet** — data is uploaded
   per `clean()` call.

5. **For each ref shot**: call `cleaner.clean(ref_idx, neighbor_indices)`.
   The indices refer to positions in the ordered list from step 2.

6. **Assemble output**: the cleaned depth (from GPU) is combined with the
   original raw normal and confidence (from step 3). A zero-filled score
   array is used (the raw cost is discarded). Each shot is saved as a clean
   depthmap via `data.save_clean_depthmaps_parallel(save_items)`.

7. **Release**: `cleaner.clear()` frees all GPU resources + host view data.
   `gc.collect()` reclaims Python-side memory.

### Data flow diagram

```
Disk (raw depthmaps)
    │
    ├── load_raw_depthmaps_parallel(ordered_shots)
    │       │
    │       ├── depth[W×H float32] ──────────► cleaner.add_view() → host views_[]
    │       ├── normal[3×W×H float32] ──────► kept in raw_normals{} (CPU)
    │       └── confidence[W×H float32] ────► kept in raw_confidence{} (CPU)
    │
    └── For each ref shot:
            cleaner.clean(ref_idx, nbr_indices)
                │
                ├── Upload ref depth → Image2D (GPU)
                ├── Upload nbr depths → 16× Image2D (GPU)
                ├── Upload cameras → Buffer (GPU)
                ├── Run kernel
                └── Read back clean_depth → Buffer → cv::Mat (CPU)
                        │
                        ├── Combine with raw_normals[ref_id]
                        ├── Combine with raw_confidence[ref_id]
                        └── save_clean_depthmaps_parallel()
```

---

## 2. Host-Side C++ (`GPUDepthmapCleaner`)

### 2.1 `AddView(K, R, t, depth)`

Stores a `ViewEntry` on the host:
```cpp
struct ViewEntry {
    cv::Mat depth;   // CV_32F, cloned from Eigen input
    Mat3d K, R;
    Vec3d t;
    int width, height;
};
```

Views accumulate in `views_` vector. Index returned = position in vector.

### 2.2 `Clean(ref_idx, neighbor_ids)`

This is where all GPU work happens:

1. **Build kernel** (once, cached in `kernel_built_`):
   ```cpp
   program_ = dev.BuildProgram(kCleanKernelSource);
   k_clean_ = cl::Kernel(program_, "acmmp_clean_depthmap");
   ```

2. **Ensure Image2D pool** matches current dimensions:
   - If `(W, H)` changed since last call, reallocate:
     - 1 × `cl_ref_depth_img_` (Image2D, CL_R, CL_FLOAT)
     - 16 × `cl_src_depth_imgs_[]` (Image2D, CL_R, CL_FLOAT)
   - Pool is **reused** if dimensions match (common case within a cluster).

3. **Upload depth data into images**:
   - Reference: `enqueueWriteImage(cl_ref_depth_img_, ..., ref.depth.data)`.
   - Neighbors: for each of `num_neighbors` (up to 16):
     - If neighbor dimensions differ from reference, **resize** via
       `cv::resize(other.depth, ..., cv::INTER_LINEAR)`.
     - `enqueueWriteImage(cl_src_depth_imgs_[vi], ..., depth.data)`.

   **Important detail**: neighbor depths that have a different resolution
   are resized to match the reference. This bilinear resize introduces
   a small amount of smoothing on the depth values used for consistency
   checks.

4. **Upload cameras**:
   - Pack `num_views` × `CLCamera` (ref + neighbors).
   - Reuse `cl_cameras_` buffer if large enough; otherwise reallocate.

5. **Allocate output buffer** `cl_clean_depth_` (W×H floats). Reused if
   large enough.

6. **Set kernel arguments and dispatch**:
   ```
   NDRange global = ceil(W/16)*16, ceil(H/16)*16
   NDRange local  = 16, 16
   ```

7. **Read back** cleaned depth into a `cv::Mat(H, W, CV_32F)`.

### 2.3 `Clear()`

Releases all GPU resources:
- `cl_cameras_`, `cl_clean_depth_` → empty `cl::Buffer`.
- `cl_ref_depth_img_`, `cl_src_depth_imgs_[]` → empty `cl::Image2D`.
- `views_` vector cleared.

---

## 3. GPU Memory Layout

### 3.1 Image2D objects (depth maps as textures)

| Image                    | Format         | Size            | Description           |
| ------------------------ | -------------- | --------------- | --------------------- |
| `cl_ref_depth_img_`      | CL_R, CL_FLOAT | W × H × 4 bytes | Reference raw depth   |
| `cl_src_depth_imgs_[0]`  | CL_R, CL_FLOAT | W × H × 4 bytes | Neighbor 0 raw depth  |
| `cl_src_depth_imgs_[1]`  | CL_R, CL_FLOAT | W × H × 4 bytes | Neighbor 1 raw depth  |
| ...                      | ...            | ...             | ...                   |
| `cl_src_depth_imgs_[15]` | CL_R, CL_FLOAT | W × H × 4 bytes | Neighbor 15 raw depth |

**Total depth images**: `(1 + 16) × W × H × 4` bytes.
At 3200×2400: 17 × 30.7 MB = **522 MB**.

These are `CL_R, CL_FLOAT` images to enable hardware bilinear interpolation
when reading neighbor depths at sub-pixel reprojected coordinates. The
reference depth uses nearest-neighbor sampling (`CLK_FILTER_NEAREST`).

### 3.2 Buffers

| Buffer            | Size                                    | Description                  |
| ----------------- | --------------------------------------- | ---------------------------- |
| `cl_cameras_`     | `num_views × sizeof(CLCamera)` ≈ 1.6 KB | Camera intrinsics/extrinsics |
| `cl_clean_depth_` | W × H × 4 bytes                         | Output cleaned depth         |

### 3.3 Total GPU memory

~**553 MB** at 3200×2400 with 16 neighbors. Dominated by the 17 depth Image2D
objects.

**Important**: unlike Phase 1, no grayscale images are loaded to GPU — only
depth maps. This is because cleaning is purely geometric (no photometric
cost computation).

### 3.4 CPU memory during cleaning

For each cluster batch, the Python side holds:
- **All loaded raw depthmaps** (union of refs + neighbors): `N × W × H × 4` bytes
  for depth, plus same for normal (float32 × 3 channels) and confidence.
- **Color images are NOT loaded** during cleaning (only during fusion).

For a typical cluster of 8 refs with 16 neighbors each (union ~30 unique
views), CPU memory ≈ 30 × 30.7 MB × 3 (depth+normal+conf) ≈ **2.8 GB**.

---

## 4. GPU Kernel: `acmmp_clean_depthmap`

### 4.1 Kernel signature

```opencl
__kernel void acmmp_clean_depthmap(
    read_only image2d_t ref_depth_img,
    read_only image2d_t src0, ..., src15,    // 16 neighbor depth images
    __global const Camera *cameras,
    __global float *clean_depth,             // output
    int width, int height,
    int num_views,
    float same_depth_threshold,              // default 0.01
    int min_consistent_views,                // default 3
    float carving_threshold,                 // default 0.2
    int max_carved_views                     // default 1
)
```

### 4.2 Algorithm (per pixel)

```
1. Read reference depth at (x, y) via nearest-neighbor sampler.
   If depth <= 0: output 0, return.

2. Backproject (x, y, ref_depth) to world point (wx, wy, wz):
   cam_xyz = (ref_depth * (x-cx)/fx, ref_depth * (y-cy)/fy, ref_depth)
   world = R^T × (cam_xyz - t)

3. Initialize: consistent = 1 (ref itself counts), carved = 0.

4. For each neighbor view v = 0..min(num_views-1, 16)-1:

   a. Project world point to neighbor camera:
      src_cam = R_v × world + t_v
      (su, sv) = (K_v × src_cam) / src_cam.z

   b. Bounds check: su, sv must be in [0, W-1) × [0, H-1).

   c. Read neighbor depth at (su, sv) via BILINEAR sampler.
      If src_depth <= 0: skip this neighbor.

   d. SPACE CARVING check:
      if src_depth > projected_depth × (1 + carving_threshold):
          carved++
          continue
      Meaning: the neighbor sees something FURTHER AWAY than where our
      point projects. The reference point is in free space from this
      neighbor's perspective → likely a floater/ghost.

   e. FORWARD DEPTH CHECK:
      if |src_depth - projected_depth| >= projected_depth × same_depth_threshold:
          skip (depths don't agree)

   f. BACKWARD GEOMETRIC CONSISTENCY:
      i.   Backproject neighbor pixel (su, sv, src_depth) to world:
           src_cam_xyz = (src_depth * (su-cx_v)/fx_v, ...)
           world2 = R_v^T × (src_cam_xyz - t_v)
      ii.  Re-project world2 to reference camera:
           ref_cam2 = R_ref × world2 + t_ref
           (ru, rv) = (K_ref × ref_cam2) / ref_cam2.z
      iii. Check pixel reprojection error:
           if |ru - x| + |rv - y| > 2.0: skip
      iv.  Check depth reprojection error:
           if |ref_cam2.z - ref_depth| >= ref_depth × same_depth_threshold: skip

   g. If all checks pass: consistent++

5. DECISION:
   keep = (consistent >= min_consistent_views) AND (carved < max_carved_views)
   clean_depth[idx] = keep ? ref_depth : 0.0
```

### 4.3 Detailed check analysis

#### Forward depth check (step 4e)

```
|src_depth - sz| < sz × same_depth_threshold
```

Where `sz` is the Z-coordinate of the world point projected into the
neighbor's camera space (i.e., the expected depth at the neighbor pixel if
the reference depth is correct).

With `same_depth_threshold = 0.01` (1%), a point at depth 10m must see
agreement within ±10cm. This is fairly tight and may reject valid pixels
at depth edges where slight calibration errors cause > 1% disagreement.

#### Backward consistency check (steps 4f.iii-iv)

This performs a full **forward-backward** roundtrip:
```
ref pixel → world → neighbor pixel → world2 → ref pixel2
```

Two conditions must hold:
- **Pixel error**: Manhattan distance `|ru-x| + |rv-y| ≤ 2.0` pixels.
- **Depth error**: `|rz - ref_depth| < ref_depth × same_depth_threshold`.

The backward check is **stricter** than just forward because it also catches
cases where the neighbor's depth map has a different surface at the projected
location (e.g., an occluding object).

#### Space carving (step 4d)

A free-space consistency check. If the neighbor sees further than expected
by more than `carving_threshold = 20%`, the reference point is in empty space
from this neighbor's view → the point is a "floater" artifact.

With `max_carved_views = 1`, a single carving vote from any neighbor discards
the pixel. This is aggressive and can remove valid thin structures that
happen to be occluded in one neighbor view.

---

## 5. Oversmoothing Impact of Cleaning

The cleaning phase **does not smooth depth values** — it only removes pixels
that fail cross-view consistency. However, it contributes to the appearance
of oversmoothing through several mechanisms:

### 5.1 Preferential removal of fine detail

Fine geometric detail (edges, corners, thin structures) tends to:
- Have fewer consistent views (less overlap at depth discontinuities).
- Fail the tight 1% depth tolerance more often (slight calibration errors
  are amplified at boundaries).
- Be carved by neighbors that see the background behind the detail.

The result is that cleaning preferentially removes high-frequency depth
variation, leaving behind the smooth, well-agreed "consensus" surface.

### 5.2 Bilinear depth sampling

Neighbor depths are read via `CLK_FILTER_LINEAR` (bilinear interpolation).
At depth discontinuities, the bilinear tap blends foreground and background
depths, producing an intermediate value that agrees with neither surface.
This causes both sides of the discontinuity to fail the depth check, creating
a gap along every depth edge.

### 5.3 Thresholds

| Parameter              | Default | Effect                                                         |
| ---------------------- | ------- | -------------------------------------------------------------- |
| `same_depth_threshold` | 0.01    | 1% relative tolerance — tight                                  |
| `min_consistent_views` | 3       | Need 3 views agreeing (including ref) — 2 neighbors must agree |
| `carving_threshold`    | 0.2     | 20% free-space margin — moderate                               |
| `max_carved_views`     | 1       | Single carve vote kills pixel — aggressive                     |

The `min_consistent_views` is clamped to `min(config_value, num_views)`.
With only 3–4 visible neighbors (common at dataset edges), this means ALL
neighbors must agree, which is very strict.

---

## 6. Differences from Phase 1 (Raw) GPU Usage

| Aspect             | Phase 1 (Raw)                      | Phase 2 (Clean)                  |
| ------------------ | ---------------------------------- | -------------------------------- |
| **Data on GPU**    | Images (grayscale) + depth buffers | Depth maps only                  |
| **Image format**   | CL_R, CL_FLOAT (grayscale)         | CL_R, CL_FLOAT (depth)           |
| **GPU objects**    | 17 Image2D + ~8 Buffers            | 17 Image2D + 2 Buffers           |
| **GPU memory**     | ~1 GB at 3200×2400                 | ~550 MB at 3200×2400             |
| **Threads/GPU**    | Up to 4 concurrent                 | 1 thread per GPU                 |
| **Work per pixel** | ~13 cost evaluations × NCC+Census  | 16 project+check operations      |
| **Output**         | depth, normal, cost, confidence    | depth only (binary keep/discard) |
| **Modifies depth** | Yes (finds best hypothesis)        | No (only zeros out bad pixels)   |

---

## 7. Summary

The cleaning phase is a **conservative geometric filter** that removes
pixels lacking cross-view depth agreement. Its key characteristics:

1. **Bilateral bilinear depth reads** at sub-pixel reprojections.
2. **Forward + backward consistency** with a tight 1% depth tolerance.
3. **Space carving** with aggressive single-vote discard.
4. **No depth modification** — purely keep/discard.

While cleaning doesn't directly smooth depths, its tight thresholds and
single-vote carving preferentially remove fine detail and depth edges,
contributing to the overall impression of oversmoothed results. The raw
PatchMatch pipeline (Phase 1) is where the actual depth value smoothing
occurs, through the checkerboard median filter and the smoothness
regularization term.
