# Raw Depthmap Estimation — Detailed Algorithm & Data Flow

## Overview

The raw depthmap estimation implements the ACMMP algorithm (Adaptive Checkerboard
Multi-scale Multi-hypothesis PatchMatch) from Xu et al. (TPAMI 2022), ported from
CUDA to OpenCL. Each reference view produces a per-pixel depth, normal, cost, and
confidence map.

The algorithm runs on GPU via OpenCL. The host-side orchestration lives in
`DepthmapClusterEstimator` (cluster.h/cc) and `DepthmapEstimator` (estimator.h/cc).
The GPU kernels are embedded as a string literal in `opencl_kernels.h`.

---

## 1. Host-Side Orchestration

### 1.1 `DepthmapClusterEstimator::Run()`

The cluster estimator manages multiple reference views that share geometric
consistency information. Flow:

```
for each ref view:
    estimator.Prepare()         → compile kernels, save originals
    n_levels = max(1, hierarchy_levels)

for level = 0 .. n_levels-1:
    for each ref view i:
        if level > 0 and geom_weight > 0:
            Feed previous-level depths from OTHER cluster refs
            (via geom_links: source_view_idx → from_ref_idx)
        estimator.RunLevel(level, n_levels, &result)
        if level < n_levels-1:
            Save intermediate depth for next level
            ReleaseGpuBuffers()        ← frees Image2D + buffers
        else:
            ReleaseBuffers()           ← frees everything
```

**Key detail**: within a level, ref views run sequentially (not in parallel),
so only one ref view's GPU buffers are live at a time. But previous-level
results from other refs are kept in CPU memory (`prev_depths[i]`).

### 1.2 `DepthmapEstimator::Prepare()`

1. Checks that images are loaded and OpenCL is available.
2. Builds kernels once (cached in `cl_initialised_`):
   - `acmmp_random_init`
   - `acmmp_patchmatch` (compiled twice: red and black instances)
   - `acmmp_upsample`
   - `acmmp_prior_reinit`
   - `acmmp_checkerboard_filter` (compiled twice: red and black)
3. Saves original full-resolution images and intrinsics.

### 1.3 `DepthmapEstimator::RunLevel(level, N, result)`

```
scale = 1 / 2^(N-1-level)     // level 0 = coarsest

if scale < 1:
    BuildImagePyramid(scale)   → cv::resize all images, scale K matrices
else:
    use original images

UploadData()                   → all images + cameras + buffers to GPU

if level == 0:
    RandomInit()               → GPU kernel
else:
    UpsampleDepthNormal()      → GPU kernel (from prev level)

UploadPreviousDepths()         → geom consistency data from other refs

if have SfM points:
    ComputeSfMPlanarPrior()    → CPU Delaunay + upload to GPU

for iter = 0 .. max_iterations-1:
    RunIteration(iter)         → red pass + black pass
    RunCheckerboardFilter()    → red filter + black filter

if last level:
    ReadBackResults(apply_median=true)   → depth + normal + cost + confidence
else:
    ReadBackResults(apply_median=false)  → intermediate for upsampling
```

---

## 2. GPU Memory Layout (per ref view at one level)

All buffers are allocated in `UploadData()` and freed in `ReleaseGpuBuffers()`.

### 2.1 Images — `cl::Image2D` with `CL_R, CL_FLOAT`

| Slot                  | Content                                            | Size                 |
| --------------------- | -------------------------------------------------- | -------------------- |
| `cl_images_[0]`       | Reference grayscale (float32, [0,1])               | W × H × 4 bytes      |
| `cl_images_[1..N]`    | Source grayscale images                            | W × H × 4 bytes each |
| `cl_images_[N+1..16]` | Dummy 1×1 images (padding to `kMaxSources+1 = 17`) | negligible           |

Images are stored as `CL_R` + `CL_FLOAT` to enable hardware bilinear
interpolation via `CLK_FILTER_LINEAR` in the sampler. This is critical for
sub-pixel warp accuracy in the NCC computation.

**Total image memory**: `(1 + num_source_views) × W × H × 4` bytes.
With 4 sources at 3200×2400: ~5 × 30.7 MB ≈ **153 MB**.

### 2.2 Buffers — `cl::Buffer`

| Buffer                 | Element type                                  | Count    | Size at 3200×2400 |
| ---------------------- | --------------------------------------------- | -------- | ----------------- |
| `cl_cameras_`          | `CLCamera` (9+9+3 floats + 2 ints = 92 bytes) | 17       | 1.5 KB            |
| `cl_plane_hypotheses_` | `cl_float4` (16 bytes)                        | W×H      | 122.9 MB          |
| `cl_costs_`            | `float` (4 bytes)                             | W×H      | 30.7 MB           |
| `cl_rand_states_`      | `cl_uint2` (8 bytes)                          | W×H      | 61.4 MB           |
| `cl_selected_views_`   | `cl_uint` (4 bytes)                           | W×H      | 30.7 MB           |
| `cl_prior_planes_`     | `cl_float4` (16 bytes)                        | W×H      | 122.9 MB          |
| `cl_plane_masks_`      | `cl_uint` (4 bytes)                           | W×H      | 30.7 MB           |
| `cl_prev_depths_`      | `float` (4 bytes)                             | 16 × W×H | 491.5 MB          |

**Total buffer memory**: ~891 MB at 3200×2400.

### 2.3 Total GPU memory per ref view

~**1.04 GB** at 3200×2400 with 4 source views. The dominant cost is
`cl_prev_depths_` (16 × W×H floats = 491 MB), which stores geometric
consistency depths from other refs at the current resolution.

At coarser levels (e.g., level 0 = 400×300), memory drops to ~16 MB.

---

## 3. Random Initialization (`acmmp_random_init`)

**Kernel grid**: `ceil(W/16) × ceil(H/16)`, workgroup `16×16`.

Per pixel `(x, y)`:

1. **PRNG init**: Philox-2x32-10 counter-based RNG, seeded from pixel index.
2. **Random depth**: drawn in **log-space** uniform between `[depth_min, depth_max]`.
   `d = exp(log(d_min) + rand * (log(d_max) - log(d_min)))`.
3. **Random normal**: `(rand[-1,1], rand[-1,1], -1.0)`, normalized.
   The z-component is fixed at -1 to ensure the normal faces the camera.
4. **Plane construction**: `plane_from_depth_normal(x, y, d, n, cam0)` →
   `n·X_cam + d = 0` where `d = -n·P_cam` and `P_cam` is the 3D point in camera space.
5. **Initial cost + view selection**: `compute_initial_cost_and_views()`:
   - Compute matching cost against ALL source images.
   - Sort costs ascending.
   - Select top-k views (bitmask stored in `selected_views`).
   - Return mean of top-k costs.

---

## 4. Matching Cost Functions

### 4.1 Bilateral NCC (`compute_ncc`)

The core photometric cost. For a pixel `(cx, cy)` and a plane hypothesis:

1. **Depth from plane** at `(cx, cy)`, `(cx+1, cy)`, `(cx, cy+1)`.
2. **Backproject** center, +x, +y to world coordinates.
3. **Project to source** → source center `(src_cx, src_cy)`.
4. **Jacobian** (finite differences): `dfdx = proj(center+dx) - proj(center)`,
   `dfdy = proj(center+dy) - proj(center)`. This models the first-order
   homography warp from ref to source.
5. **Multi-scale NCC** with strides `[1.0, 1.5, 2.0, 2.5, 3.0]`:
   - Same number of samples at each stride (no extra compute).
   - Stride > 1 gives a wider effective patch radius.
   - Returns the **best** NCC across all scales.
   - If stride=1 (base) is textureless (NCC < -0.5), larger strides are
     capped at 0 to prevent false matches across depth boundaries.
   - **Early exit**: if NCC > 0.1 at any stride, return immediately.

6. **Per-stride NCC computation** (`compute_ncc_at_stride`):
   - Loop over `(-half_patch..half_patch)²` with strided offsets.
   - **Bilateral weighting**: `w = exp(-dist²/(2σ_s²) - (ref-center)²/(2σ_c²))`.
     Spatial sigma is in **physical pixel distance** (not loop index), so
     stride > 1 gets proper wider falloff.
   - **Center-shifted accumulation**: `r = ref_val - ref_center`,
     `s = src_val - src_center` to avoid catastrophic cancellation.
   - Variance check: `var < 1.5e-10` (equivalent to 1e-5 in [0,255]) → textureless.
   - Returns ZNCC correlation in [-1, 1].

### 4.2 Census Transform (`compute_census_cost`)

Non-parametric cost robust to illumination changes and textureless surfaces.

- **5×5 window** (25 pixels), center skipped → 24-bit descriptor.
- Each bit: `pixel > center ? 1 : 0`.
- **Hamming distance** / 24 → normalized cost in [0, 1].
- **Minimum contrast gate**: if both descriptors have < 3 set bits (uniform
  patches), return cost = 1.0 to prevent false matches.
- Uses the **same homography warp** as NCC (Jacobian-based).

### 4.3 Single-View Cost (`compute_single_cost`)

```
ncc = compute_ncc_multiscale(...)
if NCC <= -0.5:                           // NCC completely failed
    if use_census:
        cost = Census                     // Census always works
    else:
        cost = 2.0                        // invalid
else:
    cost = 1 - NCC                        // NCC cost in [0, 2]
```

When NCC fails (low-texture surfaces like asphalt, walls), Census provides
a robust fallback. Census always produces a valid cost because it compares
relative pixel orderings, not absolute values.

### 4.4 Optimized Cost Vector (`compute_cost_vector`)

The reference-side geometry (depth_from_plane, backproject at center/+dx/+dy)
is computed **once** and shared across all source images. Each source only does
3 `project()` calls + NCC/Census. This eliminates ~75% of geometry computation
per hypothesis evaluation.

**OpenCL limitation**: `image2d_t` cannot be dynamically indexed, so all 16
source images are passed as separate kernel arguments and dispatched via a
`switch` statement with a macro `COST_CASE`.

---

## 5. PatchMatch Iteration (`acmmp_patchmatch`)

**Kernel grid**: same as random init. Each invocation processes only pixels of
one checkerboard color (`color_flag`: 0=red, 1=black).

### 5.1 Step 1 — Adaptive Checkerboard Sampling (8 candidates)

For each pixel, 8 candidate **neighbor positions** are identified:
- **4 near** (up, down, left, right): immediate neighbor + V-shape diagonal search (3 levels).
- **4 far** (up, down, left, right): start at offset 3, search every 2 pixels up to 23 pixels.

Each far/near search picks the neighbor with the **lowest stored cost** along
the search path. The plane hypothesis at each winning position is evaluated
at the current pixel (full cost vector against all source images).

**Why this matters for oversmoothing**: the near search with V-shape diagonals
covers a 7-pixel radius, and the far search covers up to 23 pixels. This
enables long-range propagation of good hypotheses but also means a strongly
smooth region can propagate its plane hypothesis over large areas.

### 5.2 Step 2 — Multi-hypothesis Joint View Selection

**2a. View selection priors**: from the 4 direct neighbors' `selected_views`
bitmask. If neighbor selects view `j`, prior = 0.9; otherwise 0.1.

**2b. Sampling probabilities**: for each source view, count how many of the
8 candidates have good cost (< exponentially decaying threshold). Weight by
`exp(-cost²/0.18)` × prior.

**2c. CDF sampling**: convert probabilities to CDF, draw **15 weighted samples**
(with replacement). Each sample adds 1.0 to the view's weight. Views with > 0
weight are marked in `new_selected` bitmask.

**Key detail**: the view weights are non-uniform (proportional to the number
of CDF samples that landed on each view). This soft weighting prevents a
single bad view from dominating.

### 5.3 Step 3 — Candidate Evaluation & Acceptance

For each of the 8 candidates, compute:
```
final_cost = weighted_cost_geom(photo, view_weights, ...) + smooth_weight * smoothness
```

Where:
- `weighted_cost_geom` = per-view `(photo + geom_weight * geom)` aggregated with view weights.
- `smoothness` = bilateral-gated spatial smoothness (see §6).
- `geom` = forward-backward reprojection error against prev-level depths.

The current pixel's cost is also recomputed with the **new** view weights.
The candidate with the lowest total cost is accepted if it beats the current.

### 5.4 Step 4 — Multi-hypothesis Refinement (5 diverse candidates)

Five plane hypotheses are tested, combining current depth/normal with
random perturbations:

| #   | Depth         | Normal          | Purpose                                          |
| --- | ------------- | --------------- | ------------------------------------------------ |
| 0   | random        | current         | Test alternative depth, same surface orientation |
| 1   | current       | random          | Test alternative orientation, same depth         |
| 2   | random        | random          | Fully random jump (escape local minimum)         |
| 3   | current       | perturbed ±2% π | Fine normal adjustment                           |
| 4   | perturbed ±2% | current         | Fine depth adjustment                            |

**When SfM planar prior exists**: random depth is drawn from a Gaussian
centered on the prior depth (±3σ where σ = (d_max-d_min)/32), and random
normal is perturbed by ±5° from the prior normal. This biases refinement
toward the SfM surface but doesn't force it.

Each candidate is evaluated with the same view weights and total cost
(photo + geom + smooth). Only strictly better candidates are accepted.

---

## 6. Smoothness Cost (`compute_smoothness_cost`)

**⚠️ THIS IS A KEY SOURCE OF OVERSMOOTHING**

The smoothness cost encourages neighboring pixels to have similar depth and
normal orientation. It is added to every cost evaluation with weight
`depthmap_smooth_weight` (default **0.3**).

### Algorithm

For each of the 4-connected neighbors:

1. **Bilateral gate**: `w = exp(-(relative_depth_diff)² / (2 × 0.02²))`.
   Sigma = 2% relative depth. Neighbors at very different depths contribute
   near-zero weight (preserves discontinuities).
2. **Normal disagreement**: `1 - dot(n_cur, n_neighbor)` ∈ [0, 2].
3. **Depth surface deviation**: evaluate current plane at neighbor pixel,
   compare with neighbor's actual depth: `|nb_depth - predicted| / cur_depth × 10`.
4. **Combined**: `0.5 × clamp(normal_cost, 0, 1) + 0.5 × clamp(depth_dev×10, 0, 1)`.

### Oversmoothing analysis

- With `smooth_weight = 0.3`, the smoothness term contributes up to 0.3 to
  the total cost. Given that photometric costs for good matches are typically
  0.1–0.3, **smoothness can dominate** in textureless regions.
- The bilateral gate (σ=2%) is supposed to preserve edges, but the 4-connected
  neighborhood is evaluated on the **current** plane buffer, which is being
  updated concurrently by other work-items of the same checkerboard color
  (different color pixels). This creates a feedback loop where smooth
  hypotheses reinforce each other.
- The cost is applied during **every hypothesis evaluation** (all 8 candidates
  + 5 refinements = 13 evaluations per pixel per iteration × 4 iterations),
  meaning the smoothness pressure accumulates significantly.

---

## 7. Checkerboard Filter (`acmmp_checkerboard_filter`)

**⚠️ ANOTHER KEY SOURCE OF OVERSMOOTHING**

After each PatchMatch iteration (red+black passes), a checkerboard median
filter is applied:

### Algorithm

1. Collect depths at up to **21 neighbor positions**:
   - 6 vertical: ±1, ±3, ±5 rows (opposite-color pixels).
   - 6 horizontal: ±1, ±3, ±5 columns.
   - 8 diagonal (L-shaped): (±2, ±1), (±1, ±2).
   - 1 center pixel.
2. **Bilateral depth gate**: reject neighbors whose depth differs by more
   than **15%** (`percent = 0.15`) from center. This preserves major depth
   edges but allows moderate smoothing within surfaces.
3. **Sort** the remaining depths (insertion sort, up to 21 elements).
4. **Replace** center depth with the **median**. The normal is preserved.

### Oversmoothing analysis

- The filter runs **after every iteration** (4 times by default), with each
  application replacing depth with a local median.
- The 15% bilateral gate is quite permissive — a surface with 10% depth
  variation will be flattened.
- By preserving the original normal but replacing depth, the filter creates
  inconsistencies between the plane (depth + normal) and the photometric
  evidence. The next PatchMatch iteration then "repairs" these inconsistencies
  by adjusting to match the smoothed depth, creating a ratchet effect toward
  increasingly smooth surfaces.
- **Critical**: the filter is applied on the **plane buffer** that PatchMatch
  reads in the next iteration. Smoothed planes propagate via the neighbor
  sampling in step 1, amplifying the smoothing over iterations.
- The early-exit condition (`costs[center] < 0.001f`) only skips the filter
  for very-well-converged pixels, which are rare.

---

## 8. Geometric Consistency (`compute_geom_consistency_cost`)

Forward-backward reprojection check:

1. Depth from current plane at pixel (px, py) → backproject to world.
2. Project to source camera → (src_x, src_y).
3. Read previous-level depth at that source pixel (nearest-neighbor sampling).
4. Backproject source pixel at read depth to world.
5. Re-project to reference → (back_x, back_y).
6. Cost = 2D reprojection error `||p - back||`, capped at 5.0.

Added to the photometric cost per source view: `photo + geom_weight × geom`.
Default `geom_weight = 0.05`, so max geom contribution = 0.25 per view.

**Data layout**: `cl_prev_depths_` stores depths for up to 16 source views in
a flat buffer: `prev_depths[src_idx * W*H + y*W + x]`. A bitmask
`prev_depth_mask` indicates which slots have valid data.

---

## 9. SfM Planar Prior

### CPU-side computation (`ComputeSfMPlanarPrior`)

1. **Project SfM points** (from reconstruction) into the current reference camera.
2. **Filter**: keep only projections inside the image with valid depth.
3. **Delaunay triangulation** (`cv::Subdiv2D`) of the 2D projections.
4. **Quality gates per triangle**:
   - Depth ratio > 1.5× between shallowest/deepest vertex → reject.
   - Pixel area > 20,000 px² → reject.
5. **SVD plane fitting**: for each triangle, fit `n·X + d = 0` through the
   3 vertices in camera space.
6. **Parametric rasterization**: fill each triangle's pixels in `prior_planes`
   and `plane_masks`.
7. **Validate depth**: check `depth_from_plane` against `[depth_min, depth_max]`.
8. **Upload** `cl_prior_planes_` and `cl_plane_masks_` to GPU.

### GPU-side usage

The prior is used **only for hypothesis generation** (step 4 of PatchMatch):
when a prior exists, random depth/normal in refinement are drawn near the
prior surface instead of uniformly. Acceptance remains purely photometric.

**The `acmmp_prior_reinit` kernel** (currently disabled with a TODO comment)
would re-seed masked pixels with the prior plane each iteration. It uses the
full cost function (photo + smooth + geom) for fair comparison.

---

## 10. ReadBack and Post-Processing

### `ReadBackResults(result, W, H, apply_median)`

1. **Read GPU buffers**: `cl_plane_hypotheses_` and `cl_costs_`.
2. **Decode planes to depth+normal**: for each pixel, `depth = -d / (n·ray)`.
   Guard against NaN/Inf.
3. **Range check**: only keep depths in `[depth_min, depth_max]`.
4. **Median filter** (only at final level, `apply_median=true`):
   - `cv::medianBlur(depth, filtered, 5)` — 5×5 window.
   - Reject pixels where `|depth - median| / depth > 0.10`.

   **⚠️ ADDITIONAL SMOOTHING**: this is a separate, CPU-side 5×5 median
   filter applied on top of the GPU checkerboard filter. While it only
   *removes* outlier pixels (doesn't replace depth with median), the 10%
   threshold means any surface detail finer than 10% of its depth is treated
   as noise.

5. **Confidence**: logistic function `1/(1 + exp((cost - 0.5) × 5))`.
   Cost 0.5 → confidence 0.5; cost 0.1 → confidence ~0.88; cost 0.9 → confidence ~0.12.

---

## 11. Summary of Smoothing Stages in the Raw Pipeline

| Stage                     | When                                                           | Mechanism                                              | Severity                     |
| ------------------------- | -------------------------------------------------------------- | ------------------------------------------------------ | ---------------------------- |
| **Checkerboard filter**   | After each PatchMatch iteration (4× per level, 4 levels = 16×) | Median of 21 neighbors, 15% bilateral gate             | **HIGH**                     |
| **Smoothness cost**       | Every hypothesis evaluation (~13× per pixel per iteration)     | Bilateral depth+normal regularization, weight=0.3      | **HIGH**                     |
| **Multi-scale hierarchy** | Between levels                                                 | Nearest-neighbor upsample → re-evaluate at finer scale | MODERATE                     |
| **ReadBack median**       | Once, final level only                                         | 5×5 cv::medianBlur, 10% rejection threshold            | LOW-MODERATE                 |
| **Max cost zeroing**      | Once, post-readback                                            | Pixels with cost > 0.9 set to 0                        | LOW (removal, not smoothing) |

### Root causes of oversmoothing (ranked by impact)

1. **Checkerboard median filter applied 16 times**: each application replaces
   depth with a local median, and the modified planes feed into the next
   iteration's propagation. This compounds exponentially.

2. **Smoothness weight 0.3 competing with photometric cost 0.1–0.3**: in
   textureless or low-contrast regions, the smoothness term can outweigh
   photometric evidence, causing the algorithm to prefer spatially smooth
   hypotheses over accurate ones.

3. **Multi-scale coarse-to-fine**: coarse levels have inherently smoother
   solutions (fewer pixels, less detail). These are upsampled and used as
   initialization for finer levels. While finer levels can override coarse
   estimates, the smoothness and checkerboard filter pressure tends to
   preserve the coarse structure.

4. **Bilateral NCC with multi-scale strides**: the multi-scale NCC
   (strides 1.0–3.0×) means the effective patch radius can reach
   `3 × half_patch = 6-7 pixels` at stride 3. This trades sharpness for
   robustness on textureless surfaces.
