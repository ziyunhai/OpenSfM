# Dense Reconstruction Pipeline — High-Level Overview

## Entry Point

`compute_depthmaps()` in `opensfm/dense.py` is the single entry point.
It receives an `UndistortedDataSet`, a `TracksManager`, and a `Reconstruction`.

The pipeline has **four sequential phases**:

```
Phase 1: Raw depthmaps (GPU, per cluster)
Phase 2: Cleaning       (GPU, N-hop neighbors)
Phase 3: Fusion          (CPU/GPU, per cluster)
Phase 4: Merge           (CPU, concatenation)
```

---

## Phase 0 — Setup & Clustering

### 0a. Build clusters (Leiden partitioning)

**Goal:** partition all processable shots into spatially coherent groups of
≤ `depthmap_cluster_max_size` (default 8) views.

1. `pysfm.build_covisibility_graph()` (C++, multithreaded): fuses tracks into
   super-points via kNN, computes pairwise covisibility weights.
2. Python-side Leiden/CPM partitioning (`leidenalg`) on the covisibility graph.
3. Tiny clusters (< 75 % of max) are absorbed into the best-weighted neighbor.

**Output:** `clusters: List[List[str]]` — each sub-list is one cluster of shot IDs.

### 0b. Bounding boxes

Per-cluster axis-aligned bounding boxes are computed from the 3-D tracks visible
in each cluster (or camera positions if no points exist).

### 0c. Neighbor selection (C++)

`pysfm.select_neighbors()` returns two neighbor maps per shot:
- `best_neighbors`: ranked by covisibility quality → used in Phase 1 (raw).
- `all_neighbors`: N-hop expansion → used in Phase 2 (clean) and Phase 3 (fusion).

### 0d. Depth ranges

Per-shot `(min_depth, max_depth)` computed from the tracks manager using
`compute_depth_range()`.

### 0e. Device discovery

OpenCL GPU devices are enumerated. Intel iGPUs are optionally ignored.
A slot budget of `_MAX_PARALLEL_PER_DEVICE * num_GPUs` limits concurrency.

---

## Phase 1 — Raw Depthmaps (GPU, per cluster)

### Dispatch

A shared `queue.Queue` holds `(cluster_index, shot_list)` work items.
Per-device worker threads pull from the queue until empty.
Up to `_MAX_PARALLEL_PER_DEVICE` (4) threads per GPU run concurrently.

### Per-cluster work (`_run_cluster_raw`)

For each cluster:

1. **Create a `DepthmapClusterEstimator`** (C++ object, one OpenCL context).
2. **For each ref view** in the cluster:
   - Load grayscale image from disk, resize to `depthmap_max_image_size`.
   - Call `cluster.begin_ref_view(K, R, t, image, min_d, max_d)`.
   - Optionally set SfM planar prior points.
   - For each source neighbor (up to `depthmap_num_matching_views`, default 4):
     load grayscale, resize, `cluster.add_source_view(K, R, t, image)`.
3. **Register geometric consistency links** between ref views in the cluster
   (via `cluster.add_geom_link(ref_idx, src_pos, other_ref_idx)`).
4. **`cluster.run()`** → runs multi-scale PatchMatch on GPU for all ref views.
5. **Post-process results** on CPU:
   - Sanitize NaN/Inf → zero out.
   - Zero out pixels with cost > `depthmap_max_cost`.
6. **Save raw depthmaps** to disk (parallel I/O).

### Data on GPU (per ref view)

| Buffer                 | Size                   | Description                                |
| ---------------------- | ---------------------- | ------------------------------------------ |
| `cl_images_[0]`        | W×H float Image2D      | Reference image (float32 [0,1])            |
| `cl_images_[1..N]`     | W×H float Image2D each | Source images (up to 16)                   |
| `cl_cameras_`          | 17 × `CLCamera`        | Camera intrinsics + extrinsics             |
| `cl_plane_hypotheses_` | W×H × float4           | Per-pixel plane (nx,ny,nz,d)               |
| `cl_costs_`            | W×H × float            | Per-pixel aggregated cost                  |
| `cl_rand_states_`      | W×H × uint2            | Philox PRNG state                          |
| `cl_selected_views_`   | W×H × uint             | Per-pixel view bitmask                     |
| `cl_prior_planes_`     | W×H × float4           | SfM Delaunay planar prior                  |
| `cl_plane_masks_`      | W×H × uint             | Prior validity mask                        |
| `cl_prev_depths_`      | 16 × W×H × float       | Previous-level depths for geom consistency |

**Total GPU memory per ref view** ≈ `(1+N) * W*H*4  +  W*H * (4*4 + 4 + 8 + 4 + 4*4 + 4 + 16*W*H*4)` bytes. For a 3200×2400 image with 4 source views, this is roughly **600–800 MB**.

### Multi-scale hierarchy

The estimator runs `depthmap_hierarchy_levels` (default 4) coarse-to-fine levels:
- Level 0: 1/8 resolution
- Level 1: 1/4 resolution
- Level 2: 1/2 resolution
- Level 3: full resolution

At each level:
1. Images are downscaled from originals.
2. All images + cameras re-uploaded to GPU.
3. Depth from previous level upsampled (GPU kernel `acmmp_upsample`).
4. Previous-level depths from other cluster refs uploaded for geometric consistency.
5. `max_iterations` (default 4) PatchMatch iterations, each followed by checkerboard filter.
6. Results read back; intermediate levels stored for next level's upsampling.

**Between levels, GPU buffers are released** (`ReleaseGpuBuffers`) to save memory.

### ReadBack post-processing (CPU)

After the final level:
1. Plane hypotheses decoded back to depth + normal.
2. **Median filter** (5×5 `cv::medianBlur`): pixels deviating > 10 % from median
   are zeroed out.
3. **Confidence map**: logistic `sigmoid(-(cost - 0.5) * 5)`.

---

## Phase 2 — Cleaning (GPU, N-hop neighbors)

### Dispatch

`_clean_views_batched_gpu()` distributes clusters across GPUs via a shared work
queue. **One thread per GPU device** (the cleaner's OpenCL context cannot be
shared).

### Per-cluster work (`_gpu_clean_and_save_cluster`)

1. **Select neighbors**: for each ref shot in the cluster, pick up to 16
   (`kMaxCleanSources`) neighbors from `all_neighbors` (N-hop expanded set).
   Only shots with existing raw depthmaps qualify.
2. **Load all needed views**: the union of cluster refs + their neighbors.
   Each view's raw depth is loaded from disk (parallel I/O, `load_raw_depthmaps_parallel`).
3. **Feed the GPU cleaner**: `cleaner.add_view(K, R, t, depth)` for each view.
4. **For each ref shot**: call `cleaner.clean(ref_idx, neighbor_indices)`.
5. **Save cleaned depthmaps** to disk (parallel I/O).
6. `cleaner.clear()` releases GPU memory.

### Data on GPU (per clean call)

| Buffer                      | Size                   | Description                    |
| --------------------------- | ---------------------- | ------------------------------ |
| `cl_ref_depth_img_`         | W×H float Image2D      | Reference raw depth            |
| `cl_src_depth_imgs_[0..15]` | W×H float Image2D each | Neighbor raw depths (up to 16) |
| `cl_cameras_`               | (1+N) × `CLCamera`     | Camera params                  |
| `cl_clean_depth_`           | W×H × float            | Output cleaned depth           |

**Total GPU memory**: ~ `18 × W×H × 4` bytes depth images + cameras. For 3200×2400,
this is roughly **880 MB** for 16 neighbors + ref.

### Cleaning kernel algorithm (`acmmp_clean_depthmap`)

For each pixel in the reference depth:
1. Backproject to 3D world point.
2. Project into each neighbor view.
3. Read neighbor's depth (bilinear interpolation from Image2D texture).
4. **Space carving**: if neighbor depth > ref_projected_depth × (1 + `carving_threshold`),
   count as a carve vote.
5. **Forward depth check**: `|src_depth - projected_depth| < projected_depth × same_depth_threshold`.
6. **Backward geometric consistency**: backproject neighbor pixel at neighbor
   depth to world, re-project to reference. Check both pixel distance (< 2 px)
   and depth agreement.
7. **Decision**: keep pixel if `consistent >= min_consistent_views` AND
   `carved < max_carved_views`.

Default thresholds: `same_depth_threshold=0.01` (1%), `min_consistent_views=3`,
`carving_threshold=0.2`, `max_carved_views=1`.

---

## Phase 3 — Fusion (CPU/GPU, per cluster)

Not analyzed in this document (SVO + TSDF integration).

## Phase 4 — Merge

All per-cluster `fused_batch_XXXX.ply` files are concatenated into `fused.ply`.

---

## Key Config Parameters (defaults in `config.py`)

| Parameter                          | Default | Phase | Effect                                       |
| ---------------------------------- | ------- | ----- | -------------------------------------------- |
| `depthmap_hierarchy_levels`        | 4       | Raw   | Number of coarse-to-fine levels              |
| `depthmap_max_iterations`          | 4       | Raw   | PatchMatch iterations per level              |
| `depthmap_patch_size`              | 5       | Raw   | Half-patch = 2 → 5×5 bilateral NCC window    |
| `depthmap_num_matching_views`      | 4       | Raw   | Source images per ref view                   |
| `depthmap_top_k`                   | 4       | Raw   | Top-k view selection for cost aggregation    |
| `depthmap_sigma_spatial`           | 5.0     | Raw   | Bilateral spatial sigma (pixels)             |
| `depthmap_sigma_color`             | 3/255   | Raw   | Bilateral color sigma (normalized intensity) |
| `depthmap_census_weight`           | 0.3     | Raw   | NCC/Census blend weight                      |
| `depthmap_smooth_weight`           | 0.3     | Raw   | Smoothness regularization weight             |
| `depthmap_geom_consistency_weight` | 0.05    | Raw   | Geometric consistency weight                 |
| `depthmap_sfm_planar_prior`        | True    | Raw   | Enable SfM Delaunay planar prior             |
| `depthmap_max_cost`                | 0.9     | Raw   | Post-process: zero out high-cost pixels      |
| `depthmap_same_depth_threshold`    | 0.01    | Clean | 1% relative depth tolerance                  |
| `depthmap_min_consistent_views`    | 3       | Clean | Minimum cross-view agreement                 |
| `depthmap_carving_threshold`       | 0.2     | Clean | Free-space carving threshold                 |
| `depthmap_max_carved_views`        | 1       | Clean | Max carve votes before discard               |
| `depthmap_cluster_max_size`        | 8       | Setup | Max views per geometric cluster              |
| `depthmap_max_image_size`          | 3200    | Raw   | Max image dimension                          |
