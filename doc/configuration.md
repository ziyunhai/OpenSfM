# Configuration Reference

Pipeline options are set in `config.yaml` at the root of each dataset folder. Any key present overrides the default value defined in the `OpenSfMConfig` dataclass in `opensfm/config.py`.

Example `config.yaml`:
```yaml
feature_type: SIFT
feature_process_size: 4096
matching_gps_distance: 200
processes: 4
```

---

## Metadata

| Parameter                             | Type  | Default         | Description                                                  |
| ------------------------------------- | ----- | --------------- | ------------------------------------------------------------ |
| `use_exif_size`                       | bool  | `true`          | Use image size from EXIF                                     |
| `unknown_camera_models_are_different` | bool  | `false`         | Treat images from unknown camera models as different cameras |
| `default_projection_type`             | str   | `"perspective"` | Projection type when EXIF cannot determine it                |
| `default_focal_prior`                 | float | `0.85`          | Default focal-length-to-sensor-width ratio                   |

## Features

| Parameter                          | Type        | Default   | Description                                                               |
| ---------------------------------- | ----------- | --------- | ------------------------------------------------------------------------- |
| `feature_type`                     | str         | `"HAHOG"` | Feature detector: `HAHOG`, `SIFT`, `SURF`, `AKAZE`, `ORB`                 |
| `feature_root`                     | bool        | `true`    | Apply square-root mapping to descriptors                                  |
| `feature_min_frames`               | int         | `4000`    | Target minimum features; detector threshold is relaxed if fewer are found |
| `feature_min_frames_panorama`      | int         | `16000`   | Same as above for panoramas                                               |
| `feature_process_size`             | int         | `2048`    | Resize image (longest side) before extraction. `-1` = original            |
| `feature_process_size_panorama`    | int         | `4096`    | Same as above for panoramas                                               |
| `feature_use_adaptive_suppression` | bool        | `false`   | Use adaptive non-maximal suppression                                      |
| `features_bake_segmentation`       | bool        | `false`   | Bake segmentation class/instance into feature data at extraction time     |
| `mem_ceiling`                      | int\|null   | `null`    | Max memory (MB) for feature extraction                                    |
| `mem_ratio`                        | float\|null | `null`    | Fraction of `mem_ceiling` to use                                          |

### SIFT

| Parameter             | Type  | Default | Description                  |
| --------------------- | ----- | ------- | ---------------------------- |
| `sift_peak_threshold` | float | `0.1`   | Smaller → more features      |
| `sift_edge_threshold` | int   | `10`    | Edge rejection threshold     |
| `sift_nfeatures`      | int   | `0`     | Max features (0 = unlimited) |
| `sift_octave_layers`  | int   | `3`     | Layers per octave            |
| `sift_sigma`          | float | `1.6`   | Initial Gaussian sigma       |

### SURF

| Parameter                | Type  | Default | Description                 |
| ------------------------ | ----- | ------- | --------------------------- |
| `surf_hessian_threshold` | float | `3000`  | Smaller → more features     |
| `surf_n_octaves`         | int   | `4`     | Number of octaves           |
| `surf_n_octavelayers`    | int   | `2`     | Layers per octave           |
| `surf_upright`           | int   | `0`     | Compute upright descriptors |

### AKAZE

| Parameter                       | Type  | Default   | Description                        |
| ------------------------------- | ----- | --------- | ---------------------------------- |
| `akaze_omax`                    | int   | `4`       | Maximum octave evolution           |
| `akaze_dthreshold`              | float | `0.001`   | Detector response threshold        |
| `akaze_descriptor`              | str   | `"MSURF"` | Descriptor type                    |
| `akaze_descriptor_size`         | int   | `0`       | Descriptor size in bits (0 = full) |
| `akaze_descriptor_channels`     | int   | `3`       | Feature channels (1–3)             |
| `akaze_kcontrast_percentile`    | float | `0.7`     | Contrast percentile                |
| `akaze_use_isotropic_diffusion` | bool  | `false`   | Use isotropic diffusion            |

### HAHOG

| Parameter                  | Type  | Default   | Description                            |
| -------------------------- | ----- | --------- | -------------------------------------- |
| `hahog_peak_threshold`     | float | `0.00001` | Peak threshold                         |
| `hahog_edge_threshold`     | float | `10`      | Edge threshold                         |
| `hahog_normalize_to_uchar` | bool  | `true`    | Normalize descriptors to unsigned char |

## Matching

| Parameter            | Type  | Default   | Description                       |
| -------------------- | ----- | --------- | --------------------------------- |
| `lowes_ratio`        | float | `0.8`     | Lowe's ratio test threshold       |
| `matcher_type`       | str   | `"FLANN"` | `FLANN`, `BRUTEFORCE`, or `WORDS` |
| `symmetric_matching` | bool  | `true`    | Match in both directions          |

### FLANN

| Parameter          | Type | Default    | Description                            |
| ------------------ | ---- | ---------- | -------------------------------------- |
| `flann_algorithm`  | str  | `"KMEANS"` | `KMEANS` or `KDTREE`                   |
| `flann_branching`  | int  | `8`        | Branching factor                       |
| `flann_iterations` | int  | `10`       | K-means iterations                     |
| `flann_tree`       | int  | `8`        | Number of trees (KDTREE)               |
| `flann_checks`     | int  | `20`       | Checks during search. Smaller → faster |

### Bag-of-Words (BoW)

| Parameter            | Type | Default                            | Description                  |
| -------------------- | ---- | ---------------------------------- | ---------------------------- |
| `bow_file`           | str  | `"bow_hahog_root_uchar_10000.npz"` | Visual vocabulary file       |
| `bow_words_to_match` | int  | `50`                               | Words to explore per feature |
| `bow_num_checks`     | int  | `20`                               | Matching features to check   |
| `bow_matcher_type`   | str  | `"FLANN"`                          | Matcher for word assignment  |

### VLAD

| Parameter   | Type | Default                         | Description        |
| ----------- | ---- | ------------------------------- | ------------------ |
| `vlad_file` | str  | `"bow_hahog_root_uchar_64.npz"` | VLAD codebook file |

### Guided Matching

| Parameter                         | Type  | Default | Description                                          |
| --------------------------------- | ----- | ------- | ---------------------------------------------------- |
| `guided_spanning_trees`           | int   | `5`     | Randomized spanning-tree samples                     |
| `guided_spanning_trees_random`    | float | `0.5`   | Random ratio for edge weights                        |
| `guided_matching_threshold`       | float | `0.006` | Epipolar distance threshold (radians)                |
| `guided_min_length_initial`       | int   | `3`     | Min track length for initial triangulation           |
| `guided_min_length_final`         | int   | `3`     | Min track length for final triangulation             |
| `guided_extend_threshold`         | float | `0.002` | Reprojection threshold for track extension (radians) |
| `guided_extend_image_neighbors`   | int   | `50`    | Neighbor images considered                           |
| `guided_extend_feature_neighbors` | int   | `10`    | Max reprojected neighbors to check                   |

### Pair Selection

| Parameter                     | Type  | Default | Description                                      |
| ----------------------------- | ----- | ------- | ------------------------------------------------ |
| `matching_gps_distance`       | float | `150`   | Max GPS distance (m) between images for matching |
| `matching_gps_neighbors`      | int   | `0`     | Images selected by GPS distance (0 = no limit)   |
| `matching_time_neighbors`     | int   | `0`     | Images selected by capture time (0 = disabled)   |
| `matching_order_neighbors`    | int   | `0`     | Images selected by filename order (0 = disabled) |
| `matching_bow_neighbors`      | int   | `0`     | Images selected by BoW distance (0 = disabled)   |
| `matching_bow_gps_distance`   | float | `0`     | GPS preemption radius for BoW (0 = disabled)     |
| `matching_bow_gps_neighbors`  | int   | `0`     | GPS-preempted images for BoW (0 = no limit)      |
| `matching_bow_other_cameras`  | bool  | `false` | BoW: N neighbors per camera type                 |
| `matching_vlad_neighbors`     | int   | `0`     | Images selected by VLAD distance (0 = disabled)  |
| `matching_vlad_gps_distance`  | float | `0`     | GPS preemption radius for VLAD (0 = disabled)    |
| `matching_vlad_gps_neighbors` | int   | `0`     | GPS-preempted images for VLAD (0 = no limit)     |
| `matching_vlad_other_cameras` | bool  | `false` | VLAD: N neighbors per camera type                |
| `matching_graph_rounds`       | int   | `0`     | Triangulation-based pair selection rounds        |
| `matching_use_filters`        | bool  | `false` | Remove static matches via heuristics             |
| `matching_use_segmentation`   | bool  | `false` | Use segmentation to improve matching             |
| `matching_use_opk`            | bool  | `true`  | Use orientation (OPK) to improve matching        |

## Geometric Estimation

| Parameter                             | Type  | Default  | Description                                                        |
| ------------------------------------- | ----- | -------- | ------------------------------------------------------------------ |
| `robust_matching_threshold`           | float | `0.004`  | Fundamental matrix outlier threshold (fraction of image width)     |
| `robust_matching_calib_threshold`     | float | `0.004`  | Essential matrix outlier threshold during matching (radians)       |
| `robust_matching_min_match`           | int   | `20`     | Minimum matches to accept a pair                                   |
| `five_point_algo_threshold`           | float | `0.004`  | Essential matrix outlier threshold during reconstruction (radians) |
| `five_point_algo_min_inliers`         | int   | `20`     | Min inliers for valid two-view reconstruction                      |
| `five_point_refine_match_iterations`  | int   | `10`     | LM iterations for relative pose during matching                    |
| `five_point_refine_rec_iterations`    | int   | `1000`   | LM iterations for relative pose during reconstruction              |
| `five_point_reversal_check`           | bool  | `false`  | Check Necker reversal ambiguity (useful for aerial)                |
| `five_point_reversal_ratio`           | float | `0.95`   | Non-reversed/reversed point ratio threshold                        |
| `triangulation_threshold`             | float | `0.006`  | Outlier threshold for triangulated points (radians)                |
| `triangulation_min_ray_angle`         | float | `1.0`    | Min angle between views (degrees)                                  |
| `triangulation_min_depth`             | float | `0.001`  | Min depth to accept a point                                        |
| `triangulation_type`                  | str   | `"FULL"` | `FULL` (all rays) or `ROBUST` (RANSAC)                             |
| `triangulation_refinement_iterations` | int   | `10`     | LM iterations for point refinement                                 |
| `resection_threshold`                 | float | `0.004`  | Resection outlier threshold (radians)                              |
| `resection_min_inliers`               | int   | `10`     | Min resection inliers                                              |

## Tracks

| Parameter                       | Type  | Default | Description                               |
| ------------------------------- | ----- | ------- | ----------------------------------------- |
| `min_track_length`              | int   | `2`     | Minimum images per track                  |
| `use_depth_prior`               | bool  | `false` | Use depth prior during BA                 |
| `depth_std_deviation_m_default` | float | `1.0`   | Depth prior std deviation (m)             |
| `depth_is_radial`               | bool  | `false` | Depth is distance to camera center (vs Z) |
| `depth_is_inverted`             | bool  | `false` | Depth is stored as inverse depth          |

## Bundle Adjustment

| Parameter                         | Type  | Default       | Description                                          |
| --------------------------------- | ----- | ------------- | ---------------------------------------------------- |
| `reprojection_error_sd`           | float | `0.004`       | Reprojection error std dev                           |
| `exif_focal_sd`                   | float | `0.01`        | EXIF focal length std dev (log-scale)                |
| `aspect_ratio_sd`                 | float | `0.01`        | Aspect ratio (fu/fv) std dev (log-scale)             |
| `principal_point_sd`              | float | `0.01`        | Principal point std dev                              |
| `radial_distortion_k1_sd`         | float | `0.01`        | k1 std dev                                           |
| `radial_distortion_k2_sd`         | float | `0.01`        | k2 std dev                                           |
| `radial_distortion_k3_sd`         | float | `0.01`        | k3 std dev                                           |
| `radial_distortion_k4_sd`         | float | `0.01`        | k4 std dev                                           |
| `tangential_distortion_p1_sd`     | float | `0.01`        | p1 std dev                                           |
| `tangential_distortion_p2_sd`     | float | `0.01`        | p2 std dev                                           |
| `gcp_horizontal_sd`               | float | `0.01`        | GCP horizontal std dev (m)                           |
| `gcp_vertical_sd`                 | float | `0.1`         | GCP vertical std dev (m)                             |
| `gcp_global_weight`               | float | `0.04`        | GCP weight relative to observations                  |
| `gcp_observation_sd`              | float | `0.001`       | GCP reprojection observation std dev                 |
| `gcp_annealing_steps`             | list  | `[5.0, 25.0]` | GCP weight annealing schedule. `[1.0]` = single pass |
| `rig_translation_sd`              | float | `0.1`         | Rig translation std dev                              |
| `rig_rotation_sd`                 | float | `0.1`         | Rig rotation std dev                                 |
| `bundle_outlier_filtering_type`   | str   | `"FIXED"`     | `FIXED` (threshold) or `AUTO` (distribution-based)   |
| `bundle_outlier_auto_ratio`       | float | `3.0`         | AUTO: remove if error > ratio × mean                 |
| `bundle_outlier_fixed_threshold`  | float | `0.006`       | FIXED: max reprojection error                        |
| `optimize_camera_parameters`      | bool  | `true`        | Optimize intrinsics during BA                        |
| `optimize_rig_parameters`         | bool  | `false`       | Optimize rig parameters during BA                    |
| `bundle_max_iterations`           | int   | `100`         | Max optimizer iterations                             |
| `bundle_outlier_weight_threshold` | float | `0.5`         | Weight threshold for outlier removal                 |
| `bundle_irls_density_ratio`       | float | `0.001`       | IRLS outlier/inlier density ratio (all groups)       |
| `bundle_irls_gcp_density_ratio`   | float | `0.00001`     | IRLS density ratio for GCP residuals                 |
| `bundle_irls_gps_density_ratio`   | float | `0.00001`     | IRLS density ratio for GPS residuals                 |

## Incremental Reconstruction

| Parameter                        | Type  | Default  | Description                                           |
| -------------------------------- | ----- | -------- | ----------------------------------------------------- |
| `resect_redundancy_threshold`    | float | `0.7`    | Defer resection if (candidates / tracks) exceeds this |
| `retriangulation`                | bool  | `true`   | Periodically re-triangulate all points                |
| `retriangulation_ratio`          | float | `1.2`    | Re-triangulate when points grow by this ratio         |
| `bundle_analytic_derivatives`    | bool  | `true`   | Use analytic (vs auto-diff) derivatives               |
| `bundle_interval`                | int   | `999999` | Bundle after adding this many cameras                 |
| `bundle_new_points_ratio`        | float | `1.2`    | Bundle when points grow by this ratio                 |
| `local_bundle_radius`            | int   | `3`      | Image graph distance for local BA                     |
| `local_bundle_min_common_points` | int   | `20`     | Min common points to be a neighbor                    |
| `local_bundle_max_shots`         | int   | `30`     | Max shots in local BA                                 |
| `local_bundle_grid`              | int   | `12`     | Grid divisions for track selection (local BA)         |
| `final_bundle_grid`              | int   | `32`     | Grid divisions for track selection (final BA)         |
| `incremental_max_shots_count`    | int   | `0`      | Limit max shots (0 = unlimited, for debugging)        |
| `filter_final_point_cloud`       | bool  | `false`  | Remove uncertain/isolated points                      |
| `save_partial_reconstructions`   | bool  | `false`  | Save reconstruction at every iteration                |

## GPS / GCP Alignment

| Parameter                          | Type  | Default        | Description                              |
| ---------------------------------- | ----- | -------------- | ---------------------------------------- |
| `use_altitude_tag`                 | bool  | `true`         | Use EXIF altitude tag                    |
| `align_method`                     | str   | `"auto"`       | `auto`, `orientation_prior`, or `naive`  |
| `align_orientation_prior`          | str   | `"horizontal"` | `horizontal`, `vertical`, or `no_roll`   |
| `bundle_use_gps`                   | bool  | `true`         | Enforce GPS in BA                        |
| `bundle_use_gcp`                   | bool  | `true`         | Enforce GCP in BA                        |
| `bundle_compensate_gps_bias`       | bool  | `false`        | Per-camera GPS bias correction           |
| `gcp_reprojection_error_threshold` | float | `0.05`         | GCP reprojection error outlier threshold |

## Rigs

| Parameter                      | Type  | Default | Description                        |
| ------------------------------ | ----- | ------- | ---------------------------------- |
| `rig_calibration_subset_size`  | int   | `15`    | Rig instances used for calibration |
| `rig_calibration_completeness` | float | `0.85`  | Required reconstructed-image ratio |
| `rig_calibration_max_rounds`   | int   | `10`    | Max SfM rounds for rig calibration |

## Undistortion

| Parameter                    | Type | Default  | Description                         |
| ---------------------------- | ---- | -------- | ----------------------------------- |
| `undistorted_image_format`   | str  | `"jpg"`  | Output image format                 |
| `undistorted_image_max_size` | int  | `100000` | Max dimension of undistorted images |

## Depth Estimation (PatchMatch OpenCL)

OpenSfM computes dense depthmaps with a GPU PatchMatch estimator (run by `compute_depthmaps`). These parameters control the per-view estimation and the consistency/visibility cleaning stage.

| Parameter                          | Type  | Default  | Description                                                       |
| ---------------------------------- | ----- | -------- | ---------------------------------------------------------------- |
| `depthmap_num_neighbors`           | int   | `10`     | Candidate neighboring views per image                            |
| `depthmap_num_matching_views`      | int   | `8`      | Views actually used to estimate each depthmap                    |
| `depthmap_min_depth`               | float | `0`      | Min depth (m). `0` = auto-infer from the reconstruction          |
| `depthmap_max_depth`               | float | `0`      | Max depth (m). `0` = auto-infer from the reconstruction          |
| `depthmap_max_iterations`          | int   | `4`      | Max PatchMatch iterations                                        |
| `depthmap_patch_size`              | int   | `5`      | Correlation patch window size (odd)                              |
| `depthmap_max_image_size`          | int   | `3200`   | Max image dimension processed (longer side)                      |
| `depthmap_max_cost`                | float | `0`      | Max PatchMatch cost to keep a pixel (`0` = disabled)             |
| `depthmap_same_depth_threshold`    | float | `0.05`   | Depth-closeness threshold (clean stage)                          |
| `depthmap_min_consistent_views`    | int   | `3`      | Min consistent views to keep a pixel (clean stage)               |
| `depthmap_carving_threshold`       | float | `0.01`   | Relative depth margin for a space-carving vote                   |
| `depthmap_max_carved_views`        | int   | `1`      | Max carve votes before a pixel is discarded                      |
| `depthmap_carving_two_pass`        | bool  | `true`   | Pass 1 = consistency only, pass 2 = carving                      |
| `depthmap_grazing_cos_threshold`   | float | `0.2`    | Cosine below which a pixel is "grazing" (stricter filtering)     |
| `depthmap_sigma_spatial`           | float | `1.5`    | Bilateral NCC spatial sigma                                      |
| `depthmap_sigma_color`             | float | `≈0.098` | Bilateral NCC color sigma (normalized intensity, `25/255`)       |
| `depthmap_use_census`              | bool  | `true`   | Census-transform fallback in low-texture regions                 |
| `depthmap_hierarchy_levels`        | int   | `3`      | Multi-scale levels (`1` = full-res only)                         |
| `depthmap_checkerboard_filter`     | bool  | `false`  | Checkerboard bilateral median filter after PatchMatch            |
| `depthmap_speckle_min_size`        | int   | `0`      | Remove connected components smaller than this (px; `0` = off)    |
| `depthmap_gap_max_size`            | int   | `0`      | Max gap (px) for linear depth interpolation (`0` = disabled)     |
| `depthmap_smooth_weight`           | float | `0.2`    | Depth/normal smoothness weight                                   |
| `depthmap_anchor_views`            | int   | `2`      | Views from the previous iteration kept in view selection         |
| `depthmap_geom_consistency_weight` | float | `0.05`   | Geometric-consistency cost weight (`0` = disabled)               |
| `depthmap_cluster_max_size`        | int   | `16`     | Max reference views per cluster                                  |
| `depthmap_cluster_edge_max_factor` | float | `2.0`    | Drop covisibility edges beyond this × median baseline (`0` = off) |
| `depthmap_max_cluster_views`       | int   | `48`     | Hard cap on total views loaded per cluster batch                 |
| `depthmap_sfm_planar_prior`        | bool  | `false`  | Seed a Delaunay planar prior from SfM points                     |
| `depthmap_neighbor_min_angle`      | float | `3.0`    | Min baseline angle (deg) for neighbor selection                  |
| `depthmap_neighbor_max_angle`      | float | `60.0`   | Max baseline angle (deg) for neighbor selection                  |
| `depthmap_save_debug_ply`          | bool  | `false`  | Save per-shot/per-cluster debug PLYs (slow)                      |

### Fusion

Cleaned depthmaps are fused into a point cloud (and optional mesh) by a sparse-voxel-octree (SVO) TSDF fuser, run by `fuse_depthmaps`. The voxel size is auto-derived from the data's per-pixel ground footprint; these parameters tune extraction, decimation and optional photometric refinement.

| Parameter                                     | Type  | Default      | Description                                                           |
| --------------------------------------------- | ----- | ------------ | -------------------------------------------------------------------- |
| `depthmap_fusion_svo_voxel_level`             | str   | `"fine"`     | Voxel resolution: `fine` / `half` / `quarter` (progressively coarser) |
| `depthmap_fusion_svo_trunc_factor`            | float | `8`          | TSDF truncation distance = factor × voxel size                       |
| `depthmap_fusion_svo_min_weight`              | float | `2.0`        | Min voxel weight to extract a surface point                          |
| `depthmap_fusion_svo_num_levels`              | int   | `3`          | Multi-level fill passes (`1` = fine only)                            |
| `depthmap_fusion_svo_decimate_flat`           | int   | `4`          | Flat-surface decimation (`1` = off, `4` = keep 1/4)                  |
| `depthmap_fusion_svo_edge_threshold`          | float | `0.15`       | Normal-divergence threshold protecting edges from decimation         |
| `depthmap_fusion_svo_min_count`               | int   | `2`          | Min observation count for both voxels of a zero-crossing             |
| `depthmap_fusion_svo_relative_min_weight`     | float | `0.25`       | Local adaptive weight threshold for extraction                       |
| `depthmap_fusion_svo_max_voxels`              | int   | `80000000`   | Max voxels per SVO sub-volume (clusters are split to fit)            |
| `depthmap_fusion_svo_augment_neighbors`       | int   | `4`          | Extra neighbor shots per cluster shot (boundary quality)             |
| `depthmap_fusion_svo_coarse_factor`           | int   | `8`          | Coarse pre-scan cell size = factor × voxel size                      |
| `depthmap_fusion_svo_refine_enabled`          | bool  | `false`      | Photometric TSDF refinement                                          |
| `depthmap_fusion_svo_refine_iters`            | int   | `50`         | SDF refinement iterations                                            |
| `depthmap_fusion_svo_refine_lambda_reg`       | float | `0.3`        | Laplacian regularization weight (`0` = disabled)                     |
| `depthmap_fusion_svo_refine_lambda_anchor`    | float | `0.05`       | Anchor weight pulling the refined TSDF toward the fused value        |
| `depthmap_fusion_svo_refine_early_stop_rel`   | float | `0.2`        | Early-stop once RMS surface motion drops below this fraction of peak |
| `depthmap_fusion_svo_bake_reuse_max_fusers`   | int   | `4`          | Max fusers kept resident on the GPU at once                          |
| `depthmap_fusion_svo_bake_reuse_vram_fraction`| float | `0.5`        | VRAM fraction ceiling for retained fusers                            |
| `depthmap_fusion_mesh_enabled`                | bool  | `true`       | Also extract a Surface Nets mesh (`mesh.ply`) from the fused TSDF    |
| `depthmap_fusion_mesh_delete_batches`         | bool  | `true`       | Delete per-cluster `mesh_batch_*.ply` after merging                 |

### Dense disk reclamation and export

| Parameter                          | Type | Default | Description                                                            |
| ---------------------------------- | ---- | ------- | --------------------------------------------------------------------- |
| `depthmap_delete_raw_after_clean`  | bool | `true`  | Delete each `.raw.npz` once its `.clean.npz` is produced (lossless)   |
| `depthmap_delete_fusion_batches`   | bool | `true`  | Delete per-cluster `fused_batch_*.ply` after merging into `fused.ply` |
| `depthmap_delete_geotiff_batches`  | bool | `false` | Delete per-cluster DSM/ortho tiles after the final merge              |
| `dense_pointcloud_export_las`      | bool | `true`  | Also export the dense cloud as LAS (next to `fused.ply`)              |
| `dense_pointcloud_export_laz`      | bool | `true`  | Also export the dense cloud as LAZ                                    |

## Octree Tiling

Streaming tiles for the point-cloud/web viewer, built by `dense_merging` into the `point_cloud/` directory.

| Parameter                    | Type | Default   | Description                                                       |
| ---------------------------- | ---- | --------- | ---------------------------------------------------------------- |
| `octree_max_points_per_tile` | int  | `50000`   | Max points stored in a single octree tile                        |
| `octree_max_depth`           | int  | `15`      | Max octree depth (root = 0)                                      |
| `octree_lod_sample_count`    | int  | `10000`   | LOD representative points kept in each inner (non-leaf) tile     |
| `octree_split_depth`         | int  | `4`       | Out-of-core build: depth at which points are bucketed to disk    |
| `octree_max_bucket_points`   | int  | `8000000` | Out-of-core build: clouds larger than this are bucketed to disk  |

## DSM and Orthophoto

The Digital Surface Model (`dsm.tif`) and orthophoto (`ortho.tif`) are rendered during fusion and finalized by `dense_merging`. The ground sample distance is derived automatically from the fused voxel size.

| Parameter                    | Type  | Default | Description                                                                       |
| ---------------------------- | ----- | ------- | --------------------------------------------------------------------------------- |
| `dsm_enabled`                | bool  | `true`  | Render the DSM + orthophoto during fusion                                          |
| `dsm_territory_depth_factor` | float | `2.0`   | Drop depth samples farther than this × the view's median depth                     |
| `dsm_save_cluster_tiles`     | bool  | `true`  | Also write each cluster's own DSM/ortho GeoTIFF before merging (debug)             |
| `dsm_merge_feather`          | bool  | `true`  | Merge per-cluster tiles by distance-transform feather blending                     |
| `dsm_merge_max_ram_mb`       | int   | `512`   | Soft RAM budget (MB) for the band-by-band merge                                    |
| `dsm_wall_cull_nz`           | float | `0.5`   | Rasterize a surface triangle only if \|cos\| of its normal from vertical ≥ this    |
| `ortho_bake_n_final_views`   | int   | `3`     | Sharpest inlier views blended for the final ortho color                            |
| `ortho_bake_irls_iterations` | int   | `5`     | Tukey-biweight reweighting iterations for the color consensus                      |
| `ortho_bake_dsm_occlusion`   | bool  | `true`  | Drop views occluded by a taller surface when baking hole-filled cells              |
| `ortho_median_threshold`     | float | `24.0`  | Gated 3×3 median despeckle: replace a pixel only if it differs by more than this   |
| `hole_fill_diffuse_iters`    | int   | `64`    | Diffusion iterations used to fill tiny DSM holes                                   |
| `hole_fill_small_area_max`   | int   | `256`   | Holes with ≤ this many cells are "tiny" (filled by diffusion)                      |
| `hole_fill_footprint_close`  | int   | `256`   | Morphological closing of the footprint before large-hole filling                  |
| `hole_fill_low_percentile`   | float | `20.0`  | Large holes filled as a flat surface at this boundary-height percentile            |
| `hole_fill_occlusion_drop`   | float | `0.0`   | Fallback occluded-ground heuristic (superseded by ortho occlusion; `0` = off)     |
| `dsm_shock_iterations`       | int   | `6`     | Coherence-enhancing shock-filter iterations sharpening DSM edges                   |
| `dsm_shock_window`           | int   | `5`     | Structure-tensor half-window (cells); larger = straighter edges                    |
| `dsm_shock_dt`               | float | `0.25`  | Shock-filter time step (keep ≤ 0.5 for stability)                                  |
| `dsm_shock_coherence`        | float | `0.1`   | Along-edge diffusion weight (straightens jittered boundaries)                      |
| `dsm_shock_edge_slope`       | float | `2.0`   | Edge-strength gate: shock fires only where the local slope exceeds this            |

## Multi-Processing

| Parameter      | Type | Default | Description                                  |
| -------------- | ---- | ------- | -------------------------------------------- |
| `processes`    | int  | `1`     | Number of parallel processes                 |
| `io_processes` | int  | `4`     | Image-reading threads (when `processes > 1`) |

## Submodel Split & Merge

| Parameter                          | Type  | Default                            | Description                          |
| ---------------------------------- | ----- | ---------------------------------- | ------------------------------------ |
| `submodel_size`                    | int   | `80`                               | Average images per submodel          |
| `submodel_overlap`                 | float | `30.0`                             | Overlap radius between submodels (m) |
| `submodels_relpath`                | str   | `"submodels"`                      | Submodels directory                  |
| `submodel_relpath_template`        | str   | `"submodels/submodel_%04d"`        | Submodel directory template          |
| `submodel_images_relpath_template` | str   | `"submodels/submodel_%04d/images"` | Submodel images directory template   |
