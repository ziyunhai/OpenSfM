# pyre-strict

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, IO, List, Union

import yaml


@dataclass
class OpenSfMConfig:
    ##################################
    # Params for metadata
    ##################################
    use_exif_size: bool = True
    # Treat images from unknown camera models as coming from different cameras
    unknown_camera_models_are_different: bool = False
    # Default projection type to use when it cannot be inferred from EXIF metadata
    default_projection_type: str = "perspective"
    # Default focal length to sensor size ratio to use when it cannot be inferred from EXIF metadata
    default_focal_prior: float = 0.85
    # Enable downloading missing datum grids from the PROJ CDN
    proj_cdn_enabled: bool = True
    # Additional specific folder to load/store the geoid files and grids
    proj_grid_cache_dir: str = ""

    ##################################
    # Params for features
    ##################################
    # Feature type (AKAZE, SURF, SIFT, HAHOG, ORB, DSPSIFT)
    feature_type: str = "HAHOG"
    # If true, apply square root mapping to features
    feature_root: bool = True
    # If fewer frames are detected, sift_peak_threshold/surf_hessian_threshold is reduced.
    feature_min_frames: int = 4000
    # Same as above but for panorama images
    feature_min_frames_panorama: int = 16000
    # Resize the image if its size is larger than specified. Set to -1 for original size
    feature_process_size: int = 2048
    # Same as above but for panorama images
    feature_process_size_panorama: int = 4096
    feature_use_adaptive_suppression: bool = False
    # Bake segmentation info (class and instance) in the feature data. Thus it is done once for all at extraction time.
    features_bake_segmentation: bool = False
    # Maximum amount of memory to use for feature extraction (in MB). See default in features_processing.py.
    mem_ceiling: int | None = None
    # Ratio of the memory ceiling to use for feature extraction. See default in features_processing.py.
    mem_ratio: float | None = None

    ##################################
    # Params for SIFT
    ##################################
    # Smaller value -> more features
    sift_peak_threshold: float = 0.1
    # See OpenCV doc
    sift_edge_threshold: int = 10
    # See OpenCV doc
    sift_nfeatures: int = 0
    # See OpenCV doc
    sift_octave_layers: int = 3
    # See OpenCV doc
    sift_sigma: float = 1.6

    ##################################
    # Params for DSPSIFT
    ##################################
    dspsift_peak_threshold: float = 0.006
    dspsift_edge_threshold: int = 10

    ##################################
    # Params for SURF
    ##################################
    # Smaller value -> more features
    surf_hessian_threshold: float = 3000
    # See OpenCV doc
    surf_n_octaves: int = 4
    # See OpenCV doc
    surf_n_octavelayers: int = 2
    # See OpenCV doc
    surf_upright: int = 0

    ##################################
    # Params for AKAZE (See details in lib/src/third_party/akaze/AKAZEConfig.h)
    ##################################
    # Maximum octave evolution of the image 2^sigma (coarsest scale sigma units)
    akaze_omax: int = 4
    # Detector response threshold to accept point
    akaze_dthreshold: float = 0.001
    # Feature type
    akaze_descriptor: str = "MSURF"
    # Size of the descriptor in bits. 0->Full size
    akaze_descriptor_size: int = 0
    # Number of feature channels (1,2,3)
    akaze_descriptor_channels: int = 3
    akaze_kcontrast_percentile: float = 0.7
    akaze_use_isotropic_diffusion: bool = False

    ##################################
    # Params for HAHOG
    ##################################
    hahog_peak_threshold: float = 0.00001
    hahog_edge_threshold: float = 10
    hahog_normalize_to_uchar: bool = True

    ##################################
    # Params for general matching
    ##################################
    # Ratio test for matches
    lowes_ratio: float = 0.85
    # FLANN, BRUTEFORCE, WORDS, OPENCL_HAMMING or OPENCL_BF
    matcher_type: str = "OPENCL_HAMMING"
    # Match symmetrically or one-way
    symmetric_matching: bool = True
    # Number of image pairs used to train the binary projection (OPENCL_HAMMING)
    binary_training_pairs: int = 100

    ##################################
    # Params for FLANN matching
    ##################################
    # Algorithm type (KMEANS, KDTREE)
    flann_algorithm: str = "KMEANS"
    # See OpenCV doc
    flann_branching: int = 8
    # See OpenCV doc
    flann_iterations: int = 10
    # See OpenCV doc
    flann_tree: int = 8
    # Smaller -> Faster (but might lose good matches)
    flann_checks: int = 20

    ##################################
    # Params for BoW matching
    ##################################
    bow_file: str = "bow_hahog_root_uchar_10000.npz"
    # Number of words to explore per feature.
    bow_words_to_match: int = 50
    # Number of matching features to check.
    bow_num_checks: int = 20
    # Matcher type to assign words to features
    bow_matcher_type: str = "FLANN"

    ##################################
    # Params for VLAD matching
    ##################################
    vlad_file: str = "bow_hahog_root_uchar_64.npz"

    ##################################
    # Params for guided matching
    ##################################
    # Number of randomized spanning-trees to samples over the tracks-graph
    guided_spanning_trees: int = 5
    # Random ratio higher bound edges are multiplied with
    guided_spanning_trees_random: float = 0.5
    # Threshold for epipolar distance for accepting a match in radians
    guided_matching_threshold: float = 0.006
    # Minimum track length for initial triangulation
    guided_min_length_initial: int = 3
    # Minimum track length for final triangulation
    guided_min_length_final: int = 3
    # Threshold of reprojection for extending a track within a new image (in radians)
    guided_extend_threshold: float = 0.002
    # Number of images considered as neighbors of another one
    guided_extend_image_neighbors: int = 50
    # Maximum number of reprojected neighbors (in the tracks-graph) to check when extending a track within a new image
    guided_extend_feature_neighbors: int = 10

    ##################################
    # Params for matching
    ##################################
    # Maximum gps distance between two images for matching
    matching_gps_distance: float = 150
    # Number of images to match selected by GPS distance. Set to 0 to use no limit (or disable if matching_gps_distance is also 0)
    matching_gps_neighbors: int = 0
    # Number of images to match selected by time taken. Set to 0 to disable
    matching_time_neighbors: int = 0
    # Number of images to match selected by image name. Set to 0 to disable
    matching_order_neighbors: int = 0
    # Number of images to match selected by BoW distance. Set to 0 to disable
    matching_bow_neighbors: int = 0
    # Maximum GPS distance for preempting images before using selection by BoW distance. Set to 0 to disable
    matching_bow_gps_distance: float = 0
    # Number of images (selected by GPS distance) to preempt before using selection by BoW distance. Set to 0 to use no limit (or disable if matching_bow_gps_distance is also 0)
    matching_bow_gps_neighbors: int = 0
    # If True, BoW image selection will use N neighbors from the same camera + N neighbors from any different camera. If False, the selection will take the nearest neighbors from all cameras.
    matching_bow_other_cameras: bool = False
    # Number of images to match selected by VLAD distance. Set to 0 to disable
    matching_vlad_neighbors: int = 0
    # Maximum GPS distance for preempting images before using selection by VLAD distance. Set to 0 to disable
    matching_vlad_gps_distance: float = 0
    # Number of images (selected by GPS distance) to preempt before using selection by VLAD distance. Set to 0 to use no limit (or disable if matching_vlad_gps_distance is also 0)
    matching_vlad_gps_neighbors: int = 0
    # If True, VLAD image selection will use N neighbors from the same camera + N neighbors from any different camera. If False, the selection will take the nearest neighbors from all cameras.
    matching_vlad_other_cameras: bool = False
    # Number of rounds to run when running triangulation-based pair selection
    matching_graph_rounds: int = 0
    # If True, removes static matches using ad-hoc heuristics
    matching_use_filters: bool = False
    # Use segmentation information (if available) to improve matching
    matching_use_segmentation: bool = False
    # Use orientation (if available) to improve matching
    matching_use_opk: bool = True

    ##################################
    # Params for geometric estimation
    ##################################
    # Outlier threshold for fundamental matrix estimation as portion of image width
    robust_matching_threshold: float = 0.004
    # Outlier threshold for essential matrix estimation during matching in radians
    robust_matching_calib_threshold: float = 0.004
    # Minimum number of matches to accept matches between two images
    robust_matching_min_match: int = 20
    # Outlier threshold for essential matrix estimation during incremental reconstruction in radians
    five_point_algo_threshold: float = 0.004
    # Minimum number of inliers for considering a two view reconstruction valid
    five_point_algo_min_inliers: int = 20
    # Number of LM iterations to run when refining relative pose during matching
    five_point_refine_match_iterations: int = 10
    # Number of LM iterations to run when refining relative pose during reconstruction
    five_point_refine_rec_iterations: int = 1000
    # Check for Necker reversal ambiguities. Useful for long focal length with long distance capture (aerial manned)
    five_point_reversal_check: bool = False
    # Ratio of triangulated points non-reversed/reversed when checking for Necker reversal ambiguities
    five_point_reversal_ratio: float = 0.95
    # Outlier threshold for accepting a triangulated point in radians
    triangulation_threshold: float = 0.006
    # Minimum angle between views to accept a triangulated point
    triangulation_min_ray_angle: float = 1.0
    # Minimum depth to accept a triangulated point
    triangulation_min_depth: float = 0.001
    # Triangulation type : either considering all rays (FULL), or sing a RANSAC variant (ROBUST)
    triangulation_type: str = "FULL"
    # Number of LM iterations to run when refining a point
    triangulation_refinement_iterations: int = 10
    # Outlier threshold for resection in radians
    resection_threshold: float = 0.004
    # Minimum number of resection inliers to accept it
    resection_min_inliers: int = 10

    ##################################
    # Params for track creation
    ##################################
    # Minimum number of features/images per track
    min_track_length: int = 2
    # Whether to use depth prior during BA
    use_depth_prior: bool = False
    # Depth prior default std deviation
    depth_std_deviation_m_default: float = 1.0
    # Whether depth is radial (distance to camera center) or Z value
    depth_is_radial: bool = False
    # Whether depth is stored as inverted depth
    depth_is_inverted: bool = False

    ##################################
    # Params for bundle adjustment
    ##################################
    # The standard deviation of the reprojection error
    reprojection_error_sd: float = 0.004
    # The standard deviation of the exif focal length in log-scale
    exif_focal_sd: float = 0.01
    # The standard deviation of aspect ratio, i.e. fu/fv, in log-scale
    aspect_ratio_sd: float = 0.01
    # The standard deviation of the principal point coordinates
    principal_point_sd: float = 0.01
    # The standard deviation of the first radial distortion parameter
    radial_distortion_k1_sd: float = 0.01
    # The standard deviation of the second radial distortion parameter
    radial_distortion_k2_sd: float = 0.01
    # The standard deviation of the third radial distortion parameter
    radial_distortion_k3_sd: float = 0.01
    # The standard deviation of the fourth radial distortion parameter
    radial_distortion_k4_sd: float = 0.01
    # The standard deviation of the first tangential distortion parameter
    tangential_distortion_p1_sd: float = 0.01
    # The standard deviation of the second tangential distortion parameter
    tangential_distortion_p2_sd: float = 0.01
    # The default horizontal standard deviation of the GCPs (in meters)
    gcp_horizontal_sd: float = 0.01
    # The default vertical standard deviation of the GCPs (in meters)
    gcp_vertical_sd: float = 0.1
    # Global weight for GCPs relative to regular observations (scaled by sqrt of avg tracks/shot)
    gcp_global_weight: float = 0.04
    # The standard deviation of GCP reprojection observations (in normalized image coordinates)
    gcp_observation_sd: float = 0.001
    # Annealing schedule for GCP weights: list of multipliers applied to gcp_global_weight
    # across successive bundle passes. Set to [1.0] to disable annealing (single pass).
    gcp_annealing_steps: List[float] = field(
        default_factory=lambda: [5.0, 25.0]
    )
    # The standard deviation of the rig translation
    rig_translation_sd: float = 0.1
    # The standard deviation of the rig rotation
    rig_rotation_sd: float = 0.1
    # Type of threshold for filtering outlier : either fixed value (FIXED) or based on actual distribution (AUTO)
    bundle_outlier_filtering_type: str = "FIXED"
    # For AUTO filtering type, projections with larger reprojection than ratio-times-mean, are removed
    bundle_outlier_auto_ratio: float = 3.0
    # For FIXED filtering type, projections with larger reprojection error after bundle adjustment are removed
    bundle_outlier_fixed_threshold: float = 0.006
    # Optimize internal camera parameters during bundle
    optimize_camera_parameters: bool = True
    # Optimize rig parameters during bundle
    optimize_rig_parameters: bool = False
    # Maximum optimizer iterations.
    bundle_max_iterations: int = 100
    # Weight threshold below which a point is considered an outlier and removed from the reconstruction
    bundle_outlier_weight_threshold: float = 0.5
    # Default ratio of outlier to inlier density peaks for IRLS mixture model (all error groups)
    bundle_irls_density_ratio: float = 0.001
    # Density ratio override for GCP 2D projection residuals (set higher to be more lenient on GCPs)
    bundle_irls_gcp_density_ratio: float = 0.00001
    # Density ratio override for GPS residuals (set higher to be more lenient on GPS)
    bundle_irls_gps_density_ratio: float = 0.00001

    # Ratio of (resection candidates / total tracks) of a given image so that it is culled at resection and resected later
    resect_redundancy_threshold: float = 0.7
    # Retriangulate all points from time to time
    retriangulation: bool = True
    # Retriangulate when the number of points grows by this ratio
    retriangulation_ratio: float = 1.2
    # Use analytic derivatives or auto-differentiated ones during bundle adjustment
    bundle_analytic_derivatives: bool = True
    # Bundle after adding 'bundle_interval' cameras
    bundle_interval: int = 999999
    # Bundle when the number of points grows by this ratio
    bundle_new_points_ratio: float = 1.2
    # Max image graph distance for images to be included in local bundle adjustment
    local_bundle_radius: int = 3
    # Minimum number of common points betwenn images to be considered neighbors
    local_bundle_min_common_points: int = 20
    # Max number of shots to optimize during local bundle adjustment
    local_bundle_max_shots: int = 30
    # Number of grid division for seleccting tracks in local bundle adjustment
    local_bundle_grid: int = 12
    # Number of grid division for selecting tracks in final bundle adjustment
    final_bundle_grid: int = 32
    # Min number of shots above which global bundle switches to the stochastic solver
    stochastic_bundle_shot_count: int = 4000
    # Max interior cameras optimized per stochastic round (N)
    stochastic_bundle_max_shots: int = 500
    # Number of seed shots fed to each stochastic round
    stochastic_bundle_random_shots: int = 20
    # How much of stochastic_bundle_max_shots, do we reserve for GCP-anchored shots
    stochastic_bundle_gcp_seeds_ratio: float = 0.5
    # Graph-hop radius for stochastic rounds (large so seeds can reach the N budget)
    stochastic_bundle_radius: int = 100
    # Upper bound on the number of stochastic rounds K (K is derived from scene size)
    stochastic_bundle_max_rounds: int = 100
    # For debugging purpose of large datasets: limit the maximum number of shots in incremental reconstruction
    incremental_max_shots_count: int = 0
    # Number of different pairs to try for incremental reconstruction bootstrap
    incremental_bootstrap_tries: int = 10
    # Average inlier ratio on resected shots to accept a bootstrap pair and grow it
    incremental_bootstrap_min_inliers_ratio: float = 0.80

    # Remove uncertain and isolated points from the final point cloud
    filter_final_point_cloud: bool = False

    # Save reconstructions at every iteration
    save_partial_reconstructions: bool = False

    ##################################
    # Params for GPS/GCP alignment
    ##################################
    # Use or ignore EXIF altitude tag
    use_altitude_tag: bool = True
    # orientation_prior or naive
    align_method: str = "auto"
    # horizontal, vertical or no_roll
    align_orientation_prior: str = "horizontal"
    # Enforce GPS position in bundle adjustment
    bundle_use_gps: bool = True
    # Enforce Ground Control Point position in bundle adjustment
    bundle_use_gcp: bool = True
    # Compensate GPS with a per-camera similarity transform
    bundle_compensate_gps_bias: bool = False
    # Thrershold for the reprojection error of GCPs to be considered outliers
    gcp_reprojection_error_threshold: float = 0.05

    ##################################
    # Params for rigs
    ##################################
    # Number of rig instances to use when calibration rigs
    rig_calibration_subset_size: int = 15
    # Ratio of reconstructed images needed to consider a reconstruction for rig calibration
    rig_calibration_completeness: float = 0.85
    # Number of SfM tentatives to run until we get a satisfying reconstruction
    rig_calibration_max_rounds: int = 10

    ##################################
    # Params for image undistortion
    ##################################
    # Format in which to save the undistorted images
    undistorted_image_format: str = "jpg"
    # Max width and height of the undistorted image
    undistorted_image_max_size: int = 100000

    ##################################
    # Params for depth estimation (PatchMatch OpenCL)
    ##################################
    # Number of neighboring views considered as candidates
    depthmap_num_neighbors: int = 10
    # Number of neighboring views used for each depthmap
    depthmap_num_matching_views: int = 8
    # Minimum depth in meters.  Set to 0 to auto-infer from the reconstruction.
    depthmap_min_depth: float = 0
    # Maximum depth in meters.  Set to 0 to auto-infer from the reconstruction.
    depthmap_max_depth: float = 0
    # Maximum number of PatchMatch iterations
    depthmap_max_iterations: int = 4
    # Correlation patch window size (must be odd)
    depthmap_patch_size: int = 5
    # Maximum image dimension for processing (longer side)
    depthmap_max_image_size: int = 3200
    # Maximum PatchMatch cost to keep a pixel (0 = disabled)
    depthmap_max_cost: float = 0
    # Threshold to measure depth closeness (clean stage)
    depthmap_same_depth_threshold: float = 0.05
    # Min number of consistent views in clean stage
    depthmap_min_consistent_views: int = 3
    # Relative depth margin for space-carving votes; neighbor sees further by this fraction → carve vote.
    depthmap_carving_threshold: float = 0.05
    # Max carve votes a pixel can accumulate before being discarded.
    depthmap_max_carved_views: int = 1
    # Two-pass cleaning: pass 1 uses consistency only (no carving) on raw
    # depthmaps, pass 2 uses carving on the cleaned depthmaps from pass 1.
    depthmap_carving_two_pass: bool = True
    # Cosine threshold for grazing-angle detection (below → pixel is grazing, stricter filtering).
    depthmap_grazing_cos_threshold: float = 0.2
    # Save per-shot raw/clean PLYs and per-cluster debug PLYs (slow, for debugging only).
    depthmap_save_debug_ply: bool = True
    # ── Disk reclamation (intermediate depthmap/PLY cleanup) ──
    # Delete each raw depthmap (.raw.npz) once the whole cleaning phase has
    # produced its clean counterpart. Raw maps are only consumed by cleaning,
    # so this is lossless; disable to keep raw maps for inspection/re-cleaning.
    depthmap_delete_raw_after_clean: bool = True
    # Delete per-cluster fused_batch_*.ply once they are merged into fused.ply.
    # They are fully consumed by the merge and redundant afterwards.
    depthmap_delete_fusion_batches: bool = True
    # Additionally export the merged dense cloud as LAS / LAZ (next to fused.ply).
    dense_pointcloud_export_las: bool = True
    dense_pointcloud_export_laz: bool = True
    # Spatial sigma for bilateral NCC weighting
    depthmap_sigma_spatial: float = 1.5
    # Color sigma for bilateral NCC weighting, in normalized [0,1] intensity units.
    depthmap_sigma_color: float = 25.0 / 255.0
    # Use Census transform as fallback when bilateral NCC fails (low-texture regions).
    depthmap_use_census: bool = True
    # Number of multi-scale hierarchy levels (1 = full-res only, 2 = half+full,  3 = quarter+half+full, etc.).
    depthmap_hierarchy_levels: int = 3
    # Enable checkerboard bilateral median filter after PatchMatch iterations
    depthmap_checkerboard_filter: bool = False
    # Minimum connected component size in pixels; smaller segments are removed as speckle noise
    depthmap_speckle_min_size: int = 0
    # Maximum gap size in pixels for linear depth interpolation (0 = disabled)
    depthmap_gap_max_size: int = 0
    # Depth/normal smoothness weight for PatchMatch
    depthmap_smooth_weight: float = 0.2
    # Number of views from previous iteration that are forcibly kept in the view selection (anchoring).
    depthmap_anchor_views: int = 2
    # Weight for geometric consistency cost (0 = disabled). Applied per source view.
    depthmap_geom_consistency_weight: float = 0.05
    # Maximum number of reference views per cluster for geometric consistency.
    depthmap_cluster_max_size: int = 8
    # Use SfM points to seed a Delaunay planar prior before PatchMatch iterations
    depthmap_sfm_planar_prior: bool = False
    # Minimum baseline angle (degrees) for neighbor selection.
    depthmap_neighbor_min_angle: float = 3.0
    # Maximum baseline angle (degrees) for neighbor selection.
    depthmap_neighbor_max_angle: float = 60.0
    # SVO voxel size in world units (meters). Smaller = finer but more memory.
    depthmap_fusion_svo_voxel_size: float = 0.05
    # SVO truncation factor: truncation_distance = factor * voxel_size.
    depthmap_fusion_svo_trunc_factor: float = 8
    # SVO minimum weight for extracting points
    depthmap_fusion_svo_min_weight: float = 2.0
    # Number of multi-level fill passes (1=fine only, 3=fine+2 coarser).
    depthmap_fusion_svo_num_levels: int = 3
    # Flat-surface decimation factor for extraction (1=off, 4=keep 1/4). Reduces point density on flat surfaces while preserving sharp edges.
    depthmap_fusion_svo_decimate_flat: int = 4
    # Edge detection threshold for decimation (0-1). Crossings with normal divergence above this are considered edges and never decimated.
    depthmap_fusion_svo_edge_threshold: float = 0.15
    # Minimum observation count for both voxels in a zero-crossing.
    depthmap_fusion_svo_min_count: int = 2
    # Local adaptive weight threshold for extracting surfaces : allows weak areas to self-calibrate.
    depthmap_fusion_svo_relative_min_weight: float = 0.25
    # Maximum unique voxels per SVO sub-volume : clusters are spatially split so each piece stays within this budget.
    depthmap_fusion_svo_max_voxels: int = 80_000_000
    # Number of extra neighbor shots per cluster shot for fusion augmentation.
    # Adds views from outside the cluster to improve boundary quality.
    depthmap_fusion_svo_augment_neighbors: int = 4
    # Coarse grid cell size multiplier for pre-scan (cell = factor * voxel_size).
    depthmap_fusion_svo_coarse_factor: int = 8
    # Photometric TSDF refinement
    depthmap_fusion_svo_refine_enabled: bool = True
    # Number of SDF refinement iterations.
    depthmap_fusion_svo_refine_iters: int = 50
    # Laplacian regularization weight (0 = disabled).
    depthmap_fusion_svo_refine_lambda_reg: float = 0.3
    # Surface-stabilization anchor weight: pulls the refined TSDF back toward the fused value (energy mu*(phi-phi0)^2)
    depthmap_fusion_svo_refine_lambda_anchor: float = 0.05
    # Early-stop: halt refinement once the per-iteration RMS surface motion falls below this fraction of its peak (the descent has converged).
    depthmap_fusion_svo_refine_early_stop_rel: float = 0.2
    # This caps how many fusers are kept alive on the GPU at once (each holds a hash table)
    depthmap_fusion_svo_bake_reuse_max_fusers: int = 4
    # Hard ceiling, as a fraction of device VRAM, on the bytes of retained fusers (hash table + refine images) kept resident
    depthmap_fusion_svo_bake_reuse_vram_fraction: float = 0.5

    ##################################
    # Params for octree point cloud tiling (viewer streaming)
    ##################################
    # Maximum number of points stored in a single octree tile
    octree_max_points_per_tile: int = 50000
    # Maximum octree depth (root = 0)
    octree_max_depth: int = 15
    # Number of LOD representative points kept in each inner (non-leaf) tile
    octree_lod_sample_count: int = 10000
    # Out-of-core octree build: octree depth at which points are bucketed to disk (bucket count <= 8^depth).
    octree_split_depth: int = 4
    # Out-of-core octree build: clouds with more points than this are bucketed to disk so peak
    octree_max_bucket_points: int = 8_000_000

    ##################################
    # Params for DSM (Digital Surface Model)and ortho generation
    ##################################
    # Controls DSM + ortho rendering.
    dsm_enabled: bool = True
    # Ground sample distance in meters/pixel. 0 = auto from voxel size.
    dsm_gsd: float = 0.0
    # Wall cull for the DSM mesh: a surface-net triangle is rasterized only if |cos| of its normal from vertical >= this
    dsm_wall_cull_nz: float = 0.5
    # Orthophoto color baking (svo_bake_colors): number of sharpest inlier views blended for the final color
    ortho_bake_n_final_views: int = 3
    # Tukey-biweight reweighting iterations for the robust color consensus.
    ortho_bake_irls_iterations: int = 5
    # Gated 3x3 median despeckle of the baked ortho: a pixel is replaced by the local median only when it differs from it by more than this
    ortho_median_threshold: float = 24.0
    # DSM Post-process hole filling : a hole's connected component is "tiny" when it has <= hole_fill_small_area_max cells
    hole_fill_diffuse_iters: int = 64
    # DSM Post-process hole filling : when a hole is "tiny", it is filled by diffusion
    hole_fill_small_area_max: int = 256
    # DSM Post-process hole filling : hole_fill_small_area_max is exceeded, the hole is filled by linear interpolation
    hole_fill_footprint_close: int = 256
    # Coherence-enhancing shock filter on the DSM that sharpens edges
    dsm_shock_iterations: int = 6
    # Structure-tensor half-window in cells (larger = straighter / more coherent).
    dsm_shock_window: int = 5
    # Time step; keep <= 0.5 for stability.
    dsm_shock_dt: float = 0.25
    # Along-edge (tangential) diffusion weight; straightens the voxel-jittered boundary while the shock sharpens across it.
    dsm_shock_coherence: float = 0.1
    # Edge-strength gate: the shock only fires where the local slope (rise/run = height gradient / gsd) exceeds this
    dsm_shock_edge_slope: float = 2.0

    ##################################
    # Params for multi-processing/threading
    ##################################
    # Number of threads to use
    processes: int = 1
    # When processes > 1, number of threads used for reading images
    io_processes: int = 4

    ##################################
    # Params for submodel split and merge
    ##################################
    # Average number of images per submodel
    submodel_size: int = 80
    # Radius of the overlapping region between submodels
    submodel_overlap: float = 30.0
    # Relative path to the submodels directory
    submodels_relpath: str = "submodels"
    # Template to generate the relative path to a submodel directory
    submodel_relpath_template: str = "submodels/submodel_%04d"
    # Template to generate the relative path to a submodel images directory
    submodel_images_relpath_template: str = "submodels/submodel_%04d/images"

    ##################################
    # Params for report localization
    ##################################
    # Unit system for the quality report: "metric" or "imperial"
    report_unit_system: str = "metric"
    # Language for the quality report: "en", "fr", "es", "de", "it"
    report_language: str = "en"


def default_config() -> Dict[str, Any]:
    """Return default configuration"""
    return asdict(OpenSfMConfig())


def load_config(filepath: str) -> Dict[str, Any]:
    """DEPRECATED: = Load config from a config.yaml filepath"""
    if not os.path.isfile(filepath):
        return default_config()

    with open(filepath) as fin:
        return load_config_from_fileobject(fin)


def load_config_from_fileobject(
    f: Union[IO[bytes], IO[str], bytes, str],
) -> Dict[str, Any]:
    """Load config from a config.yaml fileobject"""
    config = default_config()

    new_config = yaml.safe_load(f)
    if new_config:
        for k, v in new_config.items():
            config[k] = v

    return config
