# 🛰️ RELEASE LOG

## 1.0

## 🧟 Why Now ?
It's been a long way since OpenSfM was created by Pau Gargallo in 2013 and Mapillary. Since then, OpenSfM has been internally used as the pipeline that powers
Mapillary's planet-scale 3D reconstruction, and has been used as the core SfM of OpenDroneMap and WebODM since 2016. Through these
many years, development had its high and lows, mainly focused on supporting Mapillary platform. While at Meta, development continued, but support has been stalled in the past few years, until it was announced as no longer in active development.

We loved this project and did not want it to die, so we rolled out sleeves and went back to work, eliminating major bottlenecks (speed), while adding new features. 

This 1.0 release is aimed at supporting our two biggest users : OpenDroneMap (ODM/ODX) and WebODM/ODX, hence why there's a focus on GIS/Geo workflows.

This release is a big jump and the first major so here is the list of features :

## 🪄 New Features

> 📚 Full documentation lives in [`doc/`](doc/) — new users should start with the [quickstart](doc/quickstart.md), then the [pipeline command reference](doc/using.md) and the [configuration reference](doc/configuration.md) (the doc mirror of every option in [`opensfm/config.py`](opensfm/config.py)).

> 🧰 Ready-made `config.yaml` presets tuned for common capture types ship under [`configs/`](configs/) — `aerial`, `terrestrial` and `object`. Copy one into your dataset to start from sensible defaults; see [workflow presets](doc/using.md#workflow-presets-configs).

### 🧭 Geolocation

> Guide: [Georeferencing & GIS outputs](doc/georeferencing.md)

- GPS positions with X/Y/Z standard deviation (per image) — read from EXIF or imported with [`extract_geolocation`](doc/using.md#extract_geolocation) (config: [`bundle_use_gps`, `bundle_compensate_gps_bias`](doc/configuration.md#gps--gcp-alignment))
- GCP/Checkpoints with X/Y/Z standard deviation (per point) — see [Ground control points](doc/ground_control_points.md) (config: [`bundle_use_gcp`, `gcp_horizontal_sd`, `gcp_vertical_sd`](doc/configuration.md#bundle-adjustment))
- Horizontal+Vertical coordinate system (EPSG codes, PROJ string) — compound EPSG (e.g. `EPSG:4979+5773`) and raw `+proj=…` strings; see [the output coordinate system](doc/georeferencing.md#the-output-coordinate-system)
- Geoids support through direct download to PROJ CDN (config: [`proj_cdn_enabled`, `proj_grid_cache_dir`](doc/configuration.md#metadata))
- TXT geolocation file parsing for generating geolocation input (`exif_overrides.json`) from a text file — [`extract_geolocation --geotag-file`](doc/using.md#extract_geolocation)

### 🧩 SfM

![SfM Reconstruction](doc/images/viewer.png)

> Reference: [Pipeline commands](doc/using.md) · [Configuration reference](doc/configuration.md)

- Features : HAHOG, AKAZE, SIFT, DSP-SIFT, SURF (and ORB) — config: [`feature_type`](doc/configuration.md#features)
- Descriptors : super-fast online-trained (and pre-trained) binary quantized matching with GPU (OpenCL) matching. Classic OpenCV FLANN matching. — config: [`matcher_type` (`OPENCL_HAMMING` / `OPENCL_BF` / `FLANN`), `binary_training_pairs`](doc/configuration.md#matching)
- Matching : EXIF-based (GPS), image-based (BoW/VLAD) or hybrid (both) matching. — config: [Pair selection](doc/configuration.md#pair-selection) (`matching_gps_distance`, `matching_bow_neighbors`, `matching_vlad_neighbors`)
- SfM : fast incremental and direct aerotriangulation support. — [`reconstruct`](doc/using.md#reconstruct); config: [Incremental reconstruction](doc/configuration.md#incremental-reconstruction).
- Scales to massive scenes : stochastic global bundle adjustment for very large reconstructions (config: [Stochastic bundle](doc/configuration.md#stochastic-bundle)) and split-into-submodels processing — see [Large datasets](doc/large_datasets.md)
- Georeferencing : GPS and GCP referencing with adaptive datum shift compensation — config: [`bundle_compensate_gps_bias`, `align_method`](doc/configuration.md#gps--gcp-alignment); see [Georeferencing](doc/georeferencing.md)
- Cameras : perspective, fisheye and panorama (spherical) camera models support. Many different distortion flavors (Brown and others). Rolling-shutter correction via two-step compensation. — see [Camera models](doc/geometry.md#camera-models) and the [supported projection types](doc/using.md#extract_metadata) (config: [`default_projection_type`](doc/configuration.md#metadata))
- Rig : full rig support. Can automatically calibrate the rig geometry. — see [Rig models](doc/rig.md) / [`create_rig`](doc/using.md#create_rig) (config: [`optimize_rig_parameters`](doc/configuration.md#bundle-adjustment), [Rigs](doc/configuration.md#rigs))
- JSON and GeoJSON export — plus interoperability exporters for COLMAP, OpenMVS, Bundler, VisualSFM, PMVS and PLY — see [the exporters](doc/using.md#other-exporters) and [`export_geocoords`](doc/using.md#export_geocoords)

### 🍇 Dense

![Dense Reconstruction](doc/images/dense.png)

> How-to: [Dense reconstruction & 2D maps](doc/dense.md) · math: [Dense matching notes](doc/dense_matching.md)

- GPU (OpenCL) PatchMatch — [`compute_depthmaps`](doc/using.md#dense-reconstruction-dense_clustering-compute_depthmaps-fuse_depthmaps-dense_merging); config: [Depth estimation](doc/configuration.md#depth-estimation-patchmatch-opencl)
- GPU (OpenCL) SVO TSDF depthmap fusion — [`fuse_depthmaps`](doc/dense.md); config: [Fusion](doc/configuration.md#fusion)
- TSDF photometric refinement — config: [`depthmap_fusion_svo_refine_enabled`](doc/configuration.md#fusion)
- Dual Contouring (Surface Nets) surface extraction for sharp and well defined point-clouds and meshes 
- LAS, LAZ, PLY export for point cloud — config: [`dense_pointcloud_export_las`, `dense_pointcloud_export_laz`](doc/configuration.md#dense-disk-reclamation-and-export)
- PLY export for the mesh (`mesh.ply`) — config: [`depthmap_fusion_mesh_enabled`](doc/configuration.md#fusion)
- Streaming octree (Potree-style) tiles for the web/point-cloud viewer (`point_cloud/`) — config: [Octree tiling](doc/configuration.md#octree-tiling)

### 🧇 Ortho/DSM

![DSM and Ortho Extraction](doc/images/dsm_ortho.png)

> See [2D maps: DSM and orthophoto](doc/dense.md#2d-maps-dsm-and-orthophoto) and [Georeferencing & GIS outputs](doc/georeferencing.md)

- Direct, TSDF-based DSM and orthophoto rendering — config: [DSM and orthophoto](doc/configuration.md#dsm-and-orthophoto) (`dsm_enabled`, hole filling, coherence-enhancing shock filter, robust multi-view color baking)
- Accurate geo-referencing to output coordinate system (projection 3rd-degree polynomial, TPS fallback) — output CRS derived automatically; see [the output coordinate system](doc/georeferencing.md#the-output-coordinate-system) and `dense_merging --georeferenced`
- GeoTIFF export — `dsm.tif` / `ortho.tif`; see the [output reference](doc/dense.md#output-reference)

### 🩺 Quality Report

[Example Report](doc/images/report.pdf)

> See [Statistics & quality report](doc/quality_report.md) and [reporting JSON](doc/reporting.md)

- SfM quality metrics tables — [reconstruction details](doc/quality_report.md#reconstruction-details)
- Overlap map
- DSM/Ortho map
- Global GPS and GCP errors table — [GPS/GCP errors details](doc/quality_report.md#gpsgcp-errors-details)
- Detailed GCP errors table (control points vs. checkpoints)
- Localized report : metric/imperial units and en/fr/es/de/it — config: [`report_unit_system`, `report_language`](doc/configuration.md#report-localization)
- PDF export via [`compute_statistics`](doc/using.md#compute_statistics) + [`export_report`](doc/using.md#export_report)

### 🥽 Visualisation
![Rerun Export](doc/images/rerun.png)
- Viewer-based visualiser (SfM data only) — see [viewer/README.md](viewer/README.md)
- Web/point-cloud viewer via streaming octree tiles (`point_cloud/`) — config: [Octree tiling](doc/configuration.md#octree-tiling)
- Rerun export (SfM) with GPS/GCP data, and quality report. — [`export_rerun`](doc/using.md#export_rerun)

### ⏱️ Benchmarking

> Guide: [Benchmarking](doc/benchmark.md)

- Reproducible per-commit benchmarks — each run builds a chosen commit in an isolated git worktree + dedicated conda env, leaving your checkout untouched
- Runs the pipeline on a configurable set of datasets, SfM-only or with the full dense stages (`--dense`)
- HTML A/B report comparing quality (reconstruction, reprojection, tracks, features, GPS/GCP errors) and per-step timings against a reference commit
- Crash-safe and resumable — resume an interrupted run, re-run from a given step, bootstrap early steps from a previous run, or regenerate the report only

## 🩹 Bugfixes

## 🫣 Known Issues
- Mesh support is experimental and still very sub-optimal — the Surface Nets mesh (config: [`depthmap_fusion_mesh_enabled`](doc/configuration.md#fusion)) is on by default but can leave holes in empty regions
- TSDF photometric refinement doesn't bring substantial improvement (config: [`depthmap_fusion_svo_refine_enabled`](doc/configuration.md#fusion), `false` by default)
- Orthophoto and DSM still suffers from different exposure compensation and specular surfaces (metals, asphalt).
- The split-merge is the old one and not out-of-core, neither is the dense/ortho/DSM

🛸 There is more on the table :
 - Exposure compensation 
 - Proper mesh support
 - Revamped split/merge for massive datasets (fully out-of-core)
 - Preview mode (10-20 images/sec. target)
 - Gaussian Splats !
