![OpenSfM](doc/images/logo_small.png)


OpenSfM
=======
[![Conda](https://github.com/OpenSfM/OpenSfM/actions/workflows/conda.yml/badge.svg)](https://github.com/OpenSfM/OpenSfM/actions/workflows/conda.yml) [![Docker Ubuntu 20.04](https://github.com/OpenSfM/OpenSfM/actions/workflows/docker_ubuntu20.yml/badge.svg)](https://github.com/OpenSfM/OpenSfM/actions/workflows/docker_ubuntu20.yml) [![Docker Ubuntu 24.04](https://github.com/OpenSfM/OpenSfM/actions/workflows/docker_ubuntu24.yml/badge.svg)](https://github.com/OpenSfM/OpenSfM/actions/workflows/docker_ubuntu24.yml) [![Coverage](./badges/coverage.svg)](https://github.com/OpenSfM/OpenSfM/actions/workflows/coverage.yml)

## Intro
This repository is an attempt at continuing the original [OpenSfM](https://github.com/mapillary/opensfm) project, which not longer in active development. We were maintainers and contributors of the original OpenSfM, and we will do our best to keep it alive and serve the community and our users ([OpenDronemap](https://www.opendronemap.org/), [WebODM](https://webodm.org/) and many others)

## Overview
OpenSfM is an open-source Structure from Motion (SfM) library written in Python with performance-critical code in C++. It reconstructs camera poses and 3D point clouds from unordered image collections, but also produces dense point clouds, or 2D maps (DSM, Orthophotos)

**Core pipeline**

Feature detection (SIFT, HAHOG, AKAZE, SURF, ORB), pairwise matching (OpenCL) with geometric verification, track building, incremental and direct aerotriangulation reconstruction with robust bundle adjustment ([Ceres](http://ceres-solver.org/)-based), and GPS/GCP-constrained alignment and coordinate systems shifts compensation.

![SfM Reconstruction](doc/images/viewer.png)

**Camera models**

Perspective, brown, fisheye (OpenCV model, and customs 62 and 624 parameters), spherical/equirectangular, and dual. See [geometry](doc/geometry.md).

**Dense reconstruction**

Multi-view depth estimation via GPU PatchMatch (OpenCL), SVO-vased TSDF depth fusion and refinement, DSM and Orthophoto generation.

![Dense Reconstruction](doc/images/dense.png)

**Scalability**

Large scene support via submodel splitting/merging, rig constraints for multi-camera setups, and configurable multi-processing.

**Exports**

COLMAP, Bundler, OpenMVS, PMVS, VisualSFM, PLY, LAS/LAZ, GeoJSON, GeoTIFF. In addition, a detailed quality report can be created and exported as a PDF file (see [reporting](doc/reporting.md) and [example report](doc/images/report.pdf))

![DSM and Ortho Extraction](doc/images/dsm_ortho.png)

**Visualisation**

A built-in JavaScript viewer allows interactive 3D preview and pipeline debugging. In addition, the scene can also be inspected as a [Rerun](https://rerun.io/) export. 

![Rerun Export](doc/images/rerun.png)

**Compatibility** —
Runs on Linux, macOS, and Windows. See [quickstart](doc/quickstart.md) to get started.

**Credits** —
OpenSfM was created by Pau Gargallo and bootstrapped by Mapillarians : checkout this [blog post with more demos](http://blog.mapillary.com/update/2014/12/15/sfm-preview.html)


## Getting Started

Install using conda lock files (see [building instructions](doc/building.md)):

**Linux:**
```bash
conda create --name opensfm --file conda-linux-64.lock --yes
conda activate opensfm && pip install -e .
```

**macOS (Apple Silicon):**
```bash
conda create --name opensfm --file conda-osx-arm64.lock --yes
conda activate opensfm && pip install -e .
```

Then reconstruct a dataset:
```bash
conda activate opensfm
./bin/opensfm_run_all path/to/dataset   # Linux/macOS
bin\opensfm_run_all.bat path\to\dataset  # Windows
```
## Documentation

**Getting Started**
* [Quickstart](doc/quickstart.md)
* [Building & Installation](doc/building.md)
* [Pipeline Commands](doc/using.md)

**User Guide**
* [Dataset structure](doc/dataset.md)
* [Configuration reference](doc/configuration.md)
* [Ground control points](doc/ground_control_points.md)
* [Rig models](doc/rig.md)
* [Large datasets](doc/large_datasets.md)
* [Dense reconstruction & 2D maps](doc/dense.md)
* [Georeferencing & GIS outputs](doc/georeferencing.md)
* [Quality report](doc/quality_report.md)

**Reference**
* [Camera models & coordinate systems](doc/geometry.md)
* [Reconstruction algorithm](doc/reconstruction.md)
* [Sensor / calibration database](doc/sensor_database.md)
* [Reporting](doc/reporting.md)

**Mathematical Notes**
* [Dense matching](doc/dense_matching.md)
* [Reconstruction merging](doc/merging_notes.md)

## License
OpenSfM is BSD-style licensed, as found in the LICENSE file.

Example data in the README is under [Creative Commons CC-BY 4.0 License](https://creativecommons.org/licenses/by/4.0/) by Wingtra AG, 8045 Zürich, Switzerland.