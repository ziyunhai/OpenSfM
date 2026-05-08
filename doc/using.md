# Using OpenSfM

## Quickstart

An example dataset is available at `data/berlin`. Reconstruct it by running:

```bash
bin/opensfm_run_all data/berlin
```

This runs the entire SfM pipeline and produces `data/berlin/reconstruction.meshed.json` as output.

### Running in Docker

First, build the OpenSfM Docker image as described in [building](building.md).

Start a Docker container, mounting the `data/` folder:

```bash
docker run -it -p 8080:8080 -v ${PWD}/data/:/data/ opensfm.ubuntu24 /bin/bash
```

Inside the container, run the reconstruction:

```bash
bin/opensfm_run_all /data/berlin/
```

When done, exit with Ctrl+d. The model will be available in the `data/` directory.

### Viewer

A desktop viewer is included. Launch it with:

```bash
./bin/opensfm-desktop
```

Open a reconstruction by passing the dataset path as an argument or using the file picker in the application.

![Berlin viewer](images/berlin_viewer.jpg)

### Dense Point Clouds

For a denser point cloud:

```bash
bin/opensfm undistort data/berlin
bin/opensfm compute_depthmaps data/berlin
```

This runs dense multiview stereo and produces a denser point cloud at `data/berlin/undistorted/depthmaps/merged.ply`. Visualize it with [MeshLab](http://www.meshlab.net/) or any viewer supporting [PLY](http://paulbourke.net/dataformats/ply/) files.

For the Berlin dataset you should get something like:

![Berlin point cloud](images/berlin_point_cloud.jpg)

To reconstruct your own images:

1. Put images in `data/DATASET_NAME/images/`
2. Copy `data/berlin/config.yaml` to `data/DATASET_NAME/config.yaml`


## Reconstruction Commands

The application `bin/opensfm` runs all pipeline commands. The first argument is the command, the second is the dataset path.

```
usage: opensfm [-h] command ...

positional arguments:
  command            Command to run
    extract_metadata   Extract metadata from images' EXIF tag
    detect_features    Compute features for all images
    match_features     Match features between image pairs
    create_tracks      Link matches pair-wise matches into tracks
    reconstruct        Compute the reconstruction
    bundle             Bundle a reconstruction
    mesh               Add delaunay meshes to the reconstruction
    undistort          Save radially undistorted images
    compute_depthmaps  Compute depthmap
    compute_statistics Compute statistics and save them in the stats folder
    export_ply         Export reconstruction to PLY format
    export_openmvs     Export reconstruction to openMVS format
    export_visualsfm   Export reconstruction to NVM_V3 format from VisualSfM
    export_pmvs        Export reconstruction to PMVS
    export_bundler     Export reconstruction to bundler format
    export_colmap      Export reconstruction to colmap format
    export_geocoords   Export reconstructions in geographic coordinates
    export_report      Export a nice report based on previously generated statistics
    create_submodels   Split the dataset into smaller submodels
    align_submodels    Align submodel reconstructions
```

### `extract_metadata`

Extracts EXIF metadata from images and stores them in the `exif/` folder and `camera_models.json`.

Data extracted per image:

- `width` and `height`: image size in pixels
- `gps` latitude, longitude, altitude, dop: GPS coordinates and Dilution Of Precision
- `capture_time`: capture time (UNIX timestamp)
- `camera orientation`: EXIF orientation tag (1, 3, 6, or 8)
- `projection_type`: determined from GPano metadata. Supported types:
  - `perspective`: perspective projection with 2nd and 4th order radial distortion
  - `radial`: same with principal point and aspect ratio
  - `simple_radial`: perspective with 2nd order radial, principal point, aspect ratio
  - `brown`: perspective with 2nd/4th/6th radial, two tangential, principal point, aspect ratio
  - `spherical` (or `equirectangular`): spherical projection for 360 images
  - `fisheye`: fisheye with 2nd and 4th order radial distortion
  - `fisheye_opencv`: fisheye identical to OpenCV's fisheye
  - `fisheye62`: fisheye with 1–6th order radial, 2 tangential, principal point, aspect ratio
  - `dual`: linear interpolation between fisheye and perspective models
- `focal_ratio`: EXIF focal length divided by sensor width
- `make` and `model`: camera make and model
- `camera`: camera ID string

Camera model parameters are chosen by this priority:
1. `camera_models_overrides.json` if the camera ID is present
2. Internal calibration database if the camera ID is present
3. Inferred from EXIF metadata

#### Providing Additional Metadata

When EXIF metadata is missing or incorrect, override it via `exif_overrides.json`:

```json
{
    "image_name.jpg": {
        "gps": {
            "latitude": 52.51891,
            "longitude": 13.40029,
            "altitude": 27.0,
            "dop": 5.0
        }
    }
}
```

Rerun `extract_metadata` after writing this file.

#### Providing Your Own Camera Parameters

Put custom camera parameters in `camera_models_overrides.json` in the project folder (same structure as `camera_models.json`).

Use `"all"` as camera ID to override all cameras:

```json
{
    "all": {
        "projection_type": "perspective",
        "width": 1920,
        "height": 1080,
        "focal": 0.9,
        "k1": 0.0,
        "k2": 0.0
    }
}
```

For fisheye:

```json
{
    "all": {
        "projection_type": "fisheye",
        "width": 1920,
        "height": 1080,
        "focal": 0.5,
        "k1": 0.0,
        "k2": 0.0
    }
}
```

For equirectangular 360:

```json
{
    "all": {
        "projection_type": "equirectangular",
        "width": 2000,
        "height": 1000
    }
}
```

### `detect_features`

Detects feature points in images and stores them in the `feature/` folder.

### `match_features`

Matches feature points between images and stores results in the `matches/` folder. Determines image pairs to run first, then runs matching for each pair.

Matching can be restricted (and sped up) by GPS distance, capture time, or file name order.

### `create_tracks`

Links pairwise matches into feature point tracks stored in `tracks.csv`. A track is a set of feature points from different images recognized as the same physical point.

Format of `tracks.csv`:

```
image_name  track_id  feature_index  normalized_x  normalized_y  size  R  G  B
02.jpg      1479      2594           0.0379803     -0.0481853    0.00155505  155  145  143
```

### `reconstruct`

Runs the incremental reconstruction process to find 3D positions of tracks (structure) and camera positions (motion). Output is stored in `reconstruction.json`.

### `mesh`

Computes a rough triangular mesh of the scene seen by each image, used for smooth motion simulation in the viewer. Output is `reconstruction.meshed.json`.

### `undistort`

Creates undistorted versions of the reconstruction, tracks, and images. Required before `compute_depthmaps`.

### `compute_depthmaps`

Computes a dense point cloud by computing and merging depth maps. Requires an undistorted reconstruction. Output is in the `depthmaps/` folder; the merged cloud is at `undistorted/depthmaps/merged.ply`.

### `compute_statistics`

Computes statistics stored in JSON at `stats/stats.json`, along with diagram images (`heatmap`, `matchgraph`, `topview`, `residuals`).

### `export_report`

Exports `stats/stats.json` and diagrams as a PDF report at `stats/report.pdf`.


## Configuration

SfM options can be tuned by editing `DATASET_PATH/config.yaml`. Any option present overrides the default values defined in `opensfm/config.py`.


## Ground Control Points

When EXIF data contains GPS location, OpenSfM uses it to georeference the reconstruction. Ground control points (GCPs) can be used for additional accuracy.

GCPs are landmarks visible in images with known geospatial positions (latitude, longitude, altitude). A single GCP can be observed in one or more images.

OpenSfM uses GCPs in two steps:
- **Alignment**: globally moves the reconstruction to align observed GCPs with their GPS positions. Requires at least two observations per GCP.
- **Bundle adjustment**: uses GCP observations as constraints to refine the reconstruction. No minimum number of observations required.

### JSON Format

Add a file named `ground_control_points.json` at the dataset root:

```json
{
  "points": [
    {
      "id": "gcp_id",
      "position": {
        "latitude": 52.519,
        "longitude": 13.400,
        "altitude": 14.946
      },
      "observations": [
        {
          "shot_id": "image.jpg",
          "projection": [0.1, -0.2]
        }
      ]
    }
  ]
}
```

Coordinates are in [WGS84](https://en.wikipedia.org/wiki/World_Geodetic_System). Altitude is optional. Projections are in [normalized image coordinates](geometry.md#normalized-image-coordinates).

### TXT Format

Add a file named `gcp_list.txt` at the dataset root:

- First line: projection name
- Following lines: one observation per line:

  ```
  <geo_x> <geo_y> <geo_z> <im_x> <im_y> <image_name>
  ```

  Where `<im_x> <im_y>` are pixel coordinates and `<geo_z>` can be `NaN` if altitude is unknown.

**Supported projections:**

- `WGS84`: `geo_x = longitude`, `geo_y = latitude`, `geo_z = altitude`
- `WGS84 UTM 32N` (or any UTM zone): `geo_x = E`, `geo_y = N`, `geo_z = altitude`
- Any valid [proj4](http://proj4.org/) string, e.g. `+proj=utm +zone=32 +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs`

**Example `gcp_list.txt`:**

```
WGS84
13.400740745 52.519134104 12.0792090446 2335.0 1416.7 01.jpg
13.400740745 52.519134104 12.0792090446 2639.1 938.0  02.jpg
13.400502446 52.519251158 16.7021233002 766.0  1133.1 01.jpg
```
