# Dataset Structure

```
project/
├── config.yaml
├── images/
│   └── image_filename
├── masks/
│   └── image_filename.png
├── gcp_list.txt
├── exif/
├── camera_models.json
├── features/
├── matches/
├── tracks.csv
├── reconstruction.json
├── reconstruction.meshed.json
├── undistorted/
│   ├── images/
│   │   └── image_filename
│   ├── masks/
│   │   └── image_filename.png
│   ├── tracks.csv
│   ├── reconstruction.json
│   └── depthmaps/
│       └── merged.ply
└── stats/
    ├── stats.json
    ├── report.pdf
    ├── topview.png
    ├── matchgraph.png
    ├── heatmap_XXX.png
    └── residuals_XXX.png
```

> Note: Previous versions of OpenSfM used a different folder structure where undistorted data was not grouped into a single folder. Use `bin/migrate_undistort.sh` to port old datasets to the new structure.

## Reconstruction File Format

The main output of OpenSfM is `reconstruction.json`, containing estimated camera parameters, camera positions, and a sparse set of 3D points.

```
reconstruction.json: [RECONSTRUCTION, ...]

RECONSTRUCTION: {
    "cameras": {
        CAMERA_ID: CAMERA,
        ...
    },
    "shots": {
        SHOT_ID: SHOT,
        ...
    },
    "points": {
        POINT_ID: POINT,
        ...
    }
}

CAMERA: {
    "projection_type": "perspective",  # perspective, brown, fisheye or equirectangular
    "width": NUMBER,                   # Image width in pixels
    "height": NUMBER,                  # Image height in pixels

    # Perspective camera parameters:
    "focal": NUMBER,                   # Estimated focal length
    "k1": NUMBER,                      # Estimated distortion coefficient
    "k2": NUMBER                       # Estimated distortion coefficient
}

SHOT: {
    "camera": CAMERA_ID,
    "rotation": [X, Y, Z],      # Estimated rotation as an angle-axis vector
    "translation": [X, Y, Z],   # Estimated translation
    "gps_position": [X, Y, Z],  # GPS coordinates in the reconstruction reference frame
    "gps_dop": METERS,          # GPS accuracy in meters
    "orientation": NUMBER,      # EXIF orientation tag (1, 3, 6, or 8)
    "capture_time": SECONDS     # Capture time as a UNIX timestamp
}

POINT: {
    "coordinates": [X, Y, Z],   # Estimated position of the point
    "color": [R, G, B]          # Color of the point
}
```
