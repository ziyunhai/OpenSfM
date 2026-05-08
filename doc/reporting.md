# Reporting

OpenSfM commands write reports on the work done. Reports are stored in the `reports/` folder in JSON format.

## Feature Detection

Stored in `features.json`:

```json
{
    "wall_time": "<total time computing features>",
    "image_reports": [
        {
            "wall_time": "<feature extraction time>",
            "image": "<image name>",
            "num_features": "<number of features>"
        }
    ]
}
```

## Matching

Stored in `matches.json`:

```json
{
    "wall_time": "<total time computing matches>",
    "pairs": "<list of candidate image pairs>",
    "num_pairs": "<number of candidate image pairs>",
    "num_pairs_distance": "<number of pairs selected based on distance>",
    "num_pairs_time": "<number of pairs selected based on time>",
    "num_pairs_order": "<number of pairs selected based on order>"
}
```

## Create Tracks

Stored in `tracks.json`:

```json
{
    "wall_time": "<total time computing tracks>",
    "wall_times": {
        "load_features": "<time loading features>",
        "load_matches": "<time loading matches>",
        "compute_tracks": "<time computing tracks>"
    },
    "num_images": "<number of images with tracks>",
    "num_tracks": "<number of tracks>",
    "view_graph": "<number of image tracks for each image pair>"
}
```

## Reconstruction

Stored in `reconstruction.json`:

```json
{
    "wall_times": {
        "compute_reconstructions": "<time computing the reconstruction>",
        "compute_image_pairs": "<time computing the candidate initial pairs>"
    },
    "num_candidate_image_pairs": "<number of candidate image pairs for initialization>",
    "reconstructions": [
        {
            "bootstrap": {
                "memory_usage": "<memory usage at the end of the process>",
                "image_pair": "<initial image pair>",
                "common_tracks": "<number of common tracks of the image pair>",
                "two_view_reconstruction": {
                    "5_point_inliers": "<number of inliers for the 5-point algorithm>",
                    "plane_based_inliers": "<number of inliers for plane based initialization>",
                    "method": "<'5_point' or 'plane_based'>"
                },
                "triangulated_points": "<number of triangulated points>",
                "decision": "<'Success' or reason for failure>"
            },
            "grow": {
                "steps": [
                    {
                        "image": "<image name>",
                        "resection": {
                            "num_inliers": "<number of inliers>",
                            "num_common_points": "<number of reconstructed points visible on the new image>"
                        },
                        "triangulated_points": "<number of newly triangulated points>",
                        "memory_usage": "<memory usage after adding the image>",
                        "bundle": {
                            "wall_times": {
                                "setup": "<time setting up bundle>",
                                "run": "<time running bundle>",
                                "teardown": "<time updating values after bundle>"
                            },
                            "brief_report": "<Ceres brief report>"
                        }
                    }
                ]
            }
        }
    ],
    "not_reconstructed_images": "<images that could not be reconstructed>"
}
```
