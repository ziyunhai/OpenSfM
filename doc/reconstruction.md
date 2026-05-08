# Incremental Reconstruction Algorithm

OpenSfM implements an incremental structure-from-motion algorithm. It starts from a single image pair and iteratively adds other images one at a time.

The algorithm is in `opensfm/reconstruction.py` and the main entry point is `incremental_reconstruction()`.

The algorithm has three main steps:

1. Find good initial pairs
2. Bootstrap the reconstruction with two images
3. Grow the reconstruction by adding images one at a time

If images remain unreconstructed after step 3, steps 2 and 3 are repeated to generate additional reconstructions.

## 1. Finding Good Initial Pairs

To initialize from two images, there must be enough parallax between them (the camera must have moved significantly relative to the scene distance).

To find sufficient parallax, a rotation-only camera model is fit to each pair. A pair is accepted only if a significant portion of correspondences cannot be explained by pure rotation — specifically, if the outlier fraction exceeds 30%.

Accepted pairs are sorted by number of outliers (descending). Implemented in `compute_image_pairs()`.

## 2. Bootstrapping the Reconstruction

The first accepted image pair is used to initialize. If initialization fails, the next pair is tried.

Two algorithms are available depending on scene geometry:
- **Flat scene**: plane-based initialization
- **Non-flat scene**: five-point algorithm

Since the geometry is unknown a priori, both are computed and the one producing more points is retained (`two_view_reconstruction_general()`).

If the pair produces enough inliers, a reconstruction is initialized with the computed poses, the matches are triangulated, and bundle adjustment is performed.

## 3. Growing the Reconstruction

Starting from the initial two-image reconstruction, images are added one by one — starting with the image that sees the most already-reconstructed points.

For each new image:

1. **Resectioning** (`resect()`): finds the camera position that makes reconstructed 3D points project to their corresponding locations in the new image.
2. If resectioning succeeds, the image is added and its features that are visible in other reconstructed images are triangulated.
3. **Bundle adjustment** and **re-triangulation** are run as needed. The parameters `bundle_interval`, `bundle_new_points_ratio`, `retriangulation`, and `retriangulation_ratio` control when these are triggered.

Finally, if GPS positions or Ground Control Points are available, the reconstruction is rigidly aligned to them.
