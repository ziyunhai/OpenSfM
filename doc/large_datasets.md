# Large Datasets

## Splitting a Large Dataset into Submodels

Large datasets can be slow to process. An option is to split them into smaller *submodels* that run faster (fewer images per bundle adjustment) and can be reconstructed in parallel.

Since submodel reconstructions are done independently, they will not be aligned with each other. Only GPS positions and ground control points determine alignment. When neighboring reconstructions share cameras or points, the alignment of those can be enforced.

## Creating Submodels

The `create_submodels` command splits a dataset based on GPS positions. `extract_metadata` must be run first. Feature extraction and matching can also be run before splitting so submodels can reuse computed data:

```bash
bin/opensfm extract_metadata path/to/dataset
bin/opensfm detect_features path/to/dataset
bin/opensfm match_features path/to/dataset
bin/opensfm create_submodels path/to/dataset
```

### Submodel Folder Structure

Submodels are created in the `submodels/` folder. Each submodel is a valid OpenSfM dataset. Images, EXIF metadata, features, and matches are shared via symbolic links.

```
project/
├── images/
├── image_list.txt
├── image_list_with_gps.csv
├── exif/
├── features/
├── matches/
└── submodels/
    ├── clusters_with_neighbors.geojson
    ├── clusters_with_neighbors.npz
    ├── clusters.npz
    ├── image_list_with_gps.tsv
    ├── submodel_0000/
    │   ├── image_list.txt
    │   ├── config.yaml
    │   ├── images/              # symlink to global
    │   ├── exif/                # symlink to global
    │   ├── features/            # symlink to global
    │   ├── matches/             # symlink to global
    │   ├── camera_models.json   # symlink to global
    │   └── reference_lla.json   # symlink to global
    ├── submodel_0001/
    └── ...
```

### Configuration Parameters

- **`submodel_size`**: Average number of images per submodel. K-means clustering is used with `k = num_images / submodel_size`.
- **`submodel_overlap`**: Radius of the overlapping region between submodels (in meters). Images closer to a neighboring cluster than this value are added to both.
- **`submodels_relpath`**: Relative path to the submodels directory.
- **`submodel_relpath_template`**: Template for the relative path to a submodel directory.
- **`submodel_images_relpath_template`**: Template for the relative path to a submodel images directory.

### Providing Image Groups Manually

If you already know how to split the dataset, provide `image_groups.txt` in the main dataset folder with one line per image:

```
01.jpg A
02.jpg A
03.jpg B
04.jpg B
05.jpg C
```

This creates 3 submodels. The `create_submodels` command will still add overlap images based on `submodels_overlap`.

## Running Reconstruction for Each Submodel

Each submodel is a valid OpenSfM dataset. Assuming features and matches are already computed:

```bash
bin/opensfm create_tracks path/to/dataset/submodels/submodel_XXXX
bin/opensfm reconstruct path/to/dataset/submodels/submodel_XXXX
```

These can be run in parallel since submodels are independent.

## Aligning Submodels

Once every submodel has a reconstruction, align them with:

```bash
bin/opensfm align_submodels path/to/dataset
```

This loads all reconstructions, finds cameras and points shared between them, and rigidly transforms each reconstruction to best align the common elements.
