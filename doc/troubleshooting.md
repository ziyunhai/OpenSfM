# Troubleshooting

Quick fixes for the most common issues, focused on the GPU dense stages where hardware limits bite first. All settings below go in your dataset's `config.yaml`; see the [configuration reference](configuration.md) for full descriptions.

## Hardware requirements

The dense pipeline (depthmaps, fusion, DSM/ortho) needs an **OpenCL-capable GPU**. The sparse SfM stages run on CPU, except for the matching, which is GPU OpenCL by default, and can be deactivated by setting `matcher_type` to `FLANN`,

|               | RAM   | GPU (OpenCL) |
| ------------- | ----- | ------------ |
| **Minimum**   | 16 GB | 8 GB VRAM    |
| **Preferred** | 32 GB | 16 GB VRAM   |

## GPU / OpenCL selection

- **Depthmap estimation** (`compute_depthmaps`) uses **all detected non-Intel OpenCL GPUs in parallel** — add a second card and it is used automatically.
- **Fusion** (`fuse_depthmaps`) runs on the **first GPU (device 0)** only.
- **Intel GPUs are skipped by default** (they are slow/buggy for this workload). Set [`opencl_ignore_intel_gpu_device`](configuration.md#depth-estimation-patchmatch-opencl) to `false` to allow them.
- There is no option to pin a specific device index.
- `No OpenCL devices found — cannot compute depthmaps` means no usable GPU was detected; the dense stages cannot run.

## Dense: running out of memory (RAM or VRAM)

The single biggest driver of memory use and point count is the **processed image resolution**. In order of impact:

1. **Lower [`depthmap_max_image_size`](configuration.md#depth-estimation-patchmatch-opencl)** (default `3200`). This is the main lever — fewer points, less RAM and VRAM.
2. **Halve [`depthmap_fusion_svo_max_voxels`](configuration.md#fusion)** (default `80000000`) **and [`depthmap_max_cluster_views`](configuration.md#depth-estimation-patchmatch-opencl)** (default `48`) to cap the fusion/cleaning peak memory.
3. **Set [`depthmap_fusion_svo_voxel_level`](configuration.md#fusion) to `half` or `quarter`** — coarser voxels mean faster processing and lower memory, at the cost of detail.

## Dense: point-cloud density

[`depthmap_fusion_svo_decimate_flat`](configuration.md#fusion) trims points on flat surfaces while preserving sharp edges:

| Value | Effect                                                         |
| ----- | -------------------------------------------------------------- |
| `1.0` | No decimation — densest cloud                                  |
| `2.0` | Balanced (default)                                             |
| `4.0` | More adaptive decimation — sparser flats, detail kept at edges |

## Dense: cropping the output

By default the dense outputs (point cloud + DSM/ortho) are cropped to the PCA-trimmed convex hull of the SfM ground points, trimming the sparse fringe beyond the surveyed area.

- **Crop more aggressively:** raise [`dense_crop_percentile`](configuration.md#dsm-and-orthophoto) to `5.0` or `10.0`.
- **Disable cropping entirely:** set [`dense_crop_to_sfm_hull`](configuration.md#dsm-and-orthophoto) to `false`.

## Dense: skipping LAS/LAZ export

LAS and LAZ export are on by default (alongside `fused.ply`). To skip them, set both [`dense_pointcloud_export_las`](configuration.md#dense-disk-reclamation-and-export) and [`dense_pointcloud_export_laz`](configuration.md#dense-disk-reclamation-and-export) to `false`.

## See also

- [Configuration reference](configuration.md) — every parameter and its default.
- [Dense reconstruction & 2D maps](dense.md) — the dense pipeline and its outputs.
- [Workflow presets](using.md#workflow-presets-configs) — sensible starting configs per capture type.
