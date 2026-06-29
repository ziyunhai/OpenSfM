# Ground Control Points

Ground control points (GCPs) are landmarks with known geospatial positions, marked in one or more images. They are the most accurate way to georeference a reconstruction and to set its [output coordinate system](georeferencing.md#the-output-coordinate-system). GCPs complement — or replace — the GPS positions read from image EXIF.

OpenSfM uses GCPs in two steps, **both of which require a point to be observed in at least two reconstructed images** (the point's position is triangulated from its image rays; a point seen in fewer than two reconstructed shots is ignored):

- **Alignment**: globally rotates, translates and scales the reconstruction so the triangulated GCPs match their survey positions.
- **Bundle adjustment**: adds each GCP as a position prior plus per-observation reprojection constraints, refining the reconstruction.

Each point also carries a **role**:

- `gcp` (default) — used in both alignment and bundle adjustment.
- `checkpoint` — *excluded* from optimization and used only to measure accuracy; checkpoint residuals are reported separately in the [quality report](quality_report.md#gpsgcp-errors-details).

Key config parameters: `bundle_use_gcp`, `gcp_horizontal_sd`, `gcp_vertical_sd`, `gcp_global_weight`, `gcp_annealing_steps`, `gcp_reprojection_error_threshold`. See the [configuration reference](configuration.md#bundle-adjustment). Per-point standard deviations (see below) override `gcp_horizontal_sd` / `gcp_vertical_sd` for that point.

GCPs can be supplied in two interchangeable formats — `ground_control_points.json` or `gcp_list.txt` — placed at the dataset root. Use [`convert_gcp`](using.md#convert_gcp) to convert between them (it backs up the target file as `.bak` first).

## JSON Format

Add a file named `ground_control_points.json` at the dataset root:

```json
{
  "crs": "WGS84",
  "points": [
    {
      "id": "gcp_id",
      "role": "gcp",
      "position": {
        "latitude": 52.519,
        "longitude": 13.400,
        "altitude": 14.946,
        "latitude_std": 0.02,
        "longitude_std": 0.02,
        "altitude_std": 0.05
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

Fields:

- **`crs`** *(optional, default `WGS84`)* — coordinate reference system of the point positions. Accepts `WGS84`, `EPSG:xxxx`, a compound EPSG (e.g. `EPSG:2056+5773` for an explicit vertical datum), or a raw `+proj=…` string. A *projected* `crs` also sets the reconstruction's [output coordinate system](georeferencing.md#the-output-coordinate-system).
- **`id`** — identifier of the point; observations sharing an `id` belong to the same physical landmark.
- **`role`** *(optional, default `gcp`)* — `gcp` or `checkpoint` (see above).
- **`position`** — either geographic `latitude`/`longitude` (degrees) **or** projected `easting`/`northing` (in `crs` units, transformed internally to WGS84). `altitude` is optional (the point is treated as horizontal-only when omitted). The optional `latitude_std` / `longitude_std` / `altitude_std` are positional standard deviations in **metres** — all three must be present to take effect, and they override the global `gcp_horizontal_sd` / `gcp_vertical_sd` for this point.
- **`observations`** — list of `{shot_id, projection}`. `projection` is in [normalized image coordinates](geometry.md#normalized-image-coordinates).

> A `coordinates` array (topocentric `[x, y, z]`) may also appear on a point. It is written by the pipeline and is not needed in hand-authored files.

## TXT Format

Add a file named `gcp_list.txt` at the dataset root:

- **First line**: the CRS / projection string.
- **Following lines**: one observation per line —

  ```
  <geo_x> <geo_y> <geo_z> <im_x> <im_y> <image_name> [<gcp_id>]
  ```

  - `<im_x> <im_y>` are **pixel** coordinates (not normalized).
  - `<geo_z>` may be `NaN` when the altitude is unknown.
  - `<gcp_id>` (optional) names the point: lines sharing the same name are grouped into one GCP. When omitted, observations are grouped by identical 3-D position and the point is auto-named `unnamed-N`.

  `#` comments and blank lines are ignored. The TXT format carries neither roles nor per-point standard deviations — use the JSON format for those.

**Supported CRS strings** (same set for the JSON `crs` and the TXT header line):

- `WGS84`: `geo_x = longitude`, `geo_y = latitude`, `geo_z = altitude`
- `WGS84 UTM 32N` (or any UTM zone): `geo_x = easting`, `geo_y = northing`, `geo_z = altitude`
- Any valid [proj4](http://proj4.org/) string, e.g. `+proj=utm +zone=32 +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs`

**Example `gcp_list.txt`** (two points, two observations each):

```
WGS84
13.400740745 52.519134104 12.0792090446 2335.0 1416.7 01.jpg corner_A
13.400740745 52.519134104 12.0792090446 2639.1 938.0  02.jpg corner_A
13.400502446 52.519251158 16.7021233002 766.0  1133.1 01.jpg corner_B
13.400502446 52.519251158 16.7021233002 1090.2 1499.0 02.jpg corner_B
```
