# Calibration Database

## Overview

Structure-from-motion needs accurate estimates of the imaging sensor geometry: lens type, focal length, distortion, and principal point. See [geometry.md](geometry.md) for the full list of camera calibration parameters.

During reconstruction, OpenSfM adjusts calibration values to best explain the observed geometry. However, starting with a good initial guess produces better and more reliable results. By default, OpenSfM reads focal length from image EXIFs. When EXIF values are missing, incorrect, or insufficient, the calibration databases fill the gap.

Calibration databases are stored under `opensfm/data/`:
- `sensor_data_detailed.json`
- `sensor_data.json`
- `camera_calibration.yaml`

## `sensor_data_detailed.json`

Contains physical sensor width and height (in millimetres) for a given `model make` sensor. When only focal length is available from EXIF, the full sensor geometry can be recovered using this physical size.

## `sensor_data.json`

Contains a multiplicative factor for a given `model make` sensor. Applied to the EXIF focal length, this factor gives the 35mm equivalent focal length. Since the 35mm equivalent dimensions (24×32 mm) are known, the full sensor geometry can be recovered.

## `camera_calibration.yaml`

Contains full camera calibration definitions in OpenSfM format, keyed by `make`. Within each make, calibrations are further refined:

- **`ALL`**: the calibration is valid for all `make model` combinations, regardless of `model`
- **`MODEL`**: calibrations are per actual `model`
- **`FOCAL`**: calibrations are per focal length read from EXIF
