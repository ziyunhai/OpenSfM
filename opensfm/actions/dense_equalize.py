# pyre-strict
import os
from timeit import default_timer as timer

from opensfm import dataset, dense, io
from opensfm.dataset import DataSet


def run_dataset(data: DataSet, subfolder: str) -> None:
    """Estimate per-image radiometric corrections from the SfM tracks.

    Computes a per-image, per-channel gain (exposure + white balance) and a
    radial vignetting falloff from the track correspondences (the same 3-D point
    seen by several images should have the same colour) and writes
    ``equalization.json`` in the undistorted subfolder, for the colour bake to
    divide out.  Runs on the undistorted reconstruction/tracks so the radii and
    colours match the images the bake samples.

    Args:
        subfolder: dataset's subfolder where to load and store data
    """
    start = timer()
    udata_path = os.path.join(data.data_path, subfolder)
    udataset = dataset.UndistortedDataSet(
        data, udata_path, io_handler=data.io_handler
    )
    reconstructions = udataset.load_undistorted_reconstruction()
    tracks_manager = udataset.load_undistorted_tracks_manager()
    dense.run_equalize(udataset, tracks_manager, reconstructions[0])
    data.save_report(
        io.json_dumps({"wall_time": timer() - start}), "dense_equalize.json"
    )
