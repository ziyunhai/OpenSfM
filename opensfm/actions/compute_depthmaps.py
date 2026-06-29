# pyre-strict
import os
from timeit import default_timer as timer

from opensfm import dataset, dense, io
from opensfm.dataset import DataSet


def run_dataset(data: DataSet, subfolder: str, interactive: bool) -> None:
    """Compute raw + clean depthmaps on a dataset that has SfM ran already.

    Requires the ``dense_clustering`` stage to have run first (clusters,
    neighbours and depth ranges are loaded from disk).

    Args:
        subfolder: dataset's subfolder where to store results
        interactive : display plot of computed depthmaps

    """

    start = timer()
    udata_path = os.path.join(data.data_path, subfolder)
    udataset = dataset.UndistortedDataSet(data, udata_path, io_handler=data.io_handler)
    udataset.config["interactive"] = interactive
    reconstructions = udataset.load_undistorted_reconstruction()
    tracks_manager = udataset.load_undistorted_tracks_manager()
    dense.run_depthmaps(udataset, tracks_manager, reconstructions[0])
    data.save_report(
        io.json_dumps({"wall_time": timer() - start}), "dense_depthmaps.json"
    )
