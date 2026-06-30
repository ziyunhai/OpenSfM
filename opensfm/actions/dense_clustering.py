# pyre-strict
import os
from timeit import default_timer as timer

from opensfm import dataset, dense, io
from opensfm.dataset import DataSet


def run_dataset(data: DataSet, subfolder: str) -> None:
    """Build dense clusters, neighbours and depth ranges (dense stage 1).

    Produces the on-disk artefacts (clusters, super-points, best/all
    neighbours, depth ranges) consumed by ``compute_depthmaps`` and
    ``fuse_depthmaps``.

    Args:
        subfolder: dataset's subfolder where to load and store data
    """

    start = timer()
    udata_path = os.path.join(data.data_path, subfolder)
    udataset = dataset.UndistortedDataSet(data, udata_path, io_handler=data.io_handler)
    reconstructions = udataset.load_undistorted_reconstruction()
    tracks_manager = udataset.load_undistorted_tracks_manager()
    dense.run_clustering(udataset, tracks_manager, reconstructions[0])
    data.save_report(
        io.json_dumps({"wall_time": timer() - start}), "dense_clustering.json"
    )
