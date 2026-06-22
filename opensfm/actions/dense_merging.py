# pyre-strict
import os

from opensfm import dataset, dense
from opensfm.dataset import DataSet


def run_dataset(data: DataSet, subfolder: str) -> None:
    """Merge per-cluster fused PLYs and DSM/ortho tiles, then export (dense stage 4).

    Requires ``fuse_depthmaps`` to have run first.  Produces the final
    ``fused.ply`` (and ``dsm.tif`` / ``ortho.tif`` when DSM is enabled) plus the
    optional LAS/LAZ and octree-tile exports.

    Args:
        subfolder: dataset's subfolder where to load and store data
    """

    udata_path = os.path.join(data.data_path, subfolder)
    udataset = dataset.UndistortedDataSet(data, udata_path, io_handler=data.io_handler)
    reconstructions = udataset.load_undistorted_reconstruction()
    dense.run_merge(udataset, reconstructions[0])
