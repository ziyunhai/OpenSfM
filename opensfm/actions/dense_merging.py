# pyre-strict
import os

from opensfm import dataset, dense
from opensfm.dataset import DataSet


def run_dataset(data: DataSet, subfolder: str, georeferenced: bool = False) -> None:
    """Merge per-cluster fused PLYs and DSM/ortho tiles, then export (dense stage 4).

    Requires ``fuse_depthmaps`` to have run first.  Produces the final
    ``fused.ply`` (and ``dsm.tif`` / ``ortho.tif`` when DSM is enabled) plus the
    optional LAS/LAZ and octree-tile exports.

    Args:
        subfolder: dataset's subfolder where to load and store data
        georeferenced: when True, write the LAS/LAZ and DSM/ortho products in the
            output coordinate system (projected GCP CRS if any, else UTM from the
            reference LLA); otherwise keep the topocentric-based coordinates.
    """

    udata_path = os.path.join(data.data_path, subfolder)
    udataset = dataset.UndistortedDataSet(data, udata_path, io_handler=data.io_handler)
    reconstructions = udataset.load_undistorted_reconstruction()
    output_crs = data.output_coordinate_system() if georeferenced else None
    dense.run_merge(udataset, reconstructions[0], output_crs=output_crs)
