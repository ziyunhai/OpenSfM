# pyre-strict
import os
from timeit import default_timer as timer

from opensfm import dataset, dense, io
from opensfm.dataset import DataSet


def run_dataset(data: DataSet, subfolder: str) -> None:
    """Fuse cleaned depthmaps per cluster into batch point clouds (dense stage 3).

    Requires ``dense_clustering`` and ``compute_depthmaps`` to have run first.
    Produces ``fused_batch_*.ply`` (and, when DSM is enabled, per-cluster
    DSM/ortho tiles) consumed by ``dense_merging``.

    Args:
        subfolder: dataset's subfolder where to load and store data
    """

    start = timer()
    udata_path = os.path.join(data.data_path, subfolder)
    udataset = dataset.UndistortedDataSet(data, udata_path, io_handler=data.io_handler)
    reconstructions = udataset.load_undistorted_reconstruction()
    dense.run_fusion(udataset, reconstructions[0])
    data.save_report(
        io.json_dumps({"wall_time": timer() - start}), "dense_fusion.json"
    )
