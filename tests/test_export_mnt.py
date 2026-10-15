"""MNT export: from the extracted sequence and straight from the bag, the
mission is written by the extractor alone (the MNT reader left apairo's core)."""
from __future__ import annotations

import numpy as np
import pytest

from apairo_extractor import MntExportConfig
from apairo_extractor.bag import read_bag_info
from apairo_extractor.runner import run_extraction

from conftest import CLOUD_POINTS, IMU_TOPIC, LIDAR_COUNT, LIDAR_TOPIC, write_tiny_bag

zarr = pytest.importorskip("zarr")


@pytest.mark.parametrize("keep_kitti", [True, False], ids=["from_sequence", "from_bag"])
def test_mnt_mission_written(tmp_path, fmt, keep_kitti):
    bag = read_bag_info(
        write_tiny_bag(tmp_path / ("seq.bag" if fmt == "ros1" else "seq"), fmt)
    )
    config = MntExportConfig(points_channel="lidar", output_dir=tmp_path / "mnt")
    results = run_extraction(
        [bag],
        [LIDAR_TOPIC, IMU_TOPIC],
        tmp_path / "kitti" if keep_kitti else None,
        mnt_config=config,
        workers=1,
    )
    assert [err for _, _, err in results] == [None]

    mission = tmp_path / "mnt" / bag.path.stem
    points = np.asarray(zarr.open(str(mission / "points.zarr"), mode="r"))
    assert points.shape == (LIDAR_COUNT, *CLOUD_POINTS.shape)
    np.testing.assert_array_equal(points[0], CLOUD_POINTS)
    assert (mission / "metadata.yaml").is_file()
