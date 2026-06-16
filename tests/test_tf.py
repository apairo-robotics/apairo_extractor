"""TF handling: /tf and /tf_static are demuxed into per-edge pose channels."""
from __future__ import annotations

import numpy as np
import pytest

from apairo_extractor.bag import read_bag_info
from apairo_extractor.extract import extract_bag

from conftest import TF_COUNT, TF_STATIC_TOPIC, TF_TOPIC, write_tf_bag


def _extract_tf(tmp_path, fmt, *, static=False):
    name = ("tfs" if static else "tf") + (".bag" if fmt == "ros1" else "")
    bag = write_tf_bag(tmp_path / name, fmt, static=static)
    topic = TF_STATIC_TOPIC if static else TF_TOPIC
    return extract_bag(read_bag_info(bag), [topic], tmp_path / "out")


def test_tf_demuxed_into_per_edge_channels(tmp_path, fmt):
    from apairo.core.config import read_config

    seq_dir, skipped = _extract_tf(tmp_path, fmt)
    assert TF_TOPIC not in skipped

    channels = read_config(seq_dir)["channels"]
    assert {"odom__base_link", "base_link__lidar"} <= set(channels)

    edge = channels["odom__base_link"]
    assert edge["loader"] == "npy"
    assert edge["transform"] == {
        "parent": "odom", "child": "base_link", "format": "t_xyz_q_xyzw",
    }

    arr = np.load(seq_dir / "odom__base_link" / "odom__base_link.npy")
    assert arr.shape == (TF_COUNT, 7)                       # one pose per /tf message
    np.testing.assert_allclose(arr[0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    ts = np.loadtxt(seq_dir / "odom__base_link" / "timestamps.txt")
    assert ts.shape == (TF_COUNT,)


def test_tf_static_marked_and_single_row(tmp_path, fmt):
    from apairo.core.config import read_config

    seq_dir, _ = _extract_tf(tmp_path, fmt, static=True)
    channels = read_config(seq_dir)["channels"]
    assert channels["base_link__lidar"]["transform"]["static"] is True
    arr = np.load(seq_dir / "base_link__lidar" / "base_link__lidar.npy")
    assert arr.shape == (1, 7)                              # latched -> single sample


def test_tf_output_loads_with_rawdataset(tmp_path, fmt):
    apairo = pytest.importorskip("apairo")
    if not hasattr(apairo, "RawDataset"):
        pytest.skip("installed apairo has no RawDataset")

    seq_dir, _ = _extract_tf(tmp_path, fmt)
    ds = apairo.RawDataset(seq_dir)
    assert {"odom__base_link", "base_link__lidar"} <= set(ds.available)
