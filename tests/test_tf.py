"""TF handling: /tf and /tf_static are demuxed into per-edge pose channels.

Every edge is preserved verbatim, keyed by its source topic
(``<source>__<parent>__<child>``) -- two sources of the same edge coexist as
distinct channels, so nothing is dropped or merged.
"""
from __future__ import annotations

import numpy as np
import pytest

from apairo_extractor.bag import read_bag_info
from apairo_extractor.extract import extract_bag

from conftest import (
    TF_COUNT, TF_STATIC_TOPIC, TF_TOPIC, write_tf_bag, write_tf_bag_with_static,
)


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
    assert {"tf__odom__base_link", "tf__base_link__lidar"} <= set(channels)

    edge = channels["tf__odom__base_link"]
    assert edge["loader"] == "npy"
    assert edge["transform"] == {
        "parent": "odom", "child": "base_link", "source": TF_TOPIC,
        "format": "t_xyz_q_xyzw",
    }

    arr = np.load(seq_dir / "tf__odom__base_link" / "tf__odom__base_link.npy")
    assert arr.shape == (TF_COUNT, 7)
    np.testing.assert_allclose(arr[0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    ts = np.loadtxt(seq_dir / "tf__odom__base_link" / "timestamps.txt")
    assert ts.shape == (TF_COUNT,)


def test_tf_static_goes_to_calibration_not_channels(tmp_path, fmt):
    from apairo.core.config import read_calibration

    seq_dir, _ = _extract_tf(tmp_path, fmt, static=True)
    # Static edges become calibration, not channels.
    assert not list(seq_dir.glob("tf_static__*"))
    assert (seq_dir / ".apairo" / "calibration.yaml").exists()

    calib = read_calibration(seq_dir)
    assert {"odom_to_base_link", "base_link_to_lidar"} <= set(calib)
    M = calib["base_link_to_lidar"]
    assert M.shape == (4, 4)
    np.testing.assert_allclose(M[:3, 3], [0.0, 0.0, 0.5])     # translation
    np.testing.assert_allclose(M[:3, :3], np.eye(3), atol=1e-9)  # identity rotation


def test_tf_dynamic_channels_static_calibration(tmp_path, fmt):
    """/tf -> per-edge pose channels; /tf_static -> calibration. A shared edge is
    preserved in BOTH stores (a dynamic channel and a static extrinsic)."""
    from apairo.core.config import read_calibration, read_config

    bag = write_tf_bag_with_static(
        tmp_path / ("both.bag" if fmt == "ros1" else "both"), fmt,
        dynamic_edges=[("odom", "base_link", (1.0, 0.0, 0.0)),
                       ("base_link", "lidar", (0.0, 0.0, 0.5))],   # 'base_link__lidar' shared
        static_edges=[("base_link", "lidar", (0.0, 0.0, 0.5)),     # shared
                      ("map", "odom", (2.0, 0.0, 0.0))],           # static-only
    )
    seq_dir, skipped = extract_bag(
        read_bag_info(bag), [TF_TOPIC, TF_STATIC_TOPIC], tmp_path / "out"
    )
    assert not ({TF_TOPIC, TF_STATIC_TOPIC} & set(skipped))

    channels = read_config(seq_dir)["channels"]
    assert {"tf__odom__base_link", "tf__base_link__lidar"} <= set(channels)
    assert not any(k.startswith("tf_static__") for k in channels)  # static -> calibration

    calib = read_calibration(seq_dir)
    assert {"base_link_to_lidar", "map_to_odom"} <= set(calib)

    # The shared edge is preserved in BOTH stores -- comparable, nothing dropped.
    assert np.load(
        seq_dir / "tf__base_link__lidar" / "tf__base_link__lidar.npy"
    ).shape == (TF_COUNT, 7)                          # dynamic timeseries
    np.testing.assert_allclose(calib["base_link_to_lidar"][:3, 3], [0.0, 0.0, 0.5])  # static


def test_tf_output_loads_with_rawdataset(tmp_path, fmt):
    apairo = pytest.importorskip("apairo")
    if not hasattr(apairo, "RawDataset"):
        pytest.skip("installed apairo has no RawDataset")

    seq_dir, _ = _extract_tf(tmp_path, fmt)
    ds = apairo.RawDataset(seq_dir)
    assert {"tf__odom__base_link", "tf__base_link__lidar"} <= set(ds.available)
