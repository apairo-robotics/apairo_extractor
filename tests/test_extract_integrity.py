"""Extraction must not silently corrupt or mislabel data.

Real bags written with rosbags' Writer, where the record (log) time and the
header stamp differ -- as in a simulator bag, whose headers carry sim time
while the recorder stamps wall time.
"""
from __future__ import annotations

import numpy as np
import pytest
import yaml
from rosbags.typesys import Stores, get_typestore

from apairo_extractor.bag import read_bag_info
from apairo_extractor.extract import extract_bag

from conftest import _open_writer, _tf_message

LOG0 = 1_782_735_780_000_000_000   # wall clock, as the recorder writes it
HDR0 = 35_000_000_000              # sim clock, as the publisher stamps it
DT = 100_000_000


def _ts():
    ts = get_typestore(Stores.ROS2_HUMBLE)
    return ts


def _header(ts, stamp_ns, frame_id):
    Header = ts.types["std_msgs/msg/Header"]
    Time = ts.types["builtin_interfaces/msg/Time"]
    return Header(stamp=Time(sec=stamp_ns // 10**9, nanosec=stamp_ns % 10**9), frame_id=frame_id)


def _image(ts, stamp_ns, arr, encoding, frame_id="cam"):
    Image = ts.types["sensor_msgs/msg/Image"]
    h, w = arr.shape[:2]
    data = np.frombuffer(np.ascontiguousarray(arr).tobytes(), dtype=np.uint8)
    return Image(header=_header(ts, stamp_ns, frame_id), height=h, width=w,
                 encoding=encoding, is_bigendian=0, step=len(data) // h, data=data)


def _write(path, messages):
    """messages: list of (topic, log_ns, msg)."""
    ts = _ts()
    conns = {}
    with _open_writer(path, "ros2") as writer:
        for topic, log_ns, msg in messages:
            if topic not in conns:
                conns[topic] = writer.add_connection(topic, msg.__msgtype__, typestore=ts)
            writer.write(conns[topic], log_ns, ts.serialize_cdr(msg, msg.__msgtype__))
    return path


def test_32fc1_depth_extracted_as_float32(tmp_path):
    ts = _ts()
    depth = np.array([[1.25, np.inf], [0.5, 80.0]], dtype=np.float32)
    bag = _write(tmp_path / "bag", [("/depth", LOG0 + i * DT, _image(ts, HDR0 + i * DT, depth, "32FC1"))
                                    for i in range(2)])
    seq_dir, skipped = extract_bag(read_bag_info(bag), ["/depth"], tmp_path / "out")
    assert "/depth" not in skipped
    arr = np.load(seq_dir / "depth" / "000000.npy")
    assert arr.dtype == np.float32 and arr.shape == (2, 2)
    np.testing.assert_array_equal(arr, depth)
    meta = yaml.safe_load((seq_dir / "depth" / "metadata.yaml").read_text())
    assert meta["encoding"] == ["32FC1"]


def test_unsupported_encoding_names_the_topic(tmp_path):
    ts = _ts()
    img = _image(ts, HDR0, np.zeros((2, 4), np.uint8), "yuv422")
    bag = _write(tmp_path / "bag", [("/cam/yuv", LOG0, img)])
    with pytest.raises(ValueError, match="/cam/yuv"):
        extract_bag(read_bag_info(bag), ["/cam/yuv"], tmp_path / "out")


def test_tf_edges_share_the_clock_of_the_other_channels(tmp_path):
    # /tf used to take the HEADER stamp (sim time, ~35 s) while every other
    # channel takes the record time (wall clock, ~1.78e9 s): the pose stream
    # and the sensor streams of one sequence were on two unrelated clocks.
    ts = _ts()
    if "tf2_msgs/msg/TFMessage" not in ts.types:
        pytest.skip("typestore without tf2_msgs")
    msgs = []
    for i in range(3):
        msgs.append(("/depth", LOG0 + i * DT,
                     _image(ts, HDR0 + i * DT, np.zeros((1, 1), np.float32), "32FC1")))
        msgs.append(("/tf", LOG0 + i * DT + 1, _tf_message(ts, "ros2", HDR0 + i * DT)))
    bag = _write(tmp_path / "bag", msgs)
    seq_dir, _ = extract_bag(read_bag_info(bag), ["/depth", "/tf"], tmp_path / "out")
    t_img = np.atleast_1d(np.loadtxt(seq_dir / "depth" / "timestamps.txt"))
    t_tf = np.atleast_1d(np.loadtxt(seq_dir / "tf__odom__base_link" / "timestamps.txt"))
    assert abs(t_tf[0] - t_img[0]) < 1.0
    np.testing.assert_allclose(t_tf, (LOG0 + np.arange(3) * DT + 1) / 1e9)


def test_topic_with_changing_frame_id_is_not_labelled_with_the_first(tmp_path):
    # The channel frame came from the first message only: a topic whose
    # frame_id changes mid-bag was labelled with a frame most frames are not in.
    ts = _ts()
    a = np.zeros((1, 1), np.float32)
    msgs = [("/depth", LOG0 + i * DT, _image(ts, HDR0 + i * DT, a, "32FC1",
                                             frame_id="cam_left" if i == 0 else "cam_right"))
            for i in range(3)]
    bag = _write(tmp_path / "bag", msgs)
    from apairo.core.config import read_config
    seq_dir, _ = extract_bag(read_bag_info(bag), ["/depth"], tmp_path / "out")
    ch = read_config(seq_dir)["channels"]["depth"]
    assert ch.get("frame") in (None, "")
    meta = yaml.safe_load((seq_dir / "depth" / "metadata.yaml").read_text())
    assert meta["frame_ids"] == ["cam_left", "cam_right"]


def test_constant_frame_id_still_labels_the_channel(tmp_path):
    ts = _ts()
    a = np.zeros((1, 1), np.float32)
    bag = _write(tmp_path / "bag", [("/depth", LOG0 + i * DT, _image(ts, HDR0 + i * DT, a, "32FC1"))
                                    for i in range(2)])
    from apairo.core.config import read_config
    seq_dir, _ = extract_bag(read_bag_info(bag), ["/depth"], tmp_path / "out")
    assert read_config(seq_dir)["channels"]["depth"]["frame"] == "cam"
