"""LiveSource and ChainNode give, message by message, the frames the same
chain gives offline on a recording of the same messages."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from apairo import Compose, Sample
from apairo.dataset.stream import StreamDataset

pytest.importorskip("apairo.core.chain", reason="apairo_ros.live needs apairo >= 0.10")

from apairo_ros.live import ChainNode, LiveSource  # noqa: E402


class Scale:
    def __init__(self, k):
        self.k = k

    def __call__(self, x):
        return x * self.k


class Jitter:
    inference = False

    def __call__(self, x):
        return x + 1e6


class Stack:
    input_keys = ["lidar"]
    output_keys = ["lidar"]

    def __call__(self, samples):
        data = dict(samples[-1].data)
        data["lidar"] = np.concatenate([np.asarray(s.data["lidar"]) for s in samples])
        return Sample(data=data, timestamp=samples[-1].timestamp)


class Score:
    input_keys = ["lidar", "imu"]
    output_keys = ["score"]

    def __call__(self, s):
        s.data["score"] = float(
            np.asarray(s.data["lidar"]).sum() + np.asarray(s.data["imu"]).sum()
        )
        return s


def _messages(n=20, seed=0):
    """Interleaved lidar (10 Hz) and imu (~33 Hz, jittered) messages."""
    rng = np.random.default_rng(seed)
    msgs = [("lidar", 0.1 * i, rng.normal(size=(5, 3))) for i in range(n)]
    t_imu = np.cumsum(rng.uniform(0.02, 0.04, size=3 * n))
    msgs += [("imu", float(t), rng.normal(size=6)) for t in t_imu]
    msgs.sort(key=lambda m: m[1])
    return msgs


def _recording(msgs):
    streams = {}
    for c, t, v in msgs:
        streams.setdefault(c, ([], []))
        streams[c][0].append(t)
        streams[c][1].append(v)
    return StreamDataset({c: (np.asarray(ts), vs) for c, (ts, vs) in streams.items()})


def _deployed(msgs, method="previous", size=3):
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        ds = _recording(msgs).synchronize(reference="lidar", method={"imu": method})
        ds = ds.window(size=size, reduce=Stack())
        ds = ds.transform("lidar", Compose([Scale(2.0), Jitter()]), in_place=False)
        ds = ds.transform(Score(), in_place=False)
        return ds.inference("score")


def test_live_frames_equal_the_offline_frames():
    msgs = _messages()
    deployed = _deployed(msgs)
    offline = {
        deployed[i].timestamp: deployed[i].data["score"] for i in range(len(deployed))
    }

    live = LiveSource(deployed)
    seen = {}
    for c, t, v in msgs:
        frame = live.push(c, t, v)
        if frame is not None:
            assert c == "lidar" and frame.timestamp == t
            seen[t] = frame.data["score"]
    assert seen == offline
    assert len(seen) >= len(msgs) // 4 - 1


def test_buffers_stay_bounded():
    msgs = _messages(n=200)
    live = LiveSource(_deployed(msgs[:80]))
    for c, t, v in msgs:
        live.push(c, t, v)
    assert len(live._t["lidar"]) == live.history == 2 * 2 + 2
    assert len(live._t["imu"]) <= 4 * live.history


def test_a_non_causal_chain_is_refused():
    with pytest.raises(ValueError, match="causal.*method='previous'"):
        LiveSource(_deployed(_messages(), method="nearest"))


def test_late_messages_are_dropped_and_counted():
    live = LiveSource(_deployed(_messages()))
    live.push("imu", 1.0, np.zeros(6))
    with pytest.warns(RuntimeWarning, match="older"):
        assert live.push("imu", 0.5, np.zeros(6)) is None
    assert live.late == 1


def test_nothing_before_every_channel_has_spoken():
    live = LiveSource(_deployed(_messages()))
    assert live.push("lidar", 0.0, np.zeros((5, 3))) is None


def test_a_chain_without_synchronize_answers_every_message():
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        rec = StreamDataset({"lidar": (np.arange(4) * 0.1, [np.ones((2, 3))] * 4)})
        deployed = rec.transform("lidar", Scale(3.0), in_place=False).inference("lidar")
    live = LiveSource(deployed)
    assert live.reference is None
    frame = live.push("lidar", 0.0, np.ones((2, 3)))
    np.testing.assert_array_equal(frame.data["lidar"], np.full((2, 3), 3.0))


# ------------------------------------------------------------ ChainNode


class _Clock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        clock = self

        class _Now:
            nanoseconds = int(round(clock.t * 1e9))

        return _Now()


class FakeNode:
    """What ChainNode uses of an rclpy Node."""

    def __init__(self):
        self.clock = _Clock()
        self.subs = {}

    def create_subscription(self, msg_class, topic, callback, qos):
        self.subs[topic] = callback
        return (topic, msg_class)

    def get_clock(self):
        return self.clock


def test_chain_node_replays_a_bag_as_bag_dataset_reads_it(tmp_path):
    """The node fed a bag's messages at their recorded times gives the
    frames the deployed chain gives on the same bag read in place."""
    rosbags = pytest.importorskip("rosbags")  # noqa: F841
    from apairo_ros import BagDataset
    from rosbags.highlevel import AnyReader

    bag = _write_bag(tmp_path / "bag")
    deployed = _deployed_on_bag(bag)
    expected = deployed.on(BagDataset(bag))
    offline = {
        expected[i].timestamp: expected[i].data["score"] for i in range(len(expected))
    }

    node, frames = FakeNode(), {}
    ts = _typestore()
    ChainNode(
        node,
        deployed,
        topics={
            "lidar": ("/lidar", ts.types["sensor_msgs/msg/PointCloud2"]),
            "imu": ("/imu", ts.types["sensor_msgs/msg/Imu"]),
        },
        on_output=lambda f: frames.__setitem__(f.timestamp, f.data["score"]),
    )
    with AnyReader([bag]) as reader:
        for conn, t_ns, raw in reader.messages():
            node.clock.t = t_ns / 1e9
            node.subs[conn.topic](reader.deserialize(raw, conn.msgtype))
    assert frames == offline and frames


def test_chain_node_needs_a_topic_per_channel():
    with pytest.raises(ValueError, match=r"\['imu'\]"):
        ChainNode(
            FakeNode(),
            _deployed(_messages()),
            topics={"lidar": ("/l", object)},
            on_output=print,
        )


# ---------------------------------------------------------------- bag


def _typestore():
    from rosbags.typesys import Stores, get_typestore

    return get_typestore(Stores.ROS2_HUMBLE)


def _write_bag(path):
    from rosbags.rosbag2 import Writer

    ts = _typestore()
    PC2, PF = (
        ts.types["sensor_msgs/msg/PointCloud2"],
        ts.types["sensor_msgs/msg/PointField"],
    )
    Imu, V3, Q = (
        ts.types["sensor_msgs/msg/Imu"],
        ts.types["geometry_msgs/msg/Vector3"],
        ts.types["geometry_msgs/msg/Quaternion"],
    )
    Header, Time = (
        ts.types["std_msgs/msg/Header"],
        ts.types["builtin_interfaces/msg/Time"],
    )
    rng = np.random.default_rng(1)

    def header(ns):
        return Header(stamp=Time(sec=ns // 10**9, nanosec=ns % 10**9), frame_id="base")

    with Writer(path, version=Writer.VERSION_LATEST) as w:
        cl = w.add_connection("/lidar", PC2.__msgtype__, typestore=ts)
        ci = w.add_connection("/imu", Imu.__msgtype__, typestore=ts)
        events = [(10**9 + i * 10**8, "l") for i in range(8)]
        events += [(10**9 + 5 * 10**6 + i * 3 * 10**7, "i") for i in range(27)]
        for ns, kind in sorted(events):
            if kind == "l":
                pts = rng.normal(size=(4, 3)).astype(np.float32)
                fields = [
                    PF(name=n, offset=4 * k, datatype=7, count=1)
                    for k, n in enumerate("xyz")
                ]
                msg = PC2(
                    header=header(ns),
                    height=1,
                    width=4,
                    fields=fields,
                    is_bigendian=False,
                    point_step=12,
                    row_step=48,
                    data=np.frombuffer(pts.tobytes(), np.uint8),
                    is_dense=True,
                )
                w.write(cl, ns, ts.serialize_cdr(msg, PC2.__msgtype__))
            else:
                a = rng.normal(size=3)
                msg = Imu(
                    header=header(ns),
                    orientation=Q(x=0.0, y=0.0, z=0.0, w=1.0),
                    orientation_covariance=np.zeros(9),
                    angular_velocity=V3(x=a[0], y=a[1], z=a[2]),
                    angular_velocity_covariance=np.zeros(9),
                    linear_acceleration=V3(x=1.0, y=2.0, z=3.0),
                    linear_acceleration_covariance=np.zeros(9),
                )
                w.write(ci, ns, ts.serialize_cdr(msg, Imu.__msgtype__))
    return path


def _deployed_on_bag(bag):
    from apairo_ros import BagDataset

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        ds = BagDataset(bag).synchronize(reference="lidar")
        ds = ds.window(size=2, reduce=Stack())
        ds = ds.transform("lidar", Compose([Scale(2.0), Jitter()]), in_place=False)
        ds = ds.transform(Score(), in_place=False)
        return ds.inference("score")
