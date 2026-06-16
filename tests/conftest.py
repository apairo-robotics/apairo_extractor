"""Shared test fixtures.

We generate *real* tiny bags with rosbags' Writer rather than committing binary
blobs: the fixtures stay diffable, exercise the real bag format end-to-end, and
cover both ROS1 (``*.bag`` file) and ROS2 (directory) so the same tests guard
the future ROS2 path. ``rosbags`` is already a runtime dependency.
"""
from __future__ import annotations

import numpy as np
import pytest

from rosbags.typesys import Stores, get_types_from_msg, get_typestore

# Each format maps to (typestore store, serialize method name).
_FORMATS = {
    "ros1": (Stores.ROS1_NOETIC, "serialize_ros1"),
    "ros2": (Stores.ROS2_HUMBLE, "serialize_cdr"),
}

# Expected contents of the tiny bag, independent of format.
LIDAR_TOPIC = "/lidar"          # PointCloud2 -> frame channel (one .npy per msg)
IMU_TOPIC = "/imu"              # Imu         -> seq channel (one stacked .npy)
CHATTER_TOPIC = "/chatter"     # String      -> unsupported, must be skipped

LIDAR_COUNT = 3
IMU_COUNT = 4
CHATTER_COUNT = 2

TF_TOPIC = "/tf"               # TFMessage -> demuxed into one channel per edge
TF_STATIC_TOPIC = "/tf_static"
TF_COUNT = 5                   # dynamic /tf messages
# Edges present in every TF message: (parent, child, translation).
TF_EDGES = [("odom", "base_link", (1.0, 0.0, 0.0)),
            ("base_link", "lidar", (0.0, 0.0, 0.5))]

# Point cloud payload: 2 xyz points, float32.
CLOUD_POINTS = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)


def _header(ts, fmt, stamp_ns):
    Header = ts.types["std_msgs/msg/Header"]
    Time = ts.types["builtin_interfaces/msg/Time"]
    stamp = Time(sec=stamp_ns // 1_000_000_000, nanosec=stamp_ns % 1_000_000_000)
    if fmt == "ros1":  # ROS1 Header has a 'seq' field, ROS2 does not.
        return Header(seq=0, stamp=stamp, frame_id="base")
    return Header(stamp=stamp, frame_id="base")


def _imu_msg(ts, fmt, stamp_ns):
    Imu = ts.types["sensor_msgs/msg/Imu"]
    Vec3 = ts.types["geometry_msgs/msg/Vector3"]
    Quat = ts.types["geometry_msgs/msg/Quaternion"]
    return Imu(
        header=_header(ts, fmt, stamp_ns),
        orientation=Quat(x=0.0, y=0.0, z=0.0, w=1.0),
        orientation_covariance=np.zeros(9),
        angular_velocity=Vec3(x=0.1, y=0.2, z=0.3),
        angular_velocity_covariance=np.zeros(9),
        linear_acceleration=Vec3(x=1.0, y=2.0, z=3.0),
        linear_acceleration_covariance=np.zeros(9),
    )


def _cloud_msg(ts, fmt, stamp_ns):
    PC2 = ts.types["sensor_msgs/msg/PointCloud2"]
    PF = ts.types["sensor_msgs/msg/PointField"]
    fields = [
        PF(name="x", offset=0, datatype=7, count=1),   # 7 = FLOAT32
        PF(name="y", offset=4, datatype=7, count=1),
        PF(name="z", offset=8, datatype=7, count=1),
    ]
    data = np.frombuffer(CLOUD_POINTS.tobytes(), dtype=np.uint8)
    return PC2(
        header=_header(ts, fmt, stamp_ns),
        height=1,
        width=CLOUD_POINTS.shape[0],
        fields=fields,
        is_bigendian=False,
        point_step=12,
        row_step=12 * CLOUD_POINTS.shape[0],
        data=data,
        is_dense=True,
    )


def _string_msg(ts, stamp_ns):
    String = ts.types["std_msgs/msg/String"]
    return String(data="hello")


def _open_writer(path, fmt):
    if fmt == "ros1":
        from rosbags.rosbag1 import Writer
        return Writer(path)
    from rosbags.rosbag2 import Writer
    return Writer(path, version=Writer.VERSION_LATEST)


def write_tiny_bag(path, fmt):
    """Write a tiny bag with lidar/imu/chatter topics. Returns the bag path.

    For ros1, ``path`` is the ``*.bag`` file; for ros2 it is the bag directory.
    """
    store, serialize_name = _FORMATS[fmt]
    ts = get_typestore(store)
    serialize = getattr(ts, serialize_name)
    path.parent.mkdir(parents=True, exist_ok=True)

    # (topic, msgtype, builder, count) — interleaved timestamps below.
    specs = [
        (LIDAR_TOPIC, ts.types["sensor_msgs/msg/PointCloud2"].__msgtype__, _cloud_msg, LIDAR_COUNT),
        (IMU_TOPIC, ts.types["sensor_msgs/msg/Imu"].__msgtype__, _imu_msg, IMU_COUNT),
        (CHATTER_TOPIC, ts.types["std_msgs/msg/String"].__msgtype__, _string_msg, CHATTER_COUNT),
    ]

    with _open_writer(path, fmt) as writer:
        for topic, msgtype, builder, count in specs:
            conn = writer.add_connection(topic, msgtype, typestore=ts)
            for i in range(count):
                stamp_ns = 1_000_000_000 + i * 100_000_000  # 0.1s apart
                try:
                    msg = builder(ts, fmt, stamp_ns)
                except TypeError:  # _string_msg ignores fmt
                    msg = builder(ts, stamp_ns)
                writer.write(conn, stamp_ns, serialize(msg, msgtype))
    return path


def _tf_message(ts, fmt, stamp_ns):
    """A TFMessage carrying all TF_EDGES at *stamp_ns* (identity rotation)."""
    TFMessage = ts.types["tf2_msgs/msg/TFMessage"]
    TransformStamped = ts.types["geometry_msgs/msg/TransformStamped"]
    Transform = ts.types["geometry_msgs/msg/Transform"]
    Vec3 = ts.types["geometry_msgs/msg/Vector3"]
    Quat = ts.types["geometry_msgs/msg/Quaternion"]
    Header = ts.types["std_msgs/msg/Header"]
    Time = ts.types["builtin_interfaces/msg/Time"]
    stamp = Time(sec=stamp_ns // 1_000_000_000, nanosec=stamp_ns % 1_000_000_000)

    stamped = []
    for parent, child, (tx, ty, tz) in TF_EDGES:
        if fmt == "ros1":
            header = Header(seq=0, stamp=stamp, frame_id=parent)
        else:
            header = Header(stamp=stamp, frame_id=parent)
        stamped.append(TransformStamped(
            header=header,
            child_frame_id=child,
            transform=Transform(
                translation=Vec3(x=tx, y=ty, z=tz),
                rotation=Quat(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
        ))
    return TFMessage(transforms=stamped)


def write_tf_bag(path, fmt, *, static=False):
    """Write a bag with a single TF topic (``/tf`` or ``/tf_static``)."""
    store, serialize_name = _FORMATS[fmt]
    ts = get_typestore(store)
    # tf2_msgs is not bundled in the ROS1 typestore; register it (its definition
    # is just a TransformStamped array, exactly what a real ROS1 bag carries).
    if "tf2_msgs/msg/TFMessage" not in ts.types:
        ts.register(get_types_from_msg(
            "geometry_msgs/TransformStamped[] transforms", "tf2_msgs/msg/TFMessage"))
    serialize = getattr(ts, serialize_name)
    path.parent.mkdir(parents=True, exist_ok=True)

    topic = TF_STATIC_TOPIC if static else TF_TOPIC
    msgtype = ts.types["tf2_msgs/msg/TFMessage"].__msgtype__
    count = 1 if static else TF_COUNT

    with _open_writer(path, fmt) as writer:
        conn = writer.add_connection(topic, msgtype, typestore=ts)
        for i in range(count):
            stamp_ns = 1_000_000_000 + i * 100_000_000
            writer.write(conn, stamp_ns, serialize(_tf_message(ts, fmt, stamp_ns), msgtype))
    return path


@pytest.fixture(params=["ros1", "ros2"])
def fmt(request):
    return request.param


@pytest.fixture
def tiny_bag(tmp_path, fmt):
    """A real tiny bag in the parametrized format. Returns its path."""
    path = tmp_path / ("tiny.bag" if fmt == "ros1" else "tiny")
    return write_tiny_bag(path, fmt)
