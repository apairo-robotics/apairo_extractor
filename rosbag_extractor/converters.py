from __future__ import annotations

import numpy as np

# Message types that produce one file per frame (npys loader)
FRAME_MSGTYPES: frozenset[str] = frozenset({
    "PointCloud2",
    "Image",
    "CompressedImage",
    "LaserScan",
    "PointCloud",
})

# PointCloud2 datatype enum → numpy dtype
_PC2_DTYPE: dict[int, type] = {
    1: np.int8,
    2: np.uint8,
    3: np.int16,
    4: np.uint16,
    5: np.int32,
    6: np.uint32,
    7: np.float32,
    8: np.float64,
}


def msgtype_short(msgtype: str) -> str:
    """'sensor_msgs/msg/PointCloud2' → 'PointCloud2'"""
    return msgtype.split("/")[-1]


def is_frame_type(msgtype: str) -> bool:
    return msgtype_short(msgtype) in FRAME_MSGTYPES


def convert_message(msg, msgtype: str) -> np.ndarray | None:
    """Convert a deserialized ROS message to numpy.  Returns None if unsupported."""
    short = msgtype_short(msgtype)

    if short == "PointCloud2":
        return _pointcloud2(msg)
    if short == "Image":
        return _image(msg)
    if short == "CompressedImage":
        return _compressed_image(msg)
    if short == "LaserScan":
        return _laser_scan(msg)
    if short == "Imu":
        return _imu(msg)
    if short == "Twist":
        return _twist(msg.linear, msg.angular)
    if short == "TwistStamped":
        return _twist(msg.twist.linear, msg.twist.angular)
    if short == "TwistWithCovarianceStamped":
        return _twist(msg.twist.twist.linear, msg.twist.twist.angular)
    if short == "Odometry":
        return _odometry(msg)
    if short == "NavSatFix":
        return _navsatfix(msg)
    if short == "PoseStamped":
        p, q = msg.pose.position, msg.pose.orientation
        return np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w], dtype=np.float64)
    if short == "WrenchStamped":
        f, t = msg.wrench.force, msg.wrench.torque
        return np.array([f.x, f.y, f.z, t.x, t.y, t.z], dtype=np.float64)
    return None


def pointcloud2_field_names(msg) -> list[str]:
    return [f.name for f in msg.fields]


# ── Private converters ─────────────────────────────────────────────────────────


def _pointcloud2(msg) -> np.ndarray:
    n_pts = msg.width * msg.height
    if n_pts == 0:
        return np.zeros((0, 3), dtype=np.float32)

    endian = ">" if msg.is_bigendian else "<"
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    expected = n_pts * msg.point_step
    if len(raw) != expected:
        n_pts = len(raw) // msg.point_step
        raw = raw[: n_pts * msg.point_step]

    raw = raw.reshape(n_pts, msg.point_step)
    cols = []
    for f in msg.fields:
        base_dt = np.dtype(_PC2_DTYPE.get(f.datatype, np.float32))
        dt = base_dt.newbyteorder(endian)
        nb = base_dt.itemsize
        col = raw[:, f.offset : f.offset + nb]
        col = col.reshape(-1, nb).view(dt).reshape(-1).astype(np.float32)
        cols.append(col)

    return np.column_stack(cols) if cols else np.zeros((n_pts, 3), dtype=np.float32)


def _image(msg) -> np.ndarray:
    data = np.frombuffer(msg.data, dtype=np.uint8)
    channels = msg.step // msg.width if msg.width else 1
    if channels > 1:
        return data.reshape(msg.height, msg.width, channels)
    return data.reshape(msg.height, msg.width)


def _compressed_image(msg) -> np.ndarray:
    try:
        from PIL import Image as PILImage
        import io
        return np.array(PILImage.open(io.BytesIO(bytes(msg.data))))
    except ImportError:
        return np.frombuffer(msg.data, dtype=np.uint8)


def _laser_scan(msg) -> np.ndarray:
    return np.array(msg.ranges, dtype=np.float32)


def _imu(msg) -> np.ndarray:
    la, av, q = msg.linear_acceleration, msg.angular_velocity, msg.orientation
    return np.array(
        [la.x, la.y, la.z, av.x, av.y, av.z, q.x, q.y, q.z, q.w],
        dtype=np.float64,
    )


def _twist(linear, angular) -> np.ndarray:
    return np.array(
        [linear.x, linear.y, linear.z, angular.x, angular.y, angular.z],
        dtype=np.float64,
    )


def _odometry(msg) -> np.ndarray:
    p, q = msg.pose.pose.position, msg.pose.pose.orientation
    return np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w], dtype=np.float64)


def _navsatfix(msg) -> np.ndarray:
    return np.array([msg.latitude, msg.longitude, msg.altitude], dtype=np.float64)
