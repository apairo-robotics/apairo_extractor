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

# Message types stored as a single stacked .npy (one row per message).
# Keep in sync with the branches in convert_message().
SEQ_MSGTYPES: frozenset[str] = frozenset({
    "Imu",
    "Twist",
    "TwistStamped",
    "TwistWithCovarianceStamped",
    "Odometry",
    "NavSatFix",
    "PoseStamped",
    "WrenchStamped",
})

# Everything convert_message() can handle.
SUPPORTED_MSGTYPES: frozenset[str] = FRAME_MSGTYPES | SEQ_MSGTYPES

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


def is_supported_msgtype(msgtype: str) -> bool:
    """Whether convert_message() can turn this message type into an array."""
    return msgtype_short(msgtype) in SUPPORTED_MSGTYPES


def convert_message(msg, msgtype: str) -> np.ndarray | None:
    """Convert a deserialized ROS message to numpy.  Returns None if unsupported.

    Raises:
        ValueError: The message is of a supported type but cannot be laid out
            as its header says (a truncated cloud, an unknown image encoding).
            Callers count and report such messages; none is read by guessing.
    """
    short = msgtype_short(msgtype)

    if short == "PointCloud2":
        return _pointcloud2(msg)
    if short == "Image":
        return _image(msg)
    if short == "CompressedImage":
        return _compressed_image(msg)
    if short == "LaserScan":
        return _laser_scan(msg)
    if short == "PointCloud":
        return _pointcloud(msg)
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
    """One name per column of the converted cloud: a field with ``count > 1``
    gives ``<name>_0 .. <name>_<count-1>``."""
    names: list[str] = []
    for f in msg.fields:
        count = max(int(getattr(f, "count", 1) or 1), 1)
        names += [f.name] if count == 1 else [f"{f.name}_{j}" for j in range(count)]
    return names


# ── Private converters ─────────────────────────────────────────────────────────


def _pointcloud2(msg) -> np.ndarray:
    """A PointCloud2 as ``(N, C)``, one column per scalar the fields declare.

    The dtype is the smallest that holds every field exactly: float32 for the
    usual all-float32 cloud, float64 as soon as a field would lose values in
    float32 (a uint32 or float64 per-point time, a range in millimetres).
    Row padding of an organized cloud (``row_step > width * point_step``) is
    skipped. A payload that does not hold the points the header declares is
    refused rather than read as fewer, or other, points.
    """
    fields = list(msg.fields)
    dtypes = []
    for f in fields:
        base = _PC2_DTYPE.get(f.datatype)
        if base is None:
            raise ValueError(
                f"PointCloud2 field '{f.name}' has unknown datatype {f.datatype}."
            )
        dtypes.append(np.dtype(base))
    counts = [max(int(getattr(f, "count", 1) or 1), 1) for f in fields]
    n_cols = sum(counts)
    out_dtype = np.result_type(*dtypes) if dtypes else np.dtype(np.float32)
    if out_dtype.kind != "f":  # integer-only clouds still mix with float xyz downstream
        out_dtype = np.result_type(out_dtype, np.float32)

    width, height, step = int(msg.width), int(msg.height), int(msg.point_step)
    n_pts = width * height
    if n_pts == 0:
        return np.zeros((0, n_cols or 3), dtype=out_dtype)
    if step <= 0:
        raise ValueError(f"PointCloud2 has point_step {step}.")

    raw = np.frombuffer(msg.data, dtype=np.uint8)
    row_step = int(getattr(msg, "row_step", 0) or 0) or width * step
    if row_step < width * step:
        raise ValueError(
            f"PointCloud2 row_step {row_step} is smaller than width * point_step "
            f"({width} * {step})."
        )
    if len(raw) == height * row_step:
        raw = raw.reshape(height, row_step)[:, : width * step]
    elif len(raw) == n_pts * step:  # unpadded, whatever row_step says
        pass
    else:
        raise ValueError(
            f"PointCloud2 payload holds {len(raw)} byte(s) for {width} x {height} "
            f"point(s) of {step} byte(s) (row_step {row_step}): truncated, or its "
            f"header is wrong."
        )
    raw = np.ascontiguousarray(raw).reshape(n_pts, step)

    endian = ">" if msg.is_bigendian else "<"
    out = np.empty((n_pts, n_cols), dtype=out_dtype)
    col = 0
    for f, base, count in zip(fields, dtypes, counts, strict=True):
        nb = base.itemsize
        if f.offset + nb * count > step:
            raise ValueError(
                f"PointCloud2 field '{f.name}' (offset {f.offset}, {count} x "
                f"{nb} byte(s)) does not fit in point_step {step}."
            )
        block = raw[:, f.offset : f.offset + nb * count]
        values = np.ascontiguousarray(block).view(base.newbyteorder(endian))
        out[:, col : col + count] = values.reshape(n_pts, count)
        col += count
    if not fields:
        return np.zeros((n_pts, 3), dtype=np.float32)
    return out


# Image encodings (sensor_msgs/image_encodings.h) -> (numpy dtype, channels).
# Bayer patterns are stored as their single raw channel.
_IMAGE_ENCODINGS: dict[str, tuple[type, int]] = {
    "mono8": (np.uint8, 1), "8uc1": (np.uint8, 1), "8sc1": (np.int8, 1),
    "mono16": (np.uint16, 1), "16uc1": (np.uint16, 1), "16sc1": (np.int16, 1),
    "32sc1": (np.int32, 1), "32fc1": (np.float32, 1), "64fc1": (np.float64, 1),
    "rgb8": (np.uint8, 3), "bgr8": (np.uint8, 3), "8uc3": (np.uint8, 3),
    "rgba8": (np.uint8, 4), "bgra8": (np.uint8, 4), "8uc4": (np.uint8, 4),
    "rgb16": (np.uint16, 3), "bgr16": (np.uint16, 3), "16uc3": (np.uint16, 3),
    "rgba16": (np.uint16, 4), "bgra16": (np.uint16, 4), "16uc4": (np.uint16, 4),
    "32fc3": (np.float32, 3), "32fc4": (np.float32, 4),
    "8uc2": (np.uint8, 2), "16uc2": (np.uint16, 2), "32fc2": (np.float32, 2),
    "yuv422": (np.uint8, 2), "yuv422_yuy2": (np.uint8, 2),
    **{f"bayer_{p}{b}": (np.uint8 if b == "8" else np.uint16, 1)
       for p in ("rggb", "bggr", "gbrg", "grbg") for b in ("8", "16")},
}


def _image(msg) -> np.ndarray:
    """An Image as ``(H, W)`` or ``(H, W, C)`` in the dtype its encoding names
    (a 16UC1 depth stays uint16, a 32FC1 float32). Row padding
    (``step > width * bytes per pixel``) is skipped, ``is_bigendian`` honoured,
    and BGR(A) stored as RGB(A). An encoding that is not known is refused: its
    bytes cannot be laid out by guessing."""
    enc = (getattr(msg, "encoding", "") or "").lower()
    height, width, step = int(msg.height), int(msg.width), int(msg.step)
    if enc not in _IMAGE_ENCODINGS:
        raise ValueError(
            f"Image encoding '{getattr(msg, 'encoding', '')}' is not one "
            f"apairo_ros converts ({', '.join(sorted(_IMAGE_ENCODINGS))})."
        )
    base, channels = _IMAGE_ENCODINGS[enc]
    dtype = np.dtype(base)
    row_bytes = width * channels * dtype.itemsize
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    if step < row_bytes or len(raw) != height * step:
        raise ValueError(
            f"Image '{enc}' {width} x {height} needs {row_bytes} byte(s) per row; "
            f"got step {step} and {len(raw)} byte(s) for {height} row(s)."
        )
    rows = np.ascontiguousarray(raw.reshape(height, step)[:, :row_bytes])
    endian = ">" if getattr(msg, "is_bigendian", 0) else "<"
    img = rows.view(dtype.newbyteorder(endian)).astype(dtype, copy=False)
    img = img.reshape(height, width) if channels == 1 else img.reshape(height, width, channels)
    # ROS colour topics are commonly published as bgr8/bgra8 (the cv_bridge /
    # OpenCV default); the stored array is canonical RGB(A).
    if enc.startswith("bgra"):
        img = np.ascontiguousarray(img[..., [2, 1, 0, 3]])
    elif enc.startswith("bgr"):
        img = np.ascontiguousarray(img[..., ::-1])
    return np.ascontiguousarray(img)


def _compressed_image(msg) -> np.ndarray:
    """A CompressedImage decoded with Pillow. image_transport's
    ``compressedDepth`` puts a 12-byte header before the PNG: it is skipped (a
    16-bit PNG then gives the raw depth; a 32FC1 depth was quantised by the
    publisher and is returned as stored)."""
    import io

    try:
        from PIL import Image as PILImage
    except ImportError as exc:
        raise ImportError(
            "Decoding a CompressedImage needs Pillow (pip install Pillow)."
        ) from exc
    data = bytes(msg.data)
    fmt = (getattr(msg, "format", "") or "").lower()
    if "compresseddepth" in fmt:
        data = data[12:]
    try:
        return np.array(PILImage.open(io.BytesIO(data)))
    except Exception as exc:
        raise ValueError(
            f"CompressedImage (format '{getattr(msg, 'format', '')}') could not "
            f"be decoded: {exc}"
        ) from exc


def _pointcloud(msg) -> np.ndarray:
    """A sensor_msgs/PointCloud as ``(N, 3 + channels)`` float32: x, y, z, then
    one column per channel (their values are float32 in the message)."""
    xyz = np.array([[p.x, p.y, p.z] for p in msg.points], dtype=np.float32).reshape(-1, 3)
    extra = [np.asarray(c.values, dtype=np.float32) for c in getattr(msg, "channels", [])]
    for c in extra:
        if len(c) != len(xyz):
            raise ValueError(
                f"PointCloud channel holds {len(c)} value(s) for {len(xyz)} point(s)."
            )
    return np.column_stack([xyz, *extra]) if extra else xyz


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
