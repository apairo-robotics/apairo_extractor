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
        if f.datatype not in _PC2_DTYPE:
            raise ValueError(
                f"PointCloud2 field {f.name!r} has unknown datatype {f.datatype} "
                f"(sensor_msgs/PointField defines 1..8)"
            )
        if int(getattr(f, "count", 1)) != 1:
            raise ValueError(
                f"PointCloud2 field {f.name!r} has count {f.count}: only scalar "
                f"fields (count 1) are decoded"
            )
        base_dt = np.dtype(_PC2_DTYPE[f.datatype])
        dt = base_dt.newbyteorder(endian)
        nb = base_dt.itemsize
        col = raw[:, f.offset : f.offset + nb]
        col = col.reshape(-1, nb).view(dt).reshape(-1).astype(np.float32)
        cols.append(col)

    return np.column_stack(cols) if cols else np.zeros((n_pts, 3), dtype=np.float32)


# sensor_msgs/image_encodings: named encodings -> (numpy dtype, channels).
_NAMED_IMAGE_ENCODINGS: dict[str, tuple[type, int]] = {
    "mono8": (np.uint8, 1), "mono16": (np.uint16, 1),
    "rgb8": (np.uint8, 3), "bgr8": (np.uint8, 3),
    "rgba8": (np.uint8, 4), "bgra8": (np.uint8, 4),
    "rgb16": (np.uint16, 3), "bgr16": (np.uint16, 3),
    "rgba16": (np.uint16, 4), "bgra16": (np.uint16, 4),
    # Bayer mosaics are kept raw (one channel, not demosaiced).
    **{f"bayer_{p}{b}": (np.uint16 if b == 16 else np.uint8, 1)
       for p in ("rggb", "bggr", "gbrg", "grbg") for b in (8, 16)},
}

# Generic "<bits><U|S|F>C<n>" encodings ("32FC1", "16UC1", "8UC3", ...).
_GENERIC_IMAGE_DTYPES: dict[str, type] = {
    "8U": np.uint8, "8S": np.int8, "16U": np.uint16, "16S": np.int16,
    "32S": np.int32, "32F": np.float32, "64F": np.float64,
}

# Channel reordering to canonical RGB(A) for BGR-ordered encodings.
_TO_RGB: dict[str, list[int]] = {
    "bgr8": [2, 1, 0], "bgr16": [2, 1, 0],
    "bgra8": [2, 1, 0, 3], "bgra16": [2, 1, 0, 3],
}


def image_encoding_layout(encoding: str) -> tuple[np.dtype, int]:
    """``(dtype, channels)`` of a ``sensor_msgs/Image`` encoding.

    Raises ``ValueError`` for an encoding the decoder does not handle (packed
    YUV, empty, unknown): its bytes are never reinterpreted as something else.
    """
    enc = (encoding or "").strip()
    named = _NAMED_IMAGE_ENCODINGS.get(enc.lower())
    if named is not None:
        return np.dtype(named[0]), named[1]
    depth, sep, n = enc.upper().partition("C")
    if sep and depth in _GENERIC_IMAGE_DTYPES and (n == "" or (n.isdigit() and int(n) >= 1)):
        return np.dtype(_GENERIC_IMAGE_DTYPES[depth]), int(n) if n else 1
    raise ValueError(
        f"unsupported sensor_msgs/Image encoding {encoding!r}: supported are "
        f"{sorted(_NAMED_IMAGE_ENCODINGS)} and <8U|8S|16U|16S|32S|32F|64F>C<n>"
    )


def _image(msg) -> np.ndarray:
    """Decode an Image by its encoding: dtype, channels, endianness, row step.

    The result is ``(H, W)`` for one channel, ``(H, W, C)`` otherwise, in the
    encoding's dtype (native byte order). BGR(A) encodings are reordered to
    canonical RGB(A) -- the cv_bridge / OpenCV default would otherwise show up
    as swapped red and blue downstream.
    """
    enc = (getattr(msg, "encoding", "") or "").strip()
    dtype, channels = image_encoding_layout(enc)
    h, w, step = int(msg.height), int(msg.width), int(msg.step)
    row_bytes = w * channels * dtype.itemsize
    if step < row_bytes:
        raise ValueError(
            f"Image step {step} < width {w} x {channels} channel(s) x "
            f"{dtype.itemsize} byte(s) = {row_bytes} for encoding {enc!r}"
        )
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    if raw.size < h * step:
        raise ValueError(
            f"Image data holds {raw.size} bytes, expected height {h} x step {step} "
            f"= {h * step} for encoding {enc!r}"
        )
    rows = raw[: h * step].reshape(h, step)[:, :row_bytes]   # drop row padding
    order = ">" if getattr(msg, "is_bigendian", False) else "<"
    img = np.ascontiguousarray(rows).view(dtype.newbyteorder(order))
    img = img.astype(dtype, copy=False)                       # native byte order
    img = img.reshape(h, w) if channels == 1 else img.reshape(h, w, channels)
    perm = _TO_RGB.get(enc.lower())
    if perm is not None:
        img = np.ascontiguousarray(img[..., perm])
    return img


def _compressed_image(msg) -> np.ndarray:
    # Without Pillow the compressed bytes are NOT returned as if they were
    # pixels: that would store a JPEG/PNG byte stream as an "image".
    try:
        from PIL import Image as PILImage
    except ImportError as exc:
        raise ImportError(
            "decoding sensor_msgs/CompressedImage requires Pillow (pip install Pillow)"
        ) from exc
    import io
    return np.array(PILImage.open(io.BytesIO(bytes(msg.data))))


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
