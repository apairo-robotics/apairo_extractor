"""Unit tests for the pure message->numpy converters (no bag needed)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from apairo_extractor import converters as cv


def ns(**kw):
    return SimpleNamespace(**kw)


def test_msgtype_short_and_is_frame_type():
    assert cv.msgtype_short("sensor_msgs/msg/PointCloud2") == "PointCloud2"
    assert cv.msgtype_short("sensor_msgs/PointCloud2") == "PointCloud2"
    assert cv.is_frame_type("sensor_msgs/msg/PointCloud2")
    assert cv.is_frame_type("sensor_msgs/msg/Image")
    assert not cv.is_frame_type("sensor_msgs/msg/Imu")


def test_convert_message_unsupported_returns_none():
    assert cv.convert_message(ns(data="x"), "std_msgs/msg/String") is None


def test_imu():
    msg = ns(
        linear_acceleration=ns(x=1.0, y=2.0, z=3.0),
        angular_velocity=ns(x=0.1, y=0.2, z=0.3),
        orientation=ns(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    arr = cv.convert_message(msg, "sensor_msgs/msg/Imu")
    assert arr.dtype == np.float64
    np.testing.assert_array_equal(
        arr, [1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
    )


def test_twist_variants_equivalent():
    lin, ang = ns(x=1.0, y=2.0, z=3.0), ns(x=4.0, y=5.0, z=6.0)
    expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    plain = cv.convert_message(ns(linear=lin, angular=ang), "geometry_msgs/msg/Twist")
    stamped = cv.convert_message(
        ns(twist=ns(linear=lin, angular=ang)), "geometry_msgs/msg/TwistStamped"
    )
    cov = cv.convert_message(
        ns(twist=ns(twist=ns(linear=lin, angular=ang))),
        "geometry_msgs/msg/TwistWithCovarianceStamped",
    )
    np.testing.assert_array_equal(plain, expected)
    np.testing.assert_array_equal(stamped, expected)
    np.testing.assert_array_equal(cov, expected)


def test_navsatfix():
    msg = ns(latitude=48.85, longitude=2.35, altitude=35.0)
    np.testing.assert_array_equal(
        cv.convert_message(msg, "sensor_msgs/msg/NavSatFix"), [48.85, 2.35, 35.0]
    )


def test_odometry_and_posestamped():
    pose = ns(position=ns(x=1.0, y=2.0, z=3.0), orientation=ns(x=0.0, y=0.0, z=0.0, w=1.0))
    odom = cv.convert_message(ns(pose=ns(pose=pose)), "nav_msgs/msg/Odometry")
    posest = cv.convert_message(ns(pose=pose), "geometry_msgs/msg/PoseStamped")
    expected = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]
    np.testing.assert_array_equal(odom, expected)
    np.testing.assert_array_equal(posest, expected)


def test_laser_scan():
    msg = ns(ranges=[0.5, 1.5, 2.5])
    arr = cv.convert_message(msg, "sensor_msgs/msg/LaserScan")
    assert arr.dtype == np.float32
    np.testing.assert_array_equal(arr, [0.5, 1.5, 2.5])


def test_pointcloud2_decodes_xyz():
    pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    fields = [
        ns(name="x", offset=0, datatype=7, count=1),
        ns(name="y", offset=4, datatype=7, count=1),
        ns(name="z", offset=8, datatype=7, count=1),
    ]
    msg = ns(
        width=2, height=1, point_step=12, is_bigendian=False,
        fields=fields, data=pts.tobytes(),
    )
    arr = cv.convert_message(msg, "sensor_msgs/msg/PointCloud2")
    assert arr.shape == (2, 3)
    np.testing.assert_allclose(arr, pts)
    assert cv.pointcloud2_field_names(msg) == ["x", "y", "z"]


def test_pointcloud2_empty():
    msg = ns(width=0, height=0, point_step=12, is_bigendian=False, fields=[], data=b"")
    arr = cv.convert_message(msg, "sensor_msgs/msg/PointCloud2")
    assert arr.shape == (0, 3)


def test_pointcloud2_truncated_data_is_clamped():
    # 2 points' worth of header but only 1.5 points of data -> clamp to 1 point.
    pts = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    raw = pts.tobytes() + b"\x00\x00\x00\x00\x00\x00"  # 6 stray bytes
    fields = [
        ns(name="x", offset=0, datatype=7, count=1),
        ns(name="y", offset=4, datatype=7, count=1),
        ns(name="z", offset=8, datatype=7, count=1),
    ]
    msg = ns(width=2, height=1, point_step=12, is_bigendian=False, fields=fields, data=raw)
    arr = cv.convert_message(msg, "sensor_msgs/msg/PointCloud2")
    assert arr.shape == (1, 3)
    np.testing.assert_allclose(arr, pts)


def _img_msg(data: bytes, *, w: int, h: int, channels: int, encoding: str):
    return ns(width=w, height=h, step=w * channels, encoding=encoding, data=data)


def test_image_bgr8_is_reordered_to_rgb():
    # Two pixels stored BGR; the converter must return canonical RGB.
    data = bytes([10, 20, 30, 40, 50, 60])  # px0 BGR=(10,20,30), px1 BGR=(40,50,60)
    arr = cv.convert_message(_img_msg(data, w=2, h=1, channels=3, encoding="bgr8"),
                             "sensor_msgs/msg/Image")
    np.testing.assert_array_equal(arr, [[[30, 20, 10], [60, 50, 40]]])


def test_image_rgb8_is_left_untouched():
    data = bytes([10, 20, 30, 40, 50, 60])
    arr = cv.convert_message(_img_msg(data, w=2, h=1, channels=3, encoding="rgb8"),
                             "sensor_msgs/msg/Image")
    np.testing.assert_array_equal(arr, [[[10, 20, 30], [40, 50, 60]]])


def test_image_bgra8_is_reordered_to_rgba():
    data = bytes([10, 20, 30, 40])  # one pixel BGRA
    arr = cv.convert_message(_img_msg(data, w=1, h=1, channels=4, encoding="bgra8"),
                             "sensor_msgs/msg/Image")
    np.testing.assert_array_equal(arr, [[[30, 20, 10, 40]]])


def test_image_mono8_stays_2d():
    data = bytes([10, 20, 30, 40])
    arr = cv.convert_message(_img_msg(data, w=2, h=2, channels=1, encoding="mono8"),
                             "sensor_msgs/msg/Image")
    np.testing.assert_array_equal(arr, [[10, 20], [30, 40]])


# -- Image encodings beyond 8-bit -------------------------------------------
# The decoder used to read every Image as uint8 with step // width channels:
# a 32FC1 depth image came back as (H, W, 4) uint8 bytes.

def _raw_img(arr: np.ndarray, encoding: str, *, bigendian: bool = False, pad: int = 0):
    """An Image message holding ``arr`` (H, W[, C]) with optional row padding."""
    h, w = arr.shape[:2]
    a = arr.astype(arr.dtype.newbyteorder(">" if bigendian else "<"))
    rows = a.reshape(h, -1).view(np.uint8).reshape(h, -1)
    rows = np.concatenate([rows, np.zeros((h, pad), np.uint8)], axis=1)
    return ns(width=w, height=h, step=rows.shape[1], encoding=encoding,
              is_bigendian=int(bigendian), data=rows.tobytes())


def _decode(msg):
    return cv.convert_message(msg, "sensor_msgs/msg/Image")


@pytest.mark.parametrize("encoding, dtype, channels", [
    ("32FC1", np.float32, 1),
    ("32SC1", np.int32, 1),
    ("16UC1", np.uint16, 1),
    ("mono16", np.uint16, 1),
    ("16SC1", np.int16, 1),
    ("64FC1", np.float64, 1),
    ("32FC3", np.float32, 3),
    ("8UC3", np.uint8, 3),
    ("8SC1", np.int8, 1),
    ("16UC4", np.uint16, 4),
])
def test_image_decoded_with_encoding_dtype_and_channels(encoding, dtype, channels):
    rng = np.random.default_rng(0)
    shape = (3, 5) if channels == 1 else (3, 5, channels)
    arr = (rng.standard_normal(shape) * 1000).astype(dtype)
    out = _decode(_raw_img(arr, encoding))
    assert out.dtype == np.dtype(dtype)
    assert out.shape == shape
    np.testing.assert_array_equal(out, arr)


def test_image_32fc1_keeps_inf_and_nan():
    arr = np.array([[1.5, np.inf], [np.nan, 0.25]], dtype=np.float32)
    out = _decode(_raw_img(arr, "32FC1"))
    np.testing.assert_array_equal(out, arr)


def test_image_big_endian_is_byteswapped():
    arr = np.array([[1, 2], [300, 65535]], dtype=np.uint16)
    out = _decode(_raw_img(arr, "16UC1", bigendian=True))
    assert out.dtype.isnative
    np.testing.assert_array_equal(out, arr)


def test_image_row_padding_is_dropped():
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    out = _decode(_raw_img(arr, "32FC1", pad=4))
    np.testing.assert_array_equal(out, arr)


def test_image_bgr16_is_reordered_to_rgb():
    arr = np.array([[[1, 2, 3]]], dtype=np.uint16)
    out = _decode(_raw_img(arr, "bgr16"))
    np.testing.assert_array_equal(out, [[[3, 2, 1]]])
    assert out.dtype == np.uint16


@pytest.mark.parametrize("encoding", ["yuv422", "nv21", "", "made_up"])
def test_image_unsupported_encoding_raises(encoding):
    msg = _raw_img(np.zeros((2, 2), np.uint8), "mono8")
    msg.encoding = encoding
    with pytest.raises(ValueError, match="encoding"):
        _decode(msg)


def test_image_step_too_small_for_encoding_raises():
    # A 32FC1 header on a buffer sized for mono8 must not be reinterpreted.
    msg = _raw_img(np.zeros((2, 4), np.uint8), "mono8")
    msg.encoding = "32FC1"
    with pytest.raises(ValueError):
        _decode(msg)


def test_image_truncated_data_raises():
    msg = _raw_img(np.zeros((2, 2), np.float32), "32FC1")
    msg.data = msg.data[:-1]
    with pytest.raises(ValueError):
        _decode(msg)


# -- PointCloud2 field layouts the decoder cannot read faithfully ------------

def _cloud(fields, point_step, data):
    return ns(width=1, height=1, point_step=point_step, is_bigendian=False,
              fields=fields, data=data)


def test_pointcloud2_unknown_datatype_raises():
    # An unknown datatype used to be read as float32 whatever its size.
    msg = _cloud([ns(name="x", offset=0, datatype=42, count=1)], 4, b"\x00" * 4)
    with pytest.raises(ValueError, match="datatype"):
        cv.convert_message(msg, "sensor_msgs/msg/PointCloud2")


def test_pointcloud2_multi_count_field_raises():
    # count > 1 used to keep only the first element of the field.
    vals = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    msg = _cloud([ns(name="v", offset=0, datatype=7, count=3)], 12, vals.tobytes())
    with pytest.raises(ValueError, match="count"):
        cv.convert_message(msg, "sensor_msgs/msg/PointCloud2")


# -- MNT images.tar holds 8-bit JPEGs only -----------------------------------

def test_mnt_images_refuse_non_uint8(tmp_path):
    # A float depth image used to be clipped to [0, 255] and cast to uint8.
    from apairo_extractor.export_mnt import _write_images

    depth = np.full((4, 4), 12.5, dtype=np.float32)
    with pytest.raises(ValueError, match="uint8"):
        _write_images(tmp_path, [depth], None)
    assert not (tmp_path / "images.tar").exists()
