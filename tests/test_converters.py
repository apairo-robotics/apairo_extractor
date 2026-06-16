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
