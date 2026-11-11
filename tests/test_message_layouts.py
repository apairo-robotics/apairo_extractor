"""Messages are laid out as their header says: no value is lost to a cast, no
padding read as data, and what cannot be laid out is refused."""
from __future__ import annotations

import io
from types import SimpleNamespace as ns

import numpy as np
import pytest

from apairo_ros import converters as cv

PC2 = "sensor_msgs/msg/PointCloud2"
IMG = "sensor_msgs/msg/Image"


def _cloud(dtype, *, width=None, height=1, row_pad=0, counts=None):
    """A PointCloud2 from a structured array, optionally padded per row."""
    counts = counts or {}
    fields, datatypes = [], {v: k for k, v in cv._PC2_DTYPE.items()}
    for name in dtype.names:
        base = dtype[name].base
        fields.append(ns(name=name, offset=dtype.fields[name][1],
                         datatype=datatypes[base.type], count=counts.get(name, 1)))
    def make(points):
        w = width or len(points)
        rows = points.reshape(height, w)
        data = b"".join(r.tobytes() + b"\xff" * row_pad for r in rows)
        return ns(width=w, height=height, point_step=dtype.itemsize,
                  row_step=w * dtype.itemsize + row_pad, is_bigendian=False,
                  fields=fields, data=data)
    return make


def test_wide_fields_keep_their_values():
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("t", "<u4"), ("stamp", "<f8")])
    points = np.array([(1, 2, 3, 16777217, 1700000000.123456),
                       (4, 5, 6, 4000000001, 1700000000.223456)], dtype=dtype)
    out = cv.convert_message(_cloud(dtype)(points), PC2)
    assert out.dtype == np.float64  # float32 would round both columns
    assert out[:, 3].tolist() == [16777217, 4000000001]
    assert out[:, 4].tolist() == [1700000000.123456, 1700000000.223456]


def test_an_all_float32_cloud_stays_float32():
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4")])
    out = cv.convert_message(_cloud(dtype)(np.ones(3, dtype=dtype)), PC2)
    assert out.dtype == np.float32 and out.shape == (3, 4)


def test_a_field_with_a_count_gives_one_column_per_value():
    dtype = np.dtype([("x", "<f4"), ("normal", "<f4", (3,))])
    points = np.array([(1.0, (0.1, 0.2, 0.3)), (2.0, (0.4, 0.5, 0.6))], dtype=dtype)
    msg = _cloud(dtype, counts={"normal": 3})(points)
    out = cv.convert_message(msg, PC2)
    np.testing.assert_allclose(out, [[1, 0.1, 0.2, 0.3], [2, 0.4, 0.5, 0.6]], rtol=1e-6)
    assert cv.pointcloud2_field_names(msg) == ["x", "normal_0", "normal_1", "normal_2"]


def test_row_padding_of_an_organized_cloud_is_skipped():
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
    points = np.arange(1, 13, dtype=np.float32).view(dtype)
    out = cv.convert_message(_cloud(dtype, width=2, height=2, row_pad=8)(points), PC2)
    np.testing.assert_array_equal(out, np.arange(1, 13, dtype=np.float32).reshape(4, 3))


def test_an_empty_cloud_has_its_fields_width():
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4")])
    out = cv.convert_message(_cloud(dtype, width=0)(np.zeros(0, dtype=dtype)), PC2)
    assert out.shape == (0, 4)


def _image(array, encoding, *, pad=0, bigendian=0):
    h, w = array.shape[:2]
    row = array.reshape(h, -1)
    data = b"".join(r.tobytes() + b"\x00" * pad for r in row)
    return ns(height=h, width=w, encoding=encoding, is_bigendian=bigendian,
              step=row.shape[1] * array.dtype.itemsize + pad, data=data)


@pytest.mark.parametrize(
    "encoding, dtype, shape",
    [("16UC1", np.uint16, (2, 3)), ("mono16", np.uint16, (2, 3)),
     ("32FC1", np.float32, (2, 3)), ("mono8", np.uint8, (2, 3)),
     ("rgb8", np.uint8, (2, 3, 3)), ("rgba8", np.uint8, (2, 3, 4))],
)
def test_an_image_keeps_the_type_its_encoding_names(encoding, dtype, shape):
    array = (np.arange(np.prod(shape)).reshape(shape) * 1000 % 60000).astype(dtype)
    out = cv.convert_message(_image(array, encoding), IMG)
    assert out.dtype == dtype and out.shape == shape
    np.testing.assert_array_equal(out, array)


def test_image_row_padding_is_skipped():
    array = np.arange(1, 19, dtype=np.uint8).reshape(2, 3, 3)
    out = cv.convert_message(_image(array, "rgb8", pad=3), IMG)
    np.testing.assert_array_equal(out, array)
    grey = np.arange(10, dtype=np.uint8).reshape(2, 5)
    np.testing.assert_array_equal(cv.convert_message(_image(grey, "mono8", pad=3), IMG), grey)


def test_bgr16_is_stored_as_rgb_and_big_endian_is_read():
    bgr = np.array([[[1, 2, 3]]], dtype=np.uint16)
    np.testing.assert_array_equal(cv.convert_message(_image(bgr, "bgr16"), IMG), [[[3, 2, 1]]])
    depth = np.array([[1000, 2000]], dtype=">u2")
    out = cv.convert_message(_image(depth, "16UC1", bigendian=1), IMG)
    assert out.tolist() == [[1000, 2000]]


def test_an_unknown_encoding_or_a_short_image_is_refused():
    array = np.zeros((2, 3), np.uint8)
    with pytest.raises(ValueError, match="encoding 'weird' is not one"):
        cv.convert_message(_image(array, "weird"), IMG)
    msg = _image(array, "mono8")
    msg.data = msg.data[:-1]
    with pytest.raises(ValueError, match="byte"):
        cv.convert_message(msg, IMG)


def test_compressed_depth_skips_its_header():
    image = pytest.importorskip("PIL.Image")
    depth = np.array([[1000, 2000, 3000], [4000, 5000, 60000]], dtype=np.uint16)
    buf = io.BytesIO()
    image.fromarray(depth).save(buf, format="PNG")
    msg = ns(format="16UC1; compressedDepth", data=b"\x00" * 12 + buf.getvalue())
    out = cv.convert_message(msg, "sensor_msgs/msg/CompressedImage")
    np.testing.assert_array_equal(out, depth)
    with pytest.raises(ValueError, match="could not be decoded"):
        cv.convert_message(ns(format="png", data=b"not an image"),
                           "sensor_msgs/msg/CompressedImage")


def test_a_sensor_msgs_pointcloud_is_converted():
    msg = ns(points=[ns(x=1.0, y=2.0, z=3.0), ns(x=4.0, y=5.0, z=6.0)],
             channels=[ns(name="intensity", values=[0.5, 0.25])])
    out = cv.convert_message(msg, "sensor_msgs/msg/PointCloud")
    np.testing.assert_allclose(out, [[1, 2, 3, 0.5], [4, 5, 6, 0.25]])
    assert cv.is_supported_msgtype("sensor_msgs/msg/PointCloud")
