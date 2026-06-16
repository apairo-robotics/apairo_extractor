"""Tests for bag metadata reading and discovery (ROS1 fast path + ROS2)."""
from __future__ import annotations

import pytest

from apairo_extractor.bag import (
    compute_topic_coverage,
    find_bags,
    is_ros1_bag,
    read_bag_info,
    topic_to_dir,
    topics_in_bag,
)
from apairo_extractor.converters import msgtype_short

from conftest import (
    CHATTER_COUNT,
    CHATTER_TOPIC,
    IMU_COUNT,
    IMU_TOPIC,
    LIDAR_COUNT,
    LIDAR_TOPIC,
    write_tiny_bag,
)


def test_read_bag_info_topics_and_counts(tiny_bag):
    info = read_bag_info(tiny_bag)

    assert info.message_count == LIDAR_COUNT + IMU_COUNT + CHATTER_COUNT
    assert topics_in_bag(info) == {LIDAR_TOPIC, IMU_TOPIC, CHATTER_TOPIC}

    by_name = {t.name: t for t in info.topics}
    assert by_name[LIDAR_TOPIC].msgcount == LIDAR_COUNT
    assert by_name[IMU_TOPIC].msgcount == IMU_COUNT
    assert msgtype_short(by_name[LIDAR_TOPIC].msgtype) == "PointCloud2"
    assert msgtype_short(by_name[IMU_TOPIC].msgtype) == "Imu"


def test_read_bag_info_duration_and_size(tiny_bag):
    info = read_bag_info(tiny_bag)
    # Timestamps span 1.0s .. 1.3s (lidar/imu interleaved at 0.1s steps).
    assert info.duration_s == pytest.approx(0.3, abs=0.05)
    assert info.size_bytes > 0
    assert info.size_mb == pytest.approx(info.size_bytes / 1e6)


def test_topics_sorted(tiny_bag):
    info = read_bag_info(tiny_bag)
    names = [t.name for t in info.topics]
    assert names == sorted(names)


def test_is_ros1_bag(tiny_bag, fmt):
    assert is_ros1_bag(tiny_bag) == (fmt == "ros1")


def test_read_bag_info_invalid(tmp_path):
    bad = tmp_path / "broken.bag"
    bad.write_bytes(b"not a bag at all")
    with pytest.raises(ValueError):
        read_bag_info(bad)


def test_find_bags_discovers_both_formats(tmp_path):
    write_tiny_bag(tmp_path / "a.bag", "ros1")
    write_tiny_bag(tmp_path / "nested" / "b.bag", "ros1")
    write_tiny_bag(tmp_path / "ros2_a", "ros2")

    found = find_bags(tmp_path)
    names = {p.name for p in found}
    assert names == {"a.bag", "b.bag", "ros2_a"}


def test_find_bags_ignores_output_metadata(tmp_path):
    # An output sequence dir has metadata.yaml but no .db3/.mcap -> not a bag.
    out = tmp_path / "seq_out"
    out.mkdir()
    (out / "metadata.yaml").write_text("bag: whatever\n")
    assert find_bags(tmp_path) == []


def test_topic_to_dir():
    assert topic_to_dir("/ouster/points") == "ouster_points"
    assert topic_to_dir("/imu") == "imu"
    assert topic_to_dir("/") == "unknown"


def test_compute_topic_coverage(tiny_bag, tmp_path):
    full = read_bag_info(tiny_bag)
    # A second bag missing the chatter topic -> chatter becomes partial.
    partial_path = write_tiny_bag(tmp_path / "second.bag", "ros1")
    second = read_bag_info(partial_path)
    second.topics = [t for t in second.topics if t.name != CHATTER_TOPIC]

    common, partial = compute_topic_coverage([full, second])
    assert IMU_TOPIC in common and LIDAR_TOPIC in common
    assert CHATTER_TOPIC in partial
