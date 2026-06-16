"""Tests for the dataset-level .apairo/dataset.yaml init written on extraction."""
from __future__ import annotations

import yaml

from apairo_extractor.bag import read_bag_info
from apairo_extractor.converters import is_supported_msgtype
from apairo_extractor.runner import run_extraction

from conftest import CHATTER_TOPIC, IMU_TOPIC, LIDAR_TOPIC, write_tiny_bag


def _two_bags(tmp_path, fmt):
    paths = [write_tiny_bag(tmp_path / (f"{n}.bag" if fmt == "ros1" else n), fmt)
             for n in ("a", "b")]
    return [read_bag_info(p) for p in paths]


def test_is_supported_msgtype():
    assert is_supported_msgtype("sensor_msgs/msg/PointCloud2")
    assert is_supported_msgtype("sensor_msgs/msg/Imu")
    assert not is_supported_msgtype("std_msgs/msg/String")


def test_manifest_written_on_kitti_extraction(tmp_path, fmt):
    bags = _two_bags(tmp_path / "bags", fmt)
    out = tmp_path / "my_dataset"
    run_extraction(bags, [LIDAR_TOPIC, IMU_TOPIC], out, workers=1)

    manifest = yaml.safe_load((out / ".apairo" / "dataset.yaml").read_text())
    assert manifest["version"] == 1
    assert manifest["name"] == "my_dataset"          # = output folder name
    assert manifest["sequences"] == ["a", "b"]       # = bag stems
    assert set(manifest["channels"]) == {"lidar", "imu"}
    assert all(c["kind"] == "raw" for c in manifest["channels"].values())


def test_manifest_excludes_unsupported_topics(tmp_path, fmt):
    bags = _two_bags(tmp_path / "bags", fmt)
    out = tmp_path / "ds"
    # /chatter is a String → unsupported, must not appear as a channel.
    run_extraction(bags, [LIDAR_TOPIC, CHATTER_TOPIC], out, workers=1)

    manifest = yaml.safe_load((out / ".apairo" / "dataset.yaml").read_text())
    assert set(manifest["channels"]) == {"lidar"}
    assert "chatter" not in manifest["channels"]


def test_manifest_sequences_only_count_extracted(tmp_path, fmt):
    bags = _two_bags(tmp_path / "bags", fmt)
    out = tmp_path / "ds"
    out.mkdir(parents=True)
    # A stray subdir without .apairo must not be listed as a sequence.
    (out / "not_a_sequence").mkdir()
    run_extraction(bags, [LIDAR_TOPIC], out, workers=1)

    manifest = yaml.safe_load((out / ".apairo" / "dataset.yaml").read_text())
    assert manifest["sequences"] == ["a", "b"]
