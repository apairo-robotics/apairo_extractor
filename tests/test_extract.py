"""End-to-end tests for the streaming extractor (ROS1 + ROS2)."""
from __future__ import annotations

import numpy as np
import pytest
import yaml

from apairo_extractor.bag import read_bag_info
from apairo_extractor.extract import extract_bag

from conftest import (
    CLOUD_POINTS,
    IMU_COUNT,
    IMU_TOPIC,
    LIDAR_COUNT,
    LIDAR_TOPIC,
    CHATTER_TOPIC,
)


def _extract(tiny_bag, tmp_path, topics, progress=None):
    info = read_bag_info(tiny_bag)
    out = tmp_path / "out"
    return info, extract_bag(info, topics, out, progress_cb=progress)


def test_frame_topic_written_as_per_frame_npy(tiny_bag, tmp_path):
    _, (seq_dir, skipped) = _extract(tiny_bag, tmp_path, [LIDAR_TOPIC])
    lidar_dir = seq_dir / "lidar"

    npys = sorted(lidar_dir.glob("[0-9]*.npy"))
    assert [p.name for p in npys] == [f"{i:06d}.npy" for i in range(LIDAR_COUNT)]
    for p in npys:
        np.testing.assert_allclose(np.load(p), CLOUD_POINTS)

    timestamps = np.loadtxt(lidar_dir / "timestamps.txt")
    assert timestamps.shape == (LIDAR_COUNT,)
    assert np.all(np.diff(timestamps) > 0)

    meta = yaml.safe_load((lidar_dir / "metadata.yaml").read_text())
    assert meta["count"] == LIDAR_COUNT
    assert meta["fields"] == ["x", "y", "z"]


def test_seq_topic_written_as_single_stacked_npy(tiny_bag, tmp_path):
    _, (seq_dir, skipped) = _extract(tiny_bag, tmp_path, [IMU_TOPIC])
    imu_dir = seq_dir / "imu"

    # One stacked array, not per-frame files.
    assert not list(imu_dir.glob("[0-9]*.npy"))
    stacked = np.load(imu_dir / "imu.npy")
    assert stacked.shape == (IMU_COUNT, 10)  # Imu -> 10-vector
    # linear_acceleration was (1,2,3) for every message.
    np.testing.assert_allclose(stacked[:, 0:3], np.tile([1.0, 2.0, 3.0], (IMU_COUNT, 1)))

    timestamps = np.loadtxt(imu_dir / "timestamps.txt")
    assert timestamps.shape == (IMU_COUNT,)


def test_unsupported_and_missing_topics_are_skipped(tiny_bag, tmp_path):
    _, (seq_dir, skipped) = _extract(
        tiny_bag, tmp_path, [LIDAR_TOPIC, CHATTER_TOPIC, "/does_not_exist"]
    )
    assert CHATTER_TOPIC in skipped          # unsupported msgtype
    assert "/does_not_exist" in skipped      # not in bag
    assert not (seq_dir / "chatter").exists()
    assert (seq_dir / "lidar").exists()      # supported topic still extracted


def test_progress_callback_reaches_total(tiny_bag, tmp_path):
    calls = []
    _extract(tiny_bag, tmp_path, [LIDAR_TOPIC, IMU_TOPIC], progress=lambda d, t: calls.append((d, t)))
    assert calls, "progress callback never fired"
    done, total = calls[-1]
    assert done == total == LIDAR_COUNT + IMU_COUNT


def test_sequence_metadata_written(tiny_bag, tmp_path):
    info, (seq_dir, _) = _extract(tiny_bag, tmp_path, [LIDAR_TOPIC, IMU_TOPIC])
    meta = yaml.safe_load((seq_dir / "metadata.yaml").read_text())
    assert meta["message_count"] == info.message_count
    assert set(meta["channels"]) == {"lidar", "imu"}
    assert meta["bag"] == str(info.path)


def test_channel_frame_id_recorded(tiny_bag, tmp_path):
    # The message header.frame_id is recorded as the channel's `frame`.
    from apairo.core.config import read_config

    _, (seq_dir, _) = _extract(tiny_bag, tmp_path, [LIDAR_TOPIC, IMU_TOPIC])
    channels = read_config(seq_dir)["channels"]
    assert channels["lidar"]["frame"] == "base"   # frame topic (npys)
    assert channels["imu"]["frame"] == "base"      # seq topic (npy)


def test_extract_is_idempotent_across_sessions(tiny_bag, tmp_path):
    # Two passes (e.g. resuming) must not corrupt output or channel registration.
    info = read_bag_info(tiny_bag)
    out = tmp_path / "out"
    extract_bag(info, [LIDAR_TOPIC], out)
    seq_dir, _ = extract_bag(info, [IMU_TOPIC], out)

    meta = yaml.safe_load((seq_dir / "metadata.yaml").read_text())
    assert set(meta["channels"]) == {"lidar", "imu"}
