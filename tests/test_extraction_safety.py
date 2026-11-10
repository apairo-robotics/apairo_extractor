"""Extraction never mixes two sources into one output, never writes into its
input, and leaves nothing stale or half-written."""
from __future__ import annotations

import numpy as np
import pytest

from apairo_ros.bag import (
    colliding_channels,
    read_bag_info,
    sequence_name,
    topic_to_dir,
)
from apairo_ros.extract import extract_bag
from apairo_ros.runner import run_extraction

from conftest import IMU_TOPIC, LIDAR_COUNT, LIDAR_TOPIC, write_tiny_bag


def _bag(path, fmt):
    return read_bag_info(write_tiny_bag(path, fmt))


def test_two_bags_of_the_same_name_are_refused_before_any_work(tmp_path):
    bags = [_bag(tmp_path / "in" / day / "run.bag", "ros1") for day in ("day1", "day2")]
    with pytest.raises(ValueError, match="same sequence .*'run'"):
        run_extraction(bags, [LIDAR_TOPIC], tmp_path / "out", workers=1)
    assert not any((tmp_path / "out").glob("*/*"))


def test_ros2_bag_directories_keep_their_whole_name(tmp_path):
    a = write_tiny_bag(tmp_path / "run_1.0", "ros2")
    b = write_tiny_bag(tmp_path / "run_1.1", "ros2")
    assert (sequence_name(a), sequence_name(b)) == ("run_1.0", "run_1.1")
    results = run_extraction(
        [read_bag_info(a), read_bag_info(b)], [LIDAR_TOPIC], tmp_path / "out", workers=1
    )
    assert [err for _, _, err in results] == [None, None]
    assert sorted(p.name for p in (tmp_path / "out").iterdir() if p.is_dir() and p.name != ".apairo") == ["run_1.0", "run_1.1"]


def test_topics_mapping_to_one_channel_are_named():
    assert colliding_channels(["/a/b", "/a_b", "/c"]) == {"a_b": ["/a/b", "/a_b"]}
    assert colliding_channels(["/a/b", "/c"]) == {}
    with pytest.raises(ValueError, match="cannot be a channel name"):
        topic_to_dir("/..")


def test_a_ros2_bag_is_not_extracted_into_itself(tmp_path):
    bag = write_tiny_bag(tmp_path / "data" / "run_42", "ros2")
    before = (bag / "metadata.yaml").read_text()
    with pytest.raises(ValueError, match="in the bag itself"):
        extract_bag(read_bag_info(bag), [LIDAR_TOPIC], tmp_path / "data")
    assert (bag / "metadata.yaml").read_text() == before
    assert sorted(p.name for p in bag.iterdir()) == sorted(
        p.name for p in bag.iterdir() if p.suffix in (".db3", ".mcap", ".yaml")
    )
    read_bag_info(bag)  # still a readable bag


def test_re_extracting_leaves_no_frame_of_the_earlier_run(tmp_path, fmt):
    info = _bag(tmp_path / ("run.bag" if fmt == "ros1" else "run"), fmt)
    seq, _ = extract_bag(info, [LIDAR_TOPIC, IMU_TOPIC], tmp_path / "out")
    stale = seq / "lidar" / "000099.npy"
    np.save(stale, np.zeros((1, 4), np.float32))  # as a longer earlier run would leave
    extract_bag(info, [LIDAR_TOPIC, IMU_TOPIC], tmp_path / "out")
    assert not stale.exists()
    frames = sorted(p.name for p in (seq / "lidar").glob("*.npy"))
    assert len(frames) == LIDAR_COUNT == len(np.loadtxt(seq / "lidar" / "timestamps.txt"))


def test_a_failed_extraction_leaves_nothing_behind(tmp_path, monkeypatch):
    import apairo_ros.extract as extract

    info = _bag(tmp_path / "run.bag", "ros1")
    calls = {"n": 0}
    real = extract.convert_message

    def boom(msg, msgtype):
        calls["n"] += 1
        if calls["n"] == 4:
            raise RuntimeError("disk on fire")  # not a per-message ValueError
        return real(msg, msgtype)

    monkeypatch.setattr(extract, "convert_message", boom)
    with pytest.raises(RuntimeError):
        extract_bag(info, [LIDAR_TOPIC, IMU_TOPIC], tmp_path / "out")
    assert not (tmp_path / "out" / "run").exists()


def test_a_message_that_cannot_be_laid_out_is_counted_not_hidden(tmp_path, monkeypatch):
    import apairo_ros.extract as extract

    info = _bag(tmp_path / "run.bag", "ros1")
    real = extract.convert_message
    seen = {"lidar": 0}

    def flaky(msg, msgtype):
        if msgtype.endswith("PointCloud2"):
            seen["lidar"] += 1
            if seen["lidar"] == 2:
                raise ValueError("truncated cloud")
        return real(msg, msgtype)

    monkeypatch.setattr(extract, "convert_message", flaky)
    with pytest.warns(UserWarning, match="1 message.*truncated cloud"):
        seq, skipped = extract_bag(info, [LIDAR_TOPIC], tmp_path / "out")
    assert any("1 message(s) could not be read" in s for s in skipped)
    assert len(list((seq / "lidar").glob("*.npy"))) == LIDAR_COUNT - 1
    assert len(np.loadtxt(seq / "lidar" / "timestamps.txt")) == LIDAR_COUNT - 1
