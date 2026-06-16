"""Tests for the parallel (multi-process) extraction runner in cli.py."""
from __future__ import annotations

import yaml

from apairo_extractor.bag import read_bag_info
from apairo_extractor.runner import resolve_workers, run_extraction

from conftest import LIDAR_TOPIC, IMU_TOPIC, write_tiny_bag


def test_resolve_workers_env_override(monkeypatch):
    monkeypatch.setenv("APAIRO_EXTRACT_WORKERS", "3")
    assert resolve_workers(10) == 3
    assert resolve_workers(2) == 2          # never exceeds bag count
    monkeypatch.setenv("APAIRO_EXTRACT_WORKERS", "garbage")
    assert resolve_workers(10) == 1         # invalid → sequential
    monkeypatch.delenv("APAIRO_EXTRACT_WORKERS")
    assert resolve_workers(0) == 1          # at least one, even with no bags


def _two_bags(tmp_path, fmt):
    a = write_tiny_bag(tmp_path / ("a.bag" if fmt == "ros1" else "a"), fmt)
    b = write_tiny_bag(tmp_path / ("b.bag" if fmt == "ros1" else "b"), fmt)
    return [read_bag_info(a), read_bag_info(b)]


def _assert_extracted(out_dir):
    for stem in ("a", "b"):
        seq_dir = out_dir / stem
        assert (seq_dir / "lidar").exists()
        assert (seq_dir / "imu" / "imu.npy").exists()
        meta = yaml.safe_load((seq_dir / "metadata.yaml").read_text())
        assert set(meta["channels"]) == {"lidar", "imu"}


def test_parallel_extraction_matches_sequential(tmp_path, fmt):
    bags = _two_bags(tmp_path, fmt)
    topics = [LIDAR_TOPIC, IMU_TOPIC]

    out_par = tmp_path / "parallel"
    res_par = run_extraction(bags, topics, out_par, workers=2)
    _assert_extracted(out_par)

    out_seq = tmp_path / "sequential"
    res_seq = run_extraction(bags, topics, out_seq, workers=1)
    _assert_extracted(out_seq)

    # Both paths report a result per bag, with no errors.
    for results in (res_par, res_seq):
        assert len(results) == 2
        assert all(err is None for _, _, err in results)


def test_run_extraction_progress_callbacks(tmp_path, fmt):
    bags = _two_bags(tmp_path, fmt)
    progress_bags, done_bags = set(), []

    def on_progress(bag_id, phase, done, total):
        progress_bags.add(bag_id)

    def on_bag_done(name, skipped, err):
        done_bags.append(name)

    run_extraction(
        bags, [LIDAR_TOPIC], tmp_path / "out", workers=1,
        on_progress=on_progress, on_bag_done=on_bag_done,
    )
    assert progress_bags == {b.path.stem for b in bags}
    assert sorted(done_bags) == sorted(b.path.name for b in bags)
