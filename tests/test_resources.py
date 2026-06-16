"""Tests for system-resource probing and worker recommendation."""
from __future__ import annotations

from pathlib import Path

from apairo_extractor.bag import BagInfo, TopicInfo, read_bag_info
from apairo_extractor.resources import (
    BASE_RAM_MB,
    HDD_WORKER_CAP,
    PER_WORKER_RAM_MB,
    SEQ_BYTES_PER_MSG,
    WORKER_SOFT_CAP,
    SystemResources,
    estimate_output_bytes,
    estimate_worker_ram_mb,
    projected_ram_mb,
    read_resources,
    recommend_workers,
    worker_fit,
)

from conftest import (
    CHATTER_TOPIC,
    IMU_COUNT,
    IMU_TOPIC,
    LIDAR_COUNT,
    LIDAR_TOPIC,
)


def _res(cpu=20, ram_avail=16000, ram_total=32000, rotational=False):
    return SystemResources(
        cpu_count=cpu,
        load_avg_1m=0.0,
        ram_total_mb=ram_total,
        ram_available_mb=ram_avail,
        disk_free_mb=100_000,
        disk_total_mb=900_000,
        disk_rotational=rotational,
    )


def test_read_resources_real_machine(tmp_path):
    res = read_resources(tmp_path)
    assert res.cpu_count >= 1
    assert res.ram_total_mb > 0
    assert res.disk_total_mb > 0
    assert 0 <= res.ram_used_pct <= 100


def test_read_resources_nonexistent_output_dir_uses_parent(tmp_path):
    # Disk is measured on the nearest existing ancestor, not the (absent) dir.
    res = read_resources(tmp_path / "does" / "not" / "exist")
    assert res.disk_free_mb > 0


def test_recommend_capped_by_bag_count():
    # Plenty of CPU/RAM, but only 3 bags → at most 3 workers.
    assert recommend_workers(3, _res(cpu=20, ram_avail=64000)) == 3


def test_recommend_capped_by_soft_cap():
    # Many bags, many cores, lots of RAM → capped at the soft cap.
    assert recommend_workers(100, _res(cpu=64, ram_avail=256000)) == WORKER_SOFT_CAP


def test_recommend_capped_by_cores():
    # Leaves one core for the OS/UI.
    assert recommend_workers(100, _res(cpu=4, ram_avail=256000)) == 3


def test_recommend_capped_by_ram():
    # Only room for 2 workers' worth of RAM.
    res = _res(cpu=64, ram_avail=2 * PER_WORKER_RAM_MB + 50)
    assert recommend_workers(100, res) == 2


def test_recommend_at_least_one():
    assert recommend_workers(0, _res(cpu=1, ram_avail=10)) == 1


def test_projected_ram_scales_linearly():
    assert projected_ram_mb(5) == 5 * PER_WORKER_RAM_MB


def test_worker_fit_verdicts():
    res = _res(cpu=8, ram_avail=8000)
    assert worker_fit(2, res)[0] == "ok"
    assert worker_fit(8, res)[0] == "warn"          # all cores
    assert worker_fit(9, res)[0] == "bad"           # more than cores
    # RAM exceeded
    tight = _res(cpu=64, ram_avail=PER_WORKER_RAM_MB)
    assert worker_fit(4, tight)[0] == "bad"


# ── Rotational-disk awareness ────────────────────────────────────────────────


def test_recommend_capped_on_rotational_disk():
    # Plenty of CPU/RAM/bags, but an HDD → capped to HDD_WORKER_CAP.
    res = _res(cpu=64, ram_avail=256000, rotational=True)
    assert recommend_workers(100, res) == HDD_WORKER_CAP


def test_recommend_not_capped_on_ssd():
    res = _res(cpu=64, ram_avail=256000, rotational=False)
    assert recommend_workers(100, res) == WORKER_SOFT_CAP


def test_worker_fit_warns_on_hdd_contention():
    res = _res(cpu=16, ram_avail=64000, rotational=True)
    kind, note = worker_fit(HDD_WORKER_CAP + 1, res)
    assert kind == "warn" and "HDD" in note


def test_disk_kind_label():
    assert _res(rotational=True).disk_kind == "HDD"
    assert _res(rotational=False).disk_kind == "SSD/NVMe"
    res = _res()
    res.disk_rotational = None
    assert res.disk_kind == "unknown"


# ── Per-worker RAM estimate ──────────────────────────────────────────────────


def _bag(name, topics):
    return BagInfo(path=Path(name), size_bytes=0, duration_s=0.0,
                   message_count=0, topics=topics)


def test_estimate_ram_ignores_frame_topics():
    # Frame topics stream to disk → no buffer term, just the baseline.
    bags = [_bag("a.bag", [TopicInfo("/lidar", "sensor_msgs/msg/PointCloud2", 10_000_000)])]
    assert estimate_worker_ram_mb(bags, ["/lidar"]) == BASE_RAM_MB


def test_estimate_ram_counts_buffered_seq_topics():
    n = 5_000_000
    bags = [_bag("a.bag", [TopicInfo("/imu", "sensor_msgs/msg/Imu", n)])]
    expected = BASE_RAM_MB + (n * SEQ_BYTES_PER_MSG) // (1024 * 1024)
    assert estimate_worker_ram_mb(bags, ["/imu"]) == expected
    assert expected > BASE_RAM_MB          # this is the term that grows


def test_estimate_ram_uses_worst_single_bag_not_sum():
    # A worker handles one bag at a time → peak is the largest bag, not the sum.
    big = _bag("big.bag", [TopicInfo("/imu", "sensor_msgs/msg/Imu", 4_000_000)])
    small = _bag("small.bag", [TopicInfo("/imu", "sensor_msgs/msg/Imu", 1_000_000)])
    both = estimate_worker_ram_mb([big, small], ["/imu"])
    just_big = estimate_worker_ram_mb([big], ["/imu"])
    assert both == just_big


def test_estimate_ram_ignores_unselected_topics():
    bags = [_bag("a.bag", [
        TopicInfo("/imu", "sensor_msgs/msg/Imu", 5_000_000),
        TopicInfo("/lidar", "sensor_msgs/msg/PointCloud2", 100),
    ])]
    assert estimate_worker_ram_mb(bags, ["/lidar"]) == BASE_RAM_MB


# ── Output-size estimate (channel-aware, by sampling) ────────────────────────


def test_estimate_output_reflects_selected_channels(tiny_bag):
    bags = [read_bag_info(tiny_bag)]
    # 2 points × 3 × float32 = 24 B/frame; 10 × float64 = 80 B/imu message.
    assert estimate_output_bytes(bags, [LIDAR_TOPIC]) == 24 * LIDAR_COUNT
    assert estimate_output_bytes(bags, [IMU_TOPIC]) == 80 * IMU_COUNT
    assert estimate_output_bytes(bags, [LIDAR_TOPIC, IMU_TOPIC]) == \
        24 * LIDAR_COUNT + 80 * IMU_COUNT


def test_estimate_output_smaller_for_smaller_selection(tiny_bag):
    bags = [read_bag_info(tiny_bag)]
    assert estimate_output_bytes(bags, [IMU_TOPIC]) < \
        estimate_output_bytes(bags, [LIDAR_TOPIC, IMU_TOPIC])


def test_estimate_output_zero_for_unsupported_topic(tiny_bag):
    # An unsupported (skipped) topic produces no output.
    assert estimate_output_bytes([read_bag_info(tiny_bag)], [CHATTER_TOPIC]) == 0


def test_estimate_output_falls_back_when_unsamplable(tiny_bag):
    # Topic absent from the bag → nothing to sample → conservative bag-size proxy.
    bag = read_bag_info(tiny_bag)
    assert estimate_output_bytes([bag], ["/not_in_bag"]) == bag.size_bytes
