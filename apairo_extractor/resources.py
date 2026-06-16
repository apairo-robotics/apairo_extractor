"""System-resource probing and worker-count recommendation.

Used by the TUI to show, before extraction, what the machine has (CPU/RAM/disk),
what it's currently using, and what N parallel workers would cost — so the user
can pick a sane ``workers`` value each run.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from apairo_extractor.bag import BagInfo
from apairo_extractor.converters import convert_message, is_frame_type

# Baseline resident memory of one worker: Python + numpy + rosbags + the reader's
# current decompressed chunk + the peak single decoded frame (a point cloud or
# image is at most a few tens of MB). Measured ~357 MB on a real point-cloud run.
BASE_RAM_MB = 350

# Per-message cost of a *buffered* sequence topic. These are tiny fixed vectors
# (Imu = 10 floats), but extract.py keeps every message as a (int, ndarray)
# tuple in a list until the end, so the dominant cost is Python object overhead,
# not the data. This is the only term that grows without bound.
SEQ_BYTES_PER_MSG = 220

# Fallback per-worker estimate when we can't see the selected topics (env/auto
# path). The TUI computes a real figure via :func:`estimate_worker_ram_mb`.
PER_WORKER_RAM_MB = 400

# Beyond this many workers the gain flattens (the per-frame small-file write
# path, not CPU or raw bandwidth, becomes the ceiling). Caps the recommendation.
WORKER_SOFT_CAP = 8

# On a rotational disk, parallel writers make the head thrash between N output
# dirs and throughput collapses — so keep parallelism low there.
HDD_WORKER_CAP = 2


@dataclass
class SystemResources:
    cpu_count: int
    load_avg_1m: float
    ram_total_mb: int
    ram_available_mb: int
    disk_free_mb: int
    disk_total_mb: int
    disk_rotational: Optional[bool] = None  # None = unknown

    @property
    def ram_used_mb(self) -> int:
        return max(0, self.ram_total_mb - self.ram_available_mb)

    @property
    def ram_used_pct(self) -> float:
        return 100.0 * self.ram_used_mb / self.ram_total_mb if self.ram_total_mb else 0.0

    @property
    def disk_kind(self) -> str:
        if self.disk_rotational is None:
            return "unknown"
        return "HDD" if self.disk_rotational else "SSD/NVMe"


def _meminfo_mb() -> tuple[int, int]:
    """(MemTotal, MemAvailable) in MB from /proc/meminfo; (0, 0) if unreadable."""
    total = avail = 0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) // 1024
                elif line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) // 1024
                if total and avail:
                    break
    except OSError:
        pass
    return total, avail


def _existing_parent(path: Path) -> Path:
    """Nearest existing ancestor of ``path`` (the dir may not exist yet)."""
    p = path
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


def _disk_rotational(path: Path) -> Optional[bool]:
    """Whether the disk backing ``path`` is rotational (HDD). None if unknown.

    Resolves the filesystem's block device via /sys and reads
    ``queue/rotational``, walking up from the partition to its parent disk
    (partitions don't carry the flag themselves).
    """
    try:
        st = os.stat(path)
        link = Path(f"/sys/dev/block/{os.major(st.st_dev)}:{os.minor(st.st_dev)}")
        dev = Path(os.path.realpath(link))
        for cand in (dev, dev.parent):
            rot = cand / "queue" / "rotational"
            if rot.exists():
                return rot.read_text().strip() == "1"
    except OSError:
        pass
    return None


def read_resources(output_dir: Path | None) -> SystemResources:
    """Snapshot CPU/RAM/disk. Disk is measured on the output filesystem."""
    total_mb, avail_mb = _meminfo_mb()
    try:
        load = os.getloadavg()[0]
    except (OSError, AttributeError):
        load = 0.0
    target = _existing_parent(output_dir) if output_dir else Path.cwd()
    try:
        du = shutil.disk_usage(target)
        disk_free = du.free // (1024 * 1024)
        disk_total = du.total // (1024 * 1024)
    except OSError:
        disk_free = disk_total = 0
    return SystemResources(
        cpu_count=os.cpu_count() or 1,
        load_avg_1m=load,
        ram_total_mb=total_mb,
        ram_available_mb=avail_mb,
        disk_free_mb=disk_free,
        disk_total_mb=disk_total,
        disk_rotational=_disk_rotational(target),
    )


def estimate_worker_ram_mb(bags: list[BagInfo], topics: list[str]) -> int:
    """Estimate one worker's peak RAM for the selected topics.

    A worker processes one bag at a time, so the buffered-sequence term is the
    worst single bag's selected seq-message count — not the sum across bags.
    Frame topics stream to disk (bounded), so they fold into ``BASE_RAM_MB``.
    """
    selected = set(topics)
    worst_seq_msgs = 0
    for bag in bags:
        seq_msgs = sum(
            t.msgcount for t in bag.topics
            if t.name in selected and not is_frame_type(t.msgtype)
        )
        worst_seq_msgs = max(worst_seq_msgs, seq_msgs)
    seq_mb = (worst_seq_msgs * SEQ_BYTES_PER_MSG) // (1024 * 1024)
    return BASE_RAM_MB + seq_mb


def _sample_message_bytes(bag: BagInfo, topics: set[str]) -> dict[str, int]:
    """Decoded ``.npy`` size of one message per topic, by reading one each.

    Returns {topic: bytes_per_message}; unsupported topics map to 0 (they are
    skipped at extraction, so they contribute nothing to the output).
    """
    from rosbags.highlevel import AnyReader

    sizes: dict[str, int] = {}
    with AnyReader([bag.path]) as reader:
        conns = [c for c in reader.connections if c.topic in topics]
        wanted = {c.topic for c in conns}
        for conn, _ts, raw in reader.messages(connections=conns):
            if conn.topic in sizes:
                continue
            try:
                arr = convert_message(reader.deserialize(raw, conn.msgtype), conn.msgtype)
                sizes[conn.topic] = int(arr.nbytes) if arr is not None else 0
            except Exception:
                sizes[conn.topic] = 0
            if sizes.keys() >= wanted:
                break
    return sizes


def estimate_output_bytes(bags: list[BagInfo], topics: list[str]) -> int:
    """Estimate the on-disk ``.npy`` output size for the selected topics.

    Per-message size is sampled once (from the first bag that carries each
    topic) and multiplied by every bag's message count — so the figure reflects
    the *selected channels*, not the whole bag. Falls back to the summed bag
    size if sampling fails entirely (e.g. unreadable bags).
    """
    selected = set(topics)
    per_msg: dict[str, int] = {}
    for bag in bags:
        need = ({t.name for t in bag.topics} & selected) - per_msg.keys()
        if need:
            try:
                per_msg.update(_sample_message_bytes(bag, need))
            except Exception:
                pass
    if not per_msg:  # sampling unavailable → conservative proxy
        return sum(b.size_bytes for b in bags)
    return sum(
        t.msgcount * per_msg.get(t.name, 0)
        for bag in bags for t in bag.topics if t.name in selected
    )


def estimate_output_mb(bags: list[BagInfo], topics: list[str]) -> int:
    return estimate_output_bytes(bags, topics) // (1024 * 1024)


def projected_ram_mb(workers: int, per_worker_mb: int = PER_WORKER_RAM_MB) -> int:
    """Estimated total resident memory for ``workers`` parallel extractors."""
    return workers * per_worker_mb


def recommend_workers(
    n_bags: int,
    res: SystemResources,
    per_worker_mb: int = PER_WORKER_RAM_MB,
) -> int:
    """Suggested worker count.

    Capped by cores (less one for OS/UI), free RAM / per-worker estimate, bag
    count, the soft cap, and — on a rotational disk — :data:`HDD_WORKER_CAP`,
    since parallel small-file writes thrash an HDD head.
    """
    cpu_cap = max(1, res.cpu_count - 1)
    ram_cap = max(1, res.ram_available_mb // max(1, per_worker_mb))
    rec = min(cpu_cap, ram_cap, max(1, n_bags), WORKER_SOFT_CAP)
    if res.disk_rotational:
        rec = min(rec, HDD_WORKER_CAP)
    return max(1, rec)


def worker_fit(
    workers: int,
    res: SystemResources,
    per_worker_mb: int = PER_WORKER_RAM_MB,
) -> tuple[str, str]:
    """An (``ok``/``warn``/``bad``, reason) verdict for running ``workers``."""
    ram = projected_ram_mb(workers, per_worker_mb)
    if res.ram_available_mb and ram > res.ram_available_mb:
        return "bad", "exceeds free RAM"
    if workers > res.cpu_count:
        return "bad", "more workers than cores"
    if res.disk_rotational and workers > HDD_WORKER_CAP:
        return "warn", "HDD: write contention"
    if res.ram_available_mb and ram > 0.8 * res.ram_available_mb:
        return "warn", "RAM tight"
    if workers == res.cpu_count:
        return "warn", "uses all cores"
    return "ok", ""
