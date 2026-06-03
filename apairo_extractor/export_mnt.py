"""Export ROS bags or KITTI sequences to MNT/zarr layout.

Two entry points:
  extract_bag_to_mnt() — bag → MNT directly (no intermediate KITTI files)
  export_to_mnt()      — KITTI seq dir → MNT (for "both" mode)

MNT mission layout produced::

    <mission>/
    ├── images.tar             ← Image / CompressedImage topic
    ├── points.zarr/           ← PointCloud2 (n_frames, max_pts, n_fields) float32
    ├── trajectory.zarr/
    │   ├── positions.zarr/    ← Odometry/PoseStamped x,y  float32
    │   ├── yaws.zarr/         ← yaw from quaternion        float32
    │   └── timestamps.zarr/   ← timestamps in seconds      float64
    └── metadata.yaml
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

from apairo import WRITERS, MNTDataset
from apairo.dataset.mnt.dataset import RAW_CHANNEL_PATHS

if TYPE_CHECKING:
    from apairo_extractor.bag import BagInfo


@dataclass
class MntExportConfig:
    points_channel:   str | None = None   # KITTI channel dir name → points.zarr
    image_channel:    str | None = None   # KITTI channel dir name → images.tar
    odometry_channel: str | None = None   # KITTI channel dir name → trajectory.zarr
    max_points:       int | None = None   # truncate/pad clouds (None = max across frames)
    output_dir: Path = field(default_factory=lambda: Path.home() / "mnt_exported")


# ── Public entry points ────────────────────────────────────────────────────────


def extract_bag_to_mnt(
    bag_info: BagInfo,
    mission_dir: Path,
    points_topic: str | None,
    image_topic: str | None,
    odometry_topic: str | None,
    max_points: int | None = None,
    progress_cb: Optional[Callable[[str, int, int], None]] = None,
) -> None:
    """Read a ROS bag and write directly to MNT/zarr format.

    Bypasses KITTI entirely — no intermediate .npy files are written.

    Args:
        bag_info: Bag metadata.
        mission_dir: Target MNT mission directory (will be created).
        points_topic: ROS topic name for PointCloud2 data.
        image_topic: ROS topic name for Image / CompressedImage data.
        odometry_topic: ROS topic name for Odometry / PoseStamped data.
        max_points: Truncate/pad all point clouds to this size.
                    ``None`` → use the maximum across all frames.
        progress_cb: Optional callback(stage, done, total).
    """
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore
    from apairo_extractor.converters import convert_message

    typestore = get_typestore(Stores.ROS1_NOETIC)
    mission_dir.mkdir(parents=True, exist_ok=True)

    wanted = {t for t in (points_topic, image_topic, odometry_topic) if t is not None}
    total  = sum(t.msgcount for t in bag_info.topics if t.name in wanted)
    done   = 0

    pts_frames:  list[np.ndarray]               = []
    img_frames:  list[np.ndarray]               = []
    odom_frames: list[tuple[float, np.ndarray]] = []  # (timestamp_s, array)

    with Reader(bag_info.path) as reader:
        conns = [c for c in reader.connections if c.topic in wanted]
        for conn, ts_ns, rawdata in reader.messages(connections=conns):
            try:
                msg = typestore.deserialize_ros1(rawdata, conn.msgtype)
            except Exception:
                done += 1
                if progress_cb and done % 50 == 0:
                    progress_cb("read", done, total)
                continue

            arr = convert_message(msg, conn.msgtype)
            if arr is not None:
                if conn.topic == points_topic:
                    pts_frames.append(arr)
                elif conn.topic == image_topic:
                    img_frames.append(arr)
                elif conn.topic == odometry_topic:
                    odom_frames.append((ts_ns / 1e9, arr))

            done += 1
            if progress_cb and done % 50 == 0:
                progress_cb("read", done, total)

    if progress_cb:
        progress_cb("read", total, total)

    if pts_frames:
        _write_points(mission_dir, pts_frames, max_points, progress_cb)
    if img_frames:
        _write_images(mission_dir, img_frames, progress_cb)
    if odom_frames:
        _write_trajectory(mission_dir, odom_frames, progress_cb)

    MNTDataset.init(mission_dir)
    _write_metadata(mission_dir, bag_info.path)


def export_to_mnt(
    seq_dir: Path,
    mission_dir: Path,
    config: MntExportConfig,
    progress_cb: Optional[Callable[[str, int, int], None]] = None,
) -> None:
    """Convert an existing KITTI sequence directory to MNT/zarr format.

    Use this for "both" mode where KITTI output is also kept.

    Args:
        seq_dir: Extracted KITTI sequence directory (output of extract_bag).
        mission_dir: Target MNT mission directory (will be created).
        config: Channel mapping (KITTI dir names).
        progress_cb: Optional callback(stage, done, total).
    """
    from apairo.core.config import config_exists, read_config

    mission_dir.mkdir(parents=True, exist_ok=True)

    channels_cfg: dict = {}
    if config_exists(seq_dir):
        channels_cfg = read_config(seq_dir).get("channels", {})

    if config.points_channel:
        frames = _load_kitti_npys(seq_dir / config.points_channel)
        if frames:
            _write_points(mission_dir, frames, config.max_points, progress_cb)

    if config.image_channel:
        frames = _load_kitti_npys(seq_dir / config.image_channel)
        if frames:
            _write_images(mission_dir, frames, progress_cb)

    if config.odometry_channel:
        odom_frames = _load_kitti_odom(
            seq_dir / config.odometry_channel,
            channels_cfg.get(config.odometry_channel, {}),
        )
        if odom_frames:
            _write_trajectory(mission_dir, odom_frames, progress_cb)

    MNTDataset.init(mission_dir)
    _write_metadata(mission_dir, seq_dir)


# ── Write helpers ──────────────────────────────────────────────────────────────


def _write_points(
    mission_dir: Path,
    frames: list[np.ndarray],
    max_points: int | None,
    progress_cb,
) -> None:
    n = len(frames)
    n_fields = frames[0].shape[1] if frames[0].ndim == 2 else 1

    if max_points is None:
        max_points = max(len(f) for f in frames)

    out = np.full((n, max_points, n_fields), np.nan, dtype=np.float32)
    for i, pts in enumerate(frames):
        pts = pts.astype(np.float32)
        if pts.ndim == 1:
            pts = pts[:, None]
        actual = min(len(pts), max_points)
        out[i, :actual] = pts[:actual]
        if progress_cb:
            progress_cb("points", i + 1, n)

    path = mission_dir / Path(*RAW_CHANNEL_PATHS["points"])
    WRITERS["zarr"]().write(out, path)


def _write_images(
    mission_dir: Path,
    frames: list[np.ndarray],
    progress_cb,
) -> None:
    n = len(frames)
    clipped = []
    for i, arr in enumerate(frames):
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        clipped.append(arr)
        if progress_cb:
            progress_cb("image", i + 1, n)

    stacked = np.stack(clipped)  # (N, H, W[, C])
    WRITERS["img"]().write(stacked, mission_dir / "images.tar")


def _write_trajectory(
    mission_dir: Path,
    odom_frames: list[tuple[float, np.ndarray]],
    progress_cb,
) -> None:
    """Write positions, yaws, timestamps from [x, y, z, qx, qy, qz, qw] arrays."""
    timestamps = np.array([ts for ts, _ in odom_frames], dtype=np.float64)
    data       = np.stack([arr for _, arr in odom_frames])  # (N, D)

    w = WRITERS["zarr"]()

    if data.shape[1] >= 2:
        w.write(data[:, :2].astype(np.float32),
                mission_dir / Path(*RAW_CHANNEL_PATHS["position"]))

    if data.shape[1] >= 7:
        qx, qy, qz, qw = data[:, 3], data[:, 4], data[:, 5], data[:, 6]
        yaws = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))
        w.write(yaws.astype(np.float32),
                mission_dir / Path(*RAW_CHANNEL_PATHS["yaw"]))

    w.write(timestamps, mission_dir / Path(*RAW_CHANNEL_PATHS["timestamp"]))

    if progress_cb:
        progress_cb("trajectory", len(odom_frames), len(odom_frames))


# ── KITTI loaders (for export_to_mnt only) ────────────────────────────────────


def _load_kitti_npys(channel_dir: Path) -> list[np.ndarray]:
    return [np.load(f) for f in sorted(channel_dir.glob("*.npy"))]


def _load_kitti_odom(channel_dir: Path, channel_cfg: dict) -> list[tuple[float, np.ndarray]]:
    """Load stacked odometry + timestamps.txt → list of (ts_s, array) pairs."""
    ts_path = channel_dir / "timestamps.txt"
    timestamps = np.loadtxt(ts_path) if ts_path.exists() else None

    loader = channel_cfg.get("loader", "npy")
    if loader == "npy":
        npy_files = list(channel_dir.glob("*.npy"))
        if not npy_files:
            return []
        data = np.load(npy_files[0])
        if data.ndim == 1:
            data = data[None, :]
    else:
        files = sorted(channel_dir.glob("*.npy"))
        if not files:
            return []
        data = np.stack([np.load(f) for f in files])

    n = len(data)
    ts = timestamps if timestamps is not None else np.arange(n, dtype=np.float64)
    return [(float(ts[i]), data[i]) for i in range(n)]


# ── Misc ───────────────────────────────────────────────────────────────────────


def _write_metadata(mission_dir: Path, source: Path) -> None:
    import yaml
    with open(mission_dir / "metadata.yaml", "w") as f:
        yaml.dump({"source": str(source)}, f, default_flow_style=False)
