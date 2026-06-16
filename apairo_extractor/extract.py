from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np

from apairo.core.config import register_raw_channel

from apairo_extractor.bag import BagInfo, topic_to_dir, topics_in_bag
from apairo_extractor.converters import (
    convert_message,
    is_frame_type,
    pointcloud2_field_names,
    msgtype_short,
)


def extract_bag(
    bag_info: BagInfo,
    topics: list[str],
    output_dir: Path,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> tuple[Path, list[str]]:
    """Extract selected topics from one bag to KITTI format.

    Args:
        bag_info: Bag metadata.
        topics: ROS topic names to extract.
        output_dir: Root output directory; sequence lands in output_dir/<bag_stem>/.
        progress_cb: Optional callback(messages_done, messages_total).

    Returns:
        (seq_dir, skipped_topics) where skipped_topics were missing or unsupported.
    """
    from rosbags.highlevel import AnyReader

    seq_dir = output_dir / bag_info.path.stem
    seq_dir.mkdir(parents=True, exist_ok=True)

    bag_topic_names = topics_in_bag(bag_info)
    available = [t for t in topics if t in bag_topic_names]
    missing = [t for t in topics if t not in bag_topic_names]

    total = sum(t.msgcount for t in bag_info.topics if t.name in set(available))

    # Streaming state. Frame topics (point clouds, images, …) are the memory
    # hog, so each message is written to its own .npy as it arrives and only
    # lightweight bookkeeping (per-topic frame count + timestamps) is held in
    # RAM — keeping memory bounded regardless of bag size. Sequence topics
    # produce tiny fixed-length vectors, so they are buffered and stacked once
    # at the end (their footprint is ∝ message count, not data volume, and
    # np.stack needs the whole array anyway).
    frame_counts: dict[str, int] = {}
    frame_timestamps: dict[str, list[float]] = {}
    seq_data: dict[str, list[tuple[int, np.ndarray]]] = {}
    msgtypes: dict[str, str] = {}
    pc2_fields: dict[str, list[str]] = {}
    done = 0

    with AnyReader([bag_info.path]) as reader:
        topic_conns: dict[str, list] = {}
        for conn in reader.connections:
            if conn.topic in available:
                topic_conns.setdefault(conn.topic, []).append(conn)
                if conn.topic not in msgtypes:
                    msgtypes[conn.topic] = conn.msgtype

        all_conns = [c for conns in topic_conns.values() for c in conns]

        for conn, timestamp_ns, rawdata in reader.messages(connections=all_conns):
            topic = conn.topic
            try:
                msg = reader.deserialize(rawdata, conn.msgtype)
            except Exception:
                done += 1
                if progress_cb and done % 50 == 0:
                    progress_cb(done, total)
                continue

            arr = convert_message(msg, conn.msgtype)
            if arr is not None:
                if is_frame_type(conn.msgtype):
                    channel_dir = seq_dir / topic_to_dir(topic)
                    if topic not in frame_counts:
                        channel_dir.mkdir(exist_ok=True)
                    idx = frame_counts.get(topic, 0)
                    np.save(channel_dir / f"{idx:06d}.npy", arr)
                    frame_counts[topic] = idx + 1
                    frame_timestamps.setdefault(topic, []).append(timestamp_ns / 1e9)
                    if msgtype_short(conn.msgtype) == "PointCloud2" and topic not in pc2_fields:
                        pc2_fields[topic] = pointcloud2_field_names(msg)
                else:
                    seq_data.setdefault(topic, []).append((timestamp_ns, arr))

            done += 1
            if progress_cb and done % 50 == 0:
                progress_cb(done, total)

    if progress_cb:
        progress_cb(total, total)

    # ── Finalize on-disk channels ────────────────────────────────────────────
    # Frame .npy files are already written; here we only flush their sidecar
    # files (timestamps, metadata) and stack/write the buffered seq topics.
    skipped: list[str] = list(missing)

    for topic in available:
        dir_name = topic_to_dir(topic)
        msgtype = msgtypes.get(topic, "unknown")
        channel_dir = seq_dir / dir_name

        if frame_counts.get(topic, 0) > 0:
            timestamps = np.array(frame_timestamps[topic])
            np.savetxt(channel_dir / "timestamps.txt", timestamps)
            _write_channel_metadata(
                channel_dir, topic, msgtype,
                frame_counts[topic], pc2_fields.get(topic),
            )
            register_raw_channel(seq_dir, dir_name, "npys")

        elif topic in seq_data and seq_data[topic]:
            channel_dir.mkdir(exist_ok=True)
            frames = seq_data[topic]
            timestamps = np.array([ts / 1e9 for ts, _ in frames])
            stacked = np.stack([arr for _, arr in frames])
            np.save(channel_dir / f"{dir_name}.npy", stacked)
            np.savetxt(channel_dir / "timestamps.txt", timestamps)
            _write_channel_metadata(channel_dir, topic, msgtype, len(frames))
            register_raw_channel(seq_dir, dir_name, "npy")

        else:
            skipped.append(topic)

    _write_sequence_metadata(seq_dir, bag_info)

    return seq_dir, skipped


def _write_channel_metadata(
    channel_dir: Path,
    topic: str,
    msgtype: str,
    count: int,
    pc2_fields: list[str] | None = None,
) -> None:
    import yaml
    meta: dict = {"topic": topic, "msgtype": msgtype, "count": count}
    if pc2_fields:
        meta["fields"] = pc2_fields
    with open(channel_dir / "metadata.yaml", "w") as f:
        yaml.dump(meta, f, default_flow_style=False)


def _write_sequence_metadata(seq_dir: Path, bag_info: BagInfo) -> None:
    import yaml
    from apairo.core.config import config_exists, read_config

    channels: list[str] = []
    if config_exists(seq_dir):
        channels = sorted(read_config(seq_dir).get("channels", {}).keys())

    meta = {
        "bag": str(bag_info.path),
        "duration_s": round(bag_info.duration_s, 3),
        "size_mb": round(bag_info.size_mb, 1),
        "message_count": bag_info.message_count,
        "channels": channels,
    }
    with open(seq_dir / "metadata.yaml", "w") as f:
        yaml.dump(meta, f, default_flow_style=False)
