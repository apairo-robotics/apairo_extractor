from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np

from apairo.core.config import register_raw_channel, register_static_transform

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
    frame_ids: dict[str, str] = {}
    # TF: dynamic edges -> one pose channel per (source, parent, child);
    # static edges (/tf_static) -> calibration (time-independent, not channels).
    tf_edges: dict[tuple[str, str, str], list[tuple[float, np.ndarray]]] = {}
    tf_static: dict[tuple[str, str], np.ndarray] = {}
    tf_topics: set[str] = set()
    tf_produced: set[str] = set()
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

            # /tf and /tf_static carry many parent→child edges per message;
            # demultiplex them into per-edge pose streams instead of one channel.
            if msgtype_short(conn.msgtype) == "TFMessage":
                tf_topics.add(topic)
                _accumulate_tf(msg, topic, tf_edges, tf_static, tf_produced)
                done += 1
                if progress_cb and done % 50 == 0:
                    progress_cb(done, total)
                continue

            # Coordinate frame of the channel, from the first message's header
            # (ROS1 leading slash stripped). Headerless messages leave it empty.
            if topic not in frame_ids:
                hdr = getattr(msg, "header", None)
                fid = getattr(hdr, "frame_id", "") if hdr is not None else ""
                fid = fid.strip() if isinstance(fid, str) else ""
                frame_ids[topic] = fid[1:] if fid.startswith("/") else fid

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
        if topic in tf_topics:
            continue  # handled as per-edge channels below
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
            register_raw_channel(seq_dir, dir_name, "npys", frame=frame_ids.get(topic) or None)

        elif topic in seq_data and seq_data[topic]:
            channel_dir.mkdir(exist_ok=True)
            frames = seq_data[topic]
            timestamps = np.array([ts / 1e9 for ts, _ in frames])
            stacked = np.stack([arr for _, arr in frames])
            np.save(channel_dir / f"{dir_name}.npy", stacked)
            np.savetxt(channel_dir / "timestamps.txt", timestamps)
            _write_channel_metadata(channel_dir, topic, msgtype, len(frames))
            register_raw_channel(seq_dir, dir_name, "npy", frame=frame_ids.get(topic) or None)

        else:
            skipped.append(topic)

    _write_tf_edges(seq_dir, tf_edges)
    _write_tf_calibration(seq_dir, tf_static)
    skipped.extend(sorted(tf_topics - tf_produced))
    _write_sequence_metadata(seq_dir, bag_info)

    return seq_dir, skipped


def _strip_slash(name: str) -> str:
    name = (name or "").strip()
    return name[1:] if name.startswith("/") else name


def _is_static_tf_topic(topic: str) -> bool:
    """A ``/tf_static``-style topic (latched static transforms)."""
    return "static" in topic.rstrip("/").rsplit("/", 1)[-1]


def _accumulate_tf(msg, topic: str, edges: dict, static: dict, produced: set) -> None:
    """Split a TFMessage into transform samples.

    Dynamic edges (``/tf``) accumulate into ``edges`` keyed by *source topic*
    (preserved verbatim, never merged/dropped). Static edges (``/tf_static``)
    are time-independent, so they go to ``static`` (latest value per edge) and
    become *calibration* rather than channels. ``produced`` collects the topics
    that yielded at least one transform.
    """
    is_static = _is_static_tf_topic(topic)
    for ts in getattr(msg, "transforms", []):
        parent = _strip_slash(ts.header.frame_id)
        child = _strip_slash(ts.child_frame_id)
        if not parent or not child:
            continue
        produced.add(topic)
        tr, q = ts.transform.translation, ts.transform.rotation
        pose = np.array([tr.x, tr.y, tr.z, q.x, q.y, q.z, q.w], dtype=np.float64)
        if is_static:
            static[(parent, child)] = pose
        else:
            stamp = ts.header.stamp
            t = stamp.sec + stamp.nanosec * 1e-9
            edges.setdefault((topic, parent, child), []).append((t, pose))


def _write_one_edge(seq_dir: Path, topic: str, parent: str, child: str, samples: list) -> None:
    """Write one dynamic transform edge as a stacked pose channel."""
    if not samples:
        return
    dir_name = f"{topic_to_dir(topic)}__{parent}__{child}".replace("/", "_")
    channel_dir = seq_dir / dir_name
    channel_dir.mkdir(exist_ok=True)
    samples = sorted(samples, key=lambda s: s[0])
    timestamps = np.array([t for t, _ in samples])
    stacked = np.stack([pose for _, pose in samples])
    np.save(channel_dir / f"{dir_name}.npy", stacked)
    np.savetxt(channel_dir / "timestamps.txt", timestamps)
    _write_channel_metadata(channel_dir, f"{topic}:{parent}->{child}",
                            "tf2_msgs/msg/TFMessage", len(samples))
    register_raw_channel(seq_dir, dir_name, "npy", transform={
        "parent": parent, "child": child, "source": topic, "format": "t_xyz_q_xyzw",
    })


def _write_tf_edges(seq_dir: Path, edges: dict) -> None:
    """Flush every dynamic transform edge to its own pose channel (faithful)."""
    for (topic, parent, child), samples in edges.items():
        _write_one_edge(seq_dir, topic, parent, child, samples)


def _pose7_to_matrix(pose) -> np.ndarray:
    """``[tx,ty,tz, qx,qy,qz,qw]`` -> 4x4 homogeneous transform."""
    tx, ty, tz, qx, qy, qz, qw = (float(v) for v in pose)
    n = qx * qx + qy * qy + qz * qz + qw * qw
    s = 0.0 if n == 0.0 else 2.0 / n
    m = np.eye(4)
    m[0, 0] = 1 - s * (qy * qy + qz * qz)
    m[0, 1] = s * (qx * qy - qz * qw)
    m[0, 2] = s * (qx * qz + qy * qw)
    m[1, 0] = s * (qx * qy + qz * qw)
    m[1, 1] = 1 - s * (qx * qx + qz * qz)
    m[1, 2] = s * (qy * qz - qx * qw)
    m[2, 0] = s * (qx * qz - qy * qw)
    m[2, 1] = s * (qy * qz + qx * qw)
    m[2, 2] = 1 - s * (qx * qx + qy * qy)
    m[0, 3], m[1, 3], m[2, 3] = tx, ty, tz
    return m


def _write_tf_calibration(seq_dir: Path, static: dict) -> None:
    """Write static transforms as extrinsics in ``.apairo/calibration.yaml``."""
    for (parent, child), pose in static.items():
        register_static_transform(seq_dir, parent, child, _pose7_to_matrix(pose))


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
