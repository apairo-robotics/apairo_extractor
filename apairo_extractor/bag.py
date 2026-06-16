from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TopicInfo:
    name: str
    msgtype: str
    msgcount: int


@dataclass
class BagInfo:
    path: Path
    size_bytes: int
    duration_s: float
    message_count: int
    topics: list[TopicInfo]

    @property
    def size_mb(self) -> float:
        return self.size_bytes / 1e6


# ── Fast metadata reader ───────────────────────────────────────────────────────
# rosbags.Reader.open() seeks to every chunk to build a per-message index,
# which is O(n_chunks) disk seeks and kills performance on large bags.
# We only need CONNECTION and CHUNK_INFO records from the index section.


def _read_fields(f) -> dict[str, bytes]:
    """Read a ROS1 header block: uint32 len + key=value pairs."""
    raw = f.read(struct.unpack('<I', f.read(4))[0])
    fields: dict[str, bytes] = {}
    pos = 0
    while pos < len(raw):
        flen = struct.unpack_from('<I', raw, pos)[0]
        pos += 4
        kv = raw[pos : pos + flen]
        pos += flen
        sep = kv.index(b'=')
        fields[kv[:sep].decode('ascii', errors='replace')] = kv[sep + 1 :]
    return fields


def _u32(b: bytes) -> int:
    return struct.unpack('<I', b)[0]


def _u64(b: bytes) -> int:
    return struct.unpack('<Q', b)[0]


def _time_ns(b: bytes) -> int:
    secs, nsecs = struct.unpack('<II', b)
    return secs * 1_000_000_000 + nsecs


def is_ros1_bag(path: Path) -> bool:
    """A ROS1 bag is a single ``*.bag`` file; a ROS2 bag is a directory."""
    return path.is_file() and path.suffix == '.bag'


def read_bag_info(path: Path) -> BagInfo:
    """Read bag metadata, dispatching on bag format.

    ROS1 ``*.bag`` files use a fast index-only parser: connections and
    chunk_infos are read from the index at the end of the file, without seeking
    into chunk data, so performance is O(n_topics + n_chunks), not
    O(n_messages). On any parsing error it falls back to the generic reader.

    ROS2 bags (directories) and the fallback both use ``AnyReader``, whose
    metadata (per-connection message counts, duration) is itself cheap — it
    comes from the bag's own ``metadata.yaml`` / index, not a message scan.
    """
    if is_ros1_bag(path):
        try:
            return _fast_read(path)
        except Exception:
            pass  # fall through to the generic reader
    try:
        return _reader_read(path)
    except Exception as exc:
        raise ValueError(f'Could not read bag metadata: {exc}') from exc


def _fast_read(path: Path) -> BagInfo:
    size = path.stat().st_size

    with open(path, 'rb') as f:
        # Magic line: "#ROSBAG V2.0\n"
        magic = f.readline()
        if not magic.startswith(b'#ROSBAG V2.0'):
            raise ValueError(f'Not a ROS1 v2.0 bag: {path.name}')

        # Bag header record
        hdr = _read_fields(f)
        index_pos   = _u64(hdr['index_pos'])
        conn_count  = _u32(hdr['conn_count'])
        chunk_count = _u32(hdr['chunk_count'])

        if index_pos == 0:
            raise ValueError(f'Bag not indexed (interrupted recording?): {path.name}')

        # Jump straight to the index
        f.seek(index_pos)

        # Read CONNECTION records → topic name + message type
        conn_map: dict[int, tuple[str, str]] = {}  # conn_id → (topic, msgtype)
        for _ in range(conn_count):
            hdr  = _read_fields(f)               # record header: op, conn, topic
            data = _read_fields(f)               # data section: type, md5sum, msgdef, …
            conn_id  = _u32(hdr['conn'])
            topic    = hdr['topic'].decode('utf-8', errors='replace')
            msgtype  = data.get('type', b'unknown').decode('utf-8', errors='replace')
            conn_map[conn_id] = (topic, msgtype)

        # Read CHUNK_INFO records → timestamps + per-topic message counts
        # Format: header(ver, chunk_pos, count, start_time, end_time)
        #         + uint32 data_len (skipped) + count×(conn_id uint32, msg_count uint32)
        min_ns: int = 2**63 - 1
        max_ns: int = 0
        topic_counts: dict[str, int]    = {}
        topic_msgtypes: dict[str, str]  = {}

        for _ in range(chunk_count):
            hdr   = _read_fields(f)
            count = _u32(hdr['count'])

            if count:
                start_ns = _time_ns(hdr['start_time'])
                end_ns   = _time_ns(hdr['end_time'])
                min_ns   = min(min_ns, start_ns)
                max_ns   = max(max_ns, end_ns)

            f.seek(4, 1)  # skip data_len uint32

            for _ in range(count):
                cid       = struct.unpack('<I', f.read(4))[0]
                msg_count = struct.unpack('<I', f.read(4))[0]
                if cid in conn_map:
                    topic, msgtype = conn_map[cid]
                    topic_counts[topic]    = topic_counts.get(topic, 0) + msg_count
                    topic_msgtypes[topic]  = msgtype

    duration_s = max(0.0, (max_ns - min_ns) / 1e9) if chunk_count else 0.0
    total_msgs = sum(topic_counts.values())
    topics = sorted(
        [TopicInfo(t, topic_msgtypes[t], topic_counts[t]) for t in topic_counts],
        key=lambda t: t.name,
    )
    return BagInfo(
        path=path, size_bytes=size,
        duration_s=duration_s, message_count=total_msgs,
        topics=topics,
    )


# ── Generic reader (ROS2 + ROS1 fallback) ───────────────────────────────────────


def _path_size(path: Path) -> int:
    """Byte size of a bag: file size for ROS1, sum of files for a ROS2 dir."""
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _reader_read(path: Path) -> BagInfo:
    """Read metadata via rosbags AnyReader (auto-detects ROS1/ROS2).

    Connection message counts and duration come from the bag index / metadata,
    so this does not scan messages.
    """
    from rosbags.highlevel import AnyReader

    topic_counts: dict[str, int]   = {}
    topic_msgtypes: dict[str, str] = {}
    with AnyReader([path]) as reader:
        for conn in reader.connections:
            topic_counts[conn.topic]   = topic_counts.get(conn.topic, 0) + conn.msgcount
            topic_msgtypes[conn.topic] = conn.msgtype
        duration_s = max(0.0, reader.duration / 1e9) if topic_counts else 0.0

    topics = sorted(
        [TopicInfo(t, topic_msgtypes[t], topic_counts[t]) for t in topic_counts],
        key=lambda t: t.name,
    )
    return BagInfo(
        path=path, size_bytes=_path_size(path),
        duration_s=duration_s, message_count=sum(topic_counts.values()),
        topics=topics,
    )


# ── Helpers ────────────────────────────────────────────────────────────────────


def find_bags(directory: Path) -> list[Path]:
    """Find ROS1 (``*.bag`` files) and ROS2 (directories) bags under ``directory``.

    A ROS2 bag is identified by a ``metadata.yaml`` alongside a storage file
    (``*.db3`` or ``*.mcap``); the storage-file check avoids matching the
    ``metadata.yaml`` this tool writes into its own output sequence dirs.
    """
    ros1 = list(directory.rglob("*.bag"))
    ros2 = [
        meta.parent
        for meta in directory.rglob("metadata.yaml")
        if any(meta.parent.glob("*.db3")) or any(meta.parent.glob("*.mcap"))
    ]
    return sorted(set(ros1 + ros2))


def topic_to_dir(topic: str) -> str:
    """'/ouster/points' → 'ouster_points'"""
    return topic.lstrip("/").replace("/", "_") or "unknown"


def compute_topic_coverage(bags: list[BagInfo]) -> tuple[list[str], list[str]]:
    """Return (common_topics, partial_topics) sorted by name."""
    if not bags:
        return [], []
    topic_bag_count: dict[str, int] = {}
    for bag in bags:
        for t in bag.topics:
            topic_bag_count[t.name] = topic_bag_count.get(t.name, 0) + 1
    n = len(bags)
    common  = sorted(t for t, c in topic_bag_count.items() if c == n)
    partial = sorted(t for t, c in topic_bag_count.items() if c < n)
    return common, partial


def get_topic_info(bags: list[BagInfo], topic: str) -> TopicInfo | None:
    for bag in bags:
        for t in bag.topics:
            if t.name == topic:
                return t
    return None


def topics_in_bag(bag: BagInfo) -> set[str]:
    return {t.name for t in bag.topics}
