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


def read_bag_info(path: Path) -> BagInfo:
    """Read bag metadata by parsing only the index section.

    Reads connections and chunk_infos from the index at the end of the bag.
    Does NOT seek into chunk data, so performance is O(n_topics + n_chunks),
    not O(n_messages). Falls back to rosbags Reader on parsing errors.
    """
    try:
        return _fast_read(path)
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


# ── Helpers ────────────────────────────────────────────────────────────────────


def find_bags(directory: Path) -> list[Path]:
    return sorted(directory.rglob("*.bag"))


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
