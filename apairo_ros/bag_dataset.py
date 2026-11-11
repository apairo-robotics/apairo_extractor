"""A rosbag read in place as an apairo dataset — no extraction step.

``BagDataset`` reads the chosen topics of a ROS 1 (``.bag``) or ROS 2
(directory, ``.db3`` / ``.mcap``) bag into an asynchronous apairo dataset.
Each message is converted exactly as :func:`~apairo_ros.extract.extract_bag`
converts it, and stamped with the same time (the bag's receive time by
default), so a bag read in place and the same bag extracted to disk give the
same samples. That makes it the source to replay a deployed chain on::

    from apairo import RawDataset, load_chain
    from apairo_ros import BagDataset

    train = RawDataset("/data/extracted", keys=["ouster_points", "odom"])
    ...                                             # build and train
    deployed = train.inference(["trav_pred"])

    bag = BagDataset("/data/run_42.mcap", topics={"ouster_points": "/ouster/points",
                                                  "odom": "/odom"})
    predictions = deployed.on(bag)                  # the bag parity test

Channel names default to the extractor's (``/ouster/points`` →
``ouster_points``), so a chain built on extracted data finds its channels.

Messages are decoded when the dataset is built and held in memory: select
the topics you need. ``/tf`` is not read here (the extractor demultiplexes it
into one channel per edge).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from apairo.dataset.stream import StreamDataset

from apairo_ros.bag import colliding_channels, topic_to_dir
from apairo_ros.converters import (
    convert_message,
    is_supported_msgtype,
    msgtype_short,
)

__all__ = ["BagDataset"]


def _header_stamp(msg) -> float | None:
    stamp = getattr(getattr(msg, "header", None), "stamp", None)
    if stamp is None:
        return None
    sec = getattr(stamp, "sec", None)
    if sec is None:  # ROS 1 Time: secs / nsecs
        sec, nsec = getattr(stamp, "secs", None), getattr(stamp, "nsecs", 0)
    else:
        nsec = getattr(stamp, "nanosec", 0)
    return None if sec is None else sec + nsec / 1e9


class BagDataset(StreamDataset):
    """The topics of a rosbag as an asynchronous apairo dataset.

    Args:
        path: A ROS 1 ``.bag`` file or a ROS 2 bag directory.
        topics: Which topics to read, and under which channel names:

            - ``None``: every topic whose message type the extractor
              converts, named as the extractor names them;
            - a list of topics, named as the extractor names them;
            - a dict ``{channel: topic}``, to choose the names (to match the
              channels a chain was trained on).
        stamp: ``"receive"`` (default) stamps a message with the time the bag
            recorded it, as the extractor does; ``"header"`` uses the
            message's ``header.stamp`` (messages are then sorted by it).

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If a requested topic is absent from the bag, has no
            message, or has a message type that cannot be converted.
    """

    def __init__(
        self,
        path: str | Path,
        topics: dict[str, str] | list[str] | None = None,
        *,
        stamp: str = "receive",
    ) -> None:
        from rosbags.highlevel import AnyReader

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No bag at {path}.")
        if stamp not in ("receive", "header"):
            raise ValueError(f"stamp must be 'receive' or 'header', got {stamp!r}.")

        with AnyReader([path]) as reader:
            msgtypes: dict[str, str] = {}
            for conn in reader.connections:
                msgtypes.setdefault(conn.topic, conn.msgtype)
            channels = self._resolve_topics(topics, msgtypes, path)
            # One topic may be read under several channel names.
            by_topic: dict[str, list[str]] = {}
            for channel, topic in channels.items():
                by_topic.setdefault(topic, []).append(channel)
            conns = [c for c in reader.connections if c.topic in by_topic]

            stamps: dict[str, list[float]] = {c: [] for c in channels}
            items: dict[str, list[np.ndarray]] = {c: [] for c in channels}
            dropped: dict[str, list] = {}  # topic -> [count, first reason]
            for conn, t_ns, raw in reader.messages(connections=conns):
                # A message that cannot be read is counted and reported, as an
                # extraction does: the two give the same samples.
                try:
                    msg = reader.deserialize(raw, conn.msgtype)
                    arr = convert_message(msg, conn.msgtype)
                except Exception as exc:
                    entry = dropped.setdefault(
                        conn.topic, [0, f"{type(exc).__name__}: {exc}"]
                    )
                    entry[0] += 1
                    continue
                if arr is None:
                    continue
                t = t_ns / 1e9
                if stamp == "header":
                    # The header's time when there is a header, a zero stamp
                    # included: the live node reads it the same way.
                    header_t = _header_stamp(msg)
                    t = t if header_t is None else header_t
                for channel in by_topic[conn.topic]:
                    stamps[channel].append(t)
                    items[channel].append(arr)
            for topic, (count, reason) in sorted(dropped.items()):
                warnings.warn(
                    f"{path.name}: {count} message(s) on {topic} could not be "
                    f"read ({reason}); the channel holds the others.",
                    UserWarning,
                    stacklevel=2,
                )

        empty = [f"{c} ({channels[c]})" for c in channels if not items[c]]
        if empty:
            raise ValueError(f"No message could be read on: {', '.join(empty)} in {path}.")

        streams = {}
        for channel in channels:
            ts = np.asarray(stamps[channel], dtype=np.float64)
            order = np.argsort(ts, kind="stable")
            streams[channel] = (ts[order], [items[channel][i] for i in order])
        super().__init__(streams)
        self.path = path
        self.topics = dict(channels)
        self.msgtypes = {c: msgtypes[t] for c, t in channels.items()}

    @staticmethod
    def _resolve_topics(topics, msgtypes: dict[str, str], path: Path) -> dict[str, str]:
        if topics is None or isinstance(topics, list):
            named = (
                [t for t, mt in msgtypes.items() if is_supported_msgtype(mt)]
                if topics is None
                else [t for t in topics if t in msgtypes]
            )
            clashes = colliding_channels(named)
            if clashes:
                detail = "; ".join(
                    f"{', '.join(ts)} -> '{name}'" for name, ts in clashes.items()
                )
                raise ValueError(
                    f"Several topics of {path} would get the same channel name "
                    f"({detail}). Name the channels yourself: "
                    f"topics={{channel: topic}}."
                )
        if topics is None:
            chosen = {
                topic_to_dir(t): t
                for t, mt in msgtypes.items()
                if is_supported_msgtype(mt)
            }
            skipped = sorted(t for t, mt in msgtypes.items() if not is_supported_msgtype(mt))
            if skipped:
                warnings.warn(
                    f"BagDataset skips topics whose type it cannot convert: "
                    f"{', '.join(f'{t} ({msgtype_short(msgtypes[t])})' for t in skipped)}.",
                    UserWarning,
                    stacklevel=3,
                )
            if not chosen:
                raise ValueError(f"{path} has no topic of a type BagDataset converts.")
            return chosen
        mapping = (
            {topic_to_dir(t): t for t in topics} if isinstance(topics, list) else dict(topics)
        )
        missing = sorted(t for t in mapping.values() if t not in msgtypes)
        if missing:
            raise ValueError(
                f"Topics absent from {path}: {missing}. The bag has: {sorted(msgtypes)}."
            )
        unsupported = sorted(
            f"{t} ({msgtype_short(msgtypes[t])})"
            for t in mapping.values()
            if not is_supported_msgtype(msgtypes[t])
        )
        if unsupported:
            raise ValueError(f"BagDataset cannot convert these topics: {unsupported}.")
        return mapping

    def __repr__(self) -> str:
        sizes = {k: len(v) for k, v in self.loaders.items()}
        return f"BagDataset({self.path.name}, events={len(self)}, channels={sizes})"
