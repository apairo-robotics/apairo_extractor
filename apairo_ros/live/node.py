"""Wire a deployed chain to ROS 2 subscriptions.

:class:`ChainNode` subscribes to one topic per channel the chain reads,
converts each message as :mod:`apairo_ros` converts it during an
extraction, feeds a :class:`~apairo_ros.live.source.LiveSource`, and hands every
output frame to your callback::

    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import PointCloud2
    from nav_msgs.msg import Odometry
    from apairo import load_chain
    from apairo_ros.live import ChainNode

    rclpy.init()
    node = Node("traversability")
    chain = load_chain("trav_chain.json", trusted=["my_robot"])
    ChainNode(
        node,
        chain,
        topics={"ouster_points": ("/ouster/points", PointCloud2),
                "odom": ("/odom", Odometry)},
        on_output=lambda frame: publish(frame.data["trav_pred"]),
    )
    rclpy.spin(node)

Messages are stamped with the node clock when they arrive, as a bag stamps
them when it records (and as ``BagDataset`` reads them by default), so the
live node and the bag replay see the same clock. ``stamp="header"`` uses the
message header instead.

Only the node object is used (``create_subscription``, ``get_clock``), so
this module does not import ``rclpy`` itself.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

from apairo_ros.converters import convert_message

from apairo_ros.live.source import LiveSource

__all__ = ["ChainNode", "msgtype_of"]


def msgtype_of(msg_class: type) -> str:
    """``sensor_msgs.msg.PointCloud2`` -> ``"sensor_msgs/msg/PointCloud2"``.

    Reads ``__msgtype__`` when the class carries it (rosbags message classes).
    """
    declared = getattr(msg_class, "__msgtype__", None)
    if isinstance(declared, str):
        return declared
    package = msg_class.__module__.split(".")[0]
    return f"{package}/msg/{msg_class.__name__}"


def _header_time(msg: Any) -> float | None:
    stamp = getattr(getattr(msg, "header", None), "stamp", None)
    if stamp is None:
        return None
    return stamp.sec + stamp.nanosec / 1e9


class ChainNode:
    """Run a deployed chain on the topics of a ROS 2 node.

    Args:
        node: An ``rclpy.node.Node`` (anything with ``create_subscription``
            and ``get_clock``).
        chain: A chain from :func:`apairo.load_chain`, or an ``InferenceView``.
        topics: ``{channel: (topic, message class)}`` for every channel the
            chain reads (``chain.channels_needed()``).
        on_output: Called with each output frame (an ``apairo.Sample``
            holding the chain's outputs, stamped with the reference time).
        stamp: ``"receive"`` (default) or ``"header"``.
        qos: QoS profile or history depth passed to ``create_subscription``.

    Raises:
        ValueError: If a channel the chain reads has no topic, or a message
            class cannot be converted.
    """

    def __init__(
        self,
        node: Any,
        chain: Any,
        topics: dict[str, tuple[str, type]],
        on_output: Callable[[Any], None],
        *,
        stamp: str = "receive",
        qos: Any = 10,
        history: int | None = None,
    ) -> None:
        from apairo_ros.converters import is_supported_msgtype

        if stamp not in ("receive", "header"):
            raise ValueError(f"stamp must be 'receive' or 'header', got {stamp!r}.")
        self.source = LiveSource(chain, history=history)
        missing = [c for c in self.source.channels if c not in topics]
        if missing:
            raise ValueError(
                f"The chain reads {missing} but no topic is given for them: pass "
                f"topics={{channel: (topic, message class)}} for "
                f"{self.source.channels}."
            )
        self._node = node
        self._stamp = stamp
        self._on_output = on_output
        self.subscriptions = []
        for channel in self.source.channels:
            topic, msg_class = topics[channel]
            msgtype = msgtype_of(msg_class)
            if not is_supported_msgtype(msgtype):
                raise ValueError(f"{topic}: {msgtype} cannot be converted to an array.")
            callback = partial(self._on_message, channel, msgtype)
            self.subscriptions.append(
                node.create_subscription(msg_class, topic, callback, qos)
            )

    def _now(self) -> float:
        return self._node.get_clock().now().nanoseconds / 1e9

    def _on_message(self, channel: str, msgtype: str, msg: Any) -> None:
        t = _header_time(msg) if self._stamp == "header" else None
        if t is None:
            t = self._now()
        value = convert_message(msg, msgtype)
        if value is None:
            return
        frame = self.source.push(channel, t, value)
        if frame is not None:
            self._on_output(frame)
