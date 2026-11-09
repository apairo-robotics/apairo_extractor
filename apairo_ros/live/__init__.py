"""apairo_ros.live -- run a chain deployed with apairo in a live ROS 2 node.

    >>> from apairo import load_chain
    >>> from apairo_ros.live import LiveSource, ChainNode

``LiveSource`` feeds a chain one message at a time (no ROS dependency);
``ChainNode`` wires it to the subscriptions of an ``rclpy`` node. Both need
apairo >= 0.10 (``inference()`` and ``load_chain()``). ``rclpy`` is not a
dependency: it comes with a ROS 2 distribution, and ``ChainNode`` only uses
the node it is given.
"""

from apairo_ros.live.node import ChainNode, msgtype_of
from apairo_ros.live.source import LiveSource

__all__ = ["ChainNode", "LiveSource", "msgtype_of"]
