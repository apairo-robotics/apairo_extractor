"""Run a deployed chain on messages as they arrive.

:class:`LiveSource` takes a chain saved from training (``deployed.save()``
then :func:`apairo.load_chain`) and is fed one message at a time. When a
message arrives on the chain's reference channel, it runs the chain on the
recent messages it keeps and returns the frame for that instant: the same
frame the chain gives offline, on a recording of the same messages.

It holds no ROS dependency: :mod:`apairo_ros.live.node` wires it to ROS 2
subscriptions, and a test can push messages directly.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
from apairo.dataset.stream import StreamDataset

if TYPE_CHECKING:
    from apairo import Sample

__all__ = ["LiveSource"]


class LiveSource:
    """Feed a deployed chain one message at a time.

    Args:
        chain: A :class:`apairo.Chain` (``load_chain(path)``) or the
            ``InferenceView`` returned by ``ds.inference(...)``.
        history: How many reference frames to keep. Defaults to twice the
            chain's ``lookback`` (its windows' reach) plus two, so a frame
            dropped by a synchronization tolerance does not shorten a window.

    Raises:
        ValueError: If the chain needs future events (it is not causal), or
            its source channels are unknown.

    Example::

        chain = load_chain("trav_chain.json", trusted=["my_robot"])
        live = LiveSource(chain)
        for channel, t, value in messages:          # in arrival order
            frame = live.push(channel, t, value)
            if frame is not None:
                publish(frame.data["trav_pred"])
    """

    def __init__(self, chain: Any, *, history: int | None = None) -> None:
        chain = getattr(chain, "chain", chain)  # an InferenceView carries its chain
        if not chain.causal:
            non_causal = [
                e.item for e in chain.summary.flags if e.reason == "non-causal"
            ]
            raise ValueError(
                "A live source can only run a causal chain: "
                + "; ".join(non_causal)
                + ". Synchronize these channels with method='previous'."
            )
        channels = chain.channels_needed()
        if not channels:
            raise ValueError("The chain does not say which channels it reads.")
        self._chain = chain
        self.reference: str | None = chain.reference
        self.channels: list[str] = sorted(set(channels) | {self.reference} - {None})
        self.history = history if history is not None else 2 * chain.lookback + 2
        self._t: dict[str, list[float]] = {c: [] for c in self.channels}
        self._v: dict[str, list[Any]] = {c: [] for c in self.channels}
        self.late = 0  # messages older than the last one on their channel

    @property
    def outputs(self) -> list[str]:
        return self._chain.outputs

    def push(self, channel: str, t: float, value: Any) -> Sample | None:
        """Add a message; return the chain's frame at *t* if one is due.

        A frame is due when *channel* is the chain's reference (or, for a
        chain without ``synchronize``, on every message), once every channel
        the chain reads has had a message. ``None`` otherwise, and when the
        chain gives no frame at *t* (no match within a tolerance, a window
        that ``boundary="drop"`` discards).

        A message older than the last one on its channel is dropped and
        counted in :attr:`late`. Messages on channels the chain does not
        read are ignored.
        """
        if channel not in self._t:
            return None
        times = self._t[channel]
        if times and t < times[-1]:
            self.late += 1
            if self.late == 1:
                warnings.warn(
                    f"LiveSource dropped a message on {channel!r} older than the "
                    f"previous one ({t} < {times[-1]}); see .late for the count.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return None
        times.append(float(t))
        self._v[channel].append(value)
        if self.reference is not None and channel != self.reference:
            return None
        self._trim()
        if any(not self._t[c] for c in self.channels):
            return None
        source = StreamDataset(
            {c: (np.asarray(self._t[c]), list(self._v[c])) for c in self.channels}
        )
        frames = self._chain.on(source)
        if len(frames) == 0:
            return None
        frame = frames[len(frames) - 1]
        if frame.timestamp is not None and frame.timestamp != float(t):
            return None  # the chain gives no frame at this instant
        return frame

    def _trim(self) -> None:
        """Keep the last ``history`` reference frames, and on every other
        channel what they can match: the messages since the oldest kept frame,
        plus the one just before it (``"previous"`` matching)."""
        if self.reference is None:
            for c in self.channels:
                del self._t[c][: -self.history], self._v[c][: -self.history]
            return
        ref = self._t[self.reference]
        if len(ref) > self.history:
            cut = len(ref) - self.history
            del ref[:cut], self._v[self.reference][:cut]
        oldest = ref[0]
        for c in self.channels:
            if c == self.reference:
                continue
            ts = self._t[c]
            first = int(np.searchsorted(ts, oldest, side="left"))
            keep_from = max(first - 1, 0)
            del ts[:keep_from], self._v[c][:keep_from]

    def __repr__(self) -> str:
        sizes = {c: len(v) for c, v in self._t.items()}
        return (
            f"LiveSource(reference={self.reference!r}, outputs={self.outputs}, "
            f"buffered={sizes})"
        )
