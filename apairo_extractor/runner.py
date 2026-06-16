"""UI-agnostic extraction engine.

This is the layer that orchestrates extracting many bags — sequentially or in
parallel across processes — without any knowledge of the TUI. The interactive
TUI, the headless CLI, and the public Python API all drive the same
:func:`run_extraction` and receive progress through plain callbacks.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import queue as _queue
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional

from apairo_extractor.bag import BagInfo, topic_to_dir
from apairo_extractor.export_mnt import MntExportConfig, export_to_mnt, extract_bag_to_mnt
from apairo_extractor.extract import extract_bag
from apairo_extractor.preprocess import PreprocessConfig, run_preprocessors

# Progress within one bag: report(phase, done, total). ``phase`` is a short
# label ("extract", "preprocess", "MNT …", "done"); each pipeline phase resets
# the counters with its own total.
ReportFn = Callable[[str, int, int], None]

# Engine callbacks. on_progress(bag_id, phase, done, total) fires continuously;
# on_bag_done(bag_name, skipped, error) fires once per finished bag (error is a
# repr string, or None on success).
ProgressFn = Callable[[str, str, int, int], None]
BagDoneFn = Callable[[str, list, Optional[str]], None]

# One (bag_name, skipped_topics, error_repr) tuple per bag.
BagResult = tuple[str, list, Optional[str]]


def topic_for_channel(channel: str | None, topics: list[str]) -> str | None:
    """Reverse-map a KITTI channel dir name to its original ROS topic name."""
    if channel is None:
        return None
    for t in topics:
        if topic_to_dir(t) == channel:
            return t
    return None


@contextmanager
def _pool_context():
    """A multiprocessing context for the worker pool.

    Prefers ``fork`` (POSIX): it works from any caller — plain scripts,
    notebooks, the REPL — with no ``if __name__ == "__main__"`` guard, unlike
    ``spawn``/``forkserver`` which re-import the caller's module. Falls back to
    ``spawn`` where fork is unavailable (e.g. Windows).

    Our workers only touch the progress queue, numpy and rosbags — never the
    parent's rich display — so the "fork in a multi-threaded process" warning
    doesn't apply here; we silence just that one message.
    """
    method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=DeprecationWarning,
            message=r".*fork.*may lead to deadlocks.*",
        )
        yield mp.get_context(method)


def resolve_workers(n_bags: int) -> int:
    """Default worker count when the caller doesn't specify one.

    min(4, CPU count), overridable via ``APAIRO_EXTRACT_WORKERS``; never exceeds
    the bag count and is at least 1 (sequential).
    """
    env = os.environ.get("APAIRO_EXTRACT_WORKERS")
    if env:
        try:
            workers = int(env)
        except ValueError:
            workers = 0
    else:
        workers = min(4, os.cpu_count() or 1)
    return max(1, min(workers, max(1, n_bags)))


def _do_bag(
    bag: BagInfo,
    topics: list[str],
    preprocess_configs: list[PreprocessConfig],
    output_dir: Optional[Path],
    mnt_config: Optional[MntExportConfig],
    mnt_only: bool,
    mnt_points_topic: str | None,
    mnt_image_topic: str | None,
    mnt_odom_topic: str | None,
    report: ReportFn,
) -> list[str]:
    """Run the full extraction pipeline for one bag; return its skipped topics.

    The single source of truth for per-bag work, shared by the sequential and
    parallel paths — only ``report`` differs between them.
    """
    mission_dir = mnt_config.output_dir / bag.path.stem if mnt_config else None
    skipped: list[str] = []

    if mnt_only:
        mnt_total = sum(t.msgcount for t in bag.topics if t.name in
                        {mnt_points_topic, mnt_image_topic, mnt_odom_topic} - {None})

        def _mnt_direct_cb(stage, done, total):
            report(stage, done, max(total, 1))

        extract_bag_to_mnt(
            bag, mission_dir,
            points_topic=mnt_points_topic,
            image_topic=mnt_image_topic,
            odometry_topic=mnt_odom_topic,
            max_points=mnt_config.max_points,
            progress_cb=_mnt_direct_cb,
        )
        report("done", max(mnt_total, 1), max(mnt_total, 1))
        return skipped

    # KITTI extraction (+ optional preprocessing / MNT conversion from KITTI).
    topic_total = sum(t.msgcount for t in bag.topics if t.name in set(topics))

    def _extract_cb(done, total):
        report("extract", done, max(total, 1))

    seq_dir, sk = extract_bag(bag, topics, output_dir, progress_cb=_extract_cb)
    skipped.extend(sk)

    if preprocess_configs:
        report("preprocess", 0, 1)
        run_preprocessors(seq_dir, preprocess_configs)
        report("preprocess", 1, 1)

    if mnt_config is not None:
        def _mnt_cb(stage, done, total):
            report(f"MNT {stage}", done, max(total, 1))

        export_to_mnt(seq_dir, mission_dir, mnt_config, progress_cb=_mnt_cb)

    report("done", max(topic_total, 1), max(topic_total, 1))
    return skipped


def _extract_one_bag_worker(args) -> BagResult:
    """Process-pool entry point: run ``_do_bag`` and stream progress to a queue.

    Takes a single picklable tuple so it can be submitted directly. Returns
    ``(bag_name, skipped, error_repr)``; ``error_repr`` is None on success.
    """
    (bag, topics, preprocess_configs, output_dir, mnt_config,
     mnt_only, mnt_points_topic, mnt_image_topic, mnt_odom_topic, q) = args

    bag_id = bag.path.stem

    def report(phase: str, done: int, total: int) -> None:
        q.put((bag_id, phase, done, total))

    try:
        skipped = _do_bag(
            bag, topics, preprocess_configs, output_dir, mnt_config,
            mnt_only, mnt_points_topic, mnt_image_topic, mnt_odom_topic, report,
        )
        return bag.path.name, skipped, None
    except Exception as exc:  # noqa: BLE001 — report, don't kill sibling bags
        return bag.path.name, [], repr(exc)


def _finalize_dataset(output_dir: Optional[Path], mnt_only: bool) -> None:
    """Dataset-level apairo init, written once after extraction (KITTI only).

    Delegates to :meth:`apairo.dataset.raw.RawDataset.init`: it (idempotently)
    ensures each sequence's ``.apairo/channels.yaml`` and writes the root
    ``.apairo/dataset.yaml`` manifest. apairo owns the ``.apairo`` layout, so the
    extractor keeps a single source of truth instead of its own manifest writer.
    """
    if mnt_only or output_dir is None:
        return
    from apairo.dataset.raw import RawDataset

    try:
        RawDataset.init(output_dir, merge=True)
    except FileNotFoundError:
        pass  # no sequences extracted (e.g. every bag failed) -> nothing to init


def run_extraction(
    bags: list[BagInfo],
    topics: list[str],
    output_dir: Optional[Path],
    *,
    mnt_config: Optional[MntExportConfig] = None,
    preprocess_configs: list[PreprocessConfig] | tuple = (),
    workers: Optional[int] = None,
    on_progress: Optional[ProgressFn] = None,
    on_bag_done: Optional[BagDoneFn] = None,
) -> list[BagResult]:
    """Extract ``topics`` from each of ``bags`` into ``output_dir``.

    Bags are independent and extracted in parallel across ``workers`` processes
    (spawn) when ``workers > 1``; pass ``workers=1`` for sequential. A failing
    bag is reported via its result tuple and does not abort the others.

    Args:
        output_dir: KITTI output root, or None for direct MNT-only extraction.
        mnt_config: MNT/zarr export config (KITTI→MNT, or direct if no output_dir).
        preprocess_configs: apairo preprocessors to run after each KITTI extract.
        workers: parallel bag count; default :func:`resolve_workers`. Clamped to
            ``[1, len(bags)]``.
        on_progress: called as (bag_id, phase, done, total) during extraction.
        on_bag_done: called as (bag_name, skipped, error_repr) per finished bag.

    Returns:
        One ``(bag_name, skipped_topics, error_repr)`` per bag.
    """
    preprocess_configs = list(preprocess_configs)
    mnt_only = output_dir is None
    if not mnt_only:
        output_dir.mkdir(parents=True, exist_ok=True)
    if mnt_config is not None:
        mnt_config.output_dir.mkdir(parents=True, exist_ok=True)

    mnt_points_topic = topic_for_channel(mnt_config.points_channel, topics) if mnt_config else None
    mnt_image_topic  = topic_for_channel(mnt_config.image_channel,  topics) if mnt_config else None
    mnt_odom_topic   = topic_for_channel(mnt_config.odometry_channel, topics) if mnt_config else None

    workers = workers if workers is not None else resolve_workers(len(bags))
    workers = max(1, min(workers, max(1, len(bags))))
    bag_args = (topics, preprocess_configs, output_dir, mnt_config,
                mnt_only, mnt_points_topic, mnt_image_topic, mnt_odom_topic)

    results: list[BagResult] = []

    def _emit(bag_id, phase, done, total):
        if on_progress:
            on_progress(bag_id, phase, done, total)

    def _finish(name, skipped, err):
        results.append((name, skipped, err))
        if on_bag_done:
            on_bag_done(name, skipped, err)

    if workers <= 1:
        for bag in bags:
            bag_id = bag.path.stem
            try:
                skipped = _do_bag(
                    bag, *bag_args,
                    lambda phase, done, total, _b=bag_id: _emit(_b, phase, done, total),
                )
                _finish(bag.path.name, skipped, None)
            except Exception as exc:  # noqa: BLE001 — match the parallel path
                _finish(bag.path.name, [], repr(exc))
        _finalize_dataset(output_dir, mnt_only)
        return results

    with _pool_context() as ctx, ctx.Manager() as manager:
        q = manager.Queue()

        def _drain() -> None:
            try:
                while True:
                    _emit(*q.get_nowait())
            except _queue.Empty:
                pass

        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
            pending = {
                executor.submit(_extract_one_bag_worker, (bag, *bag_args, q))
                for bag in bags
            }
            while pending:
                _drain()
                for fut in [f for f in pending if f.done()]:
                    _finish(*fut.result())
                    pending.discard(fut)
                time.sleep(0.05)
            _drain()

    _finalize_dataset(output_dir, mnt_only)
    return results
