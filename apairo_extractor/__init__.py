"""apairo_extractor — extract ROS bags to KITTI/MNT with optional preprocessing.

Public API:
    >>> from pathlib import Path
    >>> from apairo_extractor import find_bags, read_bag_info, run_extraction
    >>> bags = [read_bag_info(p) for p in find_bags(Path("/data/bags"))]
    >>> run_extraction(bags, ["/lidar", "/imu"], Path("/out"), workers=4)

``run_extraction`` is UI-agnostic; pass ``on_progress`` / ``on_bag_done``
callbacks to observe progress, or drive the interactive TUI / headless CLI via
the ``apairo-extractor`` command.
"""
from __future__ import annotations

from apairo_extractor.bag import (
    BagInfo,
    TopicInfo,
    compute_topic_coverage,
    find_bags,
    read_bag_info,
    topics_in_bag,
)
from apairo_extractor.export_mnt import MntExportConfig
from apairo_extractor.extract import extract_bag
from apairo_extractor.runner import resolve_workers, run_extraction

__version__ = "0.1.0"

__all__ = [
    "BagInfo",
    "TopicInfo",
    "MntExportConfig",
    "find_bags",
    "read_bag_info",
    "topics_in_bag",
    "compute_topic_coverage",
    "extract_bag",
    "run_extraction",
    "resolve_workers",
    "__version__",
]
