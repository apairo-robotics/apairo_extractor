"""apairo_ros — ROS for apairo: extract bags to an apairo dataset and/or MNT, or read them in place.

Public API:
    >>> from pathlib import Path
    >>> from apairo_ros import find_bags, read_bag_info, run_extraction
    >>> bags = [read_bag_info(p) for p in find_bags(Path("/data/bags"))]
    >>> run_extraction(bags, ["/lidar", "/imu"], Path("/out"), workers=4)

``run_extraction`` is UI-agnostic; pass ``on_progress`` / ``on_bag_done``
callbacks to observe progress, or drive the interactive TUI / headless CLI via
the ``apairo-extractor`` command.

``BagDataset`` reads a bag in place, as an apairo dataset, with the same
conversion and timestamps as an extraction: the source to replay a deployed
chain on (``deployed.on(BagDataset(path))``).
"""
from __future__ import annotations

from apairo_ros.bag import (
    BagInfo,
    TopicInfo,
    compute_topic_coverage,
    find_bags,
    read_bag_info,
    topics_in_bag,
)
from apairo_ros.bag_dataset import BagDataset
from apairo_ros.export_mnt import MntExportConfig
from apairo_ros.extract import extract_bag
from apairo_ros.runner import resolve_workers, run_extraction

__version__ = "0.1.0"

__all__ = [
    "BagDataset",
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
