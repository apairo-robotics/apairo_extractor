"""Cross-repo contract test: extractor output loads with ``apairo.RawDataset``.

The extractor writes apairo sidecars at two levels -- a per-sequence
``.apairo/channels.yaml`` (channel -> loader/kind/timestamps) and a root
``.apairo/dataset.yaml`` (name + sequence order). These tests extract real tiny
bags end to end and assert the result loads back exactly as apairo expects:
single sequence, whole root, and synchronized frames. Skipped when the installed
apairo predates ``RawDataset``.
"""
from __future__ import annotations

import pytest

from apairo_extractor.bag import read_bag_info
from apairo_extractor.runner import run_extraction

from conftest import IMU_COUNT, IMU_TOPIC, LIDAR_COUNT, LIDAR_TOPIC, write_tiny_bag

apairo = pytest.importorskip("apairo")
if not hasattr(apairo, "RawDataset"):
    pytest.skip(
        "installed apairo has no RawDataset (pre-RawDataset version)",
        allow_module_level=True,
    )


@pytest.fixture
def extracted_root(tmp_path, fmt):
    """Extract two tiny bags into one dataset root; return the root directory."""
    bags = [
        read_bag_info(
            write_tiny_bag(tmp_path / (f"{n}.bag" if fmt == "ros1" else n), fmt)
        )
        for n in ("seq_a", "seq_b")
    ]
    out = tmp_path / "my_dataset"
    run_extraction(bags, [LIDAR_TOPIC, IMU_TOPIC], out, workers=1)
    return out


def test_root_loads_with_rawdataset(extracted_root):
    ds = apairo.RawDataset(extracted_root)
    assert ds.name == "my_dataset"
    assert ds.sequence_ids == ["seq_a", "seq_b"]
    assert ds.available == frozenset({"lidar", "imu"})
    # Asynchronous: per sequence, LIDAR_COUNT + IMU_COUNT interleaved events.
    assert len(ds) == 2 * (LIDAR_COUNT + IMU_COUNT)
    # One event populates exactly one channel.
    assert len(ds[0].data) == 1


def test_single_sequence_loads_standalone(extracted_root):
    seq = apairo.RawDataset(extracted_root / "seq_a")
    assert seq.available == frozenset({"lidar", "imu"})
    assert len(seq) == LIDAR_COUNT + IMU_COUNT


def test_synchronize_yields_full_frames(extracted_root):
    ds = apairo.RawDataset(extracted_root)
    sync = ds.synchronize(reference="lidar")
    assert len(sync) == 2 * LIDAR_COUNT  # one synchronous frame per lidar message
    assert {"lidar", "imu"} <= set(sync[0].data)


def test_channel_loaders_match_kind(extracted_root):
    """Frame topic -> npys (per-frame .npy); buffered seq topic -> npy (stacked)."""
    from apairo.core.config import read_config

    channels = read_config(extracted_root / "seq_a")["channels"]
    assert channels["lidar"]["loader"] == "npys"
    assert channels["imu"]["loader"] == "npy"
