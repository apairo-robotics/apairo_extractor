"""Tests for the public API surface and the headless CLI."""
from __future__ import annotations

import pytest
import yaml

from apairo_extractor.cli import (
    _build_parser,
    _load_bags,
    _parse_preprocess_spec,
    main,
)

from conftest import LIDAR_TOPIC, IMU_TOPIC, LIDAR_COUNT, write_tiny_bag


# A FramePreprocessor that reduces each lidar frame to its centroid.
PREPROC_SRC = '''
import numpy as np
from apairo import FramePreprocessor

class Centroid(FramePreprocessor):
    input_keys = ["points"]
    output_key = "centroid"

    def process(self, sample):
        return np.mean(sample.data["points"], axis=0)
'''


class _DummyPreproc:
    input_keys = ["points", "pose"]
    output_key = "out_default"


def test_public_api_exports():
    import apairo_extractor as pkg

    for name in ("run_extraction", "extract_bag", "read_bag_info",
                 "find_bags", "BagInfo", "MntExportConfig"):
        assert hasattr(pkg, name), name
        assert name in pkg.__all__


def _make_bags(d, fmt, names=("a", "b")):
    return [write_tiny_bag(d / (f"{n}.bag" if fmt == "ros1" else n), fmt) for n in names]


def test_load_bags_filters_by_name(tmp_path, fmt):
    _make_bags(tmp_path, fmt, names=("a", "b", "c"))
    bags = _load_bags(tmp_path, ["a", "c"])
    assert {b.path.stem for b in bags} == {"a", "c"}


def test_load_bags_all_when_unfiltered(tmp_path, fmt):
    _make_bags(tmp_path, fmt, names=("a", "b"))
    assert len(_load_bags(tmp_path, None)) == 2


def test_parser_defaults_to_interactive():
    # No --input → interactive mode (input is None).
    args = _build_parser().parse_args([])
    assert args.input is None


def test_parser_headless_flags():
    args = _build_parser().parse_args(
        ["-i", "/in", "-o", "/out", "-t", "/lidar", "/imu", "-w", "3"]
    )
    assert args.input == "/in" and args.output == "/out"
    assert args.topics == ["/lidar", "/imu"] and args.workers == 3


def test_headless_list_exits_zero(tmp_path, fmt, capsys):
    _make_bags(tmp_path, fmt)
    code = _exit_code(["-i", str(tmp_path), "--list"])
    assert code == 0


def test_headless_requires_topics(tmp_path, fmt):
    _make_bags(tmp_path, fmt)
    # No --topics and not --list → usage error (exit 2).
    assert _exit_code(["-i", str(tmp_path), "-o", str(tmp_path / "out")]) == 2


def test_headless_extraction_end_to_end(tmp_path, fmt):
    src = tmp_path / "bags"
    src.mkdir()
    _make_bags(src, fmt)
    out = tmp_path / "out"
    code = _exit_code(
        ["-i", str(src), "-t", LIDAR_TOPIC, IMU_TOPIC, "-o", str(out), "-w", "2"]
    )
    assert code == 0
    for stem in ("a", "b"):
        meta = yaml.safe_load((out / stem / "metadata.yaml").read_text())
        assert set(meta["channels"]) == {"lidar", "imu"}


def test_headless_creates_nested_output_dir(tmp_path, fmt):
    src = tmp_path / "bags"
    src.mkdir()
    _make_bags(src, fmt)
    out = tmp_path / "deep" / "nested" / "out"   # does not exist yet
    assert not out.exists()
    assert _exit_code(["-i", str(src), "-t", LIDAR_TOPIC, "-o", str(out), "-w", "1"]) == 0
    assert out.is_dir()


# ── Preprocessing (headless) ─────────────────────────────────────────────────


def test_parse_preprocess_spec_defaults():
    cfg = _parse_preprocess_spec("f.Centroid", {"f.Centroid": _DummyPreproc}, ["points", "pose"])
    assert cfg.output_key == "out_default"
    assert cfg.key_map == {"points": "points", "pose": "pose"}


def test_parse_preprocess_spec_overrides():
    cfg = _parse_preprocess_spec(
        "f.Centroid:output=clean,points=lidar", {"f.Centroid": _DummyPreproc}, ["lidar", "pose"]
    )
    assert cfg.output_key == "clean"
    assert cfg.key_map == {"points": "lidar", "pose": "pose"}


def test_parse_preprocess_spec_unknown_raises():
    with pytest.raises(SystemExit):
        _parse_preprocess_spec("f.Nope", {"f.Centroid": _DummyPreproc}, [])


def test_parse_preprocess_spec_bad_mapping_raises():
    with pytest.raises(SystemExit):
        _parse_preprocess_spec("f.Centroid:points", {"f.Centroid": _DummyPreproc}, [])


def test_headless_preprocess_requires_dir(tmp_path, fmt):
    src = tmp_path / "bags"
    src.mkdir()
    _make_bags(src, fmt)
    # --preprocess without --preprocess-dir → usage error.
    assert _exit_code(
        ["-i", str(src), "-t", LIDAR_TOPIC, "-o", str(tmp_path / "out"),
         "--preprocess", "f.Centroid"]
    ) == 2


def test_headless_preprocessing_end_to_end(tmp_path, fmt):
    src = tmp_path / "bags"
    src.mkdir()
    _make_bags(src, fmt)
    pp = tmp_path / "pp"
    pp.mkdir()
    (pp / "mypp.py").write_text(PREPROC_SRC)
    out = tmp_path / "out"

    code = _exit_code([
        "-i", str(src), "-t", LIDAR_TOPIC, "-o", str(out), "-w", "1",
        "--preprocess-dir", str(pp),
        "--preprocess", "mypp.Centroid:points=lidar",
    ])
    assert code == 0
    for stem in ("a", "b"):
        centroid = out / stem / "centroid"
        assert centroid.is_dir()
        # One output .npy per lidar frame, plus its timestamps sidecar.
        assert len(list(centroid.glob("[0-9]*.npy"))) == LIDAR_COUNT
        assert (centroid / "timestamps.txt").exists()


def _exit_code(argv) -> int:
    """Run main(argv) and capture the SystemExit code (0 if it returns)."""
    try:
        main(argv)
    except SystemExit as exc:
        return exc.code or 0
    return 0
