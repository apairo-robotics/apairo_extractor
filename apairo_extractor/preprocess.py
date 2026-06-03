from __future__ import annotations

import importlib.util
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PreprocessConfig:
    cls: type
    output_key: str
    key_map: dict[str, str]  # preprocessor_input_key → extracted_channel_dir_name


def discover_preprocessors(folder: Path) -> dict[str, type]:
    """Scan a folder for FramePreprocessor / SequencePreprocessor subclasses.

    Returns a dict keyed by '<filename>.<ClassName>'.
    """
    from apairo import FramePreprocessor, SequencePreprocessor

    result: dict[str, type] = {}
    for py_file in sorted(folder.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        mod_name = f"_rosbag_preproc_{py_file.stem}"
        spec = importlib.util.spec_from_file_location(mod_name, py_file)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            print(f"[warn] Could not import {py_file.name}: {exc}")
            continue

        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj in (FramePreprocessor, SequencePreprocessor):
                continue
            if issubclass(obj, (FramePreprocessor, SequencePreprocessor)):
                result[f"{py_file.stem}.{name}"] = obj

    return result


def run_preprocessors(
    seq_dir: Path,
    configs: list[PreprocessConfig],
    progress_cb=None,
) -> None:
    """Apply a list of preprocessors to an extracted sequence directory."""
    from apairo import FramePreprocessor, SequencePreprocessor
    from apairo.core.config import config_exists, read_config, register_channel

    channels_meta = read_config(seq_dir).get("channels", {}) if config_exists(seq_dir) else {}

    for cfg in configs:
        out_dir = seq_dir / cfg.output_key
        out_dir.mkdir(exist_ok=True)

        preprocessor = cfg.cls()
        out_loader = getattr(cfg.cls, "output_loader", "npys")
        timestamps_from = getattr(cfg.cls, "timestamps_from", None)
        sources = list(cfg.key_map.values())

        if issubclass(cfg.cls, FramePreprocessor):
            _run_frame(preprocessor, seq_dir, cfg, channels_meta, out_dir, progress_cb)
        elif issubclass(cfg.cls, SequencePreprocessor):
            _run_sequence(preprocessor, seq_dir, cfg, channels_meta, out_dir)

        register_channel(
            seq_dir,
            cfg.output_key,
            out_loader,
            timestamps_from=timestamps_from,
            sources=sources or None,
        )


# ── Internal helpers ───────────────────────────────────────────────────────────


def _ref_channel(cfg: PreprocessConfig) -> str:
    """Return the extracted channel name for the first input key."""
    first = cfg.cls.input_keys[0]
    return cfg.key_map.get(first, first)


def _load_frame(seq_dir: Path, channel: str, idx: int) -> np.ndarray:
    return np.load(seq_dir / channel / f"{idx:06d}.npy")


def _load_seq(seq_dir: Path, channel: str) -> np.ndarray:
    files = list((seq_dir / channel).glob("*.npy"))
    return np.load(files[0])


def _run_frame(preprocessor, seq_dir, cfg, channels_meta, out_dir, progress_cb):
    from apairo.core.sample import Sample

    ref = _ref_channel(cfg)
    loader = channels_meta.get(ref, {}).get("loader", "npys")
    ts_path = seq_dir / ref / "timestamps.txt"
    timestamps = np.loadtxt(ts_path) if ts_path.exists() else np.array([])

    if loader == "npys":
        n = len(list((seq_dir / ref).glob("*.npy")))
    else:
        n = len(timestamps)

    out_ts = []
    for idx in range(n):
        data = {}
        for preproc_key, channel in cfg.key_map.items():
            ch_loader = channels_meta.get(channel, {}).get("loader", "npys")
            data[preproc_key] = (
                _load_frame(seq_dir, channel, idx)
                if ch_loader == "npys"
                else _load_seq(seq_dir, channel)[idx]
            )
        ts = float(timestamps[idx]) if idx < len(timestamps) else 0.0
        result = np.asarray(preprocessor.process(Sample(data=data, timestamp=ts)))
        np.save(out_dir / f"{idx:06d}.npy", result)
        out_ts.append(ts)
        if progress_cb:
            progress_cb(cfg.output_key, idx + 1, n)

    np.savetxt(out_dir / "timestamps.txt", out_ts)


def _run_sequence(preprocessor, seq_dir, cfg, channels_meta, out_dir):
    from apairo.core.sample import Sample

    ref = _ref_channel(cfg)
    loader = channels_meta.get(ref, {}).get("loader", "npys")
    ts_path = seq_dir / ref / "timestamps.txt"
    timestamps = np.loadtxt(ts_path) if ts_path.exists() else np.array([])

    n = (
        len(list((seq_dir / ref).glob("*.npy")))
        if loader == "npys"
        else len(timestamps)
    )

    def _iter():
        for idx in range(n):
            data = {}
            for preproc_key, channel in cfg.key_map.items():
                ch_loader = channels_meta.get(channel, {}).get("loader", "npys")
                data[preproc_key] = (
                    _load_frame(seq_dir, channel, idx)
                    if ch_loader == "npys"
                    else _load_seq(seq_dir, channel)[idx]
                )
            ts = float(timestamps[idx]) if idx < len(timestamps) else 0.0
            yield Sample(data=data, timestamp=ts)

    result = np.asarray(preprocessor.process(_iter()))
    np.save(out_dir / f"{cfg.output_key}.npy", result)
    np.savetxt(out_dir / "timestamps.txt", timestamps)
