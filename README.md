# Apairo Extractor

Using [apairo](../apairo), extract ROS bags into an apairo-native dataset
(KITTI-style `.apairo` layout; Zarr WIP) — ready to load with `apairo.RawDataset`.
It writes the `.apairo` sidecars as it goes (per-sequence `channels.yaml`,
root `dataset.yaml`), tags each channel with its `header.frame_id`, and handles
`/tf` / `/tf_static` (see [Transforms](#transforms)).


## Quickstart

Run `apairo-extractor` with no arguments for the interactive TUI (equivalently
`apairo extractor`, if apairo's CLI is installed — see *CLI* below):
```
╭──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│      _    ____   _    ___ ____   ___                                                                                                                                                 │
│     / \  |  _ \ / \  |_ _|  _ \ / _ \                                                                                                                                                │
│    / _ \ | |_) / _ \  | || |_) | | | |                                                                                                                                               │
│   / ___ \|  __/ ___ \ | ||  _ <| |_| |                                                                                                                                               │
│  /_/   \_\_| /_/   \_\___|_| \_\\___/                                                                                                                                                │
│                                                                                                                                                                                      │
│               E X T R A C T                                                                                                                                                          │
│                                                                                                                                                                                      │
│            → KITTI • ZARR                                                                                                                                                            │
│                                                                                                                                                                                      │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

? Directory containing rosbags:

```

### Data Management

You indicate the root directory of your rosbags

```sh
/input/path/
├── rosbag_yvettes/    
│   ├── yvette-mountain-020606.bag
├── rosbag_tartan/  
└────── ...
``` 

It will extract it with the options indicated.

Here it is in KITTI-style `.apairo` layout with `cmd`, `ouster` and `/tf`:
```
output/path/                          # dataset root — load with apairo.RawDataset(...)
├── .apairo/
│   └── dataset.yaml                  # dataset name + sequence order
├── <sequence_name>/                  # e.g. yvette-mountain-020606
│   ├── .apairo/
│   │   ├── channels.yaml             # channel -> loader / kind / frame / transform
│   │   └── calibration.yaml          # static extrinsics (from /tf_static)
│   ├── cmd/                          # seq topic -> one stacked .npy
│   │   ├── cmd.npy
│   │   └── timestamps.txt
│   ├── ouster/                       # frame topic -> one .npy per message
│   │   ├── 000000.npy  000000_intensity.npy  ...  001735.npy
│   │   ├── metadata.yaml
│   │   └── timestamps.txt
│   └── tf__odom__base_link/          # /tf edge -> pose channel
│       ├── tf__odom__base_link.npy
│       └── timestamps.txt
└── ...
```

Load the result directly:
```python
import apairo
ds = apairo.RawDataset("output/path")     # whole dataset (all sequences)
ds.calibration                            # static extrinsics from /tf_static
```

## CLI

Interactive TUI (no arguments) or fully headless with flags:

```sh
# headless: list, then extract selected topics
apairo-extractor -i /input/path -l                      # discover bags + topics
apairo-extractor -i /input/path -o /output/path -t /ouster /cmd /tf /tf_static
```

`apairo-extractor` is also exposed as `apairo extractor` through apairo's CLI
dispatcher (same command, plugin-discovered — no extra dependency).

## Transforms

`/tf` and `/tf_static` are understood:

- **`/tf`** (time-varying) is demultiplexed into one pose channel per edge,
  named `<source>__<parent>__<child>` (e.g. `tf__odom__base_link`), stored as
  `[tx,ty,tz, qx,qy,qz,qw]`. Edges from different sources are kept as distinct
  channels — nothing is merged or dropped.
- **`/tf_static`** (time-independent) goes to `calibration` — one 4x4 entry per
  edge in `.apairo/calibration.yaml`, not a channel — so a tree of fixed mounts
  stays one small file. Read it via `ds.calibration`.

Applying transforms (reframing point clouds, etc.) is a downstream concern, not
the extractor's job — it only stores the transforms faithfully.

## Process

Can directly take into account preprocess functions from apairo.

