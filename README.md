# apairo-ros

ROS for [apairo](https://github.com/apairo-robotics/apairo). ROS 1 and ROS 2
bags are read with [rosbags](https://gitlab.com/ternaris/rosbags), so no ROS
installation is needed:

- **Extract** a bag to an apairo dataset with `apairo-extractor` (below);
- **read it in place** with `BagDataset`, converted and stamped as an
  extraction would, to replay a deployed chain on it (see
  [Read a bag in place](#read-a-bag-in-place--bagdataset)).

## Installation

```bash
pip install apairo-ros
pip install "apairo-ros[mnt]"           # the Zarr MNT export
```

Requires Python ≥ 3.11 and apairo ≥ 0.9.

## Extraction

Using [apairo](https://github.com/apairo-robotics/apairo), extract ROS bags into an apairo-native dataset
(one directory per channel, described by the `.apairo` sidecar; a Zarr MNT export is in progress) — ready to load with `apairo.RawDataset`.
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
│            → .apairo • MNT                                                                                                                                                           │
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

Here it is as an apairo dataset with `cmd`, `ouster` and `/tf`:
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

## Read a bag in place — `BagDataset`

No extraction step: the topics of a bag as an asynchronous apairo dataset,
converted and stamped exactly as an extraction would, so a bag read in place
and the same bag extracted give the same samples.

```python
from apairo_ros import BagDataset

bag = BagDataset("run_42", topics=["/ouster/points", "/odom"])   # channels: ouster_points, odom
bag = BagDataset("run_42", topics={"lidar": "/ouster/points"})  # or name them
frames = bag.synchronize(reference="ouster_points")
```

It is the source to replay a deployed chain on, to check a deployment
against a recording (`deployed.on(bag)`, see apairo's *Inference &
Deployment*). Messages are held in memory: choose the topics you need.

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

