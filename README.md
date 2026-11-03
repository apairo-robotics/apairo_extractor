# Apairo Extractor

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

## Timestamps and frames

Every channel, `/tf` edges included, is stamped with the bag's record time
(when the recorder received the message), so all channels of a sequence share
one clock. Header stamps are not used: in a simulator bag they are sim time
while the record time is wall time, and the two run at different rates.

A channel's `frame` is its messages' `header.frame_id`. If that frame changes
within a topic, the channel gets no frame and the frames seen are listed under
`frame_ids` in its `metadata.yaml`.

## Images

`sensor_msgs/Image` is decoded by its `encoding`: `mono8/16`, `rgb8/16`,
`bgr8/16` (reordered to RGB), `rgba8/16`, `bgra8/16`, raw Bayer, and the
generic `<8U|8S|16U|16S|32S|32F|64F>C<n>` (e.g. a `32FC1` depth image is stored
as float32, a `32SC1` label image as int32). Any other encoding raises an error
naming the topic. The encoding is recorded in the channel's `metadata.yaml`.

## Process

Can directly take into account preprocess functions from apairo.

