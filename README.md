# apairo-ros

ROS for [apairo](https://github.com/apairo-robotics/apairo). ROS 1 and ROS 2
bags are read with [rosbags](https://gitlab.com/ternaris/rosbags), so no ROS
installation is needed:

- **Extract** a bag to an apairo dataset with `apairo-extractor` (below);
- **read it in place** with `BagDataset`, converted and stamped as an
  extraction would, to replay a deployed chain on it (see
  [Read a bag in place](#read-a-bag-in-place--bagdataset));
- **run a deployed chain in a live ROS 2 node** with `apairo_ros.live`
  (see [Live](#live--a-deployed-chain-in-a-ros-2-node)).

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

## Live — a deployed chain in a ROS 2 node

The preprocessing a model was trained on, unchanged, on the robot. Needs
apairo ≥ 0.10 (`inference()`, `load_chain()`).

```python
# training side
deployed = train.inference(["trav_pred"])     # no augmentations, only what the output needs
deployed.on(BagDataset("run_42.mcap"))        # parity check on a recording
deployed.save("trav_chain.json")              # beside the checkpoint
```

```python
# robot side
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from apairo import load_chain
from apairo_ros.live import ChainNode

rclpy.init()
node = Node("traversability")
ChainNode(
    node,
    load_chain("trav_chain.json", trusted=["my_robot"]),
    topics={"ouster_points": ("/ouster/points", PointCloud2),
            "odom": ("/odom", Odometry)},
    on_output=lambda frame: publish(frame.data["trav_pred"]),
)
rclpy.spin(node)
```

- **`LiveSource`** is fed one message at a time. On each message of the
  chain's reference channel (the `synchronize(reference=...)` of the
  training chain), it runs the chain on the recent messages it keeps and
  returns the frame for that instant: the frame the chain gives offline on a
  recording of the same messages. It keeps the frames the chain's windows
  reach back to, and no more. It has no ROS dependency.
- **`ChainNode`** subscribes to one topic per channel, converts each message
  as an extraction does, and stamps it with the node clock on arrival, as a
  bag stamps it when recording. A live message and a recorded one therefore
  give the same array at the same time.

A live source only runs a **causal** chain: `synchronize` with
`method="previous"` (an interpolator or `"nearest"` needs the next event).
`LiveSource` refuses any other chain. A message on a channel arriving after
its reference frame was computed is not seen by that frame: expect what the
same recording gives, with the same message order.

`rclpy` is not a dependency: it comes with a ROS 2 distribution
(`source /opt/ros/<distro>/setup.bash`), and `ChainNode` only uses the node
it is given.

**Status.** Tested without ROS: `LiveSource` against the offline chain, and
`ChainNode` on a fake node fed a real ROS 2 bag's messages, against
`BagDataset` on the same bag. Not yet tested in a running ROS 2 system
(`ros2 bag play` into the node). Calibration from `/tf_static` is not read
yet. Each frame re-runs the chain over the kept history: fine for short
windows, an incremental evaluation is planned for high rates.

## CLI

Interactive TUI (no arguments) or fully headless with flags:

```sh
# headless: list, then extract selected topics
apairo-extractor -i /input/path -l                      # discover bags + topics
apairo-extractor -i /input/path -o /output/path -t /ouster /cmd /tf /tf_static
```

The same command has four names, all installed by the package:

| Name | Where |
|---|---|
| `apairo ros extract` | through apairo's CLI (plugin-discovered, no extra dependency) |
| `apairo-ros extract` | on its own |
| `apairo extractor`, `apairo-extractor` | the names it had before the package became `apairo-ros`; they stay |

`apairo ros` with no argument lists the package's commands.

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

