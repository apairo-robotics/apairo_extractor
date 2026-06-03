# Apairo Extractor

Using [apairo](../apairo), extract a rosbag to Kitti or Zarr(WIP) format.


## Quickstart

By typing apairo-extract, 
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

Here it is in `Kitti` format with `cmd` and `ouster`.
```
output/path/
├── <sequence_name>/             # e.g. yvette-mountain-020606
│   ├── cmd/
│   │   ├── twist.npy
│   │   └── timestamps.txt
│   └── ouster/
│       ├── 000000.npy
│       ├── 000000_intensity.npy
│       ├── ...
│       ├── 001735.npy
│       ├── 001735_intensity.npy
│       ├── metadata.yaml
│       └── timestamps.txt
└── output_datasets/
```

## CLI

Working with a CLI on terminal for easy use.


## Process

Can directly take into account preprocess function from apairo.

