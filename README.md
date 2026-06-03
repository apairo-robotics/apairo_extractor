# Rosbag Extractor

Extract a rosbag to the Kitti format :

```sh
/input/path/
├── rosbag_yvettes/    
│   ├── yvette-mountain-020606.bag
├── rosbag_tartan/  
└────── ...
``` 

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

