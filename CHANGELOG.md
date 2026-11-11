# Changelog

## [0.1.1] - 2026-11-10

A code review of 0.1.0 found messages converted to the wrong values without an
error, and extractions that could mix or destroy data. If you extracted clouds
with integer or float64 fields, depth images, or several bags of the same
name, extract them again.

### Wrong data, now fixed

- **PointCloud2 fields keep their values.** Every field was cast to float32: a
  uint32 `t` or `range` was rounded past 16,777,216, and a float64 per-point
  time lost its fraction. The cloud is now float32 only when every field fits,
  float64 otherwise.
- **A field with `count > 1`** (a `normal` of 3) gives one column per value; only
  the first was kept.
- **Row padding of an organized cloud** (`row_step > width * point_step`) is
  skipped; it was read as points.
- **Images keep the type their encoding names.** A `16UC1` depth came out as
  `(H, W, 2) uint8`, a `32FC1` as `(H, W, 4) uint8`. Row padding
  (`step > width * bytes per pixel`) is skipped, `is_bigendian` is read, and
  `bgr16` is stored as RGB like `bgr8`.
- **An empty cloud** has the width its fields give, not always 3 columns.
- **`sensor_msgs/PointCloud`** is converted; it was announced as supported and
  silently dropped.
- **A `compressedDepth` CompressedImage** is decoded past its 12-byte header.

### Refused instead of read by guessing

- A cloud or an image whose payload does not hold what its header declares, and
  an image encoding that is not known, raise an error for that message. The
  message is counted, the topic is reported with the count and the reason (in
  the skipped list, in a warning, and in the sequence's `metadata.yaml`), and
  the rest of the bag is extracted. A truncated cloud used to be read as a
  shorter one. `BagDataset` now does the same instead of failing on the bag.

### Extraction

- **Two bags that would be written to the same sequence** (`day1/run.bag` and
  `day2/run.bag`) are refused before anything is read; their frames were
  mixed. A ROS 2 bag directory keeps its whole name (`run_1.0` and `run_1.1`
  were both `run_1`).
- **Two topics that map to the same channel** (`/a/b` and `/a_b`) are refused;
  the second overwrote the first. `BagDataset` refuses them too and asks for
  explicit channel names. A topic that would name the parent directory
  (`/..`) is refused.
- **A ROS 2 bag is never extracted into itself.** With the output directory
  equal to the input directory, the bag's `metadata.yaml` was overwritten and
  the bag could no longer be opened.
- **Re-extracting clears the channel first.** Frames of an earlier, longer
  extraction stayed beside the new ones and were counted by the
  preprocessors and the MNT export.
- **A failed bag leaves nothing behind**, so it is no longer registered half
  written, and the command line exits with status 1 when a bag failed. It
  printed "Done" and exited 0.
- A topic with a single message no longer breaks the preprocessors and the MNT
  export.
- A preprocessor with several inputs is refused when they are not on one
  clock; rows were paired by position, each frame with an unrelated sample.
- `BagDataset(stamp="header")` uses the header's time whenever there is a
  header, a zero stamp included, as the live node does.
- One topic can be read under several channel names with `BagDataset`.

## [0.1.0] - 2026-11-09

First release on PyPI, as `apairo-ros` (the former `apairo_extractor`).
