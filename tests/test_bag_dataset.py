"""BagDataset: a bag read in place gives the samples of the same bag extracted.

The parity test of a deployment: a chain built on extracted data, deployed
with ``inference()``, replayed on the bag it was extracted from with ``on()``,
gives the same outputs.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
from apairo_extractor import BagDataset
from apairo_extractor.bag import read_bag_info
from apairo_extractor.extract import extract_bag
from conftest import CLOUD_POINTS, IMU_COUNT, LIDAR_COUNT

apairo = pytest.importorskip("apairo")
needs_inference = pytest.mark.skipif(
    not hasattr(apairo.RawDataset, "inference"),
    reason="installed apairo has no inference() (apairo >= 0.10)",
)


def test_reads_every_convertible_topic(tiny_bag):
    with pytest.warns(UserWarning, match=r"/chatter \(String\)"):
        bag = BagDataset(tiny_bag)
    assert sorted(bag.keys) == ["imu", "lidar"]
    assert len(bag.timestamps["lidar"]) == LIDAR_COUNT
    assert len(bag.timestamps["imu"]) == IMU_COUNT
    np.testing.assert_array_equal(bag.loaders["lidar"][0][:, :3], CLOUD_POINTS)


def test_channel_names_can_be_chosen(tiny_bag):
    bag = BagDataset(tiny_bag, topics={"velodyne": "/lidar"})
    assert bag.keys == ["velodyne"]
    assert bag.topics == {"velodyne": "/lidar"}


def test_absent_and_unconvertible_topics_are_refused(tiny_bag):
    with pytest.raises(ValueError, match=r"absent.*\['/nope'\]"):
        BagDataset(tiny_bag, topics=["/nope"])
    with pytest.raises(ValueError, match=r"cannot convert.*String"):
        BagDataset(tiny_bag, topics=["/chatter"])


def test_same_samples_as_the_extracted_bag(tiny_bag, tmp_path):
    seq_dir, _ = extract_bag(read_bag_info(tiny_bag), ["/lidar", "/imu"], tmp_path / "out")
    extracted = apairo.RawDataset(seq_dir)
    bag = BagDataset(tiny_bag, topics=["/lidar", "/imu"])
    for key in ("lidar", "imu"):
        np.testing.assert_allclose(bag.timestamps[key], extracted.timestamps[key])
        for i in range(len(bag.timestamps[key])):
            np.testing.assert_array_equal(bag.loaders[key][i], extracted.loaders[key][i])


class Scale:
    def __init__(self, k):
        self.k = k

    def __call__(self, x):
        return x * self.k


class Jitter:
    inference = False

    def __call__(self, x):
        return x + np.random.normal(size=x.shape)


class Score:
    input_keys = ["lidar", "imu"]
    output_keys = ["score"]

    def __call__(self, s):
        s.data["score"] = float(s.data["lidar"][:, :3].sum() + s.data["imu"].sum())
        return s


@needs_inference
def test_a_chain_trained_on_extracted_data_replays_on_the_bag(tiny_bag, tmp_path):
    seq_dir, _ = extract_bag(read_bag_info(tiny_bag), ["/lidar", "/imu"], tmp_path / "out")
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        train = apairo.RawDataset(seq_dir).synchronize(reference="lidar")
        train = train.transform("lidar", apairo.Compose([Scale(2.0), Jitter()]), in_place=False)
        train = train.transform(Score(), in_place=False)
        deployed = train.inference("score")

    assert [e.item for e in deployed.summary.removed] == ["Jitter"]
    replayed = deployed.on(BagDataset(tiny_bag, topics=["/lidar", "/imu"]))
    assert len(replayed) == len(deployed)
    for i in range(len(deployed)):
        assert replayed[i].data["score"] == deployed[i].data["score"]

    chain = apairo.load_chain(deployed.save(tmp_path / "chain.json"), trusted=["test_bag_dataset"])
    assert chain.on(BagDataset(tiny_bag))[0].data["score"] == deployed[0].data["score"]
