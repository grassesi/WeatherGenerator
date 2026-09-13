# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the wrapper interface of HoldingReader."""

import numpy as np
import pytest

from weathergen.datasets.data_reader_base import (
    DataReaderBase,
    ReaderData,
    TimeWindowHandler,
)
from weathergen.datasets.holding import HoldingReader

START = np.datetime64("2023-01-01T00:00")
END = np.datetime64("2023-02-01T00:00")
WINDOW = np.timedelta64(6, "h")
NUM_POINTS = 3

# one window in four carries data, as a 24 h producer read at 6 h does
CADENCE = 4


class FakeReader(DataReaderBase):
    """Minimal reader with statistics of its own, as fesom, mesh and cams have."""

    def __init__(self, stream_info: dict) -> None:
        super().__init__(TimeWindowHandler(START, END, WINDOW, WINDOW), stream_info)

        self.source_channels = ["sst"]
        self.target_channels = ["sst"]
        self.geoinfo_channels = []
        self.source_idx = [0]
        self.target_idx = [0]
        self.geoinfo_idx = []
        self.target_channel_weights = [1.0]

        self.mean = np.zeros(1)
        self.stdev = np.ones(1)
        self.mean_geoinfo = np.zeros(0)
        self.stdev_geoinfo = np.ones(0)

        self.colnames = ["fake"]

    def length(self) -> int:
        return 100

    def _get(self, idx, channels_idx) -> ReaderData:
        raise AssertionError("the wrapper must not read through _get")

    def get_source(self, idx) -> ReaderData:
        if int(idx) % CADENCE:
            return ReaderData.empty(1, 0)
        return ReaderData(
            coords=np.zeros((NUM_POINTS, 2), dtype=np.float32),
            geoinfos=np.zeros((NUM_POINTS, 0), dtype=np.float32),
            data=np.full((NUM_POINTS, 1), 7.0, dtype=np.float32),
            datetimes=np.full((NUM_POINTS,), START, dtype="datetime64[ns]"),
        )

    def get_target(self, idx) -> ReaderData:
        return self.get_source(idx)

    def normalize_source_channels(self, source):
        return source - 10.0

    def normalize_target_channels(self, target):
        return target - 20.0

    def denormalize_source_channels(self, source):
        return source + 10.0

    def denormalize_target_channels(self, data):
        return data + 20.0

    def normalize_geoinfos(self, geoinfos):
        return geoinfos - 30.0


def make_reader(max_hold: int = CADENCE - 1) -> HoldingReader:
    return HoldingReader(FakeReader({"name": "FAKE"}), max_hold=max_hold)


def test_empty_windows_are_held():
    reader = make_reader()

    for idx in range(CADENCE):
        assert reader.get_source(np.int64(idx)).data[0, 0] == pytest.approx(7.0)


def test_target_is_not_held():
    reader = make_reader()

    assert reader.get_target(np.int64(1)).is_empty()


def test_normalization_is_forwarded_to_the_wrapped_reader():
    reader = make_reader()

    ones = np.ones((1, 1), dtype=np.float32)

    assert reader.normalize_source_channels(ones)[0, 0] == pytest.approx(-9.0)
    assert reader.normalize_target_channels(ones)[0, 0] == pytest.approx(-19.0)
    assert reader.denormalize_source_channels(ones)[0, 0] == pytest.approx(11.0)
    assert reader.denormalize_target_channels(ones)[0, 0] == pytest.approx(21.0)
    assert reader.normalize_geoinfos(ones)[0, 0] == pytest.approx(-29.0)


def test_direct_get_is_refused():
    reader = make_reader()

    with pytest.raises(NotImplementedError):
        reader._get(np.int64(0), reader.source_idx)


def test_unknown_attributes_are_not_delegated():
    reader = make_reader()

    assert reader._wrapped_reader.colnames == ["fake"]
    with pytest.raises(AttributeError):
        _ = reader.colnames
