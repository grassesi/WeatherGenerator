# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for UpsamplingReader."""

import numpy as np
import pytest

from weathergen.datasets.data_reader_base import (
    DataReaderTimestep,
    ReaderData,
    TimeWindowHandler,
)
from weathergen.datasets.upsampling import UpsamplingReader

START = np.datetime64("2023-01-01T00:00")
END = np.datetime64("2023-03-01T00:00")
WINDOW = np.timedelta64(6, "h")
SOURCE_PERIOD = np.timedelta64(24, "h")
STRIDE = 4

NUM_POINTS = 3
GEOINFO_CHANNELS = ["z", "cos_local_time"]

_ZERO = np.timedelta64(0, "s")


class FakeReader(DataReaderTimestep):
    """A fixed-period reader that, like a real one, carries rows only on its own timesteps."""

    def __init__(
        self,
        stream_info: dict,
        period: np.timedelta64 = SOURCE_PERIOD,
        data_start_time: np.datetime64 = START,
        missing: tuple[int, ...] = (),
    ) -> None:
        super().__init__(
            TimeWindowHandler(START, END, WINDOW, WINDOW),
            stream_info,
            data_start_time,
            END,
            period,
        )

        self.source_channels = ["sst"]
        self.target_channels = ["sst"]
        self.geoinfo_channels = GEOINFO_CHANNELS
        self.source_idx = [0]
        self.target_idx = [0]
        self.geoinfo_idx = [0, 1]
        self.target_channel_weights = [1.0]

        self.mean = np.zeros(1)
        self.stdev = np.ones(1)
        self.mean_geoinfo = np.zeros(2)
        self.stdev_geoinfo = np.ones(2)

        self.colnames = ["fake"]

        self._missing = set(missing)
        self.source_reads: list[int] = []

    def length(self) -> int:
        return 200

    def _get(self, idx, channels_idx) -> ReaderData:
        raise AssertionError("the wrapper must not read through _get")

    def get_source(self, idx) -> ReaderData:
        self.source_reads.append(int(idx))
        return self._sample(idx)

    def get_target(self, idx) -> ReaderData:
        return self._sample(idx)

    def _sample(self, idx) -> ReaderData:
        """Rows for the window only if one of the dataset's timesteps falls on its start."""

        start = self.time_window_handler.window(idx).start
        offset = start - self.data_start_time
        empty = ReaderData.empty(1, len(GEOINFO_CHANNELS))

        if offset < _ZERO or offset % self.period != _ZERO:
            return empty

        step = int(offset // self.period)
        if step in self._missing:
            return empty

        geoinfos = np.zeros((NUM_POINTS, 2), dtype=np.float32)
        geoinfos[:, 0] = 1.0  # z, static

        return ReaderData(
            coords=np.zeros((NUM_POINTS, 2), dtype=np.float32),
            geoinfos=geoinfos,
            # value = which source sample this is, so a test can tell them apart
            data=np.full((NUM_POINTS, 1), float(step), dtype=np.float32),
            datetimes=np.full((NUM_POINTS,), start, dtype="datetime64[ns]"),
        )


def make_stream_info() -> dict:
    return {"name": "FAKE"}


def make_reader(**kwargs) -> UpsamplingReader:
    return UpsamplingReader(FakeReader(make_stream_info(), **kwargs))


def test_windows_between_source_steps_serve_the_covering_sample():
    reader = make_reader()

    served = [reader.get_source(np.int64(idx)).data[0, 0] for idx in range(2 * STRIDE)]

    assert served == [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]


def test_only_the_covering_window_is_read():
    """A map, not a search: no walking back over the windows in between."""
    wrapped = FakeReader(make_stream_info())
    reader = UpsamplingReader(wrapped)

    reader.get_source(np.int64(STRIDE - 1))

    assert wrapped.source_reads == [0]


def test_missing_source_sample_stays_empty():
    """The point of the change: a gap is not filled from the sample before it."""
    reader = make_reader(missing=(1,))

    for idx in range(STRIDE, 2 * STRIDE):
        assert reader.get_source(np.int64(idx)).is_empty()

    assert reader.get_source(np.int64(0)).data[0, 0] == pytest.approx(0.0)
    assert reader.get_source(np.int64(2 * STRIDE)).data[0, 0] == pytest.approx(2.0)


def test_phase_offset_is_respected():
    reader = make_reader(data_start_time=START + WINDOW)

    # the source grid starts one window in, so windows 1..4 are covered by its first sample
    for idx in range(1, STRIDE + 1):
        assert reader.get_source(np.int64(idx)).data[0, 0] == pytest.approx(0.0)

    assert reader.get_source(np.int64(STRIDE + 1)).data[0, 0] == pytest.approx(1.0)
    # window 0 precedes the dataset and is covered by nothing
    assert reader.get_source(np.int64(0)).is_empty()


def test_period_that_does_not_divide_the_window_step_is_rejected():
    with pytest.raises(ValueError, match="whole multiple"):
        make_reader(period=np.timedelta64(9, "h"))


def test_period_finer_than_the_window_step_is_rejected():
    with pytest.raises(ValueError, match="nothing to upsample"):
        make_reader(period=np.timedelta64(3, "h"))


def test_start_off_the_window_grid_is_rejected():
    with pytest.raises(ValueError, match="whole number of window steps"):
        make_reader(data_start_time=START + np.timedelta64(1, "h"))


def test_stride_one_serves_every_window_unchanged():
    reader = make_reader(period=WINDOW)

    for idx in range(4):
        rdata = reader.get_source(np.int64(idx))
        assert rdata.data[0, 0] == pytest.approx(float(idx))
        assert rdata.datetimes[0] == reader.time_window_handler.window(np.int64(idx)).start


def test_served_timestamps_land_in_the_requested_window():
    reader = make_reader()

    for idx in range(STRIDE):
        window = reader.time_window_handler.window(np.int64(idx))
        stamps = reader.get_source(np.int64(idx)).datetimes
        assert (stamps >= window.start).all()
        assert (stamps < window.end).all()


def test_time_varying_geoinfos_are_recomputed():
    reader = make_reader()

    at_source = reader.get_source(np.int64(0)).geoinfos
    served = reader.get_source(np.int64(2)).geoinfos

    # z is constant in time and is served as it comes
    np.testing.assert_allclose(served[:, 0], at_source[:, 0])
    # cos_local_time describes the window being served, twelve hours later
    assert served[0, 1] != pytest.approx(at_source[0, 1])


def test_target_is_not_upsampled():
    reader = make_reader()

    assert reader.get_target(np.int64(1)).is_empty()
    assert reader.get_target(np.int64(0)).data[0, 0] == pytest.approx(0.0)


def test_wrapped_reader_buffer_is_not_mutated():
    """A reader handing back a view must not see the restamping accumulate across reads."""

    class CachingReader(FakeReader):
        def __init__(self, stream_info):
            super().__init__(stream_info)
            self.buffer = np.full((NUM_POINTS,), START, dtype="datetime64[ns]")

        def _sample(self, idx) -> ReaderData:
            rdata = super()._sample(idx)
            if rdata.is_empty():
                return rdata
            return ReaderData(
                coords=rdata.coords,
                geoinfos=rdata.geoinfos,
                data=rdata.data,
                datetimes=self.buffer,
            )

    wrapped = CachingReader(make_stream_info())
    reader = UpsamplingReader(wrapped)

    for _ in range(3):
        stamps = reader.get_source(np.int64(2)).datetimes
        assert stamps[0] == START + 2 * WINDOW
    assert wrapped.buffer[0] == START


def test_normalization_is_forwarded_to_the_wrapped_reader():
    class OwnStatsReader(FakeReader):
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

    reader = UpsamplingReader(OwnStatsReader(make_stream_info()))

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


def test_metadata_is_forwarded():
    reader = make_reader()

    assert reader.source_channels == ["sst"]
    assert reader.period == SOURCE_PERIOD
    assert reader.data_start_time == START
    assert reader.get_source_num_channels() == 1
    assert reader.get_geoinfo_size() == 2
    assert len(reader) == 200
