# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import contextlib
import logging

import numpy as np
import pytest
from omegaconf import OmegaConf

from weathergen.datasets.data_reader_base import (
    DataReaderBase,
    ReaderData,
    TimeWindowHandler,
)
from weathergen.datasets.elevation import ElevatingReader

START = np.datetime64("2023-01-01T00:00")
END = np.datetime64("2023-02-01T00:00")
WINDOW_LEN = np.timedelta64(6, "h")
WINDOW_STEP = np.timedelta64(6, "h")
# One forecast step is a day, i.e. four window steps, so step and window index cannot be confused.
FORECAST_STEP = np.timedelta64(24, "h")

NUM_POINTS = 5


class FakeReader(DataReaderBase):
    """Minimal reader over three synthetic channels, so the tests need no data on disk."""

    def __init__(self, stream_info: dict, num_points: int = NUM_POINTS) -> None:
        tw_handler = TimeWindowHandler(START, END, WINDOW_LEN, WINDOW_STEP)
        super().__init__(tw_handler, stream_info)

        self.num_points = num_points
        self.channels = ["sst", "2t", "msl"]

        self.source_channels = self.channels
        self.target_channels = self.channels
        self.geoinfo_channels = ["z"]
        self.source_idx = [0, 1, 2]
        self.target_idx = [0, 1, 2]
        self.geoinfo_idx = [0]
        self.target_channel_weights = [1.0, 1.0, 1.0]

        self.mean = np.zeros(3)
        self.stdev = np.ones(3)
        self.mean_geoinfo = np.zeros(1)
        self.stdev_geoinfo = np.ones(1)

        # Not part of the base interface: used to check that delegation reaches through.
        self.colnames = ["fake"]

        self.get_calls: list[tuple] = []

    def length(self) -> int:
        return 100

    def _get(self, idx, channels_idx) -> ReaderData:
        self.get_calls.append((int(idx), tuple(channels_idx)))
        n = self.num_points
        # value = 100 * channel index, so every column is distinguishable
        data = np.tile(np.array([100.0 * ch for ch in channels_idx], dtype=np.float32), (n, 1))
        return ReaderData(
            coords=np.zeros((n, 2), dtype=np.float32),
            geoinfos=np.zeros((n, 1), dtype=np.float32),
            data=data,
            datetimes=np.full((n,), START, dtype="datetime64[ns]"),
        )


def make_stream_info(elevation=None) -> dict:
    info = {"name": "FAKE", "type": "fake"}
    if elevation is not None:
        info["elevation"] = elevation
    return OmegaConf.create(info)


def make_reader(elevation=None, num_points: int = NUM_POINTS) -> ElevatingReader:
    wrapped = FakeReader(make_stream_info(elevation), num_points=num_points)
    return ElevatingReader(wrapped, START, FORECAST_STEP)


def window_idx_of_step(step: int) -> int:
    """Index of the time window in which forecast step `step` falls."""
    return int(step * FORECAST_STEP / WINDOW_STEP)


@contextlib.contextmanager
def capture_elevation_logs():
    """Collect this module's log records without depending on pytest's logging plugin."""
    logger = logging.getLogger("weathergen.datasets.elevation")
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append

    # Force the level: without it an ambient WARNING level drops these INFO records before any
    # handler sees them, and the "does not warn" assertion below would pass on silence alone.
    previous_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


def test_empty_elevation_is_identity():
    reader = make_reader({})
    baseline = FakeReader(make_stream_info())

    np.testing.assert_array_equal(
        reader.get_source(np.int64(40)).data, baseline.get_source(np.int64(40)).data
    )


def test_offset_applies_from_the_scheduled_step_onward():
    reader = make_reader({"sst": {"offset": 2.0, "step": 4}})
    onset = window_idx_of_step(4)

    # every window before the threshold is untouched
    for idx in range(0, onset):
        assert reader.get_source(np.int64(idx)).data[0, 0] == pytest.approx(0.0)

    # and it stays applied for the rest of the run, not just at the onset step
    for idx in range(onset, onset + 40):
        assert reader.get_source(np.int64(idx)).data[0, 0] == pytest.approx(2.0)


def test_step_zero_is_active_immediately():
    reader = make_reader({"sst": {"offset": 2.0, "step": 0}})
    assert reader.get_source(np.int64(0)).data[0, 0] == pytest.approx(2.0)


def test_step_defaults_to_zero():
    reader = make_reader({"sst": {"offset": 2.0}})
    assert reader.get_source(np.int64(0)).data[0, 0] == pytest.approx(2.0)


def test_offset_can_be_negative():
    reader = make_reader({"2t": {"offset": -1.5, "step": 0}})
    # channel "2t" sits at index 1, whose synthetic value is 100.0
    assert reader.get_source(np.int64(0)).data[0, 1] == pytest.approx(98.5)


def test_other_channels_are_untouched():
    reader = make_reader({"sst": {"offset": 2.0, "step": 0}})
    data = reader.get_source(np.int64(0)).data

    assert data[0, 0] == pytest.approx(2.0)
    assert data[0, 1] == pytest.approx(100.0)
    assert data[0, 2] == pytest.approx(200.0)


def test_per_channel_schedules_are_independent():
    reader = make_reader(
        {
            "sst": {"offset": 2.0, "step": 0},
            "msl": {"offset": -50.0, "step": 4},
        }
    )

    early = reader.get_source(np.int64(0)).data
    assert early[0, 0] == pytest.approx(2.0)
    assert early[0, 2] == pytest.approx(200.0)

    late = reader.get_source(np.int64(window_idx_of_step(4))).data
    assert late[0, 0] == pytest.approx(2.0)
    assert late[0, 2] == pytest.approx(150.0)


def test_source_and_target_are_both_elevated():
    reader = make_reader({"sst": {"offset": 2.0, "step": 0}})

    assert reader.get_source(np.int64(0)).data[0, 0] == pytest.approx(2.0)
    assert reader.get_target(np.int64(0)).data[0, 0] == pytest.approx(2.0)


def test_direct_get_is_refused():
    """The schedule is per side, and a raw channel selection does not say which side it is."""
    reader = make_reader({"sst": {"offset": 2.0, "step": 0}})

    with pytest.raises(NotImplementedError):
        reader._get(np.int64(0), reader.source_idx)

    assert reader.get_source(np.int64(0)).data[0, 0] == pytest.approx(2.0)


def test_all_rows_are_elevated():
    reader = make_reader({"sst": {"offset": 2.0, "step": 0}})
    np.testing.assert_allclose(reader.get_source(np.int64(0)).data[:, 0], np.full(NUM_POINTS, 2.0))


def test_from_date_matches_the_equivalent_step():
    by_step = make_reader({"sst": {"offset": 2.0, "step": 4}})
    by_date = make_reader({"sst": {"offset": 2.0, "from_date": "2023-01-05T00:00"}})

    for idx in (0, window_idx_of_step(4) - 1, window_idx_of_step(4), window_idx_of_step(9)):
        assert by_step.get_source(np.int64(idx)).data[0, 0] == pytest.approx(
            by_date.get_source(np.int64(idx)).data[0, 0]
        )


def test_unmatched_channel_name_is_ignored():
    reader = make_reader({"not_a_channel": {"offset": 2.0, "step": 0}})
    np.testing.assert_array_equal(
        reader.get_source(np.int64(0)).data[0], np.array([0.0, 100.0, 200.0], dtype=np.float32)
    )


def test_unmatched_is_reported_only_when_it_matches_neither_side():
    """A name absent from both channel sets is worth a line; anything else is a false alarm."""
    with capture_elevation_logs() as records:
        make_reader({"not_a_channel": {"offset": 2.0, "step": 0}})

    messages = [r.getMessage() for r in records]
    assert any("Unmatched" in m and "not_a_channel" in m for m in messages), messages


def test_source_only_channel_does_not_warn():
    """
    The case this check exists for: a forcing stream with target: [].

    ERA5-Ocean is source: [sst], target: [] on every gen02 atmo model, so resolving the target
    side finds nothing. That is normal and must stay silent -- warning per side made every
    ablation run print a spurious "Unmatched elevation channels ... ['sst']".
    """
    wrapped = FakeReader(make_stream_info({"sst": {"offset": 2.0, "step": 0}}))
    wrapped.target_channels = []
    wrapped.target_idx = []

    with capture_elevation_logs() as records:
        reader = ElevatingReader(wrapped, START, FORECAST_STEP)

    messages = [r.getMessage() for r in records]
    # capture is demonstrably working -- the schedule line is there -- so the absence below is real
    assert any("elevating source channel" in m for m in messages), messages
    assert not any("Unmatched" in m for m in messages), messages
    # and the source side still works
    assert reader.get_source(np.int64(0)).data[0, 0] == pytest.approx(2.0)


def test_missing_offset_is_rejected():
    with pytest.raises(ValueError, match="offset"):
        make_reader({"sst": {"step": 4}})


def test_step_and_from_date_together_are_rejected():
    with pytest.raises(ValueError, match="from_date"):
        make_reader({"sst": {"offset": 2.0, "step": 4, "from_date": "2023-01-05T00:00"}})


def test_empty_reader_data_is_handled():
    reader = make_reader({"sst": {"offset": 2.0, "step": 0}}, num_points=0)
    assert reader.get_source(np.int64(0)).is_empty()


def test_metadata_is_forwarded():
    reader = make_reader({"sst": {"offset": 2.0, "step": 0}})

    assert reader.source_channels == ["sst", "2t", "msl"]
    assert reader.target_channel_weights == [1.0, 1.0, 1.0]
    assert reader.get_source_num_channels() == 3
    assert reader.get_geoinfo_size() == 1
    assert len(reader) == 100
    np.testing.assert_array_equal(reader.stdev, np.ones(3))


def test_unknown_attributes_are_not_delegated():
    """No blanket __getattr__: a name the wrapper does not forward has to fail loudly."""
    reader = make_reader({"sst": {"offset": 2.0, "step": 0}})

    # set on the wrapped reader, outside the base interface
    assert reader._wrapped_reader.colnames == ["fake"]
    with pytest.raises(AttributeError):
        _ = reader.colnames


def test_normalization_sees_the_offset():
    """The offset is added in physical units, so normalization scales it by 1/stdev."""
    wrapped = FakeReader(make_stream_info({"sst": {"offset": 2.0, "step": 0}}))
    wrapped.stdev = np.array([4.0, 1.0, 1.0])
    reader = ElevatingReader(wrapped, START, FORECAST_STEP)

    normalized = reader.normalize_source_channels(reader.get_source(np.int64(0)).data)
    assert normalized[0, 0] == pytest.approx(0.5)


def test_wrapped_reader_buffer_is_not_mutated():
    """A reader handing back a view must not see the offset accumulate across reads."""

    class CachingReader(FakeReader):
        def __init__(self, stream_info):
            super().__init__(stream_info)
            self.buffer = np.zeros((NUM_POINTS, 3), dtype=np.float32)

        def _get(self, idx, channels_idx) -> ReaderData:
            return ReaderData(
                coords=np.zeros((NUM_POINTS, 2), dtype=np.float32),
                geoinfos=np.zeros((NUM_POINTS, 1), dtype=np.float32),
                data=self.buffer,
                datetimes=np.full((NUM_POINTS,), START, dtype="datetime64[ns]"),
            )

    wrapped = CachingReader(make_stream_info({"sst": {"offset": 2.0, "step": 0}}))
    reader = ElevatingReader(wrapped, START, FORECAST_STEP)

    for _ in range(3):
        assert reader.get_source(np.int64(0)).data[0, 0] == pytest.approx(2.0)
    assert wrapped.buffer[0, 0] == pytest.approx(0.0)


def test_normalization_is_forwarded_to_the_wrapped_reader():
    """fesom, mesh and cams normalize with statistics of their own; inheriting would shadow them."""

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

    wrapped = OwnStatsReader(make_stream_info({"sst": {"offset": 2.0, "step": 0}}))
    reader = ElevatingReader(wrapped, START, FORECAST_STEP)

    ones = np.ones((1, 3), dtype=np.float32)
    geo = np.ones((1, 1), dtype=np.float32)

    assert reader.normalize_source_channels(ones)[0, 0] == pytest.approx(-9.0)
    assert reader.normalize_target_channels(ones)[0, 0] == pytest.approx(-19.0)
    assert reader.denormalize_source_channels(ones)[0, 0] == pytest.approx(11.0)
    assert reader.denormalize_target_channels(ones)[0, 0] == pytest.approx(21.0)
    assert reader.normalize_geoinfos(geo)[0, 0] == pytest.approx(-29.0)
