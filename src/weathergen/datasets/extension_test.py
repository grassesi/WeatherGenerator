# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Contract of ExtendingReader: target windows past the end of the data, sources never."""

import datetime

import numpy as np
import pytest
import torch

from weathergen.datasets.data_reader_base import (
    DataReaderTimestep,
    ReaderData,
    TimeWindowHandler,
    check_reader_data,
)
from weathergen.datasets.extension import (
    ExtendingReader,
    _cos_julian_day,
    _cos_local_time,
    _cos_solar_zenith_angle,
    _sin_julian_day,
    _sin_local_time,
)
from weathergen.datasets.stream_data import StreamData

N_POINTS = 6
# z, lsm are constant in the store and get persisted; insolation and the time terms are recomputed
GEOINFO_CHANNELS = ["z", "lsm", "insolation", "cos_local_time", "sin_julian_day"]
DATA_START = np.datetime64("2023-01-01T00:00")
DATA_END = np.datetime64("2023-01-05T00:00")
WINDOW = np.timedelta64(6, "h")


class FakeReader(DataReaderTimestep):
    """A gridded reader over a short, closed time range."""

    def __init__(self, twh: TimeWindowHandler, data_end_time=DATA_END) -> None:
        super().__init__(twh, {"name": "FAKE", "stream_id": 0}, DATA_START, data_end_time, WINDOW)

        self.source_channels = ["u", "v"]
        self.source_idx = [0, 1]
        self.target_channels = ["u"]
        self.target_idx = [0]
        self.geoinfo_channels = list(GEOINFO_CHANNELS)
        self.geoinfo_idx = list(range(2, 2 + len(GEOINFO_CHANNELS)))
        self.target_channel_weights = [1.0]

        n_vars = 2 + len(GEOINFO_CHANNELS)
        self.mean = np.arange(n_vars, dtype=np.float32) + 100.0
        self.stdev = np.ones(n_vars, dtype=np.float32)
        self.mean_geoinfo = np.zeros(len(GEOINFO_CHANNELS), dtype=np.float32)
        self.stdev_geoinfo = np.ones(len(GEOINFO_CHANNELS), dtype=np.float32)

        self.coords = np.stack(
            [np.linspace(-75, 75, N_POINTS), np.linspace(-150, 150, N_POINTS)], axis=-1
        ).astype(np.float32)

    def length(self) -> int:
        return 16

    def _get(self, idx, channels_idx) -> ReaderData:
        (t_idxs, dtr) = self._get_dataset_idxs(idx)
        if len(t_idxs) == 0:
            return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

        # a recognisable, index-dependent payload so passthrough is easy to assert
        return ReaderData(
            coords=self.coords.copy(),
            geoinfos=np.full(
                (N_POINTS, len(self.geoinfo_idx)), float(idx), dtype=np.float32
            ),
            data=np.full((N_POINTS, len(channels_idx)), float(idx), dtype=np.float32),
            datetimes=np.repeat(dtr.start, N_POINTS),
        )


@pytest.fixture
def time_window_handler() -> TimeWindowHandler:
    return TimeWindowHandler(DATA_START, DATA_START + np.timedelta64(20, "D"), WINDOW, WINDOW)


@pytest.fixture
def reader(time_window_handler) -> ExtendingReader:
    return ExtendingReader(FakeReader(time_window_handler))


def test_last_real_idx_is_the_last_window_inside_the_data(reader):
    """DATA_END is 4 days after DATA_START, so the last full 6h window starts 6h before it."""
    assert reader._last_real_idx == 15
    assert not reader._wrapped_reader.get_target(np.int64(15)).is_empty()
    assert reader._wrapped_reader.get_target(np.int64(16)).is_empty()


def test_in_range_windows_pass_through_untouched(reader):
    for idx in (0, 7, 15):
        extended = reader.get_target(np.int64(idx))
        original = reader._wrapped_reader.get_target(np.int64(idx))

        assert not extended.is_extended
        assert np.array_equal(extended.data, original.data)
        assert np.array_equal(extended.coords, original.coords)
        assert np.array_equal(extended.geoinfos, original.geoinfos)


def test_out_of_range_target_is_a_full_grid_not_a_two_point_spoof(reader):
    rdata = reader.get_target(np.int64(20))

    assert rdata.is_extended
    assert not rdata.is_spoof
    assert rdata.coords.shape == (N_POINTS, 2)
    assert np.array_equal(rdata.coords, reader._wrapped_reader.coords)


def test_out_of_range_target_values_are_absent(reader):
    rdata = reader.get_target(np.int64(20))

    assert rdata.data.shape == (N_POINTS, len(reader.target_idx))
    assert np.isnan(rdata.data).all()


def test_datetimes_are_restamped_into_the_requested_window(reader, time_window_handler):
    idx = np.int64(20)
    rdata = reader.get_target(idx)
    window = time_window_handler.window(idx)

    # check_reader_data enforces exactly this, so an extended window has to satisfy it
    assert (rdata.datetimes >= window.start).all()
    assert (rdata.datetimes < window.end).all()


def test_extended_window_satisfies_the_reader_data_contract(reader, time_window_handler):
    """Whatever a reader returns has to pass check_reader_data, extended windows included."""
    idx = np.int64(20)

    check_reader_data(reader.get_target(idx), time_window_handler.window(idx))


def test_constant_geoinfos_are_persisted_and_time_varying_ones_recomputed(reader):
    rdata = reader.get_target(np.int64(20))
    template_value = float(reader._last_real_idx)

    for pos, channel in enumerate(GEOINFO_CHANNELS):
        column = rdata.geoinfos[:, pos]
        if channel in ("z", "lsm"):
            assert np.allclose(column, template_value), f"{channel} should have been persisted"
        else:
            assert not np.allclose(column, template_value), f"{channel} should be recomputed"

    # and the recomputed columns are the real quantities, not noise
    date = rdata.datetimes[0].astype("datetime64[s]").astype(datetime.datetime)
    lats, lons = rdata.coords[:, 0], rdata.coords[:, 1]
    assert np.allclose(rdata.geoinfos[:, 2], _cos_solar_zenith_angle(date, lats, lons))
    assert np.allclose(rdata.geoinfos[:, 3], _cos_local_time(date, lats, lons))
    assert np.allclose(rdata.geoinfos[:, 4], _sin_julian_day(date, lats, lons))


def test_recomputed_geoinfos_change_with_the_step(reader):
    early = reader.get_target(np.int64(20))
    later = reader.get_target(np.int64(24))  # a full day later

    assert not np.allclose(early.geoinfos[:, 2], later.geoinfos[:, 2])  # insolation
    assert np.allclose(early.geoinfos[:, 0], later.geoinfos[:, 0])  # z stays put


def test_sources_are_never_extended(reader, caplog):
    """The hard restriction: a source past the data end stays empty, it is not invented."""
    passthrough = reader._wrapped_reader.get_source(np.int64(20))
    rdata = reader.get_source(np.int64(20))

    assert rdata.is_empty()
    assert not rdata.is_extended
    assert rdata.data.shape == passthrough.data.shape
    assert any("source window" in message for message in caplog.messages)


def test_get_does_not_fabricate_sources_either(reader):
    for channels_idx in ([0, 1], [0]):
        assert reader._get(np.int64(20), channels_idx).is_empty()


def test_degrades_to_a_noop_without_a_data_end(time_window_handler):
    """An open-ended reader has no edge to extend from; it must not crash or invent one."""
    reader = ExtendingReader(FakeReader(time_window_handler, data_end_time=None))

    assert reader._template is None
    assert reader.get_target(np.int64(20)).is_empty()


def test_attributes_and_bounds_are_forwarded(reader):
    assert reader.length() == 16
    assert reader.data_end_time == DATA_END, "the wrapper must not lie about the real data end"
    assert reader.target_channels == ["u"]
    assert reader.geoinfo_channels == GEOINFO_CHANNELS


def test_stored_template_survives_repeated_reads(reader):
    first = reader.get_target(np.int64(20))
    first.geoinfos[:] = -999.0
    first.coords[:] = -999.0
    second = reader.get_target(np.int64(20))

    assert not np.allclose(second.geoinfos, -999.0)
    assert not np.allclose(second.coords, -999.0)


def _stream_data_with(n_steps: int, extended: list[int]) -> StreamData:
    """A StreamData whose target tokens are all NaN, with `extended` steps flagged."""
    sdata = StreamData(idx=0, input_steps=1, output_steps=n_steps, healpix_cells=48)
    for step in range(n_steps):
        sdata.target_tokens[step] = torch.full((4, 1), torch.nan)
        sdata.target_is_extended[step] = step in extended
    return sdata


def test_extended_steps_do_not_make_a_batch_look_all_nan():
    """
    Blocker this guards: is_nan() rejects a batch whose targets are all NaN, inside a `while True`
    that then draws the next index. A rollout entirely past the data end would spin forever.
    """
    assert not _stream_data_with(3, extended=[0, 1, 2]).target_nan()

    # a step that really is all-NaN still counts, so the existing guard keeps working
    assert _stream_data_with(3, extended=[1, 2]).target_nan()


def test_is_extended_is_step_scoped_unlike_is_spoof():
    sdata = _stream_data_with(3, extended=[2])

    assert not sdata.is_extended(0)
    assert sdata.is_extended(2)
    # and it does not leak into the spoof flag, which the writer keys off
    assert not sdata.is_spoof(2)


@pytest.mark.parametrize(
    "date",
    [
        datetime.datetime(2024, 1, 1, 0),
        datetime.datetime(2024, 3, 21, 6),
        datetime.datetime(2024, 6, 4, 12),
        datetime.datetime(2025, 12, 31, 18),
    ],
)
def test_formulas_match_earthkit(date):
    """
    The computed geoinfos must reproduce what anemoi baked into the zarr.

    They are reimplemented here rather than imported, because earthkit-meteo is only an optional
    extra of anemoi-datasets. This pins the reimplementation wherever earthkit is installed.
    """
    ek_solar = pytest.importorskip("earthkit.meteo.solar")
    forcings = pytest.importorskip("earthkit.data.sources.forcings")

    rng = np.random.default_rng(0)
    lats = rng.uniform(-90, 90, 200)
    lons = rng.uniform(-180, 180, 200)

    assert np.allclose(
        _cos_solar_zenith_angle(date, lats, lons),
        ek_solar.cos_solar_zenith_angle(date, lats, lons),
    )

    class _Field:
        shape = (200,)

    class _Maker(forcings.ForcingMaker):
        def __init__(self):
            self.field = _Field()

        def longitude(self, date):
            return lons

    maker = _Maker()
    assert np.allclose(_cos_local_time(date, lats, lons), maker.cos_local_time(date))
    assert np.allclose(_sin_local_time(date, lats, lons), maker.sin_local_time(date))
    assert np.allclose(_cos_julian_day(date, lats, lons), maker.cos_julian_day(date))
    assert np.allclose(_sin_julian_day(date, lats, lons), maker.sin_julian_day(date))
