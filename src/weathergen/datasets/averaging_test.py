# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Contract of AveragingReader: what it averages, and the column order it returns it in."""

import numpy as np
import pytest

from weathergen.datasets.averaging import AveragingReader
from weathergen.datasets.data_reader_base import (
    DataReaderTimestep,
    ReaderData,
    TimeWindowHandler,
)

# insolation is the only averaged one (_WINDOW_MEAN_GEOINFOS); the rest -- z included, it is the
# static surface geopotential -- are carried over from the first datapoint in the window. The
# averaged channel sits last, so a wrapper that returns [averaged, rest] permutes the columns.
GEOINFOS = ["z", "lsm", "slor", "sdor", "insolation"]
N_POINTS = 3
SAMPLES_PER_WINDOW = 2


class FakeReader(DataReaderTimestep):
    """Two samples per window at the same coordinates, so averaging has something to do."""

    def __init__(self, twh: TimeWindowHandler, geoinfo_channels: list[str] | None = None) -> None:
        super().__init__(
            twh,
            {"stream_id": 0},
            np.datetime64("2023-01-01T00:00"),
            np.datetime64("2023-12-31T00:00"),
            np.timedelta64(3, "h"),
        )
        self.source_channels = ["t2m"]
        self.source_idx = [0]
        self.target_channels = ["t2m"]
        self.target_idx = [0]
        self.target_channel_weights = [1.0]
        self.geoinfo_channels = list(geoinfo_channels if geoinfo_channels else GEOINFOS)
        self.geoinfo_idx = list(range(len(self.geoinfo_channels)))
        self.mean = np.zeros(1, dtype=np.float32)
        self.stdev = np.ones(1, dtype=np.float32)
        self.mean_geoinfo = np.zeros(len(self.geoinfo_channels), dtype=np.float32)
        self.stdev_geoinfo = np.ones(len(self.geoinfo_channels), dtype=np.float32)

    def length(self) -> int:
        return 1000

    def _get(self, idx, channels_idx) -> ReaderData:
        coords = np.tile(
            np.stack(
                [np.linspace(-40, 40, N_POINTS), np.linspace(-120, 120, N_POINTS)], axis=-1
            ).astype(np.float32),
            (SAMPLES_PER_WINDOW, 1),
        )
        # one column per geoinfo, valued by its position so a permutation is visible, and
        # differing between the two samples so an average differs from either
        geoinfos = np.tile(
            np.arange(len(self.geoinfo_channels), dtype=np.float32), (len(coords), 1)
        )
        geoinfos[N_POINTS:] += 2.0
        data = np.arange(len(coords), dtype=np.float32).reshape(-1, 1)
        datetimes = np.repeat(
            np.array(
                ["2023-01-01T00:00", "2023-01-01T03:00"],
                dtype="datetime64[ns]",
            ),
            N_POINTS,
        )
        return ReaderData(coords=coords, geoinfos=geoinfos, data=data, datetimes=datetimes)


@pytest.fixture
def reader() -> AveragingReader:
    twh = TimeWindowHandler(
        np.datetime64("2023-01-01T00:00"),
        np.datetime64("2023-12-31T00:00"),
        np.timedelta64(6, "h"),
        np.timedelta64(6, "h"),
    )
    return AveragingReader(FakeReader(twh))


def test_geoinfos_keep_the_declared_column_order(reader):
    """Everything downstream indexes geoinfos by position, so the order must be preserved.

    `normalize_geoinfos` pairs column i with `mean_geoinfo[i]`/`stdev_geoinfo[i]`, which are
    in `geoinfo_channels` order. Returning the averaged channels first would normalize
    insolation with lsm's statistics, and sdor with insolation's.
    """
    rdata = reader._get(np.int64(0), [0])

    assert rdata.geoinfos.shape == (N_POINTS, len(GEOINFOS))
    # the carried-over columns keep the first sample's values, which are position + 0
    for name in ["z", "lsm", "slor", "sdor"]:
        col = GEOINFOS.index(name)
        assert np.allclose(rdata.geoinfos[:, col], float(col)), f"{name} is not in column {col}"
    # the averaged column sits at its declared position, halfway between the two samples
    col = GEOINFOS.index("insolation")
    assert np.allclose(rdata.geoinfos[:, col], col + 1.0), f"insolation is not in column {col}"


def test_averaged_channels_are_averaged_and_the_rest_are_not(reader):
    rdata = reader._get(np.int64(0), [0])

    z, insolation = GEOINFOS.index("z"), GEOINFOS.index("insolation")
    # sample values are v and v + 2, so a mean sits at v + 1 and a carry-over at v
    assert np.allclose(rdata.geoinfos[:, insolation], insolation + 1.0)
    assert np.allclose(rdata.geoinfos[:, GEOINFOS.index("lsm")], 1.0)
    # z is static in every store that declares it, so it is carried, not averaged
    assert np.allclose(rdata.geoinfos[:, z], float(z))


def test_the_window_average_is_stamped_at_the_window_start(reader):
    """[T, T+len) averages to one datapoint at T, not at the last sample in the window.

    The window is anchored on its start and `end` is never a sampled time, so the average of
    a window has to carry the start as well -- otherwise an averaged stream and an
    un-averaged one on the same grid disagree about when the same sample happened.
    """
    rdata = reader._get(np.int64(0), [0])

    window = reader.time_window_handler.window(np.int64(0))
    assert (rdata.datetimes == np.datetime64(window.start, "ns")).all()
    assert rdata.datetimes.shape == (N_POINTS,)
    assert rdata.data.shape == (N_POINTS, 1)


def test_cyclic_time_geoinfos_are_recomputed_at_the_stamp(reader):
    """The cyclic terms describe the stamp, so they follow it rather than a carried sample.

    Checked against the closed form rather than against the reader's own helper: the window
    starts at midnight, where local solar time is lon/15 hours, so cos_local_time is
    cos(lon * 2pi/360) -- [-0.5, 1.0, -0.5] for the fake reader's three longitudes. The
    carried value would be the column index, 1.0 everywhere.
    """
    channels = ["z", "cos_local_time", "insolation"]
    reader = AveragingReader(FakeReader(reader.time_window_handler, channels))

    rdata = reader._get(np.int64(0), [0])

    lons = np.linspace(-120, 120, N_POINTS)
    expected = np.cos(np.deg2rad(lons))
    assert np.allclose(rdata.geoinfos[:, 1], expected, atol=1e-6)
    assert not np.allclose(rdata.geoinfos[:, 1], 1.0)


def test_insolation_keeps_its_window_mean_when_the_cyclic_terms_are_recomputed(reader):
    """insolation is computable too; evaluating it at the stamp would undo the averaging.

    It is a window-mean channel, so it stays the mean over the window (position + 1 for the
    fake reader's two samples) rather than the instantaneous value at the stamp. On real ERA5
    the two differ by up to 0.7 of a unit-range channel.
    """
    channels = ["z", "cos_local_time", "insolation"]
    reader = AveragingReader(FakeReader(reader.time_window_handler, channels))

    rdata = reader._get(np.int64(0), [0])
    assert np.allclose(rdata.geoinfos[:, 2], 2 + 1.0)
    assert np.allclose(rdata.geoinfos[:, 0], 0.0)  # z, carried


def test_geoinfo_width_matches_what_the_reader_declares(reader):
    """The invariant normalize_geoinfos asserts on, checked on the wrapper's own output."""
    rdata = reader._get(np.int64(0), [0])

    assert rdata.geoinfos.shape[-1] == len(reader.geoinfo_channels)
    assert rdata.geoinfos.shape[-1] == len(reader.geoinfo_idx)
    reader.normalize_geoinfos(rdata.geoinfos)


def test_an_empty_window_averages_to_an_empty_window(reader):
    """Reading empty is normal: past the end of the data, or a window a producer has not emitted.

    This reader sits above the coupling reader on a coupled stream, so an unproduced window
    reaches it as empty rather than as an error.
    """
    reader._wrapped_reader._get = lambda idx, channels_idx: ReaderData.empty(
        len(channels_idx), len(GEOINFOS)
    )

    rdata = reader._get(np.int64(0), [0])

    assert rdata.is_empty()
