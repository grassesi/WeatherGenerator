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

# z and insolation are the averaged ones (AVERAGING_GEOINFOS); the rest are carried over
# from the last datapoint in the window. They are deliberately not adjacent, so a wrapper
# that returns [averaged, rest] permutes the columns between them.
GEOINFOS = ["z", "lsm", "slor", "sdor", "insolation"]
N_POINTS = 3
SAMPLES_PER_WINDOW = 2


class FakeReader(DataReaderTimestep):
    """Two samples per window at the same coordinates, so averaging has something to do."""

    def __init__(self, twh: TimeWindowHandler) -> None:
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
        self.geoinfo_channels = list(GEOINFOS)
        self.geoinfo_idx = list(range(len(GEOINFOS)))
        self.mean = np.zeros(1, dtype=np.float32)
        self.stdev = np.ones(1, dtype=np.float32)
        self.mean_geoinfo = np.zeros(len(GEOINFOS), dtype=np.float32)
        self.stdev_geoinfo = np.ones(len(GEOINFOS), dtype=np.float32)

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
        geoinfos = np.tile(np.arange(len(GEOINFOS), dtype=np.float32), (len(coords), 1))
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
    # the carried-over columns keep the last sample's values, which are position + 2
    for name in ["lsm", "slor", "sdor"]:
        col = GEOINFOS.index(name)
        assert np.allclose(rdata.geoinfos[:, col], col + 2.0), f"{name} is not in column {col}"
    # the averaged columns sit at their declared positions, halfway between the two samples
    for name in ["z", "insolation"]:
        col = GEOINFOS.index(name)
        assert np.allclose(rdata.geoinfos[:, col], col + 1.0), f"{name} is not in column {col}"


def test_averaged_channels_are_averaged_and_the_rest_are_not(reader):
    rdata = reader._get(np.int64(0), [0])

    z, insolation = GEOINFOS.index("z"), GEOINFOS.index("insolation")
    # sample values are v and v + 2, so a mean sits at v + 1 and a carry-over at v + 2
    assert np.allclose(rdata.geoinfos[:, z], z + 1.0)
    assert np.allclose(rdata.geoinfos[:, insolation], insolation + 1.0)
    assert np.allclose(rdata.geoinfos[:, GEOINFOS.index("lsm")], 1 + 2.0)


def test_geoinfo_width_matches_what_the_reader_declares(reader):
    """The invariant normalize_geoinfos asserts on, checked on the wrapper's own output."""
    rdata = reader._get(np.int64(0), [0])

    assert rdata.geoinfos.shape[-1] == len(reader.geoinfo_channels)
    assert rdata.geoinfos.shape[-1] == len(reader.geoinfo_idx)
    reader.normalize_geoinfos(rdata.geoinfos)
