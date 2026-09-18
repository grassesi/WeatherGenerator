# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Tests for the recomputation of time-varying geoinfo channels.

These pin the properties that make the reimplementation usable as a stand-in for the columns
anemoi-datasets bakes into a zarr, including the two quirks it reproduces on purpose, and the
window policy `recompute_geoinfos` owns: which columns are carried, which are evaluated at a
time, and which are averaged over the window.
"""

import datetime

import numpy as np
import pytest
from numpy.typing import NDArray

from weathergen.datasets.geoinfo import (
    computed_columns,
    recompute_geoinfos,
    to_datetime,
)

LATS = np.array([0.0, 45.0, -30.0, 80.0])
LONS = np.array([0.0, 90.0, -120.0, 180.0])


def test_computed_columns_picks_only_the_time_varying_channels():
    channels = ["z", "insolation", "lsm", "cos_local_time", "sdor", "sin_julian_day"]

    computed = computed_columns(channels)

    assert sorted(computed) == [1, 3, 5]


def test_computed_columns_is_empty_without_time_varying_channels():
    assert computed_columns(["z", "lsm", "slor", "sdor"]) == {}
    assert computed_columns([]) == {}


def test_to_datetime_returns_a_scalar_datetime():
    stamp = np.datetime64("2023-06-21T12:30:00")

    result = to_datetime(stamp)

    assert isinstance(result, datetime.datetime)
    assert (result.year, result.month, result.day, result.hour) == (2023, 6, 21, 12)


def _recompute(channels: list[str], stamp: str, lats=LATS, lons=LONS) -> NDArray:
    geoinfos = np.full((len(lats), len(channels)), np.nan, dtype=np.float64)
    coords = np.stack([lats, lons], axis=1)
    datetimes = np.full(len(lats), np.datetime64(stamp))

    return recompute_geoinfos(geoinfos, coords, datetimes, channels)


def test_insolation_is_clipped_at_zero_and_bounded_by_one():
    values = _recompute(["insolation"], "2023-06-21T12:00:00")[:, 0]

    assert (values >= 0.0).all()
    assert (values <= 1.0).all()


def test_insolation_is_zero_on_the_night_side():
    # local midnight at lon 0
    values = _recompute(
        ["insolation"], "2023-06-21T00:00:00", lats=np.array([0.0]), lons=np.array([0.0])
    )

    assert values[0, 0] == pytest.approx(0.0)


def test_insolation_peaks_near_the_subsolar_point():
    # northern summer solstice, local noon at lon 0: highest sun over the tropics, not the pole
    values = _recompute(
        ["insolation"], "2023-06-21T12:00:00", lats=np.array([23.5, 0.0, 80.0]), lons=np.zeros(3)
    )[:, 0]

    assert values[0] > values[1] > values[2]


def test_the_hour_angle_steps_in_whole_hours():
    """
    The upstream implementation builds its hour angle from `date.hour` alone, so minutes move
    the result only through the much weaker declination term. That is a quirk, but it is what
    the stored column contains, so it is reproduced deliberately.
    """
    on_the_hour = _recompute(["insolation"], "2023-03-15T09:00:00")
    almost_the_next_hour = _recompute(["insolation"], "2023-03-15T09:59:00")
    the_next_hour = _recompute(["insolation"], "2023-03-15T10:00:00")

    within_the_hour = np.abs(almost_the_next_hour - on_the_hour).max()
    across_the_hour = np.abs(the_next_hour - on_the_hour).max()

    assert within_the_hour < 0.01 * across_the_hour


def test_local_time_terms_lie_on_the_unit_circle():
    values = _recompute(["cos_local_time", "sin_local_time"], "2023-03-15T09:00:00")

    np.testing.assert_allclose(values[:, 0] ** 2 + values[:, 1] ** 2, 1.0)


def test_local_time_terms_vary_with_longitude():
    values = _recompute(["cos_local_time"], "2023-03-15T09:00:00")[:, 0]

    assert len(np.unique(values)) == len(LONS)


def test_local_time_advances_over_the_day():
    morning = _recompute(["cos_local_time", "sin_local_time"], "2023-03-15T06:00:00")
    evening = _recompute(["cos_local_time", "sin_local_time"], "2023-03-15T18:00:00")

    # half a day apart is half a turn of the diurnal cycle
    np.testing.assert_allclose(morning, -evening, atol=1e-12)


def test_julian_day_terms_are_uniform_over_space():
    values = _recompute(["cos_julian_day", "sin_julian_day"], "2023-03-15T09:00:00")

    assert len(np.unique(values[:, 0])) == 1
    assert len(np.unique(values[:, 1])) == 1
    np.testing.assert_allclose(values[:, 0] ** 2 + values[:, 1] ** 2, 1.0)


def test_julian_day_terms_are_seasonal_not_diurnal():
    """Within a day they barely move; over a season they swing across their whole range."""
    morning = _recompute(["cos_julian_day"], "2023-03-15T00:00:00")
    evening = _recompute(["cos_julian_day"], "2023-03-15T18:00:00")
    months_later = _recompute(["cos_julian_day"], "2023-06-15T00:00:00")

    within_a_day = np.abs(evening - morning).max()
    across_seasons = np.abs(months_later - morning).max()

    assert within_a_day < 0.02
    assert across_seasons > 1.0


def test_static_columns_are_left_alone():
    channels = ["z", "insolation", "lsm"]
    geoinfos = np.tile(np.array([123.0, -999.0, 1.0]), (len(LATS), 1))
    coords = np.stack([LATS, LONS], axis=1)
    datetimes = np.full(len(LATS), np.datetime64("2023-03-15T09:00:00"))

    out = recompute_geoinfos(geoinfos, coords, datetimes, channels)

    np.testing.assert_allclose(out[:, 0], 123.0)
    np.testing.assert_allclose(out[:, 2], 1.0)
    assert (out[:, 1] != -999.0).all()


def test_rows_with_different_stamps_are_computed_separately():
    channels = ["cos_local_time"]
    geoinfos = np.full((2, 1), np.nan)
    coords = np.zeros((2, 2))
    datetimes = np.array(
        [np.datetime64("2023-03-15T00:00:00"), np.datetime64("2023-03-15T12:00:00")]
    )

    out = recompute_geoinfos(geoinfos, coords, datetimes, channels)

    # same coordinates, different times, so the two rows must differ
    assert out[0, 0] != out[1, 0]
    np.testing.assert_allclose(out[0, 0], -out[1, 0], atol=1e-12)


def test_recompute_without_computed_columns_is_a_noop():
    geoinfos = np.full((len(LATS), 2), 5.0)
    coords = np.stack([LATS, LONS], axis=1)
    datetimes = np.full(len(LATS), np.datetime64("2023-03-15T09:00:00"))

    out = recompute_geoinfos(geoinfos, coords, datetimes, ["z", "lsm"])

    np.testing.assert_allclose(out, 5.0)


# --------------------------------------------------------------------------- the window policy


def _window(stamps: list[str], channels: list[str], stored: NDArray | None = None):
    """A gridded window: the same points once per stamp, sample-major, as a reader returns it."""
    n = len(LATS)
    coords = np.tile(np.stack([LATS, LONS], axis=1), (len(stamps), 1))
    datetimes = np.repeat(np.array(stamps, dtype="datetime64[ns]"), n)
    if stored is None:
        stored = np.zeros((n * len(stamps), len(channels)))
    return stored, coords, datetimes


def test_recompute_does_not_write_to_its_argument():
    """Pure: a wrapped reader may hand out its own array, so it must survive untouched."""
    channels = ["cos_local_time", "insolation"]
    geoinfos, coords, datetimes = _window(["2023-03-15T00:00", "2023-03-15T06:00"], channels)
    before = geoinfos.copy()

    recompute_geoinfos(geoinfos, coords, datetimes, channels)
    recompute_geoinfos(geoinfos, coords, datetimes, channels, datetimes[0])

    np.testing.assert_array_equal(geoinfos, before)


def test_collapsing_keeps_the_rows_at_the_stamp_and_evaluates_phases_there():
    stamps = ["2023-03-15T00:00", "2023-03-15T06:00", "2023-03-15T12:00", "2023-03-15T18:00"]
    channels = ["z", "cos_local_time"]
    stored = np.tile(np.array([[7.0, np.nan]]), (len(LATS) * len(stamps), 1))
    geoinfos, coords, datetimes = _window(stamps, channels, stored)

    out = recompute_geoinfos(geoinfos, coords, datetimes, channels, np.datetime64(stamps[0]))

    assert out.shape == (len(LATS), len(channels)), "one row per point, not per sample"
    np.testing.assert_allclose(out[:, 0], 7.0)
    np.testing.assert_allclose(out[:, 1], _recompute(["cos_local_time"], stamps[0])[:, 0])


def test_collapsed_insolation_is_the_mean_over_the_window():
    """insolation is averaged, not evaluated at the stamp: the mean is a real quantity, the
    snapshot at 00Z would be zero over half the globe."""
    stamps = ["2023-03-15T00:00", "2023-03-15T06:00", "2023-03-15T12:00", "2023-03-15T18:00"]
    n = len(LATS)
    # stored values that differ per sample and per point, so a wrong mean cannot pass by luck
    stored = np.concatenate([np.arange(n)[:, None] + 10.0 * k for k in range(len(stamps))])
    geoinfos, coords, datetimes = _window(stamps, ["insolation"], stored)

    out = recompute_geoinfos(geoinfos, coords, datetimes, ["insolation"], np.datetime64(stamps[0]))

    np.testing.assert_allclose(out[:, 0], np.arange(n) + 15.0)


def test_a_window_whose_samples_carry_different_points_is_refused():
    """Averaging pairs rows by position, so unequal samples would mix different points."""
    geoinfos, coords, datetimes = _window(["2023-03-15T00:00", "2023-03-15T06:00"], ["insolation"])

    with pytest.raises(AssertionError, match="expected"):
        recompute_geoinfos(
            geoinfos[:-1], coords[:-1], datetimes[:-1], ["insolation"], datetimes[0]
        )
