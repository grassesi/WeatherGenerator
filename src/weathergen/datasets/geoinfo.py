# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Recomputing the time-varying geoinfo channels of a window.

Geoinfos are not all static: alongside z, lsm, slor and sdor an anemoi stream carries
insolation and the cyclic time terms, which are functions of the window's time and the
points' coordinates. Any wrapper that serves one window's values at another window's time
has to recompute those columns, or it hands the model a stale time of day -- an error of up
to a full diurnal cycle, in the channels whose whole purpose is to say where in that cycle
the model is.

Taken from `datasets/extension.py` on the `dataset-extension` branch, which needs the same
thing to serve windows past the end of the data. Lifted into its own module so the two
wrappers share one implementation rather than each carrying a copy; when that branch lands it
should import from here.
"""

import datetime

import numpy as np
from numpy.typing import NDArray

# --------------------------------------------------------------------------------------------
# Computed geoinfos.
#
# These reproduce the "computed forcings" that anemoi-datasets bakes into a zarr at build time, so
# that an extended window carries the same quantities as a window read from the store. They are
# reimplemented rather than imported: earthkit-meteo is only an optional extra of anemoi-datasets
# and is not a dependency of this package. The quirks below are deliberate -- they are what the
# stored columns actually contain.
# --------------------------------------------------------------------------------------------

# earthkit.meteo.solar.array.solar, DAYS_PER_YEAR
_DAYS_PER_YEAR = 365.25


def _julian_day(date: datetime.datetime) -> float:
    """Day of year as a float. After earthkit.meteo.solar.array.solar.julian_day."""
    year_start = datetime.datetime(date.year, 1, 1, tzinfo=date.tzinfo)
    delta = date - year_start
    return delta.days + delta.seconds / 86400.0


def _solar_declination_angle(date: datetime.datetime) -> tuple[float, float]:
    """
    Solar declination and time correction, both in degrees.

    After earthkit.meteo.solar.array.solar.solar_declination_angle.
    """
    angle = _julian_day(date) / _DAYS_PER_YEAR * np.pi * 2

    declination = (
        0.396372
        - 22.91327 * np.cos(angle)
        + 4.025430 * np.sin(angle)
        - 0.387205 * np.cos(2 * angle)
        + 0.051967 * np.sin(2 * angle)
        - 0.154527 * np.cos(3 * angle)
        + 0.084798 * np.sin(3 * angle)
    )
    time_correction = (
        0.004297
        + 0.107029 * np.cos(angle)
        - 1.837877 * np.sin(angle)
        - 0.837378 * np.cos(2 * angle)
        - 2.340475 * np.sin(2 * angle)
    )
    return declination, time_correction


def _cos_solar_zenith_angle(
    date: datetime.datetime, lats: NDArray, lons: NDArray
) -> NDArray[np.float64]:
    """
    Cosine of the solar zenith angle, clipped at zero.

    After earthkit.meteo.solar.array.solar.cos_solar_zenith_angle. Two properties of that
    implementation are reproduced on purpose, because the stored columns have them:
    the result is clipped at 0 (so the night side is 0, despite what its docstring says), and the
    hour angle uses whole hours only, ignoring minutes.

    This is what the `insolation` channel of an anemoi dataset holds: earthkit-data's forcings
    source defines insolation as an alias of this function, so the values are dimensionless in
    [0, 1] rather than a radiative flux.
    """
    declination, time_correction = _solar_declination_angle(date)

    declination = np.deg2rad(declination)
    lats_rad = np.deg2rad(lats)

    sindec_sinlat = np.sin(declination) * np.sin(lats_rad)
    cosdec_coslat = np.cos(declination) * np.cos(lats_rad)

    solar_angle = np.deg2rad((date.hour - 12) * 15 + lons + time_correction)
    zenith_angle = sindec_sinlat + cosdec_coslat * np.cos(solar_angle)

    return np.clip(zenith_angle, 0.0, None)


def _local_time(date: datetime.datetime, lons: NDArray) -> NDArray[np.float64]:
    """Local solar time in hours. After earthkit-data's ForcingMaker.local_time."""
    day_start = datetime.datetime(date.year, date.month, date.day, tzinfo=date.tzinfo)
    delta = date - day_start
    hours_since_midnight = (delta.days + delta.seconds / 86400.0) * 24
    return (lons / 360.0 * 24.0 + hours_since_midnight) % 24


def _cos_local_time(date, lats, lons):
    return np.cos(_local_time(date, lons) / 24 * np.pi * 2)


def _sin_local_time(date, lats, lons):
    return np.sin(_local_time(date, lons) / 24 * np.pi * 2)


def _cos_julian_day(date, lats, lons):
    return np.full(lons.shape, np.cos(_julian_day(date) / _DAYS_PER_YEAR * np.pi * 2))


def _sin_julian_day(date, lats, lons):
    return np.full(lons.shape, np.sin(_julian_day(date) / _DAYS_PER_YEAR * np.pi * 2))


def _insolation(date, lats, lons):
    return _cos_solar_zenith_angle(date, lats, lons)


# Geoinfo channels that vary with time and are therefore recomputed rather than carried.
# Everything else -- z, lsm, slor, sdor and the lat/lon terms -- is constant in time in the store,
# so carrying it from any sample of the window is exact rather than an approximation.
_GEOINFO_COMPUTERS = {
    "insolation": _insolation,
    "cos_local_time": _cos_local_time,
    "sin_local_time": _sin_local_time,
    "cos_julian_day": _cos_julian_day,
    "sin_julian_day": _sin_julian_day,
}

# Of the time-varying channels, the ones a collapsed window is represented by the *mean* of
# rather than by the value at one instant. insolation is a clipped cosine, so it is linear in
# the solar flux and its window mean is proportional to mean insolation over the window -- a
# real quantity, still varying with latitude and season. The others are phases, and a window
# spanning a whole cycle averages them to nothing: four samples at 00/06/12/18Z are equally
# spaced around the local-time circle and sum to the null vector. Those take the value at the
# stamp instead.
_WINDOW_MEAN_GEOINFOS = frozenset({"insolation"})


def to_datetime(stamp) -> datetime.datetime:
    """numpy datetime64 to a plain datetime, which the solar formulas expect."""
    # np.datetime64(...) keeps this a scalar; going through np.asarray would yield a 0-d array
    # whose .astype(datetime) is another 0-d array rather than a datetime.
    return np.datetime64(stamp, "s").astype(datetime.datetime)


def computed_columns(geoinfo_channels: list[str]) -> dict[int, object]:
    """Position -> computer, for the geoinfo channels that vary with time."""
    return {
        pos: _GEOINFO_COMPUTERS[channel]
        for pos, channel in enumerate(geoinfo_channels)
        if channel in _GEOINFO_COMPUTERS
    }


def recompute_geoinfos(
    geoinfos: NDArray[np.float32],
    coords: NDArray[np.float32],
    datetimes: NDArray,
    geoinfo_channels: list[str],
    stamp: np.datetime64 | None = None,
) -> NDArray[np.float32]:
    """Return the geoinfos a window should carry, given every row the window holds.

    The whole geoinfo policy, so a reader that reshapes a window keeps no per-channel
    bookkeeping of its own. Each column is one of three kinds: **static** (z, lsm, slor, sdor)
    is carried unchanged, **phase** (the cyclic time terms) is evaluated at the row's own time,
    and **window mean** (insolation) is averaged over the window -- see `_WINDOW_MEAN_GEOINFOS`.

    With `stamp` None every row is kept and its computed columns are evaluated at its own
    datetime: the restamping case, where a window is served at a time other than the one it was
    read for. With `stamp` given the window collapses onto the rows carrying that datetime.

    Pure: the argument is never written to, so a caller may pass a wrapped reader's own array
    without copying it first. A new array comes back in every case.
    """
    channels = list(geoinfo_channels or [])
    computed = {
        pos: channel for pos, channel in enumerate(channels) if channel in _GEOINFO_COMPUTERS
    }

    if stamp is None:
        out = geoinfos.copy()
        # One evaluation per distinct datetime: the solar terms are scalar in time, so the work
        # scales with the number of stamps in the window, not with the number of points.
        for value in np.unique(datetimes):
            rows = datetimes == value
            _evaluate_into(out, rows, coords, to_datetime(value), computed)
        return out

    rows = datetimes == stamp
    assert rows.any(), f"no datapoint at {stamp} to collapse the window onto"

    out = geoinfos[rows].copy()
    _evaluate_into(out, slice(None), coords[rows], to_datetime(stamp), computed)

    for pos, channel in computed.items():
        if channel in _WINDOW_MEAN_GEOINFOS:
            out[:, pos] = _window_mean(geoinfos[:, pos], datetimes, len(out))

    return out


def _evaluate_into(out: NDArray, rows, coords: NDArray, date, computed: dict[int, str]) -> None:
    """Fill the computed columns of `rows` from the coordinates, at `date`."""
    if not computed:
        return
    lats, lons = coords[:, 0], coords[:, 1]
    for pos, channel in computed.items():
        out[rows, pos] = _GEOINFO_COMPUTERS[channel](date, lats, lons)


def _window_mean(column: NDArray, datetimes: NDArray, n_points: int) -> NDArray[np.float64]:
    """Mean of one stored column over the window, per point.

    Accumulates the stamps' blocks rather than grouping by coordinate, which is the same
    alignment the caller already relies on: every sample of a gridded window carries the same
    points in the same order. The assert makes that assumption fail loudly instead of averaging
    two different points together.
    """
    stamps = np.unique(datetimes)
    total = np.zeros(n_points, dtype=np.float64)
    for value in stamps:
        block = column[datetimes == value]
        assert len(block) == n_points, (
            f"window sample at {value} has {len(block)} points, expected {n_points}; "
            "averaging assumes every sample of the window carries the same points"
        )
        total += block
    return total / len(stamps)
