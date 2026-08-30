# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Target windows beyond the end of an arbitrary data reader's dataset.

A rollout that runs past the last date in the store gets empty target windows, which degrade to a
two-point spoof; since the decoder only evaluates at the target coordinates, such a step yields two
points instead of a field. This wrapper supplies the missing geometry instead: coordinates and
constant geoinfos persisted from the last window that is fully inside the data, time-varying
geoinfos recomputed for the requested window, and NaN in place of values that do not exist.

Sources are never extended. A source window past the end of the data means the initial condition
itself is missing, and inventing one would feed the model data it has no business seeing.
"""

import dataclasses
import datetime
import logging
import typing

import numpy as np
from numpy.typing import NDArray

from weathergen.datasets.data_reader_base import (
    NPDT64,
    DataReaderBase,
    ReaderData,
    TIndex,
)

_logger = logging.getLogger(__name__)

# Key in the stream config that turns the wrapper on.
EXTENSION_KEY = "extend_beyond_data"

# How far back to look for a readable window when the last in-range one is missing from the store.
_MAX_TEMPLATE_PROBES = 8


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


# Geoinfo channels that vary with time and are therefore recomputed for an extended window.
# Everything else -- z, lsm, slor, sdor and the lat/lon terms -- is constant in time in the store,
# so persisting it from the template is exact rather than an approximation.
_GEOINFO_COMPUTERS = {
    "insolation": _insolation,
    "cos_sza": _insolation,  # the same quantity under the name the observation streams use
    "cos_local_time": _cos_local_time,
    "sin_local_time": _sin_local_time,
    "cos_julian_day": _cos_julian_day,
    "sin_julian_day": _sin_julian_day,
}


@dataclasses.dataclass(frozen=True)
class _Template:
    """The geometry of the last window that is fully inside the data."""

    coords: NDArray[np.float32]
    geoinfos: NDArray[np.float32]
    # datetimes relative to the start of the window they were read from, so that they can be
    # re-stamped into any later window while keeping the structure of a multi-timestep window
    time_offsets: NDArray


class ExtendingReader(DataReaderBase):
    """
    Serve target windows past the end of the wrapped reader's dataset.

    Only `get_target` is extended. `get_source` and `_get` delegate unchanged, so no source value
    is ever invented -- see the module docstring. An extended window carries real coordinates and
    real geoinfos with NaN values, and is flagged `is_extended` so the loss can exclude it while
    the writer can still record what was predicted there.

    Operates in physical units: `collect_datasources` normalizes what a reader returns.
    """

    def __init__(self, wrapped_reader: DataReaderBase) -> None:
        """
        Parameters
        ----------
        wrapped_reader :
            The reader whose target windows are to be extended.

        Returns
        -------
        None
        """

        self._wrapped_reader = wrapped_reader

        super().__init__(wrapped_reader.time_window_handler, wrapped_reader.stream_info)

        # These have to be set on the instance rather than reached through __getattr__: the
        # ABCMeta in utils.better_abc walks dir() after construction and refuses to instantiate
        # while any of them still resolves to the abstract placeholder on the base class.
        self.source_channels = wrapped_reader.source_channels
        self.target_channels = wrapped_reader.target_channels
        self.geoinfo_channels = wrapped_reader.geoinfo_channels
        self.source_idx = wrapped_reader.source_idx
        self.target_idx = wrapped_reader.target_idx
        self.geoinfo_idx = wrapped_reader.geoinfo_idx
        self.target_channel_weights = wrapped_reader.target_channel_weights

        self.mean = wrapped_reader.mean
        self.stdev = wrapped_reader.stdev
        self.mean_geoinfo = wrapped_reader.mean_geoinfo
        self.stdev_geoinfo = wrapped_reader.stdev_geoinfo

        # which geoinfo columns are recomputed rather than persisted, by position
        self._computed = {
            pos: _GEOINFO_COMPUTERS[channel]
            for pos, channel in enumerate(self.geoinfo_channels)
            if channel in _GEOINFO_COMPUTERS
        }

        self._warned_about_source = False
        self._last_real_idx = self._find_last_real_idx()
        self._template = self._read_template()

        name = self.stream_info.get("name")
        if self._template is None:
            _logger.warning(
                f"{name}: no window could be read to extend from, "
                f"{EXTENSION_KEY} has no effect for this stream."
            )
        else:
            persisted = [ch for ch in self.geoinfo_channels if ch not in _GEOINFO_COMPUTERS]
            computed = [ch for ch in self.geoinfo_channels if ch in _GEOINFO_COMPUTERS]
            _logger.info(
                f"{name}: extending target windows after index {self._last_real_idx} with "
                f"{len(self._template.coords)} points; persisting geoinfos {persisted}, "
                f"recomputing {computed}."
            )

    def __getattr__(self, name: str):
        """
        Delegate anything not set here to the wrapped reader.

        Readers carry state the base interface does not describe (data_end_time and period on the
        timestep readers, colnames on the observation reader). Forwarding keeps the wrapper
        transparent, and in particular keeps `data_end_time` reporting where the real data ends.
        Only reached for names that are not found on this instance or its class.
        """
        if name.startswith("_"):
            # Guards against recursion before _wrapped_reader is assigned.
            raise AttributeError(name)
        return getattr(self._wrapped_reader, name)

    @typing.override
    def length(self) -> int:
        return self._wrapped_reader.length()

    @typing.override
    def init_empty(self) -> None:
        self._wrapped_reader.init_empty()

    @typing.override
    def get_source(self, idx: TIndex) -> ReaderData:
        """Sources are never extended; delegate, but say so once if one is out of range."""
        if idx > self._last_real_idx and not self._warned_about_source:
            self._warned_about_source = True
            _logger.warning(
                f"{self.stream_info.get('name')}: source window {idx} lies beyond the end of the "
                "data, so the initial condition does not exist. Sources are not extended -- this "
                "run is misconfigured, and the window falls back to spoofed input."
            )
        return self._wrapped_reader.get_source(idx)

    @typing.override
    def get_target(self, idx: TIndex) -> ReaderData:
        rdata = self._wrapped_reader.get_target(idx)

        # in range, or a window the dataset genuinely has no data for -- neither is ours to fill
        if not rdata.is_empty() or idx <= self._last_real_idx or self._template is None:
            return rdata

        return self._extend(idx)

    @typing.override
    def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
        """
        Delegate unconditionally.

        A raw channel selection cannot be attributed to a side: a stream that reads and predicts
        the same channels has source_idx == target_idx, so comparing against either would
        sometimes fabricate a source. Both public entry points are overridden, and nothing calls
        _get on the outermost reader, so declining to guess costs nothing.
        """
        return self._wrapped_reader._get(idx, channels_idx)

    def _find_last_real_idx(self) -> int:
        """Largest time window index that still lies entirely inside the wrapped dataset."""
        data_end_time = getattr(self._wrapped_reader, "data_end_time", None)
        if data_end_time is None:
            return np.iinfo(np.int64).max

        twh = self.time_window_handler
        # the window at idx spans [t_start + step*idx, ... + window_len]
        return int((data_end_time - twh.t_window_len - twh.t_start) // twh.t_window_step)

    def _read_template(self) -> _Template | None:
        """
        Read the geometry of the last in-range window, once, at construction.

        Priming eagerly rather than on the first empty read keeps this deterministic under
        `num_workers > 0`, where each worker holds its own copy of the reader and may never read an
        in-range window first.
        """
        if self._last_real_idx == np.iinfo(np.int64).max:
            return None

        for probe in range(_MAX_TEMPLATE_PROBES):
            idx = self._last_real_idx - probe
            if idx < 0:
                break
            rdata = self._wrapped_reader.get_target(idx)
            if rdata.is_empty():
                continue

            window_start = self.time_window_handler.window(idx).start
            return _Template(
                coords=np.asarray(rdata.coords, dtype=np.float32).copy(),
                geoinfos=np.asarray(rdata.geoinfos, dtype=np.float32).copy(),
                time_offsets=np.asarray(rdata.datetimes) - window_start,
            )

        return None

    def _extend(self, idx: TIndex) -> ReaderData:
        """Build the target window at `idx` from the template, with no values."""

        template = self._template
        window_start = self.time_window_handler.window(idx).start
        datetimes = window_start + template.time_offsets

        geoinfos = template.geoinfos.copy()
        if self._computed:
            self._recompute_geoinfos(geoinfos, template.coords, datetimes)

        # the values do not exist; NaN is the only way to say so in an array that has to line up
        # with the coordinates row for row
        data = np.full((len(template.coords), len(self.target_idx)), np.nan, dtype=np.float32)

        return ReaderData(
            coords=template.coords.copy(),
            geoinfos=geoinfos,
            data=data,
            datetimes=datetimes,
            is_spoof=False,
            is_extended=True,
        )

    def _recompute_geoinfos(
        self, geoinfos: NDArray[np.float32], coords: NDArray[np.float32], datetimes: NDArray
    ) -> None:
        """Fill the time-varying geoinfo columns in place, one call per distinct datetime."""

        lats, lons = coords[:, 0], coords[:, 1]

        for stamp in np.unique(datetimes):
            rows = datetimes == stamp
            date = _to_datetime(stamp)
            for pos, compute in self._computed.items():
                geoinfos[rows, pos] = compute(date, lats[rows], lons[rows])


def _to_datetime(stamp: NPDT64) -> datetime.datetime:
    """numpy datetime64 to a plain datetime, which the solar formulas expect."""
    # np.datetime64(...) keeps this a scalar; going through np.asarray would yield a 0-d array
    # whose .astype(datetime) is another 0-d array rather than a datetime.
    return np.datetime64(stamp, "s").astype(datetime.datetime)
