# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Scheduled per-channel offsets on top of an arbitrary data reader.

Used for perturbation experiments: "what does the model do if SST is 2 K warmer from day 4 of
the rollout onward". The offset is declared per channel in the stream config and applied by a
wrapper around the stream's real reader, so nothing downstream needs to know about it.
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
    DType,
    NPTDel64,
    ReaderData,
    TIndex,
)

_logger = logging.getLogger(__name__)

# Key in the stream config that carries the per-channel schedule.
ELEVATION_KEY = "elevation"


@dataclasses.dataclass(frozen=True)
class Elevation:
    """A single channel's schedule: add `offset` to every value from `from_time` onward."""

    offset: float
    from_time: NPDT64


def _as_datetime64(value) -> NPDT64:
    """Parse a `from_date` config value, which YAML may hand over as str or as a datetime."""
    if isinstance(value, np.datetime64):
        return value
    if isinstance(value, datetime.datetime | datetime.date):
        return np.datetime64(value)
    # OmegaConf stores unquoted timestamps as strings; np.datetime64 insists on the 'T' separator
    return np.datetime64(str(value).strip().replace(" ", "T"))


def parse_elevations(
    stream_info: dict,
    channels: list[str],
    channels_idx: list[int],
    anchor_time: NPDT64,
    forecast_time_step: NPTDel64,
) -> dict[int, Elevation]:
    """
    Resolve the stream config's `elevation` block for one set of channels.

    The schedule is given per channel as `{offset: <float>, step: <int>}`, where `step` is the
    forecast step from which the offset applies. It is resolved here, once, into the absolute
    time `anchor_time + step * forecast_time_step`; `from_date` may be given instead to state
    that time directly.

    Parameters
    ----------
    stream_info :
        the stream config, which may or may not carry an `elevation` key
    channels :
        channel names, parallel to channels_idx
    channels_idx :
        the channels' indices in the underlying dataset, used as the returned keys
    anchor_time :
        start of the run, i.e. forecast step 0
    forecast_time_step :
        time between two forecast steps

    Returns
    -------
    Schedule keyed by dataset channel index. Empty if nothing is configured.
    """

    elevations = stream_info.get(ELEVATION_KEY)
    if not elevations:
        return {}

    schedule = {}
    for channel, idx in zip(channels, channels_idx, strict=True):
        entry = elevations.get(channel, None)
        if entry is None:
            continue

        if "offset" not in entry:
            raise ValueError(f"{ELEVATION_KEY} entry for channel '{channel}' has no 'offset'.")
        if "step" in entry and "from_date" in entry:
            raise ValueError(
                f"{ELEVATION_KEY} entry for channel '{channel}' sets both 'step' and 'from_date'; "
                "they are two spellings of the same threshold, so give only one."
            )

        if "from_date" in entry:
            from_time = _as_datetime64(entry["from_date"])
        else:
            from_time = anchor_time + int(entry.get("step", 0)) * forecast_time_step

        schedule[idx] = Elevation(offset=float(entry["offset"]), from_time=from_time)

    return schedule


class ElevatingReader(DataReaderBase):
    """
    Wraps a data reader and adds a configured constant to selected channels.

    The offset is applied in the reader's own physical units, i.e. before the normalization
    that happens further down in collect_datasources and ForcingInput._collect_forcing_data,
    so a configured 2.0 on a temperature channel really is 2 K.

    The schedule is resolved against absolute time, not per-sample lead time. For an inference
    rollout, which has a single initial condition at the run's start_date, that is exactly
    "from forecast step N onward". In training every sample starts at a different time, so a
    step-derived threshold there means "from the fixed calendar date that step N maps to",
    the same date for every sample. Use `from_date` when that is what is meant.
    """

    def __init__(
        self,
        wrapped_reader: DataReaderBase,
        anchor_time: NPDT64,
        forecast_time_step: NPTDel64,
    ) -> None:
        """
        Parameters
        ----------
        wrapped_reader :
            the stream's real reader, which supplies both the data and all of the metadata
        anchor_time :
            start of the run, i.e. the time of forecast step 0
        forecast_time_step :
            time between two forecast steps
        """

        self._wrapped_reader = wrapped_reader

        super().__init__(wrapped_reader.time_window_handler, wrapped_reader.stream_info)

        # These have to be copied onto the instance rather than read off the wrapped reader on
        # demand: the ABCMeta in utils.better_abc walks dir() after construction and refuses to
        # instantiate while any of them still resolves to the abstract placeholder on the base.
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

        # Source and target are resolved separately: readers that draw the two sides from
        # different files (fesom, mesh) number their channels per file, so an index means
        # nothing without knowing which side it came from.
        self._source_elevations = parse_elevations(
            self.stream_info,
            self.source_channels,
            self.source_idx,
            anchor_time,
            forecast_time_step,
        )
        self._target_elevations = parse_elevations(
            self.stream_info,
            self.target_channels,
            self.target_idx,
            anchor_time,
            forecast_time_step,
        )

        configured = self.stream_info.get(ELEVATION_KEY) or {}
        known = set(self.source_channels) | set(self.target_channels)
        unmatched = [ch for ch in configured if ch not in known]
        if unmatched:
            _logger.info(
                f"Unmatched {ELEVATION_KEY} channels in {self.stream_info.get('name')}: {unmatched}"
            )

        for side, schedule in (
            ("source", self._source_elevations),
            ("target", self._target_elevations),
        ):
            for idx, elevation in schedule.items():
                _logger.info(
                    f"{self.stream_info.get('name')}: elevating {side} channel index {idx} by "
                    f"{elevation.offset} from {elevation.from_time}."
                )

    @typing.override
    def length(self) -> int:
        return self._wrapped_reader.length()

    @typing.override
    def init_empty(self) -> None:
        self._wrapped_reader.init_empty()

    @typing.override
    def get_source(self, idx: TIndex) -> ReaderData:
        rdata = self._wrapped_reader.get_source(idx)
        return self._elevate(rdata, self.source_idx, self._source_elevations, idx)

    @typing.override
    def get_target(self, idx: TIndex) -> ReaderData:
        rdata = self._wrapped_reader.get_target(idx)
        return self._elevate(rdata, self.target_idx, self._target_elevations, idx)

    @typing.override
    def normalize_source_channels(self, source: NDArray[DType]) -> NDArray[DType]:
        return self._wrapped_reader.normalize_source_channels(source)

    @typing.override
    def normalize_target_channels(self, target: NDArray[DType]) -> NDArray[DType]:
        return self._wrapped_reader.normalize_target_channels(target)

    @typing.override
    def denormalize_source_channels(self, source: NDArray[DType]) -> NDArray[DType]:
        return self._wrapped_reader.denormalize_source_channels(source)

    @typing.override
    def denormalize_target_channels(self, data: NDArray[DType]) -> NDArray[DType]:
        return self._wrapped_reader.denormalize_target_channels(data)

    @typing.override
    def normalize_geoinfos(self, geoinfos: NDArray[DType]) -> NDArray[DType]:
        return self._wrapped_reader.normalize_geoinfos(geoinfos)

    @typing.override
    def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
        raise NotImplementedError(
            "This is a decorator use the public interface (get_source / get_target)"
        )

    def _elevate(
        self,
        rdata: ReaderData,
        channels_idx: list[int],
        elevations: dict[int, Elevation],
        idx: TIndex,
    ) -> ReaderData:
        """Add each due channel's offset to the data read for time window `idx`."""

        if not elevations or rdata.is_empty():
            return rdata

        window_start = self.time_window_handler.window(idx).start
        active = [
            (pos, elevations[ch].offset)
            for pos, ch in enumerate(channels_idx)
            if ch in elevations and window_start >= elevations[ch].from_time
        ]
        if not active:
            return rdata

        # Copy rather than write in place: a reader is free to hand back a view into a buffer it
        # keeps, and an in-place addition would corrupt it and compound over reads.
        positions = np.fromiter((pos for pos, _ in active), dtype=np.intp, count=len(active))
        offsets = np.fromiter(
            (offset for _, offset in active), dtype=rdata.data.dtype, count=len(active)
        )
        data = rdata.data.copy()
        data[:, positions] += offsets

        return dataclasses.replace(rdata, data=data)
