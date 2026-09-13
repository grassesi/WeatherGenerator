# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Present a fixed-resolution stream on a finer time grid.

A dataset with a period coarser than the model's time window step carries no data in most
windows: `get_dataset_indexes_timestep` returns no rows for a window that falls between two
timesteps, so a 24 h stream read on a 6 h grid is empty three windows in four. The motivating
case is coupled inference, where an ocean component predicts SST once per 24 h while the
atmosphere consuming it as a dynamic forcing asks every 6 h, and an empty forcing window is not
a stale field but an absent one, which the atmosphere never saw in training.

This wrapper maps each requested window onto the source sample covering it -- a zero-order hold
on a known grid, not a search for the last window that happened to carry data. A source sample
that is genuinely missing therefore stays missing: the wrapper never reaches past the covering
sample to fill a gap or to run off the end of the dataset.

The staleness is real and is the cost of the approach: with a 24 h source and a 6 h consumer,
three of every four windows see a field up to 18 h old. That is far closer to what the model
trained on than an absent field, but it is not the same thing, and the served field is piecewise
constant where training data varied smoothly.
"""

import logging
import typing

import numpy as np
from numpy.typing import NDArray

from weathergen.datasets.data_reader_base import (
    DataReaderTimestep,
    DType,
    ReaderData,
    TIndex,
)
from weathergen.datasets.geoinfo import computed_columns, recompute_geoinfos

_logger = logging.getLogger(__name__)

_ZERO = np.timedelta64(0, "s")


class UpsamplingReader(DataReaderTimestep):
    """
    Wraps a fixed-period reader and serves every window from the source sample covering it.

    The served values are re-stamped onto the requested window. That is what a zero-order hold
    means -- the value is presented as the current best estimate of the field, not as an
    observation from 18 h ago -- and it is also what keeps the data inside the window the
    consumer is tokenizing. Handing back the original timestamps would put every served point
    outside the requested window, where the tokenizer's time binning may drop it, turning a
    served forcing back into an absent one by a longer route.
    """

    def __init__(self, wrapped_reader: DataReaderTimestep) -> None:
        """
        Parameters
        ----------
        wrapped_reader :
            the reader to upsample; supplies the data, the time grid and all of the metadata
        """

        self._wrapped_reader = wrapped_reader

        super().__init__(
            wrapped_reader.time_window_handler,
            wrapped_reader.stream_info,
            wrapped_reader.data_start_time,
            wrapped_reader.data_end_time,
            wrapped_reader.period,
        )

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

        self._stride = self._resolve_stride()
        self._anchor_idx = self._resolve_anchor()

        # Geoinfos are not all static: insolation and the cyclic time terms are functions of the
        # window's time, so serving them unchanged would state the wrong time of day by up to a
        # full source period. These columns are recomputed for the window actually being served;
        # everything else (z, lsm, slor, sdor) is constant in time and is served as it comes.
        self._computed_geoinfos = computed_columns(list(self.geoinfo_channels or []))

        self._announce()

    def _resolve_stride(self) -> int:
        """How many consecutive windows one source sample serves."""

        name = self.stream_info.get("name")
        step = self.time_window_handler.t_window_step

        if self.period is None:
            msg = (
                f"Stream '{name}' has no period, so the windows one of its samples covers "
                "cannot be determined."
            )
            raise ValueError(msg)

        if self.period < step:
            msg = (
                f"Stream '{name}' has period {self.period}, finer than the window step {step}; "
                "there is nothing to upsample."
            )
            raise ValueError(msg)

        if self.period % step != _ZERO:
            # a window would straddle two source samples, and neither of them is the answer
            msg = (
                f"Stream '{name}' has period {self.period}, which is not a whole multiple of "
                f"the window step {step}."
            )
            raise ValueError(msg)

        return int(self.period // step)

    def _resolve_anchor(self) -> int:
        """Window index the source grid starts on, i.e. its phase against the handler origin."""

        name = self.stream_info.get("name")
        handler = self.time_window_handler
        offset = self.data_start_time - handler.t_start

        if offset % handler.t_window_step != _ZERO:
            msg = (
                f"Stream '{name}' starts at {self.data_start_time}, which is {offset} from the "
                f"timeline origin {handler.t_start} and not a whole number of window steps."
            )
            raise ValueError(msg)

        return int(offset // handler.t_window_step)

    def _announce(self) -> None:
        """One line per run, so a log can be checked for what the upsampling actually did."""

        name = self.stream_info.get("name")
        step = self.time_window_handler.t_window_step

        if self._stride == 1:
            _logger.info(
                f"UpsamplingReader on '{name}': source period {self.period} equals the window "
                f"step {step}, serving every window unchanged."
            )
            return

        _logger.info(
            f"UpsamplingReader on '{name}': source period {self.period} over window step "
            f"{step}, so one source sample serves {self._stride} windows."
        )
        if self._computed_geoinfos:
            varying = [self.geoinfo_channels[pos] for pos in self._computed_geoinfos]
            _logger.info(
                f"UpsamplingReader on '{name}': recomputing time-varying geoinfos {varying} "
                "for every served window."
            )

    def _covering(self, idx: TIndex) -> TIndex:
        """Index of the window holding the source sample that covers window `idx`."""
        return idx - ((idx - self._anchor_idx) % self._stride)

    @typing.override
    def length(self) -> int:
        return self._wrapped_reader.length()

    @typing.override
    def init_empty(self) -> None:
        self._wrapped_reader.init_empty()

    @typing.override
    def get_source(self, idx: TIndex) -> ReaderData:
        src = self._covering(idx)
        rdata = self._wrapped_reader.get_source(src)
        if src == idx or rdata.is_empty():
            return rdata
        return self._restamp(rdata, src, idx)

    @typing.override
    def get_target(self, idx: TIndex) -> ReaderData:
        # Targets are scored against the time they are valid for, so serving one on a window it
        # was not observed in would fabricate truth. Only the source side is a forcing.
        return self._wrapped_reader.get_target(idx)

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

    def _restamp(self, rdata: ReaderData, source_idx: TIndex, target_idx: TIndex) -> ReaderData:
        """Move a source window's timestamps onto the requested window.

        Shifts by the difference between the two window starts rather than overwriting, so
        structure inside the window (several observation times, say) survives the move.
        """

        handler = self.time_window_handler
        shift = handler.window(target_idx).start - handler.window(source_idx).start

        # the wrapped reader may hand out its stored arrays; never mutate them in place
        coords = rdata.coords.copy()
        geoinfos = rdata.geoinfos.copy()
        datetimes = (rdata.datetimes + np.asarray(shift)).copy()

        # the served values keep their own time only in the sense that they are the last known
        # state of the field; the geoinfos describe the window being served, so they follow the
        # new stamps rather than the old ones
        recompute_geoinfos(geoinfos, coords, datetimes, self._computed_geoinfos)

        return ReaderData(
            coords=coords,
            geoinfos=geoinfos,
            data=rdata.data.copy(),
            datetimes=datetimes,
            is_spoof=rdata.is_spoof,
        )
