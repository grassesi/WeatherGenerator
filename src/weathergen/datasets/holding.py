# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Zero-order hold on top of an arbitrary data reader.

Serves a fine window cadence from a source that only carries data at a coarse one, by
repeating the last valid step until the next one arrives. The motivating case is coupled
inference: an ocean component predicts SST once per 24 h, while the atmosphere consuming it
as a dynamic forcing asks every 6 h. Three of every four windows are then empty, and an
empty forcing window is not a stale field -- it is an absent one, which the atmosphere never
saw in training, where its SST came from a 6-hourly dataset at every single step.
"""

import logging
import typing

import numpy as np
from numpy.typing import NDArray

from weathergen.datasets.data_reader_base import (
    DataReaderBase,
    DType,
    ReaderData,
    TIndex,
)
from weathergen.datasets.geoinfo import computed_columns, recompute_geoinfos

_logger = logging.getLogger(__name__)


class HoldingReader(DataReaderBase):
    """
    Wraps a data reader and fills empty windows with the most recent non-empty one.

    A window is filled only if a valid one exists within `max_hold` windows behind it, so a
    stream that genuinely stops producing still reports empty rather than being held forever.
    Nothing is invented: an empty window before the first valid one stays empty.

    The held values are re-stamped onto the requested window. That is what a zero-order hold
    means -- the value is presented as the current best estimate of the field, not as an
    observation from 18 h ago -- and it is also what keeps the data inside the window the
    consumer is tokenizing. Handing back the original timestamps would put every held point
    outside the requested window, where the tokenizer's time binning may drop it, turning a
    held forcing back into an absent one by a longer route.

    The staleness is real and is the cost of the approach: with a 24 h producer and a 6 h
    consumer, three of every four steps see a field up to 18 h old. That is far closer to
    what the model trained on than an absent field, but it is not the same thing, and the
    held field is piecewise constant where training data varied smoothly.
    """

    def __init__(self, wrapped_reader: DataReaderBase, max_hold: int) -> None:
        """
        Parameters
        ----------
        wrapped_reader :
            the reader to hold over; supplies the data and all of the metadata
        max_hold :
            how many consecutive empty windows may be filled from one valid window. For a
            24 h source read at 6 h this is 3. Must be >= 1.
        """

        if max_hold < 1:
            msg = f"max_hold must be >= 1, got {max_hold}."
            raise ValueError(msg)

        self._wrapped_reader = wrapped_reader
        self._max_hold = int(max_hold)
        self._held = 0

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

        # Geoinfos are not all static: insolation and the cyclic time terms are functions of
        # the window's time, so holding them unchanged would state the wrong time of day by
        # up to a full diurnal cycle. These columns are recomputed for the window actually
        # being served; everything else (z, lsm, slor, sdor) is constant in time and is
        # persisted exactly.
        self._computed_geoinfos = computed_columns(list(self.geoinfo_channels or []))
        if self._computed_geoinfos:
            varying = [self.geoinfo_channels[pos] for pos in self._computed_geoinfos]
            _logger.info(
                f"HoldingReader on '{self.stream_info.get('name')}': recomputing "
                f"time-varying geoinfos {varying} for every held window."
            )

    @typing.override
    def length(self) -> int:
        return self._wrapped_reader.length()

    @typing.override
    def init_empty(self) -> None:
        self._wrapped_reader.init_empty()

    @typing.override
    def get_source(self, idx: TIndex) -> ReaderData:
        return self._hold(idx, self._wrapped_reader.get_source)

    @typing.override
    def get_target(self, idx: TIndex) -> ReaderData:
        # Targets are scored against the time they are valid for, so holding one would
        # fabricate truth. Only the source side is a forcing.
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

    @property
    def held_windows(self) -> int:
        """How many windows this reader has filled by holding, over its lifetime."""
        return self._held

    def _hold(self, idx: TIndex, read) -> ReaderData:
        """Return the window at idx, or the most recent non-empty one within max_hold."""

        rdata = read(idx)
        if not rdata.is_empty():
            return rdata

        for back in range(1, self._max_hold + 1):
            previous = idx - back
            if previous < 0:
                break

            rdata = read(previous)
            if rdata.is_empty():
                continue

            self._held += 1
            _logger.debug(
                f"Holding stream '{self.stream_info.get('name')}' at window {idx} from "
                f"window {previous} ({back} window(s) back)."
            )
            return self._restamp(rdata, previous, idx)

        return rdata

    def _restamp(self, rdata: ReaderData, source_idx: TIndex, target_idx: TIndex) -> ReaderData:
        """Move a held window's timestamps onto the requested window.

        Shifts by the difference between the two window starts rather than overwriting, so
        structure inside the window (several observation times, say) survives the move.
        """

        handler = self.time_window_handler
        shift = handler.window(target_idx).start - handler.window(source_idx).start

        # the wrapped reader may hand out its stored arrays; never mutate them in place
        coords = rdata.coords.copy()
        geoinfos = rdata.geoinfos.copy()
        datetimes = (rdata.datetimes + np.asarray(shift)).copy()

        # the held values keep their own time only in the sense that they are the last known
        # state of the field; the geoinfos describe the window being served, so they follow
        # the new stamps rather than the old ones
        recompute_geoinfos(geoinfos, coords, datetimes, self._computed_geoinfos)

        return ReaderData(
            coords=coords,
            geoinfos=geoinfos,
            data=rdata.data.copy(),
            datetimes=datetimes,
            is_spoof=rdata.is_spoof,
        )
