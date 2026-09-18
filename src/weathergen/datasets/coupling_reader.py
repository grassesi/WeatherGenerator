# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The one reader every dynamic forcing stream is read through.

`DataReaderCoupling` resolves a forcing request on the *consumer's* timeline, shifted by the
stream's forcing lag, and gathers the source windows that lie inside it. Where the rows come
from -- the consumer's own dataset, the producing component's ground truth, or the producing
component's dispatched predictions -- is a separate question, answered by `is_forced` and the
initialization time. The window arithmetic is the same either way, which is what makes a lag
mean the same thing in training and in coupled inference (`forcing_lag_design.md` L6, L7).

It lives here rather than beside the `Coupler` because the training path needs it too, and
`weathergen.common.coupling` imports the `Trainer`.
"""

from __future__ import annotations

import dataclasses
import logging
import typing

import numpy as np
import torch
from numpy.typing import NDArray

from weathergen.common.config import parse_timedelta, timedelta_to_str
from weathergen.datasets.data_reader_base import (
    NPDT64,
    DataReaderBase,
    DataReaderTimestep,
    DTRange,
    NPTDel64,
    ReaderData,
    TimeWindowHandler,
    TIndex,
    restamp,
    t_epsilon,
)
from weathergen.datasets.tokenizer_utils import TIMES_WIDTH
from weathergen.model.chunking import ChunkInfo

if typing.TYPE_CHECKING:
    from weathergen.datasets.batch import ModelBatch
    from weathergen.model.model import ModelOutput

logger = logging.getLogger(__name__)

_ZERO = np.timedelta64(0, "ms")


@dataclasses.dataclass
class ForcingProvenance:
    """Where a forcing stream's rows actually came from, tallied over a whole run.

    The point of counting is that every failure mode of the coupled exchange is silent: an
    unresolved window falls through to a climatological spoof at `logger.debug`, and a window
    resolved against the wrong timeline reads back empty exactly the same way. Wiring
    assertions -- couplings live, files written -- pass through all of it. These counts do not.

    Counted per *source window*, not per request: one request on a 24 h window against a 6 h
    producer resolves to four of them, and collapsing that to one number is what hid the
    defect the gathering fixes.
    """

    stream: str
    requests: int = 0
    predicted: int = 0
    primed: int = 0
    disk: int = 0
    unresolved: int = 0
    held: int = 0

    @property
    def resolved(self) -> int:
        return self.predicted + self.primed + self.disk

    def describe(self, consumer: str | None = None) -> str:
        who = f"'{self.stream}'" if consumer is None else f"'{self.stream}' of '{consumer}'"
        return (
            f"Forcing {who}: {self.requests} request(s) -> {self.predicted} predicted, "
            f"{self.primed} primed, {self.disk} from disk, {self.unresolved} unresolved "
            f"({self.held} held from a covering window)."
        )


@dataclasses.dataclass
class _StoredChunk:
    """One dispatched rollout chunk, indexed by the valid time of each emitted window.

    `expected` is `len(tile.predicted_steps)` taken off the tile the chunk arrived with, not
    counted off a step list whose first tile is padded down to global step 0 and whose last
    may be short.
    """

    index: int
    expected: int
    windows: dict[NPDT64, ReaderData]

    @property
    def is_complete(self) -> bool:
        return len(self.windows) == self.expected


class DataReaderCoupling(DataReaderTimestep):
    """Serve a dynamic forcing stream, from disk or from another component's predictions.

    Implements only the interface required by ForcingInput: `stream_info`, `get_source`,
    `normalize_source_channels`, `normalize_geoinfos` and `get_geoinfo_size`. The consumer-side
    members those need (`source_idx`, `mean`, `stdev`, `geoinfo_idx`, `mean_geoinfo`,
    `stdev_geoinfo`) are taken from the consumer's own reader for the same stream, so a coupled
    forcing is tokenized and normalized exactly like the real stream it stands in for.

    **Two timelines.** Requests arrive on the consumer's index space, and are resolved on
    `request_handler` -- the consumer's handler shifted earlier by the stream's forcing lag.
    The rows live on the *source* timeline: the producer's handler when a producer supplies
    them, the consumer's own when they come from disk. A request window gathers every source
    window whose start lies inside it, so a consumer asking for a 24 h window against a
    6 h producer receives four rows, exactly as its own disk reader would have returned, and
    the stream's own `AveragingReader` above then performs the reduction it was trained
    through. Only when the gather is empty -- the source is coarser than the request window --
    is the covering window taken and restamped.
    """

    def __init__(
        self,
        dataset: DataReaderBase,
        producer_stream: str,
        producer: DataReaderBase | None = None,
        request_handler: TimeWindowHandler | None = None,
        is_forced: bool = True,
        init_time: NPDT64 | None = None,
        forecast_step_stride: int = 1,
        max_chunks: int = 2,
        provenance: ForcingProvenance | None = None,
    ) -> None:
        """
        Parameters
        ----------
        dataset :
            The consumer's own reader for this stream. Supplies the tokenization and
            normalization context the forcing has to be dressed up in, and the rows themselves
            when `is_forced` is False.
        producer_stream :
            Name the stream carries in the producing component, i.e. the key its predictions
            are stored under in `ModelOutput.physical`.
        producer :
            The producing component's reader for the stream, used to bring its predictions
            back to physical space and to serve the initial condition. If omitted, producer and
            consumer are assumed to share statistics for these channels.
        request_handler :
            The timeline requests are resolved on: the consumer's own handler shifted earlier
            by this stream's forcing lag. Defaults to the consumer's handler, i.e. no lag.
        is_forced :
            Whether a partner component supplies this stream. False is the training and
            uncoupled-inference case: the same window arithmetic and the same lag, with the
            rows read from the consumer's own dataset.
        init_time :
            Start of the trajectory's initialization window. A source window at or before it is
            an initial condition and comes from the producer's own data however far into the
            rollout it is requested. Pushed in per batch; it cannot be learned from the first
            dispatched chunk, because the first request precedes the first dispatch.
        forecast_step_stride :
            Dataset indices advanced per forecast step on the producer side. Only a fallback:
            a dispatched chunk carries its own `ChunkInfo`, which is preferred.
        max_chunks :
            Dispatched chunks kept. Two -- the current one and the one before it -- covers
            every lag up to `chunk_length`, because a request window's left edge is always
            served by the preceding chunk.
        provenance :
            Tally to record into. Shared with the readers this one replaces on the next batch,
            so the counts span the run rather than one trajectory.

        Returns
        -------
        None
        """

        super().__init__(
            dataset.time_window_handler,
            dataset.stream_info,
            dataset.data_start_time,
            dataset.data_end_time,
            dataset.period,
        )

        self._producer_stream = producer_stream
        self._is_forced = bool(is_forced)
        self._init_time = init_time
        self._stride = int(forecast_step_stride)
        self._max_chunks = max(1, int(max_chunks))
        self._length = dataset.length()
        self.provenance = provenance or ForcingProvenance(stream=producer_stream)

        # A coupled stream is gridded and periodic on both sides. That is an assumption, not
        # something the type system enforces -- `DataReaderObs` has no period at all -- so it is
        # asserted here, where the message can name the stream, rather than surfacing later as a
        # missing attribute inside a wrapper.
        for role, reader in (("consumer", dataset), ("producer", producer)):
            if reader is not None and getattr(reader, "period", None) is None:
                msg = (
                    f"Coupled stream {producer_stream!r}: its {role} reader "
                    f"{type(reader).__name__} is not periodic. A forcing window is resolved by "
                    "gathering the source windows inside it, so both sides must be gridded "
                    "readers with a period."
                )
                raise ValueError(msg)

        # The timeline a request is resolved on: the consumer's, shifted by the forcing lag.
        self._request_handler = request_handler or dataset.time_window_handler
        # The timeline the rows live on. The producer's when it supplies them -- predictions
        # and primed ground truth alike are stamped on its grid -- the consumer's otherwise.
        self._source_handler = (
            producer.time_window_handler if producer is not None else dataset.time_window_handler
        )
        # Gathering only exists to reconcile two grids. With no producer there is one grid, and
        # the consumer's own disk reader is the authority on what a window holds -- it already
        # returns every sample inside it. Gathering there would concatenate overlapping windows
        # and, at a lag finer than the window step, reach the very window being predicted.
        self._gathers = producer is not None

        if self._gathers:
            source = producer.time_window_handler
            if source.t_window_len > source.t_window_step:
                msg = (
                    f"Coupled stream {producer_stream!r}: its producer's windows are "
                    f"{timedelta_to_str(source.t_window_len)} long on a "
                    f"{timedelta_to_str(source.t_window_step)} step, so consecutive emissions "
                    "overlap and a request spanning several of them would count the shared "
                    "points twice."
                )
                raise ValueError(msg)

        # consumer side: how ForcingInput will tokenize and normalize what we return
        self.source_channels = list(dataset.source_channels)
        self.source_idx = list(dataset.source_idx)
        self.geoinfo_channels = list(dataset.geoinfo_channels)
        self.geoinfo_idx = list(dataset.geoinfo_idx)
        self.mean = dataset.mean
        self.stdev = dataset.stdev
        self.mean_geoinfo = dataset.mean_geoinfo
        self.stdev_geoinfo = dataset.stdev_geoinfo

        # a coupled forcing stream is never a prediction target of its consumer
        self.target_channels = []
        self.target_idx = []
        self.target_channel_weights = []

        self._resolve_producer_columns(dataset, producer)

        # Dispatched chunks, oldest first. Each is published atomically by `add_chunk`, so a
        # window is never read out of a chunk that is still being emitted.
        self._chunks: list[_StoredChunk] = []
        self._dispatched = 0

        # The consumer's own reader: the metadata mirrored above, and the rows themselves when
        # nothing produces this stream.
        self._dataset = dataset
        # Windows at or before the init time are primed from the PRODUCER's target data: a
        # primed window has to be the same quantity a prediction is, on the same grid and
        # selectable by the same channel map, or priming and prediction reach the consumer by
        # two independent routes that agree only by coincidence.
        self._producer = producer

        # where the producer's geoinfos sit in the target tokens it hands over
        offset = 1 + TIMES_WIDTH
        offered = len(self._geoinfo_channels_offered)
        self._geoinfo_slice = slice(offset, offset + offered)

    # ------------------------------------------------------------------ construction

    @property
    def disk_reader(self) -> DataReaderBase:
        """The consumer's own reader for this stream, i.e. what this one was built on.

        Rebuilding a coupling reader -- binding a producer to one the training path already
        wrapped -- must start from this, never from the wrapper, or a run ends up reading a
        coupling reader through a coupling reader.
        """
        return self._dataset

    @property
    def request_handler(self) -> TimeWindowHandler:
        """The lagged timeline requests are resolved on."""
        return self._request_handler

    @property
    def is_forced(self) -> bool:
        return self._is_forced

    def _resolve_producer_columns(
        self, dataset: DataReaderBase, producer: DataReaderBase | None
    ) -> None:
        """Check the producer supplies what the consumer needs, and say which columns to take.

        Both halves of a window go through this: the data channels the consumer sources, and
        the geoinfos that travel with them. Neither has to match the producer column for
        column -- the consumer's channels only have to be a *subset* of what the producer
        offers, and are selected by name. An ocean predicting only `sst` can force an
        atmosphere that sources only `sst`; an atmosphere predicting seventy channels can
        force an ocean that sources three of them.
        """

        stats = producer if producer is not None else dataset
        if producer is None:
            # producer and consumer are assumed to share statistics for these channels
            data_channels, data_idx = list(dataset.source_channels), list(dataset.source_idx)
        else:
            data_channels, data_idx = list(producer.target_channels), list(producer.target_idx)

        self._pred_cols, self._pred_mean, self._pred_stdev = self._select_columns(
            "channels",
            needed=list(self.source_channels),
            offered=data_channels,
            # `mean`/`stdev` are indexed by dataset channel, reached through the offered idx
            mean_of=lambda col: stats.mean[data_idx[col]],
            stdev_of=lambda col: stats.stdev[data_idx[col]],
            guard_zero_stdev=False,
        )

        # what the producer writes into its target tokens, which sets the slice width
        self._geoinfo_channels_offered = list(stats.geoinfo_channels or [])

        self._geo_cols, self._geo_mean, self._geo_stdev = self._select_columns(
            "geoinfo channels",
            needed=list(self.geoinfo_channels or []),
            offered=self._geoinfo_channels_offered,
            # `mean_geoinfo`/`stdev_geoinfo` are indexed by position, as normalize_geoinfos does
            mean_of=lambda col: stats.mean_geoinfo[col],
            stdev_of=lambda col: stats.stdev_geoinfo[col],
            guard_zero_stdev=True,
        )

    def _select_columns(
        self,
        kind: str,
        needed: list[str],
        offered: list[str],
        mean_of,
        stdev_of,
        guard_zero_stdev: bool,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Positions of `needed` within `offered`, with the statistics to denormalize them."""

        missing = [ch for ch in needed if ch not in offered]
        if missing:
            msg = (
                f"Coupled stream '{self._producer_stream}' does not supply the {kind} its "
                f"consumer needs as forcing: missing {missing}, offered {offered}."
            )
            raise ValueError(msg)

        cols = np.asarray([offered.index(ch) for ch in needed], dtype=np.int64)
        mean = np.asarray([mean_of(col) for col in cols], dtype=np.float32)
        stdev = np.asarray([stdev_of(col) for col in cols], dtype=np.float32)
        if guard_zero_stdev:
            # constant fields are centered but not scaled; mirrors normalize_geoinfos
            stdev = np.where(np.isclose(stdev, 0.0), 1.0, stdev)

        return cols, mean, stdev

    def _slice_geoinfos(self, source_data, fstep: int, num_points: int) -> NDArray:
        """Recover the producer's geoinfos for one forecast step from its source tokens.

        Requires the column layout to not have changed.
        """

        coords_local = source_data.target_coords[fstep]
        geoinfos = _to_numpy(coords_local[..., self._geoinfo_slice])

        expected = (num_points, len(self._geoinfo_channels_offered))
        assert geoinfos.shape == expected, (
            f"Coupled stream '{self._producer_stream}': sliced geoinfos of shape "
            f"{geoinfos.shape} at forecast step {fstep}, expected {expected}. Rows come from "
            "the source sample's tokenized target coords and points from the target sample's "
            "raw ones, so a row mismatch means the two no longer describe the same points; a "
            "column mismatch means get_target_coords_local's layout moved away from "
            "TIMES_WIDTH."
        )

        # take the consumer's subset, then back to physical space; the consumer normalizes
        # again with its own statistics, exactly as it would for data read from disk
        return geoinfos[:, self._geo_cols] * self._geo_stdev + self._geo_mean

    # ------------------------------------------------------------------ serving

    def length(self) -> int:
        """Length of the timeline this reader is defined on, not of what it holds."""
        return self._length

    def set_init_time(self, init_time: NPDT64 | None) -> None:
        """Declare the trajectory's initialization window (G5)."""
        self._init_time = init_time

    def _source_idxs_within(self, win: DTRange) -> list[TIndex]:
        """Source windows whose start lies in `[win.start, win.end)`, ascending."""

        handler = self._source_handler
        step = handler.t_window_step
        # ceil, so a source window starting exactly on the left edge is included; floor of the
        # open right edge, so one starting exactly on it is not
        first = -((handler.t_start - win.start) // step)
        last = (win.end - t_epsilon - handler.t_start) // step
        return [np.int64(j) for j in range(int(first), int(last) + 1)]

    def _covering_source_idx(self, when: NPDT64) -> TIndex:
        """Source window holding `when`, i.e. the last one starting at or before it."""
        handler = self._source_handler
        return np.int64((when - handler.t_start) // handler.t_window_step)

    def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
        """Resolve one forcing request: gather the source windows inside it, or hold one.

        `idx` is an index on the *consumer's* timeline. It is resolved against
        `request_handler`, which carries the stream's forcing lag, so the window returned is
        the partner state the consumer's model was trained to be forced with -- never the
        index as the producer would have read it.
        """

        win = self._request_handler.window(idx)
        self.provenance.requests += 1

        if self._gathers:
            source_idxs = self._source_idxs_within(win)
            if source_idxs:
                parts = [self._fetch(j, channels_idx) for j in source_idxs]
                rdata = _concatenate(parts, len(self.source_idx), len(self.geoinfo_idx))
                return self._clip(rdata, win)

        # Either the source is coarser than the request window, so nothing starts inside it, or
        # there is only one grid and the request names one window on it. Hold the covering
        # window: the value is the last known state of the field, presented at the time it is
        # being served for (`upsampling.py`), not as an observation from 18 h ago.
        #
        # Quantising *backwards* is what keeps a lag finer than the source step honest. Rounding
        # to the nearest window would let a 3 h lag on a 6 h grid serve the window the consumer
        # is predicting, i.e. no lag at all.
        src = self._covering_source_idx(win.start)
        rdata = self._fetch(src, channels_idx)
        if rdata.is_empty():
            return rdata

        source_start = self._source_handler.window(src).start
        if source_start == win.start:
            return self._clip(rdata, win)

        self.provenance.held += 1
        shift = win.start - source_start
        logger.debug(
            f"Stream '{self._producer_stream}': request {win.start} .. {win.end} held from "
            f"source window {source_start} (shift {shift})."
        )
        return self._clip(restamp(rdata, shift, list(self.geoinfo_channels or [])), win)

    def _clip(self, rdata: ReaderData, win: DTRange) -> ReaderData:
        """Drop rows outside the requested window, which `check_reader_data` insists on."""

        if rdata.is_empty():
            return rdata

        inside = (rdata.datetimes >= win.start) & (rdata.datetimes < win.end)
        if inside.all():
            return rdata

        return dataclasses.replace(
            rdata,
            coords=rdata.coords[inside],
            geoinfos=rdata.geoinfos[inside],
            data=rdata.data[inside],
            datetimes=rdata.datetimes[inside],
        )

    def _fetch(self, src: TIndex, channels_idx: list[int]) -> ReaderData:
        """One source window, from wherever this stream's rows come from (D3).

        | | source of rows |
        | --- | --- |
        | not forced | the consumer's own dataset |
        | forced, at or before the init time | the producer's targets |
        | forced, beyond it | the producer's dispatched predictions |
        """

        if not self._is_forced:
            self.provenance.disk += 1
            return self._dataset._get(src, channels_idx)

        start = self._source_handler.window(src).start

        if self._init_time is not None:
            # An initial condition, however far into the rollout it is requested: a request
            # reaching back across the init window is a normal state of the first chunk, and
            # of every chunk when the lag is long.
            if start <= self._init_time:
                self.provenance.primed += 1
                return self._prime(src)
        elif not self._chunks:
            # Before the first batch nobody has declared an init time yet, and nothing has been
            # dispatched. Serving ground truth is what an uncoupled run would have read.
            self.provenance.primed += 1
            return self._prime(src)

        window = self._lookup(start)
        if window is not None:
            self.provenance.predicted += 1
            # A window is served once per consumer of it, so never hand out the stored arrays
            return window.copy()

        self.provenance.unresolved += 1
        logger.debug(
            f"No prediction for stream '{self._producer_stream}' at {start} (source index "
            f"{src}), forcing falls back to spoof."
        )
        return ReaderData.empty(len(self.source_idx), len(self.geoinfo_idx))

    def _lookup(self, valid_time: NPDT64) -> ReaderData | None:
        """The stored prediction for a valid time, newest chunk first."""

        for stored in reversed(self._chunks):
            assert stored.is_complete, (
                f"Coupled stream '{self._producer_stream}': chunk {stored.index} holds "
                f"{len(stored.windows)} windows but its tile predicts {stored.expected}. A "
                "chunk is published atomically, so serving a partial one means add_chunk "
                "raised part-way and the store was left inconsistent."
            )
            window = stored.windows.get(valid_time)
            if window is not None:
                return window

        return None

    def _prime(self, src: TIndex) -> ReaderData:
        """The producer's ground truth for a source window, dressed as the consumer's source.

        Runs through the same `_pred_cols` map a prediction does, so the two paths differ only
        in where the numbers came from.
        """

        if self._producer is None:
            # no producer to read from: fall back to the consumer's own view of the stream
            return self._dataset.get_source(src)

        rdata = self._producer.get_target(src)
        if rdata.is_empty():
            return ReaderData.empty(len(self.source_idx), len(self.geoinfo_idx))

        data = rdata.data[:, self._pred_cols]
        geoinfos = rdata.geoinfos[:, self._geo_cols] if len(self._geo_cols) else rdata.geoinfos
        return dataclasses.replace(rdata, data=data, geoinfos=geoinfos)

    # ------------------------------------------------------------------ dispatch

    def add_chunk(self, chunk: ModelOutput, batch: ModelBatch) -> None:
        """Index the predictions of one rollout chunk by the time they are valid for.

        `batch` is needed because a ModelOutput carries only the predicted values: the
        coordinates, times and geoinfos they live on sit on the batch's samples, split
        across the source and target halves as `_lower_prediction` describes.

        The chunk is built locally and assigned once, so `get_source` never reads out of a
        chunk that is still being emitted (G4).
        """

        tile = self._producer_tile(chunk)
        windows: dict[NPDT64, ReaderData] = {}

        for fstep in chunk.forecast_steps:
            preds = chunk.get_physical_prediction(chunk.chunk_idx(fstep), self._producer_stream)
            if preds is None:
                # leading empty steps of the first chunk, or a stream without a decoder
                continue

            for i_source, pred in enumerate(preds):
                valid_time, window = self._lower_prediction(
                    chunk.batch_idx(fstep), i_source, pred, batch, tile
                )
                windows[valid_time] = window

        expected = len(tile.predicted_steps) if tile is not None else len(windows)
        stored = _StoredChunk(index=self._dispatched, expected=expected, windows=windows)
        assert stored.is_complete, (
            f"Coupled stream '{self._producer_stream}': chunk {stored.index} emitted "
            f"{len(windows)} windows, but its tile predicts {expected} steps "
            f"{tile.predicted_steps if tile is not None else ()}. A consumer would read a "
            "gap as an absent forcing rather than as a missing emission."
        )

        # atomic publish, oldest first, bounded by max_chunks
        self._chunks = [*self._chunks, stored][-self._max_chunks :]
        self._dispatched += 1

    def _producer_tile(self, chunk: ModelOutput) -> ChunkInfo | None:
        """The producer's own description of the chunk it just emitted.

        The timeline a prediction is stamped on and the stride between its forecast steps
        belong to the *producing* component. They used to be constructor arguments of this
        reader, which is how they came to be left unset: `subscribe()` filled three of six.
        Riding along inside the ModelOutput, they cannot be forgotten -- so prefer them, and
        fall back to what the constructor was given only when a chunk carries no tile.
        """

        tile = getattr(chunk, "chunk", None)
        if tile is None or tile.time_window_handler is None:
            return None

        return tile

    def _lower_prediction(
        self,
        fstep: int,
        i_source: int,
        pred: torch.Tensor,
        batch: ModelBatch,
        tile: ChunkInfo | None = None,
    ) -> tuple[NPDT64, ReaderData]:
        """Pair one prediction with its target geometry and bring it to physical space.

        The geometry is split across two samples, so both are needed. `target_coords_raw`,
        `target_times_raw` and `idxs_inv` are written only by `add_target_values`, which runs
        under `target_select`; the tokenized `target_coords` the geoinfos are recovered from
        is written only by `add_target_coords`, which runs under `source_select`
        (`multi_stream_data_sampler.py:694-697`). Neither sample carries both halves -- the
        other half is left at its empty `StreamData.__init__` default, so reading it off the
        wrong sample yields nothing rather than raising.

        Zipping the two rests on their rows describing the same points in the same order,
        which holds because both are tokenized from the same windows under the same target
        mask. Nothing states that invariant, so the row count is checked below.
        """

        i_target = batch.get_target_idx_for_source(i_source)
        stream_data = batch.get_target_sample(i_target).streams_data.get(self._producer_stream)
        if (
            stream_data is None
            or stream_data.is_spoof(fstep)
            or len(stream_data.target_coords_raw[fstep]) == 0
        ):
            raise ValueError("Cannot pair prediction with its target geometry")

        source_data = batch.get_source_sample(i_source).streams_data.get(self._producer_stream)
        if source_data is None:
            raise ValueError(
                f"Coupled stream '{self._producer_stream}' has no source sample {i_source}, so "
                "the geoinfos its prediction carries cannot be recovered."
            )

        coords = _to_numpy(stream_data.target_coords_raw[fstep])
        times = np.asarray(stream_data.target_times_raw[fstep])
        geoinfos = self._slice_geoinfos(source_data, fstep, len(coords))

        # ensemble members are equivalent forcings, use their mean
        data = pred.mean(dim=0).to(torch.float32).detach().cpu().numpy()

        assert data.shape[0] == coords.shape[0] == times.shape[0], (
            f"Prediction for '{self._producer_stream}' at step {fstep} has {data.shape[0]} points "
            f"but {coords.shape[0]} coordinates and {times.shape[0]} times."
        )

        # restore the ordering of the original data, as the output writer does
        idxs_inv = stream_data.idxs_inv[fstep]
        if len(idxs_inv) > 0:
            idxs_inv = _to_numpy(idxs_inv)
            data, coords, times = data[idxs_inv], coords[idxs_inv], times[idxs_inv]
            geoinfos = geoinfos[idxs_inv]

        # select the channels the consumer expects and denormalize them
        data = data[:, self._pred_cols] * self._pred_stdev + self._pred_mean

        if tile is not None:
            valid_idx = tile.window_idx(stream_data.sample_idx, fstep)
            valid_time = tile.time_window_handler.window(valid_idx).start
        else:
            valid_idx = stream_data.sample_idx + fstep * self._stride
            valid_time = self._source_handler.window(valid_idx).start

        return valid_time, ReaderData(
            coords=coords.astype(np.float32),
            geoinfos=geoinfos.astype(np.float32),
            data=data.astype(np.float32),
            datetimes=times.astype("datetime64[ns]"),
        )


def _to_numpy(tensor) -> NDArray:
    """Detach a tensor to a numpy array; pass numpy arrays through."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def _concatenate(parts: list[ReaderData], num_data: int, num_geo: int) -> ReaderData:
    """Stack the source windows a request gathered into one window.

    The body of `IOReaderData.combine`, over `ReaderData` and without its spoof accounting:
    nothing that reaches here is a spoof, because a spoofed producer window aborts the run at
    `_lower_prediction` and a spoofed disk window is created above this reader, not below it.
    """

    parts = [part for part in parts if not part.is_empty()]
    if not parts:
        return ReaderData.empty(num_data, num_geo)
    if len(parts) == 1:
        return parts[0]

    return ReaderData(
        coords=np.concatenate([p.coords for p in parts]),
        geoinfos=np.concatenate([p.geoinfos for p in parts]),
        data=np.concatenate([p.data for p in parts]),
        datetimes=np.concatenate([p.datetimes for p in parts]),
    )


def forcing_lag(stream_info: dict, window_step: NPTDel64, forecast_offset: int) -> NPTDel64:
    """How far before the window it predicts a component samples this forcing stream.

    A property of the *model*, so it is read off the stream's own config and travels in the
    checkpoint: the dynamic-forcing response is trained at a particular lag, and the same lag
    must hold through finetuning and inference. A coupled run validates it; it never overrides
    it (`forcing_lag_design.md` L3).

    Absent, or the documented alias `step`, means `forecast_offset * window_step` -- what the
    code did before the key existed. Deriving the default from `forecast_offset` rather than
    hard-coding one window leaves the `offset: 0` path alone.
    """

    raw = stream_info.get("forcing_lag") if stream_info is not None else None
    if raw is None or (isinstance(raw, str) and raw.strip() == "step"):
        return np.timedelta64(int(forecast_offset) * window_step, "ms")

    lag = parse_timedelta(raw)
    if lag < _ZERO:
        name = (stream_info or {}).get("name", "?")
        msg = (
            f"Stream '{name}' has a negative forcing_lag {raw!r}. A forcing is sampled before "
            "the window it forces, never after it."
        )
        raise ValueError(msg)

    return lag
