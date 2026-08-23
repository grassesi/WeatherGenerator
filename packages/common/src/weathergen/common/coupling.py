from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from omegaconf import OmegaConf

import weathergen.common.config as config
from weathergen.common.logger import init_loggers
from weathergen.datasets.batch import ModelBatch
from weathergen.datasets.data_reader_base import (
    DataReaderBase,
    ReaderData,
    TimeWindowHandler,
    TIndex,
)
from weathergen.model.forcing import ForcingInput
from weathergen.model.model import ModelOutput
from weathergen.train.trainer import Trainer

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ModelCheckpoint:
    _checkpoint_str: dataclasses.InitVar[str]
    run_id: str = dataclasses.field(init=False)
    mini_epoch: int = dataclasses.field(init=False)
    model_timestep: int #!?

    def __post_init__(self, _checkpoint_str: str):
        run_id, mini_epoch = _checkpoint_str.split("@")
        self.run_id = run_id
        self.mini_epoch = int(mini_epoch)

        # peek config for istep
        self.istep = config.load_merge_configs(
            None, self.run_id, self.mini_epoch
        ).general.istep

    def get_component(
        self,
        private_config: Path | None,
        configs: list[Path] | None,
        options: list[str],
        shared_config: config.Config
    ) -> tuple[Trainer, config.Config]:
        options_overwrite = config.from_cli_arglist(options)
        cf = config.load_merge_configs(
            private_config,
            self.run_id,
            self.mini_epoch,
            None,
            *configs,
            options_overwrite,
        )
        cf = OmegaConf.merge(cf, shared_config)
        return Trainer(cf.logging), cf

    def launch_inference(self):
        trainer = Trainer()

@dataclasses.dataclass
class Coupling:
    name: str
    producer: str
    consumer: str
    stream: str

@dataclasses.dataclass
class Couplings:
    couplings: dict[str, Coupling]
    checkpoints: dict[str, ModelCheckpoint]

    @classmethod
    def from_args(cls, couplings: Path, components: list[str]):
        components_kv = (component.split("=") for component in components)
        components = {
            component: ModelCheckpoint(checkpoint_str) for component, checkpoint_str in components_kv
        }
        parsed_couplings = {
            name: Coupling(**coupling)
            for name, coupling in OmegaConf.load(couplings).items()
        }

        return cls(parsed_couplings, components)

    def run(
        self,
        private_config: Path,
        run_id: str,
        configs: list[Path],
        options: list[str]
    ):
        # global setup
        cf = OmegaConf.create({"general": {}})
        cf = config.set_run_id(cf, run_id, False)
        devices = Trainer.init_torch()
        cf = Trainer.init_ddp(cf)
        init_loggers(cf.general.run_id)
        logger.info(f"DDP initialization: rank={cf.rank}, world_size={cf.world_size}")
        cf.general.run_history += [
            [(checkpoint.run_id, checkpoint.istep) for checkpoint in self.checkpoints]
        ]

        # instantiate all components
        components = {
            name: checkpoint.get_component(
                private_config, None, None, cf,
            ) for name, checkpoint in self.checkpoints.items()
        }

        producers = {
            name: [
                coupling for coupling in self.couplings.values()
                if name == coupling.consumer
            ]
            for name in components.keys()
        }

        # run all components
        for name, component in components.items():
            trainer, ccf = component
            checkpoint = self.checkpoints[name]
            # TODO: make sure training artifacts (checkpoints, configs) dont collide
            trainer.inference(
                config, ccf, devices, checkpoint.run_id, checkpoint.mini_epoch, name=name
            )


def _to_numpy(tensor) -> NDArray:
    """Detach a tensor to a numpy array; pass numpy arrays through."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


@dataclasses.dataclass(frozen=True)
class _PredictedWindow:
    """One producer prediction, lowered to the data model a reader hands out.

    `data` is in physical space and already ordered like the consumer's source
    channels, so that ForcingInput can normalize it with the consumer's own
    statistics exactly as it would normalize data read from disk.
    """

    coords: NDArray[np.float32]
    data: NDArray[np.float32]
    datetimes: NDArray[np.datetime64]


class DataReaderCoupling(DataReaderBase):
    """Serve another component's predicted chunks as forcing input.

    Implements only the interface required by ForcingInput: `stream_info`,
    `get_source`, `normalize_source_channels`, `normalize_geoinfos` and
    `get_geoinfo_size`. The consumer-side members those need (`source_idx`,
    `mean`, `stdev`, `geoinfo_idx`, `mean_geoinfo`, `stdev_geoinfo`) are taken
    from the consumer's own reader for the same stream, so a coupled forcing is
    tokenized and normalized exactly like the real stream it stands in for.

    Predictions arrive through `add_chunk` and are indexed by the time window
    they are valid for. `get_source` translates the requested time index with
    the consumer's own `TimeWindowHandler`, which keeps producer and consumer
    aligned even when they do not share a dataset index origin. A window that
    has not been produced (yet) reads back empty, which makes ForcingInput fall
    back to spoofed forcing rather than fail -- the normal state at the first
    rollout step, before the producer has run.
    """

    def __init__(
        self,
        dataset: DataReaderBase,
        producer_stream: str,
        producer: DataReaderBase | None = None,
        producer_time_window_handler: TimeWindowHandler | None = None,
        forecast_step_stride: int = 1,
        max_pending_windows: int = 64,
    ) -> None:
        """
        Parameters
        ----------
        dataset :
            The consumer's own reader for this stream. Supplies the tokenization
            and normalization context the prediction has to be dressed up in.
        producer_stream :
            Name the stream carries in the producing component, i.e. the key its
            predictions are stored under in `ModelOutput.physical`.
        producer :
            The producing component's reader for the stream, used to bring its
            predictions back to physical space. If omitted, producer and consumer
            are assumed to share statistics for these channels.
        producer_time_window_handler :
            The producing component's time window handler. Defaults to the
            consumer's, i.e. both components walk the same timeline.
        forecast_step_stride :
            Dataset indices advanced per forecast step on the producer side,
            `forecast.time_step // time_window_step` of the producing sampler.
        max_pending_windows :
            Number of predicted windows kept before the oldest ones are dropped.

        Returns
        -------
        None
        """

        super().__init__(dataset.time_window_handler, dataset.stream_info)

        self._producer_stream = producer_stream
        self._producer_twh = producer_time_window_handler or dataset.time_window_handler
        self._stride = int(forecast_step_stride)
        self._max_pending = int(max_pending_windows)
        self._length = dataset.length()

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

        # geoinfos are not predicted; hand out the climatological mean so that the
        # consumer's normalization maps them to zero
        self._geoinfo_fill = np.asarray(self.mean_geoinfo, dtype=np.float32).reshape(1, -1)

        # producer side: which prediction columns to take and how to denormalize them
        if producer is None:
            producer_channels, producer_idx, stats = (
                list(dataset.source_channels),
                list(dataset.source_idx),
                dataset,
            )
        else:
            producer_channels, producer_idx, stats = (
                list(producer.target_channels),
                list(producer.target_idx),
                producer,
            )

        missing = [ch for ch in self.source_channels if ch not in producer_channels]
        if missing:
            raise ValueError(
                f"Coupled stream '{producer_stream}' does not predict the channels its consumer "
                f"needs as forcing: missing {missing}, predicted {producer_channels}."
            )

        self._pred_cols = np.asarray(
            [producer_channels.index(ch) for ch in self.source_channels], dtype=np.int64
        )
        chs = [producer_idx[col] for col in self._pred_cols]
        self._pred_mean = np.asarray([stats.mean[ch] for ch in chs], dtype=np.float32)
        self._pred_stdev = np.asarray([stats.stdev[ch] for ch in chs], dtype=np.float32)

        self._windows: dict[np.datetime64, _PredictedWindow] = {}

    def length(self) -> int:
        """Length of the timeline this reader is defined on, not of what it holds."""
        return self._length

    def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
        raise NotImplementedError(
            "DataReaderCoupling serves predicted chunks, not stored data; use get_source()."
        )

    def get_source(self, idx: TIndex) -> ReaderData:
        """
        Get the prediction valid for time window idx

        Parameters
        ----------
        idx : int
            Index of temporal window

        Returns
        -------
        source data, empty if no chunk covering idx has been dispatched yet
        """

        valid_time = self.time_window_handler.window(idx).start
        window = self._windows.get(valid_time)

        if window is None:
            logger.debug(
                f"No chunk for stream '{self._producer_stream}' at {valid_time} (index {idx}), "
                "forcing falls back to spoof."
            )
            return ReaderData.empty(len(self.source_idx), len(self.geoinfo_idx))

        # ForcingInput normalizes in place, so never hand out the stored arrays
        return ReaderData(
            coords=window.coords.copy(),
            geoinfos=np.tile(self._geoinfo_fill, (len(window.data), 1)),
            data=window.data.copy(),
            datetimes=window.datetimes.copy(),
            is_spoof=False,
        )

    def add_chunk(self, chunk: ModelOutput, batch: ModelBatch) -> None:
        """Index the predictions of one rollout chunk by the time they are valid for.

        `batch` is needed because a ModelOutput only carries the source samples,
        while the coordinates and times the predictions live on sit on the
        target samples.
        """

        for fstep in chunk.forecast_steps:
            preds = chunk.get_physical_prediction(chunk.chunk_idx(fstep), self._producer_stream)
            if preds is None:
                # leading empty steps of the first chunk, or a stream without a decoder
                continue

            for i_source, pred in enumerate(preds):
                entry = self._lower_prediction(chunk.batch_idx(fstep), i_source, pred, batch)
                if entry is not None:
                    valid_time, window = entry
                    self._windows[valid_time] = window

        self._evict()

    def _lower_prediction(
        self, fstep: int, i_source: int, pred: torch.Tensor, batch: ModelBatch
    ) -> tuple[np.datetime64, _PredictedWindow] | None:
        """Pair one prediction with its target geometry and bring it to physical space."""

        i_target = batch.get_target_idx_for_source(i_source)
        stream_data = batch.get_target_sample(i_target).streams_data.get(self._producer_stream)
        if stream_data is None:
            return None

        # spoofed steps carry made-up geometry, forcing on them would be worse than spoofing
        if stream_data.is_spoof(fstep):
            return None

        coords = _to_numpy(stream_data.target_coords_raw[fstep])
        times = np.asarray(stream_data.target_times_raw[fstep])
        if len(coords) == 0:
            return None

        # ensemble members are equivalent forcings, use their mean
        data = pred.mean(dim=0).to(torch.float32).detach().cpu().numpy()

        assert data.shape[0] == coords.shape[0] == times.shape[0], (
            f"Prediction for '{self._producer_stream}' at step {fstep} has {data.shape[0]} points "
            f"but {coords.shape[0]} coordinates and {times.shape[0]} times."
        )

        # restore the ordering of the original data, as the output writer does
        idxs_inv = stream_data.idxs_inv[fstep]
        if idxs_inv is not None and len(idxs_inv) > 0:
            idxs_inv = _to_numpy(idxs_inv)
            data, coords, times = data[idxs_inv], coords[idxs_inv], times[idxs_inv]

        # select the channels the consumer expects and denormalize them
        data = data[:, self._pred_cols] * self._pred_stdev + self._pred_mean

        valid_idx = stream_data.sample_idx + fstep * self._stride
        valid_time = self._producer_twh.window(valid_idx).start

        return valid_time, _PredictedWindow(
            coords=coords.astype(np.float32),
            data=data.astype(np.float32),
            datetimes=times.astype("datetime64[ns]"),
        )

    def _evict(self) -> None:
        """Drop the oldest windows once more than max_pending_windows are held."""
        excess = len(self._windows) - self._max_pending
        if excess <= 0:
            return
        for valid_time in sorted(self._windows)[:excess]:
            del self._windows[valid_time]


class Coupler:
    """Routes predicted chunks from producing components to consuming ForcingInputs."""

    def __init__(self, couplings: dict[str, Coupling] | None = None) -> None:
        self._couplings = dict(couplings or {})
        # (consumer, stream) -> reader serving that stream to that consumer
        self._readers: dict[tuple[str, str], DataReaderCoupling] = {}
        # producer -> readers waiting for its chunks
        self._by_producer: dict[str, list[DataReaderCoupling]] = {}
        # (consumer, stream) -> the ForcingInput reading through those readers
        self._subscribers: dict[tuple[str, str], ForcingInput] = {}

    def get_forcings(
        self,
        consumer: str,
        forcing_streams: dict[str, list[DataReaderBase]],
        producer_readers: dict[str, DataReaderBase] | None = None,
        forecast_step_stride: int = 1,
    ) -> dict[str, list[DataReaderBase]]:
        """Substitute coupling readers for the real readers of coupled streams.

        Takes the forcing streams a component would read from disk, i.e. the
        `{name: stream.readers}` mapping the Trainer builds for ForcingInput, and
        returns the same mapping with every stream this component consumes from
        another one backed by that component's chunks instead.
        """

        coupled = {
            coupling.stream: coupling
            for coupling in self._couplings.values()
            if coupling.consumer == consumer
        }

        forcings: dict[str, list[DataReaderBase]] = {}
        for stream, readers in forcing_streams.items():
            coupling = coupled.get(stream)
            if coupling is None:
                forcings[stream] = readers
                continue

            reader = DataReaderCoupling(
                readers[0],
                stream,
                producer=(producer_readers or {}).get(stream),
                forecast_step_stride=forecast_step_stride,
            )
            self._readers[(consumer, stream)] = reader
            self._by_producer.setdefault(coupling.producer, []).append(reader)
            forcings[stream] = [reader]

            logger.info(
                f"Stream '{stream}' of component '{consumer}' is forced by "
                f"component '{coupling.producer}'."
            )

        return forcings

    def subscribe_stream(self, consumer: str, subscriber: ForcingInput) -> None:
        """Record which ForcingInput reads the coupled streams of a component."""
        for stream in subscriber.forcing_streams.keys():
            self._subscribers[(consumer, stream)] = subscriber

    def dispatch_chunk(self, producer: str, chunk: ModelOutput, batch: ModelBatch) -> None:
        """Hand a finished rollout chunk to everything forced by producer.

        Called once per chunk in the rollout loop, where the ModelOutput and the
        batch it was computed from are both in scope.
        """
        for reader in self._by_producer.get(producer, ()):
            reader.add_chunk(chunk, batch)
