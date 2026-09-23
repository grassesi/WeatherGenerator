from __future__ import annotations

import copy
import logging
import typing

import numpy as np
import torch
import tqdm

import weathergen.common.config as config
from weathergen.common.config import timedelta_to_str
from weathergen.datasets.averaging import AveragingReader
from weathergen.datasets.batch import ModelBatch
from weathergen.datasets.coupling_reader import DataReaderCoupling
from weathergen.datasets.data_reader_base import (
    DataReaderBase,
    WrappedDataReader,
    rebase_innermost,
)
from weathergen.datasets.upsampling import UpsamplingReader
from weathergen.model.forcing import ForcingInput
from weathergen.model.model import ModelOutput
from weathergen.train.coupling.derivation import derive_component_configs
from weathergen.train.coupling.spec import Coupling, Rollout, produced_streams
from weathergen.train.trainer import Trainer
from weathergen.train.utils import (
    extract_batch_metadata,
    get_target_idxs_from_cfg,
    resolve_stage_configs,
)

if typing.TYPE_CHECKING:
    # annotation only: run imports this module to build the Coupler
    from weathergen.train.coupling.run import ModelCheckpoint

logger = logging.getLogger(__name__)

_ZERO = np.timedelta64(0, "ms")

# `ForcingEngine.init_weights_final` draws every block weight from normal(0, 0.001), so a
# freshly built engine is a near-identity by construction (`forcing.py`). An engine still
# sitting at that spread never learned anything, which is what generation 01's xcpk26es and
# j5h3is35 turned out to be.
_FFE_INIT_STD = 0.001
# Relative distance from _FFE_INIT_STD within which an engine is called untrained.
_FFE_INIT_TOL = 0.25


def _unwrap_model(model):
    """The ForcedModel underneath any DDP wrapper.

    `init_model_and_shard` wraps in DistributedDataParallel when running without FSDP, while
    `fully_shard` mutates in place and leaves the module itself reachable.
    """
    return getattr(model, "module", model)


def _reader_stack(reader: DataReaderBase) -> list[DataReaderBase]:
    """Every reader in a wrapper stack, outermost first."""
    stack = [reader]
    while isinstance(stack[-1], WrappedDataReader):
        stack.append(stack[-1]._wrapped_reader)
    return stack


def _ffe_param_count(engine) -> int:
    """Parameters in a forcing engine's blocks, counted regardless of requires_grad.

    `get_num_parameters` filters on requires_grad, which an inference run may have switched
    off, and a trained engine reading as empty would fail the identity check below for the
    wrong reason.
    """
    if engine is None:
        return 0
    return sum(p.numel() for p in engine.blocks.parameters())


def _ffe_weight_spread(engine) -> float | None:
    """Standard deviation over every forcing-engine block parameter, or None if unmeasurable.

    Under FSDP the parameters are DTensors whose `.std()` would be a collective, and a check
    that can deadlock is worse than a check that abstains. `to_local()` keeps it rank-local.
    """
    values = []
    for param in engine.blocks.parameters():
        tensor = param.detach()
        to_local = getattr(tensor, "to_local", None)
        if to_local is not None:
            tensor = to_local()
        values.append(tensor.reshape(-1).float())
    if not values:
        return None
    try:
        return float(torch.cat(values).std().item())
    except RuntimeError:
        return None


def _stream_files(reader: DataReaderBase) -> list[str]:
    """The files a reader's stream resolves to, as its stream_info records them."""
    info = getattr(reader, "stream_info", None) or {}
    return sorted(str(f) for f in (info.get("filenames") or []))


class Coupler:
    """Central driver: owns the batch and chunk loops of every component.

    The components never step themselves. The Coupler pulls one batch per component,
    then advances every component by one chunk before any of them advances to the next,
    and writes each chunk's output as it is produced. That makes the chunk boundary the
    single point where components meet, which is where coupling will later be inserted.

    At this milestone nothing is exchanged: the components run side by side, and each
    must reproduce exactly what it produces in a standalone inference run.
    """

    # test-stage keys that must agree so all components walk the same sample index space
    _SHARED_WINDOW_KEYS = ("start_date", "end_date", "time_window_len", "time_window_step")

    # policies that draw the per-batch forecast step count from a rank-dependent RNG;
    # under FSDP they make ranks issue different numbers of collectives and the run hangs
    _RANK_UNIFORM_POLICIES = ("fixed", "sequential")

    def __init__(
        self,
        components: dict[str, tuple[Trainer, config.Config]],
        couplings: dict[str, Coupling] | None = None,
        rollout: Rollout | None = None,
    ):
        # deterministic order: every rank must issue its collectives in the same sequence
        self._names = sorted(components)
        self._components = components
        self._couplings = couplings or {}
        self._rollout = rollout

        # component -> the forcings the Trainer built, before any coupling substitution
        self._pristine_forcings: dict[str, ForcingInput] = {}
        # (consumer, stream) pairs already reported, so a resubscribe stays quiet
        # (consumer, stream) pairs whose reader subscribe() actually replaced. Recorded at the
        # point of substitution, which is the only place that knows: asking the reader stack
        # afterwards means asking what type its outermost reader is, and that answer changes
        # the moment anything wraps it.
        self._substituted: set[tuple[str, str]] = set()
        # producer -> readers waiting for its chunks
        self._subscribers: dict[str, list[DataReaderCoupling]] = {}
        # component -> start of the current batch's initialization window, pushed into every
        # reader that primes from it. It cannot be learned from the first dispatched chunk,
        # because the first request precedes the first dispatch.
        self._init_times: dict[str, np.datetime64] = {}

    def trainer(self, name: str) -> Trainer:
        return self._components[name][0]

    def config(self, name: str) -> config.Config:
        return self._components[name][1]

    def setup(self, devices, checkpoints: dict[str, ModelCheckpoint]) -> None:
        """Derive every component's rollout settings, then build them."""
        self._check_couplings()
        self._names = self._step_order()
        derive_component_configs(
            {name: self.config(name) for name in self._names}, self._rollout, self._couplings
        )

        for name in self._names:
            trainer, ccf = self._components[name]
            checkpoint = checkpoints[name]
            logger.info(f"Setting up component {name!r} from {checkpoint.run_id}.")
            trainer.setup_inference(
                ccf, devices, checkpoint.run_id, checkpoint.mini_epoch, name
            )
            # Keep what the Trainer built before any substitution: subscribe() replaces the
            # real readers with coupling readers, so resubscribing from an already-coupled
            # ForcingInput would wrap a coupling reader in another one.
            self._pristine_forcings[name] = trainer.dynamic_forcings

        # Subscribe only once every component is built.
        for name in self._names:
            trainer = self.trainer(name)
            trainer.dynamic_forcings = self.subscribe(name, self._pristine_forcings[name])

        self._check_forcing_lags()
        self._announce_couplings()
        self._check_forcing_engines()
        self._check_exchange_grid()
        self._check_exchange_masks()
        self._report_checkpoint_provenance(checkpoints)

    def _step_order(self) -> list[str]:
        """Components in the order they are stepped, from the couplings file (C3).

        The producer of the first coupling is stepped first, so a consumer ordered after its
        producer sees the chunk just dispatched and one ordered before it sees the previous
        chunk. That is what makes the two directions of an asymmetric pairing fall out of the
        order alone, with no staging slot and no chunk-index bookkeeping.

        A component that produces nothing is stepped last. File order is as rank-deterministic
        as the alphabetical order it replaces, so the collectives guarantee is untouched.
        """

        order: list[str] = []
        for coupling in self._couplings.values():
            if coupling.producer not in order:
                order.append(coupling.producer)
        for name in sorted(self._components):
            if name not in order:
                order.append(name)

        logger.info(f"Step order, from couplings-file declaration order: {order}.")
        return order

    def _check_forcing_lags(self) -> None:
        """Every source window a coupled request touches must exist when it is asked for.

        With `len_c` the consumer's window length, `P` the producer's emission cadence and
        `D` a whole chunk when the producer is stepped *after* the consumer, the lag must
        satisfy `L >= len_c - P + D`. Below that the consumer reaches into a window its
        producer has not emitted yet, which reads back as an absent forcing rather than as an
        error (`forcing_lag_design.md` D4).
        """

        if self._rollout is None:
            return

        for coupling in self._couplings.values():
            if coupling.consumer is None:
                continue

            forcings = self._pristine_forcings.get(coupling.consumer)
            if forcings is None or coupling.stream not in forcings.lags:
                continue

            lag = forcings.lags[coupling.stream]
            len_c = self._window_len(coupling.consumer)
            cadence = self._emission_cadence(coupling.producer)
            before = self._names.index(coupling.producer) < self._names.index(coupling.consumer)
            slack = _ZERO if before else self._rollout.chunk_length
            required = len_c - cadence + slack

            where = (
                f"Coupling {coupling.name!r} ({coupling.producer!r} -> {coupling.consumer!r}, "
                f"stream {coupling.stream!r})"
            )
            if lag < required:
                msg = (
                    f"{where} declares forcing_lag {timedelta_to_str(lag)}, but the producer is "
                    f"stepped {'before' if before else 'after'} the consumer, so a request "
                    "reaches windows that do not exist yet unless the lag is at least "
                    f"{timedelta_to_str(required)} (consumer window {timedelta_to_str(len_c)} "
                    f"- producer cadence {timedelta_to_str(cadence)} + "
                    f"{timedelta_to_str(slack)})."
                )
                raise ValueError(msg)

            if lag > self._rollout.chunk_length:
                msg = (
                    f"{where} declares forcing_lag {timedelta_to_str(lag)}, longer than "
                    f"rollout.chunk_length {timedelta_to_str(self._rollout.chunk_length)}. The "
                    "reader keeps the current chunk and the one before it, so a longer lag "
                    "reaches past what is stored."
                )
                raise ValueError(msg)

            if cadence > _ZERO and lag % cadence != _ZERO:
                logger.warning(
                    f"{where} declares forcing_lag {timedelta_to_str(lag)}, which is not a "
                    f"whole multiple of the producer's {timedelta_to_str(cadence)} cadence. "
                    "The request lands between two emissions and is quantised to the window "
                    "covering it."
                )

            logger.info(
                f"{where}: forcing_lag {timedelta_to_str(lag)}, at least "
                f"{timedelta_to_str(required)} required."
            )

    def _window_len(self, name: str) -> np.timedelta64:
        """The component's own time window length, as its test config resolves it."""
        _, _, test_cfg = resolve_stage_configs(self.config(name))
        return config.parse_timedelta(test_cfg.get("time_window_len"))

    def _emission_cadence(self, name: str) -> np.timedelta64:
        """Wall-clock spacing of the windows this component emits, i.e. its forecast step."""
        _, _, test_cfg = resolve_stage_configs(self.config(name))
        time_step = test_cfg.get("forecast", {}).get("time_step")
        if time_step is None:
            return config.parse_timedelta(test_cfg.get("time_window_step"))
        return config.parse_timedelta(time_step)

    # ------------------------------------------------------------------ checks

    def _check_couplings(self) -> None:
        """Coupling names must resolve, and each stream may have only one producer.

        A coupling whose producer does not carry the stream is dropped rather than rejected.
        """
        producers: dict[str, str] = {}
        live: dict[str, Coupling] = {}
        for coupling in self._couplings.values():
            if coupling.producer not in self._components:
                msg = (
                    f"Coupling {coupling.name!r} names producer {coupling.producer!r}, "
                    f"which is not one of the components {self._names}."
                )
                raise ValueError(msg)

            if coupling.consumer is not None and coupling.consumer not in self._components:
                msg = (
                    f"Coupling {coupling.name!r} names consumer {coupling.consumer!r}, "
                    f"which is not one of the components {self._names}."
                )
                raise ValueError(msg)

            streams = self.config(coupling.producer).streams
            if coupling.stream not in streams:
                logger.warning(
                    f"Coupling {coupling.name!r} dropped: its producer "
                    f"{coupling.producer!r} has no stream {coupling.stream!r}. Expected when "
                    "one couplings file covers several component pairs; check the stream "
                    "name if this pair was meant to exchange it."
                )
                continue

            # one producer per stream keeps the shared output store collision-free by
            # construction, rather than by asking the configs to be disjoint
            if coupling.stream in producers:
                msg = (
                    f"Couplings {producers[coupling.stream]!r} and {coupling.name!r} both "
                    f"produce stream {coupling.stream!r}. Each stream may be produced once, "
                    "since components share one output store keyed <sample>/<stream>/<step>."
                )
                raise ValueError(msg)
            producers[coupling.stream] = coupling.name
            live[coupling.name] = coupling

        self._couplings = live

    def _producer_reader(self, producer: str, stream: str) -> DataReaderBase:
        """The producing component's own reader for `stream`.

        This is what the coupling reader resolves its channel map against, and it has to be the
        producer's rather than the consumer's: the two components carry separate configs for the
        same stream, and their channel lists agree in neither order nor length.
        """

        dataset = self.trainer(producer).dataset
        if dataset is None:
            msg = (
                f"Component {producer!r} has no dataset yet, so the channels it produces for "
                f"stream {stream!r} cannot be resolved. Every component must be set up before "
                "any is subscribed."
            )
            raise ValueError(msg)

        stream_data = dataset.streams_datasets.get(stream)
        if stream_data is None or not stream_data.readers:
            msg = (
                f"Component {producer!r} carries no reader for stream {stream!r}, so it cannot "
                "produce it. _check_couplings should have dropped this coupling."
            )
            raise ValueError(msg)

        return stream_data.readers[0]

    def _announce_couplings(self) -> None:
        """A run that exchanges nothing looks exactly like one that works. Say which it is.

        Reports what the wiring actually did, not what the config asked for: a coupling only
        takes effect if the consumer already reads that stream as a dynamic forcing, so a
        declared consumer can substitute nothing at all.

        The record comes from `_substituted`, written by subscribe() as it substitutes. Deriving
        it instead from the reader stack -- "is the outermost reader a DataReaderCoupling" -- is
        what made this function report a live exchange as dead for as long as subscribe() wrapped
        that reader in a HoldingReader.
        """
        exchanged: dict[str, list[str]] = {name: [] for name in self._names}
        for consumer, stream in self._substituted:
            exchanged.setdefault(consumer, []).append(stream)
        for streams in exchanged.values():
            streams.sort()

        declared = [c for c in self._couplings.values() if c.consumer is not None]
        live = [c for c in declared if c.stream in exchanged.get(c.consumer, [])]
        logger.info(
            f"{len(self._couplings)} coupling(s) declared, {len(declared)} with a consumer, "
            f"{len(live)} actually exchanging."
        )

        for coupling in declared:
            if coupling not in live:
                logger.warning(
                    f"Coupling {coupling.name!r} names {coupling.consumer!r} as the consumer "
                    f"of '{coupling.stream}', but that component does not read that stream as "
                    "a dynamic forcing, so nothing is substituted and the coupling has no "
                    "effect. Use a checkpoint whose stream config sets is_dynamic_forcing."
                )

        for name in self._names:
            forcing = self.trainer(name).dynamic_forcings
            if forcing is None or forcing.is_empty:
                continue
            coupled = exchanged.get(name, [])
            from_disk = sorted(set(forcing.forcing_streams) - set(coupled))
            if coupled:
                logger.info(f"Component {name!r} is forced by another component on: {coupled}.")
            if from_disk:
                logger.info(f"Component {name!r} samples from disk: {from_disk}.")

    def _live_couplings(self) -> list[Coupling]:
        """Couplings subscribe() actually substituted, i.e. the ones exchanging data.

        `_substituted` is the only honest record: a declared coupling whose consumer does not
        read the stream as a dynamic forcing substitutes nothing, and asking the reader stack
        what it is answers a different question (`_announce_couplings`).
        """
        return [
            coupling
            for coupling in self._couplings.values()
            if coupling.consumer is not None
            and (coupling.consumer, coupling.stream) in self._substituted
        ]

    def _check_forcing_engines(self) -> None:
        """A component receiving a coupled forcing must have a forcing engine with weights.

        `ffe_num_blocks` defaults to 0 and a zero-block engine is the identity: no parameters,
        an empty state dict, and `ForcedModel.forward` passes the latent through untouched, so
        the forcing tokens are discarded. That is a legitimate configuration for an unforced
        component (`open_actions.md` item 1) and it is exactly what a *forced* component looks
        like when its engine failed to load -- `load_state_dict` runs with `strict=False`, so
        unmatched keys are a warning and nothing else.

        Under coupling the forcing engine is the path the exchanged field travels. An identity
        engine there transports nothing while every provenance count still reports a live
        exchange, which is the `_announce_couplings` failure one layer down, in the weights
        instead of the wiring.
        """

        for coupling in self._live_couplings():
            consumer = coupling.consumer
            where = (
                f"Component {consumer!r} is forced by {coupling.producer!r} on "
                f"'{coupling.stream}'"
            )
            model = _unwrap_model(self.trainer(consumer).model)
            engine = getattr(model, "forcing_engine", None)
            num_params = _ffe_param_count(engine)

            if num_params == 0:
                msg = (
                    f"{where}, but its forcing engine has no parameters "
                    f"(ffe_num_blocks={self.config(consumer).get('ffe_num_blocks', 0)}). A "
                    "zero-block engine is the identity, so the forcing tokens are built, "
                    "gathered, reported as exchanged -- and then discarded. Use a checkpoint "
                    "trained with ffe_num_blocks > 0 as the consumer."
                )
                raise ValueError(msg)

            std = _ffe_weight_spread(engine)
            if std is None:
                logger.info(
                    f"{where}: forcing engine has {num_params} parameters; weight spread not "
                    "measurable on this sharding."
                )
                continue

            logger.info(
                f"{where}: forcing engine has {num_params} parameters, weight std {std:.3e}."
            )
            if abs(std - _FFE_INIT_STD) <= _FFE_INIT_TOL * _FFE_INIT_STD:
                logger.error(
                    f"{where}, but its forcing engine's weight std {std:.3e} is within "
                    f"{_FFE_INIT_TOL:.0%} of the {_FFE_INIT_STD} it is initialized to, so it "
                    "looks untrained and the coupling will transport almost nothing. "
                    "Generation 01's xcpk26es and j5h3is35 failed exactly this way. Check "
                    "that the checkpoint really finetuned the forcing engine before trusting "
                    "this run."
                )

    def _report_checkpoint_provenance(self, checkpoints: dict[str, ModelCheckpoint]) -> None:
        """One line per component recording which weights this run actually paired.

        Continuation pairs a checkpoint with itself, so its provenance is the run_id. A coupled
        run pairs two, and which two is the experiment -- but nothing in the artifacts records
        it, which is what the run registry keeps having to reconstruct after the fact. No
        policy here, just the record.
        """

        for name in self._names:
            checkpoint = checkpoints.get(name)
            ccf = self.config(name)
            model = _unwrap_model(self.trainer(name).model)
            engine = getattr(model, "forcing_engine", None)
            ffe_params = _ffe_param_count(engine)

            size = mtime = "unknown"
            if checkpoint is not None:
                # A record that can abort the run it is recording is worse than an incomplete
                # one: resolving the model path goes through the private config, which is not
                # reachable everywhere this runs.
                try:
                    path = (
                        config.get_path_model(run_id=checkpoint.run_id)
                        / f"{checkpoint.run_id}_chkpt{checkpoint.mini_epoch:05d}.chkpt"
                    )
                    stat = path.stat()
                except (OSError, AssertionError, ValueError, KeyError):
                    pass
                else:
                    size = f"{stat.st_size / 1024**3:.2f} GiB"
                    mtime = str(np.datetime64(int(stat.st_mtime), "s"))

            logger.info(
                f"Component {name!r} provenance: "
                f"checkpoint={checkpoint.run_id if checkpoint else '?'}"
                f"@{checkpoint.mini_epoch if checkpoint else '?'}, "
                f"size={size}, mtime={mtime}, "
                f"healpix_level={ccf.get('healpix_level')}, "
                f"ffe_num_blocks={ccf.get('ffe_num_blocks', 0)}, ffe_params={ffe_params}, "
                f"produces={produced_streams(self._couplings, name) or None}."
            )

    def _check_exchange_grid(self) -> None:
        """The two sides of a live coupling must mean the same spatio-temporal grid.

        Only the grid. `token_size` and `healpix_level` are how each component chops its own
        input up, not what the exchanged points are, and the consumer re-tokenizes what it
        receives with its own tokenizer either way -- so they are recorded by
        `_report_checkpoint_provenance` and left alone here.

        **Space.** The producer emits predictions on the point set of its own reader for the
        stream, and the consumer tokenizes them as if they had come off its own disk reader.
        Two readers resolving to different files are two different point sets, and nothing
        downstream would say so.

        **Time.** A gap in temporal resolution is legitimate -- it is what the averaging and
        upsampling wrappers exist for -- but only in the direction each of them bridges.
        `DataReaderCoupling` gathers every source window whose start falls in the request, so
        a producer *finer* than the stream's native period hands the consumer more rows per
        window than it ever trained on, and only an `AveragingReader` above the coupling
        reader reduces them back. The comparison is producer cadence against the period the
        consumer's disk reader declares, because the request window length is the same in
        training and under coupling: their ratio is the factor by which the row count moved.
        This is not `coupling_reader_placement.md` P6's withdrawn criterion, which compared
        periods to *choose* a wrapper to insert; here the stack is fixed by the checkpoint and
        the comparison only asks whether it still fits (P8). A producer *coarser* than the
        native period empties the gather, and the coupling reader takes the covering window
        and restamps it, which is a zero-order hold the consumer did not necessarily train
        through.
        """

        for coupling in self._live_couplings():
            producer, consumer, stream = coupling.producer, coupling.consumer, coupling.stream
            where = f"Coupling {coupling.name!r} ({producer!r} -> {consumer!r}, '{stream}')"

            producer_reader = self._producer_reader(producer, stream)
            consumer_readers = self._pristine_forcings[consumer].forcing_streams[stream]
            stack = _reader_stack(consumer_readers[0])
            consumer_reader = stack[-1]

            # -- space: the same underlying dataset, hence the same points
            producer_files = _stream_files(producer_reader)
            consumer_files = _stream_files(consumer_reader)
            if producer_files != consumer_files:
                msg = (
                    f"{where} exchanges a stream the two components read from different "
                    f"files: {producer!r} has {producer_files}, {consumer!r} has "
                    f"{consumer_files}. The producer emits on its own point set and the "
                    "consumer tokenizes the result as its own, so the forcing would land on "
                    "a grid the consumer never trained on."
                )
                raise ValueError(msg)

            # -- time: the producer's cadence against the period the consumer trained on
            cadence = self._emission_cadence(producer)
            period = getattr(consumer_reader, "period", None)
            if period is None or cadence == _ZERO:
                logger.info(
                    f"{where}: grid check covered files only; "
                    f"{consumer!r}'s reader for '{stream}' declares no period."
                )
                continue

            bridges = [
                type(r).__name__
                for r in stack
                if isinstance(r, AveragingReader | UpsamplingReader)
            ]

            if period == cadence:
                logger.info(
                    f"{where}: grid agrees, producer cadence "
                    f"{timedelta_to_str(cadence)} = stream period "
                    f"{timedelta_to_str(period)}"
                    + (f", bridged by {bridges}." if bridges else ".")
                )
            elif cadence < period:
                if not any(b == "AveragingReader" for b in bridges):
                    msg = (
                        f"{where}: {producer!r} emits every {timedelta_to_str(cadence)}, "
                        f"finer than the {timedelta_to_str(period)} period {consumer!r} reads "
                        f"'{stream}' at, so every request window gathers "
                        f"{period // cadence}x the rows it did in training. The stream carries "
                        f"no AveragingReader to reduce them ({bridges or 'no wrappers'}), so "
                        "the tokenizer would see a row count the model never trained on."
                    )
                    raise ValueError(msg)
                logger.info(
                    f"{where}: producer cadence {timedelta_to_str(cadence)} is finer than the "
                    f"{timedelta_to_str(period)} stream period; AveragingReader bridges the "
                    f"{period // cadence}x increase in rows per window."
                )
            else:
                logger.warning(
                    f"{where}: {producer!r} emits every {timedelta_to_str(cadence)}, coarser "
                    f"than the {timedelta_to_str(period)} period {consumer!r} reads '{stream}' "
                    "at. The coupling reader serves the covering window restamped, so the "
                    f"consumer sees a field up to {timedelta_to_str(cadence - period)} stale "
                    "where training varied every window"
                    + (f" ({bridges} also in the stack)." if bridges else ".")
                )

    def _check_exchange_masks(self) -> None:
        """An exchanged channel that is NaN on disk must be masked in the producer's prediction.

        `_prime` hands the consumer the producer's ground truth for chunk 0, NaN wherever the
        data is (SST over land), and the consumer's tokenizer turns that NaN into the same
        `mask_value` it saw in training. Every later chunk is a prediction, finite everywhere
        unless the producer masks it via `streams.<stream>.mask_predictions`. Unmasked, the
        consumer is forced from chunk 1 on by land values it never saw, with nothing to show
        for it (`optional_target_sst_masking.md` C, M3).

        The producer's disk reader knows which channels carry NaNs (`nan_channels`); None
        means it cannot say, and then the check abstains rather than guesses.
        """

        for coupling in self._live_couplings():
            producer, consumer, stream = coupling.producer, coupling.consumer, coupling.stream
            where = f"Coupling {coupling.name!r} ({producer!r} -> {consumer!r}, '{stream}')"

            # what crosses is what the coupling reader maps: the consumer's source channels,
            # taken by name from the producer's target channels
            producer_reader = self._producer_reader(producer, stream)
            consumer_readers = self._pristine_forcings[consumer].forcing_streams[stream]
            needed = set(_reader_stack(consumer_readers[0])[-1].source_channels)
            exchanged = [ch for ch in producer_reader.target_channels if ch in needed]

            stream_cfg = self.config(producer).streams[stream]
            mask = list(stream_cfg.get("mask_predictions", None) or [])

            readers = self.trainer(producer).dataset.streams_datasets[stream].readers
            nans = [getattr(_reader_stack(r)[-1], "nan_channels", None) for r in readers]
            if any(nan is None for nan in nans):
                logger.warning(
                    f"{where}: {producer!r}'s reader cannot say which channels carry NaNs, so "
                    f"whether the exchanged channels {exchanged} need masking was not checked. "
                    f"Masked: {[ch for ch in exchanged if ch in mask]}."
                )
                continue

            nan = frozenset().union(*nans)
            unmasked = [ch for ch in exchanged if ch in nan and ch not in mask]
            if unmasked:
                msg = (
                    f"{where}: {producer!r} hands over {unmasked}, which are NaN on disk, but "
                    f"its stream config does not mask them (mask_predictions={mask}). Chunk 0 "
                    f"reaches {consumer!r} with NaN there and every later chunk with finite "
                    "predictions the consumer never trained on. Pass "
                    f"--options '{producer}:streams.{stream}.mask_predictions="
                    f"[{','.join(sorted(set(mask) | set(unmasked)))}]'."
                )
                raise ValueError(msg)

            logger.info(
                f"{where}: exchanged channels {exchanged}, masked "
                f"{[ch for ch in exchanged if ch in mask]}, NaN on disk {sorted(nan)}."
            )

    # ------------------------------------------------------------------ driving

    def validate(self, mini_epoch: int = 0) -> None:
        """Run every component over the validation set, interleaved by chunk."""
        for name in self._names:
            self.trainer(name).model.eval()

        # _batch_size_one derives this, and the loop below reads a batch count as a sample
        # count, so check the post-condition rather than trusting the derivation.
        for name in self._names:
            batch_size = self.trainer(name).batch_size_test_per_gpu
            if batch_size != 1:
                msg = f"Component {name!r} has batch size {batch_size}, not 1."
                raise ValueError(msg)

        iters = {name: iter(self.trainer(name).data_loader_validation) for name in self._names}
        # len() is the per-rank batch count. samples_per_mini_epoch counts samples across all
        # ranks, so it is not a bound on this loop; using it over-runs the loader.
        total = min(len(self.trainer(name).data_loader_validation) for name in self._names)
        with_ddp = self.config(self._names[0]).with_ddp

        with torch.no_grad(), tqdm.tqdm(total=total, disable=with_ddp) as pbar:
            for bidx in range(total):
                self._run_batch(self._next_batches(iters), bidx, mini_epoch)
                pbar.update(1)

        self._report_provenance()

        for name in self._names:
            trainer = self.trainer(name)
            trainer.finish_validation(
                mini_epoch, inference_only=trainer.test_cfg.get("inference_only", False)
            )

    def _report_provenance(self) -> None:
        """Say where every forcing window a run served actually came from.

        The cheapest assertion that kills the whole class of silent-exchange defects. Every
        way the exchange can fail -- a request resolved on the wrong timeline, a window the
        producer never emitted, a coupling that was never substituted -- ends at the same
        climatological spoof, behind the same `logger.debug`, with every wiring assertion
        still green. These counts distinguish them.
        """

        for name in self._names:
            forcings = self._pristine_forcings.get(name)
            if forcings is None or forcings.is_empty:
                continue
            for stream, provenance in forcings.provenance.items():
                logger.info(provenance.describe(name))
                if provenance.unresolved:
                    logger.warning(
                        f"Forcing '{stream}' of '{name}' left {provenance.unresolved} of "
                        f"{provenance.requests + provenance.unresolved} source window(s) "
                        "unresolved; each was replaced by a climatological spoof."
                    )

    def _next_batches(self, iters) -> dict:
        """One batch per component."""
        batches = {}
        for name in self._names:
            try:
                batches[name] = next(iters[name])
            except StopIteration as e:
                msg = (
                    f"Component {name!r} ran out of batches early. Its loader declared "
                    f"{len(self.trainer(name).data_loader_validation)} batches; the "
                    "components are no longer on one time axis."
                )
                raise RuntimeError(msg) from e

        return batches

    @staticmethod
    def _autocast(trainer: Trainer):
        cf = trainer.cf
        return torch.autocast(
            device_type=f"cuda:{cf.local_rank}",
            dtype=trainer.mixed_precision_dtype,
            enabled=cf.with_mixed_precision,
        )

    @staticmethod
    def _compute_targets(trainer: Trainer, batch) -> dict:
        # Trainer.validate's rule: under inference_only there is no target half, and the
        # prediction geometry the writer needs sits on the source samples
        inference_only = trainer.test_cfg.get("inference_only", False)
        targets_and_auxs = {}
        for loss_name, target_aux in trainer.target_and_aux_calculators_val.items():
            target_idxs = get_target_idxs_from_cfg(trainer.test_cfg, loss_name)
            targets_and_auxs[loss_name] = target_aux.compute(
                trainer.cf.general.istep,
                (
                    batch.get_source_samples()
                    if inference_only
                    else batch.get_target_samples(target_idxs)
                ),
                trainer.model_params,
                trainer.model,
            )

        return targets_and_auxs

    def _run_batch(self, batches: dict, bidx: int, mini_epoch: int) -> None:
        targets, plans, states, accumulated = {}, {}, {}, {}

        self._resubscribe(batches)

        # per-component preparation, in deterministic order
        for name in self._names:
            trainer = self.trainer(name)
            batch = batches[name]
            if trainer.cf.data_loading.get("memory_pinning", False):
                batch = batch.pin_memory()
            batch.to_device(trainer.device)
            batches[name] = batch

            with self._autocast(trainer):
                targets[name] = self._compute_targets(trainer, batch)
                plans[name] = trainer.prepare_chunks(
                    batch,
                    trainer.test_cfg,
                    trainer.batch_size_test_per_gpu,
                    bidx,
                    targets[name],
                )
            states[name] = batch.get_source_samples()
            accumulated[name] = ([], [])

        # interleave: every component advances one chunk before any advances two
        for i in range(max(len(plan.tiles) for plan in plans.values())):
            for name in self._names:
                plan = plans[name]
                if i >= len(plan.tiles):
                    continue

                trainer = self.trainer(name)
                with self._autocast(trainer):
                    states[name] = trainer.step_chunk(states[name], plan.tiles[i])

                    self.dispatch_chunk(trainer.name, states[name], batches[name])
                    if plan.should_write_output:
                        trainer.write_chunk_output(
                            plan,
                            trainer.test_cfg,
                            trainer.batch_size_test_per_gpu,
                            mini_epoch,
                            bidx,
                            batches[name],
                            states[name],
                            targets[name],
                        )

                if plan.should_accumulate_chunks:
                    physical, latent = accumulated[name]
                    physical += states[name].physical
                    latent += states[name].latent

        # losses outside autocast, matching the single-model path
        for name in self._names:
            plan = plans[name]
            trainer = self.trainer(name)
            # inference_only builds no targets, so there is nothing to score against
            if not plan.should_accumulate_chunks or trainer.test_cfg.get("inference_only", False):
                continue

            physical, latent = accumulated[name]
            preds = trainer.assemble_chunks(plan, physical, latent, batches[name])
            _ = trainer.loss_calculator_val.compute_loss(
                preds=preds,
                targets_and_aux=targets[name],
                metadata=extract_batch_metadata(batches[name]),
            )

    # -------------------------------------------------- coupling (not yet active)

    def subscribe(
        self,
        consumer: str,
        forcings: ForcingInput,
    ) -> ForcingInput:
        """Substitute coupling readers for the real readers of coupled streams.

        Takes the forcing streams a component would read from disk, i.e. the
        `{name: stream.readers}` mapping the Trainer builds for ForcingInput, and
        returns the same mapping with every stream this component consumes from
        another one backed by that component's chunks instead.
        """
        forcings = copy.copy(forcings)
        forcing_streams = forcings.forcing_streams

        coupled = {
            coupling.stream: coupling
            for coupling in self._couplings.values()
            if coupling.consumer == consumer
        }

        forcing_streams_coupled: dict[str, list[DataReaderBase]] = {}
        # TODO move this loop to ForcingInput, move DataReaderCoupling into separate module.
        for stream, readers in forcing_streams.items():
            coupling = coupled.get(stream)
            if coupling is not None:
                if len(readers) > 1:
                    msg = (
                        f"Coupled stream {stream!r} of component {consumer!r} has "
                        f"{len(readers)} readers. A producer replaces a stream, not one of its "
                        "files, so which reader to rebase is undefined."
                    )
                    raise ValueError(msg)

                producer_reader = self._producer_reader(coupling.producer, stream)
                init_time = self._init_times.get(coupling.producer)
                built: list[DataReaderCoupling] = []

                def _make_inner(base, _p=producer_reader, _s=stream, _b=built, _init=init_time):
                    # The training path already read this stream through a coupling reader, on
                    # the lagged timeline. Rebuild from the disk reader underneath it rather
                    # than wrapping it, and keep its handler and its tally: the lag is the
                    # model's, not the run's, and the counts have to span every trajectory.
                    disk = base.disk_reader if isinstance(base, DataReaderCoupling) else base
                    request_handler = (
                        base.request_handler if isinstance(base, DataReaderCoupling) else None
                    )
                    provenance = (
                        base.provenance if isinstance(base, DataReaderCoupling) else None
                    )
                    coupled = DataReaderCoupling(
                        disk,
                        _s,
                        producer=_p,
                        request_handler=request_handler,
                        is_forced=True,
                        init_time=_init,
                        provenance=provenance,
                    )
                    _b.append(coupled)
                    return coupled

                # innermost, so the stream's own wrappers apply to a coupled forcing exactly as
                # they did in training, instead of being bypassed by it
                reader = rebase_innermost(readers[0], _make_inner)

                # register coupling for producer: add_chunk belongs to the coupling reader
                # itself, not to whatever now wraps it
                self._subscribers.setdefault(coupling.producer, []).append(built[0])

                # once per run, not once per batch: the readers are rebuilt for every
                # trajectory, but the wiring they describe is the same every time, and this
                # line is what a run is checked against to prove the exchange is live
                if (consumer, stream) not in self._substituted:
                    self._substituted.add((consumer, stream))
                    logger.info(
                        f"Stream '{stream}' of component '{consumer}' is forced by "
                        f"component '{coupling.producer}'."
                    )
                readers = [reader]

            forcing_streams_coupled[stream] = readers

        forcings.forcing_streams = forcing_streams_coupled
        return forcings

    def _resubscribe(self, batches: dict) -> None:
        """Give every component fresh coupling readers, discarding the previous batch's.

        Each batch is a separate initialisation time, i.e. an independent trajectory, but a
        DataReaderCoupling indexes what it holds by wall-clock valid time alone. Two
        trajectories overlap in wall-clock time -- a run started on Jan 1 covers Jan 2, and so
        does one started on Jan 2 -- so without this the second trajectory would read the
        first one's predictions for the shared windows, silently and with no error. Rebuilding
        the readers from the pristine forcings drops the old chunks, so every trajectory
        bootstraps from ground truth exactly as the first one did.

        It is also where each component's initialization window is read off the batch and
        pushed into the readers that prime from it (G5): a window at or before that time is an
        initial condition however far into the rollout it is requested, and the first request
        of a trajectory precedes the first dispatch, so it cannot be captured from a chunk.
        """

        self._init_times = {
            name: self._init_time(name, batch) for name, batch in batches.items()
        }
        self._subscribers.clear()
        for name in self._names:
            trainer = self.trainer(name)
            pristine = self._pristine_forcings.get(name)
            if pristine is None:
                continue
            trainer.dynamic_forcings = self.subscribe(name, pristine)

    def _init_time(self, name: str, batch: ModelBatch) -> np.datetime64:
        """Start of the window this batch's trajectory is initialized from.

        Read off the batch rather than derived from the rollout spec: the sampler may have
        moved the sample, and what the readers must agree with is the window the model was
        actually handed. Batch size is one per component, so one date per batch is well
        defined.
        """

        samples = batch.get_source_samples().samples
        assert len(samples) == 1, (
            f"Component {name!r} handed {len(samples)} source samples; coupled inference runs "
            "at batch size 1, which is what makes one initialization time per batch."
        )

        idxs = {
            stream_data.sample_idx
            for stream_data in samples[0].streams_data.values()
            if stream_data is not None
        }
        assert len(idxs) == 1, (
            f"Component {name!r} sampled window indices {sorted(idxs)} in one batch; the "
            "streams of a sample must share one initialization window."
        )

        return self.trainer(name).dataset.time_window_handler.window(next(iter(idxs))).start

    def dispatch_chunk(self, producer: str, chunk: ModelOutput, batch: ModelBatch) -> None:
        """Hand a finished rollout chunk to everything forced by producer.

        Called once per chunk in the rollout loop, where the ModelOutput and the
        batch it was computed from are both in scope.
        """
        for reader in self._subscribers.get(producer, ()):
            reader.add_chunk(chunk, batch)
