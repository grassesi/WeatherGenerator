from __future__ import annotations

import copy
import logging
import typing

import numpy as np
import torch
import tqdm

import weathergen.common.config as config
from weathergen.datasets.batch import ModelBatch
from weathergen.datasets.coupling_reader import DataReaderCoupling
from weathergen.datasets.data_reader_base import (
    DataReaderBase,
    rebase_innermost,
)
from weathergen.model.forcing import ForcingInput
from weathergen.model.model import ModelOutput
from weathergen.train.coupling.checks import (
    announce_couplings,
    check_forcing_lags,
    resolve_couplings,
)
from weathergen.train.coupling.compatibility import (
    check_exchange_grid,
    check_exchange_masks,
    check_forcing_engines,
    report_checkpoint_provenance,
)
from weathergen.train.coupling.derivation import derive_component_configs
from weathergen.train.coupling.spec import Coupling, Rollout
from weathergen.train.trainer import Trainer
from weathergen.train.utils import (
    extract_batch_metadata,
    get_target_idxs_from_cfg,
)

if typing.TYPE_CHECKING:
    # annotation only: run imports this module to build the Coupler
    from weathergen.train.coupling.run import ModelCheckpoint

logger = logging.getLogger(__name__)


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
        self._couplings = resolve_couplings(
            self._couplings, {name: self.config(name) for name in self._components}
        )
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

        check_forcing_lags(self)
        announce_couplings(self)
        check_forcing_engines(self)
        check_exchange_grid(self)
        check_exchange_masks(self)
        report_checkpoint_provenance(self, checkpoints)

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

    # -- read-only view for the setup checks in checks.py and compatibility.py

    @property
    def names(self) -> list[str]:
        """Components in the order they are stepped."""
        return self._names

    @property
    def couplings(self) -> dict[str, Coupling]:
        return self._couplings

    @property
    def rollout(self) -> Rollout | None:
        return self._rollout

    @property
    def substituted(self) -> frozenset[tuple[str, str]]:
        """(consumer, stream) pairs whose reader subscribe() actually replaced."""
        return frozenset(self._substituted)

    def pristine_forcings(self, name: str) -> ForcingInput | None:
        """The forcings the component's Trainer built, before any coupling substitution."""
        return self._pristine_forcings.get(name)

    def producer_reader(self, producer: str, stream: str) -> DataReaderBase:
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
                "produce it. resolve_couplings should have dropped this coupling."
            )
            raise ValueError(msg)

        return stream_data.readers[0]

    def live_couplings(self) -> list[Coupling]:
        """Couplings subscribe() actually substituted, i.e. the ones exchanging data.

        `_substituted` is the only honest record: a declared coupling whose consumer does not
        read the stream as a dynamic forcing substitutes nothing, and asking the reader stack
        what it is answers a different question (`checks.announce_couplings`).
        """
        return [
            coupling
            for coupling in self._couplings.values()
            if coupling.consumer is not None
            and (coupling.consumer, coupling.stream) in self._substituted
        ]

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

                producer_reader = self.producer_reader(coupling.producer, stream)
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
