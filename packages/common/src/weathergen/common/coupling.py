from __future__ import annotations

import copy
import dataclasses
import logging
from pathlib import Path

import numpy as np
import torch
import tqdm
from numpy.typing import NDArray
from omegaconf import OmegaConf, open_dict

import weathergen.common.config as config
from weathergen.common.logger import init_loggers
from weathergen.datasets.averaging import AveragingReader
from weathergen.datasets.batch import ModelBatch
from weathergen.datasets.data_reader_base import (
    DataReaderBase,
    DataReaderTimestep,
    PassthroughReader,
    ReaderData,
    TimeWindowHandler,
    TIndex,
    rebase_innermost,
)
from weathergen.datasets.tokenizer_utils import TIMES_WIDTH
from weathergen.datasets.upsampling import UpsamplingReader
from weathergen.model.chunking import ChunkInfo
from weathergen.model.forcing import ForcingInput
from weathergen.model.model import ModelOutput
from weathergen.train.trainer import Trainer
from weathergen.train.utils import (
    extract_batch_metadata,
    get_target_idxs_from_cfg,
    resolve_stage_configs,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ModelCheckpoint:
    component: str
    _checkpoint_str: dataclasses.InitVar[str]
    run_id: str = dataclasses.field(init=False)
    mini_epoch: int = dataclasses.field(init=False)
    istep: int = dataclasses.field(init=False)
    
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
        options: list[str] | None,
        shared_config: config.Config
    ) -> tuple[Trainer, config.Config]:
        configs = [] if configs is None else configs
        # never None: OmegaConf.from_cli(None) falls back to sys.argv[1:], which would merge
        # the shell command line into every component's config
        options_overwrite = config.from_cli_arglist(options or [])
        cf = config.load_merge_configs(
            private_config,
            self.run_id,
            self.mini_epoch,
            None,
            *configs,
            options_overwrite,
        )
        cf = OmegaConf.merge(cf, shared_config)
        return Trainer(cf.train_logging, self.component), cf


@dataclasses.dataclass
class Coupling:
    """One exchange surface: a stream a producer emits and a consumer may later ingest.

    consumer is None while the components only run side by side, which is what lets the
    interleaving be exercised before any field is actually exchanged.

    Every coupling is optional: one that names a stream its producer does not carry is
    dropped rather than fataly
    """

    name: str
    producer: str
    stream: str
    consumer: str | None = None


@dataclasses.dataclass
class Rollout:
    """Global rollout spec - the single source of truth for how a coupled run advances.

    Everything the components must agree on lives here and is pushed down onto each
    component's test_config, rather than being checked between components that have no way to
    be made to agree. chunk_length is a physical duration, so a component's chunk size is
    chunk_length // its own forecast.time_step; chunk i then covers the same wall-clock
    interval for every component, which is what makes interleaving by chunk index correct.
    """

    start_date: str
    end_date: str
    chunk_length: np.timedelta64
    num_chunks: int
    num_samples: int
    forecast_offset: int = 1
    num_workers: int = 0 # Dont use forked pools of workers for dataloaders
    accumulate_chunks: bool = False

    @classmethod
    def from_config(cls, cfg) -> Rollout:
        required = ("start_date", "end_date", "chunk_length", "num_chunks", "num_samples")
        missing = [key for key in required if cfg.get(key) is None]
        if missing:
            msg = f"The couplings file's 'rollout' section is missing: {missing}."
            raise ValueError(msg)

        rollout = cls(
            start_date=str(cfg.start_date),
            end_date=str(cfg.end_date),
            chunk_length=config.parse_timedelta(cfg.chunk_length),
            num_chunks=int(cfg.num_chunks),
            num_samples=int(cfg.num_samples),
            forecast_offset=int(cfg.get("forecast_offset", 1)),
            num_workers=int(cfg.get("num_workers", 0)),
            accumulate_chunks=bool(cfg.get("accumulate_chunks", False)),
        )

        for key in ("num_chunks", "num_samples"):
            if getattr(rollout, key) < 1:
                msg = f"rollout.{key} must be >= 1, got {getattr(rollout, key)}."
                raise ValueError(msg)
        if rollout.chunk_length <= np.timedelta64(0, "ms"):
            msg = f"rollout.chunk_length must be positive, got {cfg.chunk_length!r}."
            raise ValueError(msg)
        if rollout.forecast_offset not in (0, 1):
            msg = f"rollout.forecast_offset must be 0 or 1, got {rollout.forecast_offset}."
            raise ValueError(msg)
        if rollout.num_workers < 0:
            msg = f"rollout.num_workers must be >= 0, got {rollout.num_workers}."
            raise ValueError(msg)

        return rollout

@dataclasses.dataclass
class Couplings:
    rollout: Rollout
    couplings: dict[str, Coupling]
    checkpoints: dict[str, ModelCheckpoint]
    _devices: str = dataclasses.field(init=False, default=None)
    _run_history: list[tuple[str, int]] = dataclasses.field(init=False, default=None)

    def __str__(self) -> str:
        lines = [f"rollout: {self.rollout}", "couplings:"]
        lines += [f"  {coupling}" for coupling in self.couplings.values()] or ["  (none)"]
        lines += ["components:"]
        lines += [
            f"  {name} <- {ckpt.run_id}@{ckpt.mini_epoch}"
            for name, ckpt in self.checkpoints.items()
        ]
        return "\n".join(lines)

    @classmethod
    def from_args(cls, couplings: Path, components: list[str]):
        components_kv = (component.split("=") for component in components)
        components = {
            component: ModelCheckpoint(component, checkpoint_str)
            for component, checkpoint_str in components_kv
        }

        spec = OmegaConf.load(couplings)
        if spec.get("rollout") is None:
            msg = (
                f"{couplings} has no top-level 'rollout' section. The rollout spec is the "
                "single source of truth for a coupled run; see config/example_coupling.yml."
            )
            raise ValueError(msg)

        rollout = Rollout.from_config(spec.rollout)
        parsed_couplings = {
            name: Coupling(name=name, **coupling)
            for name, coupling in (spec.get("couplings") or {}).items()
        }

        return cls(rollout=rollout, couplings=parsed_couplings, checkpoints=components)
    
    def global_intialization(self, run_id: str) -> config.Config:
        # global setup
        cf = OmegaConf.create({"general": {}})
        cf = config.set_run_id(cf, run_id, False)
        self._devices = Trainer.init_torch()
        cf = Trainer.init_ddp(cf)
        init_loggers(cf.general.run_id)
        logger.info(f"DDP initialization: rank={cf.rank}, world_size={cf.world_size}")
        self._run_history = [
            [(checkpoint.run_id, checkpoint.istep) for checkpoint in self.checkpoints.values()]
        ]
        return cf

    def run(
        self,
        private_config: Path,
        global_cf: config.Config,
        configs: list[Path] | None = None,
        options: list[str] | None = None,
    ):
        """Build every component and hand them to the Coupler.

        `configs` and `options` are the run's own --config files and --options, applied to
        each component on top of its checkpoint. They are the only way to reach a component's
        config at all -- a coupled run takes its rollout from the couplings file, but settings
        that are properties of the *execution* rather than the rollout (with_fsdp, say) have
        nowhere else to live. Whatever they set, `Coupler._derive_component_configs` still
        overrides the rollout keys afterwards, so the couplings file stays authoritative for
        the horizon, window and chunking.
        """

        # the coupling keys are this run's own arguments, not component config; they arrive
        # here inside config_command_line.yaml because the launcher has no coupled stage
        configs = [] if configs is None else list(configs)
        options = [] if options is None else list(options)

        # instantiate all components
        components = {
            name: checkpoint.get_component(private_config, configs, options, global_cf)
            for name, checkpoint in self.checkpoints.items()
        }
        
        for _, ccf in components.values():
            ccf.general.run_history += self._run_history

        # the coupler owns the loops: components are stepped, they do not run themselves
        coupler = Coupler(components, self.couplings, self.rollout)
        coupler.setup(self._devices, self.checkpoints)

        run_id = global_cf.general.run_id
        logger.info(f"Starting coupled inference with id={run_id} over {list(components)}.")
        coupler.validate(mini_epoch=0)
        logger.info(f"Finished coupled inference run with id: {run_id}")

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
        # per component, filled by _derive_component_configs and read by subscribe()
        self._fsteps_per_chunk: dict[str, int] = {}
        # producer -> readers waiting for its chunks
        self._subscribers: dict[str, list[DataReaderCoupling]] = {}

    def trainer(self, name: str) -> Trainer:
        return self._components[name][0]

    def config(self, name: str) -> config.Config:
        return self._components[name][1]

    def setup(self, devices, checkpoints: dict[str, ModelCheckpoint]) -> None:
        """Derive every component's rollout settings, then build them."""
        self._check_couplings()
        self._derive_component_configs()

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

        self._announce_couplings()

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

    @staticmethod
    def _level(
        coupled: DataReaderCoupling, base: DataReaderBase, stream: str
    ) -> DataReaderBase:
        """Wrap the coupling reader so it emits at the consumer's own stream cadence.

        The quantity compared is the spacing between consecutive samples on each side --
        (window length) / (samples per window) -- which is exactly a reader's `period`. Not the
        forecast step: a 24 h window reading a 6 h stream carries four samples, and the forecast
        step says nothing about that.

        Faster producer -> average down, slower producer -> upsample, equal -> nothing, made
        explicit (coupling_reader_placement.md P6).
        """

        produced = getattr(coupled, "period", None)
        consumed = getattr(base, "period", None)
        if produced is None or consumed is None:
            msg = (
                f"Coupled stream {stream!r} cannot be levelled: "
                f"producer period={produced}, consumer period={consumed}. Both sides must be "
                "periodic for the cadences to be comparable."
            )
            raise ValueError(msg)

        if produced == consumed:
            logger.info(
                f"Stream {stream!r}: producer and consumer both sample every {consumed}, "
                "no levelling needed."
            )
            return PassthroughReader(coupled)

        if produced < consumed:
            logger.info(
                f"Stream {stream!r}: producer samples every {produced} against the consumer's "
                f"{consumed}, averaging down."
            )
            return AveragingReader(coupled)

        logger.info(
            f"Stream {stream!r}: producer samples every {produced} against the consumer's "
            f"{consumed}, upsampling."
        )
        return UpsamplingReader(coupled)

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

    def _produced_streams(self, name: str) -> list[str]:
        """Streams this component is the producer of, in declaration order."""
        return [c.stream for c in self._couplings.values() if c.producer == name]

    # ------------------------------------------------------------------ derivation

    def _derive_component_configs(self) -> None:
        """Push the global rollout spec down onto every component's test_config.

        The components cannot be made to agree by hand - nothing overrides what is baked into
        their checkpoints - so instead of checking them against each other, everything they
        must share is derived here from one spec and written into test_config, the last layer
        of the training -> validation -> test cascade.
        """
        if self._rollout is None:
            msg = "Coupler requires a Rollout spec; none was provided."
            raise ValueError(msg)

        rollout = self._rollout
        for name in self._names:
            _, ccf = self._components[name]
            # the effective test config, resolved the same way Trainer.init will resolve it
            _, _, test_cfg = resolve_stage_configs(ccf)

            time_step = test_cfg.get("forecast", {}).get("time_step")
            window_step = test_cfg.get("time_window_step")
            if time_step is None or window_step is None:
                msg = (
                    f"Component {name!r} is missing test_config.forecast.time_step or "
                    "test_config.time_window_step; both are needed to place it on the "
                    "shared time axis."
                )
                raise ValueError(msg)

            fsteps_per_chunk = self._exact_ratio(
                rollout.chunk_length, time_step, name, "forecast.time_step"
            )
            sample_stride = self._exact_ratio(
                rollout.chunk_length, window_step, name, "time_window_step"
            )

            overrides = {
                "start_date": f"${{{config._DATETIME_TYPE_NAME}:{rollout.start_date}}}",
                "end_date": f"${{{config._DATETIME_TYPE_NAME}:{rollout.end_date}}}",
                "_start_date": rollout.start_date,
                "_end_date": rollout.end_date,
                # successive initial conditions are one chunk apart; the sampler strides in
                # index units, so each component covers the same absolute times
                "sample_stride": sample_stride,
                # inference writes every sample it runs, which only holds at batch size 1
                "samples_per_mini_epoch": rollout.num_samples,
                # each component shuffles with its own rng_seed, so they would otherwise
                # visit different windows
                "shuffle": False,
                # rollout.start_date must actually draw the first sample. The sampler
                # otherwise substitutes the next usable window for an empty or NaN one, and
                # does so per component, which walks two components onto different dates.
                "strict_batches": True,
                "forecast": {
                    "offset": rollout.forecast_offset,
                    "chunk_size": fsteps_per_chunk,
                    "num_steps": rollout.num_chunks * fsteps_per_chunk,
                    # 'random' policies draw the step count from a rank-dependent seed, so
                    # ranks would issue different numbers of collectives and FSDP would hang
                    "policy": "fixed",
                    "accumulate_chunks": rollout.accumulate_chunks,
                },
                "output": {
                    "num_samples": rollout.num_samples,
                    "streams": self._produced_streams(name),
                },
                "model_input": self._batch_size_one(name, test_cfg),
            }

            self._fsteps_per_chunk[name] = fsteps_per_chunk

            self._warn_on_overwrite(
                name, test_cfg, overrides, fsteps_per_chunk, sample_stride, rollout.num_workers
            )

            if not overrides["output"]["streams"]:
                # no coupling names this component as a producer, so it has nothing to write
                logger.info(
                    f"Component {name!r} produces no coupled stream and will write no output."
                )
                overrides["output"]["num_samples"] = 0

            with open_dict(ccf):
                if ccf.get("test_config") is None:
                    ccf.test_config = {}
                # deep merge: forecast.time_step and the rest of the cascade survive
                ccf.test_config = OmegaConf.merge(ccf.test_config, OmegaConf.create(overrides))

                # Each component opens its own loader and the worker pools are forked one
                # after the other, so this is a per-run global, not a per-component knob.
                if ccf.get("data_loading") is None:
                    ccf.data_loading = {}
                ccf.data_loading.num_workers = rollout.num_workers

    @staticmethod
    def _exact_ratio(chunk_length, step, name: str, key: str) -> int:
        """chunk_length // step, rejecting a remainder rather than flooring it silently."""
        if step <= np.timedelta64(0, "ms"):
            msg = f"Component {name!r} has a non-positive test_config.{key}: {step}."
            raise ValueError(msg)
        if chunk_length % step != np.timedelta64(0, "ms"):
            msg = (
                f"rollout.chunk_length ({chunk_length}) is not an exact multiple of component "
                f"{name!r}'s test_config.{key} ({step}). The components would land on "
                "different times and the interleaving would be meaningless."
            )
            raise ValueError(msg)

        # a positive chunk_length that divides step exactly is necessarily >= one step
        return int(chunk_length // step)

    @staticmethod
    def _batch_size_one(name: str, test_cfg) -> dict:
        """Force batch size 1, which is what get_batch_size_from_config sums to.

        Inference has no reason to batch, and the invariant that every sample run is a sample
        written only holds at 1: validate() runs samples_per_mini_epoch // batch_size batches
        but writes output.num_samples * batch_size of them.
        """
        entries = [
            key
            for key, cfg in test_cfg.get("model_input", {}).items()
            if cfg.get("enabled", True)
        ]
        if len(entries) != 1:
            msg = (
                f"Component {name!r} has {len(entries)} enabled test_config.model_input "
                f"entries ({entries}); coupled inference needs exactly one so that the batch "
                "size is 1."
            )
            raise ValueError(msg)

        return {entries[0]: {"num_samples": 1}}

    @staticmethod
    def _warn_on_overwrite(
        name: str,
        test_cfg,
        overrides: dict,
        fsteps_per_chunk: int,
        sample_stride: int,
        num_workers: int,
    ) -> None:
        """Say what the rollout spec is taking over, so nothing changes silently."""
        if test_cfg.get("shuffle", False):
            logger.warning(
                f"Component {name!r}: test_config.shuffle was True, forced to False. Each "
                "component shuffles with its own rng_seed, so they would visit different "
                "time windows."
            )

        policy = test_cfg.get("forecast", {}).get("policy")
        if policy != "fixed":
            logger.warning(
                f"Component {name!r}: test_config.forecast.policy was {policy!r}, forced to "
                "'fixed'. Rank-dependent step counts deadlock the FSDP collectives."
            )

        num_steps = test_cfg.get("forecast", {}).get("num_steps")
        logger.info(
            f"Component {name!r}: chunk_size={fsteps_per_chunk}, "
            f"num_steps={overrides['forecast']['num_steps']} (was {num_steps}), "
            f"sample_stride={sample_stride}, "
            f"accumulate_chunks={overrides['forecast']['accumulate_chunks']}, "
            f"num_workers={num_workers}, "
            f"output.streams={overrides['output']['streams']}"
        )

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

        for name in self._names:
            self.trainer(name).finish_validation(mini_epoch)

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
        targets_and_auxs = {}
        for loss_name, target_aux in trainer.target_and_aux_calculators_val.items():
            target_idxs = get_target_idxs_from_cfg(trainer.test_cfg, loss_name)
            targets_and_auxs[loss_name] = target_aux.compute(
                trainer.cf.general.istep,
                batch.get_target_samples(target_idxs),
                trainer.model_params,
                trainer.model,
            )

        return targets_and_auxs

    def _run_batch(self, batches: dict, bidx: int, mini_epoch: int) -> None:
        targets, plans, states, accumulated = {}, {}, {}, {}

        self._resubscribe()

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
            if not plan.should_accumulate_chunks:
                continue

            trainer = self.trainer(name)
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

                forecast_step_stride = 1 # TODO determine automatically (from chunkizes)
                producer_reader = self._producer_reader(coupling.producer, stream)
                built: list[DataReaderCoupling] = []

                def _make_inner(base, _p=producer_reader, _s=stream, _b=built,
                                _stride=forecast_step_stride):
                    coupled = DataReaderCoupling(
                        base,
                        _s,
                        producer=_p,
                        forecast_step_stride=_stride,
                        # TODO point it exactly at initialization date
                    )
                    _b.append(coupled)
                    return self._level(coupled, base, _s)

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

    def _resubscribe(self) -> None:
        """Give every component fresh coupling readers, discarding the previous batch's.

        Each batch is a separate initialisation time, i.e. an independent trajectory, but a
        DataReaderCoupling indexes what it holds by wall-clock valid time alone. Two
        trajectories overlap in wall-clock time -- a run started on Jan 1 covers Jan 2, and so
        does one started on Jan 2 -- so without this the second trajectory would read the
        first one's predictions for the shared windows, silently and with no error. Rebuilding
        the readers from the pristine forcings drops the old windows and resets the priming,
        so every trajectory bootstraps from ground truth exactly as the first one did.
        """

        self._subscribers.clear()
        for name in self._names:
            trainer = self.trainer(name)
            pristine = self._pristine_forcings.get(name)
            if pristine is None:
                continue
            trainer.dynamic_forcings = self.subscribe(name, pristine)

    def dispatch_chunk(self, producer: str, chunk: ModelOutput, batch: ModelBatch) -> None:
        """Hand a finished rollout chunk to everything forced by producer.

        Called once per chunk in the rollout loop, where the ModelOutput and the
        batch it was computed from are both in scope.
        """
        for reader in self._subscribers.get(producer, ()):
            reader.add_chunk(chunk, batch)


def _to_numpy(tensor) -> NDArray:
    """Detach a tensor to a numpy array; pass numpy arrays through."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


class DataReaderCoupling(DataReaderTimestep):
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

        # The reader stands on the PRODUCER's grid: its windows are the producer's windows and
        # its period the producer's sampling period. Everything below the levelling wrapper is
        # therefore on one grid -- predictions and primed ground truth alike -- and that wrapper
        # alone converts to the consumer's cadence (coupling_reader_placement.md P6).
        if producer is not None:
            super().__init__(
                producer.time_window_handler,
                dataset.stream_info,
                getattr(producer, "data_start_time", None),
                getattr(producer, "data_end_time", None),
                getattr(producer, "period", None),
            )

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

        # One producer prediction per valid time, held in the data model a reader hands
        # out. `data` and `geoinfos` are physical and already ordered like the consumer's
        # source channels, so ForcingInput normalizes them with the consumer's own
        # statistics exactly as it would data read from disk. The geoinfos are the
        # producer's own, recovered from its target tokens -- never a stand-in, there is
        # no correct substitute for them here.
        self._windows: dict[np.datetime64, ReaderData] = {}

        # The consumer's own reader, kept only for the metadata mirrored above.
        self._dataset = dataset
        # Windows the producer has not emitted *yet* are primed from the PRODUCER's target data:
        # a primed window has to be the same quantity a prediction is, on the same grid and
        # selectable by the same channel map, or priming and prediction reach the consumer by
        # two independent routes that agree only by coincidence.
        self._producer = producer
        self._dispatched = 0

        # where the producer's geoinfos sit in the target tokens it hands over
        offset = 1 + TIMES_WIDTH
        offered = len(self._geoinfo_channels_offered)
        self._geoinfo_slice = slice(offset, offset + offered)

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

    def length(self) -> int:
        """Length of the timeline this reader is defined on, not of what it holds."""
        return self._length

    def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
        """Serve one window of the producer's grid: its prediction, or its ground truth.

        `idx` is an index into the producer's own timeline, because that is the grid this reader
        stands on. The levelling wrapper above converts to whatever the consumer asks for; nothing
        here knows or cares what that cadence is.
        """

        valid_time = self.time_window_handler.window(idx).start
        window = self._windows.get(valid_time)

        if window is not None:
            # A window is served once per consumer of it, so never hand out the stored arrays
            return window.copy()

        if self._dispatched == 0:
            # Bootstrap. The producer is stepped inside the same chunk loop as the consumer, so
            # on the first chunk one direction of the exchange necessarily has nothing to hand
            # over yet. Serving a spoof there feeds the model a field that is not merely stale
            # but absent, at the one step the whole rollout is conditioned on. The producer's own
            # ground truth for that window is what it would have predicted, and is what an
            # uncoupled run would have read.
            logger.debug(
                f"Priming stream '{self._producer_stream}' at {valid_time} (index {idx}) from "
                "the producer's target data: no chunk has been dispatched yet."
            )
            return self._prime(idx)

        logger.debug(
            f"No chunk for stream '{self._producer_stream}' at {valid_time} (index {idx}), "
            "forcing falls back to spoof."
        )
        return ReaderData.empty(len(self.source_idx), len(self.geoinfo_idx))

    def _prime(self, idx: TIndex) -> ReaderData:
        """The producer's ground truth for a window, dressed as the consumer's source.

        Runs through the same `_pred_cols` map a prediction does, so the two paths differ only in
        where the numbers came from.
        """

        if self._producer is None:
            # no producer to read from: fall back to the consumer's own view of the stream
            return self._dataset.get_source(idx)

        rdata = self._producer.get_target(idx)
        if rdata.is_empty():
            return ReaderData.empty(len(self.source_idx), len(self.geoinfo_idx))

        data = rdata.data[:, self._pred_cols]
        geoinfos = rdata.geoinfos[:, self._geo_cols] if len(self._geo_cols) else rdata.geoinfos
        return dataclasses.replace(rdata, data=data, geoinfos=geoinfos)

    def add_chunk(self, chunk: ModelOutput, batch: ModelBatch) -> None:
        """Index the predictions of one rollout chunk by the time they are valid for.

        `batch` is needed because a ModelOutput carries only the predicted values: the
        coordinates, times and geoinfos they live on sit on the batch's samples, split
        across the source and target halves as `_lower_prediction` describes.
        """

        tile = self._producer_tile(chunk)

        for fstep in chunk.forecast_steps:
            preds = chunk.get_physical_prediction(chunk.chunk_idx(fstep), self._producer_stream)
            if preds is None:
                # leading empty steps of the first chunk, or a stream without a decoder
                continue

            for i_source, pred in enumerate(preds):
                entry = self._lower_prediction(
                    chunk.batch_idx(fstep), i_source, pred, batch, tile
                )
                valid_time, window = entry
                self._windows[valid_time] = window

        self._dispatched += 1
        self._evict()

    def _producer_tile(self, chunk: ModelOutput) -> ChunkInfo | None:
        """The producer's own description of the chunk it just emitted.

        The timeline a prediction is stamped on and the stride between its forecast steps
        belong to the *producing* component. They used to be constructor arguments of this
        reader, which is how they came to be left unset: `subscribe()` fills three of six.
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
    ) -> tuple[np.datetime64, ReaderData]:
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
            valid_time = self._producer_twh.window(valid_idx).start

        return valid_time, ReaderData(
            coords=coords.astype(np.float32),
            geoinfos=geoinfos.astype(np.float32),
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
    
    def reset(self): # duplicates resubsribe()
        """Reset the stream for running the next batch => next initialization time."""
        del self._windows
        self._windows = {}
