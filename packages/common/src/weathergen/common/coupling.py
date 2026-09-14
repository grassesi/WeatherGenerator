from __future__ import annotations

import copy
import dataclasses
import logging
from pathlib import Path

import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf, open_dict

import weathergen.common.config as config
from weathergen.common.config import timedelta_to_str
from weathergen.common.logger import init_loggers
from weathergen.datasets.batch import ModelBatch
from weathergen.datasets.coupling_reader import DataReaderCoupling
from weathergen.datasets.data_reader_base import (
    DataReaderBase,
    rebase_innermost,
)
from weathergen.model.forcing import ForcingInput
from weathergen.model.model import ModelOutput
from weathergen.train.trainer import Trainer
from weathergen.train.utils import (
    extract_batch_metadata,
    get_target_idxs_from_cfg,
    resolve_stage_configs,
)

logger = logging.getLogger(__name__)

_ZERO = np.timedelta64(0, "ms")


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
            name: checkpoint.get_component(
                private_config, configs, self._options_for(name, options), global_cf
            )
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

    def _options_for(self, name: str, options: list[str]) -> list[str]:
        """The --options entries that apply to one component.

        An option may be addressed at a single component by prefixing it with that
        component's name and a colon -- `Ocean:with_fsdp=False`. Everything else applies to
        every component, which is what a run-wide execution setting wants.

        Without this a coupled run has no route to one component's config: each is loaded from
        its own checkpoint, and the couplings file is deliberately about the rollout rather
        than about how a particular half is executed. Colons are matched only against the
        known component names, so a value carrying one -- a duration, say -- is untouched.
        """

        prefixes = {f"{component}:" for component in self.checkpoints}
        mine, shared = [], []
        for option in options:
            for prefix in prefixes:
                if option.startswith(prefix):
                    if prefix == f"{name}:":
                        mine.append(option[len(prefix) :])
                    break
            else:
                shared.append(option)

        if mine:
            logger.info(f"Component {name!r} takes the per-component overrides {mine}.")
        return [*shared, *mine]

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

        self._check_forcing_lags()
        self._announce_couplings()

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

        self._report_provenance()

        for name in self._names:
            self.trainer(name).finish_validation(mini_epoch)

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
                init_time = self._init_times.get(coupling.producer)
                built: list[DataReaderCoupling] = []

                def _make_inner(base, _p=producer_reader, _s=stream, _b=built,
                                _stride=forecast_step_stride, _init=init_time):
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
                        forecast_step_stride=_stride,
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
