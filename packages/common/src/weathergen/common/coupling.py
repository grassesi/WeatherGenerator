from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf, open_dict

import weathergen.common.config as config
from weathergen.common.io import IOReaderData
from weathergen.common.logger import init_loggers
from weathergen.datasets.data_reader_base import DataReaderBase
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
        return Trainer(cf.train_logging), cf

@dataclasses.dataclass
class Coupling:
    """One exchange surface: a stream a producer emits and a consumer may later ingest.

    consumer is None while the components only run side by side, which is what lets the
    interleaving be exercised before any field is actually exchanged.
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
            component: ModelCheckpoint(checkpoint_str)
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
        global_cf: config.Config
    ):
        # instantiate all components
        components = {
            name: checkpoint.get_component(private_config, [], [], global_cf)
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
        self._subscribers: dict[str, list[ForcingInput]] = {}

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

        self._check_derivation()
        self._announce_couplings()

    # ------------------------------------------------------------------ checks

    def _check_couplings(self) -> None:
        """Coupling names must resolve, and each stream may have only one producer."""
        producers: dict[str, str] = {}
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
                msg = (
                    f"Coupling {coupling.name!r} uses stream {coupling.stream!r}, which its "
                    f"producer {coupling.producer!r} does not have."
                )
                raise ValueError(msg)

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

    def _check_derivation(self) -> None:
        """Post-condition: the derivation really did put the components on one axis.

        These can no longer fail on a user's config - they catch a bug in the derivation.
        """
        reference = self._names[0]
        ref_cfg = self.trainer(reference).test_cfg
        keys = ("start_date", "end_date", "shuffle", "samples_per_mini_epoch")

        for name in self._names[1:]:
            cfg = self.trainer(name).test_cfg
            for key in keys:
                if cfg.get(key) != ref_cfg.get(key):
                    msg = (
                        f"Derivation failed: {reference!r} and {name!r} disagree on "
                        f"test_config.{key}: {ref_cfg.get(key)} vs {cfg.get(key)}."
                    )
                    raise ValueError(msg)

            if cfg.forecast.offset != ref_cfg.forecast.offset:
                msg = (
                    f"Derivation failed: {reference!r} and {name!r} disagree on "
                    f"test_config.forecast.offset."
                )
                raise ValueError(msg)

        for name in self._names:
            batch_size = self.trainer(name).batch_size_test_per_gpu
            if batch_size != 1:
                msg = f"Derivation failed: component {name!r} has batch size {batch_size}, not 1."
                raise ValueError(msg)

    def _announce_couplings(self) -> None:
        """A run that exchanges nothing looks exactly like one that works. Say which it is."""
        active = [c.name for c in self._couplings.values() if c.consumer is not None]
        logger.info(
            f"{len(self._couplings)} coupling(s) declared, {len(active)} with a consumer. "
            "No field is exchanged at this milestone: components run side by side and each "
            "must reproduce its standalone output exactly."
        )
        for name in self._names:
            forcing = self.trainer(name).dynamic_forcings
            if forcing is not None and not forcing.is_empty:
                logger.info(
                    f"Component {name!r} samples its dynamic forcings from disk, not from "
                    f"another component: {sorted(forcing.forcing_streams)}."
                )

    # ------------------------------------------------------------------ driving

    def validate(self, mini_epoch: int = 0) -> None:
        """Run every component over the validation set, interleaved by chunk."""
        for name in self._names:
            self.trainer(name).model.eval()

        iters = {name: iter(self.trainer(name).data_loader_validation) for name in self._names}
        total = min(len(self.trainer(name).data_loader_validation) for name in self._names)
        max_samples = min(
            self.trainer(name).test_cfg.samples_per_mini_epoch for name in self._names
        )
        batch_size = self.trainer(self._names[0]).batch_size_test_per_gpu
        with_ddp = self.config(self._names[0]).with_ddp

        with torch.no_grad(), tqdm.tqdm(total=total, disable=with_ddp) as pbar:
            bidx = 0
            while True:
                batches = self._next_batches(iters)
                if batches is None:
                    break

                self._assert_aligned(batches, bidx)
                self._run_batch(batches, bidx, mini_epoch)

                pbar.update(batch_size)
                bidx += 1
                if (bidx * batch_size) > max_samples:
                    break

        for name in self._names:
            self.trainer(name).finish_validation(mini_epoch)

    def _next_batches(self, iters) -> dict | None:
        """One batch per component, or None once any component is exhausted."""
        batches = {}
        for name in self._names:
            try:
                batches[name] = next(iters[name])
            except StopIteration:
                logger.info(f"Component {name!r} exhausted its validation set.")
                return None

        return batches

    @staticmethod
    def _sample_idxs(batch) -> list[int]:
        return [
            next(iter(sample.streams_data.values())).sample_idx
            for sample in batch.get_source_samples().get_samples()
        ]

    def _valid_times(self, name: str, batch) -> list:
        """Absolute start times of a batch's source windows, for this component's grid.

        Sample indices are not comparable across components: each strides its own index
        space (window(idx) = start_date + idx * time_window_step), so aligned components sit
        at different indices that denote the same instant. Comparing the times instead is
        both correct under striding and stricter than comparing indices ever was.
        """
        twh = self.trainer(name).dataset.time_window_handler
        return [twh.window(idx).start for idx in self._sample_idxs(batch)]

    def _assert_aligned(self, batches: dict, bidx: int) -> None:
        """All components must be looking at the same time windows.

        MultiStreamDataSampler.__iter__ skips empty and NaN batches independently per
        component, so equal batch positions do not imply equal dates. The shared output
        store keys samples by position, so a drift here would silently file two different
        dates under one sample index.
        """
        reference = self._names[0]
        ref_times = self._valid_times(reference, batches[reference])
        for name in self._names[1:]:
            times = self._valid_times(name, batches[name])
            if times != ref_times:
                msg = (
                    f"Components drifted apart at batch {bidx}: {reference!r} is at "
                    f"{ref_times}, {name!r} at {times}. A component skipped an empty or NaN "
                    "batch that the others did not."
                )
                raise RuntimeError(msg)

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
        for i in range(max(len(plan.chunks) for plan in plans.values())):
            for name in self._names:
                plan = plans[name]
                if i >= len(plan.chunks):
                    continue

                trainer = self.trainer(name)
                with self._autocast(trainer):
                    states[name] = trainer.step_chunk(states[name], plan.chunks[i])

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

    def subscribe_stream(self, subscriber: ForcingInput) -> DataReaderCoupling:
        for stream in subscriber.forcing_streams.keys():
            self._subscribers[stream] = subscriber

    def get_forcings(self, forcing_streams):
        pass

    def dispatch_chunk(self, chunk: ModelOutput):
        pass

class DataReaderCoupling(DataReaderBase):
    """Implments only the interface required by ForcingInput"""
    def __init__(self, dataset: DataReaderBase):
        # inherit all if possible from corresponding "real" stream
        
        # required for tokenization
        self.stream_info = dataset.stream_info
        
        # required for normalize_source_channels
        self.source_idx
        self.mean
        self.stdev
        
        # required for normalize_geoinfos
        self.geoinfo_idx
        self.mean_geoinfo
        self.stdev_geoinfo
        
        self._chunks: list[ModelOutput]
    
    def get_source(idx: int) -> IOReaderData:
        pass
    
    # required for spoofing?
    def get_geoinfo_size() -> int:
        pass
    
    def add_chunk(self, chunk: ModelOutput):
        self._chunks.append(chunk)