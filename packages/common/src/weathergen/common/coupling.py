from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import torch
import tqdm
from omegaconf import OmegaConf

import weathergen.common.config as config
from weathergen.common.io import IOReaderData
from weathergen.common.logger import init_loggers
from weathergen.datasets.data_reader_base import DataReaderBase
from weathergen.model.forcing import ForcingInput
from weathergen.model.model import ModelOutput
from weathergen.train.trainer import Trainer
from weathergen.train.utils import extract_batch_metadata, get_target_idxs_from_cfg

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
        options: list[str],
        shared_config: config.Config
    ) -> tuple[Trainer, config.Config]:
        configs = [] if configs is None else configs
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
        return Trainer(cf.train_logging), cf

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
    _devices: str = dataclasses.field(init=False)
    _run_history: list[tuple[str, int]] = dataclasses.field(init=False)
    
    def __str__(self) -> str:
        return f"couplings: {self.couplings}\n{self.checkpoints}"

    @classmethod
    def from_args(cls, couplings: Path, components: list[str]):
        components_kv = (component.split("=") for component in components)
        components = {
            component: ModelCheckpoint(checkpoint_str) for component, checkpoint_str in components_kv
        }
        parsed_couplings = {
            name: Coupling(name, **coupling)
            for name, coupling in OmegaConf.load(couplings).items()
        }
        
        return cls(parsed_couplings, components)
    
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
            name: checkpoint.get_component(
                private_config, None, None, global_cf, 
            ) for name, checkpoint in self.checkpoints.items()
        }
        
        for _, ccf in components.values():
            ccf.general.run_history += self._run_history

        # the coupler owns the loops: components are stepped, they do not run themselves
        coupler = Coupler(components, self.couplings)
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
    ):
        # deterministic order: every rank must issue its collectives in the same sequence
        self._names = sorted(components)
        self._components = components
        self._couplings = couplings or {}
        self._subscribers: dict[str, list[ForcingInput]] = {}

    def trainer(self, name: str) -> Trainer:
        return self._components[name][0]

    def config(self, name: str) -> config.Config:
        return self._components[name][1]

    def setup(self, devices, checkpoints: dict[str, ModelCheckpoint]) -> None:
        """Build every component, then verify they can share one run."""
        for name in self._names:
            trainer, ccf = self._components[name]
            checkpoint = checkpoints[name]
            logger.info(f"Setting up component {name!r} from {checkpoint.run_id}.")
            trainer.setup_inference(
                ccf, devices, checkpoint.run_id, checkpoint.mini_epoch, name
            )

        self._check_couplings()
        self._check_alignment()
        self._check_output_streams()

    # ------------------------------------------------------------------ checks

    def _check_couplings(self) -> None:
        """Couplings are not acted on yet, but their names must resolve."""
        for coupling in self._couplings.values():
            for role, component in (
                ("producer", coupling.producer),
                ("consumer", coupling.consumer),
            ):
                if component not in self._components:
                    msg = (
                        f"Coupling {coupling.name!r} names {role} {component!r}, "
                        f"which is not one of the components {self._names}."
                    )
                    raise ValueError(msg)

            for role, component in (
                ("producer", coupling.producer),
                ("consumer", coupling.consumer),
            ):
                streams = self.config(component).streams
                if coupling.stream not in streams:
                    msg = (
                        f"Coupling {coupling.name!r} uses stream {coupling.stream!r}, "
                        f"which {role} {component!r} does not have."
                    )
                    raise ValueError(msg)

    def _check_alignment(self) -> None:
        """Verify the components share one sample index space and one collective order.

        The shared output store keys samples by position, so the components must walk the
        same time windows. And under FSDP every rank must issue the same collectives in
        the same order, which constrains the forecast policy.
        """
        reference = self._names[0]
        ref_cfg = self.trainer(reference).test_cfg

        for name in self._names[1:]:
            cfg = self.trainer(name).test_cfg
            for key in self._SHARED_WINDOW_KEYS:
                if cfg.get(key) != ref_cfg.get(key):
                    msg = (
                        f"Components {reference!r} and {name!r} disagree on "
                        f"test_config.{key}: {ref_cfg.get(key)} vs {cfg.get(key)}. "
                        "All components must span the same time windows."
                    )
                    raise ValueError(msg)

            ref_batch = self.trainer(reference).batch_size_test_per_gpu
            batch = self.trainer(name).batch_size_test_per_gpu
            if batch != ref_batch:
                msg = (
                    f"Components {reference!r} and {name!r} disagree on test batch size: "
                    f"{ref_batch} vs {batch}. Output samples are keyed by "
                    "batch_idx * batch_size, so the batch sizes must match."
                )
                raise ValueError(msg)

            ref_offset = ref_cfg.get("forecast", {}).get("offset")
            offset = cfg.get("forecast", {}).get("offset")
            if offset != ref_offset:
                msg = (
                    f"Components {reference!r} and {name!r} disagree on "
                    f"test_config.forecast.offset: {ref_offset} vs {offset}."
                )
                raise ValueError(msg)

        for name in self._names:
            cfg = self.trainer(name).test_cfg
            if cfg.get("shuffle", False):
                msg = (
                    f"Component {name!r} has test_config.shuffle=True. Each component "
                    "shuffles with its own rng_seed, so the components would visit "
                    "different time windows. Set shuffle: False."
                )
                raise ValueError(msg)

            policy = cfg.get("forecast", {}).get("policy")
            if policy not in self._RANK_UNIFORM_POLICIES:
                msg = (
                    f"Component {name!r} uses forecast.policy={policy!r}. The per-batch "
                    "forecast step count is then drawn from a seed that "
                    "MultiStreamDataSampler.worker_workset makes rank-dependent, so ranks "
                    "would run different numbers of steps and the collectives would "
                    f"deadlock. Use one of {list(self._RANK_UNIFORM_POLICIES)}."
                )
                raise ValueError(msg)

        time_steps = {
            name: str(self.trainer(name).test_cfg.get("forecast", {}).get("time_step"))
            for name in self._names
        }
        if len(set(time_steps.values())) > 1:
            logger.warning(
                "Components advance at different forecast time steps: "
                f"{time_steps}. They are interleaved by chunk index, so chunk i is a "
                "different physical time for each of them. Harmless while they do not "
                "interact, but it must become a shared time axis before coupling."
            )

    def _output_streams(self, name: str) -> list[str]:
        """Streams this component will write, or empty if it writes nothing."""
        trainer = self.trainer(name)
        output_cfg = trainer.test_cfg.get("output", {})
        if output_cfg.get("num_samples", 0) <= 0:
            return []

        streams = output_cfg.get("streams")
        if streams is None:
            return list(trainer.cf.streams.keys())

        return list(streams)

    def _check_output_streams(self) -> None:
        """All components write into one store, keyed by <sample>/<stream>/<step>."""
        owner: dict[str, str] = {}
        for name in self._names:
            for stream in self._output_streams(name):
                if stream in owner:
                    msg = (
                        f"Components {owner[stream]!r} and {name!r} would both write "
                        f"stream {stream!r} into the shared output store, overwriting "
                        "each other. Give each component a disjoint "
                        "test_config.output.streams."
                    )
                    raise ValueError(msg)
                owner[stream] = name

        logger.info(f"Output streams per component: {owner}")

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

    def _assert_aligned(self, batches: dict, bidx: int) -> None:
        """All components must be looking at the same time windows.

        MultiStreamDataSampler.__iter__ skips empty and NaN batches independently per
        component, so equal batch positions do not imply equal dates. The shared output
        store keys samples by position, so a drift here would silently file two different
        dates under one sample index.
        """
        reference = self._names[0]
        ref_idxs = self._sample_idxs(batches[reference])
        for name in self._names[1:]:
            idxs = self._sample_idxs(batches[name])
            if idxs != ref_idxs:
                msg = (
                    f"Components drifted apart at batch {bidx}: {reference!r} is at "
                    f"samples {ref_idxs}, {name!r} at {idxs}. A component skipped an "
                    "empty or NaN batch that the others did not."
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