# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The coupled_inference entry point: load every component, then hand them to the Coupler."""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

from omegaconf import OmegaConf

import weathergen.common.config as config
from weathergen.common.logger import init_loggers
from weathergen.train.coupling.coupler import Coupler
from weathergen.train.coupling.spec import Coupling, Rollout
from weathergen.train.trainer import Trainer

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
        nowhere else to live. Whatever they set, `derive_component_configs` still
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
