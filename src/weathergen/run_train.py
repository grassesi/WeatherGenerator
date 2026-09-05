# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
The entry point for training and inference weathergen-atmo
"""

import argparse
import logging
import os
import pdb
import sys
import time
import traceback
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

import weathergen.common.config as config
import weathergen.utils.cli as cli
from weathergen.common.coupling import Couplings
from weathergen.common.logger import init_loggers
from weathergen.train.trainer import Trainer

logger = logging.getLogger(__name__)


def train() -> None:
    """Entry point for calling the training code from the command line."""
    main([cli.Stage.train] + sys.argv[1:])


def train_continue() -> None:
    """Entry point for calling train_continue from the command line."""
    main([cli.Stage.train_continue] + sys.argv[1:])


def inference():
    """Entry point for calling the inference code from the command line."""
    main([cli.Stage.inference] + sys.argv[1:])


def main(argl: list[str]):
    try:
        argl = _fix_argl(argl)
    except ValueError as e:
        logger.error(str(e))

    argl = _fix_argl_coupled_inference(argl)

    parser = cli.get_main_parser()
    args = parser.parse_args(argl)

    match args.stage:
        case cli.Stage.train:
            run_train(args)
        case cli.Stage.train_continue:
            run_continue(args)
        case cli.Stage.inference:
            run_inference(args)
        case cli.Stage.coupled_inference:
            run_coupled_inference(args)
        case _:
            logger.error("No stage was found.")


def _fix_argl(argl):  # TODO remove this fix after grace period
    """Ensure `stage` positional argument is in arglist."""
    if argl[0] not in cli.Stage:
        try:
            stage = os.environ.get("WEATHERGEN_STAGE")
        except KeyError as e:
            msg = (
                "`stage` postional argument and environment variable 'WEATHERGEN_STAGE' missing.",
                "Provide either one or the other.",
            )
            raise ValueError(msg) from e

        argl = [stage] + argl

    return argl


# The keys an inference stage carries to mean "this is really a coupled run". They are the
# two positional arguments of the coupled_inference parser, passed as options instead.
_COUPLED_KEYS = ("couplings", "components")

# Model loading flags that inference requires and coupled inference has no parser entry for:
# every component names its own checkpoint in `components`, so these are dropped, not mapped.
# Maps flag -> number of values it consumes.
_INFERENCE_ONLY_FLAGS = {
    "--from-run-id": 1,
    "-id": 1,
    "--mini-epoch": 1,
    "-e": 1,
    "--reuse-run-id": 0,
}


def _fix_argl_coupled_inference(argl: list[str]) -> list[str]:
    """Rewrite an `inference` invocation carrying coupling options into a coupled one."""

    if not argl or argl[0] != cli.Stage.inference:
        return argl

    couplings, components = _peek_coupled_options(argl[1:])

    if couplings is None and components is None:
        return argl

    logger.info(
        f"Coupling options found ({_COUPLED_KEYS[0]}={couplings}, "
        f"{_COUPLED_KEYS[1]}={components}); dispatching to {cli.Stage.coupled_inference}."
    )

    rest = _drop_flags(argl[1:], _INFERENCE_ONLY_FLAGS)

    return [cli.Stage.coupled_inference, couplings, *components, *rest]


def _peek_coupled_options(args: list[str]) -> tuple[str | None, list[str] | None]:
    """Read the coupling keys out of an inference arglist"""

    parser = argparse.ArgumentParser(allow_abbrev=False, add_help=False)
    parser.add_argument("--config", type=Path, nargs="*", default=[])
    parser.add_argument("--options", nargs="+", default=[])
    known, _ = parser.parse_known_args(args)

    found: dict[str, object] = {}

    sources = []
    for path in known.config:
        try:
            sources.append(OmegaConf.load(path))
        except Exception as e:  # a broken or missing config is the config loader's to report
            logger.debug(f"Could not peek at config {path} for coupling options: {e}")
    if known.options:
        sources.append(OmegaConf.from_dotlist(known.options))

    for source in sources:
        if not isinstance(source, DictConfig):
            continue
        for key in _COUPLED_KEYS:
            if source.get(key) is not None:
                found[key] = source.get(key)

    couplings = found.get(_COUPLED_KEYS[0])
    components = found.get(_COUPLED_KEYS[1])

    return (
        None if couplings is None else str(couplings),
        None if components is None else _split_components(components),
    )


def _split_components(value) -> list[str]:
    """Normalize the `components` option into the argv entries the coupled parser takes.

    Accepts the whitespace or comma separated string a dotlist option produces
    (`components="Atmo=a@16 Ocean=b@16"`) as well as a yaml list, since both survive the
    launcher's round trip through `config_command_line.yaml`.
    """

    if isinstance(value, str):
        entries = value.replace(",", " ").split()
    else:
        entries = [str(entry) for entry in value]

    if not entries:
        msg = "The 'components' option is empty; coupled inference needs at least one component."
        raise ValueError(msg)

    for entry in entries:
        name, _, checkpoint = entry.partition("=")
        if not name or "@" not in checkpoint:
            msg = (
                f"Coupled inference component {entry!r} is malformed. Expected "
                "'<Component>=<run_id>@<mini_epoch>', e.g. 'Atmo=zhyqsbi8@16'."
            )
            raise ValueError(msg)

    return entries


def _drop_flags(args: list[str], flags: dict[str, int]) -> list[str]:
    """Remove `flags` and the values they consume from an arglist, `--flag=value` included."""

    kept: list[str] = []
    skip = 0

    for arg in args:
        if skip:
            skip -= 1
            continue

        name = arg.split("=", 1)[0]
        if name in flags:
            if "=" not in arg:
                skip = flags[name]
            continue

        kept.append(arg)

    return kept


def run_inference(args):
    """
    Inference function for WeatherGenerator model.

    Note: Additional configuration for inference (`test_config`) is set in the function.
    """

    cli_overwrite = config.from_cli_arglist(args.options)
    cf = config.load_merge_configs(
        args.private_config,
        args.from_run_id,
        args.mini_epoch,
        args.base_config,
        *args.config,
        {},
        cli_overwrite,
    )
    cf = config.set_run_id(cf, args.run_id, args.reuse_run_id)

    devices = Trainer.init_torch()
    cf = Trainer.init_ddp(cf)

    init_loggers(cf.general.run_id)

    logger.info(f"DDP initialization: rank={cf.rank}, world_size={cf.world_size}")

    cf.general.run_history += [(args.from_run_id, cf.general.istep)]

    trainer = Trainer(cf.train_logging)

    try:
        trainer.inference(cf, devices, args.from_run_id, args.mini_epoch)
    except Exception:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        if cf.world_size == 1:
            pdb.post_mortem(tb)


def run_coupled_inference(args):
    couplings = Couplings.from_args(args.couplings, args.components)
    logger.info(f"Coupled inference setup:\n{couplings}")

    global_cf = couplings.global_intialization(args.run_id)
    try:
        couplings.run(args.private_config, global_cf)
    except Exception:
        # Never swallow the exception: the coupled setup validates itself with ValueErrors,
        # and returning normally here would report success for a run that never happened.
        traceback.print_exc()
        if global_cf.world_size == 1 and sys.stdin.isatty():
            _, _, tb = sys.exc_info()
            pdb.post_mortem(tb)
        raise


def run_continue(args):
    """
    Function to continue training for WeatherGenerator model.

    Note: All model configurations are set in the function body.
    """

    cli_overwrite = config.from_cli_arglist(args.options)
    cf = config.load_merge_configs(
        args.private_config,
        args.from_run_id,
        args.mini_epoch,
        args.base_config,
        *args.config,
        {},
        cli_overwrite,
    )
    cf = config.set_run_id(cf, args.run_id, args.reuse_run_id)

    mp_method = cf.general.get("multiprocessing_method", "fork")
    devices = Trainer.init_torch(multiprocessing_method=mp_method)
    cf = Trainer.init_ddp(cf)

    init_loggers(cf.general.run_id)

    # track history of run to ensure traceability of results
    cf.general.run_history += [(args.from_run_id, cf.general.istep)]

    trainer = Trainer(cf.train_logging)

    try:
        trainer.run(cf, devices, args.from_run_id, args.mini_epoch)
    except Exception:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        if cf.world_size == 1:
            pdb.post_mortem(tb)


def run_train(args):
    """
    Training function for WeatherGenerator model.

    Note: All model configurations are set in the function body.
    """

    cli_overwrite = config.from_cli_arglist(args.options)

    cf = config.load_merge_configs(
        args.private_config, None, None, args.base_config, *args.config, cli_overwrite
    )
    cf = config.set_run_id(cf, args.run_id, False)

    cf.data_loading.rng_seed = int(time.time())
    mp_method = cf.general.get("multiprocessing_method", "fork")
    devices = Trainer.init_torch(multiprocessing_method=mp_method)
    cf = Trainer.init_ddp(cf)

    # this line should probably come after the processes have been sorted out else we get lots
    # of duplication due to multiple process in the multiGPU case
    init_loggers(cf.general.run_id)

    logger.info(f"DDP initialization: rank={cf.rank}, world_size={cf.world_size}")

    if cf.with_flash_attention:
        assert cf.with_mixed_precision

    trainer = Trainer(cf.train_logging)

    try:
        trainer.run(cf, devices)
    except Exception:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        if cf.world_size == 1:
            pdb.post_mortem(tb)


if __name__ == "__main__":
    main(sys.argv[1:])
