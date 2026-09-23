# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Push the couplings file's rollout spec down onto every component's config."""

from __future__ import annotations

import logging

import numpy as np
from omegaconf import OmegaConf, open_dict

import weathergen.common.config as config
from weathergen.train.coupling.spec import Coupling, Rollout, produced_streams
from weathergen.train.utils import resolve_stage_configs

logger = logging.getLogger(__name__)


def derive_component_configs(
    configs: dict[str, config.Config],
    rollout: Rollout | None,
    couplings: dict[str, Coupling],
) -> None:
    """Push the global rollout spec down onto every component's test_config.

    The components cannot be made to agree by hand - nothing overrides what is baked into
    their checkpoints - so instead of checking them against each other, everything they
    must share is derived here from one spec and written into test_config, the last layer
    of the training -> validation -> test cascade.

    `configs` maps each component to its config, in the order they are stepped; each is
    updated in place. `couplings` decides which streams a component writes.
    """
    if rollout is None:
        msg = "Coupler requires a Rollout spec; none was provided."
        raise ValueError(msg)

    if rollout.accumulate_chunks:
        msg = (
            "rollout.accumulate_chunks is true, but a coupled rollout always runs "
            "test_config.inference_only, which builds no targets: there is nothing to score "
            "the assembled chunks against. Remove accumulate_chunks from the couplings file."
        )
        raise ValueError(msg)

    for name, ccf in configs.items():
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

        fsteps_per_chunk = _exact_ratio(
            rollout.chunk_length, time_step, name, "forecast.time_step"
        )
        windows_per_chunk = _exact_ratio(
            rollout.chunk_length, window_step, name, "time_window_step"
        )
        init_stride_fsteps = rollout.init_stride_chunks * windows_per_chunk

        overrides = {
            "start_date": f"${{{config._DATETIME_TYPE_NAME}:{rollout.start_date}}}",
            "end_date": f"${{{config._DATETIME_TYPE_NAME}:{rollout.end_date}}}",
            "_start_date": rollout.start_date,
            "_end_date": rollout.end_date,
            # successive initial conditions are init_stride_chunks chunks apart; the
            # sampler strides in index units, so each component covers the same absolute
            # times (init_stride_design.md D, S2)
            "init_stride_fsteps": init_stride_fsteps,
            # inference writes every sample it runs, which only holds at batch size 1
            "samples_per_mini_epoch": rollout.num_samples,
            # each component shuffles with its own rng_seed, so they would otherwise
            # visit different windows
            "shuffle": False,
            # rollout.start_date must actually draw the first sample. The sampler
            # otherwise substitutes the next usable window for an empty or NaN one, and
            # does so per component, which walks two components onto different dates.
            "strict_batches": True,
            # a coupled rollout is judged on the arrays it writes, and a rollout past the
            # data end has no targets for its later chunks, so the target half is not built
            "inference_only": True,
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
                "streams": produced_streams(couplings, name),
            },
            "model_input": _batch_size_one(name, test_cfg),
        }

        _warn_on_overwrite(
            name, test_cfg, overrides, fsteps_per_chunk, init_stride_fsteps, rollout.num_workers
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


def _warn_on_overwrite(
    name: str,
    test_cfg,
    overrides: dict,
    fsteps_per_chunk: int,
    init_stride_fsteps: int,
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

    if not test_cfg.get("inference_only", True):
        logger.warning(
            f"Component {name!r}: test_config.inference_only was False, forced to True. A "
            "coupled rollout builds no targets and computes no loss."
        )

    num_steps = test_cfg.get("forecast", {}).get("num_steps")
    logger.info(
        f"Component {name!r}: chunk_size={fsteps_per_chunk}, "
        f"num_steps={overrides['forecast']['num_steps']} (was {num_steps}), "
        f"init_stride_fsteps={init_stride_fsteps}, "
        f"accumulate_chunks={overrides['forecast']['accumulate_chunks']}, "
        f"num_workers={num_workers}, "
        f"output.streams={overrides['output']['streams']}"
    )
