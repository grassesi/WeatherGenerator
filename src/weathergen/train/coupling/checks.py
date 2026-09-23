# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Setup checks on the wiring of a coupled run: which couplings exist and whether they can work."""

from __future__ import annotations

import logging
import typing

import numpy as np

import weathergen.common.config as config
from weathergen.common.config import timedelta_to_str
from weathergen.train.coupling.spec import Coupling
from weathergen.train.utils import resolve_stage_configs

if typing.TYPE_CHECKING:
    from weathergen.train.coupling.coupler import Coupler

logger = logging.getLogger(__name__)

_ZERO = np.timedelta64(0, "ms")


def resolve_couplings(
    couplings: dict[str, Coupling], configs: dict[str, config.Config]
) -> dict[str, Coupling]:
    """Coupling names must resolve, and each stream may have only one producer.

    A coupling whose producer does not carry the stream is dropped rather than rejected, so the
    couplings that survive are returned rather than all of them. `configs` holds every
    component's config, keyed by name.
    """
    producers: dict[str, str] = {}
    live: dict[str, Coupling] = {}
    for coupling in couplings.values():
        if coupling.producer not in configs:
            msg = (
                f"Coupling {coupling.name!r} names producer {coupling.producer!r}, "
                f"which is not one of the components {sorted(configs)}."
            )
            raise ValueError(msg)

        if coupling.consumer is not None and coupling.consumer not in configs:
            msg = (
                f"Coupling {coupling.name!r} names consumer {coupling.consumer!r}, "
                f"which is not one of the components {sorted(configs)}."
            )
            raise ValueError(msg)

        streams = configs[coupling.producer].streams
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

    return live


def check_forcing_lags(coupler: Coupler) -> None:
    """Every source window a coupled request touches must exist when it is asked for.

    With `len_c` the consumer's window length, `P` the producer's emission cadence and
    `D` a whole chunk when the producer is stepped *after* the consumer, the lag must
    satisfy `L >= len_c - P + D`. Below that the consumer reaches into a window its
    producer has not emitted yet, which reads back as an absent forcing rather than as an
    error (`forcing_lag_design.md` D4).
    """

    if coupler.rollout is None:
        return

    for coupling in coupler.couplings.values():
        if coupling.consumer is None:
            continue

        forcings = coupler.pristine_forcings(coupling.consumer)
        if forcings is None or coupling.stream not in forcings.lags:
            continue

        lag = forcings.lags[coupling.stream]
        len_c = window_len(coupler.config(coupling.consumer))
        cadence = emission_cadence(coupler.config(coupling.producer))
        before = coupler.names.index(coupling.producer) < coupler.names.index(coupling.consumer)
        slack = _ZERO if before else coupler.rollout.chunk_length
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

        if lag > coupler.rollout.chunk_length:
            msg = (
                f"{where} declares forcing_lag {timedelta_to_str(lag)}, longer than "
                f"rollout.chunk_length {timedelta_to_str(coupler.rollout.chunk_length)}. The "
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


def window_len(cfg: config.Config) -> np.timedelta64:
    """A component's own time window length, as its test config resolves it."""
    _, _, test_cfg = resolve_stage_configs(cfg)
    return config.parse_timedelta(test_cfg.get("time_window_len"))


def emission_cadence(cfg: config.Config) -> np.timedelta64:
    """Wall-clock spacing of the windows a component emits, i.e. its forecast step."""
    _, _, test_cfg = resolve_stage_configs(cfg)
    time_step = test_cfg.get("forecast", {}).get("time_step")
    if time_step is None:
        return config.parse_timedelta(test_cfg.get("time_window_step"))
    return config.parse_timedelta(time_step)


def announce_couplings(coupler: Coupler) -> None:
    """A run that exchanges nothing looks exactly like one that works. Say which it is.

    Reports what the wiring actually did, not what the config asked for: a coupling only
    takes effect if the consumer already reads that stream as a dynamic forcing, so a
    declared consumer can substitute nothing at all.

    The record comes from `Coupler.substituted`, written by subscribe() as it substitutes. Deriving
    it instead from the reader stack -- "is the outermost reader a DataReaderCoupling" -- is
    what made this function report a live exchange as dead for as long as subscribe() wrapped
    that reader in a HoldingReader.
    """
    exchanged: dict[str, list[str]] = {name: [] for name in coupler.names}
    for consumer, stream in coupler.substituted:
        exchanged.setdefault(consumer, []).append(stream)
    for streams in exchanged.values():
        streams.sort()

    declared = [c for c in coupler.couplings.values() if c.consumer is not None]
    live = [c for c in declared if c.stream in exchanged.get(c.consumer, [])]
    logger.info(
        f"{len(coupler.couplings)} coupling(s) declared, {len(declared)} with a consumer, "
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

    for name in coupler.names:
        forcing = coupler.trainer(name).dynamic_forcings
        if forcing is None or forcing.is_empty:
            continue
        coupled = exchanged.get(name, [])
        from_disk = sorted(set(forcing.forcing_streams) - set(coupled))
        if coupled:
            logger.info(f"Component {name!r} is forced by another component on: {coupled}.")
        if from_disk:
            logger.info(f"Component {name!r} samples from disk: {from_disk}.")
