# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""What a couplings file declares: the exchange surfaces and the shared rollout."""

from __future__ import annotations

import dataclasses

import numpy as np

import weathergen.common.config as config


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
    # spacing between successive initialization times, in chunks (init_stride_design.md D).
    # Chunks are the unit shared by every component regardless of its own cadence. 1 keeps the
    # previous behaviour: each batch's initialization is exactly one chunk after the last.
    init_stride_chunks: int = 1

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
            init_stride_chunks=int(cfg.get("init_stride_chunks", 1)),
        )

        for key in ("num_chunks", "num_samples", "init_stride_chunks"):
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


def produced_streams(couplings: dict[str, Coupling], name: str) -> list[str]:
    """Streams a component is the producer of, in declaration order."""
    return [c.stream for c in couplings.values() if c.producer == name]
