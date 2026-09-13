# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
One tile of a chunked rollout, and everything a consumer needs to place it in time.

A rollout is split into tiles of at most `chunk_size` forecast steps, and each tile is
stepped independently -- by the Trainer for a single model, or by the Coupler when several
models are interleaved. `ChunkInfo` is what describes one such tile.

It exists because the tile used to be described by nothing but its list of global forecast
steps, which cannot answer two questions its consumers ask:

- *which* tile is this? Nothing carried an index. Deriving one from the step list needs the
  chunk width from somewhere else and is wrong on the first tile, whose step list is padded
  down to global step 0 (see `pads_to_zero`), and wrong again on a short final tile.
- *whose timeline* are these steps on? In a coupled run the tile travels from the producing
  component to the consuming one inside a `ModelOutput`, and the producer's window handler
  and step stride have to travel with it. Passing them separately is how they came to be
  omitted: they were constructor arguments of the coupling reader that nobody filled in.

Both are recorded here rather than re-derived, so that a consumer never divides by
`len(steps)` to recover a cadence -- the padded first tile and a short final tile make that
division wrong exactly where it matters.
"""

from __future__ import annotations

import dataclasses
import typing

if typing.TYPE_CHECKING:
    from weathergen.datasets.data_reader_base import TimeWindowHandler


@dataclasses.dataclass(frozen=True)
class ChunkInfo:
    """One tile of a rollout: which forecast steps it covers, and on whose timeline.

    Attributes
    ----------
    index :
        Position of this tile in the rollout, counting from 0.
    steps :
        The global forecast steps this tile actually rolls out, ascending and contiguous.
        This is the *unpadded* list -- what the model iterates over. See `forecast_steps`.
    forecast_offset :
        The batch's forecast offset, i.e. the first global step that carries a target.
        Steps below it exist only as empty leading slots of the first tile.
    chunk_size :
        The configured tile width, `forecast.chunk_size` as read from the config. Equal to
        `len(steps)` for every tile except a short final one, and deliberately kept
        separate from it: a consumer that needs the cadence must use this, never the
        length of a particular tile.
    pads_to_zero :
        Whether `forecast_steps` is padded down to global step 0. True for the first tile,
        which keeps its leading `forecast_offset` steps as empty slots so that concatenating
        the tiles of a rollout stays indexed by global forecast step.
    time_window_handler :
        The handler of the component that produces this tile, i.e. the timeline its steps
        are indices on. None when the tile is not handed to another component.
    step_stride :
        Dataset window indices advanced per forecast step on that timeline,
        `forecast.time_step // time_window_step`.
    """

    index: int
    steps: tuple[int, ...]
    forecast_offset: int
    chunk_size: int
    pads_to_zero: bool = False
    time_window_handler: TimeWindowHandler | None = None
    step_stride: int = 1

    def __post_init__(self) -> None:
        if not self.steps:
            msg = "A ChunkInfo must cover at least one forecast step."
            raise ValueError(msg)
        if list(self.steps) != list(range(self.steps[0], self.steps[-1] + 1)):
            msg = f"ChunkInfo steps must be ascending and contiguous, got {self.steps}."
            raise ValueError(msg)
        for name in ("index", "forecast_offset"):
            if getattr(self, name) < 0:
                msg = f"ChunkInfo.{name} must be >= 0, got {getattr(self, name)}."
                raise ValueError(msg)
        for name in ("chunk_size", "step_stride"):
            if getattr(self, name) < 1:
                msg = f"ChunkInfo.{name} must be >= 1, got {getattr(self, name)}."
                raise ValueError(msg)

    @classmethod
    def tiles(
        cls,
        output_idxs: list[int],
        chunk_size: int,
        time_window_handler: TimeWindowHandler | None = None,
        step_stride: int = 1,
    ) -> list[ChunkInfo]:
        """Split a rollout's global forecast steps into tiles of at most chunk_size steps.

        The first tile is the only one that pads down to step 0, which is what keeps the
        concatenation of all tiles indexed by global forecast step.
        """

        if chunk_size < 1:
            msg = f"forecast.chunk_size must be >= 1, got {chunk_size}."
            raise ValueError(msg)

        return [
            cls(
                index=index,
                steps=tuple(output_idxs[start : start + chunk_size]),
                forecast_offset=output_idxs[0],
                chunk_size=chunk_size,
                pads_to_zero=index == 0,
                time_window_handler=time_window_handler,
                step_stride=step_stride,
            )
            for index, start in enumerate(range(0, len(output_idxs), chunk_size))
        ]

    @classmethod
    def whole(
        cls,
        output_idxs: list[int],
        chunk_size: int,
        time_window_handler: TimeWindowHandler | None = None,
        step_stride: int = 1,
    ) -> ChunkInfo:
        """The entire rollout as a single tile.

        Used where a rollout is run or reassembled in one piece rather than tile by tile.
        `chunk_size` stays the configured width, so this tile may be wider than it -- it
        describes the whole rollout, not one chunk of it. It pads to zero for the same
        reason the first tile does, and is otherwise indistinguishable from the first tile
        of an unchunked rollout, which it is.
        """

        return cls(
            index=0,
            steps=tuple(output_idxs),
            forecast_offset=output_idxs[0],
            chunk_size=chunk_size,
            pads_to_zero=True,
            time_window_handler=time_window_handler,
            step_stride=step_stride,
        )

    @property
    def forecast_steps(self) -> tuple[int, ...]:
        """Global steps this tile is indexed by, including the padded leading slots."""
        if self.pads_to_zero:
            return tuple(range(0, self.steps[-1] + 1))
        return self.steps

    @property
    def predicted_steps(self) -> tuple[int, ...]:
        """Steps of this tile that carry a target, i.e. that a producer emits a window for."""
        return tuple(step for step in self.steps if step >= self.forecast_offset)

    @property
    def is_full(self) -> bool:
        """Whether this tile carries a full chunk_size of steps, i.e. is not a short tail."""
        return len(self.steps) == self.chunk_size

    def chunk_idx(self, fstep: int) -> int:
        """Index of global forecast step fstep into tile-local data, e.g. predictions."""
        return fstep - self.forecast_steps[0]

    def window_idx(self, base_idx: int, fstep: int) -> int:
        """Window index on this tile's timeline that global forecast step fstep is valid for.

        `base_idx` is the trajectory's initialization window index in that same index space.
        """
        return base_idx + fstep * self.step_stride
