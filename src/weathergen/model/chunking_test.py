# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Contract of ChunkInfo: the facts a tile must carry rather than let a consumer derive."""

import numpy as np
import pytest

from weathergen.datasets.data_reader_base import TimeWindowHandler
from weathergen.model.chunking import ChunkInfo


def rollout(num_chunks: int, chunk_size: int, offset: int) -> list[int]:
    """The global forecast steps a batch carries, as ModelBatch builds them."""
    # batch.py: output_idxs = list(range(output_offset, output_offset + num_steps))
    return list(range(offset, offset + num_chunks * chunk_size))


@pytest.fixture
def handler() -> TimeWindowHandler:
    return TimeWindowHandler(
        np.datetime64("2023-01-01T00:00"),
        np.datetime64("2023-12-31T00:00"),
        np.timedelta64(6, "h"),
        np.timedelta64(6, "h"),
    )


# ------------------------------------------------------------------ tiling


def test_only_the_first_tile_pads_down_to_step_zero():
    tiles = ChunkInfo.tiles(rollout(3, 4, offset=1), 4)

    assert [t.index for t in tiles] == [0, 1, 2]
    assert [t.pads_to_zero for t in tiles] == [True, False, False]
    assert tiles[0].forecast_steps == (0, 1, 2, 3, 4)
    assert tiles[1].forecast_steps == (5, 6, 7, 8)


def test_the_index_is_carried_not_derived():
    """The derivation this type exists to remove returns -1 on the padded first tile."""
    tiles = ChunkInfo.tiles(rollout(3, 4, offset=1), 4)

    naive = [(t.forecast_steps[0] - t.forecast_offset) // t.chunk_size for t in tiles]
    assert naive == [-1, 1, 2], "the derivation is wrong on the first tile"
    assert [t.index for t in tiles] == [0, 1, 2], "the carried index is not"


def test_a_short_final_tile_keeps_the_configured_chunk_size():
    """len(steps) is not the cadence: dividing by it mis-indexes the tail."""
    tiles = ChunkInfo.tiles(rollout(1, 10, offset=1), 4)

    assert [len(t.steps) for t in tiles] == [4, 4, 2]
    assert [t.chunk_size for t in tiles] == [4, 4, 4]
    assert [t.is_full for t in tiles] == [True, True, False]
    assert (tiles[2].steps[0] - 1) // len(tiles[2].steps) == 4, "divides to the wrong tile"
    assert tiles[2].index == 2


def test_a_one_step_component_still_pads_its_first_tile():
    """The 24h ocean against a 24h chunk: chunk_size 1, first tile two slots long."""
    tiles = ChunkInfo.tiles(rollout(3, 1, offset=1), 1)

    assert [t.forecast_steps for t in tiles] == [(0, 1), (2,), (3,)]
    assert [t.index for t in tiles] == [0, 1, 2]


def test_offset_zero_pads_nothing_away():
    tiles = ChunkInfo.tiles(rollout(2, 4, offset=0), 4)

    assert tiles[0].forecast_steps == (0, 1, 2, 3)
    assert tiles[0].predicted_steps == (0, 1, 2, 3)


def test_predicted_steps_drops_the_leading_slots():
    tiles = ChunkInfo.tiles(rollout(2, 4, offset=1), 4)

    assert tiles[0].steps == (1, 2, 3, 4)
    assert tiles[0].predicted_steps == (1, 2, 3, 4)
    # the padded slot is in forecast_steps but is not a step anyone predicts
    assert 0 in tiles[0].forecast_steps
    assert 0 not in tiles[0].predicted_steps


# ------------------------------------------------------------------ indexing


def test_chunk_idx_round_trips_on_every_tile():
    for tile in ChunkInfo.tiles(rollout(3, 4, offset=1), 4):
        for step in tile.steps:
            assert tile.forecast_steps[tile.chunk_idx(step)] == step


def test_window_idx_advances_by_the_stride():
    tile = ChunkInfo.tiles(rollout(2, 4, offset=1), 4, step_stride=4)[0]

    assert tile.window_idx(100, 0) == 100
    assert tile.window_idx(100, 2) == 108


def test_the_timeline_travels_with_the_tile(handler):
    tile = ChunkInfo.tiles(rollout(2, 4, offset=1), 4, handler, 1)[1]

    assert tile.time_window_handler is handler
    assert tile.time_window_handler.window(tile.window_idx(0, 5)).start == np.datetime64(
        "2023-01-02T06:00"
    )


# ------------------------------------------------------------------ the whole rollout


def test_whole_spans_the_rollout_and_pads_like_a_first_tile():
    output_idxs = rollout(3, 4, offset=1)
    whole = ChunkInfo.whole(output_idxs, 4)

    assert whole.steps == tuple(output_idxs)
    assert whole.pads_to_zero
    assert whole.chunk_size == 4, "stays the configured width, not the rollout length"


def test_tiles_reassemble_into_exactly_the_whole_tile():
    """The invariant assemble_chunks asserts: per-tile slots must total the whole rollout's."""
    for offset in (0, 1):
        for num_chunks, chunk_size in ((3, 4), (1, 10), (3, 1)):
            output_idxs = rollout(num_chunks, chunk_size, offset)
            tiles = ChunkInfo.tiles(output_idxs, chunk_size)
            whole = ChunkInfo.whole(output_idxs, chunk_size)

            slots = sum(len(t.forecast_steps) for t in tiles)
            assert slots == len(whole.forecast_steps), (offset, num_chunks, chunk_size)


# ------------------------------------------------------------------ rejected input


@pytest.mark.parametrize(
    "kwargs",
    [
        {"steps": ()},
        {"steps": (1, 3, 4)},
        {"index": -1},
        {"chunk_size": 0},
        {"step_stride": 0},
        {"forecast_offset": -1},
    ],
)
def test_a_tile_that_cannot_be_placed_is_rejected(kwargs):
    valid = {
        "index": 0,
        "steps": (1, 2),
        "forecast_offset": 1,
        "chunk_size": 2,
    }
    with pytest.raises(ValueError):
        ChunkInfo(**{**valid, **kwargs})
