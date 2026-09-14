"""Contract of DataReaderCoupling: the subset of the reader interface ForcingInput uses."""

import dataclasses

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from weathergen.common.coupling import Coupler, Coupling, DataReaderCoupling
from weathergen.datasets.batch import ModelBatch, SampleMetaData
from weathergen.datasets.data_reader_base import (
    DataReaderTimestep,
    PassthroughReader,
    ReaderData,
    TimeWindowHandler,
    WrappedDataReader,
)
from weathergen.datasets.stream_data import StreamData
from weathergen.datasets.tokenizer_utils import TIMES_WIDTH, VERTEX_WIDTH
from weathergen.model.chunking import ChunkInfo
from weathergen.model.forcing import ForcingInput
from weathergen.model.model import ModelOutput

STREAM = "ERA5-Ocean"
N_POINTS = 7
SAMPLE_IDX = 100
FORECAST_OFFSET = 1
FSTEPS = [1, 2]

# The producer's geoinfos travel inside its tokenized target coords, one column here since
# FakeReader declares a single geoinfo channel. Normalized, as the tokenizer stored them.
GEOINFO_COL = 1 + TIMES_WIDTH
GEOINFOS = np.linspace(-1.0, 1.0, N_POINTS, dtype=np.float32)


def _target_tokens(geoinfos: NDArray[np.float32]) -> torch.Tensor:
    """One row per point in get_target_coords_local's layout, geoinfos in their column.

    Everything else is filled with a value that is wrong if it is ever read as a geoinfo,
    so a slice off by one column fails the assertions rather than passing by luck.
    """
    row_width = 1 + TIMES_WIDTH + 1 + VERTEX_WIDTH
    tokens = torch.full((len(geoinfos), row_width), -99.0, dtype=torch.float32)
    tokens[:, GEOINFO_COL] = torch.from_numpy(geoinfos)
    return tokens


class FakeReader(DataReaderTimestep):
    """Stand-in for the consumer's own reader for the coupled stream."""

    def __init__(self, twh: TimeWindowHandler) -> None:
        super().__init__(
            twh,
            {"stream_id": 0, "token_size": 4, "tokenize_spacetime": False},
            np.datetime64("2023-01-01T00:00"),
            np.datetime64("2023-12-31T00:00"),
            np.timedelta64(6, "h"),
        )
        # variable table: two data channels followed by one geoinfo channel
        self.source_channels = ["sst", "sea_ice"]
        self.source_idx = [0, 1]
        self.target_channels = ["sst", "sea_ice"]
        self.target_idx = [0, 1]
        self.geoinfo_channels = ["lsm"]
        self.geoinfo_idx = [2]
        self.target_channel_weights = [1.0, 1.0]
        self.mean = np.array([290.0, 0.3, 0.5], dtype=np.float32)
        self.stdev = np.array([10.0, 0.1, 0.25], dtype=np.float32)
        self.mean_geoinfo = np.array([0.5], dtype=np.float32)
        self.stdev_geoinfo = np.array([0.25], dtype=np.float32)

    def length(self) -> int:
        return 1000

    def _get(self, idx, channels_idx) -> ReaderData:
        """Ground truth for one window, valued so its origin is identifiable.

        Channel j carries `100 + j`, which is distinguishable from any prediction the chunk
        fixture emits, so a primed window cannot be confused with a served one.
        """
        coords = np.stack(
            [np.linspace(-80, 80, N_POINTS), np.linspace(-170, 170, N_POINTS)], axis=-1
        ).astype(np.float32)
        data = np.tile(
            np.asarray([100.0 + c for c in channels_idx], dtype=np.float32), (N_POINTS, 1)
        )
        geoinfos = np.zeros((N_POINTS, len(self.geoinfo_idx)), dtype=np.float32)
        times = np.array(["2023-01-01T00:00"] * N_POINTS, dtype="datetime64[ns]")
        return ReaderData(coords=coords, geoinfos=geoinfos, data=data, datetimes=times)


@pytest.fixture
def time_window_handler() -> TimeWindowHandler:
    return TimeWindowHandler(
        np.datetime64("2023-01-01T00:00"),
        np.datetime64("2023-12-31T00:00"),
        np.timedelta64(6, "h"),
        np.timedelta64(6, "h"),
    )


@pytest.fixture
def consumer(time_window_handler) -> FakeReader:
    return FakeReader(time_window_handler)


@pytest.fixture
def coords() -> NDArray[np.float32]:
    return np.stack(
        [np.linspace(-80, 80, N_POINTS), np.linspace(-170, 170, N_POINTS)], axis=-1
    ).astype(np.float32)


def _innermost(reader):
    """Walk a stack down to the reader at the bottom."""
    while isinstance(reader, WrappedDataReader):
        reader = reader._wrapped_reader
    return reader


class FakeStreamData:
    """The `streams_datasets` entry a Coupler reaches through to reach a producer's readers."""

    def __init__(self, reader: FakeReader) -> None:
        self.readers = [reader]


class FakeDataset:
    def __init__(self, streams: dict[str, FakeReader]) -> None:
        self.streams_datasets = {n: FakeStreamData(r) for n, r in streams.items()}


class FakeTrainer:
    def __init__(self, name: str, streams: dict[str, FakeReader]) -> None:
        self.name = name
        self.dataset = FakeDataset(streams)


@pytest.fixture
def chunk(coords) -> tuple[ModelOutput, ModelBatch]:
    """One rollout chunk: predictions plus the batch carrying the target geometry."""
    times = np.array(["2023-01-01T00:00"] * N_POINTS, dtype="datetime64[ns]")

    batch = ModelBatch([STREAM], 1, 1, FORECAST_OFFSET, FSTEPS[-1] + 1)
    source = StreamData(SAMPLE_IDX, 1, FSTEPS[-1] + 1, healpix_cells=48)
    target = StreamData(SAMPLE_IDX, 1, FSTEPS[-1] + 1, healpix_cells=48)
    for fstep in FSTEPS:
        # the two halves of the geometry live on different samples: add_target_values writes
        # the raw coords, times and idxs_inv, add_target_coords the tokenized target coords
        target.target_coords_raw[fstep] = torch.tensor(coords)
        target.target_times_raw[fstep] = times
        # a non-trivial permutation, so applying it is observable
        target.idxs_inv[fstep] = torch.arange(N_POINTS - 1, -1, -1)
        target.target_is_spoof[fstep] = False
        source.target_coords[fstep] = _target_tokens(GEOINFOS)
        source.target_is_spoof[fstep] = False

    batch.add_source_stream(0, 0, STREAM, source, SampleMetaData(params={}, mask=None))
    batch.add_target_stream(0, 0, STREAM, target, SampleMetaData(params={}, mask=None))

    tile = ChunkInfo.tiles(FSTEPS, len(FSTEPS))[0]
    output = ModelOutput(tile, batch.get_source_samples())
    for fstep in FSTEPS:
        # normalized prediction, constant per step so it is easy to check
        pred = torch.full((1, N_POINTS, 2), float(fstep), dtype=torch.float32)
        output.add_physical_prediction(output.chunk_idx(fstep), STREAM, [pred])

    return output, batch


def collect(reader: DataReaderCoupling, idx: int):
    """The call sequence ForcingInput._collect_forcing_data runs on a reader."""
    rdata = reader.get_source(np.int64(idx)).shuffle(None, False, -1)
    rdata = rdata.remove_nan_coords_and_geoinfos()
    rdata = dataclasses.replace(
        rdata,
        data=reader.normalize_source_channels(rdata.data),
        geoinfos=reader.normalize_geoinfos(rdata.geoinfos),
    )
    return rdata


def test_primes_from_the_producer_before_any_chunk_arrives(consumer, time_window_handler):
    """C2: the first rollout step precedes the producer's first chunk, so serve ground truth.

    From the *producer's* target data, not the consumer's source data: a primed window has to be
    the same quantity a prediction is, on the same grid and through the same channel map.
    """
    producer = _producer_with_wider_target(time_window_handler)

    reader = DataReaderCoupling(consumer, STREAM, producer=producer)
    rdata = reader.get_source(np.int64(SAMPLE_IDX))

    assert not rdata.is_empty(), "priming must not hand back a spoof"
    assert rdata.data.shape == (N_POINTS, len(consumer.source_idx))
    # the producer's targets are [10u, sea_ice, 2d, sst, msl]; the consumer sources [sst, sea_ice],
    # which are columns 3 and 1, carrying 103 and 101
    assert np.allclose(rdata.data[:, 0], 103.0)
    assert np.allclose(rdata.data[:, 1], 101.0)


def test_reads_empty_once_a_chunk_has_been_dispatched(consumer, chunk, time_window_handler):
    """Past the first chunk an unproduced window is a real gap, so it reads back empty."""
    # the producer matches the chunk fixture's two predicted channels
    producer = FakeReader(time_window_handler)
    reader = DataReaderCoupling(consumer, STREAM, producer=producer)
    reader.add_chunk(*chunk)

    rdata = reader.get_source(np.int64(SAMPLE_IDX + FSTEPS[-1] + 5))

    assert rdata.is_empty()
    assert rdata.data.shape == (0, len(consumer.source_idx))
    assert rdata.geoinfos.shape == (0, len(consumer.geoinfo_idx))


def test_prediction_is_served_at_its_valid_time(consumer, chunk, coords):
    reader = DataReaderCoupling(consumer, STREAM)
    reader.add_chunk(*chunk)

    for fstep in FSTEPS:
        rdata = reader.get_source(np.int64(SAMPLE_IDX + fstep))

        assert rdata.data.shape == (N_POINTS, 2)
        # handed out in physical space, so the consumer can normalize it as usual
        expected = np.float32(fstep) * consumer.stdev[:2] + consumer.mean[:2]
        assert np.allclose(rdata.data[0], expected)
        # idxs_inv was applied to coordinates and data alike
        assert np.allclose(rdata.coords[0], coords[-1])

    # steps the producer has not reached read back empty
    assert reader.get_source(np.int64(SAMPLE_IDX + FSTEPS[-1] + 1)).is_empty()


def test_normalization_round_trip(consumer, chunk):
    """What ForcingInput normalizes must come back to the prediction it started from."""
    reader = DataReaderCoupling(consumer, STREAM)
    reader.add_chunk(*chunk)

    for fstep in FSTEPS:
        rdata = collect(reader, SAMPLE_IDX + fstep)

        assert np.allclose(rdata.data, float(fstep))
        # geoinfos are the producer's own, recovered from its target tokens and denormalized
        # on the way out, so normalizing again returns the values the tokenizer stored --
        # reordered by idxs_inv alongside the data. Never zero: that was the defect this
        # replaces, where the consumer's climatological mean stood in for them.
        assert np.allclose(rdata.geoinfos[:, 0], GEOINFOS[::-1])


def test_stored_windows_survive_in_place_normalization(consumer, chunk):
    reader = DataReaderCoupling(consumer, STREAM)
    reader.add_chunk(*chunk)

    collect(reader, SAMPLE_IDX + FSTEPS[0])
    again = reader.get_source(np.int64(SAMPLE_IDX + FSTEPS[0]))

    expected = np.float32(FSTEPS[0]) * consumer.stdev[:2] + consumer.mean[:2]
    assert np.allclose(again.data[0], expected)


def test_spoofed_producer_steps_abort_the_coupling(consumer, chunk):
    """A spoof means the producer has no data there, which is not recoverable.

    `spoof` stands in for a window that came back empty, so during a rollout it says the
    trajectory was initialized outside the producer's dataset range -- a setup error, not a
    per-step condition to skip. Skipping it would force the consumer on climatological means
    with no signal that it happened.
    """
    output, batch = chunk
    batch.get_target_sample(0).streams_data[STREAM].target_is_spoof[FSTEPS[0]] = True

    reader = DataReaderCoupling(consumer, STREAM)
    with pytest.raises(ValueError, match="Cannot pair prediction with its target geometry"):
        reader.add_chunk(output, batch)


def test_eviction_bounds_the_window_store(consumer, chunk):
    reader = DataReaderCoupling(consumer, STREAM, max_pending_windows=1)
    reader.add_chunk(*chunk)

    # the newest window survives, the older one is dropped
    assert reader.get_source(np.int64(SAMPLE_IDX + FSTEPS[0])).is_empty()
    assert not reader.get_source(np.int64(SAMPLE_IDX + FSTEPS[-1])).is_empty()


def _producer_with_wider_target(time_window_handler) -> FakeReader:
    """A producer predicting five channels, of which the consumer sources two.

    The consumer's `sst` and `sea_ice` sit at positions 3 and 1, so taking the first two
    columns positionally yields `10u` and `sea_ice` instead.
    """
    producer = FakeReader(time_window_handler)
    producer.target_channels = ["10u", "sea_ice", "2d", "sst", "msl"]
    producer.target_idx = [0, 1, 2, 3, 4]
    producer.mean = np.zeros(5, dtype=np.float32)
    producer.stdev = np.ones(5, dtype=np.float32)
    return producer


def test_channels_are_selected_by_name_not_position(consumer, time_window_handler):
    """A producer's channel list agrees with its consumer's in neither order nor length.

    Selecting positionally is what delivered the atmosphere's `2d` (~273 K) to the ocean
    labelled `z_1000` (~10^3 m^2 s^-2): the ocean's source channels were read off the
    producer's first three columns.
    """
    producer = _producer_with_wider_target(time_window_handler)

    reader = DataReaderCoupling(consumer, STREAM, producer=producer)

    assert list(reader._pred_cols) == [3, 1], "sst is producer column 3, sea_ice column 1"


def test_named_channels_survive_the_round_trip(consumer, coords, time_window_handler):
    """End to end: the values the consumer reads back are the ones it named."""
    producer = _producer_with_wider_target(time_window_handler)
    times = np.array(["2023-01-01T00:00"] * N_POINTS, dtype="datetime64[ns]")

    batch = ModelBatch([STREAM], 1, 1, FORECAST_OFFSET, FSTEPS[-1] + 1)
    source = StreamData(SAMPLE_IDX, 1, FSTEPS[-1] + 1, healpix_cells=48)
    target = StreamData(SAMPLE_IDX, 1, FSTEPS[-1] + 1, healpix_cells=48)
    for fstep in FSTEPS:
        target.target_coords_raw[fstep] = torch.tensor(coords)
        target.target_times_raw[fstep] = times
        target.idxs_inv[fstep] = torch.arange(N_POINTS - 1, -1, -1)
        target.target_is_spoof[fstep] = False
        source.target_coords[fstep] = _target_tokens(GEOINFOS)
        source.target_is_spoof[fstep] = False
    batch.add_source_stream(0, 0, STREAM, source, SampleMetaData(params={}, mask=None))
    batch.add_target_stream(0, 0, STREAM, target, SampleMetaData(params={}, mask=None))

    tile = ChunkInfo.tiles(FSTEPS, len(FSTEPS))[0]
    output = ModelOutput(tile, batch.get_source_samples())
    for fstep in FSTEPS:
        # channel j carries the value j * 10, so which column was taken is visible
        pred = torch.arange(5, dtype=torch.float32).mul(10.0).expand(1, N_POINTS, 5).contiguous()
        output.add_physical_prediction(output.chunk_idx(fstep), STREAM, [pred])

    reader = DataReaderCoupling(consumer, STREAM, producer=producer)
    reader.add_chunk(output, batch)
    rdata = reader.get_source(np.int64(SAMPLE_IDX + FSTEPS[0]))

    # consumer order is [sst, sea_ice] -> producer columns [3, 1] -> values [30, 10].
    # Positional selection would have given [0, 10].
    assert np.allclose(rdata.data[:, 0], 30.0), "sst must come from the channel named sst"
    assert np.allclose(rdata.data[:, 1], 10.0)


def test_missing_producer_channel_is_reported(consumer, time_window_handler):
    producer = FakeReader(time_window_handler)
    producer.target_channels = ["sst"]
    producer.target_idx = [0]

    with pytest.raises(ValueError, match="sea_ice"):
        DataReaderCoupling(consumer, STREAM, producer=producer)


def test_coupler_resolves_the_channel_map_against_the_producer(consumer, time_window_handler):
    """F1: subscribe() must hand the reader the producing component's own reader.

    The reader has always resolved by name when given a producer; the defect was that
    `subscribe()` never passed one, so every coupling fell back to the branch that assumes
    producer and consumer share a channel list.
    """
    producer = _producer_with_wider_target(time_window_handler)
    coupler = Coupler(
        {"atmo": (FakeTrainer("atmo", {STREAM: producer}), None)},
        {"c": Coupling(name="c", producer="atmo", consumer="ocean", stream=STREAM)},
    )
    forcings = ForcingInput(
        "validation", time_window_handler, {STREAM: [consumer]}, tokenizer=None
    )

    reader = coupler.subscribe("ocean", forcings).forcing_streams[STREAM][0]

    assert list(reader._pred_cols) == [3, 1], "channel map must come from the producer"


def test_coupler_substitutes_only_coupled_streams(consumer, chunk, time_window_handler):
    # components first, couplings second: this driver owns the components too
    # the producer has to be resolvable: its reader is what the channel map is built against
    producer = FakeReader(time_window_handler)
    coupler = Coupler(
        {"atmo": (FakeTrainer("atmo", {STREAM: producer}), None)},
        {"c": Coupling(name="c", producer="atmo", consumer="ocean", stream=STREAM)},
    )
    forcings = ForcingInput(
        "validation",
        time_window_handler,
        {STREAM: [consumer], "era5": [consumer]},
        tokenizer=None,
    )

    subscribed = coupler.subscribe("ocean", forcings)
    streams = subscribed.forcing_streams

    # the coupling reader is innermost now, under whatever levelling wrapper the cadences call
    # for, so the substitution is not visible from the outermost reader's type -- which is the
    # same trap _announce_couplings fell into twice
    assert isinstance(streams[STREAM][0], PassthroughReader), "cadences match, so no levelling"
    assert isinstance(_innermost(streams[STREAM][0]), DataReaderCoupling)
    # a stream no coupling names is left on its own reader
    assert streams["era5"][0] is consumer

    coupler.dispatch_chunk("atmo", *chunk)
    assert not streams[STREAM][0].get_source(np.int64(SAMPLE_IDX + FSTEPS[0])).is_empty()

    # a producer nothing is subscribed to is a no-op, not an error
    coupler.dispatch_chunk("nobody", *chunk)
