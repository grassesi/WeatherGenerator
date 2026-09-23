"""Contract of DataReaderCoupling: the subset of the reader interface ForcingInput uses.

The consumer and the producer stand on *different* timelines here -- different origins and
different window steps -- because one shared handler fixture is what made the index space go
unrebased for as long as it did. With one handler the two index spaces coincide and every
arithmetic error is invisible; with two, the same integer denotes times a factor of four apart.

The fake readers encode each window's own valid time in the values they hand out, so a served
row names where it came from. That is what separates a window that was gathered from one that
was held and restamped: a held row carries the requested stamp but the covering window's value.
"""

import dataclasses

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from weathergen.common.coupling import Coupler, Coupling
from weathergen.datasets.averaging import AveragingReader
from weathergen.datasets.batch import ModelBatch, SampleMetaData
from weathergen.datasets.coupling_reader import DataReaderCoupling, ForcingProvenance
from weathergen.datasets.data_reader_base import (
    DataReaderTimestep,
    ReaderData,
    TimeWindowHandler,
    WrappedDataReader,
    rebase_innermost,
    shifted,
)
from weathergen.datasets.elevation import ElevatingReader
from weathergen.datasets.stream_data import StreamData
from weathergen.datasets.tokenizer_utils import TIMES_WIDTH, VERTEX_WIDTH
from weathergen.model.chunking import ChunkInfo
from weathergen.model.forcing import ForcingInput
from weathergen.model.model import ModelOutput

STREAM = "ERA5-Ocean"
N_POINTS = 7
FORECAST_OFFSET = 1
FSTEPS = [1, 2]

HOUR = np.timedelta64(1, "h")
H6 = np.timedelta64(6, "h")
H24 = np.timedelta64(24, "h")
ZERO = np.timedelta64(0, "h")

# Deliberately different origins: the consumer's index 0 and the producer's index 0 are six
# days apart, so an index carried across the two timelines unchanged lands nowhere near the
# time it names.
CONSUMER_START = np.datetime64("2023-01-01T00:00")
PRODUCER_START = np.datetime64("2022-12-25T00:00")
END = np.datetime64("2023-06-01T00:00")

# The producer's geoinfos travel inside its tokenized target coords, one column here since
# StampedReader declares a single geoinfo channel. Normalized, as the tokenizer stored them.
GEOINFO_COL = 1 + TIMES_WIDTH
GEOINFOS = np.linspace(-1.0, 1.0, N_POINTS, dtype=np.float32)


def value_of(when: np.datetime64, channel: int) -> float:
    """The value a window valid at `when` carries in `channel`.

    Encoding the time in the data is what lets an assertion name the window a row came from
    rather than merely counting rows.
    """
    return float((when - CONSUMER_START) / HOUR) * 10.0 + channel


def _target_tokens(geoinfos: NDArray[np.float32]) -> torch.Tensor:
    """One row per point in get_target_coords_local's layout, geoinfos in their column.

    Everything else is filled with a value that is wrong if it is ever read as a geoinfo,
    so a slice off by one column fails the assertions rather than passing by luck.
    """
    row_width = 1 + TIMES_WIDTH + 1 + VERTEX_WIDTH
    tokens = torch.full((len(geoinfos), row_width), -99.0, dtype=torch.float32)
    tokens[:, GEOINFO_COL] = torch.from_numpy(geoinfos)
    return tokens


def handler(start: np.datetime64, step: np.timedelta64) -> TimeWindowHandler:
    return TimeWindowHandler(start, END, step, step)


def idx_of(h: TimeWindowHandler, when: np.datetime64) -> np.int64:
    """The index a time sits at on a handler, so a test never hardcodes one."""
    return np.int64((when - h.t_start) // h.t_window_step)


class StampedReader(DataReaderTimestep):
    """A gridded reader whose every window says which window it is.

    One sample per window, at the window's start, valued by `value_of`. That is enough to
    exercise the window arithmetic without also modelling a stream whose period is finer than
    its window -- the reduction that case needs belongs to `AveragingReader`, above this.
    """

    def __init__(self, twh: TimeWindowHandler) -> None:
        super().__init__(
            twh,
            {"stream_id": 0, "token_size": 4, "tokenize_spacetime": False, "name": STREAM},
            twh.t_start,
            END,
            twh.t_window_step,
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
        start = self.time_window_handler.window(idx).start
        coords = np.stack(
            [np.linspace(-80, 80, N_POINTS), np.linspace(-170, 170, N_POINTS)], axis=-1
        ).astype(np.float32)
        data = np.tile(
            np.asarray([value_of(start, c) for c in channels_idx], dtype=np.float32),
            (N_POINTS, 1),
        )
        geoinfos = np.zeros((N_POINTS, len(self.geoinfo_idx)), dtype=np.float32)
        times = np.full(N_POINTS, start, dtype="datetime64[ns]")
        return ReaderData(coords=coords, geoinfos=geoinfos, data=data, datetimes=times)


@pytest.fixture
def consumer_handler() -> TimeWindowHandler:
    return handler(CONSUMER_START, H6)


@pytest.fixture
def producer_handler() -> TimeWindowHandler:
    """A coarser grid on a different origin, which is the whole point of the pair."""
    return handler(PRODUCER_START, H24)


@pytest.fixture
def consumer(consumer_handler) -> StampedReader:
    return StampedReader(consumer_handler)


@pytest.fixture
def producer(producer_handler) -> StampedReader:
    return StampedReader(producer_handler)


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

    def __init__(self, reader: StampedReader) -> None:
        self.readers = [reader]


class FakeDataset:
    def __init__(self, streams: dict[str, StampedReader]) -> None:
        self.streams_datasets = {n: FakeStreamData(r) for n, r in streams.items()}


class FakeTrainer:
    def __init__(self, name: str, streams: dict[str, StampedReader]) -> None:
        self.name = name
        self.dataset = FakeDataset(streams)


def build_chunk(
    producer_handler: TimeWindowHandler,
    init: np.datetime64,
    coords: NDArray[np.float32],
    fsteps: list[int] = FSTEPS,
    channels: int = 2,
    decoy_target_half: bool = False,
) -> tuple[ModelOutput, ModelBatch]:
    """One rollout chunk on the producer's timeline, valid from `init`.

    The tile carries the producer's handler, which is what stamps a prediction: a chunk that
    travels without one leaves the consumer guessing which timeline its steps are indices on.

    All geometry sits on the source half, as `add_target_coords` leaves it; under
    `inference_only`, which a coupled run always sets, there is no target half at all.
    `decoy_target_half` adds one anyway, with the pre-branch layout (raw coords, times and a
    reversing `idxs_inv`) but different points and times, so reading any of it is visible.
    """

    base = idx_of(producer_handler, init)
    step = producer_handler.t_window_step

    batch = ModelBatch([STREAM], 1, 1, FORECAST_OFFSET, fsteps[-1] + 1)
    source = StreamData(base, 1, fsteps[-1] + 1, healpix_cells=48)
    for fstep in fsteps:
        # add_target_coords writes the tokenized coords and, with the branch, the raw coords
        # and times, all in prediction row order
        source.target_coords[fstep] = _target_tokens(GEOINFOS)
        source.target_coords_raw[fstep] = coords.copy()
        # stamped at the time the step is valid for, which is what the reader serves it in
        source.target_times_raw[fstep] = np.full(
            N_POINTS, init + fstep * step, dtype="datetime64[ns]"
        )
        source.target_is_spoof[fstep] = False
    batch.add_source_stream(0, 0, STREAM, source, SampleMetaData(params={}, mask=None))

    if decoy_target_half:
        target = StreamData(base, 1, fsteps[-1] + 1, healpix_cells=48)
        for fstep in fsteps:
            target.target_coords_raw[fstep] = torch.tensor(coords + 1.0)
            target.target_times_raw[fstep] = np.full(
                N_POINTS, init - H24 * 100, dtype="datetime64[ns]"
            )
            target.idxs_inv[fstep] = torch.arange(N_POINTS - 1, -1, -1)
            target.target_is_spoof[fstep] = False
        batch.add_target_stream(0, 0, STREAM, target, SampleMetaData(params={}, mask=None))

    tile = ChunkInfo.tiles(fsteps, len(fsteps), producer_handler, 1)[0]
    output = ModelOutput(tile, batch.get_source_samples())
    for fstep in fsteps:
        # normalized prediction, constant per step so it is easy to check
        pred = torch.full((1, N_POINTS, channels), float(fstep), dtype=torch.float32)
        output.add_physical_prediction(output.chunk_idx(fstep), STREAM, [pred])

    return output, batch


@pytest.fixture
def chunk(producer_handler, coords) -> tuple[ModelOutput, ModelBatch]:
    return build_chunk(producer_handler, np.datetime64("2023-01-02T00:00"), coords)


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


def served_times(rdata: ReaderData) -> list[np.datetime64]:
    """Distinct valid times in a served window, ascending."""
    return sorted({np.datetime64(t, "m") for t in rdata.datetimes})


def served_origins(rdata: ReaderData) -> list[np.datetime64]:
    """The windows the rows actually came from, read back out of their values."""
    hours = sorted({float(v) / 10.0 for v in rdata.data[:, 0]})
    return [CONSUMER_START + np.timedelta64(int(round(h)), "h") for h in hours]


# --------------------------------------------------------------------------- the window matrix

# consumer step, producer step, lag, the request's valid time, and the source windows it must
# resolve to. Derived by hand from the window geometry, not from the code under test.
WINDOW_CASES = [
    # equal cadences: one source window per request, moved by the lag
    (H6, H6, ZERO, "2023-01-03T00:00", ["2023-01-03T00:00"], False),
    (H6, H6, H6, "2023-01-03T00:00", ["2023-01-02T18:00"], False),
    # half a step: the request straddles two source windows and only one starts inside it
    (H6, H6, np.timedelta64(3, "h"), "2023-01-03T00:00", ["2023-01-03T00:00"], False),
    # slow consumer, fast producer: four source windows in one request. This is the case a
    # single exact-valid-time lookup served one of, and the consumer then averaged one row.
    (
        H24,
        H6,
        H24,
        "2023-01-04T00:00",
        [
            "2023-01-03T00:00",
            "2023-01-03T06:00",
            "2023-01-03T12:00",
            "2023-01-03T18:00",
        ],
        False,
    ),
    # fast consumer, slow producer: nothing starts inside the request, so the covering window
    # is held and restamped onto it
    (H6, H24, H6, "2023-01-03T00:00", ["2023-01-02T00:00"], True),
    (H24, H24, H24, "2023-01-04T00:00", ["2023-01-03T00:00"], False),
]


@pytest.mark.parametrize(
    ("consumer_step", "producer_step", "lag", "predicts", "expected", "held"), WINDOW_CASES
)
def test_a_request_resolves_to_the_source_windows_inside_it(
    consumer_step, producer_step, lag, predicts, expected, held
):
    """L6: the coupled stack returns what the consumer's own disk reader would have returned.

    A request is resolved on the consumer's timeline shifted by the lag, and every source
    window starting inside it is gathered. Only when the source is coarser than the request
    window -- nothing starts inside it -- is the covering window held and restamped.
    """

    c_handler = handler(CONSUMER_START, consumer_step)
    p_handler = handler(PRODUCER_START, producer_step)
    consumer, producer = StampedReader(c_handler), StampedReader(p_handler)

    reader = DataReaderCoupling(
        consumer,
        STREAM,
        producer=producer,
        request_handler=shifted(c_handler, lag),
        # everything is an initial condition, so this test is about the geometry alone
        init_time=np.datetime64("2024-01-01T00:00"),
    )
    rdata = reader.get_source(idx_of(c_handler, np.datetime64(predicts)))

    wanted = [np.datetime64(t, "m") for t in expected]
    assert served_origins(rdata) == wanted, "the rows must come from these source windows"

    if held:
        # the value still names the covering window; the stamp names the window served
        request_start = np.datetime64(predicts) - lag
        assert served_times(rdata) == [np.datetime64(request_start, "m")]
        assert reader.provenance.held == 1
    else:
        assert served_times(rdata) == wanted, "a gathered row keeps its own timestamp (L5)"
        assert reader.provenance.held == 0

    assert rdata.data.shape == (N_POINTS * len(wanted), len(consumer.source_idx))


@pytest.mark.parametrize("lag", [ZERO, H6, H24])
def test_an_unforced_stream_reads_the_same_window_through_the_same_arithmetic(lag):
    """L7: the same lag and the same window must hold when the rows come from disk.

    This is the row that pins train/inference consistency. An uncoupled run reads exactly what
    it reads today, but through the reader a coupled run reads through, so the two cannot drift
    apart by one path changing and the other not. Whole multiples of the window step only --
    the fractional case is the next test, where the two paths part on purpose.
    """

    c_handler = handler(CONSUMER_START, H6)
    consumer = StampedReader(c_handler)
    producer = StampedReader(c_handler)
    request_handler = shifted(c_handler, lag)
    idx = idx_of(c_handler, np.datetime64("2023-01-03T00:00"))

    from_disk = DataReaderCoupling(
        consumer, STREAM, request_handler=request_handler, is_forced=False
    ).get_source(idx)
    primed = DataReaderCoupling(
        consumer,
        STREAM,
        producer=producer,
        request_handler=request_handler,
        is_forced=True,
        init_time=np.datetime64("2024-01-01T00:00"),
    ).get_source(idx)

    assert served_times(from_disk) == served_times(primed)
    assert np.allclose(from_disk.data, primed.data)


class SampledReader(StampedReader):
    """A stream sampled every `period` inside a longer window, as a real 6 h ERA5 read into a
    24 h ocean window is: the window returns one row per sample it contains, not one per window.
    """

    def __init__(self, twh: TimeWindowHandler, period: np.timedelta64) -> None:
        super().__init__(twh)
        self.period = period

    def _get(self, idx, channels_idx) -> ReaderData:
        didx, _ = self._get_dataset_idxs(idx)
        times = np.repeat(self.data_start_time + didx * self.period, N_POINTS)
        one = super()._get(idx, channels_idx)  # coords and geoinfos of one sample
        return ReaderData(
            coords=np.tile(one.coords, (len(didx), 1)),
            geoinfos=np.tile(one.geoinfos, (len(didx), 1)),
            data=np.array([[value_of(t, c) for c in channels_idx] for t in times], np.float32),
            datetimes=times.astype("datetime64[ns]"),
        )


@pytest.mark.parametrize("lag_hours", [0, 6, 18, 24])
def test_training_and_inference_read_the_same_samples_off_the_window_grid(lag_hours):
    """L7 at the DLESyM geometry: a 24 h ocean window over a 6 h atmosphere, lagged 18 h.

    18 h is a multiple of the atmosphere's 6 h sampling period but not of the ocean's 24 h
    window step. Unforced, the rows are read on the stream's sampling grid, so training gathers
    exactly the samples a coupled run gathers from its producer. Resolving it on the window
    grid instead trained on the atmosphere up to T - 6 h while inference served it up to T --
    losing the one sample, A(T), that DLESyM exists to use.
    """

    ocean = handler(CONSUMER_START, H24)
    lag = np.timedelta64(lag_hours, "h")
    predicts = np.datetime64("2023-01-06T00:00")
    idx = idx_of(ocean, predicts)

    training = DataReaderCoupling(
        SampledReader(ocean, H6), STREAM, request_handler=shifted(ocean, lag), is_forced=False
    )
    inference = DataReaderCoupling(
        SampledReader(ocean, H6),
        STREAM,
        producer=StampedReader(handler(PRODUCER_START, H6)),
        request_handler=shifted(ocean, lag),
        init_time=np.datetime64("2024-01-01T00:00"),  # every window primed from ground truth
    )

    trained, served = training.get_source(idx), inference.get_source(idx)
    start = predicts - lag
    expected = [np.datetime64(start + k * H6, "m") for k in range(4)]

    assert served_origins(trained) == expected, "training reads the samples inside the window"
    assert served_origins(served) == expected, "inference gathers the same samples"
    assert np.array_equal(trained.data, served.data)
    assert served_times(trained) == served_times(served)
    if lag_hours == 18:
        assert expected[-1] == predicts, "DLESyM: the last sample is the atmosphere at T"


def test_an_overlapping_producer_grid_is_refused(consumer, consumer_handler):
    """Gathering concatenates, so overlapping source windows would count shared points twice."""

    overlapping = StampedReader(handler(PRODUCER_START, H6))
    overlapping.time_window_handler = TimeWindowHandler(PRODUCER_START, END, H24, H6)

    with pytest.raises(ValueError, match="consecutive emissions"):
        DataReaderCoupling(consumer, STREAM, producer=overlapping)


def test_the_lag_is_what_moves_the_window(consumer, consumer_handler, producer):
    """The index is not what carries the lag; the handler is.

    `ForcedModel` passes `step * step_stride` and nothing else, so a stream forced at a
    different lag differs only in the handler it was built with.
    """

    idx = idx_of(consumer_handler, np.datetime64("2023-01-03T00:00"))
    seen = {}
    for lag in (ZERO, H6, H24):
        reader = DataReaderCoupling(
            consumer,
            STREAM,
            producer=StampedReader(consumer_handler),
            request_handler=shifted(consumer_handler, lag),
            init_time=np.datetime64("2024-01-01T00:00"),
        )
        seen[lag] = served_origins(reader.get_source(idx))

    assert seen[ZERO] == [np.datetime64("2023-01-03T00:00")]
    assert seen[H6] == [np.datetime64("2023-01-02T18:00")]
    assert seen[H24] == [np.datetime64("2023-01-02T00:00")]


# --------------------------------------------------------------------------- where rows come from


def test_a_window_at_the_init_time_is_an_initial_condition(consumer, consumer_handler, producer):
    """G1: it comes from the producer's own data however far into the rollout it is asked for.

    Not "before anything was dispatched": with a lag, a request made deep into a rollout still
    reaches back across the init window, and the gate this replaces was already false by then.
    """

    init = np.datetime64("2023-01-02T00:00")
    reader = DataReaderCoupling(
        consumer,
        STREAM,
        producer=producer,
        request_handler=shifted(consumer_handler, H24),
        init_time=init,
    )
    # dispatch a chunk first, so the old "nothing dispatched yet" gate would be shut
    reader.add_chunk(*build_chunk(producer.time_window_handler, init, _coords()))

    rdata = reader.get_source(idx_of(consumer_handler, np.datetime64("2023-01-03T00:00")))

    assert served_origins(rdata) == [init], "the init window must come from ground truth"
    assert reader.provenance.primed == 1
    assert reader.provenance.predicted == 0


def test_a_window_past_the_init_time_is_a_prediction(consumer, consumer_handler, producer):
    """G2: served with boundary conditions from the producer's chunk."""

    init = np.datetime64("2023-01-02T00:00")
    reader = DataReaderCoupling(
        consumer,
        STREAM,
        producer=producer,
        request_handler=shifted(consumer_handler, H24),
        init_time=init,
    )
    reader.add_chunk(*build_chunk(producer.time_window_handler, init, _coords()))

    # forecast step 1 of the chunk is valid a producer window (24 h) past the init
    rdata = reader.get_source(idx_of(consumer_handler, np.datetime64("2023-01-04T00:00")))

    assert not rdata.is_empty()
    expected = np.float32(1.0) * consumer.stdev[:2] + consumer.mean[:2]
    assert np.allclose(rdata.data[0], expected), "physical space, as read from disk"
    assert reader.provenance.predicted == 1
    assert reader.provenance.primed == 0


def test_an_unproduced_window_past_the_init_time_is_unresolved(
    consumer, consumer_handler, producer
):
    """Past the init window an unemitted window is a real gap, counted as one."""

    init = np.datetime64("2023-01-02T00:00")
    reader = DataReaderCoupling(
        consumer,
        STREAM,
        producer=producer,
        request_handler=shifted(consumer_handler, H24),
        init_time=init,
    )
    reader.add_chunk(*build_chunk(producer.time_window_handler, init, _coords()))

    # a week past anything the chunk emitted
    rdata = reader.get_source(idx_of(consumer_handler, np.datetime64("2023-01-11T00:00")))

    assert rdata.is_empty()
    assert rdata.data.shape == (0, len(consumer.source_idx))
    assert reader.provenance.unresolved == 1


def test_provenance_counts_every_source_window_a_request_touched(producer_handler):
    """A request on a wide window resolves to several source windows, and all of them count.

    Counting requests instead would report the four-into-one case as one served window, which
    is exactly the number that stayed right while the exchange was wrong.
    """

    c_handler = handler(CONSUMER_START, H24)
    reader = DataReaderCoupling(
        StampedReader(c_handler),
        STREAM,
        producer=StampedReader(producer_handler),
        request_handler=shifted(c_handler, H24),
        init_time=np.datetime64("2024-01-01T00:00"),
    )
    # gathers the four 6 h windows of 2023-01-03 -- except the producer here is 24 h, so use
    # a 6 h one to make the gather real
    reader = DataReaderCoupling(
        StampedReader(c_handler),
        STREAM,
        producer=StampedReader(handler(PRODUCER_START, H6)),
        request_handler=shifted(c_handler, H24),
        init_time=np.datetime64("2024-01-01T00:00"),
    )
    reader.get_source(idx_of(c_handler, np.datetime64("2023-01-04T00:00")))

    assert reader.provenance.requests == 1
    assert reader.provenance.primed == 4
    assert reader.provenance.resolved == 4


def test_provenance_is_shared_across_the_readers_of_successive_trajectories(
    consumer, consumer_handler, producer
):
    """Readers are rebuilt per batch; the tally has to span the run, not one trajectory."""

    tally = ForcingProvenance(stream=STREAM)
    for _ in range(3):
        DataReaderCoupling(
            consumer,
            STREAM,
            producer=producer,
            request_handler=shifted(consumer_handler, H6),
            init_time=np.datetime64("2024-01-01T00:00"),
            provenance=tally,
        ).get_source(idx_of(consumer_handler, np.datetime64("2023-01-03T00:00")))

    assert tally.requests == 3
    assert tally.primed == 3


# --------------------------------------------------------------------------- the chunk store


def test_the_store_keeps_the_current_chunk_and_the_one_before_it(
    consumer, consumer_handler, producer
):
    """C4: a request's left edge is always served by the preceding chunk, never the current one.

    Two is therefore the bound, and anything older is genuinely past.
    """

    init = np.datetime64("2023-01-02T00:00")
    p_handler = producer.time_window_handler
    reader = DataReaderCoupling(
        consumer,
        STREAM,
        producer=producer,
        request_handler=shifted(consumer_handler, H24),
        init_time=init,
    )
    # three successive chunks of two steps each, 48 h apart on the producer's 24 h grid
    for i in range(3):
        reader.add_chunk(*build_chunk(p_handler, init + np.timedelta64(48 * i, "h"), _coords()))

    def served(when: str) -> ReaderData:
        return reader.get_source(idx_of(consumer_handler, np.datetime64(when) + H24))

    # chunk i emits at init + 48 i + 24 h and + 48 h, so the three chunks cover
    # 01-03/01-04, 01-05/01-06 and 01-07/01-08. The newest two survive.
    assert not served("2023-01-07T00:00").is_empty()
    assert not served("2023-01-05T00:00").is_empty()
    # the first chunk has been dropped, so its windows are past rather than pending
    assert served("2023-01-04T00:00").is_empty()
    assert served("2023-01-03T00:00").is_empty()


def test_an_incompletely_emitted_chunk_is_refused(consumer, consumer_handler, producer, coords):
    """C5/G4: a chunk is published whole or not at all.

    The expected count is `len(tile.predicted_steps)` carried on the chunk, not a length read
    off a step list whose first tile is padded and whose last may be short.
    """

    init = np.datetime64("2023-01-02T00:00")
    output, batch = build_chunk(producer.time_window_handler, init, coords)
    # drop one step's prediction, as a mid-emission chunk would be missing it
    output.physical[output.chunk_idx(FSTEPS[-1])].pop(STREAM)

    reader = DataReaderCoupling(
        consumer,
        STREAM,
        producer=producer,
        request_handler=shifted(consumer_handler, H24),
        init_time=init,
    )
    with pytest.raises(AssertionError, match="emitted 1 windows"):
        reader.add_chunk(output, batch)


def test_the_reader_has_no_reset(consumer):
    """F11: `_resubscribe` rebuilds the readers, so a second mechanism cannot drift from it."""
    assert not hasattr(DataReaderCoupling(consumer, STREAM), "reset")


# --------------------------------------------------------------------------- values and channels


def test_prediction_is_served_at_its_valid_time(consumer, consumer_handler, producer, coords):
    init = np.datetime64("2023-01-02T00:00")
    reader = DataReaderCoupling(
        consumer,
        STREAM,
        producer=producer,
        request_handler=shifted(consumer_handler, H24),
        init_time=init,
    )
    reader.add_chunk(*build_chunk(producer.time_window_handler, init, coords))

    for fstep in FSTEPS:
        valid = init + fstep * H24
        rdata = reader.get_source(idx_of(consumer_handler, valid + H24))

        assert rdata.data.shape == (N_POINTS, 2)
        # handed out in physical space, so the consumer can normalize it as usual
        expected = np.float32(fstep) * consumer.stdev[:2] + consumer.mean[:2]
        assert np.allclose(rdata.data[0], expected)
        # served in prediction row order; nothing reorders it
        assert np.allclose(rdata.coords, coords)


def test_normalization_round_trip(consumer, consumer_handler, producer, coords):
    """What ForcingInput normalizes must come back to the prediction it started from."""
    init = np.datetime64("2023-01-02T00:00")
    reader = DataReaderCoupling(
        consumer,
        STREAM,
        producer=producer,
        request_handler=shifted(consumer_handler, H24),
        init_time=init,
    )
    reader.add_chunk(*build_chunk(producer.time_window_handler, init, coords))

    for fstep in FSTEPS:
        rdata = collect(reader, idx_of(consumer_handler, init + (fstep + 1) * H24))

        assert np.allclose(rdata.data, float(fstep))
        # geoinfos are the producer's own, recovered from its target tokens and denormalized
        # on the way out, so normalizing again returns the values the tokenizer stored. Never
        # zero: that was the defect this replaces, where the consumer's climatological mean
        # stood in for them.
        assert np.allclose(rdata.geoinfos[:, 0], GEOINFOS)


def test_stored_windows_survive_in_place_normalization(
    consumer, consumer_handler, producer, coords
):
    init = np.datetime64("2023-01-02T00:00")
    reader = DataReaderCoupling(
        consumer,
        STREAM,
        producer=producer,
        request_handler=shifted(consumer_handler, H24),
        init_time=init,
    )
    reader.add_chunk(*build_chunk(producer.time_window_handler, init, coords))

    idx = idx_of(consumer_handler, init + 2 * H24)
    collect(reader, idx)
    again = reader.get_source(idx)

    expected = np.float32(FSTEPS[0]) * consumer.stdev[:2] + consumer.mean[:2]
    assert np.allclose(again.data[0], expected)


def test_spoofed_producer_steps_abort_the_coupling(consumer, producer, coords):
    """A spoof means the producer has no data there, which is not recoverable.

    `spoof` stands in for a window that came back empty, so during a rollout it says the
    trajectory was initialized outside the producer's dataset range -- a setup error, not a
    per-step condition to skip. Skipping it would force the consumer on climatological means
    with no signal that it happened.
    """
    output, batch = build_chunk(
        producer.time_window_handler, np.datetime64("2023-01-02T00:00"), coords
    )
    batch.get_source_sample(0).streams_data[STREAM].target_is_spoof[FSTEPS[0]] = True

    reader = DataReaderCoupling(consumer, STREAM, producer=producer)
    with pytest.raises(ValueError, match="Cannot pair prediction with its target geometry"):
        reader.add_chunk(output, batch)


def test_a_chunk_without_a_timeline_is_refused(consumer, producer, coords):
    """A tile without its handler cannot be placed, and there is no stride to guess with.

    The constructor stride this used to fall back to was never set from the producer, so a
    producer stepping more than one window per forecast step would have been stamped on the
    wrong windows -- and the completeness assert, which counted the windows against
    themselves in that branch, would have passed.
    """
    output, batch = build_chunk(
        producer.time_window_handler, np.datetime64("2023-01-02T00:00"), coords
    )
    output.chunk = dataclasses.replace(output.chunk, time_window_handler=None)

    reader = DataReaderCoupling(consumer, STREAM, producer=producer)
    with pytest.raises(ValueError, match="carries no ChunkInfo with a time_window_handler"):
        reader.add_chunk(output, batch)


def test_geometry_is_read_off_the_source_sample(consumer, consumer_handler, producer, coords):
    """Coords and times come from the source half, never from a target half that is present.

    The decoy target half carries other points, a stamp 100 days early and a reversing
    idxs_inv, so reading any of it, or reordering by it, changes what is served.
    """
    init = np.datetime64("2023-01-02T00:00")
    reader = DataReaderCoupling(
        consumer,
        STREAM,
        producer=producer,
        request_handler=shifted(consumer_handler, H24),
        init_time=init,
    )
    reader.add_chunk(
        *build_chunk(producer.time_window_handler, init, coords, decoy_target_half=True)
    )

    for fstep in FSTEPS:
        valid = init + fstep * H24
        rdata = reader.get_source(idx_of(consumer_handler, valid + H24))

        assert np.allclose(rdata.coords, coords)
        assert served_times(rdata) == [np.datetime64(valid, "m")]
        assert np.allclose(rdata.geoinfos[:, 0], GEOINFOS * 0.25 + 0.5)


def test_a_batch_without_a_target_half_is_lowered(consumer, consumer_handler, producer, coords):
    """Under inference_only the target sample holds nothing for the stream."""
    init = np.datetime64("2023-01-02T00:00")
    output, batch = build_chunk(producer.time_window_handler, init, coords)
    assert batch.get_target_sample(0).streams_data[STREAM] is None

    reader = DataReaderCoupling(
        consumer,
        STREAM,
        producer=producer,
        request_handler=shifted(consumer_handler, H24),
        init_time=init,
    )
    reader.add_chunk(output, batch)

    rdata = reader.get_source(idx_of(consumer_handler, init + FSTEPS[0] * H24 + H24))
    assert rdata.data.shape == (N_POINTS, 2)


def test_a_masked_prediction_reaches_the_consumer_as_nan(
    consumer, consumer_handler, producer, coords
):
    """A NaN the producer wrote (SST over land) must survive denormalization and the store.

    That NaN is what `_prime` hands over for chunk 0 from disk, and what the consumer's
    tokenizer turns into the mask value it trained on. Filled in anywhere on the way, the
    consumer would be forced by a land value from chunk 1 on.
    """
    init = np.datetime64("2023-01-02T00:00")
    output, batch = build_chunk(producer.time_window_handler, init, coords)
    land = [1, 4]
    for fstep in FSTEPS:
        pred = torch.full((1, N_POINTS, 2), float(fstep), dtype=torch.float32)
        pred[:, land, 0] = float("nan")
        output.physical[output.chunk_idx(fstep)][STREAM] = [pred]

    reader = DataReaderCoupling(
        consumer,
        STREAM,
        producer=producer,
        request_handler=shifted(consumer_handler, H24),
        init_time=init,
    )
    reader.add_chunk(output, batch)

    for fstep in FSTEPS:
        rdata = reader.get_source(idx_of(consumer_handler, init + fstep * H24 + H24))
        nan = np.isnan(rdata.data)
        assert nan[land, 0].all()
        # only the masked points of the masked channel, nothing else
        assert nan.sum() == len(land)
        assert np.allclose(rdata.data[~nan[:, 0], 0], fstep * consumer.stdev[0] + consumer.mean[0])


def _producer_with_wider_target(twh) -> StampedReader:
    """A producer predicting five channels, of which the consumer sources two.

    The consumer's `sst` and `sea_ice` sit at positions 3 and 1, so taking the first two
    columns positionally yields `10u` and `sea_ice` instead.
    """
    producer = StampedReader(twh)
    producer.target_channels = ["10u", "sea_ice", "2d", "sst", "msl"]
    producer.target_idx = [0, 1, 2, 3, 4]
    producer.mean = np.zeros(5, dtype=np.float32)
    producer.stdev = np.ones(5, dtype=np.float32)
    return producer


def _coords() -> NDArray[np.float32]:
    return np.stack(
        [np.linspace(-80, 80, N_POINTS), np.linspace(-170, 170, N_POINTS)], axis=-1
    ).astype(np.float32)


def test_channels_are_selected_by_name_not_position(consumer, producer_handler):
    """A producer's channel list agrees with its consumer's in neither order nor length.

    Selecting positionally is what delivered the atmosphere's `2d` (~273 K) to the ocean
    labelled `z_1000` (~10^3 m^2 s^-2): the ocean's source channels were read off the
    producer's first three columns.
    """
    producer = _producer_with_wider_target(producer_handler)

    reader = DataReaderCoupling(consumer, STREAM, producer=producer)

    assert list(reader._pred_cols) == [3, 1], "sst is producer column 3, sea_ice column 1"


def test_named_channels_survive_the_round_trip(
    consumer, consumer_handler, producer_handler, coords
):
    """End to end: the values the consumer reads back are the ones it named."""
    producer = _producer_with_wider_target(producer_handler)
    init = np.datetime64("2023-01-02T00:00")
    output, batch = build_chunk(producer_handler, init, coords, channels=5)
    for fstep in FSTEPS:
        # channel j carries the value j * 10, so which column was taken is visible
        pred = torch.arange(5, dtype=torch.float32).mul(10.0).expand(1, N_POINTS, 5).contiguous()
        output.physical[output.chunk_idx(fstep)][STREAM] = [pred]

    reader = DataReaderCoupling(
        consumer,
        STREAM,
        producer=producer,
        request_handler=shifted(consumer_handler, H24),
        init_time=init,
    )
    reader.add_chunk(output, batch)
    rdata = reader.get_source(idx_of(consumer_handler, init + 2 * H24))

    # consumer order is [sst, sea_ice] -> producer columns [3, 1] -> values [30, 10].
    # Positional selection would have given [0, 10].
    assert np.allclose(rdata.data[:, 0], 30.0), "sst must come from the channel named sst"
    assert np.allclose(rdata.data[:, 1], 10.0)


def test_a_non_periodic_reader_is_refused(consumer, producer_handler):
    """Coupled streams are gridded and periodic; the assumption is asserted, not assumed."""

    producer = StampedReader(producer_handler)
    producer.period = None  # as an obs reader arrives: no sampling period at all

    with pytest.raises(ValueError, match="not periodic"):
        DataReaderCoupling(consumer, STREAM, producer=producer)


def test_missing_producer_channel_is_reported(consumer, producer_handler):
    producer = StampedReader(producer_handler)
    producer.target_channels = ["sst"]
    producer.target_idx = [0]

    with pytest.raises(ValueError, match="sea_ice"):
        DataReaderCoupling(consumer, STREAM, producer=producer)


# --------------------------------------------------------------------------- the Coupler's wiring


def _forcings(consumer_handler, streams) -> ForcingInput:
    return ForcingInput("validation", consumer_handler, streams, tokenizer=None)


def test_coupler_resolves_the_channel_map_against_the_producer(
    consumer, consumer_handler, producer_handler
):
    """F1: subscribe() must hand the reader the producing component's own reader.

    The reader has always resolved by name when given a producer; the defect was that
    `subscribe()` never passed one, so every coupling fell back to the branch that assumes
    producer and consumer share a channel list.
    """
    producer = _producer_with_wider_target(producer_handler)
    coupler = Coupler(
        {"atmo": (FakeTrainer("atmo", {STREAM: producer}), None)},
        {"c": Coupling(name="c", producer="atmo", consumer="ocean", stream=STREAM)},
    )
    forcings = _forcings(consumer_handler, {STREAM: [consumer]})

    reader = coupler.subscribe("ocean", forcings).forcing_streams[STREAM][0]

    assert list(reader._pred_cols) == [3, 1], "channel map must come from the producer"


def test_coupler_substitutes_only_coupled_streams(
    consumer, consumer_handler, producer_handler, coords
):
    # components first, couplings second: this driver owns the components too
    # the producer has to be resolvable: its reader is what the channel map is built against
    producer = StampedReader(producer_handler)
    coupler = Coupler(
        {"atmo": (FakeTrainer("atmo", {STREAM: producer}), None)},
        {"c": Coupling(name="c", producer="atmo", consumer="ocean", stream=STREAM)},
    )
    forcings = _forcings(consumer_handler, {STREAM: [consumer], "era5": [consumer]})

    subscribed = coupler.subscribe("ocean", forcings)
    streams = subscribed.forcing_streams

    # every forcing stream is read through a coupling reader; only a coupled one carries a
    # producer, which is the whole difference between the two paths
    assert isinstance(_innermost(streams[STREAM][0]), DataReaderCoupling)
    assert streams[STREAM][0].is_forced
    assert not _innermost(streams["era5"][0]).is_forced

    coupler.dispatch_chunk("atmo", *build_chunk(producer_handler, CONSUMER_START, coords))
    # the chunk's first step is valid one producer window (24 h) past the init; the request is
    # made one consumer window later still, because the default lag is one window
    served = idx_of(consumer_handler, CONSUMER_START + H24 + consumer_handler.t_window_step)
    assert not streams[STREAM][0].get_source(served).is_empty()

    # a producer nothing is subscribed to is a no-op, not an error
    coupler.dispatch_chunk("nobody", *build_chunk(producer_handler, CONSUMER_START, coords))


def test_an_uncoupled_forcing_stream_still_goes_through_the_lagged_reader(
    consumer, consumer_handler
):
    """L7: training builds the same reader, with `is_forced=False`."""

    forcings = _forcings(consumer_handler, {"era5": [consumer]})
    reader = forcings.forcing_streams["era5"][0]

    assert isinstance(reader, DataReaderCoupling)
    assert not reader.is_forced
    # default lag is one window, i.e. what the call site used to encode in its index
    assert forcings.lags["era5"] == np.timedelta64(consumer_handler.t_window_step, "ms")


def test_the_configured_lag_reaches_the_reader(consumer_handler):
    """D2: the lag is a property of the stream's own config, so it rides in the checkpoint."""

    consumer = StampedReader(consumer_handler)
    consumer.stream_info = dict(consumer.stream_info, forcing_lag="12:00:00")

    forcings = _forcings(consumer_handler, {"era5": [consumer]})

    assert forcings.lags["era5"] == np.timedelta64(12, "h")
    assert forcings.handler("era5").window(0).start == CONSUMER_START - np.timedelta64(12, "h")


def test_a_negative_lag_is_refused(consumer_handler):
    consumer = StampedReader(consumer_handler)
    consumer.stream_info = dict(consumer.stream_info, forcing_lag="-06:00:00")

    with pytest.raises(ValueError, match="negative forcing_lag"):
        _forcings(consumer_handler, {"era5": [consumer]})


def test_step_is_a_documented_alias_for_the_default(consumer_handler):
    consumer = StampedReader(consumer_handler)
    consumer.stream_info = dict(consumer.stream_info, forcing_lag="step")

    forcings = _forcings(consumer_handler, {"era5": [consumer]})

    assert forcings.lags["era5"] == np.timedelta64(consumer_handler.t_window_step, "ms")


# --------------------------------------------------------------- the published coupling schemes

# The deployed pairing, driven through the interleave the `Coupler` actually runs: a 6 h
# atmosphere against a 24 h ocean in 24 h chunks, the atmosphere stepped first. Chunk `i` emits
# the producer's global forecast steps `i*n+1 .. (i+1)*n`, so with `forecast_offset` 1 a chunk
# ends on its boundary rather than starting there. After chunk `i` the atmosphere has therefore
# reached exactly the ocean's own target time and no further, and that frontier -- not the
# arithmetic of the window -- is what bounds how fresh a forcing can be.
#
# These cases are the empirical answer `forcing_lag_design.md` E asked for, as a test rather
# than as prose: which published scheme a lag expresses is decided by the source windows the
# request resolves to, and a scheme whose windows do not all exist is not available at all.

SCHEME_CHUNK = H24
SCHEME_INIT = np.datetime64("2023-01-02T00:00")


def _replay(consumer_step, producer_step, lag, *, producer_first, num_chunks=2):
    """Drive one direction of the exchange through the real dispatch order.

    Returns one row per request: the time predicted, the source valid times served, and how
    many of that request's source windows were unresolved.
    """

    c_h = handler(SCHEME_INIT, consumer_step)
    p_h = handler(SCHEME_INIT, producer_step)
    per_p = int(SCHEME_CHUNK // producer_step)
    per_c = int(SCHEME_CHUNK // consumer_step)

    reader = DataReaderCoupling(
        StampedReader(c_h),
        STREAM,
        producer=StampedReader(p_h),
        request_handler=shifted(c_h, lag),
        init_time=SCHEME_INIT,
    )

    rows = []
    for i in range(num_chunks):
        fsteps = list(range(i * per_p + 1, (i + 1) * per_p + 1))

        def dispatch(fsteps=fsteps, p_h=p_h):
            reader.add_chunk(*build_chunk(p_h, SCHEME_INIT, _coords(), fsteps=fsteps))

        if producer_first:
            dispatch()

        for k in range(per_c):
            target = SCHEME_INIT + (i * per_c + k + 1) * consumer_step
            before = reader.provenance.unresolved
            rdata = reader.get_source(idx_of(c_h, target))
            rows.append(
                (
                    target,
                    [] if rdata.is_empty() else served_times(rdata),
                    reader.provenance.unresolved - before,
                )
            )

        if not producer_first:
            dispatch()

    return rows


@pytest.mark.parametrize(
    ("lag_hours", "unresolved_per_request"),
    [
        (0, 3),  # the fully concurrent window DLESyM names as an instant
        (6, 2),
        (12, 1),  # SamudrACE's half of the ocean's own step
        (18, 0),  # DLESyM, as fresh as a single pass admits
        (24, 0),  # the old default
    ],
)
def test_how_fresh_an_atmosphere_the_ocean_can_reach(lag_hours, unresolved_per_request):
    """Below 18 h the ocean reaches windows the atmosphere has not emitted yet.

    Each request spans four 6 h emissions, and every hour of lag below the bound costs one of
    them. They do not fail loudly: an unresolved window reads back empty and the forcing falls
    through to a climatological spoof, which is why this counts rather than merely running.
    """

    rows = _replay(H24, H6, np.timedelta64(lag_hours, "h"), producer_first=True)

    assert [unresolved for _, _, unresolved in rows] == [unresolved_per_request] * len(rows)


def test_dlesym_reaches_the_atmosphere_at_its_own_target_time():
    """What separates DLESyM's fresh atmosphere from the lag that preceded it.

    Both resolve completely, so no count distinguishes them; the newest source window each
    gathers does. At 18 h the request runs to `T + 6h`, so it takes in the atmospheric window
    starting at `T` -- the `A(T)` of the scheme's own notation. At 24 h it stops at `T`, and
    the newest it can take is one atmospheric step earlier.
    """

    fresh = _replay(H24, H6, np.timedelta64(18, "h"), producer_first=True)
    stale = _replay(H24, H6, H24, producer_first=True)

    for (target, fresh_times, _), (_, stale_times, _) in zip(fresh, stale, strict=True):
        assert max(fresh_times) == target, "DLESyM: the atmosphere at the ocean's own time"
        assert max(stale_times) == target - H6, "the old default: strictly before it"


@pytest.mark.parametrize(
    ("lag_hours", "unresolved"),
    [
        (3, 2),  # SamudrACE's half of the atmosphere's own step
        (6, 0),  # DLESyM's stale ocean, and the default
        (12, 0),
    ],
)
def test_the_atmospheres_ocean_forcing_cannot_be_half_a_step(lag_hours, unresolved):
    """The other half of why SamudrACE is not a lag this rollout can be configured into.

    The ocean is stepped after the atmosphere, so within a chunk the atmosphere sees only the
    ocean's previous one. At 3 h the last of each chunk's four requests reaches into the ocean
    window being computed beside it and resolves to nothing.
    """

    rows = _replay(H6, H24, np.timedelta64(lag_hours, "h"), producer_first=False)

    assert sum(row[2] for row in rows) == unresolved


def test_a_single_pass_interleave_costs_one_chunk_of_round_trip_lag():
    """Why no configuration expresses SamudrACE, stated as the invariant behind both halves.

    Each direction's smallest legal lag is `len_c - P + D`, and summed over the two directions
    the cadences cancel and one `chunk_length` of slack remains. DLESyM sits exactly on that
    bound; SamudrACE's 12 h and 3 h sum to less than it, so no pair of lags can express it
    without a second pass over each chunk to supply the half-step state.
    """

    ocean_min = H24 - H6 + ZERO  # atmosphere stepped first
    atmo_min = H6 - H24 + SCHEME_CHUNK  # ocean stepped second

    assert ocean_min + atmo_min == SCHEME_CHUNK
    assert np.timedelta64(18, "h") + H6 == SCHEME_CHUNK, "DLESyM is exactly on the bound"
    assert np.timedelta64(12, "h") + np.timedelta64(3, "h") < SCHEME_CHUNK, "SamudrACE is inside"


def test_the_reduced_forcing_is_stamped_at_the_window_start():
    """What the ocean's model actually receives: one row, and the time it claims to be from.

    The consumer's own `AveragingReader` sits above the coupling reader and collapses the
    gathered windows to one datapoint stamped at the window *start*
    (`averaging.py`, `datetimes.min()`). That stamp is what `encode_times_source` turns into
    the token's time features and what the cyclic geoinfos are recomputed at, so it is a
    separate fact from which windows were gathered -- and it moved when the averaging
    semantics were levelled with an un-averaged stream's.

    The distinction matters for reading the lag: at 18 h the *content* reaches `A(T)` while
    the *label* is `T - 18h`. A forcing labelled `T`, concurrent with the ocean's own target
    window, would need lag 0, which does not resolve.
    """

    consumer_handler = handler(SCHEME_INIT, H24)
    producer_handler = handler(SCHEME_INIT, H6)
    target = SCHEME_INIT + 4 * H24

    for lag, expected_stamp in ((np.timedelta64(18, "h"), target - np.timedelta64(18, "h")),
                                (H24, target - H24)):
        coupled = DataReaderCoupling(
            StampedReader(consumer_handler),
            STREAM,
            producer=StampedReader(producer_handler),
            request_handler=shifted(consumer_handler, lag),
            # far ahead, so every gathered window is primed from the producer's own truth
            init_time=np.datetime64("2030-01-01T00:00"),
        )
        averaged = AveragingReader(coupled)
        idx = idx_of(consumer_handler, target)

        gathered = served_times(coupled.get_source(idx))
        reduced = averaged.get_source(idx)

        assert len(gathered) == 4, "four 6 h emissions inside a 24 h request"
        assert served_times(reduced) == [expected_stamp], (
            "the reduction is stamped at the start of the window it averaged"
        )
        assert min(gathered) == expected_stamp, "which is the earliest window it gathered"


def test_only_the_dlesym_lag_gathers_the_atmosphere_at_the_target_time():
    """The content claim, stated where the averaging cannot hide it.

    Averaging collapses the four gathered windows into one row, so the stamp alone can no
    longer say whether `A(T)` was in the mean. This asserts it on the gathered windows, which
    is where the lag actually decides it.
    """

    consumer_handler = handler(SCHEME_INIT, H24)
    producer_handler = handler(SCHEME_INIT, H6)
    target = SCHEME_INIT + 4 * H24

    reach = {}
    for hours in (18, 24):
        coupled = DataReaderCoupling(
            StampedReader(consumer_handler),
            STREAM,
            producer=StampedReader(producer_handler),
            request_handler=shifted(consumer_handler, np.timedelta64(hours, "h")),
            init_time=np.datetime64("2030-01-01T00:00"),
        )
        reach[hours] = max(served_times(coupled.get_source(idx_of(consumer_handler, target))))

    assert reach[18] == target, "18 h reaches the atmosphere at the ocean's own target time"
    assert reach[24] == target - H6, "24 h stops one atmospheric step short of it"


# ---------------------------------------------------------------------------------------------
# source and target are two behaviours, and _get only routes between them
# ---------------------------------------------------------------------------------------------


def _coupled(consumer_handler, producer_handler) -> DataReaderCoupling:
    return DataReaderCoupling(
        StampedReader(consumer_handler),
        STREAM,
        producer=StampedReader(producer_handler),
        request_handler=shifted(consumer_handler, H24),
        init_time=np.datetime64("2030-01-01T00:00"),
    )


def test_a_coupled_stream_refuses_to_serve_targets(consumer_handler, producer_handler):
    """A forcing has no targets, and the wrappers above delegate `get_target` down to here.

    Returning empty would let a stream that is only ever an input score as if it were an
    output; raising names the caller instead (`coupling_reader_placement.md` F1).
    """

    coupled = _coupled(consumer_handler, producer_handler)

    assert coupled.target_idx == [], "a coupled stream carries no target channels"
    with pytest.raises(NotImplementedError, match=STREAM):
        coupled.get_target(idx_of(consumer_handler, CONSUMER_START))


def test_get_routes_to_source_and_target_by_channel_list(consumer_handler, producer_handler):
    """`_get` decides which of the two it is, since the wrappers reach the reader both ways."""

    coupled = _coupled(consumer_handler, producer_handler)
    idx = idx_of(consumer_handler, CONSUMER_START + 2 * H24)

    routed = coupled._get(idx, coupled.source_idx)
    direct = _coupled(consumer_handler, producer_handler).get_source(idx)
    assert served_times(routed) == served_times(direct), "the source list routes to get_source"

    with pytest.raises(NotImplementedError):
        coupled._get(idx, coupled.target_idx)

    with pytest.raises(ValueError, match="neither"):
        coupled._get(idx, [7])


# ---------------------------------------------------------------------------------------------
# the rebase contract, and the metadata it rests on (`coupling_reader_placement.md` G.2, G.3)
# ---------------------------------------------------------------------------------------------


def test_rebasing_a_stack_leaves_the_original_untouched(consumer_handler, producer_handler):
    """P4: a stream's readers are shared with the sampler, so the rebase must not mutate them.

    `rebased` shallow-copies every wrapper on the way down rather than reconstructing it, which
    is only safe if the copy is what gets redirected. Were the wrappers shared with the
    original, redirecting a forcing would silently change what the batch itself reads.
    """

    base = StampedReader(consumer_handler)
    inner_wrapper = AveragingReader(base)
    stack = ElevatingReader(inner_wrapper, CONSUMER_START, H6)
    replacement = StampedReader(producer_handler)

    seen = []

    def make_inner(reader):
        seen.append(reader)
        return replacement

    rebased = rebase_innermost(stack, make_inner)

    assert seen == [base], "make_inner is called on the innermost reader, not on a wrapper"

    # the original stack still reads its own base, all the way down
    assert stack._wrapped_reader is inner_wrapper
    assert stack._wrapped_reader._wrapped_reader is base

    # the clone reads the replacement, and every wrapper of it is a distinct object
    assert rebased._wrapped_reader._wrapped_reader is replacement
    assert rebased is not stack
    assert rebased._wrapped_reader is not inner_wrapper


def test_a_bare_reader_is_replaced_rather_than_rebased(consumer_handler, producer_handler):
    """`rebase_innermost` also accepts a stream whose reader carries no wrappers at all."""

    base = StampedReader(consumer_handler)
    replacement = StampedReader(producer_handler)

    assert rebase_innermost(base, lambda reader: replacement) is replacement


def test_the_reader_refuses_a_producer_that_cannot_supply_the_consumers_channels(
    consumer_handler, producer_handler
):
    """P5/F1: the channel map is resolved by name at construction, and a gap raises there.

    F1's shape: the two components carry their own channel lists for the same stream, agreeing
    in neither order nor length. A subset in a different order is the normal case and must be
    accepted; a channel the producer never emits must be refused while the message can still
    name it, rather than surfacing as a silently mislabelled column at rollout.
    """

    def atmosphere() -> StampedReader:
        producer = StampedReader(producer_handler)
        producer.target_channels = ["10u", "10v", "2t"]
        producer.target_idx = [0, 1, 2]
        return producer

    def ocean(sources: list[str]) -> StampedReader:
        consumer = StampedReader(consumer_handler)
        consumer.source_channels = sources
        consumer.source_idx = list(range(len(sources)))
        return consumer

    # a subset, in the producer's reverse order: selected by name, so the order is irrelevant
    reordered = DataReaderCoupling(ocean(["10v", "10u"]), STREAM, producer=atmosphere())
    assert list(reordered._pred_cols) == [1, 0], "the columns follow the consumer's own order"

    # z_1000 is a channel this atmosphere never emits. Match the constructed message, not just
    # the channel name: with the check removed, `offered.index` raises a bare "not in list"
    # ValueError that also carries the name, so a looser assertion passes vacuously.
    with pytest.raises(ValueError, match="does not supply the channels.*z_1000"):
        DataReaderCoupling(ocean(["10u", "10v", "z_1000"]), STREAM, producer=atmosphere())
