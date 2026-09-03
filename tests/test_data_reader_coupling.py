"""Contract of DataReaderCoupling: the subset of the reader interface ForcingInput uses."""

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from weathergen.common.coupling import Coupler, Coupling, DataReaderCoupling
from weathergen.datasets.batch import ModelBatch, SampleMetaData
from weathergen.datasets.data_reader_base import DataReaderBase, TimeWindowHandler
from weathergen.datasets.stream_data import StreamData
from weathergen.model.model import ModelOutput

STREAM = "ERA5-Ocean"
N_POINTS = 7
SAMPLE_IDX = 100
FORECAST_OFFSET = 1
FSTEPS = [1, 2]


class FakeReader(DataReaderBase):
    """Stand-in for the consumer's own reader for the coupled stream."""

    def __init__(self, twh: TimeWindowHandler) -> None:
        super().__init__(twh, {"stream_id": 0, "token_size": 4, "tokenize_spacetime": False})
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

    def _get(self, idx, channels_idx):
        raise AssertionError("DataReaderCoupling must not fall through to _get")


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


@pytest.fixture
def chunk(coords) -> tuple[ModelOutput, ModelBatch]:
    """One rollout chunk: predictions plus the batch carrying the target geometry."""
    times = np.array(["2023-01-01T00:00"] * N_POINTS, dtype="datetime64[ns]")

    batch = ModelBatch([STREAM], 1, 1, FORECAST_OFFSET, FSTEPS[-1] + 1)
    source = StreamData(SAMPLE_IDX, 1, FSTEPS[-1] + 1, healpix_cells=48)
    target = StreamData(SAMPLE_IDX, 1, FSTEPS[-1] + 1, healpix_cells=48)
    for fstep in FSTEPS:
        target.target_coords_raw[fstep] = torch.tensor(coords)
        target.target_times_raw[fstep] = times
        # a non-trivial permutation, so applying it is observable
        target.idxs_inv[fstep] = torch.arange(N_POINTS - 1, -1, -1)
        target.target_is_spoof[fstep] = False

    batch.add_source_stream(0, 0, STREAM, source, SampleMetaData(params={}, mask=None))
    batch.add_target_stream(0, 0, STREAM, target, SampleMetaData(params={}, mask=None))

    output = ModelOutput(FSTEPS, FORECAST_OFFSET, batch.get_source_samples())
    for fstep in FSTEPS:
        # normalized prediction, constant per step so it is easy to check
        pred = torch.full((1, N_POINTS, 2), float(fstep), dtype=torch.float32)
        output.add_physical_prediction(output.chunk_idx(fstep), STREAM, [pred])

    return output, batch


def collect(reader: DataReaderCoupling, idx: int):
    """The call sequence ForcingInput._collect_forcing_data runs on a reader."""
    rdata = reader.get_source(np.int64(idx)).shuffle(None, False, -1)
    rdata = rdata.remove_nan_coords_and_geoinfos()
    rdata.data = reader.normalize_source_channels(rdata.data)
    rdata.geoinfos = reader.normalize_geoinfos(rdata.geoinfos)
    return rdata


def test_reads_empty_before_any_chunk_arrives(consumer):
    """The first rollout step happens before the producer has run: spoof, do not fail."""
    rdata = DataReaderCoupling(consumer, STREAM).get_source(np.int64(SAMPLE_IDX))

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
        # geoinfos are not predicted, they normalize to zero rather than to noise
        assert np.allclose(rdata.geoinfos, 0.0)


def test_stored_windows_survive_in_place_normalization(consumer, chunk):
    reader = DataReaderCoupling(consumer, STREAM)
    reader.add_chunk(*chunk)

    collect(reader, SAMPLE_IDX + FSTEPS[0])
    again = reader.get_source(np.int64(SAMPLE_IDX + FSTEPS[0]))

    expected = np.float32(FSTEPS[0]) * consumer.stdev[:2] + consumer.mean[:2]
    assert np.allclose(again.data[0], expected)


def test_spoofed_producer_steps_are_skipped(consumer, chunk):
    output, batch = chunk
    batch.get_target_sample(0).streams_data[STREAM].target_is_spoof[FSTEPS[0]] = True

    reader = DataReaderCoupling(consumer, STREAM)
    reader.add_chunk(output, batch)

    assert reader.get_source(np.int64(SAMPLE_IDX + FSTEPS[0])).is_empty()
    assert not reader.get_source(np.int64(SAMPLE_IDX + FSTEPS[1])).is_empty()


def test_eviction_bounds_the_window_store(consumer, chunk):
    reader = DataReaderCoupling(consumer, STREAM, max_pending_windows=1)
    reader.add_chunk(*chunk)

    # the newest window survives, the older one is dropped
    assert reader.get_source(np.int64(SAMPLE_IDX + FSTEPS[0])).is_empty()
    assert not reader.get_source(np.int64(SAMPLE_IDX + FSTEPS[-1])).is_empty()


def test_missing_producer_channel_is_reported(consumer, time_window_handler):
    producer = FakeReader(time_window_handler)
    producer.target_channels = ["sst"]
    producer.target_idx = [0]

    with pytest.raises(ValueError, match="sea_ice"):
        DataReaderCoupling(consumer, STREAM, producer=producer)


def test_coupler_substitutes_only_coupled_streams(consumer, chunk):
    coupler = Coupler({"c": Coupling(name="c", producer="atmo", consumer="ocean", stream=STREAM)})

    forcings = coupler.get_forcings("ocean", {STREAM: [consumer], "era5": [consumer]})

    assert isinstance(forcings[STREAM][0], DataReaderCoupling)
    assert forcings["era5"][0] is consumer

    coupler.dispatch_chunk("atmo", *chunk)
    assert not forcings[STREAM][0].get_source(np.int64(SAMPLE_IDX + FSTEPS[0])).is_empty()

    # a producer nothing is subscribed to is a no-op, not an error
    coupler.dispatch_chunk("nobody", *chunk)
