"""What a coupled run checks about the checkpoints it was handed (`open_actions.md` item 12).

Continuation pairs a checkpoint with itself and scores it against targets from the first
batch, so a half-loaded model shows up as a discontinuity in the metric history. A coupled
rollout pairs two checkpoints, free-runs, and feeds each component's output to the other, so
the same defect reads as a physics result rather than as a loading bug. These are the checks
that make it read as a loading bug instead.

The fakes here are deliberately small: none of these checks touches data, only the metadata
two components have to agree on and the weights that carry the exchange.
"""

import logging
import types

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from weathergen.common.coupling import Coupler, Coupling
from weathergen.datasets.averaging import AveragingReader
from weathergen.datasets.data_reader_base import DataReaderTimestep, TimeWindowHandler
from weathergen.datasets.upsampling import UpsamplingReader

STREAM = "ERA5-Ocean"
START = np.datetime64("2023-01-01T00:00")
END = np.datetime64("2023-06-01T00:00")
H6 = np.timedelta64(6, "h")
H24 = np.timedelta64(24, "h")
H48 = np.timedelta64(48, "h")


class PlainReader(DataReaderTimestep):
    """The metadata a grid check reads, and nothing else."""

    def __init__(self, period: np.timedelta64, filenames: list[str]) -> None:
        handler = TimeWindowHandler(START, END, period, period)
        super().__init__(
            handler,
            {"name": STREAM, "stream_id": 0, "filenames": filenames},
            START,
            END,
            period,
        )
        self.source_channels = ["sst"]
        self.target_channels = ["sst"]
        self.geoinfo_channels = ["lsm"]
        self.source_idx = [0]
        self.target_idx = [0]
        self.geoinfo_idx = [1]
        self.target_channel_weights = [1.0]
        self.mean = np.array([290.0, 0.5], dtype=np.float32)
        self.stdev = np.array([10.0, 0.25], dtype=np.float32)
        self.mean_geoinfo = np.array([0.5], dtype=np.float32)
        self.stdev_geoinfo = np.array([0.25], dtype=np.float32)

    def length(self) -> int:
        return 1

    def _get(self, idx, channels_idx):  # pragma: no cover - never read by these checks
        raise NotImplementedError

    def init_empty(self) -> None:  # pragma: no cover - never read by these checks
        pass


def engine(num_blocks: int, std: float = 0.3):
    """A stand-in for ForcingEngine: only `.blocks` is ever read.

    `std` is the spread of the block weights. The real engine initializes at
    `normal(0, 0.001)` (`forcing.py::init_weights_final`), which is what an untrained engine
    still looks like.

    Wide enough that the sample std is tight: at 4x4 the spread of 20 draws lands outside the
    check's tolerance often enough to make the test flaky, which a real engine's millions of
    parameters never do.
    """
    torch.manual_seed(0)
    blocks = torch.nn.ModuleList([torch.nn.Linear(64, 64) for _ in range(num_blocks)])
    for param in blocks.parameters():
        with torch.no_grad():
            param.normal_(0.0, std)
    return types.SimpleNamespace(blocks=blocks)


def model(num_blocks: int, std: float = 0.3):
    return types.SimpleNamespace(forcing_engine=engine(num_blocks, std))


def component_config(cadence: np.timedelta64, ffe_num_blocks: int = 1):
    hours = int(cadence / np.timedelta64(1, "h"))
    return OmegaConf.create(
        {
            "ffe_num_blocks": ffe_num_blocks,
            "healpix_level": 5,
            "training_config": {
                "time_window_step": f"{hours}h",
                "forecast": {"time_step": f"{hours}h"},
            },
        }
    )


def trainer(name: str, reader: PlainReader, num_blocks: int, std: float = 0.3):
    stream_data = types.SimpleNamespace(readers=[reader])
    dataset = types.SimpleNamespace(streams_datasets={STREAM: stream_data})
    return types.SimpleNamespace(name=name, dataset=dataset, model=model(num_blocks, std))


def coupler(
    *,
    producer_cadence: np.timedelta64 = H24,
    consumer_period: np.timedelta64 = H24,
    producer_files: list[str] | None = None,
    consumer_files: list[str] | None = None,
    consumer_blocks: int = 1,
    consumer_ffe_std: float = 0.3,
    wrappers: tuple[type, ...] = (),
) -> Coupler:
    """A two-component Coupler wired past setup(), with the state the checks read.

    `setup()` itself builds Trainers and models, which is more machinery than any of these
    checks needs; `_substituted` and `_pristine_forcings` are the two things it leaves behind
    that they do read.
    """
    producer_reader = PlainReader(producer_cadence, producer_files or ["ocean.zarr"])
    consumer_reader = PlainReader(consumer_period, consumer_files or ["ocean.zarr"])

    stack = consumer_reader
    for wrapper in wrappers:
        stack = wrapper(stack)

    coup = Coupler(
        {
            "Ocean": (
                trainer("Ocean", producer_reader, 1),
                component_config(producer_cadence),
            ),
            "Atmo": (
                trainer("Atmo", consumer_reader, consumer_blocks, consumer_ffe_std),
                component_config(H6, ffe_num_blocks=consumer_blocks),
            ),
        },
        {"c": Coupling(name="c", producer="Ocean", consumer="Atmo", stream=STREAM)},
    )
    coup._names = ["Ocean", "Atmo"]
    coup._substituted = {("Atmo", STREAM)}
    coup._pristine_forcings = {
        "Atmo": types.SimpleNamespace(forcing_streams={STREAM: [stack]}),
    }
    return coup


# ---------------------------------------------------------------- C2: forcing engines


def test_a_live_coupling_into_an_identity_forcing_engine_is_refused():
    """A zero-block engine discards the forcing it is handed, silently.

    `ffe_num_blocks` defaults to 0 and that is a legal unforced component, so nothing below
    this check distinguishes "no engine was wanted" from "the engine did not load".
    """
    coup = coupler(consumer_blocks=0)

    with pytest.raises(ValueError, match="forcing engine has no parameters"):
        coup._check_forcing_engines()


def test_a_forcing_engine_still_at_its_initialization_is_reported(caplog):
    """Generation 01's xcpk26es and j5h3is35 shipped exactly this and nobody noticed."""
    coup = coupler(consumer_ffe_std=0.001)

    with caplog.at_level(logging.ERROR, logger="weathergen.common.coupling"):
        coup._check_forcing_engines()

    assert any("looks untrained" in r.message for r in caplog.records)


def test_a_trained_forcing_engine_passes_quietly(caplog):
    """The converse, so the probe above is not simply always firing."""
    coup = coupler(consumer_ffe_std=0.3)

    with caplog.at_level(logging.ERROR, logger="weathergen.common.coupling"):
        coup._check_forcing_engines()

    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_a_dead_coupling_is_not_checked():
    """Only what subscribe() substituted is checked; a declared-but-dead coupling is not.

    `_announce_couplings` already warns about those, and refusing a run over the engine of a
    component that receives nothing would reject a legitimate uncoupled arm.
    """
    coup = coupler(consumer_blocks=0)
    coup._substituted = set()

    coup._check_forcing_engines()


# ---------------------------------------------------------------- C4: the exchanged grid


def test_the_two_sides_must_read_the_exchanged_stream_from_the_same_files():
    """Different files are different point sets, and the consumer re-tokenizes either one."""
    coup = coupler(producer_files=["ocean_v2.zarr"], consumer_files=["ocean.zarr"])

    with pytest.raises(ValueError, match="different files"):
        coup._check_exchange_grid()


def test_a_finer_producer_without_an_averaging_reader_is_refused():
    """Four rows per window where training saw one, with nothing above to reduce them."""
    coup = coupler(producer_cadence=H6, consumer_period=H24)

    with pytest.raises(ValueError, match="no AveragingReader"):
        coup._check_exchange_grid()


def test_an_averaging_reader_bridges_a_finer_producer(caplog):
    """The same gap is legitimate once the wrapper that closes it is in the stack."""
    coup = coupler(producer_cadence=H6, consumer_period=H24, wrappers=(AveragingReader,))

    with caplog.at_level(logging.INFO, logger="weathergen.common.coupling"):
        coup._check_exchange_grid()

    assert any(
        "AveragingReader bridges the 4x increase in rows per window" in r.message
        for r in caplog.records
    )


def test_a_coarser_producer_is_held_over_with_a_warning(caplog):
    """The coupling reader restamps the covering window, so this is staleness, not absence."""
    coup = coupler(producer_cadence=H48, consumer_period=H24)

    with caplog.at_level(logging.WARNING, logger="weathergen.common.coupling"):
        coup._check_exchange_grid()

    assert any("coarser than" in r.message for r in caplog.records)


def test_an_upsampling_reader_is_recognised_as_a_bridge(caplog):
    """An upsampled stream still reports which wrapper is doing the bridging.

    Matched on the grid check's own wording, not on the wrapper name alone: UpsamplingReader
    announces itself from its constructor, so `"UpsamplingReader" in message` passes whether
    or not this check ever ran.
    """
    coup = coupler(producer_cadence=H24, consumer_period=H24, wrappers=(UpsamplingReader,))

    with caplog.at_level(logging.INFO, logger="weathergen.common.coupling"):
        coup._check_exchange_grid()

    assert any(
        "grid agrees" in r.message and "bridged by ['UpsamplingReader']" in r.message
        for r in caplog.records
    )


def test_token_size_and_healpix_level_are_not_compared():
    """Relaxed deliberately: they are each component's own tokenization, not the grid.

    The consumer re-tokenizes whatever it receives with its own tokenizer, so a difference
    here changes nothing about the exchanged points. Recorded by the provenance line instead.
    """
    coup = coupler()
    coup._components["Atmo"][1].healpix_level = 3
    coup._components["Atmo"][1].training_config.token_size = 16

    coup._check_exchange_grid()


# ---------------------------------------------------------------- C3: provenance


def test_provenance_names_the_checkpoint_of_every_component(caplog):
    """Which two checkpoints were paired is the experiment, and nothing else records it."""
    coup = coupler()
    checkpoints = {
        "Ocean": types.SimpleNamespace(run_id="mgqf3zqe", mini_epoch=12),
        "Atmo": types.SimpleNamespace(run_id="ta816nwq", mini_epoch=1455),
    }

    with caplog.at_level(logging.INFO, logger="weathergen.common.coupling"):
        coup._report_checkpoint_provenance(checkpoints)

    lines = [r.message for r in caplog.records if "provenance" in r.message]
    assert len(lines) == 2
    assert any("mgqf3zqe@12" in line for line in lines)
    assert any("ta816nwq@1455" in line for line in lines)
    assert any("ffe_params=" in line for line in lines)
