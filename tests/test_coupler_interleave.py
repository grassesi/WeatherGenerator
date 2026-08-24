# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the Coupler driver: chunk interleaving and the alignment guards.

These use fakes rather than real models - what is under test is the order in which the
driver steps its components and the preconditions it refuses to run without, none of
which need a network or a GPU.
"""

import contextlib

import pytest
from omegaconf import OmegaConf

from weathergen.common.coupling import Coupler, Coupling
from weathergen.train.trainer import ChunkPlan, Trainer


class FakeSample:
    def __init__(self, sample_idx):
        self.streams_data = {"s": type("SD", (), {"sample_idx": sample_idx})()}


class FakeSourceSamples:
    def __init__(self, sample_idxs):
        self._samples = [FakeSample(i) for i in sample_idxs]

    def get_samples(self):
        return self._samples


class FakeBatch:
    def __init__(self, sample_idxs):
        self._source = FakeSourceSamples(sample_idxs)

    def get_source_samples(self):
        return self._source

    def to_device(self, device):
        pass


class FakeOutput:
    """Stands in for ModelOutput: carries the per-chunk physical/latent lists."""

    def __init__(self, tag):
        self.tag = tag
        self.physical = [tag]
        self.latent = [tag]


class FakeTrainer:
    """Minimal stand-in exposing exactly the component API the Coupler drives."""

    def __init__(self, name, n_chunks, accumulate=False, write=False):
        self.name = name
        self.cf = OmegaConf.create(
            {
                "data_loading": {"memory_pinning": False},
                "general": {"istep": 0},
                "streams": {f"{name}-stream": {}},
                "with_ddp": False,
                "with_mixed_precision": False,
                "local_rank": 0,
            }
        )
        self.test_cfg = OmegaConf.create(
            {
                "start_date": "2023-01-01T00:00",
                "end_date": "2023-01-05T00:00",
                "time_window_len": "06:00:00",
                "time_window_step": "06:00:00",
                "shuffle": False,
                "samples_per_mini_epoch": 8,
                "forecast": {"policy": "fixed", "offset": 1, "time_step": "06:00:00"},
                "output": {"num_samples": 1 if write else 0, "streams": None},
            }
        )
        self.batch_size_test_per_gpu = 1
        self.device = "cpu"
        self.mixed_precision_dtype = None
        self.model = type("M", (), {"eval": lambda self: None})()
        self.target_and_aux_calculators_val = {}
        self.loss_calculator_val = None
        self.data_loader_validation = []
        self._n_chunks = n_chunks
        self._accumulate = accumulate
        self._write = write
        self.steps = []
        self.writes = []
        self.finished = False

    def prepare_chunks(self, batch, mode_cfg, batch_size, bidx, targets_and_auxs):
        return ChunkPlan(
            output_idxs=list(range(self._n_chunks)),
            chunks=[[i] for i in range(self._n_chunks)],
            should_write_output=self._write,
            should_accumulate_chunks=self._accumulate,
            denormalize_data_fct=None,
        )

    def step_chunk(self, forecast_chunk, chunk):
        self.steps.append(chunk[0])
        return FakeOutput(f"{self.name}{chunk[0]}")

    def write_chunk_output(self, plan, mode_cfg, batch_size, mini_epoch, bidx, batch, out, targets):
        self.writes.append(out.tag)

    def assemble_chunks(self, plan, physical, latent, batch):
        return FakeOutput("+".join(physical))

    def finish_validation(self, mini_epoch):
        self.finished = True


@pytest.fixture
def no_autocast(monkeypatch):
    """The driver's autocast targets CUDA; neutralise it for a CPU test."""
    monkeypatch.setattr(Coupler, "_autocast", staticmethod(lambda t: contextlib.nullcontext()))


def make_coupler(components, couplings=None):
    return Coupler({name: (t, t.cf) for name, t in components.items()}, couplings)


def test_chunks_are_interleaved_not_run_to_completion(no_autocast):
    """Every component advances one chunk before any advances two."""
    a, b = FakeTrainer("A", 3), FakeTrainer("B", 3)
    coupler = make_coupler({"A": a, "B": b})

    order = []
    for trainer in (a, b):
        original = trainer.step_chunk

        def traced(fc, chunk, _t=trainer, _o=original):
            order.append((_t.name, chunk[0]))
            return _o(fc, chunk)

        trainer.step_chunk = traced

    coupler._run_batch({"A": FakeBatch([0]), "B": FakeBatch([0])}, bidx=0, mini_epoch=0)

    assert order == [
        ("A", 0), ("B", 0),
        ("A", 1), ("B", 1),
        ("A", 2), ("B", 2),
    ]


def test_shorter_component_idles_while_longer_finishes(no_autocast):
    a, b = FakeTrainer("A", 3), FakeTrainer("B", 1)
    coupler = make_coupler({"A": a, "B": b})

    coupler._run_batch({"A": FakeBatch([0]), "B": FakeBatch([0])}, bidx=0, mini_epoch=0)

    assert a.steps == [0, 1, 2]
    assert b.steps == [0]


def test_output_is_written_per_chunk(no_autocast):
    a = FakeTrainer("A", 3, write=True)
    coupler = make_coupler({"A": a})

    coupler._run_batch({"A": FakeBatch([0])}, bidx=0, mini_epoch=0)

    assert a.writes == ["A0", "A1", "A2"]


def test_accumulation_preserves_chunk_order(no_autocast, monkeypatch):
    monkeypatch.setattr("weathergen.common.coupling.extract_batch_metadata", lambda b: None)
    a = FakeTrainer("A", 3, accumulate=True)
    a.loss_calculator_val = type("LC", (), {"compute_loss": lambda self, **kw: kw["preds"]})()
    assembled = []
    a.assemble_chunks = lambda plan, physical, latent, batch: assembled.append(list(physical))

    coupler = make_coupler({"A": a})
    coupler._run_batch({"A": FakeBatch([0])}, bidx=0, mini_epoch=0)

    assert assembled == [["A0", "A1", "A2"]]


def test_components_are_driven_in_sorted_order(no_autocast):
    """Collective order must not depend on dict insertion order."""
    coupler = make_coupler({"Ocean": FakeTrainer("Ocean", 1), "Atmo": FakeTrainer("Atmo", 1)})

    assert coupler._names == ["Atmo", "Ocean"]


def test_drifted_samples_are_rejected():
    a, b = FakeTrainer("A", 1), FakeTrainer("B", 1)
    coupler = make_coupler({"A": a, "B": b})

    with pytest.raises(RuntimeError, match="drifted apart"):
        coupler._assert_aligned({"A": FakeBatch([7]), "B": FakeBatch([8])}, bidx=3)


def test_aligned_samples_pass():
    coupler = make_coupler({"A": FakeTrainer("A", 1), "B": FakeTrainer("B", 1)})

    coupler._assert_aligned({"A": FakeBatch([7]), "B": FakeBatch([7])}, bidx=3)


def test_shuffle_is_rejected():
    a = FakeTrainer("A", 1)
    a.test_cfg.shuffle = True

    with pytest.raises(ValueError, match="shuffle"):
        make_coupler({"A": a})._check_alignment()


@pytest.mark.parametrize("policy", ["random", "sequential_random"])
def test_rank_dependent_forecast_policy_is_rejected(policy):
    """These would make ranks issue different numbers of FSDP collectives."""
    a = FakeTrainer("A", 1)
    a.test_cfg.forecast.policy = policy

    with pytest.raises(ValueError, match="deadlock"):
        make_coupler({"A": a})._check_alignment()


def test_mismatched_date_range_is_rejected():
    a, b = FakeTrainer("A", 1), FakeTrainer("B", 1)
    b.test_cfg.end_date = "2023-02-01T00:00"

    with pytest.raises(ValueError, match="end_date"):
        make_coupler({"A": a, "B": b})._check_alignment()


def test_mismatched_forecast_offset_is_rejected():
    a, b = FakeTrainer("A", 1), FakeTrainer("B", 1)
    b.test_cfg.forecast.offset = 0

    with pytest.raises(ValueError, match="forecast.offset"):
        make_coupler({"A": a, "B": b})._check_alignment()


def test_differing_time_step_warns_but_runs(caplog):
    a, b = FakeTrainer("A", 1), FakeTrainer("B", 1)
    b.test_cfg.forecast.time_step = "24:00:00"

    make_coupler({"A": a, "B": b})._check_alignment()

    assert "different forecast time steps" in caplog.text


def test_overlapping_output_streams_are_rejected():
    """Both components writing one stream would collide in the shared store."""
    a, b = FakeTrainer("A", 1, write=True), FakeTrainer("B", 1, write=True)
    a.test_cfg.output.streams = ["ERA5-Ocean"]
    b.test_cfg.output.streams = ["ERA5-Ocean"]

    with pytest.raises(ValueError, match="both write"):
        make_coupler({"A": a, "B": b})._check_output_streams()


def test_disjoint_output_streams_pass():
    a, b = FakeTrainer("A", 1, write=True), FakeTrainer("B", 1, write=True)
    a.test_cfg.output.streams = ["ERA5"]
    b.test_cfg.output.streams = ["ERA5-Ocean"]

    make_coupler({"A": a, "B": b})._check_output_streams()


def test_component_writing_nothing_cannot_collide():
    """num_samples=0 writes no zarr, so overlapping stream names are harmless."""
    a = FakeTrainer("A", 1, write=True)
    b = FakeTrainer("B", 1, write=False)
    a.test_cfg.output.streams = ["ERA5-Ocean"]
    b.test_cfg.output.streams = ["ERA5-Ocean"]

    make_coupler({"A": a, "B": b})._check_output_streams()


def test_coupling_naming_an_unknown_component_is_rejected():
    coupling = Coupling(name="A-B", producer="Ocean", consumer="Atmo", stream="ERA5-Ocean")
    coupler = make_coupler({"Atmo": FakeTrainer("Atmo", 1)}, {"A-B": coupling})

    with pytest.raises(ValueError, match="not one of the components"):
        coupler._check_couplings()


def test_coupling_naming_an_unknown_stream_is_rejected():
    coupling = Coupling(name="A-B", producer="Ocean", consumer="Atmo", stream="missing")
    coupler = make_coupler(
        {"Atmo": FakeTrainer("Atmo", 1), "Ocean": FakeTrainer("Ocean", 1)}, {"A-B": coupling}
    )

    with pytest.raises(ValueError, match="does not have"):
        coupler._check_couplings()


# --------------------------------------------------------------------------------------
# The single-model path must keep behaving exactly as before the control inversion:
# _process_validation_chunks is now written in terms of the same three pieces the
# Coupler drives, so these pin down that it still walks the chunks the same way.
# --------------------------------------------------------------------------------------


class BatchWithSteps:
    def __init__(self, output_idxs):
        self._output_idxs = output_idxs

    def get_output_idxs(self):
        return self._output_idxs

    def get_source_samples(self):
        return "source"


def make_real_trainer(chunk_size, num_samples=0, accumulate=True):
    """A real Trainer with only the attributes the rollout path touches."""
    trainer = Trainer.__new__(Trainer)
    trainer.cf = OmegaConf.create({"general": {"run_id": "test"}})
    trainer.ema_model = None
    trainer.model_params = None
    trainer.dynamic_forcings = None
    denormalize = staticmethod(lambda *a: a)
    trainer.dataset_val = type("DS", (), {"denormalize_target_channels": denormalize})()
    trainer.model = lambda params, fc, chunk, forcings: FakeOutput(f"c{chunk[0]}")
    mode_cfg = OmegaConf.create(
        {
            "forecast": {"chunk_size": chunk_size, "accumulate_chunks": accumulate},
            "output": {"num_samples": num_samples, "normalized_samples": False},
        }
    )
    return trainer, mode_cfg


def test_single_model_path_steps_every_chunk():
    trainer, mode_cfg = make_real_trainer(chunk_size=2, accumulate=False)
    batch = BatchWithSteps([0, 1, 2, 3, 4])
    seen = []
    trainer.step_chunk = lambda fc, chunk: seen.append(list(chunk)) or FakeOutput("x")

    result = trainer._process_validation_chunks(batch, mode_cfg, 1, 0, 0, {})

    assert seen == [[0, 1], [2, 3], [4]]
    assert result is None, "accumulate_chunks=False must return None, as before"


def test_single_model_path_accumulates_in_chunk_order():
    trainer, mode_cfg = make_real_trainer(chunk_size=1, accumulate=True)
    batch = BatchWithSteps([0, 1, 2])
    captured = {}
    trainer.assemble_chunks = lambda plan, physical, latent, b: captured.update(
        physical=list(physical), latent=list(latent), chunks=plan.chunks
    )

    trainer._process_validation_chunks(batch, mode_cfg, 1, 0, 0, {})

    assert captured["physical"] == ["c0", "c1", "c2"]
    assert captured["latent"] == ["c0", "c1", "c2"]
    assert captured["chunks"] == [[0], [1], [2]]


def test_single_model_path_defaults_to_one_chunk():
    """Without chunk_size the whole rollout is a single chunk, as before."""
    trainer, mode_cfg = make_real_trainer(chunk_size=None, accumulate=False)
    del mode_cfg.forecast.chunk_size
    batch = BatchWithSteps([0, 1, 2, 3])
    seen = []
    trainer.step_chunk = lambda fc, chunk: seen.append(list(chunk)) or FakeOutput("x")

    trainer._process_validation_chunks(batch, mode_cfg, 1, 0, 0, {})

    assert seen == [[0, 1, 2, 3]]


def test_writing_output_without_targets_still_raises():
    trainer, mode_cfg = make_real_trainer(chunk_size=1, num_samples=4)
    batch = BatchWithSteps([0, 1])

    with pytest.raises(ValueError, match="requires targets"):
        trainer._process_validation_chunks(batch, mode_cfg, 1, 0, 0, {})
