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
import sys

import pytest
from omegaconf import OmegaConf

import weathergen.common.config as config
from weathergen.common.coupling import Coupler, Coupling, ModelCheckpoint, Rollout
from weathergen.datasets.data_reader_base import TimeWindowHandler
from weathergen.train.trainer import ChunkPlan, Trainer
from weathergen.train.utils import resolve_stage_configs


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


class FakeDataset:
    """Carries the real TimeWindowHandler, so index -> time is not itself faked."""

    def __init__(self, window_step):
        self.time_window_handler = TimeWindowHandler(
            config.str_to_datetime64("2023-01-01T00:00"),
            config.str_to_datetime64("2023-06-01T00:00"),
            config.parse_timedelta(window_step),
            config.parse_timedelta(window_step),
        )


class FakeTrainer:
    """Minimal stand-in exposing exactly the component API the Coupler drives."""

    def __init__(self, name, n_chunks, accumulate=False, write=False, window_step="06:00:00"):
        self.name = name
        self.dataset = FakeDataset(window_step)
        self.dynamic_forcings = None
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
                "time_window_len": window_step,
                "time_window_step": window_step,
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


def make_coupler(components, couplings=None, rollout=None):
    return Coupler(
        {name: (t, t.cf) for name, t in components.items()}, couplings, rollout
    )


def make_rollout(chunk_length="24:00:00", num_chunks=4, num_samples=2, forecast_offset=1):
    return Rollout.from_config(
        OmegaConf.create(
            {
                "start_date": "2023-01-01T00:00",
                "end_date": "2023-12-31T00:00",
                "chunk_length": chunk_length,
                "num_chunks": num_chunks,
                "num_samples": num_samples,
                "forecast_offset": forecast_offset,
            }
        )
    )


def make_component_cf(window_step, time_step, *, shuffle=True, policy="sequential", inputs=None):
    """A component config shaped like a real one, for the derivation to chew on."""
    return OmegaConf.create(
        {
            "streams": {"ERA5": {}, "ERA5-Ocean": {}},
            "training_config": {
                "time_window_step": f"${{timedelta:{window_step}}}",
                "time_window_len": f"${{timedelta:{window_step}}}",
                "start_date": "${datetime:1979-01-01T00:00}",
                "end_date": "${datetime:2022-12-31T00:00}",
                "samples_per_mini_epoch": 4096,
                "shuffle": shuffle,
                "losses": {"physical": {"type": "LossPhysical"}},
                "model_input": inputs or {"forecasting": {"masking_strategy": "forecast"}},
                "forecast": {
                    "time_step": f"${{timedelta:{time_step}}}",
                    "num_steps": 2,
                    "offset": 1,
                    "policy": policy,
                },
            },
            "validation_config": {"output": {"num_samples": 0}},
            "test_config": {},
        }
    )


def derive(components, couplings=None, rollout=None):
    """Run the derivation over raw configs and hand back each effective test_cfg."""
    coupler = Coupler(
        {name: (None, cf) for name, cf in components.items()},
        couplings,
        rollout or make_rollout(),
    )
    coupler._check_couplings()
    coupler._derive_component_configs()

    return {name: resolve_stage_configs(cf)[2] for name, cf in components.items()}


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


ATMO_OCEAN = {
    "sst": Coupling(name="sst", producer="Ocean", stream="ERA5-Ocean"),
    "atm": Coupling(name="atm", producer="Atmo", stream="ERA5"),
}


def atmo_ocean_cfs():
    return {
        "Atmo": make_component_cf("06:00:00", "06:00:00"),
        "Ocean": make_component_cf("24:00:00", "24:00:00"),
    }


# --------------------------------------------------------------------------------------
# Drift: sample indices are no longer comparable across components, valid times are.


def test_drifted_samples_are_rejected():
    a, b = FakeTrainer("A", 1), FakeTrainer("B", 1)
    coupler = make_coupler({"A": a, "B": b})

    with pytest.raises(RuntimeError, match="drifted apart"):
        coupler._assert_aligned({"A": FakeBatch([7]), "B": FakeBatch([8])}, bidx=3)


def test_aligned_samples_pass():
    coupler = make_coupler({"A": FakeTrainer("A", 1), "B": FakeTrainer("B", 1)})

    coupler._assert_aligned({"A": FakeBatch([7]), "B": FakeBatch([7])}, bidx=3)


def test_different_grids_at_the_same_instant_are_aligned():
    """The whole point of striding: index 4 on a 6h grid is index 1 on a 24h grid."""
    a = FakeTrainer("A", 1, window_step="06:00:00")
    b = FakeTrainer("B", 1, window_step="24:00:00")
    coupler = make_coupler({"A": a, "B": b})

    coupler._assert_aligned({"A": FakeBatch([4]), "B": FakeBatch([1])}, bidx=1)


def test_equal_indices_on_different_grids_are_rejected():
    """Equal sample_idx used to pass this guard while denoting different dates."""
    a = FakeTrainer("A", 1, window_step="06:00:00")
    b = FakeTrainer("B", 1, window_step="24:00:00")
    coupler = make_coupler({"A": a, "B": b})

    with pytest.raises(RuntimeError, match="drifted apart"):
        coupler._assert_aligned({"A": FakeBatch([1]), "B": FakeBatch([1])}, bidx=1)


# --------------------------------------------------------------------------------------
# Derivation: what used to be checked between components is now pushed down from `rollout`.


def test_chunk_is_a_shared_duration_not_a_shared_step_count():
    cfs = atmo_ocean_cfs()
    out = derive(cfs, ATMO_OCEAN, make_rollout(chunk_length="24:00:00", num_chunks=4))

    assert out["Atmo"].forecast.chunk_size == 4
    assert out["Ocean"].forecast.chunk_size == 1
    # 4 x 6h == 1 x 24h: chunk i is the same wall-clock interval for both
    assert (
        out["Atmo"].forecast.chunk_size * out["Atmo"].forecast.time_step
        == out["Ocean"].forecast.chunk_size * out["Ocean"].forecast.time_step
    )


def test_num_steps_follows_from_num_chunks():
    out = derive(atmo_ocean_cfs(), ATMO_OCEAN, make_rollout(num_chunks=4))

    assert out["Atmo"].forecast.num_steps == 16
    assert out["Ocean"].forecast.num_steps == 4


def test_sample_stride_puts_components_on_one_time_axis():
    out = derive(atmo_ocean_cfs(), ATMO_OCEAN, make_rollout(chunk_length="24:00:00"))

    assert out["Atmo"].sample_stride == 4
    assert out["Ocean"].sample_stride == 1


def test_component_time_step_survives_the_derivation():
    """The overrides deep-merge, so the cascade's own keys must not be clobbered."""
    out = derive(atmo_ocean_cfs(), ATMO_OCEAN)

    assert out["Atmo"].forecast.time_step == config.parse_timedelta("06:00:00")
    assert out["Ocean"].forecast.time_step == config.parse_timedelta("24:00:00")


def test_dates_offset_and_sample_count_come_from_the_rollout():
    out = derive(atmo_ocean_cfs(), ATMO_OCEAN, make_rollout(num_samples=3))

    for cfg in out.values():
        assert cfg.start_date == config.str_to_datetime64("2023-01-01T00:00")
        assert cfg.end_date == config.str_to_datetime64("2023-12-31T00:00")
        assert cfg.forecast.offset == 1
        assert cfg.samples_per_mini_epoch == 3
        # inference writes every sample it runs
        assert cfg.output.num_samples == cfg.samples_per_mini_epoch


def test_batch_size_is_forced_to_one():
    out = derive(atmo_ocean_cfs(), ATMO_OCEAN)

    for cfg in out.values():
        assert cfg.model_input.forecasting.num_samples == 1


def test_several_enabled_model_inputs_are_rejected():
    """Batch size is the sum over enabled entries, so more than one cannot give 1."""
    cfs = {
        "Atmo": make_component_cf(
            "06:00:00",
            "06:00:00",
            inputs={"a": {"masking_strategy": "forecast"}, "b": {"masking_strategy": "random"}},
        )
    }

    with pytest.raises(ValueError, match="exactly one"):
        derive(cfs, {"atm": Coupling(name="atm", producer="Atmo", stream="ERA5")})


def test_shuffle_is_forced_off_with_a_warning(caplog):
    out = derive(atmo_ocean_cfs(), ATMO_OCEAN)

    assert all(cfg.shuffle is False for cfg in out.values())
    assert "forced to False" in caplog.text


@pytest.mark.parametrize("policy", ["random", "sequential_random", "sequential"])
def test_forecast_policy_is_forced_to_fixed_with_a_warning(policy, caplog):
    """Rank-dependent step counts deadlock the FSDP collectives."""
    cfs = {
        "Atmo": make_component_cf("06:00:00", "06:00:00", policy=policy),
        "Ocean": make_component_cf("24:00:00", "24:00:00", policy=policy),
    }
    out = derive(cfs, ATMO_OCEAN)

    assert all(cfg.forecast.policy == "fixed" for cfg in out.values())
    assert "forced to" in caplog.text and "fixed" in caplog.text


def test_chunk_length_not_a_multiple_of_a_time_step_is_rejected():
    cfs = {"Atmo": make_component_cf("06:00:00", "07:00:00")}

    with pytest.raises(ValueError, match="not an exact multiple"):
        derive(cfs, {"atm": Coupling(name="atm", producer="Atmo", stream="ERA5")})


def test_chunk_shorter_than_a_time_step_is_rejected():
    """A chunk must hold whole steps, so a short chunk fails the exact-multiple test."""
    cfs = {"Atmo": make_component_cf("06:00:00", "06:00:00")}

    with pytest.raises(ValueError, match="not an exact multiple"):
        derive(
            cfs,
            {"atm": Coupling(name="atm", producer="Atmo", stream="ERA5")},
            make_rollout(chunk_length="03:00:00"),
        )


# --------------------------------------------------------------------------------------
# Output streams: disjointness is now structural rather than checked.


def test_output_streams_are_the_streams_a_component_produces():
    out = derive(atmo_ocean_cfs(), ATMO_OCEAN)

    assert list(out["Atmo"].output.streams) == ["ERA5"]
    assert list(out["Ocean"].output.streams) == ["ERA5-Ocean"]


def test_two_couplings_producing_one_stream_are_rejected():
    """One producer per stream is what keeps the shared output store collision-free."""
    couplings = {
        "a": Coupling(name="a", producer="Atmo", stream="ERA5-Ocean"),
        "b": Coupling(name="b", producer="Ocean", stream="ERA5-Ocean"),
    }

    with pytest.raises(ValueError, match="Each stream may be produced once"):
        derive(atmo_ocean_cfs(), couplings)


def test_a_component_producing_nothing_writes_nothing():
    """Interleaving without coupling is legal; it just has no output to file."""
    couplings = {"atm": Coupling(name="atm", producer="Atmo", stream="ERA5")}
    out = derive(atmo_ocean_cfs(), couplings)

    assert out["Ocean"].output.num_samples == 0
    assert list(out["Ocean"].output.streams) == []
    assert out["Atmo"].output.num_samples > 0


def test_a_consumer_is_optional():
    """consumer: null is what lets the interleaving run before any exchange exists."""
    couplings = {
        "sst": Coupling(name="sst", producer="Ocean", stream="ERA5-Ocean", consumer=None),
        "atm": Coupling(name="atm", producer="Atmo", stream="ERA5", consumer="Ocean"),
    }

    derive(atmo_ocean_cfs(), couplings)


def test_coupling_naming_an_unknown_consumer_is_rejected():
    couplings = {
        "sst": Coupling(name="sst", producer="Ocean", stream="ERA5-Ocean", consumer="Nope")
    }

    with pytest.raises(ValueError, match="not one of the components"):
        derive(atmo_ocean_cfs(), couplings)


def test_component_options_never_fall_back_to_sys_argv(monkeypatch):
    """OmegaConf.from_cli(None) reads sys.argv[1:], which would land in every component."""
    captured = {}

    def fake_load_merge_configs(
        private_home=None, from_run_id=None, mini_epoch=None, base=None, *overwrites
    ):
        captured["overwrites"] = overwrites
        return OmegaConf.create({"general": {"istep": 0}, "train_logging": {}})

    monkeypatch.setattr(config, "load_merge_configs", fake_load_merge_configs)
    monkeypatch.setattr(
        "weathergen.common.coupling.Trainer", lambda train_logging: object()
    )
    monkeypatch.setattr(
        sys, "argv", ["prog", "coupled_inference", "spec.yml", "Atmo=abc@0", "--run-id", "x"]
    )

    checkpoint = ModelCheckpoint("abc@0")
    checkpoint.get_component(None, None, None, OmegaConf.create({}))

    for overwrite in captured["overwrites"]:
        assert not overwrite, f"command line leaked into the component config: {overwrite}"


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
